# PubMed Multimodal RAG 🩺

## Overview

This repository contains the codebase for a highly advanced clinical AI assistant fine-tuned on PubMed literature. Its primary function is to assist medical professionals by analyzing clinical notes, extracting literature context, and evaluating medical imaging. The system implements a dynamic Retrieval-Augmented Generation (RAG) architecture capable of processing mixed media inputs, including text and images, into a cohesive knowledge base.

## Features

* **Dynamic Multimodal RAG**: Allows users to upload clinical notes (PDF, TXT) and medical images (PNG, JPG, JPEG) to dynamically update the system's knowledge base.
* **Vision Scan**: Enables users to process and view multiple medical images for visual queries.
* **Advanced Embeddings & Reranking**: Utilizes `Qwen3-VL-Embedding-2B` for generating multimodal embeddings and `Qwen3-VL-Reranker-2B` for high-precision document reranking.
* **Strict Clinical Constraints**: The AI is strictly programmed to base answers on the extracted context, state if an answer cannot be determined, output clear bullet points, and refrain from providing definitive medical diagnoses.
* **Custom Fine-Tuned Model**: The core generation engine is built on the `Qwen3-VL-4B-Instruct` architecture and fine-tuned specifically on the `PubMedVision-enhanced-z000` dataset.
* **Hybrid Cloud Architecture**: Isolates heavy multi-GPU inference on institutional DGX hardware while serving a highly available, low-latency Streamlit frontend via the public cloud (Render/Docker Hub).
* **Kubernetes-Native Deployment**: Production-grade deployment on Google Kubernetes Engine with autoscaling, load balancing, and public endpoint exposure.

---

## System Architecture & Components

The project consists of several independent but heavily integrated components separated into a backend compute environment and a frontend cloud environment:

### 1. Frontend (`frontend.py`)

* Built using Streamlit to provide an intuitive web interface titled "PubMed AI".
* Features a sidebar that supports multi-file document indexing and medical image uploads.
* Maintains a chat interface that communicates with a backend API for streaming text responses.
* Containerized via Docker for cloud-agnostic deployment.

### 2. Orchestrator (`orchestrator.py`)

* A FastAPI backend that orchestrates the entire RAG pipeline.
* Extracts both text chunks and images directly from uploaded PDFs using `fitz` (PyMuPDF).
* Manages a `faiss` vector database to store and search through chunk embeddings.
* Calls the Inference API for embeddings and reranking scores, selectively filtering the top 3 most relevant context blocks.
* Formats the final query encompassing the clinical system prompt, retrieved documents, images, and user prompt before routing securely to the DGX GPU server.

### 3. Support APIs (`support_apis.py`) [Runs on DGX]

* A dedicated FastAPI microservice managing the vision-language retrieval models.
* **`/embed` Endpoint**: Converts incoming text strings or base64-encoded images into numerical vectors using the Qwen3-VL embedding model.
* **`/rerank` Endpoint**: Re-scores a list of candidate documents against a given query using the Qwen3-VL reranker.

### 4. Training Pipeline (`train.py`)

* Implements a LoRA fine-tuning script for `Qwen/Qwen3-VL-4B-Instruct` using `peft` and 4-bit `BitsAndBytesConfig` quantization.
* Features an active data cleaning step for the `PubMedVision-enhanced` dataset to safely parse and drop corrupted JSON arrays or empty rows.
* Includes strict memory optimizations such as limiting image resolution (max pixels), enabling gradient checkpointing, and utilizing an 8-bit paged AdamW optimizer to prevent VRAM crashes.

### 5. Utility & Patch Scripts

* **`merge_lora.py`**: Merges the fine-tuned LoRA weights (`b22ee075/Qwen3-VL-4B-PubMed`) back into the base Qwen3-VL-4B model and saves the baked weights locally.
* **`patch_system_prompt.py`**: Dynamically edits the orchestrator file to inject a rigorous, clinical system prompt demanding objective tones and strict contextual grounding.
* **`patch_vllm.py`**: Adjusts the local `vllm` package configuration to remove strict assertion errors regarding rope scaling parameters.
* **`patch_wrapper.py`**: Uses Regex to aggressively patch the AutoProcessor paths in the Qwen3-VL reranker repository to ensure proper model routing.

---

## Submodule: Qwen3-VL Wrapper (`qwen3_vl_wrapper/`)

The project bundles the official Qwen3-VL-Embedding and Reranker tools for local usage.

* Designed to process text, images, screenshots, videos, and mixed-modal inputs within a unified framework.
* Includes comprehensive scripts for evaluating performance on major datasets (e.g., ImageNet, DocVQA, MSCOCO).
* Contains example Jupyter Notebooks (`embedding_vllm.ipynb`, `Qwen3VL_Multimodal_RAG.ipynb`) to demonstrate retrieval logic.

---

## Deployment Guide

This system supports two deployment modes: a **Hybrid Cloud** approach for direct cloud hosting, and a **Kubernetes** deployment on Google Cloud Platform for production-grade scalability.

### Phase 1: DGX Compute Backend (Required for Both Modes)

The backend requires a Linux environment with NVIDIA GPUs and CUDA configured.

1. SSH into the DGX server and activate your Conda environment:
    ```bash
    conda activate pubmed_env
    ```
2. Start the AI engines using `tmux`:
    ```bash
    # Start Support APIs (Embeddings) on GPU 1
    tmux new -d -s support_api "CUDA_VISIBLE_DEVICES=1 python support_apis.py"

    # Start LMDeploy Engine on GPU 0
    tmux new -d -s lmdeploy_engine "CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server ./qwen-pubmed-merged --model-name pubmed-adapter --server-port 8000"
    ```
3. Open **Cloudflare Quick Tunnels** to bridge the DGX to the internet securely:
    ```bash
    tmux new -d -s tunnel_vllm "./cloudflared-linux-amd64 tunnel --url http://localhost:8000 > tunnel_vllm.log 2>&1"
    tmux new -d -s tunnel_support "./cloudflared-linux-amd64 tunnel --url http://localhost:8001 > tunnel_support.log 2>&1"
    ```
4. Extract your secure URLs from the logs:
    ```bash
    cat tunnel_vllm.log | grep -o 'https://[^[:space:]]*\.trycloudflare\.com'
    cat tunnel_support.log | grep -o 'https://[^[:space:]]*\.trycloudflare\.com'
    ```

---

### Phase 2a: Cloud Frontend Deployment (Render)

The frontend is packaged as a Docker image (`docker.io/priyanshsaxena1/pubmed-rag-ui:v1`) and can be deployed anywhere without needing GPUs.

**Zero-Friction Deployment (Render)**

1. Go to [Render.com](https://render.com) and click **New Web Service**.
2. Select **Deploy an existing image from a registry** and paste: `docker.io/priyanshsaxena1/pubmed-rag-ui:v1`.
3. Under **Advanced**, add the Cloudflare URLs you generated in Phase 1 as Environment Variables:
   - `VLLM_API_URL`: `<YOUR_CLOUDFLARE_VLLM_URL>`
   - `INFERENCE_API_URL`: `<YOUR_CLOUDFLARE_SUPPORT_URL>`
4. Deploy to access the live web UI.

**Local Testing Deployment**

```bash
# Set environment variables locally
export VLLM_API_URL="<YOUR_CLOUDFLARE_VLLM_URL>"
export INFERENCE_API_URL="<YOUR_CLOUDFLARE_SUPPORT_URL>"

# Start the Orchestrator
python orchestrator.py &

# Start the Streamlit UI
streamlit run frontend.py --server.port=80
```

---

### Phase 2b: Kubernetes Deployment on GCP (Production)

For production-grade deployments, the system can be hosted on **Google Kubernetes Engine (GKE)** with autoscaling and a public load balancer.

#### Architecture

```
User Browser
     ↓
Public IP (LoadBalancer)
     ↓
Streamlit UI (Frontend)
     ↓
FastAPI Backend
     ↓
FAISS Vector Store
     ↓
Cloudflare Tunnel APIs
     ↓
Embedding + Reranking + LLM Response
```

#### Docker Container

The container image bundles the FastAPI backend, Streamlit frontend, FAISS pipeline, PDF ingestion engine, and embedding API connectors in a single image.

```dockerfile
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn streamlit requests faiss-cpu numpy pymupdf pydantic python-multipart

COPY orchestrator.py frontend.py ./

RUN echo '#!/bin/bash
python orchestrator.py &
streamlit run frontend.py --server.port=80 --server.address=0.0.0.0
' > start.sh

RUN chmod +x start.sh

EXPOSE 80 5000

CMD ["./start.sh"]
```

| Service    | Port |
|------------|------|
| Streamlit  | 80   |
| FastAPI    | 5000 |

#### Cluster Setup

```bash
# Create GKE Autopilot cluster
gcloud container clusters create-auto autopilot-cluster-1 \
  --region asia-south1

# Configure credentials
gcloud container clusters get-credentials autopilot-cluster-1 \
  --region asia-south1
```

#### Deploy the Application

```bash
# Create deployment
kubectl create deployment pubmed-rag \
  --image=docker.io/priyanshsaxena1/pubmed-rag-ui:v1

# Set inference API environment variables
kubectl set env deployment/pubmed-rag \
  VLLM_API_URL=<YOUR_CLOUDFLARE_VLLM_URL> \
  INFERENCE_API_URL=<YOUR_CLOUDFLARE_SUPPORT_URL>

# Expose via public LoadBalancer
kubectl expose deployment pubmed-rag \
  --type=LoadBalancer \
  --port=80 \
  --target-port=80
```

#### Autoscaling

```bash
kubectl autoscale deployment pubmed-rag \
  --cpu=70% \
  --min=1 \
  --max=5
```

| Load   | Pods |
|--------|------|
| Low    | 1    |
| Medium | 2–3  |
| High   | 4–5  |

Infrastructure nodes scale automatically via GKE Autopilot.

#### Monitoring

```bash
kubectl get pods
kubectl get services
kubectl get hpa
kubectl logs deployment/pubmed-rag
```

## ⚠️ Disclaimer

**This is a research project.** The model architecture, embeddings, and generated responses are strictly for educational and engineering demonstration purposes. This system is **not** intended for clinical decision-making, patient diagnosis, or as a replacement for professional medical advice. Always consult a qualified healthcare provider for medical concerns.
