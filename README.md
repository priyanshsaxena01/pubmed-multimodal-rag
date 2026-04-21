# PubMed Multimodal RAG - DomainBot 🩺
## ML-DL-Ops Course Project

**Member 1 -** Priyansh Saxena - B22EE075

**Member 2 -** Anchitya Kumar - B22BB009

---

## 🔗 Project Links

- 🤗 **Fine-Tuned Model (Hugging Face):**  
  https://huggingface.co/b22ee075/Qwen3-VL-4B-PubMed  

- 🧠 **Original Qwen Models:**
  - Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct  
  - Qwen3-VL-Embedding-2B: https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B  
  - Qwen3-VL-Reranker-2B: https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B  

- 📊 **Weights & Biases Training Logs:**  
  [Wandb Link](https://wandb.ai/priyansh-saxena/pubmed-multimodal-rag/workspace)

- 🌐 **Live Deployment (Render):**  
  [Render Link](https://pubmed-rag-ui-v1.onrender.com/)

- ☁️ **GCP / Kubernetes Endpoint:**  
  [GCP_Loadbalancer_IP](http://34.100.147.140/)

---

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

## 🏗️ System Architecture & Components

The system is composed of five deeply integrated layers spanning two distinct compute environments: a **cloud frontend** (stateless, GPU-free, publicly accessible) and a **DGX backend** (stateful, multi-GPU, institutionally firewalled). Communication between these environments is tunnelled securely via Cloudflare.

### End-to-End Request Flow

```
User (Browser)
    │
    ▼
[1] Streamlit Frontend (frontend.py)
    │  ── HTTP POST /query (with optional image) ──►
    ▼
[2] FastAPI Orchestrator (orchestrator.py)
    │
    ├── PDF/Image upload? ──► PyMuPDF extraction ──► Text chunks + embedded images
    │                                                          │
    │                                                          ▼
    │                                                  FAISS Vector Index
    │
    ├── /embed ──► Support APIs (DGX GPU 1) ──► Qwen3-VL-Embedding-2B
    │
    ├── /rerank ──► Support APIs (DGX GPU 1) ──► Qwen3-VL-Reranker-2B
    │              (top-3 chunks selected)
    │
    └── Final prompt (system prompt + retrieved context + images + user query)
             │
             ▼
    [3] LMDeploy Engine (DGX GPU 0)
             │  Qwen3-VL-4B-Instruct (LoRA-merged, PubMed fine-tuned)
             ▼
    Streaming token response ──► Orchestrator ──► Frontend (SSE/stream)
             │
             ▼
    User sees response in chat UI
```

---

### 1. Frontend (`frontend.py`)

The frontend is a **Streamlit** application that serves as the sole user-facing interface. It is intentionally kept stateless and lightweight — it holds no model weights and performs no heavy computation, making it trivially deployable on free-tier cloud services.

**Interface layout:**

- **Sidebar (Knowledge Base Management)**:
  - Multi-file uploader accepting `.pdf`, `.txt`, `.png`, `.jpg`, `.jpeg`.
  - On upload, files are `POST`-ed to the orchestrator's `/index` endpoint, which triggers extraction and embedding in the background.
  - A "Vision Scan" panel allows uploading medical images separately for image-only visual queries, displayed as a thumbnail grid before querying.
  - Session state tracks all indexed documents and images within a conversation.

- **Main Chat Panel**:
  - Standard chat history display with user/assistant message bubbles.
  - Sends the user message (plus any attached images) to the orchestrator's `/query` endpoint.
  - Handles **streaming responses** via server-sent events — tokens are written to the UI incrementally as they arrive from the DGX, giving a responsive feel despite long-context generation.
  - Displays an inline "Sources" expander below each assistant reply, listing which retrieved chunks informed the answer (chunk text preview + source filename + page number).

**Containerization:** Packaged as `docker.io/priyanshsaxena1/pubmed-rag-ui:v1`. The image contains only the Python dependencies for Streamlit and HTTP requests — no ML libraries, no GPU drivers. The two required environment variables (`VLLM_API_URL`, `INFERENCE_API_URL`) point it at the DGX tunnels at runtime.

---

### 2. Orchestrator (`orchestrator.py`)

The orchestrator is the **central nervous system** of the RAG pipeline. It is a **FastAPI** application running inside the same Docker container as the frontend (on port `5000`, internal to the container). It coordinates every step from document ingestion through to final prompt assembly.

#### 2a. Document Ingestion & Parsing

When a file is uploaded, the orchestrator uses **PyMuPDF (`fitz`)** to handle both text and visual content extraction from PDFs:

- **Text extraction**: Each PDF page is parsed into raw text, then split into overlapping chunks (configurable chunk size / stride) to preserve context across chunk boundaries. Plain `.txt` files are chunked directly.
- **Embedded image extraction**: `fitz` iterates over every PDF page's image list (`page.get_images()`), decodes each XObject image, and saves it as a base64 PNG. These images are treated as first-class retrieval candidates alongside text chunks — a page diagram or MRI scan embedded in a paper can be retrieved and passed to the vision-language model.
- **Metadata tagging**: Every chunk and image is tagged with its source filename and page number, enabling accurate citation in the final response.

#### 2b. Embedding & FAISS Indexing

After chunking, each text chunk and image is embedded via a call to the `/embed` endpoint on the Support APIs (see §3). The returned float vectors are added to a **FAISS `IndexFlatIP` (inner product) index** held in memory:

- Text and image embeddings share the same vector space, since `Qwen3-VL-Embedding-2B` is trained to produce modality-aligned representations.
- The FAISS index is rebuilt from scratch on each new upload (suitable for session-scoped knowledge bases). Persistent cross-session indexing can be enabled by serializing with `faiss.write_index`.
- Chunk metadata (text, source, page, modality) is stored in a parallel Python list indexed identically to the FAISS vectors, allowing O(1) lookup after a search.

#### 2c. Query-Time Retrieval & Reranking

On each user query:

1. The query text (and optional query image) is embedded via `/embed`.
2. FAISS performs an approximate nearest-neighbour search, returning the top-K candidates (default K=10).
3. The top-K candidates are passed to `/rerank` along with the query. The reranker scores each candidate's relevance more precisely using cross-attention over the full query–document pair.
4. The top 3 reranked candidates are selected as the final retrieved context. This two-stage retrieval (fast ANN → precise reranker) balances speed with retrieval quality.

#### 2d. Prompt Assembly & LLM Routing

The orchestrator constructs the final multimodal prompt:

```
[SYSTEM PROMPT — clinical constraints, output format rules]
[RETRIEVED CONTEXT BLOCK 1 — text or image]
[RETRIEVED CONTEXT BLOCK 2]
[RETRIEVED CONTEXT BLOCK 3]
[USER QUERY — text + optional image]
```

This assembled prompt is forwarded to the **LMDeploy** OpenAI-compatible API endpoint running on DGX GPU 0 (via Cloudflare tunnel). The response is streamed back token-by-token to the frontend.

---

### 3. Support APIs (`support_apis.py`) — Runs on DGX GPU 1

A dedicated **FastAPI microservice** running on DGX GPU 1 that hosts the two retrieval models. It is kept separate from the generation engine (GPU 0) to avoid VRAM contention and to allow independent scaling or swapping of retrieval models.

#### `/embed` — Multimodal Embedding Endpoint

- **Model**: `Qwen3-VL-Embedding-2B` loaded via the `qwen3_vl_wrapper` submodule.
- **Input**: JSON body with either a `text` string or a `image` field (base64-encoded PNG/JPEG).
- **Processing**: The model's vision encoder processes images; the language encoder processes text. Both produce vectors in the same shared latent space — enabling cross-modal similarity search.
- **Output**: A flat float array (the embedding vector). Dimensionality is model-dependent (~1536d).
- **Batching**: Accepts a list of inputs for batch embedding, which the orchestrator uses during document indexing to reduce round-trip overhead.

#### `/rerank` — Cross-Modal Reranking Endpoint

- **Model**: `Qwen3-VL-Reranker-2B`, a cross-encoder that jointly encodes the query and each candidate document to produce a relevance score.
- **Input**: A `query` (text + optional image) and a list of `candidates` (text snippets or image base64 strings).
- **Processing**: Unlike the bi-encoder embedding model, the reranker performs full cross-attention between query and each candidate — significantly more accurate but also more compute-intensive, which is why it operates only on the pre-filtered top-K FAISS results.
- **Output**: A list of float relevance scores, one per candidate. The orchestrator sorts by score and retains the top 3.

---

### 4. Training Pipeline (`train.py`)

The custom fine-tuning script that produced the `Qwen3-VL-4B-PubMed` model weights. It is not required for inference but is included for full reproducibility.

#### Dataset

- **Base dataset**: `PubMedVision-enhanced` — a curated collection of PubMed figure–caption pairs and clinical VQA examples, augmented with structured reasoning traces (`PubMedVision-enhanced-z000` variant).
- **Data cleaning**: The script includes an active preprocessing step that parses each row's JSON fields and drops any rows with corrupted arrays, null fields, or malformed image references. This is critical because the raw PubMed dataset contains a non-trivial fraction of broken entries that cause silent training failures.

#### Fine-Tuning Strategy

- **Method**: **LoRA (Low-Rank Adaptation)** via the `peft` library. Rather than updating all 4B parameters, LoRA injects small trainable rank-decomposition matrices into the attention layers. This drastically reduces VRAM usage and training time while preserving most of the base model's general reasoning.
- **Quantization**: The base model is loaded in **4-bit NF4** precision (`BitsAndBytesConfig`) to fit on a single DGX GPU during training. The LoRA adapters themselves are trained in `bfloat16`.
- **Optimizer**: **8-bit paged AdamW** (`optim="paged_adamw_8bit"`) — the "paged" variant offloads optimizer states to CPU RAM when GPU memory pressure is high, preventing out-of-memory crashes on long sequences.

#### Memory Optimizations

| Optimization | Purpose |
|---|---|
| 4-bit NF4 base model quantization | Reduces base model VRAM from ~16 GB to ~4 GB |
| LoRA adapters only | ~10–50 MB of trainable params vs. 8 GB for full fine-tune |
| `max_pixels` image cap | Prevents VRAM spikes from very high-res figures |
| Gradient checkpointing | Trades compute for memory during backprop |
| 8-bit paged AdamW | CPU-offloaded optimizer states |

#### Training Outputs

After training, LoRA adapter weights are saved to `b22ee075/Qwen3-VL-4B-PubMed` (HuggingFace Hub). These are merged into the base model by `merge_lora.py` before serving.

---

### 5. Utility & Patch Scripts

A set of targeted scripts that handle model preparation and environment compatibility.

#### `merge_lora.py` — LoRA Weight Merging

After fine-tuning, the LoRA adapter weights exist as a separate delta on top of the base model. `merge_lora.py` uses `peft`'s `merge_and_unload()` to arithmetically fold the adapter weights into the base model's weight matrices, producing a single standalone model (`./qwen-pubmed-merged`). This merged model is then served directly by LMDeploy without requiring `peft` at inference time — simplifying the serving stack and slightly improving throughput.

#### `patch_system_prompt.py` — Clinical Prompt Injection

Dynamically rewrites the system prompt string inside `orchestrator.py` to enforce strict clinical behaviour:
- Answers must be grounded in retrieved context only.
- If the context does not contain enough information to answer, the model must say so explicitly.
- Output is formatted as objective, numbered bullet points.
- The model must never provide a definitive diagnosis.

This patching approach allows prompt iteration without modifying the orchestrator's core logic.

#### `patch_vllm.py` — vLLM RoPE Scaling Compatibility Fix

The Qwen3-VL model uses a non-standard RoPE (Rotary Position Embedding) scaling configuration that triggers assertion errors in certain versions of `vllm`. This script locates the relevant validation function inside the installed `vllm` package and comments out the offending assertion, allowing the model to serve correctly. A targeted one-line fix that avoids the need to maintain a custom `vllm` fork.

#### `patch_wrapper.py` — Reranker AutoProcessor Path Fix

The `Qwen3-VL-Reranker-2B` model repository uses relative import paths for `AutoProcessor` that break when the `qwen3_vl_wrapper` submodule is loaded outside its own working directory. This script uses regex substitution to rewrite those paths to absolute references, ensuring the reranker initializes correctly regardless of where the orchestrator is invoked from.


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
