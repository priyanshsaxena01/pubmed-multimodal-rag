Here is a complete, professional, and detailed `README.md` for your GitHub repository. It covers the entire hybrid architecture, from the DGX hardware to the cloud deployment, making it an excellent showcase for your engineering portfolio.

You can copy and paste this directly into your repository.

-----

# 🏥 Qwen3-VL PubMed Multimodal RAG Pipeline

An end-to-end Medical Visual Question Answering (VQA) and Multimodal Retrieval-Augmented Generation (RAG) system. This project leverages a custom-fine-tuned **Qwen3-VL-8B-Instruct** model, grounded in clinical literature and medical imaging from PubMed.

It features a unique **Hybrid Cloud Architecture**, isolating heavy multi-GPU inference on institutional hardware (NVIDIA DGX) while serving a highly available, low-latency Streamlit frontend via the public cloud.

-----

## 🏗️ Hybrid Cloud Architecture

To optimize compute costs and bypass restrictive university firewalls, this project splits the processing pipeline across two distinct environments bridged by secure tunnels.

1.  **The Compute Backend (NVIDIA DGX Server)**

      - Houses the massive GPU requirements for the VLM.
      - Runs the **LMDeploy** inference engine for the fine-tuned Qwen3-VL model.
      - Runs **Support APIs** for multimodal embedding generation.
      - Exposed securely to the internet via **Cloudflare Quick Tunnels**.

2.  **The Cloud Frontend (Render / Docker)**

      - A lightweight Docker container hosting the **Streamlit UI** and **FastAPI Orchestrator**.
      - Manages the **FAISS Vector Database** and document chunking (PyMuPDF).
      - Routes user text and image queries to the DGX server over the Cloudflare tunnels.

-----

## ✨ Key Features

  - **Multimodal Understanding:** Processes both complex clinical text queries and medical imagery (diagrams, scans, figures).
  - **PubMed Grounding:** Utilizes a FAISS-backed RAG pipeline to retrieve relevant medical literature before generating responses, reducing hallucinations.
  - **LoRA Fine-Tuned:** The base Qwen3-VL model was fine-tuned for \~8 hours on specialized medical QA pairs.
  - **Cloud-Agnostic UI:** The frontend is fully containerized and deployable on any cloud provider (Azure, Render, AWS) without region locks.

-----

## 📂 Repository Structure

```text
├── qwen-pubmed-merged/       # (Ignored) Merged VLM weights
├── data/                     # (Ignored) FAISS vector indexes and PDF corpus
├── orchestrator.py           # FastAPI backend handling RAG logic and FAISS retrieval
├── frontend.py               # Streamlit user interface
├── support_apis.py           # Embedding generation API (Runs on DGX)
├── start_services.sh         # Master script to boot DGX tmux sessions
├── Dockerfile                # Cloud container instructions for UI/Orchestrator
└── README.md                 # Project documentation
```

-----

## 🚀 Deployment Guide

### Phase 1: Start the DGX Compute Backend

The backend requires a Linux environment with NVIDIA GPUs and CUDA configured.

1.  Clone the repository on the DGX server.
2.  Install dependencies:
    ```bash
    conda create -n pubmed_env python=3.10
    conda activate pubmed_env
    pip install lmdeploy vllm fastapi uvicorn
    ```
3.  Start the AI engines using `tmux`:
    ```bash
    # Start Support APIs (Embeddings) on GPU 1
    tmux new -d -s support_api "CUDA_VISIBLE_DEVICES=1 python support_apis.py"

    # Start LMDeploy Engine on GPU 0
    tmux new -d -s lmdeploy_engine "CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server ./qwen-pubmed-merged --model-name pubmed-adapter --server-port 8000 --cache-max-entry-count 0.8"
    ```
4.  Open the Cloudflare Tunnels to bridge the DGX to the internet:
    ```bash
    tmux new -d -s tunnel_vllm "./cloudflared-linux-amd64 tunnel --url http://localhost:8000 > tunnel_vllm.log 2>&1"
    tmux new -d -s tunnel_support "./cloudflared-linux-amd64 tunnel --url http://localhost:8001 > tunnel_support.log 2>&1"
    ```
5.  Extract your secure URLs from the logs:
    ```bash
    cat tunnel_vllm.log | grep -o 'https://[^[:space:]]*\.trycloudflare\.com'
    cat tunnel_support.log | grep -o 'https://[^[:space:]]*\.trycloudflare\.com'
    ```

### Phase 2: Deploy the Cloud Frontend

The frontend is packaged as a Docker image and can be deployed anywhere. It requires the URLs generated in Phase 1.

**Option A: Deploy via Render (Recommended/Zero-Friction)**

1.  Ensure your image is pushed to Docker Hub (e.g., `docker.io/priyanshsaxena1/pubmed-rag-ui:v1`).
2.  Go to [Render.com](https://render.com) -\> New Web Service.
3.  Select "Deploy an existing image from a registry" and paste your Docker Hub URL.
4.  Under "Advanced", add the necessary Environment Variables:
      - `VLLM_API_URL`: `<YOUR_CLOUDFLARE_VLLM_URL>`
      - `INFERENCE_API_URL`: `<YOUR_CLOUDFLARE_SUPPORT_URL>`
5.  Deploy to access the live web UI.

**Option B: Deploy Locally (For Development)**

```bash
# Set environment variables locally
export VLLM_API_URL="<YOUR_CLOUDFLARE_VLLM_URL>"
export INFERENCE_API_URL="<YOUR_CLOUDFLARE_SUPPORT_URL>"

# Start the Orchestrator
python orchestrator.py &

# Start the Streamlit UI
streamlit run frontend.py --server.port=80
```

-----

## 📊 Model Information

For comprehensive details on training hyperparameters, evaluation metrics, and dataset curation, please view the [Model Card](https://huggingface.co/) for this project.

-----

## ⚠️ Disclaimer

**This is a research project.** The model architecture, embeddings, and generated responses are strictly for educational and engineering demonstration purposes. This system is **not** intended for clinical decision-making, patient diagnosis, or as a replacement for professional medical advice. Always consult a qualified healthcare provider for medical concerns.
