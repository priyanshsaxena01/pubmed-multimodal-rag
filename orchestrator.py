import os, json, base64, requests, faiss, numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import fitz  # PyMuPDF
import uvicorn
from typing import List, Optional

app = FastAPI()
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000") + "/v1/chat/completions"
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8001")

FAISS_PATH, DOCS_PATH = "./data/kb.faiss", "./data/doc_store.json"
index, doc_store = None, []

os.makedirs("./data", exist_ok=True)
if os.path.exists(FAISS_PATH) and os.path.exists(DOCS_PATH):
    index = faiss.read_index(FAISS_PATH)
    with open(DOCS_PATH, "r") as f: doc_store = json.load(f)

SYSTEM_PROMPT = """You are a highly advanced clinical AI assistant fine-tuned on PubMed literature.
Your primary function is to assist medical professionals by analyzing clinical notes, extracting literature context, and evaluating medical imaging.

CRITICAL INSTRUCTIONS:
1. Grounding: Base your answers STRICTLY on the 'Extracted Medical Context'. 
2. Honesty: If the provided context or image does not contain the answer, state: "I cannot determine this based on the provided documents/image."
3. Tone: Maintain a highly professional, objective, and clinical tone.
4. Formatting: Use clear bullet points for differential diagnoses, findings, or literature summaries.
5. Disclaimer: You do not provide definitive medical diagnoses."""

@app.post("/clear_db")
def clear_database():
    global index, doc_store
    index = None
    doc_store = []
    if os.path.exists(FAISS_PATH): os.remove(FAISS_PATH)
    if os.path.exists(DOCS_PATH): os.remove(DOCS_PATH)
    return {"message": "Knowledge base cleared successfully."}

@app.post("/upload_doc")
async def upload_document(file: UploadFile = File(...)):
    global index, doc_store
    chunks = []
    
    if file.filename.lower().endswith(".pdf"):
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page in doc:
            text = page.get_text().strip()
            if len(text) > 20:
                chunks.append({"type": "text", "content": text})
            
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                b64_img = base64.b64encode(image_bytes).decode("utf-8")
                chunks.append({"type": "image", "content": b64_img})
    else:
        content = (await file.read()).decode('utf-8')
        text_chunks = [c.strip() for c in content.split('\n\n') if len(c.strip()) > 20]
        chunks.extend([{"type": "text", "content": c} for c in text_chunks])
        
    if not chunks: return {"message": "No extractable content found."}
    
    try:
        embeddings, valid_chunks = [], []
        for chunk in chunks:
            res = requests.post(f"{INFERENCE_API_URL}/embed", json=chunk)
            if res.status_code == 200:
                embeddings.append(res.json()["embedding"])
                valid_chunks.append(chunk)
                
        if embeddings:
            emb_matrix = np.vstack(embeddings).astype("float32")
            if index is None: index = faiss.IndexFlatL2(emb_matrix.shape[1])
            index.add(emb_matrix)
            doc_store.extend(valid_chunks)
            
            faiss.write_index(index, FAISS_PATH)
            with open(DOCS_PATH, "w") as f: json.dump(doc_store, f)
            
        return {"message": f"Ingested {len(valid_chunks)} mixed media blocks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FIX: Allow multiple images to be received
@app.post("/chat")
async def chat(query: str = Form(...), history: str = Form("[]"), images: Optional[List[UploadFile]] = File(None)):
    chat_history = json.loads(history)
    retrieved_context = []
    
    if index is not None and len(doc_store) > 0:
        embed_res = requests.post(f"{INFERENCE_API_URL}/embed", json={"type": "text", "content": query}).json()
        D, I = index.search(np.array([embed_res["embedding"]]).astype("float32"), min(15, len(doc_store)))
        retrieved = [doc_store[i] for i in I[0] if i < len(doc_store)]
        
        rerank_payload = {"query": query, "documents": retrieved}
        scores = requests.post(f"{INFERENCE_API_URL}/rerank", json=rerank_payload).json()["scores"]
        retrieved_context = [doc for _, doc in sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)][:3]

    current_msg_content = []
    
    if retrieved_context:
        current_msg_content.append({"type": "text", "text": "Extracted Medical Context:"})
        for doc in retrieved_context:
            if doc["type"] == "text":
                current_msg_content.append({"type": "text", "text": f"---\n{doc['content']}\n---"})
            elif doc["type"] == "image":
                current_msg_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{doc['content']}"}})

    # FIX: Loop through all directly uploaded UI images
    if images:
        for img in images:
            if img.filename:
                base64_img = base64.b64encode(await img.read()).decode('utf-8')
                mime_type = img.content_type or "image/jpeg"
                current_msg_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}})

    current_msg_content.append({"type": "text", "text": f"\nQuestion:\n{query}"})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history + [{"role": "user", "content": current_msg_content}]

    payload = {
        "model": "pubmed-adapter",
        "messages": messages,
        "stream": True, 
        "max_tokens": 1024
    }

    def stream():
        res = requests.post(VLLM_API_URL, json=payload, stream=True)
        if res.status_code != 200:
            yield f"data: {json.dumps({'error': res.text})}\n\n".encode('utf-8')
            return
        for line in res.iter_lines():
            if line: yield line + b"\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=5000)
