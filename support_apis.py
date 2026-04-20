import sys
import torch
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoProcessor, AutoModel
import uvicorn
from qwen_vl_utils import process_vision_info

# Dynamically link the official Qwen repository
sys.path.append("./qwen3_vl_wrapper")
from src.models.qwen3_vl_reranker import Qwen3VLReranker

app = FastAPI()

print("Loading Qwen3-VL Multimodal Embedding Model...")
embed_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-Embedding-2B")
embed_model = AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-2B", torch_dtype=torch.float16).cuda()
embed_model.eval()

print("Loading Qwen3-VL Multimodal Reranker...")
rerank_model = Qwen3VLReranker(
    model_name_or_path="Qwen/Qwen3-VL-Reranker-2B",
    torch_dtype=torch.float16
)

class EmbedRequest(BaseModel): 
    type: str  # "text" or "image"
    content: str  # text string or base64 string

class DocItem(BaseModel):
    type: str
    content: str

class RerankRequest(BaseModel): 
    query: str
    documents: List[DocItem]

@app.post("/embed")
def get_embedding(req: EmbedRequest):
    messages = [{"role": "user", "content": []}]
    if req.type == "text":
        messages[0]["content"].append({"type": "text", "text": req.content})
    elif req.type == "image":
        # Format base64 for Qwen Vision Processor
        messages[0]["content"].append({"type": "image", "image": f"data:image/jpeg;base64,{req.content}"})
        
    text = embed_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = embed_processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad(): 
        emb = embed_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().tolist()
    return {"embedding": emb[0]}

@app.post("/rerank")
def get_rerank_scores(req: RerankRequest):
    if not req.documents: 
        return {"scores": []}
        
    formatted_docs = []
    for doc in req.documents:
        if doc.type == "text":
            formatted_docs.append({"text": doc.content})
        elif doc.type == "image":
            formatted_docs.append({"image": f"data:image/jpeg;base64,{doc.content}"})
            
    rerank_inputs = {
        "instruction": "Retrieve highly relevant medical charts, imaging, or clinical text.",
        "query": {"text": req.query},
        "documents": formatted_docs
    }
    
    with torch.no_grad():
        scores = rerank_model.process(rerank_inputs)
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy().tolist()
        return {"scores": scores}

if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8001)
