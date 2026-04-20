import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

print("1. Loading Base Model (Using CPU to prevent VRAM spikes)...")
model_id = "Qwen/Qwen3-VL-4B-Instruct"
adapter_id = "b22ee075/Qwen3-VL-4B-PubMed"

processor = AutoProcessor.from_pretrained(model_id)
base_model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cpu" 
)

print("2. Attaching LoRA Adapter...")
model = PeftModel.from_pretrained(base_model, adapter_id)

print("3. Baking Weights together (This will take ~2 minutes)...")
merged_model = model.merge_and_unload()

print("4. Saving Native Merged Model to ./qwen-pubmed-merged...")
merged_model.save_pretrained("./qwen-pubmed-merged")
processor.save_pretrained("./qwen-pubmed-merged")

print("✅ Merge Complete!")
