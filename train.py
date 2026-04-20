import torch
import wandb
import json
import ast
from PIL import ImageFile
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# Fix for the PIL "Truncated File Read" warning
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. Initialize WandB
wandb.init(
    project="pubmed-multimodal-rag",
    name="qwen-4b-qlora-finetune",
    config={
        "learning_rate": 2e-4,
        "architecture": "Qwen3-VL-4B",  # DOUBLE CHECK THIS HF REPO NAME
        "dataset": "PubMedVision-enhanced-z000",
        "epochs": 1,
    }
)

print("Loading dataset...")
dataset = load_dataset("alvinl29/PubMedVision-enhanced", "z000", split="train")

# ==========================================
# FIX: Robustly parse and clean the dataset
# ==========================================
def parse_and_clean(example):
    convs = example["conversations"]
    
    if isinstance(convs, list):
        return example
        
    if isinstance(convs, str):
        if not convs.strip() or convs.strip().lower() == "none":
            example["conversations"] = None
            return example
            
        try:
            example["conversations"] = json.loads(convs)
        except json.JSONDecodeError:
            try:
                example["conversations"] = ast.literal_eval(convs)
            except (ValueError, SyntaxError):
                example["conversations"] = None
                
    return example

print("Cleaning and formatting conversations...")
dataset = dataset.map(parse_and_clean, num_proc=4)
original_len = len(dataset)
dataset = dataset.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) >= 2)
print(f"Filtered out {original_len - len(dataset)} corrupted/empty rows.")
# ==========================================


# 2. V100 Safe Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=True
)

model_id = "Qwen/Qwen3-VL-4B-Instruct" 

# OOM FIX: Limit image resolution so massive images don't explode the token count
processor = AutoProcessor.from_pretrained(model_id, max_pixels=1024 * 1024)

model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto", 
    torch_dtype=torch.float16
)

# OOM & WARNING FIXES: Prepare model for efficient training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False             # Required for gradient checkpointing
model.enable_input_require_grads()         # Fixes the "requires_grad=True" warning

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


# 3. Collate Function
def collate_fn(examples):
    texts, images = [], []
    for ex in examples:
        convs = ex["conversations"] 
            
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": ex["image"]}, 
                    {"type": "text", "text": convs[0]["value"]}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": convs[1]["value"]}
                ]
            }
        ]
        
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        img, _ = process_vision_info(messages)
        images.extend(img)
        
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    
    return batch


# 4. Training
training_args = TrainingArguments(
    output_dir="./qwen-adapter", 
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=16,
    learning_rate=2e-4, 
    fp16=True, 
    num_train_epochs=1, 
    logging_steps=10,
    eval_strategy="no",
    
    # OOM & CRASH FIXES:
    gradient_checkpointing=True,                             # Drastically reduces VRAM
    gradient_checkpointing_kwargs={'use_reentrant': False},  # Silences warning
    optim="paged_adamw_8bit",                                # Uses 8-bit memory-efficient optimizer
    dataloader_num_workers=4,                                # Speeds up data loading
    
    # SAFE SAVING (saves every ~500 steps so you don't lose data if DGX crashes)
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    
    report_to="wandb", 
    run_name="pubmed-qlora-v2-4b",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=dataset, 
    data_collator=collate_fn
)

trainer.train()

# 5. Save and Finish
model.push_to_hub("b22ee075/Qwen3-VL-4B-PubMed")
processor.push_to_hub("b22ee075/Qwen3-VL-4B-PubMed")

wandb.finish()
print("Training Complete!")