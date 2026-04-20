import re

file_path = "./qwen3_vl_wrapper/src/models/qwen3_vl_reranker.py"
with open(file_path, "r") as f:
    content = f.read()

# Aggressively catch any spacing, line breaks, or variable variations
new_content = re.sub(
    r"AutoProcessor\.from_pretrained\(\s*(?:self\.)?(?:pretrained_)?model_name_or_path",
    "AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct'",
    content
)

with open(file_path, "w") as f:
    f.write(new_content)

print("✅ Aggressive processor patch applied!")
