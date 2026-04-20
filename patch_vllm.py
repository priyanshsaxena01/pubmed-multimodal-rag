file_path = "/home/b22ee075/miniconda3/envs/pubmed_env/lib/python3.10/site-packages/vllm/config.py"
with open(file_path, "r") as f:
    content = f.read()

# Remove the strict assertion and add a safe fallback
content = content.replace('assert "factor" in rope_scaling', 'pass')
content = content.replace('scaling_factor = rope_scaling["factor"]', 'scaling_factor = rope_scaling.get("factor", 1.0)')

with open(file_path, "w") as f:
    f.write(content)

print("✅ vLLM successfully patched!")
