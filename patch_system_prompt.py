file_path = "./orchestrator.py"
with open(file_path, "r") as f:
    content = f.read()

# Define the rigorous medical system prompt
system_prompt_code = """
    SYSTEM_PROMPT = \"\"\"You are a highly advanced clinical AI assistant fine-tuned on PubMed literature.
Your primary function is to assist medical professionals by analyzing clinical notes, extracting literature context, and evaluating medical imaging.

CRITICAL INSTRUCTIONS:
1. Grounding: Base your answers STRICTLY on the 'Extracted Medical Context' provided in the prompt. 
2. Honesty: If the provided context or image does not contain the answer, you must state: "I cannot determine this based on the provided documents/image." Do not hallucinate or guess.
3. Tone: Maintain a highly professional, objective, and clinical tone.
4. Formatting: Use clear bullet points for differential diagnoses, findings, or literature summaries.
5. Disclaimer: You are an AI assistant. You do not provide definitive medical diagnoses.\"\"\"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history + [{"role": "user", "content": current_msg_content}]
"""

# Replace the old message construction with the new one
import re
new_content = re.sub(
    r"messages = chat_history \+ \[\{\"role\": \"user\", \"content\": current_msg_content\}\]",
    system_prompt_code.strip(),
    content
)

with open(file_path, "w") as f:
    f.write(new_content)

print("✅ System Prompt successfully injected into Orchestrator!")
