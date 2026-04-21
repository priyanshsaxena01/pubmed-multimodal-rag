import streamlit as st
import requests
import json

st.set_page_config(page_title="PubMed AI", page_icon="🩺", layout="wide")
BACKEND_URL = "http://localhost:5000"

# --- Sidebar ---
with st.sidebar:
    st.header("📚 Dynamic RAG")
    uploaded_docs = st.file_uploader("Upload Clinical Notes", type=["txt", "pdf"], accept_multiple_files=True)
    if st.button("Index Documents") and uploaded_docs:
        with st.spinner("Embedding..."):
            for doc in uploaded_docs:
                requests.post(f"{BACKEND_URL}/upload_doc", files={"file": (doc.name, doc.getvalue(), doc.type)})
            st.success("✅ Knowledge Base Updated!")
            
    st.markdown("---")
    st.header("🖼️ Vision Scan")
    
    # FIX: Enable multiple file selection
    uploaded_images = st.file_uploader("Upload Medical Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_images:
        for img in uploaded_images:
            st.image(img, use_container_width=True)

    st.markdown("---")
    st.header("⚙️ Controls")
    
    if st.button("💬 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    if st.button("🗑️ Clear RAG Memory", use_container_width=True, type="secondary"):
        try:
            res = requests.post(f"{BACKEND_URL}/clear_db")
            if res.status_code == 200:
                st.success("RAG Memory Cleared!")
            else:
                st.error("Failed to clear database.")
        except Exception as e:
            st.error("Backend offline.")

# --- Main Chat UI ---
st.title("🩺 PubMed Fine-Tuned AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a diagnostic or analytical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # FIX: Format multiple images into a list of tuples for requests.post
        files = [("images", (img.name, img.getvalue(), img.type)) for img in uploaded_images] if uploaded_images else None
        
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        data = {"query": prompt, "history": json.dumps(history)}
        
        try:
            res = requests.post(f"{BACKEND_URL}/chat", data=data, files=files, stream=True)
            for line in res.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: ") and "[DONE]" not in decoded:
                        try:
                            json_data = json.loads(decoded[6:])
                            if "error" in json_data:
                                st.error(f"Engine Error: {json_data['error']}")
                                break
                            if "choices" in json_data and "content" in json_data["choices"][0].get("delta", {}):
                                full_response += json_data["choices"][0]["delta"]["content"]
                                placeholder.markdown(full_response + "▌")
                        except: pass
            placeholder.markdown(full_response)
            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"🚨 Connection failed: {e}")
