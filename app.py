import streamlit as st
import time
from pypdf import PdfReader

from rag.embeddings import get_jina_embeddings
from rag.vision import describe_image
from rag.chunking import chunk_text
from rag.retriever import FAISSRetriever
from rag.reranker import simple_rerank
from rag.llm import ask_llm


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Enterprise Multimodal RAG",
    page_icon="✨",
    layout="centered",
)

# ===============================
# PROFESSIONAL LIGHT UI
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    background-color: #f4f6fb;
}

/* Title */
.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #475569;
    margin-bottom: 2rem;
    font-size: 18px;
}

/* Card Design */
.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #1d4ed8, #0891b2);
}

/* Chat Messages */
[data-testid="stChatMessage-user"] {
    background-color: #e0f2fe;
    border-radius: 10px;
}

[data-testid="stChatMessage-assistant"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}

/* Input boxes */
input {
    border-radius: 10px !important;
}

/* Section headings */
h3 {
    color: #1e293b;
}
</style>
""", unsafe_allow_html=True)


# ===============================
# HEADER
# ===============================
st.markdown("<div class='main-title'>Enterprise Multimodal RAG</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload documents or images and chat with your own AI knowledge base.</div>", unsafe_allow_html=True)


# ===============================
# SESSION STATE
# ===============================
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None


# ===============================
# API CONFIG SECTION
# ===============================
st.markdown("### 🔐 API Configuration")

with st.container():
    groq_key = st.text_input("Groq API Key", type="password")
    jina_key = st.text_input("Jina API Key", type="password")

    model = st.selectbox(
        "Select Model",
        ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]
    )


# ===============================
# KNOWLEDGE UPLOAD SECTION
# ===============================
st.markdown("### 📚 Upload Knowledge")

txt_file = st.file_uploader("Upload Document (TXT or PDF)", type=["txt", "pdf"])
img_file = st.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

index_button = st.button("🚀 Index Knowledge")


# ===============================
# INDEXING PROCESS
# ===============================
if index_button:

    if not groq_key or not jina_key:
        st.warning("Please enter both API keys.")
    elif not (txt_file or img_file):
        st.warning("Please upload at least one document or image.")
    else:
        with st.spinner("Indexing knowledge..."):

            chunks = []
            metadata = []

            # DOCUMENT
            if txt_file:
                if txt_file.name.endswith(".pdf"):
                    reader = PdfReader(txt_file)
                    raw_text = "\n".join(
                        [p.extract_text() for p in reader.pages if p.extract_text()]
                    )
                else:
                    raw_text = txt_file.read().decode("utf-8")

                text_chunks = chunk_text(raw_text)
                chunks.extend(text_chunks)
                metadata.extend([{"type": "text"} for _ in text_chunks])

            # IMAGE
            if img_file:
                image_bytes = img_file.read()
                vision_text = describe_image(image_bytes, groq_key)
                if vision_text:
                    chunks.append("Image description: " + vision_text)
                    metadata.append({"type": "image"})

            embeddings = get_jina_embeddings(chunks, jina_key)
            retriever = FAISSRetriever(embeddings, metadata)

            st.session_state.retriever = retriever
            st.session_state.chunks = chunks

        st.success("Knowledge indexed successfully ✅")


# ===============================
# CHAT SECTION
# ===============================
st.markdown("### 💬 Chat")

query = st.chat_input(
    "Ask a question about your uploaded knowledge...",
    disabled=st.session_state.retriever is None
)

if query and st.session_state.retriever:

    with st.chat_message("user"):
        st.write(query)

    start = time.time()

    query_emb = get_jina_embeddings([query], jina_key)
    ids = st.session_state.retriever.search(query_emb, top_k=5)

    retrieved_texts = [st.session_state.chunks[i] for i in ids]
    reranked = simple_rerank(query, retrieved_texts)
    top_sources = reranked[:3]

    context = "\n\n".join(top_sources)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_llm(context, query, groq_key, model)
            st.write(answer)

    latency = round(time.time() - start, 2)
    st.caption(f"⚡ Response generated in {latency} seconds")
