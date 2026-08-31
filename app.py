import streamlit as st
import os
from pdf_utils import load_and_chunk_pdfs
from embeddings import EmbeddingManager
from retrieval import HybridRetriever
from qa import QASystem
import tempfile

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI_Note_Assistant",
    page_icon="🧠",
    layout="wide"
)

# ------------------ CUSTOM CSS (SPIDERMAN HUD) ------------------
st.markdown("""
<style>

/* GLOBAL */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background: radial-gradient(circle at top, #0b0f2a, #050816);
    color: white;
}

/* TITLE */
.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #ff003c, #0066ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glowTitle 2s infinite alternate;
}

@keyframes glowTitle {
    from { text-shadow: 0 0 10px #ff003c; }
    to { text-shadow: 0 0 20px #0066ff; }
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f2a, #050816);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* GLASS CARD */
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 20px rgba(0, 102, 255, 0.2);
    transition: 0.3s;
}

.glass:hover {
    box-shadow: 0 0 30px rgba(255, 0, 60, 0.6);
    transform: translateY(-3px);
}

/* CHAT */
.stChatMessage {
    border-radius: 18px !important;
    padding: 12px !important;
    margin-bottom: 10px !important;
    transition: 0.3s;
}

.stChatMessage:hover {
    box-shadow: 0 0 20px rgba(255,0,60,0.4);
}

/* USER MESSAGE */
[data-testid="stChatMessage-user"] {
    background: linear-gradient(135deg, #0066ff, #001f5c);
}

/* BOT MESSAGE */
[data-testid="stChatMessage-assistant"] {
    background: linear-gradient(135deg, #ff003c, #5c0015);
}

/* BUTTON */
.stButton button {
    background: linear-gradient(90deg, #ff003c, #0066ff);
    border: none;
    border-radius: 12px;
    color: white;
    transition: 0.3s;
}

.stButton button:hover {
    box-shadow: 0 0 20px #ff003c;
    transform: scale(1.05);
}

/* INPUT */
textarea {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    color: black !important;
}

/* ANIMATION */
.fade-in {
    animation: fadeIn 0.6s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown('<div class="title">AI_Note_Assistant</div>', unsafe_allow_html=True)
st.markdown("### 🕸️ AI-powered PDF Intelligence System")

# ------------------ SESSION STATE ------------------
for key in ["documents", "embedding_manager", "retriever", "qa_system", "chat_history", "processed_files"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "history" in key or "files" in key else None

# ------------------ API KEYS ------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("Add GROQ_API_KEY in secrets")
    st.stop()

qdrant_api_key = st.secrets.get("QDRANT_API_KEY", None)
qdrant_endpoint = st.secrets.get("QDRANT_ENDPOINT", None)
use_qdrant = bool(qdrant_api_key and qdrant_endpoint)

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("## 🕷️ Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Drop files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    current_files = [f.name for f in uploaded_files]

    if current_files != st.session_state.processed_files:
        with st.spinner("⚡ Processing..."):
            temp_paths = []

            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    temp_paths.append(tmp.name)

            documents = load_and_chunk_pdfs(temp_paths, chunk_size=150, overlap=30)

            for path in temp_paths:
                os.unlink(path)

            st.session_state.documents = documents

            embedding_manager = EmbeddingManager(
                use_qdrant=use_qdrant,
                qdrant_api_key=qdrant_api_key,
                qdrant_endpoint=qdrant_endpoint
            )
            embedding_manager.build_index(documents)

            retriever = HybridRetriever(embedding_manager, documents, top_k=7)

            qa_system = QASystem(groq_api_key, retriever)

            st.session_state.embedding_manager = embedding_manager
            st.session_state.retriever = retriever
            st.session_state.qa_system = qa_system
            st.session_state.chat_history = []
            st.session_state.processed_files = current_files

            st.sidebar.success("🧠 Engine Ready")

# ------------------ MAIN ------------------
if st.session_state.documents:

    st.markdown('<div class="glass fade-in">💬 Conversation</div>', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""

            for chunk in st.session_state.qa_system.answer_question(
                question,
                chat_history=st.session_state.chat_history[-10:]
            ):
                full += chunk
                placeholder.markdown(full + " ▌")

            placeholder.markdown(full)

        st.session_state.chat_history.append({"role": "assistant", "content": full})

    if st.sidebar.button("🗑️ Reset"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.markdown('<div class="glass fade-in">👈 Upload PDFs to activate the engine</div>', unsafe_allow_html=True)
