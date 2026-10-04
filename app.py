"""
app.py — Streamlit web interface for the RAG PDF Chatbot.

Run with:
    streamlit run app.py

Features:
  - Upload PDF files via the sidebar
  - Ask questions about the uploaded documents
  - See the LLM's answer with source citations
  - View the retrieved chunks and their similarity scores
  - Chat history within the session
"""

import os
import tempfile

import streamlit as st

from src.rag_pipeline import RAGPipeline


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("RAG PDF Chatbot")
st.caption("Ask questions about your PDF documents — powered by local AI")


# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


# ------------------------------------------------------------------
# Sidebar — Document upload and settings
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with.",
    )

    # Settings
    st.divider()
    st.header("Settings")

    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant chunks to send to the LLM as context.",
    )

    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=200,
        max_value=1000,
        value=500,
        step=50,
        help="Target size for text chunks. Smaller = more precise, larger = more context.",
    )

    # Process uploaded files
    if uploaded_files:
        process_btn = st.button(
            "Process Documents",
            type="primary",
            use_container_width=True,
        )

        if process_btn:
            with st.spinner("Initializing pipeline..."):
                rag = RAGPipeline(
                    chunk_size=chunk_size,
                    top_k=top_k,
                )

            # Check if Ollama is available
            if not rag.llm.is_available():
                st.error(
                    "Ollama is not running. Start it with `ollama serve` "
                    "and make sure `llama3.2` is pulled."
                )
            else:
                # Save uploaded files to a temp directory
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(tmp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    # Load and index the documents
                    with st.spinner(
                        f"Processing {len(uploaded_files)} document(s)..."
                    ):
                        rag.load_documents(tmp_dir)

                st.session_state.rag = rag
                st.session_state.documents_loaded = True
                st.session_state.messages = []  # Clear chat for new docs
                st.success(
                    f"Indexed {rag.document_count} document(s), "
                    f"{rag.chunk_count} chunks"
                )

    # Status display
    st.divider()
    st.header("Status")
    if st.session_state.rag:
        status = st.session_state.rag.status()
        st.write(f"**Documents:** {status['documents']}")
        st.write(f"**Chunks:** {status['chunks']}")
        st.write(f"**Embedding model:** {status['embedding_model']}")
        st.write(f"**LLM model:** {status['llm_model']}")
        st.write(f"**LLM available:** {'Yes' if status['llm_available'] else 'No'}")
    else:
        st.info("Upload PDFs and click 'Process Documents' to start.")


# ------------------------------------------------------------------
# Main area — Chat interface
# ------------------------------------------------------------------

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources expander for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View sources", expanded=False):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**Source {i}** — {source['source']}, "
                        f"Page {source['page_number']} "
                        f"(score: {source['score']:.4f})"
                    )
                    st.text(source["text"][:300])
                    if len(source["text"]) > 300:
                        st.caption(f"... ({len(source['text'])} total chars)")
                    st.divider()


# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.documents_loaded:
        st.warning("Please upload and process documents first.")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag.ask(prompt, top_k=top_k)

            st.markdown(result["answer"])

            # Show sources
            if result["sources"]:
                with st.expander("View sources", expanded=False):
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(
                            f"**Source {i}** — {source['source']}, "
                            f"Page {source['page_number']} "
                            f"(score: {source['score']:.4f})"
                        )
                        st.text(source["text"][:300])
                        if len(source["text"]) > 300:
                            st.caption(
                                f"... ({len(source['text'])} total chars)"
                            )
                        st.divider()

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
