"""
app.py — Modern UI for the RAG PDF Chatbot.

Matches the custom design system:
  - 3-column layout (Sidebar | Chat | Source Inspector)
  - Dark & Light mode toggle with the brand color palette
  - Interactive source pills & live document citation inspector
  - Powered by our from-scratch RAG pipeline
"""

import datetime
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
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # Default to dark mode as shown in mockup
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "document_info" not in st.session_state:
    st.session_state.document_info = []
if "active_source" not in st.session_state:
    st.session_state.active_source = None
if "inspector_page_idx" not in st.session_state:
    st.session_state.inspector_page_idx = 0


# ------------------------------------------------------------------
# Dynamic CSS Design System (Brand Palette + Dark/Light Mode)
# ------------------------------------------------------------------
is_dark = st.session_state.theme == "dark"

css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  /* ===== BRAND ===== */
  --beige: #EEEBD0;
  --powder-blush: #EBB3A9;
  --petal-rouge: #E87EA1;
  --coral: #E86252;
  --neon-pink: #EE2677;

  /* ===== ACTIVE THEME ({st.session_state.theme.upper()}) ===== */
  --bg: {'#0E1016' if is_dark else '#FAF9F4'};
  --surface: {'#171922' if is_dark else '#FFFFFF'};
  --surface-2: {'#20222C' if is_dark else '#F5F0E5'};
  --border: {'#30323D' if is_dark else '#E8DCD5'};
  --text: {'#F8F5F2' if is_dark else '#202027'};
  --text-muted: {'#AAA7AE' if is_dark else '#77727A'};

  /* ===== CHAT ===== */
  --user-bg: {'#321A28' if is_dark else '#F8C4D5'};
  --user-border: {'#EE2677' if is_dark else '#E87EA1'};
  --user-text: {'#F8F5F2' if is_dark else '#202027'};

  --assistant-bg: {'#20222C' if is_dark else '#F7F1E7'};
  --assistant-border: {'#30323D' if is_dark else '#E8DCD5'};
  --assistant-text: {'#F8F5F2' if is_dark else '#202027'};

  /* ===== GRADIENTS ===== */
  --brand-gradient: linear-gradient(135deg, #E87EA1, #EE2677);
  --soft-gradient: linear-gradient(135deg, #EEEBD0, #EBB3A9);
}}

/* Base Page Styling */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}}

[data-testid="stHeader"] {{
    background-color: transparent !important;
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}}

[data-testid="stSidebar"] hr {{
    border-color: var(--border) !important;
}}

/* Typography */
.hero-title {{
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}}

.gradient-text {{
    background: var(--brand-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}}

.hero-subtitle {{
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-bottom: 1.8rem;
}}

/* Sidebar Brand Header */
.brand-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 20px 0;
}}

.brand-icon {{
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: var(--brand-gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 18px;
    box-shadow: 0 4px 12px rgba(238, 38, 119, 0.25);
}}

.brand-name {{
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
}}

/* Sidebar Nav Pills */
.nav-pill {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 6px;
    font-weight: 500;
    font-size: 0.92rem;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s ease;
}}

.nav-pill.active {{
    background: {'rgba(238, 38, 119, 0.15)' if is_dark else '#FCE5ED'};
    color: var(--neon-pink);
    font-weight: 600;
}}

/* Document Card */
.doc-card {{
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 10px;
}}

.doc-icon {{
    width: 34px;
    height: 34px;
    background: {'rgba(232, 126, 161, 0.2)' if is_dark else '#FDE7ED'};
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--coral);
    font-size: 16px;
}}

.doc-info {{
    flex: 1;
    overflow: hidden;
}}

.doc-name {{
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

.doc-meta {{
    font-size: 0.74rem;
    color: var(--text-muted);
    margin-top: 2px;
}}

/* User Profile Badge */
.profile-badge {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 6px;
    margin-top: 16px;
}}

.avatar-circle {{
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--brand-gradient);
    color: white;
    font-weight: 600;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(238, 38, 119, 0.3);
}}

.avatar-circle-sm {{
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--brand-gradient);
    color: white;
    font-weight: 600;
    font-size: 0.75rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}}

.bot-avatar {{
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: var(--coral);
    color: white;
    font-size: 17px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(232, 98, 82, 0.3);
}}

/* Chat Bubbles */
.chat-row {{
    display: flex;
    margin-bottom: 20px;
    gap: 12px;
}}

.chat-row.user {{
    justify-content: flex-end;
}}

.chat-row.assistant {{
    justify-content: flex-start;
}}

.user-bubble {{
    background: var(--user-bg);
    border: 1px solid var(--user-border);
    color: var(--user-text);
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 78%;
    font-size: 0.94rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}

.assistant-card {{
    background: var(--assistant-bg);
    border: 1px solid var(--assistant-border);
    color: var(--assistant-text);
    padding: 16px 20px;
    border-radius: 18px;
    max-width: 86%;
    font-size: 0.94rem;
    line-height: 1.6;
    box-shadow: 0 4px 16px rgba(0,0,0,{'0.2' if is_dark else '0.04'});
}}

.message-time {{
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 6px;
    text-align: right;
}}

/* Source Pills */
.sources-container {{
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px dashed var(--border);
}}

.sources-label {{
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}}

.source-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {'rgba(232, 126, 161, 0.12)' if is_dark else '#FDEBF1'};
    border: 1px solid {'rgba(232, 126, 161, 0.4)' if is_dark else '#F4B6CA'};
    color: var(--petal-rouge);
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    cursor: pointer;
    text-decoration: none;
    margin-right: 6px;
    margin-bottom: 6px;
    transition: all 0.2s ease;
}}

/* Inspector Panel */
.inspector-panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,{'0.25' if is_dark else '0.05'});
}}

.inspector-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}}

.inspector-title {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

.inspector-card {{
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}}

.page-badge {{
    background: var(--petal-rouge);
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
}}

.inspector-text {{
    font-size: 0.85rem;
    color: var(--text);
    line-height: 1.6;
    max-height: 280px;
    overflow-y: auto;
    padding-right: 6px;
    margin-top: 10px;
    font-family: inherit;
}}

.citation-card {{
    background: {'rgba(232, 126, 161, 0.08)' if is_dark else '#FDF3F6'};
    border: 1px solid {'rgba(232, 126, 161, 0.25)' if is_dark else '#FAD4E0'};
    border-radius: 12px;
    padding: 16px;
    position: relative;
    margin-top: 14px;
}}

.quote-mark {{
    font-size: 2.2rem;
    line-height: 1;
    color: var(--coral);
    font-family: Georgia, serif;
    margin-bottom: -10px;
}}

.citation-quote {{
    font-size: 0.82rem;
    font-style: italic;
    color: var(--text);
    line-height: 1.5;
}}

.citation-footer {{
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--coral);
    margin-top: 8px;
}}

/* Form Buttons Styling */
div.stButton > button {{
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}}

/* Brand Primary Button */
div.stButton > button[kind="primary"] {{
    background: var(--brand-gradient) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(238, 38, 119, 0.3) !important;
}}

/* Send Button in Chat Input */
[data-testid="stChatInput"] {{
    border-radius: 24px !important;
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
}}

[data-testid="stChatInput"] textarea {{
    color: var(--text) !important;
}}
</style>
"""

st.markdown(css, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Sidebar: Navigation, Documents, and Profile
# ------------------------------------------------------------------
with st.sidebar:
    # 1. Brand Logo Header
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-icon">📄</div>
            <div class="brand-name">RAG PDF Chatbot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Navigation Pills
    st.markdown(
        """
        <div class="nav-pill active">💬 Chat</div>
        <div class="nav-pill">📁 Documents</div>
        <div class="nav-pill">⚙️ Settings</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # 3. Documents Section
    col_d1, col_d2 = st.columns([1.8, 1])
    with col_d1:
        st.markdown(
            "<span style='font-weight: 600; font-size: 0.92rem;'>Your documents</span>",
            unsafe_allow_html=True,
        )

    # File Uploader in an expander / popover
    with st.expander("＋ Add PDF", expanded=not st.session_state.documents_loaded):
        uploaded_files = st.file_uploader(
            "Select PDF",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        chunk_size = st.slider("Chunk size", 200, 1000, 500, 50)
        top_k = st.slider("Top-k chunks", 1, 10, 5)

        if uploaded_files and st.button("Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents with RAG pipeline..."):
                rag = RAGPipeline(chunk_size=chunk_size, top_k=top_k)

                if not rag.llm.is_available():
                    st.error("Ollama is not running. Start with `ollama serve`.")
                else:
                    doc_details = []
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(tmp_dir, uploaded_file.name)
                            bytes_data = uploaded_file.getbuffer()
                            with open(file_path, "wb") as f:
                                f.write(bytes_data)

                            size_kb = round(len(bytes_data) / 1024, 1)
                            doc_details.append({
                                "name": uploaded_file.name,
                                "size": f"{size_kb} KB",
                            })

                        rag.load_documents(tmp_dir)

                    for d in doc_details:
                        d["chunks"] = f"{rag.chunk_count} chunks"

                    st.session_state.rag = rag
                    st.session_state.documents_loaded = True
                    st.session_state.document_info = doc_details
                    st.session_state.messages = []
                    st.session_state.active_source = None
                    st.success("Indexed successfully!")
                    st.rerun()

    # Show Document Cards
    if st.session_state.document_info:
        for doc in st.session_state.document_info:
            st.markdown(
                f"""
                <div class="doc-card">
                    <div class="doc-icon">📄</div>
                    <div class="doc-info">
                        <div class="doc-name">{doc['name']}</div>
                        <div class="doc-meta">{doc['size']} • {doc['chunks']}</div>
                    </div>
                    <div style="color: var(--text-muted); font-size: 16px;">⋮</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="doc-card" style="opacity: 0.6;">
                <div class="doc-icon">📄</div>
                <div class="doc-info">
                    <div class="doc-name">No documents yet</div>
                    <div class="doc-meta">Upload a PDF to start</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.divider()

    # 4. Theme Toggle (Light mode / Dark mode)
    theme_btn_label = "☀️ Light mode" if is_dark else "🌙 Dark mode"
    if st.button(theme_btn_label, use_container_width=True):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

    # 5. User Profile Footer
    st.markdown(
        """
        <div class="profile-badge">
            <div class="avatar-circle">HR</div>
            <div style="flex: 1;">
                <div style="font-weight: 600; font-size: 0.88rem;">Hafiza Rimah</div>
                <div style="font-size: 0.72rem; color: var(--text-muted);">Admin</div>
            </div>
            <div style="color: var(--text-muted); font-size: 14px;">⌄</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Main Area: Header + 2-Column Responsive Layout
# ------------------------------------------------------------------

# Hero Header
st.markdown(
    """
    <div class="hero-title">Ask your <span class="gradient-text">PDF documents</span></div>
    <div class="hero-subtitle">Get accurate answers with sources from your uploaded files.</div>
    """,
    unsafe_allow_html=True,
)

# Determine layout: Split if an inspector source is selected, otherwise full-width
has_active_inspector = st.session_state.active_source is not None

if has_active_inspector:
    chat_col, inspector_col = st.columns([1.65, 1.1], gap="large")
else:
    chat_col = st.container()
    inspector_col = None


# ------------------------------------------------------------------
# Chat Stream (Middle Column)
# ------------------------------------------------------------------
with chat_col:
    # Render messages
    for msg_idx, msg in enumerate(st.session_state.messages):
        current_time = msg.get("time", "14:22")

        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-row user">
                    <div>
                        <div class="user-bubble">{msg['content']}</div>
                        <div class="message-time">{current_time}</div>
                    </div>
                    <div class="avatar-circle-sm" style="margin-top: 4px;">HR</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Assistant Message Card
            st.markdown(
                f"""
                <div class="chat-row assistant">
                    <div class="bot-avatar">🤖</div>
                    <div class="assistant-card">
                        <div>{msg['content']}</div>
                        <div class="message-time" style="text-align: right; margin-top: 10px;">
                            {current_time} &nbsp; 📋 &nbsp; 👍 &nbsp; 👎
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Sources Pills as interactive clickable Streamlit buttons
            if msg.get("sources"):
                st.markdown(
                    "<div style='margin-left: 46px; margin-top: -10px; margin-bottom: 4px;' class='sources-label'>Sources:</div>",
                    unsafe_allow_html=True,
                )
                pill_cols = st.columns(min(len(msg["sources"]), 4) + 1)
                for s_idx, source in enumerate(msg["sources"][:4]):
                    with pill_cols[s_idx]:
                        # Extract friendly section name or page label
                        doc_name = source["source"]
                        page_num = source["page_number"]
                        pill_label = f"📄 Page {page_num}"

                        if st.button(pill_label, key=f"src_pill_{msg_idx}_{s_idx}"):
                            st.session_state.active_source = source
                            st.session_state.inspector_page_idx = s_idx
                            st.rerun()

    # Chat Input Box
    if user_prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.documents_loaded:
            st.warning("Please upload and process a PDF document in the sidebar first.")
        else:
            time_now = datetime.datetime.now().strftime("%H:%M")

            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_prompt,
                "time": time_now,
            })

            with st.spinner("Generating answer with llama3.2..."):
                rag_result = st.session_state.rag.ask(user_prompt)

            # Save assistant message
            sources_list = rag_result.get("sources", [])
            st.session_state.messages.append({
                "role": "assistant",
                "content": rag_result["answer"],
                "sources": sources_list,
                "time": datetime.datetime.now().strftime("%H:%M"),
            })

            # Automatically show the top source in the inspector
            if sources_list:
                st.session_state.active_source = sources_list[0]
                st.session_state.inspector_page_idx = 0

            st.rerun()


# ------------------------------------------------------------------
# Right Column: Source Inspector (Live Citation & Document Viewer)
# ------------------------------------------------------------------
if has_active_inspector and inspector_col:
    with inspector_col:
        source_data = st.session_state.active_source

        # Find the full sources list from the last assistant message
        last_sources = []
        for m in reversed(st.session_state.messages):
            if m["role"] == "assistant" and m.get("sources"):
                last_sources = m["sources"]
                break

        curr_idx = st.session_state.inspector_page_idx + 1
        total_sources = len(last_sources) if last_sources else 1

        # Header with close button
        h_col1, h_col2 = st.columns([4, 1])
        with h_col1:
            st.markdown(
                f"""
                <div class="inspector-title">
                    <div class="doc-icon" style="width: 28px; height: 28px; font-size: 14px;">📄</div>
                    <div>
                        <div style="font-weight: 700; font-size: 0.95rem;">Source {curr_idx}</div>
                        <div style="font-size: 0.76rem; color: var(--text-muted);">{source_data['source']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h_col2:
            if st.button("✕", key="close_inspector"):
                st.session_state.active_source = None
                st.rerun()

        st.markdown("<hr style='margin: 10px 0; border-color: var(--border);'>", unsafe_allow_html=True)

        # Document Preview Card
        st.markdown(
            f"""
            <div class="inspector-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; font-size: 0.88rem; color: var(--text);">Document Preview</span>
                    <span class="page-badge">Page {source_data['page_number']}</span>
                </div>
                <div class="inspector-text">
                    {source_data['text']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Pagination controls
        p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
        with p_col1:
            if st.button("◀", disabled=st.session_state.inspector_page_idx <= 0):
                st.session_state.inspector_page_idx -= 1
                st.session_state.active_source = last_sources[st.session_state.inspector_page_idx]
                st.rerun()
        with p_col2:
            st.markdown(
                f"<div style='text-align: center; font-size: 0.82rem; color: var(--text-muted); padding-top: 6px;'>Source {curr_idx} of {total_sources}</div>",
                unsafe_allow_html=True,
            )
        with p_col3:
            if st.button("▶", disabled=st.session_state.inspector_page_idx >= total_sources - 1):
                st.session_state.inspector_page_idx += 1
                st.session_state.active_source = last_sources[st.session_state.inspector_page_idx]
                st.rerun()

        # Citation Card at bottom
        quote_excerpt = source_data["text"][:220].strip()
        if len(source_data["text"]) > 220:
            quote_excerpt += "..."

        st.markdown(
            f"""
            <div class="citation-card">
                <div class="quote-mark">“</div>
                <div class="citation-quote">"{quote_excerpt}"</div>
                <div class="citation-footer">Page {source_data['page_number']} — {source_data['source']} (Score: {source_data['score']:.4f})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
