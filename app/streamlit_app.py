"""
VerbaQuery-Enterprise: Streamlit Web Interface

Production-grade RAG interface with:
- Chat-based query interface
- Source citation display
- Relevance score visualization
- Query history
- Error handling
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation import QueryEngine
from src.ingestion import UploadProcessor
from config import get_settings


# Page configuration
st.set_page_config(
    page_title="VerbaQuery-Enterprise",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "upload_processor" not in st.session_state:
        st.session_state.upload_processor = UploadProcessor()
    if "last_upload_result" not in st.session_state:
        st.session_state.last_upload_result = None
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
        st.session_state.engine_status = "waiting"  # Waiting for PDF upload


def display_header():
    """Display application header."""
    st.title("📚 VerbaQuery-Enterprise")
    st.caption("Upload a PDF and ask questions about it")

    # Status indicator
    if st.session_state.engine_status == "ready":
        st.success("✓ Ready to answer questions")
    elif st.session_state.engine_status == "waiting":
        st.info("📤 Upload a PDF to get started")
    elif st.session_state.engine_status == "processing":
        st.info("⏳ Processing PDF...")
    else:
        st.error(f"✗ Error: {st.session_state.get('engine_error', 'Unknown error')}")


def display_sidebar():
    """Display sidebar with configuration and info."""
    with st.sidebar:
        # Upload Documents Section - at the top
        st.header("📤 Upload PDF")
        display_upload_section()

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.header("⚙️ System Info")

        settings = get_settings()

        st.markdown("**Configuration**")
        st.code(f"""
Model: {settings.openai_chat_model}
Embeddings: {settings.openai_embedding_model}
Chunk Size: {settings.chunk_size}
        """)

        st.markdown("**Pipeline**")
        st.caption("Hybrid Retrieval → Re-ranking → Generation")


def display_upload_section():
    """Display the PDF upload section in the sidebar."""
    settings = get_settings()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF to query",
        type=["pdf"],
        help=f"Max size: {settings.max_upload_size_mb}MB"
    )

    if uploaded_file is not None:
        st.caption("This will replace any previously indexed content.")
        # Process button
        if st.button("📥 Process & Index", key="process_btn"):
            process_uploaded_document(uploaded_file)

    # Show current indexed document
    if st.session_state.last_upload_result:
        result = st.session_state.last_upload_result
        if result.get("success"):
            st.info(f"📄 Current: **{result['filename']}**")
            st.caption(f"{result['pages']} pages • {result['chunks']} chunks")
        else:
            st.error(f"✗ {result.get('error', 'Unknown error')}")


def process_uploaded_document(uploaded_file):
    """Handle PDF upload and indexing."""
    # Close existing QueryEngine BEFORE processing to release ChromaDB locks
    if st.session_state.query_engine is not None:
        try:
            st.session_state.query_engine.close()
        except Exception:
            pass
    st.session_state.query_engine = None
    st.session_state.engine_status = "processing"

    processor = st.session_state.upload_processor

    # Create progress bar
    progress_bar = st.progress(0, text="Starting...")

    def update_progress(stage: str, progress: float):
        progress_bar.progress(progress, text=stage)

    # Process the file (creates fresh indexes with only this PDF)
    result = processor.process_uploaded_file(
        uploaded_file,
        progress_callback=update_progress
    )

    # Store result
    st.session_state.last_upload_result = result

    if result.get("success"):
        # Reload query engine with new indexes
        reload_query_engine()
        # Clear progress bar and rerun to refresh UI
        progress_bar.empty()
        st.rerun()
    else:
        st.error(f"Failed: {result.get('error', 'Unknown error')}")
        progress_bar.empty()


def reload_query_engine():
    """Reinitialize QueryEngine after index updates."""
    with st.spinner("Reloading indexes..."):
        try:
            # Clear old chat history since we have new content
            st.session_state.messages = []
            # Reinitialize QueryEngine with new indexes
            st.session_state.query_engine = QueryEngine()
            st.session_state.engine_status = "ready"
        except Exception as e:
            st.session_state.engine_status = "error"
            st.session_state.engine_error = str(e)


def display_chat_history():
    """Display chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])


def display_sources(sources):
    """Display source citations with expandable details."""
    if not sources:
        return

    st.divider()
    st.caption(f"📄 **Sources ({len(sources)} documents)**")

    for idx, source in enumerate(sources, start=1):
        with st.expander(
            f"Source {idx}: {source['source']} (Page {source['page']})"
        ):
            st.markdown(f"**File:** {source['source']}")
            st.markdown(f"**Page:** {source['page']}")
            st.markdown(f"**Chunk ID:** {source['chunk_id']}")

            if source['rerank_score'] is not None:
                st.markdown(f"**Relevance Score:** {source['rerank_score']:.4f}")

                # Relevance bar
                st.progress(
                    min(source['rerank_score'], 1.0),
                    text=f"Relevance: {source['rerank_score']:.1%}"
                )

            st.markdown("**Content Preview:**")
            st.text_area(
                label="preview",
                value=source['content_preview'],
                height=100,
                disabled=True,
                label_visibility="collapsed",
                key=f"source_{idx}_{hash(source['content_preview'])}"
            )


def process_query(query_text: str):
    """Process user query through RAG pipeline."""
    query_engine = st.session_state.query_engine

    # Display user message
    with st.chat_message("user"):
        st.markdown(query_text)

    # Add to history
    st.session_state.messages.append({
        "role": "user",
        "content": query_text
    })

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = query_engine.query(query_text)

        # Display answer
        st.markdown(result["answer"])

        # Display sources
        if result["sources"]:
            display_sources(result["sources"])

        # Display metadata (expandable)
        with st.expander("🔍 Query Metadata"):
            st.json(result["metadata"])

    # Add to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "metadata": result["metadata"]
    })


def main():
    """Main application entry point."""
    initialize_session_state()
    display_header()
    display_sidebar()

    # Check if engine is ready
    if st.session_state.engine_status == "waiting":
        # Show instructions to upload a PDF
        st.markdown("### 👋 Welcome!")
        st.markdown("Upload a PDF using the sidebar to get started. You can then ask questions about its contents.")
        return
    elif st.session_state.engine_status == "error":
        st.error(f"Error: {st.session_state.get('engine_error', 'Unknown error')}")
        return

    # Display chat history
    display_chat_history()

    # Chat input
    if query := st.chat_input("Ask a question about your PDF..."):
        process_query(query)


if __name__ == "__main__":
    main()
