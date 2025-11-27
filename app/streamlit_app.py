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
    if "query_engine" not in st.session_state:
        with st.spinner("Initializing query engine..."):
            try:
                st.session_state.query_engine = QueryEngine()
                st.session_state.engine_status = "ready"
            except Exception as e:
                st.session_state.engine_status = "error"
                st.session_state.engine_error = str(e)


def display_header():
    """Display application header."""
    st.title("📚 VerbaQuery-Enterprise")
    st.caption("Industrial-Grade RAG System for Enterprise Document Querying")

    # Status indicator
    if st.session_state.engine_status == "ready":
        st.success("✓ Query engine ready")
    else:
        st.error(f"✗ Engine initialization failed: {st.session_state.get('engine_error', 'Unknown error')}")


def display_sidebar():
    """Display sidebar with configuration and info."""
    with st.sidebar:
        st.header("⚙️ System Info")

        settings = get_settings()

        st.markdown("**Configuration**")
        st.code(f"""
Model: {settings.openai_chat_model}
Embeddings: {settings.openai_embedding_model}
Chunk Size: {settings.chunk_size}
Initial Retrieval: {settings.initial_retrieval_count}
Final (Re-ranked): {settings.final_retrieval_count}
        """)

        st.divider()

        st.markdown("**Pipeline Stages**")
        st.markdown("""
        1. **Hybrid Retrieval**
           - Vector search (semantic)
           - Keyword search (BM25)
           - Ensemble ranking

        2. **Re-ranking**
           - Flashrank cross-encoder
           - Top-5 selection

        3. **Generation**
           - GPT-4 with grounding
           - Source citation
        """)

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


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
                key=f"source_{idx}_{source['chunk_id']}"
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
    if st.session_state.engine_status != "ready":
        st.error("Query engine not initialized. Please check your configuration and indexes.")
        st.info("""
        **Troubleshooting Steps:**
        1. Ensure indexes exist: Run `python scripts/ingest_documents.py --input data/`
        2. Check .env file: Verify OPENAI_API_KEY is set
        3. Check logs for detailed error messages
        """)
        return

    # Display chat history
    display_chat_history()

    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        process_query(query)


if __name__ == "__main__":
    main()
