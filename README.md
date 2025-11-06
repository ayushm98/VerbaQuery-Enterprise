# VerbaQuery-Enterprise

Industrial-grade Retrieval-Augmented Generation system for enterprise document querying.

## Features

- Hybrid search combining vector and keyword retrieval
- Advanced re-ranking for precision
- Source-grounded responses with page citations
- Streamlit web interface

## Architecture

- **Ingestion**: PDF processing → Semantic chunking → Dual indexing
- **Retrieval**: ChromaDB (vectors) + BM25 (keywords) → Ensemble ranking
- **Refinement**: Flashrank cross-encoder re-ranking
- **Generation**: OpenAI GPT with strict grounding prompts

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. Ingest documents:
   ```bash
   python scripts/ingest_documents.py --input data/raw/
   ```

4. Launch application:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## System Requirements

- Python 3.10+
- 8GB RAM minimum
- OpenAI API key
