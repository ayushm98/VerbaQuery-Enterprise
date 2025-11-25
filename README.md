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

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Ingest documents:**
   ```bash
   # Ingest all PDFs from data directory
   python scripts/ingest_documents.py --input data/

   # Force rebuild if indexes already exist
   python scripts/ingest_documents.py --input data/ --rebuild
   ```

4. **Test the pipeline:**
   ```bash
   python scripts/test_pipeline.py
   ```

5. **Launch application:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Project Structure

```
RAG/
├── app/
│   └── streamlit_app.py          # Web interface
├── config/
│   ├── settings.py                # Configuration management
│   └── prompts.py                 # LLM prompt templates
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py          # PDF extraction
│   │   ├── chunker.py             # Semantic chunking
│   │   └── indexer.py             # Dual indexing (ChromaDB + BM25)
│   ├── retrieval/
│   │   ├── vector_retriever.py    # Dense retrieval
│   │   ├── keyword_retriever.py   # Sparse retrieval (BM25)
│   │   ├── ensemble_retriever.py  # Hybrid search
│   │   └── reranker.py            # Flashrank re-ranking
│   ├── generation/
│   │   └── query_engine.py        # End-to-end RAG pipeline
│   └── utils/
│       ├── logger.py              # Structured logging
│       └── validators.py          # Input validation
├── scripts/
│   ├── ingest_documents.py        # Document ingestion CLI
│   └── test_pipeline.py           # Pipeline testing
└── data/
    ├── indexes/                   # Persisted indexes
    └── models/                    # Downloaded models
```

## System Requirements

- Python 3.10+
- 8GB RAM minimum
- OpenAI API key

## Technical Details

### RAG Pipeline

1. **Ingestion (ETL)**
   - Page-level PDF extraction with metadata
   - Semantic chunking (1000 tokens, 200 overlap)
   - Dual indexing: ChromaDB (vectors) + BM25 (keywords)

2. **Retrieval (Hybrid Search)**
   - Vector search: Semantic similarity matching
   - Keyword search: Exact term matching (BM25)
   - Ensemble: Reciprocal Rank Fusion (50/50 weighting)
   - Retrieve top 10 candidates

3. **Re-ranking (Precision Filter)**
   - Flashrank cross-encoder
   - Re-score 10 candidates → Keep top 5
   - Solves "Lost in the Middle" problem

4. **Generation (Grounded LLM)**
   - GPT-4 with strict grounding prompts
   - Source citations with page numbers
   - Hallucination prevention

### Configuration

Edit `.env` to customize:
- `CHUNK_SIZE`: Chunk size in tokens (default: 1000)
- `CHUNK_OVERLAP`: Overlap in tokens (default: 200)
- `INITIAL_RETRIEVAL_COUNT`: First-stage retrieval (default: 10)
- `FINAL_RETRIEVAL_COUNT`: After re-ranking (default: 5)
- `ENSEMBLE_WEIGHT_VECTOR`: Vector search weight (default: 0.5)
- `ENSEMBLE_WEIGHT_KEYWORD`: Keyword search weight (default: 0.5)
