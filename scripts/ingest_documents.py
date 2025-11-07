#!/usr/bin/env python3
"""
Document Ingestion CLI Script

Processes PDFs from raw directory through the complete ETL pipeline:
1. Load PDFs with page-level extraction
2. Chunk into semantic units (1000 tokens, 200 overlap)
3. Build dual indexes (ChromaDB vector + BM25 keyword)

Usage:
    python scripts/ingest_documents.py --input data/raw/
    python scripts/ingest_documents.py --input data/raw/ --rebuild

Interview Defense:
- Q: Why separate ingestion from query serving?
  A: Separation of concerns - ingestion is expensive (embeddings cost money),
     should run once. Query serving is frequent, should be fast.
- Q: Why --rebuild flag?
  A: Allows cache invalidation - if docs change, rebuild indexes.
     Without flag, skips if indexes exist (faster iteration).
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import PDFLoader, SemanticChunker, DualIndexer
from src.utils import setup_logger
from config import get_settings


logger = setup_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into VerbaQuery-Enterprise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all PDFs from data directory
  python scripts/ingest_documents.py --input data/

  # Force rebuild existing indexes
  python scripts/ingest_documents.py --input data/ --rebuild

  # Ingest specific file
  python scripts/ingest_documents.py --input data/policy.pdf
        """
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data"),
        help="Path to PDF file or directory containing PDFs (default: data/)"
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild indexes even if they exist"
    )

    return parser.parse_args()


def check_existing_indexes(settings) -> bool:
    """
    Check if indexes already exist.

    Returns:
        True if both indexes exist, False otherwise
    """
    chroma_exists = settings.chroma_persist_directory.exists()
    bm25_exists = settings.bm25_index_path.exists()

    return chroma_exists and bm25_exists


def main():
    """Execute ingestion pipeline."""
    args = parse_args()
    settings = get_settings()

    logger.info("=" * 80)
    logger.info("VerbaQuery-Enterprise: Document Ingestion Pipeline")
    logger.info("=" * 80)

    # Check if indexes exist
    if not args.rebuild and check_existing_indexes(settings):
        logger.warning(
            "Indexes already exist. Use --rebuild to force recreation. Exiting."
        )
        return

    # Step 1: Load PDFs
    logger.info(f"Step 1/3: Loading PDFs from {args.input}")
    loader = PDFLoader()

    if args.input.is_file():
        documents = loader.load_single_pdf(args.input)
    elif args.input.is_dir():
        documents = loader.load_directory(args.input)
    else:
        logger.error(f"Invalid input path: {args.input}")
        sys.exit(1)

    if not documents:
        logger.error("No documents loaded. Exiting.")
        sys.exit(1)

    logger.info(f"Loaded {len(documents)} page-level documents")

    # Step 2: Chunk documents
    logger.info(f"Step 2/3: Chunking documents (size={settings.chunk_size}, overlap={settings.chunk_overlap})")
    chunker = SemanticChunker()
    chunked_docs = chunker.chunk_documents(documents)

    logger.info(f"Created {len(chunked_docs)} semantic chunks")

    # Step 3: Build indexes
    logger.info("Step 3/3: Building dual indexes (Vector + Keyword)")
    indexer = DualIndexer()

    try:
        vector_store, bm25_index = indexer.build_indexes(chunked_docs)

        logger.info("=" * 80)
        logger.info("✓ Ingestion Complete!")
        logger.info(f"  - Vector Index: {settings.chroma_persist_directory}")
        logger.info(f"  - Keyword Index: {settings.bm25_index_path}")
        logger.info(f"  - Total Chunks Indexed: {len(chunked_docs)}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to build indexes: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
