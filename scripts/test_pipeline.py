#!/usr/bin/env python3
"""
End-to-End Pipeline Testing Script

Tests all components of VerbaQuery-Enterprise:
1. PDF loading
2. Chunking
3. Indexing
4. Retrieval (vector, keyword, hybrid)
5. Re-ranking
6. Query generation

Usage:
    python scripts/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import PDFLoader, SemanticChunker, DualIndexer
from src.retrieval import VectorRetriever, KeywordRetriever, HybridRetriever, FlashrankReranker
from src.generation import QueryEngine
from src.utils import setup_logger


logger = setup_logger(__name__)


def test_ingestion():
    """Test ingestion pipeline."""
    logger.info("=" * 80)
    logger.info("Testing Ingestion Pipeline")
    logger.info("=" * 80)

    # Load PDFs
    loader = PDFLoader()
    data_dir = Path("data")

    if not data_dir.exists():
        logger.error("Data directory not found")
        return False

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found")
        return False

    # Test single PDF
    test_pdf = pdf_files[0]
    logger.info(f"Testing with: {test_pdf.name}")

    try:
        documents = loader.load_single_pdf(test_pdf)
        logger.info(f"✓ Loaded {len(documents)} pages")

        # Test chunking
        chunker = SemanticChunker()
        chunks = chunker.chunk_documents(documents[:5])  # Test first 5 pages
        logger.info(f"✓ Created {len(chunks)} chunks")

        return True
    except Exception as e:
        logger.error(f"✗ Ingestion failed: {str(e)}")
        return False


def test_retrieval():
    """Test retrieval components."""
    logger.info("=" * 80)
    logger.info("Testing Retrieval Pipeline")
    logger.info("=" * 80)

    test_query = "What is the refund policy?"

    try:
        # Test vector retrieval
        logger.info("Testing vector retrieval...")
        vector_retriever = VectorRetriever()
        vector_results = vector_retriever.retrieve(test_query, k=5)
        logger.info(f"✓ Vector retrieval returned {len(vector_results)} docs")

        # Test keyword retrieval
        logger.info("Testing keyword retrieval...")
        keyword_retriever = KeywordRetriever()
        keyword_results = keyword_retriever.retrieve(test_query, k=5)
        logger.info(f"✓ Keyword retrieval returned {len(keyword_results)} docs")

        # Test hybrid retrieval
        logger.info("Testing hybrid retrieval...")
        hybrid_retriever = HybridRetriever()
        hybrid_results = hybrid_retriever.retrieve(test_query, k=10)
        logger.info(f"✓ Hybrid retrieval returned {len(hybrid_results)} docs")

        # Test re-ranking
        logger.info("Testing re-ranking...")
        reranker = FlashrankReranker()
        reranked = reranker.rerank(test_query, hybrid_results, top_k=5)
        logger.info(f"✓ Re-ranking returned {len(reranked)} docs")

        return True
    except Exception as e:
        logger.error(f"✗ Retrieval failed: {str(e)}")
        return False


def test_query_engine():
    """Test end-to-end query engine."""
    logger.info("=" * 80)
    logger.info("Testing Query Engine")
    logger.info("=" * 80)

    test_queries = [
        "What is the refund policy?",
        "How many vacation days do employees get?",
        "What are the working hours?"
    ]

    try:
        engine = QueryEngine()

        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            result = engine.query(query)

            logger.info(f"Answer: {result['answer'][:200]}...")
            logger.info(f"Sources: {len(result['sources'])} documents")
            logger.info(f"Metadata: {result['metadata']}")

        logger.info("\n✓ Query engine test complete")
        return True

    except Exception as e:
        logger.error(f"✗ Query engine failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting VerbaQuery-Enterprise Pipeline Tests\n")

    results = {
        "Ingestion": test_ingestion(),
        "Retrieval": test_retrieval(),
        "Query Engine": test_query_engine()
    }

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)

    for component, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{component}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.info("\n⚠️  Some tests failed. Check logs above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
