from typing import List
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from config import get_settings
from src.utils import get_logger


logger = get_logger(__name__)


class FlashrankReranker:
    """
    Cross-encoder re-ranking using Flashrank.
    Applies a trained cross-encoder model to re-score and rank retrieved candidates.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """
        Initialize Flashrank re-ranker.

        Args:
            model_name: Flashrank model identifier
        """
        self.logger = logger
        self.settings = get_settings()

        try:
            self.ranker = Ranker(model_name=model_name, cache_dir="./data/models")
            self.logger.info(f"Loaded Flashrank model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load Flashrank model: {str(e)}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Document]:
        """
        Re-rank documents using cross-encoder scoring.

        Args:
            query: User query string
            documents: List of candidate documents from retrieval
            top_k: Number of top documents to return (default: final_retrieval_count from config)

        Returns:
            Re-ranked list of top-k documents

        Algorithm:
        1. Convert LangChain Documents to Flashrank format
        2. Score each query-document pair with cross-encoder
        3. Sort by relevance score (descending)
        4. Return top-k documents
        """
        if top_k is None:
            top_k = self.settings.final_retrieval_count

        if not documents:
            self.logger.warning("No documents to rerank")
            return []

        self.logger.info(
            f"Re-ranking {len(documents)} documents, returning top {top_k}"
        )

        try:
            # Convert to Flashrank format
            passages = [
                {
                    "id": idx,
                    "text": doc.page_content,
                    "meta": doc.metadata
                }
                for idx, doc in enumerate(documents)
            ]

            # Create rerank request
            rerank_request = RerankRequest(query=query, passages=passages)

            # Get re-ranked results
            results = self.ranker.rerank(rerank_request)

            # Convert back to LangChain Documents and sort by score
            reranked_docs = []
            for result in results[:top_k]:
                original_doc = documents[result["id"]]
                # Optionally add re-rank score to metadata
                original_doc.metadata["rerank_score"] = result["score"]
                reranked_docs.append(original_doc)

            self.logger.info(
                f"Re-ranking complete. Top doc score: {reranked_docs[0].metadata.get('rerank_score', 'N/A'):.4f}"
                if reranked_docs else "No results after re-ranking"
            )

            return reranked_docs

        except Exception as e:
            self.logger.error(f"Re-ranking failed: {str(e)}")
            # Fallback: return original documents (degraded but functional)
            self.logger.warning("Falling back to original retrieval ranking")
            return documents[:top_k]
