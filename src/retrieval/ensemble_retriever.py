from typing import List, Dict
from langchain_core.documents import Document
from config import get_settings
from src.utils import get_logger
from .vector_retriever import VectorRetriever
from .keyword_retriever import KeywordRetriever


logger = get_logger(__name__)


class HybridRetriever:
    """
    Ensemble retriever combining vector (dense) and keyword (sparse) search.
    Uses Reciprocal Rank Fusion to merge results from multiple retrieval methods.
    """

    def __init__(self):
        """Initialize hybrid retriever with vector and keyword components."""
        self.settings = get_settings()
        self.logger = logger

        # Initialize retrievers
        self.vector_retriever = VectorRetriever()
        self.keyword_retriever = KeywordRetriever()

        # Ensemble weights
        self.weight_vector = self.settings.ensemble_weight_vector
        self.weight_keyword = self.settings.ensemble_weight_keyword

        self.logger.info(
            f"Initialized hybrid retriever (vector weight={self.weight_vector}, "
            f"keyword weight={self.weight_keyword})"
        )

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """
        Retrieve documents using hybrid search with reciprocal rank fusion.

        Args:
            query: User query string
            k: Number of documents to retrieve

        Returns:
            List of Document objects, ranked by ensemble score

        Algorithm:
        1. Retrieve k documents from each retriever
        2. Compute reciprocal rank scores for each document
        3. Combine scores using weighted sum
        4. Return top-k by combined score
        """
        self.logger.info(f"Hybrid retrieval: query='{query[:50]}...', k={k}")

        # Step 1: Retrieve from both indexes
        vector_results = self.vector_retriever.retrieve(query, k=k)
        keyword_results = self.keyword_retriever.retrieve(query, k=k)

        self.logger.info(
            f"Retrieved {len(vector_results)} vector docs, "
            f"{len(keyword_results)} keyword docs"
        )

        # Step 2: Compute reciprocal rank fusion scores
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Process vector results
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = self._get_doc_id(doc)
            rrf_score = 1.0 / (60 + rank)  # k=60 is standard RRF constant
            doc_scores[doc_id] = self.weight_vector * rrf_score
            doc_map[doc_id] = doc

        # Process keyword results
        for rank, doc in enumerate(keyword_results, start=1):
            doc_id = self._get_doc_id(doc)
            rrf_score = 1.0 / (60 + rank)

            if doc_id in doc_scores:
                # Document appears in both results - add scores
                doc_scores[doc_id] += self.weight_keyword * rrf_score
            else:
                doc_scores[doc_id] = self.weight_keyword * rrf_score
                doc_map[doc_id] = doc

        # Step 3: Sort by combined score and return top-k
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        results = [doc_map[doc_id] for doc_id, score in sorted_docs]

        self.logger.info(f"Ensemble retrieval returned {len(results)} documents")

        return results

    def _get_doc_id(self, doc: Document) -> str:
        """
        Generate unique document identifier from metadata.

        Uses chunk_id if available, otherwise constructs from source + page + chunk_index.
        """
        metadata = doc.metadata

        if "chunk_id" in metadata:
            return metadata["chunk_id"]

        # Fallback: construct ID from available metadata
        source = metadata.get("source", "unknown")
        page = metadata.get("page", 0)
        chunk_idx = metadata.get("chunk_index", 0)

        return f"{source}_page{page}_chunk{chunk_idx}"
