from typing import List
from pathlib import Path
import pickle
from langchain_core.documents import Document
from config import get_settings
from src.utils import get_logger


logger = get_logger(__name__)


class KeywordRetriever:
    """
    Sparse retrieval using BM25 keyword matching.
    Provides exact and lexical term matching capabilities.
    """

    def __init__(self, index_path: Path = None):
        """
        Initialize keyword retriever from persisted BM25 index.

        Args:
            index_path: Path to pickled BM25 index
        """
        self.settings = get_settings()
        self.logger = logger

        if index_path is None:
            index_path = self.settings.bm25_index_path

        # Load BM25 index and documents
        try:
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)

            self.bm25_index = index_data["bm25"]
            self.documents = index_data["documents"]

            self.logger.info(f"Loaded BM25 index from {index_path} ({len(self.documents)} docs)")
        except Exception as e:
            self.logger.error(f"Failed to load BM25 index: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """
        Retrieve top-k documents using BM25 scoring.

        Args:
            query: User query string
            k: Number of documents to retrieve

        Returns:
            List of Document objects, ranked by BM25 score
        """
        self.logger.info(f"BM25 retrieval: query='{query[:50]}...', k={k}")

        try:
            # Tokenize query (must match corpus tokenization)
            tokenized_query = query.lower().split()

            # Get BM25 scores for all documents
            scores = self.bm25_index.get_scores(tokenized_query)

            # Get top-k document indices
            top_k_indices = scores.argsort()[::-1][:k]

            # Retrieve corresponding documents
            results = [self.documents[i] for i in top_k_indices]

            self.logger.info(f"Retrieved {len(results)} documents from BM25 index")
            return results

        except Exception as e:
            self.logger.error(f"BM25 retrieval failed: {str(e)}")
            return []

    def retrieve_with_scores(self, query: str, k: int = 10) -> List[tuple[Document, float]]:
        """
        Retrieve documents with BM25 scores.

        Returns:
            List of (Document, score) tuples
        """
        self.logger.info(f"BM25 retrieval with scores: query='{query[:50]}...', k={k}")

        try:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            top_k_indices = scores.argsort()[::-1][:k]

            results = [(self.documents[i], scores[i]) for i in top_k_indices]

            self.logger.info(f"Retrieved {len(results)} documents with BM25 scores")
            return results

        except Exception as e:
            self.logger.error(f"BM25 retrieval with scores failed: {str(e)}")
            return []

    def reload_index(self, index_path: Path = None) -> None:
        """
        Reload BM25 index from disk after updates.

        Args:
            index_path: Path to pickled BM25 index (uses default if not specified)
        """
        if index_path is None:
            index_path = self.settings.bm25_index_path

        self.logger.info(f"Reloading BM25 index from {index_path}")

        try:
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)

            self.bm25_index = index_data["bm25"]
            self.documents = index_data["documents"]

            self.logger.info(f"Reloaded BM25 index ({len(self.documents)} docs)")
        except Exception as e:
            self.logger.error(f"Failed to reload BM25 index: {str(e)}")
            raise
