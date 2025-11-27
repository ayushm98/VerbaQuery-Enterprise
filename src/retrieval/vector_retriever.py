from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import get_settings
from src.utils import get_logger


logger = get_logger(__name__)


class VectorRetriever:
    """
    Dense retrieval using ChromaDB vector similarity search.
    Uses cosine similarity to find semantically similar documents.
    """

    def __init__(self, persist_directory: Path = None):
        """
        Initialize vector retriever with persisted ChromaDB.

        Args:
            persist_directory: Path to ChromaDB persistence directory
        """
        self.settings = get_settings()
        self.logger = logger

        if persist_directory is None:
            persist_directory = self.settings.chroma_persist_directory

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key
        )

        # Load persisted ChromaDB
        try:
            self.vectorstore = Chroma(
                persist_directory=str(persist_directory),
                embedding_function=self.embeddings,
                collection_name="verbaquery_docs"
            )
            self.logger.info(f"Loaded ChromaDB from {persist_directory}")
        except Exception as e:
            self.logger.error(f"Failed to load ChromaDB: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """
        Retrieve top-k most similar documents using vector similarity.

        Args:
            query: User query string
            k: Number of documents to retrieve

        Returns:
            List of Document objects, ranked by similarity
        """
        self.logger.info(f"Vector retrieval: query='{query[:50]}...', k={k}")

        try:
            results = self.vectorstore.similarity_search(query, k=k)
            self.logger.info(f"Retrieved {len(results)} documents from vector index")
            return results
        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {str(e)}")
            return []

    def retrieve_with_scores(self, query: str, k: int = 10) -> List[tuple[Document, float]]:
        """
        Retrieve documents with similarity scores.

        Returns:
            List of (Document, score) tuples, where score is cosine similarity
        """
        self.logger.info(f"Vector retrieval with scores: query='{query[:50]}...', k={k}")

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            self.logger.info(f"Retrieved {len(results)} documents with scores")
            return results
        except Exception as e:
            self.logger.error(f"Vector retrieval with scores failed: {str(e)}")
            return []

    def reload_index(self, persist_directory: Path = None) -> None:
        """
        Reload ChromaDB collection after updates.

        Args:
            persist_directory: Path to ChromaDB persistence directory
        """
        if persist_directory is None:
            persist_directory = self.settings.chroma_persist_directory

        self.logger.info(f"Reloading ChromaDB from {persist_directory}")

        # Close existing connection first
        self.close()

        try:
            self.vectorstore = Chroma(
                persist_directory=str(persist_directory),
                embedding_function=self.embeddings,
                collection_name="verbaquery_docs"
            )
            self.logger.info("Reloaded ChromaDB collection")
        except Exception as e:
            self.logger.error(f"Failed to reload ChromaDB: {str(e)}")
            raise

    def close(self) -> None:
        """Close ChromaDB connection to release database lock."""
        if hasattr(self, 'vectorstore') and self.vectorstore is not None:
            try:
                # Clear the system cache to release SQLite connections
                if hasattr(self.vectorstore, '_client'):
                    client = self.vectorstore._client
                    if hasattr(client, 'clear_system_cache'):
                        client.clear_system_cache()
                        self.logger.info("Cleared ChromaDB system cache")
                # Delete the vectorstore reference
                del self.vectorstore
                self.vectorstore = None
                self.logger.info("Closed ChromaDB connection")
            except Exception as e:
                self.logger.warning(f"Error closing ChromaDB: {e}")
