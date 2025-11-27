from pathlib import Path
from typing import List
import pickle
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from config import get_settings
from src.utils import get_logger


logger = get_logger(__name__)


class DualIndexer:
    """
    Create and manage dual indexes: vector (ChromaDB) + keyword (BM25).

    Interview Defense:
    - Q: Why maintain two separate indexes?
      A: Complementary strengths:
         - Vector index: Semantic similarity (e.g., "automobile" matches "car")
         - BM25: Exact term matching (e.g., policy numbers, product codes)
         - Hybrid approach proven to outperform single method by 15-30%
    - Q: Why persist both indexes to disk?
      A: Avoid re-embedding on every application restart (expensive and slow)
         - ChromaDB: Native persistence support
         - BM25: Pickle serialization (lightweight, <10MB for 10K docs)
    """

    def __init__(self):
        self.settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key
        )
        self.logger = logger

    def create_vector_index(
        self,
        documents: List[Document],
        persist_directory: Path = None
    ) -> Chroma:
        """
        Create ChromaDB vector index from documents.

        Args:
            documents: Chunked documents to index
            persist_directory: Directory to persist ChromaDB

        Returns:
            Initialized ChromaDB vector store
        """
        if persist_directory is None:
            persist_directory = self.settings.chroma_persist_directory

        self.logger.info(f"Creating vector index for {len(documents)} documents")

        # ChromaDB with persistence
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(persist_directory),
            collection_name="verbaquery_docs"
        )

        self.logger.info(f"Vector index created and persisted to {persist_directory}")

        return vectorstore

    def create_keyword_index(
        self,
        documents: List[Document],
        index_path: Path = None
    ) -> BM25Okapi:
        """
        Create BM25 keyword index from documents.

        Args:
            documents: Chunked documents to index
            index_path: Path to persist BM25 index

        Returns:
            Initialized BM25 index
        """
        if index_path is None:
            index_path = self.settings.bm25_index_path

        self.logger.info(f"Creating BM25 index for {len(documents)} documents")

        # Tokenize documents (simple whitespace tokenization)
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]

        # Create BM25 index
        bm25_index = BM25Okapi(tokenized_corpus)

        # Persist to disk
        index_data = {
            "bm25": bm25_index,
            "documents": documents  # Store original docs for retrieval
        }

        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)

        self.logger.info(f"BM25 index created and persisted to {index_path}")

        return bm25_index

    def build_indexes(self, documents: List[Document]) -> tuple[Chroma, BM25Okapi]:
        """
        Build both vector and keyword indexes.

        Args:
            documents: Chunked documents to index

        Returns:
            Tuple of (vector_store, bm25_index)
        """
        self.logger.info("Building dual indexes (vector + keyword)")

        vector_store = self.create_vector_index(documents)
        bm25_index = self.create_keyword_index(documents)

        self.logger.info("Dual indexing complete")

        return vector_store, bm25_index
