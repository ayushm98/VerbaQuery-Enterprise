"""
Fresh Indexer for Dynamic Document Uploads.

Creates fresh indexes for each uploaded PDF - replaces any existing indexes.
"""

from pathlib import Path
from typing import List, Dict
import pickle
import shutil
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from rank_bm25 import BM25Okapi
from config import get_settings
from src.utils import get_logger


logger = get_logger(__name__)


class IncrementalIndexer:
    """
    Creates fresh indexes for uploaded documents.
    Clears existing indexes and builds new ones with only the uploaded content.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = logger

    def _get_embeddings(self):
        """Create fresh embeddings instance to avoid stale connections."""
        return OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key
        )

    def add_documents(self, new_documents: List[Document], replace: bool = True) -> Dict:
        """
        Index documents - by default replaces all existing indexes.

        Args:
            new_documents: List of chunked Document objects to index
            replace: If True, clears existing indexes first (default: True)

        Returns:
            dict with stats: {
                "vector_count": int,
                "bm25_count": int,
                "chunks": int
            }
        """
        if not new_documents:
            self.logger.warning("No documents to index")
            return {"vector_count": 0, "bm25_count": 0, "chunks": 0}

        self.logger.info(f"Indexing {len(new_documents)} chunks (replace={replace})")

        if replace:
            # Clear existing indexes first
            self._clear_indexes()

        try:
            # Step 1: Create fresh ChromaDB index
            vector_count = self._create_vector_index(new_documents)

            # Step 2: Create fresh BM25 index
            bm25_count = self._create_bm25_index(new_documents)

            stats = {
                "vector_count": vector_count,
                "bm25_count": bm25_count,
                "chunks": len(new_documents)
            }

            self.logger.info(f"Indexing complete: {stats}")
            return stats

        except Exception as e:
            self.logger.error(f"Indexing failed: {e}")
            raise

    def _clear_indexes(self) -> None:
        """Clear all existing indexes."""
        self.logger.info("Clearing existing indexes...")

        # Clear ChromaDB
        chroma_dir = self.settings.chroma_persist_directory
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
            self.logger.info(f"Removed ChromaDB at {chroma_dir}")

        # Clear BM25
        bm25_path = self.settings.bm25_index_path
        if bm25_path.exists():
            bm25_path.unlink()
            self.logger.info(f"Removed BM25 index at {bm25_path}")

    def _create_vector_index(self, documents: List[Document]) -> int:
        """
        Create fresh ChromaDB index with documents.

        Returns:
            Number of documents indexed
        """
        persist_directory = self.settings.chroma_persist_directory

        self.logger.info(f"Creating vector index for {len(documents)} documents")

        # Create ChromaDB client with settings that work in cloud environments
        client_settings = chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )

        # Create persistent client
        client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=client_settings
        )

        # Create fresh embeddings and ChromaDB for each indexing operation
        embeddings = self._get_embeddings()
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            client=client,
            collection_name="verbaquery_docs"
        )

        self.logger.info(f"Created ChromaDB with {len(documents)} documents")

        return len(documents)

    def _create_bm25_index(self, documents: List[Document]) -> int:
        """
        Create fresh BM25 index with documents.

        Returns:
            Number of documents indexed
        """
        index_path = self.settings.bm25_index_path

        self.logger.info(f"Creating BM25 index for {len(documents)} documents")

        # Tokenize documents
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]

        # Create BM25 index
        bm25_index = BM25Okapi(tokenized_corpus)

        # Save index with documents
        index_data = {
            "bm25": bm25_index,
            "documents": documents,
            "tokenized_corpus": tokenized_corpus
        }

        # Ensure directory exists
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)

        self.logger.info(f"Created BM25 index with {len(documents)} documents")

        return len(documents)

    def indexes_exist(self) -> bool:
        """Check if both indexes exist."""
        chroma_exists = self.settings.chroma_persist_directory.exists()
        bm25_exists = self.settings.bm25_index_path.exists()
        return chroma_exists and bm25_exists
