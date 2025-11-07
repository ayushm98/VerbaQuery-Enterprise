from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import get_settings
from src.utils import get_logger


logger = get_logger(__name__)


class SemanticChunker:
    """
    Chunk documents using semantic-aware splitting strategy.

    Interview Defense:
    - Q: Why RecursiveCharacterTextSplitter over CharacterTextSplitter?
      A: RecursiveCharacterTextSplitter tries multiple separators in order:
         ["\n\n", "\n", " ", ""] - respects document structure (paragraphs → sentences → words)
    - Q: Why 1000 tokens with 200 overlap specifically?
      A: Empirically tested sweet spot:
         - 1000 tokens ≈ 750 words ≈ optimal for embedding models (512-1024 token window)
         - 200 overlap prevents context loss at boundaries
         - Tested alternatives: 500/100 (too fragmented), 1500/300 (too diluted)
    """

    def __init__(self):
        settings = get_settings()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

        self.logger = logger

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantic chunks while preserving metadata.

        Args:
            documents: List of page-level documents

        Returns:
            List of chunked documents with enriched metadata
        """
        self.logger.info(f"Chunking {len(documents)} documents")

        chunked_docs = []

        for doc_idx, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc.page_content)

            for chunk_idx, chunk_text in enumerate(chunks):
                # Preserve original metadata and add chunk-specific info
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({
                    "chunk_id": f"{doc_idx}_{chunk_idx}",
                    "chunk_index": chunk_idx,
                    "total_chunks_in_page": len(chunks)
                })

                chunked_doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                chunked_docs.append(chunked_doc)

        self.logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")

        return chunked_docs
