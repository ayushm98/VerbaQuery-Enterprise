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

    Interview Defense:
    - Q: Why vector search over keyword search alone?
      A: Vector search captures semantic similarity:
         Query: "automobile accident" → Matches: "car crash", "vehicle collision"
         This handles synonyms, paraphrasing, and conceptual similarity
    - Q: Why OpenAI embeddings vs open-source (SentenceTransformers)?
      A: Trade-off decision:
         - OpenAI: Better quality (~5% higher accuracy), pay-per-use ($0.02/1M tokens)
         - Open-source: Free, offline, but requires GPU for fast inference
         - For enterprise MVP, quality > cost (can optimize later)
    - Q: What similarity metric does ChromaDB use?
      A: Cosine similarity by default (measures angle between vectors)
         Range: -1 to 1, where 1 = identical, 0 = orthogonal
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

        Interview Defense:
        - Q: Why k=10 as default?
          A: Balancing recall vs noise:
             - Too low (k=3): May miss relevant docs (low recall)
             - Too high (k=50): Dilutes context, slows LLM (high noise)
             - k=10 is sweet spot for first-stage retrieval (then re-rank to 5)
        - Q: What's the cost of each query?
          A: Query embedding: ~10 tokens × $0.02/1M = $0.0000002 (negligible)
             Vector search: Local computation in ChromaDB (no API cost)
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
