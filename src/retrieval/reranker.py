from typing import List
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from config import get_settings
from src.utils import get_logger


logger = get_logger(__name__)


class FlashrankReranker:
    """
    Cross-encoder re-ranking using Flashrank.

    Interview Defense:
    - Q: What's the difference between retrieval and re-ranking?
      A: Two-stage process for efficiency:
         Stage 1 (Retrieval): Fast, cheap, broad recall
           - Embedding lookup in ChromaDB: ~50ms for 10K docs
           - Retrieve top 10-50 candidates (cast wide net)
         Stage 2 (Re-ranking): Slow, expensive, high precision
           - Cross-encoder model: ~200ms for 10 docs
           - Re-score candidates, keep top 5 (filter noise)
         This is much faster than cross-encoding all 10K docs upfront
    - Q: Why use a cross-encoder for re-ranking?
      A: Architecture advantage:
         Bi-encoder (used in retrieval): Encode query and doc separately, then compare
           - Fast: Pre-compute doc embeddings, query embedding at runtime
           - Less accurate: Can't see query-doc interactions
         Cross-encoder (used in re-ranking): Encode query+doc together
           - Slow: Must process each query-doc pair
           - More accurate: Sees full interaction, better relevance
    - Q: What is Flashrank specifically?
      A: Lightweight cross-encoder library:
         - Models: ms-marco-MiniLM-L-12-v2 (default), ms-marco-MultiBERT-L-12
         - Size: ~120MB (vs BERT ~400MB)
         - Speed: Optimized for CPU inference (no GPU needed)
         - Quality: Trained on MS MARCO passage ranking dataset
    - Q: Why retrieve 10 docs but only pass 5 to LLM?
      A: Context window optimization:
         - GPT-4 context: 128K tokens, but accuracy drops with "noise"
         - Research ("Lost in the Middle"): LLMs focus on start/end of context
         - Solution: Send only highest-quality docs (top 5 after re-ranking)
         - Cost savings: Fewer tokens = lower API cost
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

        Interview Defense:
        - Q: What's the computational cost?
          A: Linear in number of documents:
             - 10 docs × 200ms/doc = 2 seconds (acceptable for interactive query)
             - 100 docs × 200ms/doc = 20 seconds (too slow, hence two-stage approach)
        - Q: How accurate is re-ranking?
          A: MS MARCO benchmark results:
             - Retrieval-only (BM25): MRR@10 = 0.18
             - Retrieval + Re-ranking: MRR@10 = 0.36 (2x improvement)
             - Real-world: ~20-30% better precision in top-5 results
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
