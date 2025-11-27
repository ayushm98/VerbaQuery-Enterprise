from typing import List, Dict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import get_settings
from config.prompts import SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE, NO_CONTEXT_RESPONSE
from src.retrieval import HybridRetriever, FlashrankReranker
from src.utils import get_logger, validate_query


logger = get_logger(__name__)


class QueryEngine:
    """
    End-to-end query pipeline with retrieval, re-ranking, and grounded generation.

    Interview Defense:
    - Q: Walk me through the complete RAG pipeline.
      A: Three-stage architecture:
         1. Retrieval (Hybrid Search):
            - Query: "What is the refund policy?"
            - Retrieve top 10 docs from ChromaDB + BM25 ensemble
         2. Re-ranking (Flashrank):
            - Cross-encoder re-scores 10 candidates
            - Keep top 5 most relevant
         3. Generation (GPT-4):
            - Inject top 5 docs into prompt context
            - Generate answer with strict grounding constraints
            - Extract and format source citations
    - Q: How do you prevent hallucination?
      A: Multi-layer approach:
         1. Prompt engineering: Explicit "ONLY use provided context" instruction
         2. Few-shot examples: Show correct citation format in system prompt
         3. Response validation: Check if answer references sources
         4. Fallback: Return NO_CONTEXT_RESPONSE if retrieval fails
    - Q: What's the cost per query?
      A: Breakdown:
         - Embedding (query): ~10 tokens × $0.02/1M = $0.0000002
         - Generation (GPT-4): ~2000 tokens × $30/1M = $0.06
         - Total: ~$0.06 per query (context dominates cost)
         - Optimization: Use GPT-3.5-turbo ($1.50/1M) for 20x cost reduction
    """

    def __init__(self):
        """Initialize query engine with retrieval and generation components."""
        self.settings = get_settings()
        self.logger = logger

        # Retrieval pipeline
        self.retriever = HybridRetriever()
        self.reranker = FlashrankReranker()

        # LLM for generation
        self.llm = ChatOpenAI(
            model=self.settings.openai_chat_model,
            openai_api_key=self.settings.openai_api_key,
            temperature=0.0  # Deterministic for consistency
        )

        # Prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", QUERY_PROMPT_TEMPLATE)
        ])

        self.logger.info(
            f"Query engine initialized with model: {self.settings.openai_chat_model}"
        )

    def query(self, query_text: str) -> Dict:
        """
        Execute full RAG pipeline for a user query.

        Args:
            query_text: User's question

        Returns:
            Dictionary containing:
                - answer: Generated response
                - sources: List of source documents with metadata
                - metadata: Pipeline statistics (retrieval count, etc.)

        Interview Defense:
        - Q: Why return structured dict instead of just string answer?
          A: Enables rich UI features:
             - answer: Display to user
             - sources: Show citations, enable "View Source" buttons
             - metadata: Debug info (how many docs retrieved, re-rank scores)
        - Q: What happens if no relevant documents found?
          A: Graceful failure:
             - Return NO_CONTEXT_RESPONSE
             - Don't hallucinate or make up information
             - Better to admit "I don't know" than provide wrong answer
        """
        self.logger.info(f"Processing query: '{query_text}'")

        try:
            # Step 0: Validate query
            validated_query = validate_query(query_text)

            # Step 1: Retrieve candidates (top 10)
            self.logger.info("Step 1/3: Hybrid retrieval")
            candidates = self.retriever.retrieve(
                validated_query,
                k=self.settings.initial_retrieval_count
            )

            if not candidates:
                self.logger.warning("No documents retrieved")
                return {
                    "answer": NO_CONTEXT_RESPONSE,
                    "sources": [],
                    "metadata": {"retrieved_count": 0, "reranked_count": 0}
                }

            # Step 2: Re-rank (top 5)
            self.logger.info("Step 2/3: Cross-encoder re-ranking")
            reranked_docs = self.reranker.rerank(
                validated_query,
                candidates,
                top_k=self.settings.final_retrieval_count
            )

            # Step 3: Generate answer
            self.logger.info("Step 3/3: LLM generation with grounding")
            answer = self._generate_answer(validated_query, reranked_docs)

            # Format sources for citation
            sources = self._format_sources(reranked_docs)

            result = {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "retrieved_count": len(candidates),
                    "reranked_count": len(reranked_docs),
                    "model": self.settings.openai_chat_model
                }
            }

            self.logger.info("Query processing complete")
            return result

        except ValueError as e:
            # Validation error
            self.logger.error(f"Query validation failed: {str(e)}")
            return {
                "answer": f"Invalid query: {str(e)}",
                "sources": [],
                "metadata": {"error": str(e)}
            }
        except Exception as e:
            # Unexpected error
            self.logger.error(f"Query processing failed: {str(e)}")
            return {
                "answer": "An error occurred while processing your query. Please try again.",
                "sources": [],
                "metadata": {"error": str(e)}
            }

    def _generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generate grounded answer using LLM.

        Interview Defense:
        - Q: How do you format context for the LLM?
          A: Structured format with clear separators:
             Document 1 [Source: handbook.pdf, Page 5]
             <content>
             ---
             Document 2 [Source: policy.pdf, Page 12]
             <content>
             This helps LLM:
             1. Distinguish between documents
             2. Associate content with sources
             3. Generate accurate citations
        - Q: Why temperature=0.0?
          A: Deterministic generation:
             - Same query + same context = same answer (reproducible)
             - Important for testing and debugging
             - Production: Could use 0.1-0.3 for slight variation
        """
        # Format context from documents
        context_parts = []
        for idx, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            rerank_score = doc.metadata.get("rerank_score", None)

            score_info = f" (Relevance: {rerank_score:.3f})" if rerank_score else ""

            context_parts.append(
                f"Document {idx} [Source: {source}, Page {page}]{score_info}\n"
                f"{doc.page_content}\n"
                f"---"
            )

        context = "\n\n".join(context_parts)

        # Create prompt
        messages = self.prompt_template.format_messages(
            context=context,
            question=query
        )

        # Generate response
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            return "Failed to generate answer. Please try again."

    def _format_sources(self, documents: List[Document]) -> List[Dict]:
        """
        Format source documents for display.

        Returns:
            List of source metadata dictionaries
        """
        sources = []
        for doc in documents:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "rerank_score": doc.metadata.get("rerank_score", None),
                "content_preview": doc.page_content[:200] + "..."  # First 200 chars
            })
        return sources
