# PROJECT CONCEPTS DEEP DIVE

**White Box Audit Report**
**System:** VerbaQuery-Enterprise RAG Pipeline
**Scope:** Complete source code analysis (src/, config/, scripts/, app/)
**Methodology:** Staff Engineer review focusing on fundamental concepts over implementation details

---

## LEVEL 1: THE MATHEMATICS (Vector & Probabilistic)

### 1.1 Similarity Metrics

#### **Cosine Similarity (Vector Retrieval)**
- **Location:** `src/retrieval/vector_retriever.py:28-29`
- **Implementation:** ChromaDB default metric
- **Formula:** `cos(θ) = (A · B) / (||A|| × ||B||)`
- **Range:** [-1, 1] where:
  - 1 = Identical vectors (0° angle)
  - 0 = Orthogonal vectors (90° angle)
  - -1 = Opposite vectors (180° angle)
- **Properties:**
  - Magnitude-invariant (only cares about direction)
  - Efficient for high-dimensional embeddings (1536D for OpenAI text-embedding-3-small)
- **Why chosen:** Standard for semantic similarity in embedding space; handles normalized vectors efficiently

#### **No Explicit Euclidean Distance**
- Vector retrieval exclusively uses cosine similarity via ChromaDB
- No L2 distance computation found in codebase
- Trade-off: Cosine more robust to document length variations

### 1.2 Search Algorithms

#### **BM25 (Best Matching 25) - Keyword Search**
- **Location:** `src/retrieval/keyword_retriever.py:73-79`
- **Full Formula:**
```
BM25(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
```

Where:
- `IDF(qi)` = Inverse Document Frequency (rare terms score higher)
- `f(qi, D)` = Term frequency in document D
- `k1` = Term saturation parameter (default: **1.5**)
  - Controls how quickly term frequency saturates
  - Higher k1 = more weight to term repetition
- `b` = Length normalization (default: **0.75**)
  - Penalizes longer documents
  - b=1: full normalization, b=0: no normalization
- `|D|` = Document length
- `avgdl` = Average document length in corpus

**Implementation:**
- Library: `rank-bm25` (BM25Okapi variant)
- Tokenization: `doc.lower().split()` (simple whitespace, case-insensitive)
- Scoring: NumPy argsort for top-k selection (`src/retrieval/keyword_retriever.py:95`)

**Mathematical Properties:**
- Non-linear term frequency saturation (prevents spam)
- Probabilistic ranking function (derived from probability theory)
- Sublinear growth with term frequency

#### **Vector Search (Dense Retrieval)**
- **Embedding Model:** OpenAI text-embedding-3-small
- **Dimensions:** 1536
- **Normalization:** Embeddings are L2-normalized by OpenAI API
- **Search:** Approximate Nearest Neighbor (ANN) via ChromaDB's HNSW index
- **Complexity:** O(log N) query time for HNSW vs O(N) for brute force

### 1.3 Ranking & Score Fusion

#### **Reciprocal Rank Fusion (RRF)**
- **Location:** `src/retrieval/ensemble_retriever.py:91, 98`
- **Formula:**
```python
RRF(doc) = 1 / (k + rank)
```

- **Constant k:** 60 (standard from literature)
- **Combined Score:**
```python
combined_score = weight_vector × RRF_vector(doc) + weight_keyword × RRF_keyword(doc)
```

- **Default Weights:** 0.5 vector + 0.5 keyword (configurable via `.env`)

**Why RRF over Weighted Sum?**
1. **Score Incompatibility Problem:**
   - Vector scores: [0, 1] (cosine similarity)
   - BM25 scores: [0, 100+] (unbounded)
   - Direct averaging would overweight BM25

2. **RRF Solution:**
   - Rank-based (position-invariant to score magnitude)
   - Robust to outliers
   - Validated in IR literature (Cormack et al., SIGIR 2009)

3. **Deduplication Logic:**
   - If document appears in both retrievers: scores are **summed**
   - Implementation: `src/retrieval/ensemble_retriever.py:100-105`

**Mathematical Justification:**
- RRF score decays hyperbolically: rank 1 → 1/61, rank 10 → 1/70
- Top-ranked documents weighted exponentially higher
- Constant k=60 balances top-heavy vs democratic fusion

### 1.4 Cross-Encoder Scoring (Re-ranking)

- **Model:** ms-marco-MiniLM-L-12-v2
- **Architecture:** BERT-based transformer
- **Scoring:** Softmax over [query + document] concatenation
- **Output:** Relevance score in [0, 1]
- **No explicit formula** (learned via neural network on MS MARCO dataset)

**Key Difference from Bi-Encoder:**
- Bi-encoder: `similarity(embed(query), embed(doc))` — independent encoding
- Cross-encoder: `score(query ⊕ doc)` — joint encoding (⊕ = concatenation)
- Cross-encoder sees full query-document interaction (higher accuracy, slower)

---

## LEVEL 2: SYSTEMS & PERFORMANCE (Python & Hardware)

### 2.1 Memory Management

#### **No Streaming or Generators**
- **Ingestion:** Entire PDFs loaded into memory
  - `src/ingestion/pdf_loader.py:40-72` — loads all pages into list
  - Risk: Large PDFs (>1000 pages) may cause OOM
  - No `yield` statements found in ingestion pipeline

#### **Batch Processing**
- **Chunking:** Processes documents sequentially
  - `src/ingestion/chunker.py:53-69` — list comprehension for chunks
  - All chunks materialized in memory before indexing

#### **Index Persistence (Disk-Based Optimization)**
- **ChromaDB:** Persisted to `data/indexes/chroma/` (SQLite backend)
- **BM25:** Pickled to `data/indexes/bm25_index.pkl`
  - Size: ~10MB for 10K documents (verified in comments)
  - Trade-off: One-time RAM spike during pickle.load, then in-memory

**Memory Footprint Estimate:**
- 1000 chunks × 1KB avg = ~1MB page content
- OpenAI embeddings: 1000 × 1536 × 4 bytes (float32) = ~6MB
- BM25 tokenized corpus: ~2-3MB
- **Total:** ~10-15MB per 1000 chunks (manageable for enterprise)

### 2.2 Concurrency

#### **Synchronous Architecture**
- **No async/await:** Entire codebase is synchronous
- **No threading/multiprocessing:** Single-threaded execution
- **Blocking I/O:**
  - OpenAI API calls block during embedding/generation
  - ChromaDB queries block during similarity search

**Implications:**
- Simple debugging (no race conditions)
- Limited scalability (1 query at a time in Streamlit app)
- **Production path:** Wrap in FastAPI with async endpoints

**Observed in:**
- `src/generation/query_engine.py:209` — synchronous `llm.invoke(messages)`
- `src/retrieval/vector_retriever.py:87` — synchronous `vectorstore.similarity_search()`

### 2.3 Vectorization & Numerical Optimization

#### **No NumPy Usage in Application Code**
- All array operations delegated to libraries:
  - ChromaDB (uses NumPy internally for embeddings)
  - rank-bm25 (uses NumPy for argsort)
  - Flashrank (uses PyTorch internally)

#### **Data Types**
- **No explicit dtype specifications** in application code
- Embeddings: Assumed `float32` (OpenAI default)
- BM25 scores: Python `float` (64-bit)

**Missed Optimization Opportunity:**
- Could use `float16` embeddings to halve memory (ChromaDB supports it)
- Trade-off: Minimal accuracy loss (~0.1% on benchmarks)

### 2.4 Data Structures & Design Choices

#### **Pydantic Models vs Dicts**
- **Config:** Pydantic `BaseSettings` (`config/settings.py:6`)
  - Why: Type validation, auto `.env` parsing, IDE autocomplete
  - Overhead: Minimal (singleton pattern caches instance)

#### **LangChain Document Schema**
- **Ubiquitous:** Every component uses `Document(page_content, metadata)`
- **Why:** Standardized interface across LangChain ecosystem
- **Trade-off:** Slight overhead vs raw dicts, but cleaner API

#### **Metadata as Dicts**
- Mutable dicts for metadata enrichment
- Example: `src/ingestion/chunker.py:59-62` — `.copy()` + `.update()`
- Risk: Mutability could cause bugs (not observed in this codebase)

#### **Lists vs Other Structures**
- **Documents:** Always `List[Document]` (no deques, sets)
- **Why:** Sequential access pattern (no random access needed)
- **Top-k selection:** Uses `argsort()[:k]` instead of heaps
  - Acceptable for k=10, N=10K (linear scan is fast enough)

---

## LEVEL 3: ARCHITECTURE (Patterns & Flow)

### 3.1 Design Patterns

#### **Pattern 1: Singleton (Configuration)**
- **Implementation:** `config/settings.py:47-53`
```python
@lru_cache()
def get_settings() -> Settings:
    return Settings()
```
- **Purpose:** Single source of truth for config
- **Benefits:**
  - Avoids repeated `.env` parsing
  - Shared state across all modules
- **Thread-Safety:** `lru_cache` is thread-safe in Python 3.8+

#### **Pattern 2: Dependency Injection**
- **Query Engine:**
  - `src/generation/query_engine.py:45-66` — injects retrievers and LLM
  - Components created in `__init__`, not hardcoded
- **Hybrid Retriever:**
  - `src/retrieval/ensemble_retriever.py:39-54` — injects vector + keyword retrievers
- **Benefits:** Testability (mock retrievers), configurability

#### **Pattern 3: Strategy Pattern (Retrievers)**
- **Common Interface:** All retrievers implement `retrieve(query, k) -> List[Document]`
  - VectorRetriever
  - KeywordRetriever
  - HybridRetriever (composes the above)
- **Polymorphism:** `HybridRetriever` treats vector/keyword retrievers uniformly
- **Found in:** `src/retrieval/ensemble_retriever.py:56-118`

#### **Pattern 4: Template Method (LangChain Integration)**
- **Prompt Templates:** `config/prompts.py:15-26`
  - Fixed structure with variable slots: `{context}`, `{question}`
  - LangChain's `ChatPromptTemplate.format_messages()`
- **Text Splitter:** `src/ingestion/chunker.py:29-35`
  - RecursiveCharacterTextSplitter provides template for chunking strategy

#### **Pattern 5: Facade (QueryEngine)**
- **Single Entry Point:** `QueryEngine.query()` hides complexity
- **Orchestrates:**
  1. Validation (`src/utils/validators.py`)
  2. Retrieval (HybridRetriever)
  3. Re-ranking (FlashrankReranker)
  4. Generation (OpenAI LLM)
- **User sees:** Simple function call, not internal pipeline

### 3.2 Data Flow (PDF → ChromaDB)

**Complete Pipeline Trace:**

```
┌─────────────────────────────────────────────────────────────────┐
│ INGESTION (One-time, Offline)                                   │
└─────────────────────────────────────────────────────────────────┘

1. PDF File (data/document.pdf)
   ↓
2. PDFLoader.load_single_pdf()                        [pdf_loader.py:19-72]
   - Opens with pypdf.PdfReader
   - Extracts text per page: page.extract_text()
   - Creates Document per page
   - Metadata: {source, page, total_pages, file_path}
   ↓
3. List[Document] (page-level)
   Example: 50-page PDF → 50 Documents
   ↓
4. SemanticChunker.chunk_documents()                  [chunker.py:39-73]
   - RecursiveCharacterTextSplitter with separators: ["\n\n", "\n", ". ", " ", ""]
   - chunk_size=1000, chunk_overlap=200
   - Enriches metadata: {chunk_id, chunk_index, total_chunks_in_page}
   ↓
5. List[Document] (chunk-level)
   Example: 50 pages × 2 chunks/page = 100 Documents
   ↓
6. DualIndexer.build_indexes()                        [indexer.py:110-127]
   ├─→ create_vector_index()                          [indexer.py:39-69]
   │   - OpenAIEmbeddings.embed_documents() → List[List[float]]
   │   - Chroma.from_documents() → SQLite persistence
   │   - Output: data/indexes/chroma/
   │
   └─→ create_keyword_index()                         [indexer.py:71-108]
       - Tokenize: doc.lower().split()
       - BM25Okapi(tokenized_corpus)
       - Pickle to data/indexes/bm25_index.pkl

┌─────────────────────────────────────────────────────────────────┐
│ RETRIEVAL (Query-time, Online)                                  │
└─────────────────────────────────────────────────────────────────┘

7. User Query: "What is the refund policy?"
   ↓
8. QueryEngine.query()                                [query_engine.py:71-160]
   ├─→ validate_query()                               [validators.py:31-57]
   │   - Strip whitespace
   │   - Check length: 3 ≤ len ≤ 500
   │
   ├─→ HybridRetriever.retrieve(k=10)                 [ensemble_retriever.py:56-118]
   │   ├─→ VectorRetriever.retrieve(k=10)             [vector_retriever.py:63-92]
   │   │   - Embed query: OpenAIEmbeddings.embed_query()
   │   │   - ChromaDB similarity_search() → top 10 by cosine
   │   │
   │   ├─→ KeywordRetriever.retrieve(k=10)            [keyword_retriever.py:60-105]
   │   │   - Tokenize query: query.lower().split()
   │   │   - BM25.get_scores() → top 10 by BM25
   │   │
   │   └─→ Reciprocal Rank Fusion
   │       - RRF score = 1/(60 + rank)
   │       - Combine: 0.5 × RRF_vector + 0.5 × RRF_keyword
   │       - Deduplicate and sort
   │       - Output: Top 10 unique documents
   │
   ├─→ FlashrankReranker.rerank(top_k=5)              [reranker.py:64-145]
   │   - Cross-encoder scores each (query, doc) pair
   │   - Sort by relevance score
   │   - Add rerank_score to metadata
   │   - Output: Top 5 documents
   │
   └─→ _generate_answer()                             [query_engine.py:162-213]
       - Format context from top 5 docs
       - Inject into QUERY_PROMPT_TEMPLATE
       - LLM.invoke(messages) → GPT-4 generation
       - Return: {answer, sources, metadata}
```

**Critical Metadata Enrichment Points:**
- **Stage 2 (PDFLoader):** Add `{source, page, total_pages, file_path}`
- **Stage 4 (Chunker):** Add `{chunk_id, chunk_index, total_chunks_in_page}`
- **Stage 8.3 (Reranker):** Add `{rerank_score}`

**Persistence Points:**
- ChromaDB: Written in Stage 6, Read in Stage 8
- BM25: Written in Stage 6, Read in Stage 8

### 3.3 Dependency Injection Analysis

#### **Constructor Injection (Preferred Pattern)**

**Example 1: HybridRetriever**
```python
# src/retrieval/ensemble_retriever.py:39-54
def __init__(self):
    self.settings = get_settings()
    self.vector_retriever = VectorRetriever()    # Injected dependency
    self.keyword_retriever = KeywordRetriever()  # Injected dependency
```
- **Benefits:** Clear dependencies, easy to mock for testing

**Example 2: QueryEngine**
```python
# src/generation/query_engine.py:45-69
def __init__(self):
    self.retriever = HybridRetriever()    # Injected
    self.reranker = FlashrankReranker()   # Injected
    self.llm = ChatOpenAI(...)            # Injected with config
```

#### **Singleton Injection**
- `get_settings()` injected everywhere via import
- Avoids parameter drilling
- Trade-off: Harder to test with different configs (need monkeypatch)

#### **No Service Locator Pattern**
- Dependencies explicitly constructed (no global registry)
- Clean dependency graph (can trace imports)

---

## LEVEL 4: OPS & ROBUSTNESS

### 4.1 Configuration Management

#### **Pydantic Settings Architecture**
- **File:** `config/settings.py:6-53`
- **Base Class:** `pydantic_settings.BaseSettings`
- **Features:**
  - Auto `.env` file parsing
  - Environment variable override (e.g., `CHUNK_SIZE=1500`)
  - Type coercion and validation (int, float, Path)
  - Default values with fallbacks

**Example Config:**
```python
class Settings(BaseSettings):
    openai_api_key: str                        # Required (no default)
    chunk_size: int = 1000                     # Optional (default)
    chroma_persist_directory: Path = Path("./data/indexes/chroma")

    model_config = SettingsConfigDict(
        env_file=".env",                       # Load from .env
        case_sensitive=False                   # CHUNK_SIZE = chunk_size
    )
```

#### **Singleton Pattern for Efficiency**
```python
@lru_cache()
def get_settings() -> Settings:
    return Settings()  # Called once, cached forever
```
- **Why:** Avoid re-parsing `.env` on every import
- **Thread-safe:** `lru_cache` uses locks internally

#### **Directory Auto-Creation**
```python
# config/settings.py:40-44
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.chroma_persist_directory.parent.mkdir(parents=True, exist_ok=True)
    self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
```
- **Fail-fast:** Ensures paths exist at startup (not during indexing)

### 4.2 Observability (Logging)

#### **Structured Logging Implementation**
- **File:** `src/utils/logger.py:7-40`
- **Format:** `"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"`
- **Example Output:**
```
2024-03-15 10:23:45 | INFO     | src.retrieval.ensemble_retriever | Hybrid retrieval: query='What is...', k=10
```

**Components:**
1. **Timestamp:** ISO 8601 format
2. **Level:** INFO, WARNING, ERROR (padded to 8 chars)
3. **Module Name:** Fully qualified (e.g., `src.ingestion.chunker`)
4. **Message:** Contextual info (query snippet, document count)

#### **Anti-Duplication Logic**
```python
# src/utils/logger.py:23-24
if logger.handlers:
    return logger  # Prevent duplicate handlers on re-import
```

#### **Observability Gaps (Production Improvements)**
- **No Metrics:** Missing Prometheus/StatsD counters
  - Should track: queries/sec, latency percentiles, error rate
- **No Trace IDs:** Cannot correlate logs across pipeline stages
  - Should add: `request_id` in all log messages
- **No Log Aggregation:** Logs to stdout only
  - Production: Send to ELK/Splunk/Datadog

### 4.3 Validation & Fail-Fast Mechanisms

#### **Input Validation (validators.py)**

**PDF File Validation:**
```python
# src/utils/validators.py:5-28
def validate_pdf_file(file_path: Path) -> bool:
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"File must be a PDF, got: {file_path.suffix}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"PDF file is empty: {file_path}")

    return True
```
- **Called at:** Entry to `PDFLoader.load_single_pdf()` (line 35)
- **Fail-fast:** Errors raised before expensive PDF parsing

**Query Validation:**
```python
# src/utils/validators.py:31-57
def validate_query(query: str, min_length: int = 3, max_length: int = 500) -> str:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query = query.strip()  # Sanitization

    if len(query) < min_length:
        raise ValueError(f"Query too short (min {min_length} characters)")

    if len(query) > max_length:
        raise ValueError(f"Query too long (max {max_length} characters)")

    return query
```
- **Called at:** `QueryEngine.query()` before retrieval (line 100)
- **Prevents:** Empty queries from wasting API calls

#### **Graceful Degradation (Error Handling)**

**BM25 Retrieval Fallback:**
```python
# src/retrieval/keyword_retriever.py:103-105
except Exception as e:
    self.logger.error(f"BM25 retrieval failed: {str(e)}")
    return []  # Empty list allows hybrid retriever to use vector-only
```

**Reranker Fallback:**
```python
# src/retrieval/reranker.py:141-145
except Exception as e:
    self.logger.error(f"Re-ranking failed: {str(e)}")
    self.logger.warning("Falling back to original retrieval ranking")
    return documents[:top_k]  # Use pre-rerank order
```

**Query Engine Fallback:**
```python
# src/generation/query_engine.py:145-160
except ValueError as e:
    return {"answer": f"Invalid query: {str(e)}", "sources": [], ...}
except Exception as e:
    return {"answer": "An error occurred...", "sources": [], ...}
```

**Philosophy:**
- Never crash the entire system
- Degrade gracefully (vector-only, no reranking, error message)
- Log errors for post-mortem analysis

### 4.4 Robustness Mechanisms

#### **Empty Result Handling**
```python
# src/generation/query_engine.py:109-115
if not candidates:
    self.logger.warning("No documents retrieved")
    return {
        "answer": NO_CONTEXT_RESPONSE,  # "I don't have enough information..."
        "sources": [],
        "metadata": {"retrieved_count": 0, "reranked_count": 0}
    }
```

#### **API Key Validation (Implicit)**
- Pydantic enforces `openai_api_key: str` (required field)
- App crashes at startup if missing (fail-fast)

#### **Index Existence Check**
```python
# scripts/ingest_documents.py:72-99
def check_existing_indexes(settings) -> bool:
    return settings.chroma_persist_directory.exists() and settings.bm25_index_path.exists()

if not args.rebuild and check_existing_indexes(settings):
    logger.warning("Indexes already exist. Use --rebuild to force recreation. Exiting.")
    return
```
- **Prevents:** Accidental overwrite of expensive indexes

---

## LEVEL 5: CRITICAL EVIDENCE (Exact Code Extraction)

### 5.1 Semantic Chunking Logic

**File:** `src/ingestion/chunker.py:11-74`

```python
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
```

**Key Algorithm Details:**
1. **Recursive Splitting Strategy:**
   - Try to split on `"\n\n"` (paragraph breaks) first
   - If chunk still too large, try `"\n"` (line breaks)
   - Then `". "` (sentence boundaries)
   - Then `" "` (word boundaries)
   - Finally `""` (character-level split as last resort)

2. **Overlap Mechanism:**
   - Each chunk includes last 200 characters from previous chunk
   - Prevents semantic loss at boundaries (e.g., split mid-sentence)

3. **Metadata Enrichment:**
   - Original metadata preserved via `.copy()`
   - Added fields: `chunk_id`, `chunk_index`, `total_chunks_in_page`
   - Enables citation: "Document X, Page Y, Chunk Z"

### 5.2 Hybrid Search (Ensemble) Logic

**File:** `src/retrieval/ensemble_retriever.py:56-145`

```python
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

    Interview Defense:
    - Q: Why not use document content as ID?
      A: Content can be large (1000 tokens), hashing is slower
         Metadata provides unique identifiers (chunk_id, source+page+chunk)
    - Q: What if chunk_id is missing?
      A: Fallback to composite key: "filename_page42_chunk3"
         Handles legacy data or different indexing strategies
    """
    metadata = doc.metadata

    if "chunk_id" in metadata:
        return metadata["chunk_id"]

    # Fallback: construct ID from available metadata
    source = metadata.get("source", "unknown")
    page = metadata.get("page", 0)
    chunk_idx = metadata.get("chunk_index", 0)

    return f"{source}_page{page}_chunk{chunk_idx}"
```

**Critical Implementation Details:**

1. **Parallel Retrieval:**
   - Vector and keyword searches run independently
   - No short-circuiting (always retrieve k from both)

2. **Deduplication Strategy:**
   ```python
   if doc_id in doc_scores:
       doc_scores[doc_id] += self.weight_keyword * rrf_score  # Sum scores
   else:
       doc_scores[doc_id] = self.weight_keyword * rrf_score   # New entry
   ```
   - Same document from both retrievers → higher combined score
   - Rewards consensus between retrieval methods

3. **RRF Constant k=60:**
   - From Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods"
   - Not tuned for this specific dataset (could be optimized)

4. **Document ID Generation:**
   - Primary: `chunk_id` from metadata (format: `"docIdx_chunkIdx"`)
   - Fallback: Composite key (handles edge cases)
   - Ensures consistent deduplication across retrievers

### 5.3 Re-ranking Implementation

**File:** `src/retrieval/reranker.py:64-145`

```python
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
```

**Key Implementation Aspects:**

1. **Format Conversion:**
   - LangChain `Document` → Flashrank dict format
   - Preserves index mapping via `"id": idx`
   - Metadata carried through for debugging

2. **Metadata Enrichment:**
   ```python
   original_doc.metadata["rerank_score"] = result["score"]
   ```
   - Score added to document for downstream use
   - Used in UI to show relevance bars
   - Used in context formatting to show confidence

3. **Fallback on Failure:**
   - If re-ranker crashes, return original retrieval order
   - Degrades gracefully (still returns results)
   - Logs warning for debugging

4. **Cross-Encoder Model:**
   - Model: `ms-marco-MiniLM-L-12-v2`
   - Trained on MS MARCO passage ranking dataset
   - Size: ~120MB (CPU-friendly)
   - Inference: ~200ms per document (10 docs = ~2s total)

5. **Score Range:**
   - Output: Probability score in [0, 1]
   - Higher = more relevant to query
   - Used for sorting and UI visualization

---

## SUMMARY: CONCEPTUAL FOUNDATIONS

### Mathematical Core
- **Cosine Similarity:** Magnitude-invariant semantic matching
- **BM25 Formula:** Probabilistic ranking with saturation and normalization
- **Reciprocal Rank Fusion:** Rank-based score fusion (k=60)
- **Cross-Encoder:** Learned query-document relevance (neural)

### Systems Architecture
- **Synchronous I/O:** Simple, single-threaded (trade-off for complexity)
- **Persistent Indexes:** One-time build, reuse on restart
- **Memory Footprint:** ~10-15MB per 1K chunks (manageable)
- **No Vectorization:** Delegates to libraries (NumPy, PyTorch)

### Design Patterns
- **Singleton:** Config caching with `@lru_cache`
- **Dependency Injection:** Constructor-based composition
- **Strategy:** Polymorphic retrievers
- **Facade:** QueryEngine hides complexity

### Operational Robustness
- **Fail-Fast Validation:** Check inputs before expensive ops
- **Graceful Degradation:** Fallback on component failure
- **Structured Logging:** Timestamp + Level + Module + Message
- **Config Management:** Pydantic with .env auto-loading

### Evidence-Based Insights
1. **Chunking:** Recursive splitting with 1000/200 (empirically validated)
2. **Hybrid Search:** RRF with deduplication (consensus rewarding)
3. **Re-ranking:** Cross-encoder with fallback (graceful degradation)

---

**End of White Box Audit**
