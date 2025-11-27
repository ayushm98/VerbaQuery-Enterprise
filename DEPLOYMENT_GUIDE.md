# Deployment Guide: Make VerbaQuery-Enterprise Live

**Quick Start:** Get your RAG system running in 5 minutes for interview presentation.

---

## Pre-Flight Checklist

### ✓ Already Done
- [x] Project cloned with all source code
- [x] 4 PDF documents in `data/` directory
- [x] `.env` file created

### ⚠️ To Do Before Launch
- [ ] Verify OpenAI API key in `.env`
- [ ] Install Python dependencies
- [ ] Build indexes (one-time, ~2-3 minutes)
- [ ] Test the pipeline
- [ ] Launch Streamlit app

---

## Step 1: Verify OpenAI API Key (30 seconds)

```bash
# Open .env file
cat .env | grep OPENAI_API_KEY

# Should show: OPENAI_API_KEY=sk-...
# If it shows "sk-your-key-here", replace with your actual key
```

**Edit .env if needed:**
```bash
nano .env
# or
code .env
```

**Get API key:** https://platform.openai.com/api-keys

---

## Step 2: Install Dependencies (1 minute)

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import langchain, chromadb, streamlit; print('✓ Dependencies installed')"
```

**If you get errors:**
```bash
# Use Python 3.10+ virtual environment
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 3: Build Indexes (2-3 minutes)

**This is the critical step to make the system live.**

```bash
# Ingest all PDFs and build dual indexes
python scripts/ingest_documents.py --input data/

# Expected output:
# ================================================================================
# VerbaQuery-Enterprise: Document Ingestion Pipeline
# ================================================================================
# Step 1/3: Loading PDFs from data
# Loaded X page-level documents
# Step 2/3: Chunking documents (size=1000, overlap=200)
# Created Y semantic chunks
# Step 3/3: Building dual indexes (Vector + Keyword)
# ✓ Ingestion Complete!
#   - Vector Index: ./data/indexes/chroma
#   - Keyword Index: ./data/indexes/bm25_index.pkl
#   - Total Chunks Indexed: Y
# ================================================================================
```

**Troubleshooting:**
- **Error: OpenAI API key invalid**
  - Fix: Update `OPENAI_API_KEY` in `.env` file
- **Error: No PDF files found**
  - Fix: Ensure PDFs are in `data/` directory (not `data/raw/`)
- **Slow ingestion (>5 minutes)**
  - Normal for large PDFs (embeddings take time)
  - Cost: ~$0.02 per 1M tokens

---

## Step 4: Test the Pipeline (30 seconds)

```bash
# Run end-to-end tests
python scripts/test_pipeline.py

# Expected output:
# ================================================================================
# Testing Ingestion Pipeline
# ✓ Loaded X pages
# ✓ Created Y chunks
# ================================================================================
# Testing Retrieval Pipeline
# ✓ Vector retrieval returned 5 docs
# ✓ Keyword retrieval returned 5 docs
# ✓ Hybrid retrieval returned 10 docs
# ✓ Re-ranking returned 5 docs
# ================================================================================
# Testing Query Engine
# Query: What is the refund policy?
# Answer: ...
# ✓ Query engine test complete
# ================================================================================
# Test Summary
# Ingestion: ✓ PASSED
# Retrieval: ✓ PASSED
# Query Engine: ✓ PASSED
# 🎉 All tests passed!
# ================================================================================
```

**If tests fail:**
- Check logs for specific errors
- Ensure indexes were built successfully in Step 3
- Verify `.env` has correct API key

---

## Step 5: Launch Streamlit App (10 seconds)

```bash
# Start the web interface
streamlit run app/streamlit_app.py

# Expected output:
#   You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
#   Network URL: http://192.168.x.x:8501
```

**Your app is now live!** 🚀

Open browser to `http://localhost:8501`

---

## Interview Presentation Flow

### Demo Script (5 minutes)

1. **Show the UI (30 seconds)**
   - Open `http://localhost:8501`
   - Point out: Chat interface, sidebar with config, system status

2. **Run a Query (1 minute)**
   - Type: "What is the refund policy?"
   - Wait ~3-5 seconds for response
   - **Show:**
     - Grounded answer with citations
     - Source documents (expandable)
     - Relevance scores (progress bars)
     - Metadata (retrieved count, reranked count)

3. **Explain the Architecture (2 minutes)**
   - Open sidebar: "Pipeline Stages"
   - Walk through:
     1. **Hybrid Retrieval:** Vector (semantic) + BM25 (keyword)
     2. **Re-ranking:** Flashrank cross-encoder filters top 5
     3. **Generation:** GPT-4 with strict grounding
   - **Key point:** "This is a 3-stage pipeline, not just 'throw docs at GPT'"

4. **Show the Code (1.5 minutes)**
   - Open `src/generation/query_engine.py`
   - Highlight: `query()` function (lines 71-160)
   - Walk through: Validation → Retrieval → Re-rank → Generate
   - **Key point:** "Notice the graceful degradation in error handling"

5. **Technical Deep Dive (if asked)**
   - Open `PROJECT_CONCEPTS_DEEP_DIVE.md`
   - Jump to Section 1.3: "Reciprocal Rank Fusion"
   - Explain RRF formula: `1/(60 + rank)`
   - **Key point:** "This solves the score incompatibility problem between vector and BM25"

### Sample Interview Questions & Answers

**Q: "Walk me through how a query works."**

A: "When a user asks 'What is the refund policy?':
1. The query goes to HybridRetriever, which runs two parallel searches:
   - Vector search finds semantically similar chunks (e.g., 'return policy', 'money back')
   - BM25 finds exact keyword matches ('refund')
2. We use Reciprocal Rank Fusion to combine results - this handles the score incompatibility between the two methods
3. The top 10 candidates go to Flashrank, a cross-encoder that re-scores by looking at query-document interactions
4. Top 5 highest-quality chunks are injected into GPT-4 with a strict grounding prompt
5. GPT-4 generates an answer with page citations, which we return to the user"

**Q: "Why hybrid search instead of just vector search?"**

A: "Vector search alone has a weakness with exact matches. For example, if the user searches for a policy code like 'AU-2024-001', vector search might match 'AU-2024-002' because they're semantically similar. BM25 excels at exact lexical matching. Research shows hybrid retrieval outperforms single-method by 15-30% on most benchmarks."

**Q: "What's the cost per query?"**

A: "Breakdown:
- Query embedding: ~10 tokens × $0.02/1M = $0.0000002 (negligible)
- GPT-4 generation: ~2000 tokens × $30/1M = $0.06
- Total: ~6 cents per query

The context window dominates cost. For production, we could use GPT-3.5-turbo at $1.50/1M for 20x cost reduction with minimal quality loss."

**Q: "How do you prevent hallucination?"**

A: "Multi-layer approach:
1. Prompt engineering: Explicit 'ONLY use provided context' instruction
2. Re-ranking: Only send highest-quality docs (top 5 after cross-encoder)
3. No-context handling: If retrieval fails, return 'I don't have enough information' instead of making up an answer
4. Citation requirement: Force the LLM to cite page numbers for every claim"

**Q: "How would you scale this?"**

A: "Current bottlenecks:
1. Synchronous I/O: Move to FastAPI with async endpoints
2. Single-threaded: Add request queuing with Celery/Redis
3. Re-ranking latency: Batch re-ranking requests
4. Embedding cost: Cache frequent queries in Redis

For 1000 queries/day: Current architecture is fine
For 100K queries/day: Need async + caching + horizontal scaling"

---

## Quick Recovery Commands

**If something breaks during demo:**

```bash
# Restart Streamlit
Ctrl+C
streamlit run app/streamlit_app.py

# Rebuild indexes (if corrupted)
python scripts/ingest_documents.py --input data/ --rebuild

# Check logs
tail -f <streamlit-output>

# Test specific component
python -c "from src.retrieval import HybridRetriever; r = HybridRetriever(); print(r.retrieve('test', k=5))"
```

---

## Advanced: Deploy to Cloud (Optional)

### Option 1: Streamlit Community Cloud (Easiest)

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Add secrets (OpenAI API key)
5. Deploy with one click

**Limitations:**
- Free tier: Limited resources
- Cold starts: Slow first load
- Not suitable for high traffic

### Option 2: Docker + AWS/GCP (Production-Grade)

```dockerfile
# Dockerfile (already provided if needed)
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Build indexes at container startup
RUN python scripts/ingest_documents.py --input data/

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py"]
```

```bash
# Build and run
docker build -t verbaquery .
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY verbaquery
```

**Deploy to AWS:**
- Use AWS Fargate (serverless containers)
- Or EC2 with Docker Compose
- Add Application Load Balancer for scaling

---

## Performance Benchmarks

**Expected Response Times:**
- Query validation: <1ms
- Hybrid retrieval: ~500ms
- Re-ranking (10 docs): ~2s
- GPT-4 generation: ~1-2s
- **Total:** 3-5 seconds per query

**Resource Usage:**
- Memory: ~500MB (including models)
- CPU: Low (mostly I/O-bound waiting for OpenAI API)
- Disk: ~50MB indexes for 4 PDFs

---

## Troubleshooting Guide

### Issue: "Module not found" errors
**Fix:** `pip install -r requirements.txt`

### Issue: "No documents retrieved"
**Fix:** Rebuild indexes with `--rebuild` flag

### Issue: "OpenAI API rate limit exceeded"
**Fix:**
- Wait 60 seconds
- Or upgrade to paid tier (https://platform.openai.com/account/billing)

### Issue: "Streamlit connection error"
**Fix:**
```bash
# Check if port 8501 is in use
lsof -i :8501
# Kill process if needed
kill -9 <PID>
```

### Issue: "Slow query responses (>10s)"
**Fix:**
- Check internet connection (OpenAI API calls)
- Reduce `INITIAL_RETRIEVAL_COUNT` to 5 (faster re-ranking)
- Use `gpt-3.5-turbo` instead of `gpt-4-turbo-preview`

---

## Post-Interview Follow-Up

**What to send:**
1. GitHub repository link
2. Live demo link (if deployed to Streamlit Cloud)
3. `PROJECT_CONCEPTS_DEEP_DIVE.md` (shows deep technical understanding)
4. This deployment guide (shows you can ship production systems)

**Email template:**
```
Subject: VerbaQuery-Enterprise Demo - [Your Name]

Hi [Interviewer Name],

Thank you for the opportunity to present my RAG system. Here are the resources:

🔗 Live Demo: [Streamlit Cloud URL]
📂 GitHub: https://github.com/[your-username]/VerbaQuery-Enterprise
📄 Technical Deep Dive: [Link to PROJECT_CONCEPTS_DEEP_DIVE.md]

Key highlights:
• 3-stage pipeline: Hybrid retrieval → Cross-encoder re-ranking → Grounded generation
• Reciprocal Rank Fusion for score fusion (handles vector/BM25 incompatibility)
• Production-ready: Graceful degradation, structured logging, fail-fast validation

The system is fully functional and ready for production deployment with minimal changes.

Looking forward to discussing this further.

Best,
[Your Name]
```

---

## Summary: 5-Minute Checklist

- [ ] **Minute 1:** Verify `.env` has valid OpenAI API key
- [ ] **Minute 2:** Run `pip install -r requirements.txt`
- [ ] **Minute 3-4:** Run `python scripts/ingest_documents.py --input data/`
- [ ] **Minute 5:** Run `streamlit run app/streamlit_app.py`
- [ ] **Bonus:** Open browser to `http://localhost:8501` and test a query

**You're now ready to impress your interviewer!** 🎯
