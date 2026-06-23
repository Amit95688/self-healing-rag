# SentinelRAG

**A production-grade, self-healing RAG pipeline built with LangGraph.**

Most RAG systems stop at retrieve-and-generate. SentinelRAG adds a full reliability layer on top: input safety, cross-encoder reranking, a self-healing generation loop grounded in the official Self-RAG and CRAG research, and output safety — all running on a zero-cost stack.

---

## The Problem

Standard RAG fails in five predictable ways:

| Failure | What happens | Standard RAG |
|---|---|---|
| Bad input | Prompt injection or PII reaches the LLM | No protection |
| Wrong retrieval | Irrelevant chunks go to generation | No quality check |
| Hallucination | LLM answers confidently from nothing | No verification |
| Unuseful answer | Correct facts, wrong question answered | No detection |
| Bad output | Toxic or off-topic response reaches user | No gate |

SentinelRAG has a dedicated, bounded fix for each one.

---

## Pipeline

```
User Question
      ↓
┌─────────────────────────────────────────┐
│ NODE 1 — Guard + Route (1 LLM call)     │
│ • Blocks prompt injection, jailbreaks,  │
│   PII, unsafe content                   │
│ • Decides: retrieve vs. direct answer   │
└─────────────────────────────────────────┘
      ↓ (if retrieval needed)
┌─────────────────────────────────────────┐
│ OPTIONAL — Query Decomposition          │
│ • Splits compound questions into        │
│   atomic sub-queries (1 LLM call)       │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│ OPTIONAL — HyDE Expansion               │
│ • Generates hypothetical answer and     │
│   embeds that instead of raw query      │
│   (bridges question/answer embedding    │
│   mismatch, 1 LLM call)                 │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│ NODE 3 — Hybrid Retrieval (no LLM)      │
│ • BM25 (keyword) + dense (semantic)     │
│ • EnsembleRetriever, k=8 per retriever  │
│ • Merges + deduplicates results         │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│ NODE 4 — Cross-Encoder Reranking (no LLM│
│ • ms-marco-MiniLM-L-6-v2 (local, CPU)   │
│ • Scores every (question, chunk) pair   │
│ • Passes top-2 relevant docs forward    │
│ • If zero pass: → Query Rewrite         │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│ NODE 5 — Generate from Context          │
│ • Answers from retrieved chunks ONLY    │
│ • Adds [Source: page X] citations       │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│ NODE 6 — Merged Verify (1 LLM call)     │
│ • IsSUP: is every claim in the chunks?  │
│ • IsUSE: does the answer address the    │
│   question?                             │
│ • fully_supported + useful → accept     │
│ • no_support → revise (max 1 retry)     │
│ • not_useful → rewrite query (max 1)    │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│ NODE 9 — Output Guardrail (1 LLM call)  │
│ • Blocks toxic, PII, system leaks       │
│ • Final gate before answer reaches user │
└─────────────────────────────────────────┘
      ↓
Trusted Answer with Citations
```

### Self-Healing Loops

```
verify_answer
    ├── no_support  → revise_answer → verify_answer  (max 1 retry)
    └── not_useful  → rewrite_question → retrieve    (max 1 retry)

rerank_docs
    └── zero relevant → rewrite_question → retrieve  (max 1 retry)
```

Every loop is bounded. No infinite recursion risk.

---

## What Makes This Different from Standard RAG

| | Standard RAG | SentinelRAG |
|---|---|---|
| Input safety | ❌ | ✅ Guard node (blocks injection, jailbreaks, PII) |
| Query quality | ❌ | ✅ Optional decomposition + HyDE |
| Retrieval method | Single dense search | ✅ Hybrid BM25 + dense |
| Retrieval quality | ❌ | ✅ Cross-encoder reranking (local, no API call) |
| Hallucination check | ❌ | ✅ Grounding verification (IsSUP) |
| Answer usefulness | ❌ | ✅ Usefulness check (IsUSE) |
| Self-correction | ❌ | ✅ Revision + query rewrite loops |
| Output safety | ❌ | ✅ Output guardrail |
| Retry bounds | N/A | ✅ Explicit MAX_RETRIES counters |
| Cost | Paid API | ✅ 100% free stack |

---

## Research Backing

| Component | Paper / Source |
|---|---|
| Self-healing loop (IsSUP, IsUSE) | [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511) |
| Retrieval grading + fallback | [CRAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884) |
| HyDE (optional) | [HyDE (Gao et al., 2022)](https://arxiv.org/abs/2212.10496) |
| Cross-encoder reranking | [MS-MARCO (Nogueira et al., 2019)](https://arxiv.org/abs/1910.10687) |
| Hybrid retrieval | BM25 + dense (standard BEIR baseline) |
| Official LangGraph Self-RAG | [LangChain docs](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/) |

---

## Optimizations Applied

| Optimization | Latency Impact |
|---|---|
| FAISS index persisted to disk (`./faiss_index/`) | Cold start: 4-8 min → 1-3 sec (after first run) |
| Chunks pickled to disk (`./chunks_cache.pkl`) | BM25 rebuild skipped on reload |
| SQLite LLM response cache (`./llm_cache.db`) | Repeated queries: full pipeline → ~0s |
| Guard + retrieval decision merged (2 calls → 1) | Saves ~3-6s per query |
| IsSUP + IsUSE merged (2 calls → 1) | Saves ~3-6s per query |
| Cross-encoder replaces LLM-based grading | Saves ~5-15s per query (local model, ~50ms) |
| Retrieval candidate pool truncated to top-2 | Smaller context → faster generation |
| Single MoE model (3B active params) for all LLM calls | Low per-call latency |

**Best-case call count:** 4 LLM calls (guard, generate, verify, output guard)
**Worst-case call count:** 7 LLM calls (+ decompose, + rewrite, + revise)

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| LLM | `qwen/qwen3-next-80b-a3b-instruct` via NVIDIA free endpoint | Free |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, CPU) | Free |
| Vector store | FAISS (in-process, persisted) | Free |
| Keyword search | BM25Retriever | Free |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, CPU) | Free |
| Graph orchestration | LangGraph | Free |
| LLM cache | SQLiteCache (LangChain Community) | Free |
| Document loading | PyPDFLoader | Free |

**Total infrastructure cost: ₹0**

---

## Repository Structure

```
self-healing-rag/
│
├── srag.ipynb              ← Main notebook (full pipeline, cell-by-cell)
├── src/                    ← Modular Python source
├── data/                   ← PDF documents (not committed)
├── faiss_index/            ← Persisted FAISS vector index
├── logs/                   ← Execution logs
├── eval_questions.json     ← Golden test set for evaluation
├── test.py                 ← Standalone test runner
├── requirements.txt
├── chunks_cache.pkl        ← Pickled document chunks for BM25 (auto-generated)
├── llm_cache.db            ← SQLite LLM response cache (auto-generated)
└── .github/workflows/      ← CI configuration
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Amit95688/self-healing-rag.git
cd self-healing-rag
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install langchain langchain-openai langchain-community langchain-huggingface \
    langchain-classic langgraph faiss-cpu rank_bm25 pypdf \
    sentence-transformers python-dotenv pydantic
```

**3. Set up environment variables**

Create a `.env` file in the project root:
```
NVIDIA_API_KEY=your_nvidia_api_key_here
```

Get a free API key at [build.nvidia.com](https://build.nvidia.com).

**4. Add your documents**

Place your PDF files in `./data/`. Update the paths in `load_and_chunk_docs()` inside `srag.ipynb` or `src/` to match your filenames.

**5. Run**

```bash
# Notebook
jupyter notebook srag.ipynb

# Or run tests directly
python test.py
```

The first run embeds your documents and saves the index — expect 4-8 minutes for ~4000 pages. Every subsequent run loads from cache in 1-3 seconds.

---

## Feature Toggles

Two optional features add LLM calls but improve retrieval quality on complex queries. Both default to **off** to minimize latency:

```python
USE_QUERY_DECOMPOSITION = False  # splits "compare X and Y" into two sub-queries
USE_HYDE = False                 # embeds a hypothetical answer instead of the raw question
```

Set either to `True` in `srag.ipynb` or `src/sentinel_rag.py` to enable. Each adds exactly +1 LLM call to the best-case path regardless of how many sub-queries or passages are produced.

---

## Evaluation

The `eval_questions.json` file contains a golden test set of question/answer pairs for offline evaluation. Run `test.py` to measure:

- Faithfulness (answer grounded in retrieved chunks)
- Answer relevancy (answer addresses the question)
- Hallucination rate
- Latency (per-query, p50/p95)

---

## Latency Reference

Measured on free NVIDIA endpoint + CPU embeddings + CPU reranker:

| Scenario | Expected latency |
|---|---|
| First run (cold start, 4000 pages) | 4-8 min (one-time) |
| Subsequent cold start (index cached) | 1-3 sec setup |
| Per-query, no retries (best case) | 12-25 sec |
| Per-query, with 1 revise + 1 rewrite | 30-60 sec |
| Per-query, cache hit (repeated question) | ~0 sec |

---

## License

MIT

---

## Author

**Amit Dubey**
[github.com/Amit95688](https://github.com/Amit95688) · [linkedin.com/in/amit-dubey-45292629a](https://linkedin.com/in/amit-dubey-45292629a)
