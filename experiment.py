"""
============================================================
  RAG Strategies Experiment — Suraj's TDS Article
  "I Tested 3 RAG Retrieval Strategies on the Same Dataset
   — Here's What Actually Worked"

  Dataset  : Wikipedia articles on AI/ML topics (via wikipedia-api)
  Tools    : LangChain · FAISS · rank_bm25 · Groq (llama3)
  Strategies:
    1. Vanilla RAG      — dense vector search only
    2. Hybrid RAG       — BM25 + semantic + RRF fusion
    3. Self-Corrective  — CRAG-style relevance gating
============================================================
  SETUP (run once in Colab or terminal):
    pip install langchain langchain-community langchain-groq
    pip install faiss-cpu sentence-transformers rank-bm25
    pip install wikipedia-api pandas tabulate colorama
============================================================
"""

# ─────────────────────────────────────────────
#  0. IMPORTS & CONFIG
# ─────────────────────────────────────────────
import os
import time
import warnings
warnings.filterwarnings("ignore")

import wikipediaapi
import numpy as np
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init
init(autoreset=True)

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch
import faiss

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ── SET YOUR GROQ API KEY ──
# Get free key at: https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key")
MODEL_NAME   = "llama-3.3-70b-versatile"   # free on Groq

print(Fore.CYAN + "=" * 60)
print("  RAG Strategies Experiment")
print("  TDS Article by Suraj Kumar Jha")
print("=" * 60)


# ─────────────────────────────────────────────
#  1. FETCH WIKIPEDIA ARTICLES
# ─────────────────────────────────────────────
print(Fore.YELLOW + "\n[Step 1] Fetching Wikipedia articles on AI/ML topics...")

TOPICS = [
    "LangChain",
    "Retrieval-augmented generation",
    "Large language model",
    "FAISS",
    "Sentence embedding",
    "Transformer (deep learning architecture)",
    "Prompt engineering",
    "Vector database",
    "Hallucination (artificial intelligence)",
    "Fine-tuning (deep learning)",
    "Llama (language model)",
    "Generative pre-trained transformer",
    "Natural language processing",
    "OpenAI",
    "Hugging Face",
]

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="RAG-Experiment/1.0 (suraj.tds.article@example.com)"
)

documents = []
doc_titles = []

CHUNK_SIZE = 500
OVERLAP = 100

for topic in TOPICS:
    page = wiki.page(topic)

    if not page.exists():
        continue

    # Use FULL article instead of summary
    text = page.text.strip()

    if len(text) < 500:
        continue

    chunk_count = 0

    for start in range(0, len(text), CHUNK_SIZE - OVERLAP):
        chunk = text[start:start + CHUNK_SIZE]

        if len(chunk) < 200:
            continue

        documents.append(chunk)

        doc_titles.append(
            f"{topic}_chunk_{chunk_count}"
        )

        chunk_count += 1

    print(f"✓ {topic}: {chunk_count} chunks")

    time.sleep(0.3)

print(
    Fore.GREEN +
    f"\nLoaded {len(documents)} chunks from {len(TOPICS)} Wikipedia articles!"
)


# ─────────────────────────────────────────────
#  2. LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────
print(Fore.YELLOW + "\n[Step 2] Loading sentence-transformer embedding model...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-compute embeddings for all documents
print("  Computing document embeddings...")
doc_embeddings_tensor = embedder.encode(documents, convert_to_tensor=True, show_progress_bar=True)
doc_embeddings_np     = doc_embeddings_tensor.cpu().numpy().astype("float32")

# Build FAISS index
dimension  = doc_embeddings_np.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)   # Inner Product = cosine if normalized
faiss.normalize_L2(doc_embeddings_np)
faiss_index.add(doc_embeddings_np)

print(Fore.GREEN + f"  FAISS index built — {faiss_index.ntotal} vectors, dim={dimension}")


# ─────────────────────────────────────────────
#  3. BUILD BM25 INDEX
# ─────────────────────────────────────────────
print(Fore.YELLOW + "\n[Step 3] Building BM25 lexical index...")

tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

print(Fore.GREEN + "  BM25 index ready!")


# ─────────────────────────────────────────────
#  4. GROQ LLM
# ─────────────────────────────────────────────
print(Fore.YELLOW + "\n[Step 4] Initializing Groq LLM...")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=0.2,
    max_tokens=300,
)

def generate_answer(question: str, context: str) -> str:
    """Generate answer from LLM given retrieved context."""
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the context doesn't contain enough information, say "Not enough context."

Context:
{context}

Question: {question}

Answer (2-3 sentences):"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


# ─────────────────────────────────────────────
#  5. RETRIEVAL FUNCTIONS
# ─────────────────────────────────────────────

def retrieve_vanilla(query: str, top_k: int = 3):
    """Strategy 1: Pure dense vector search with FAISS."""
    q_emb = embedder.encode([query], convert_to_tensor=False).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = faiss_index.search(q_emb, top_k)
    return indices[0].tolist(), scores[0].tolist()


def retrieve_hybrid(query: str, top_k: int = 3):
    """Strategy 2: BM25 + Semantic fused via Reciprocal Rank Fusion (RRF)."""
    n = len(documents)
    K = 60  # RRF constant (standard academic value)

    # --- BM25 ranking ---
    bm25_scores  = bm25.get_scores(query.lower().split())
    bm25_ranked  = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)

    # --- Semantic ranking ---
    q_emb = embedder.encode([query], convert_to_tensor=False).astype("float32")
    faiss.normalize_L2(q_emb)
    sem_scores, sem_indices = faiss_index.search(q_emb, n)
    sem_ranked = sem_indices[0].tolist()

    # --- RRF fusion ---
    rrf = {i: 0.0 for i in range(n)}
    for rank, idx in enumerate(bm25_ranked):
        rrf[idx] += 1.0 / (K + rank + 1)
    for rank, idx in enumerate(sem_ranked):
        rrf[idx] += 1.0 / (K + rank + 1)

    final_ranked = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)
    top_indices  = final_ranked[:top_k]
    top_scores   = [rrf[i] for i in top_indices]
    return top_indices, top_scores


def relevance_score(query: str, doc: str) -> float:
    """
    Simple cosine similarity between query and document embeddings.
    Used as relevance gate in CRAG.
    """
    q_emb = embedder.encode([query],  convert_to_tensor=True)
    d_emb = embedder.encode([doc[:500]], convert_to_tensor=True)
    score = util.cos_sim(q_emb, d_emb).item()
    return score


def retrieve_self_corrective(query: str, top_k: int = 3, threshold: float = 0.70):
    """
    Strategy 3: CRAG-style self-corrective retrieval.

    Step 1 — retrieve vanilla top-k
    Step 2 — score each doc for relevance
    Step 3 — if ALL docs below threshold → expand search to top-6 and re-score
    Step 4 — filter out low-relevance docs; use what's left
    Step 5 — log correction events for the article's narrative
    """
    correction_triggered = False
    note = ""

    # Initial retrieval
    indices, scores = retrieve_vanilla(query, top_k=top_k)

    # Score each retrieved doc
    rel_scores = [(idx, relevance_score(query, documents[idx])) for idx in indices]

    # Check if ALL are below threshold
    max_rel = max(s for _, s in rel_scores)
    if max_rel < threshold:
        correction_triggered = True
        note = f"⚠ All docs below threshold ({threshold}). Expanding search to top-6."
        # Expand and re-score
        indices, scores = retrieve_vanilla(query, top_k=6)
        rel_scores = [(idx, relevance_score(query, documents[idx])) for idx in indices]

    # Filter: keep docs above threshold/2 (soft floor)
    filtered = [(idx, rs) for idx, rs in rel_scores if rs >= threshold * 0.6]

    if not filtered:
        # Last resort: just use the best one
        filtered = [max(rel_scores, key=lambda x: x[1])]
        note += " | Last-resort: using best available doc."

    final_indices = [idx for idx, _ in filtered[:top_k]]
    final_scores  = [rs  for _, rs  in filtered[:top_k]]

    return final_indices, final_scores, correction_triggered, note


# ─────────────────────────────────────────────
#  6. TEST QUERIES
# ─────────────────────────────────────────────

TEST_QUERIES = [
    "What is RAG and how does it reduce hallucinations?",
    "How does FAISS handle vector similarity search?",
    "What is the difference between fine-tuning and prompt engineering?",
    "How do sentence embeddings represent text semantically?",
    "What is LangChain used for in LLM applications?",
    "How do transformer models process sequential data?",
    "What is a vector database and why is it used with LLMs?",
    "What did OpenAI release that changed NLP?",
    "How does Llama differ from GPT models?",
    "What makes large language models prone to hallucination?",
    "What are the key components of a transformer architecture?",
    "How does Hugging Face contribute to the open-source AI community?",
    "What are the advantages of using retrieval-augmented generation?",
    "Can prompt engineering replace fine-tuning entirely?",
    "How do attention mechanisms work in transformers?",
    "What is the role of the encoder in a transformer?",
    "How do vector databases perform indexing for fast retrieval?",
    "What are some common causes of hallucination in AI?",
    "What is the difference between natural language processing and understanding?",
    "How does a generative pre-trained transformer generate text?",
    "What metrics are used to evaluate sentence embeddings?",
    "How can you mitigate hallucinations in LLMs without retraining?",
    "What makes FAISS efficient for large scale similarity search?",
    "What is the difference between dense and sparse embeddings?",
    "How has OpenAI's GPT series evolved over time?",
    "What are the benefits of using LangChain's memory module?",
    "How does fine-tuning improve a model's performance on a specific task?",
    "What are the ethical concerns surrounding large language models?",
    "How do you implement a simple RAG pipeline?",
    "What is the difference between open-source models like Llama and closed models?"
]


# ─────────────────────────────────────────────
#  7. RUN EXPERIMENT
# ─────────────────────────────────────────────

print(Fore.CYAN + "\n" + "=" * 60)
print(f"  RUNNING EXPERIMENT — 3 Strategies × {len(TEST_QUERIES)} Queries")
print("=" * 60)

def retrieval_relevance(query, idx):
    return relevance_score(query, documents[idx])

results = []

for q_idx, query in enumerate(TEST_QUERIES):
    print(Fore.YELLOW + f"\n[Query {q_idx+1}/{len(TEST_QUERIES)}] {query}")
    row = {"query": query}

    # ── Strategy 1: Vanilla ──
    t0 = time.time()
    v_indices, v_scores = retrieve_vanilla(query)
    v_context = "\n\n".join([f"[{doc_titles[i]}]\n{documents[i][:400]}" for i in v_indices])
    v_answer  = generate_answer(query, v_context)
    v_time    = round(time.time() - t0, 2)
    v_top_doc = doc_titles[v_indices[0]] if v_indices else "N/A"

    row["vanilla_top_doc"] = v_top_doc
    row["vanilla_score"]   = round(float(v_scores[0]), 4)
    row["vanilla_time_s"]  = v_time
    row["vanilla_answer"]  = v_answer
    print(Fore.WHITE + f"  Vanilla  → {v_top_doc} (score={v_scores[0]:.4f}, {v_time}s)")

    time.sleep(0.5)  # Groq rate limit buffer

    # ── Strategy 2: Hybrid ──
    t0 = time.time()
    h_indices, h_scores = retrieve_hybrid(query)
    h_context = "\n\n".join([f"[{doc_titles[i]}]\n{documents[i][:400]}" for i in h_indices])
    h_answer  = generate_answer(query, h_context)
    h_time    = round(time.time() - t0, 2)
    h_top_doc = doc_titles[h_indices[0]] if h_indices else "N/A"

    row["hybrid_top_doc"] = h_top_doc
    row["hybrid_score"]   = round(float(h_scores[0]), 4)
    row["hybrid_time_s"]  = h_time
    row["hybrid_answer"]  = h_answer
    if h_indices:
        row["hybrid_relevance"] = retrieval_relevance(query, h_indices[0])
    else:
        row["hybrid_relevance"] = 0.0
    print(Fore.CYAN + f"  Hybrid   → {h_top_doc} (score={h_scores[0]:.4f}, {h_time}s)")

    time.sleep(0.5)

    # ── Strategy 3: Self-Corrective ──
    t0 = time.time()
    sc_indices, sc_scores, corrected, sc_note = retrieve_self_corrective(query)
    sc_context = "\n\n".join([f"[{doc_titles[i]}]\n{documents[i][:400]}" for i in sc_indices])
    sc_answer  = generate_answer(query, sc_context)
    sc_time    = round(time.time() - t0, 2)
    sc_top_doc = doc_titles[sc_indices[0]] if sc_indices else "N/A"

    row["selfcorr_top_doc"]   = sc_top_doc
    row["selfcorr_score"]     = round(float(sc_scores[0]), 4)
    row["selfcorr_time_s"]    = sc_time
    row["selfcorr_corrected"] = "YES ⚠" if corrected else "no"
    row["selfcorr_note"]      = sc_note
    row["selfcorr_answer"]    = sc_answer

    correction_flag = Fore.RED + "CORRECTED" if corrected else Fore.GREEN + "ok"
    print(Fore.MAGENTA + f"  SelfCorr → {sc_top_doc} (score={sc_scores[0]:.4f}, {sc_time}s) [{correction_flag}{Fore.MAGENTA}]")
    if sc_note:
        print(Fore.RED + f"    Note: {sc_note}")

    time.sleep(0.5)
    results.append(row)

print(Fore.GREEN + f"\n  All {len(TEST_QUERIES)} queries processed!")


# ─────────────────────────────────────────────
#  8. RESULTS TABLE (Screenshot this!)
# ─────────────────────────────────────────────

print(Fore.CYAN + "\n" + "=" * 60)
print("  RESULTS TABLE  ← TAKE SCREENSHOT HERE")
print("=" * 60)

df = pd.DataFrame(results)

# Summary table
summary = []
for row in results:
    q_short = row["query"][:45] + "..." if len(row["query"]) > 45 else row["query"]
    summary.append([
        q_short,
        row["vanilla_top_doc"][:25],
        row["hybrid_top_doc"][:25],
        row["selfcorr_top_doc"][:25],
        row["selfcorr_corrected"],
    ])

headers = ["Query", "Vanilla top doc", "Hybrid top doc", "SelfCorr top doc", "Corrected?"]
print(tabulate(summary, headers=headers, tablefmt="rounded_outline"))


# ─────────────────────────────────────────────
#  9. STRATEGY COMPARISON METRICS
# ─────────────────────────────────────────────

print(Fore.CYAN + "\n" + "=" * 60)
print("  STRATEGY COMPARISON METRICS  ← SCREENSHOT THIS TOO")
print("=" * 60)

# Agreement: does Vanilla & Hybrid return same top doc?
v_h_agree = sum(1 for r in results if r["vanilla_top_doc"] == r["hybrid_top_doc"])
v_sc_agree = sum(1 for r in results if r["vanilla_top_doc"] == r["selfcorr_top_doc"])
corrections = sum(1 for r in results if r["selfcorr_corrected"] == "YES ⚠")

avg_v_score  = np.mean([r["vanilla_score"]  for r in results])
avg_h_relevance = np.mean([r["hybrid_relevance"] for r in results])
avg_sc_score = np.mean([r["selfcorr_score"] for r in results])

avg_v_time   = np.mean([r["vanilla_time_s"]  for r in results])
avg_h_time   = np.mean([r["hybrid_time_s"]   for r in results])
avg_sc_time  = np.mean([r["selfcorr_time_s"] for r in results])

metrics = [
    ["Vanilla RAG",        f"{avg_v_score:.4f}",  f"{avg_v_time:.2f}s",  "N/A"],
    ["Hybrid RAG (RRF)",   f"{avg_h_relevance:.4f}",  f"{avg_h_time:.2f}s",  "N/A"],
    ["Self-Corrective",    f"{avg_sc_score:.4f}", f"{avg_sc_time:.2f}s", f"{corrections}/{len(results)} triggered"],
]
print(tabulate(metrics,
               headers=["Strategy", "Avg relevance score", "Avg latency", "Corrections"],
               tablefmt="rounded_outline"))

print(f"\n  Vanilla vs Hybrid — same top doc: {v_h_agree}/{len(results)} queries")
print(f"  Vanilla vs SelfCorr — same top doc: {v_sc_agree}/{len(results)} queries")
print(f"  Self-corrective triggered: {corrections}/{len(results)} times")


# ─────────────────────────────────────────────
#  10. SAMPLE ANSWER COMPARISON (use in article)
# ─────────────────────────────────────────────

print(Fore.CYAN + "\n" + "=" * 60)
print("  SAMPLE ANSWER COMPARISON — Query 1  ← SCREENSHOT")
print("=" * 60)

r = results[0]
print(Fore.YELLOW + f"\nQuery: {r['query']}\n")
print(Fore.WHITE  + f"[Vanilla RAG]\n{r['vanilla_answer']}\n")
print(Fore.CYAN   + f"[Hybrid RAG]\n{r['hybrid_answer']}\n")
print(Fore.MAGENTA+ f"[Self-Corrective RAG]\n{r['selfcorr_answer']}\n")


# ─────────────────────────────────────────────
#  11. SAVE TO CSV (attach in article)
# ─────────────────────────────────────────────

csv_path = "rag_experiment_results.csv"
df.to_csv(csv_path, index=False)
print(Fore.GREEN + f"\n  Results saved to {csv_path}")


# ─────────────────────────────────────────────
#  12. YOUR ARTICLE OBSERVATIONS TEMPLATE
# ─────────────────────────────────────────────

print(Fore.CYAN + "\n" + "=" * 60)
print("  ARTICLE WRITING NOTES — fill these after running")
print("=" * 60)
print(f"""
Copy these observations into your TDS draft:

1. VANILLA RAG
   - Where it worked well:   _________________________
   - Where it failed:        _________________________
   - Surprising finding:     _________________________

2. HYBRID RAG (BM25 + Semantic + RRF)
   - Improvement over vanilla: _____________________
   - Query type it helped most: ____________________
   - Where it still struggled: _____________________

3. SELF-CORRECTIVE RAG
   - How many corrections triggered: __/{{len(TEST_QUERIES)}}
   - Did correction improve results: YES / NO / MIXED
   - Latency trade-off:              _________________

4. MY VERDICT (this is your TDS conclusion):
   "For [use case], I would use [strategy] because ___"
""")

print(Fore.CYAN + "=" * 60)
print("  Experiment complete! Now write your TDS article.")
print("  Remember: YOUR observations = what makes it original.")
print("=" * 60 + Style.RESET_ALL)
