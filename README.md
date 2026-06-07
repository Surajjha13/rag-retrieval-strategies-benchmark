# RAG Retrieval Strategies Benchmark
 
> **Companion code for the article:**  
> [I Built a Self-Corrective RAG System. It Corrected Itself 70% of the Time. Here's What I Did Wrong.](YOUR_TDS_ARTICLE_LINK_HERE)  
> *Published on Towards Data Science*
 
---
 
## What This Repo Is About
 
This is a single-file experiment that compares three RAG retrieval strategies on the same dataset, same 30 queries, and same LLM.
 
| Strategy | Description |
|---|---|
| **Vanilla RAG** | Pure dense vector search using FAISS |
| **Hybrid RAG** | BM25 + Semantic search fused via RRF |
| **Self-Corrective RAG** | CRAG-style relevance gating |
 
---
 
## Quickstart
 
### 1. Install dependencies
 
```bash
pip install langchain-groq langchain-community faiss-cpu sentence-transformers rank-bm25 wikipedia-api pandas tabulate colorama
```
 
### 2. Set your Groq API key
 
Get a free key at [console.groq.com](https://console.groq.com)
 
```bash
export GROQ_API_KEY="your_key_here"
```
 
Or in Google Colab:
```python
import os
os.environ["GROQ_API_KEY"] = "your_key_here"
```
 
### 3. Run
 
```bash
python experiment.py
```
 
**Runtime:** ~10–15 minutes. No GPU needed. Total cost: $0.
 
---
 
## Try the Threshold Experiment
 
Change this one line in `experiment.py` and re-run:
 
```python
threshold: float = 0.55  # Try: 0.45, 0.55, 0.65, 0.75
```
 
Watch how the correction rate changes across runs. That is the core lesson of the article.
 
---
 
## Author
 
**Suraj Kumar Jha**  
- 📝 [Towards Data Science](YOUR_TDS_ARTICLE_LINK_HERE)
- 💼 [LinkedIn](https://www.linkedin.com/in/suraj-jha1/)
---
 
*If this helped you, give the repo a ⭐*
