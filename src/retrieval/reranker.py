import os
import numpy as np
from sentence_transformers import CrossEncoder
from src.retrieval.hybrid_retriever import get_hybrid_retriever

# --- 1. GLOBAL INITIALIZATION (Warm Start) ---
# This loads the model ONCE when the app starts, not on every query.
# Using the small but powerful MiniLM-L-6-v2 (approx 22M parameters)
RERANK_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

def get_reranked_docs(query):
    # 2. Get candidates (Recall Stage)
    retriever = get_hybrid_retriever()
    # Principal Tip: Ensure your hybrid retriever is limited to top 15-20 docs
    docs = retriever.invoke(query) 
    
    if not docs:
        return []

    # 3. Aggressive Pruning for 2-vCPU Performance
    # We only rerank the top 10 candidates. Reranking 50 on a free tier is suicide.
    candidate_docs = docs[:10] 
    
    # 4. Re-rank (Precision Stage)
    # The model is already warm in memory, so predict() starts instantly.
    pairs = [[query, doc.page_content] for doc in candidate_docs]
    scores = RERANK_MODEL.predict(pairs)
    
    # 5. Sort and Return
    reranked_indices = np.argsort(scores)[::-1]
    
    # We return the re-ordered top docs
    return [candidate_docs[i] for i in reranked_indices]
