import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.retrieval.reranker import get_reranked_docs

load_dotenv()

def audit_tokens():
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    query = "give me protein recipes which can give me 25g protein"
    
    # --- PATH A: Standard Semantic Search (Top 5) ---
    # Simulating what happens without your optimizations
    retriever = get_hybrid_retriever() # Uses the ensemble
    semantic_docs = retriever.invoke(query)[:5] 
    semantic_text = "\n\n".join([d.page_content for d in semantic_docs])
    
    # --- PATH B: Your Hybrid + Reranker (Top 2) ---
    reranked_docs = get_reranked_docs(query)[:2]
    hybrid_text = "\n\n".join([d.page_content for d in reranked_docs])

    # --- CALCULATE TOKENS ---
    # Gemini has a built-in method for precise counting
    sem_tokens = llm.get_num_tokens(semantic_text)
    hyb_tokens = llm.get_num_tokens(hybrid_text)
    
    savings = sem_tokens - hyb_tokens
    cost_reduction = (savings / sem_tokens) * 100 if sem_tokens > 0 else 0

    print("="*40)
    print("ARCHITECT'S TOKEN AUDIT")
    print("="*40)
    print(f"Query: {query}")
    print("-"*40)
    print(f"Standard Semantic Tokens: {sem_tokens}")
    print(f"Your Hybrid + Rerank Tokens: {hyb_tokens}")
    print("-"*40)
    print(f"TOTAL TOKENS SAVED: {savings}")
    print(f"ESTIMATED COST REDUCTION: {cost_reduction:.2f}%")
    print("="*40)

if __name__ == "__main__":
    audit_tokens()
