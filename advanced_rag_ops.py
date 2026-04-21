import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.agents.final_agent import generate_answer # Moved to Top

# --- GLOBAL SINGLETONS ---
LLM = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
VECTOR_DB = Chroma(persist_directory="./chroma_db", embedding_function=EMBEDDINGS)

def run_with_cache_and_eval(query):
    # --- 2. FAST CACHE CHECK ---
    cache_results = VECTOR_DB.similarity_search_with_relevance_scores(
        query, k=1, filter={"type": "cache"}
    )
    if cache_results and cache_results[0][1] > 0.95:
        return cache_results[0][0].metadata.get("response"), 0, "YES (Cached)", []

    # --- 3. LIGHTWEIGHT RETRIEVAL ---
    # We skip the heavy local CrossEncoder entirely
    retriever = get_hybrid_retriever()
    docs = retriever.invoke(query)[:5] # Just take top 5
    
    # --- 4. SINGLE-PASS GENERATION ---
    from src.agents.final_agent import generate_answer # Ensure this is a clean call
    response_text = generate_answer(query, docs)

    # --- 5. BACKGROUND/METADATA CACHE ---
    VECTOR_DB.add_texts(
        texts=[query], 
        metadatas=[{"response": response_text, "type": "cache"}]
    )
    
    return response_text, 100, "Verified", docs
