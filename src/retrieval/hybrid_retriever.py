import os
from dotenv import load_dotenv

# --- THE IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

load_dotenv()

def get_hybrid_retriever():
    CHROMA_PATH = "./chroma_db"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 1. Fetch data from Chroma
    raw_data = vectorstore.get()
    
    # --- THE GUARDRAIL ---
    if not raw_data['documents']:
        print("⚠️ ARCHITECT NOTE: Chroma is empty. Triggering PDF ingestion...")
        # Option A: Raise an error so you know you need to run your ingestion script
        raise ValueError("ChromaDB is empty. Please run your ingestion script (ingest.py) first.")
        
        # Option B: You could call your ingestion function here if you have one.
    
    # 2. Build Keyword Retriever only if we have docs
    docs = [
        Document(page_content=txt, metadata=meta) 
        for txt, meta in zip(raw_data['documents'], raw_data['metadatas'])
    ]

    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 5 

    # 3. Assemble
    return EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.7, 0.3]
    )

# --- THE TEST (Only runs if you execute THIS file directly) ---
if __name__ == "__main__":
    retriever = get_hybrid_retriever()
    query = "give me protein recipes which can give me 25g protein"
    
    print(f"\n Running Local Test Query: {query}")
    print("-" * 30)
    
    results = retriever.invoke(query)
    
    for i, doc in enumerate(results):
        source = doc.metadata.get('source', 'Unknown')
        content_snippet = doc.page_content[:150].replace('\n', ' ')
        print(f"[{i+1}] Source: {source} | Content: {content_snippet}...")
        