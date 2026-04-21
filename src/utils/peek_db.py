import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# 1. Initialize with explicit logging
print(" Initializing Embedding Model...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

CHROMA_PATH = "./chroma_db"

if os.path.exists(CHROMA_PATH):
    print(f" Found {CHROMA_PATH}. Connecting...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    
    # 2. Test a very simple query to minimize token usage
    test_query = "protein"
    print(f" Sending query '{test_query}' to Google API for embedding...")
    
    start_time = time.time()
    try:
        # We use search_with_score to see the mathematical distance (the "Hard STEM" metric)
        results = vectorstore.similarity_search_with_score(test_query, k=1)
        elapsed = time.time() - start_time
        
        if results:
            doc, score = results[0]
            print(f"✅ Success! Latency: {elapsed:.2f}s | Distance Score: {score:.4f}")
            print(f"\n--- Top Result Preview ---\n{doc.page_content[:300]}...")
        else:
            print(" API returned successfully but found 0 matches in your DB.")
            
    except Exception as e:
        print(f" API Call Failed after {time.time() - start_time:.2f}s")
        print(f"Error Details: {e}")
else:
    print(" Directory './chroma_db' does not exist.")