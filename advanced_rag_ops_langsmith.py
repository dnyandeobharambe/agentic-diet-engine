import os
from dotenv import load_dotenv
from langsmith import traceable  # CRITICAL: For observability
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Importing your logic from other files
from src.retrieval.reranker import get_reranked_docs
from src.agents.final_agent import generate_answer

load_dotenv()

# --- 1. SET UP MODELS & CACHE ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
cache_db = Chroma(
    collection_name="llm_semantic_cache",
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)

# --- 2. TRACEABLE SUB-COMPONENTS ---
# We wrap these so they show up as separate bars in the LangSmith UI
@traceable(name="Document Retrieval & Rerank", run_type="retriever")
def traced_retrieval(query):
    return get_reranked_docs(query)

@traceable(name="LLM Answer Generation", run_type="llm")
def traced_generation(query, docs):
    return generate_answer(query, docs)

@traceable(name="Faithfulness Evaluation", run_type="chain")
def traced_evaluation(llm, context, answer):
    eval_prompt = f"Context: {context}\nAnswer: {answer}\nIs this answer supported? YES/NO"
    response = llm.invoke(eval_prompt)
    
    # --- GEMINI 3 CONTENT EXTRACTION ---
    # The response.content is now a list like: [{'type': 'text', 'text': 'YES', 'extras': {...}}]
    if isinstance(response.content, list):
        # Extract the 'text' key from the first block that has it
        content = next((block['text'] for block in response.content if 'text' in block), "")
    else:
        # Fallback for older models/strings
        content = str(response.content)

    return content.strip().upper() # Force to uppercase for reliable YES/NO check

# --- 3. MAIN TRACED PIPELINE ---
@traceable(name="Nutrition RAG Pipeline", run_type="chain")
def run_with_cache_and_eval(query):
    # Initialize defaults to prevent "got 0" errors
    res, tokens, verd, ctx = "No response", 0, "FAIL", "No context"
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
        
        # 1. Check Cache
        cache_results = cache_db.similarity_search_with_relevance_scores(query, k=1)
        if cache_results and cache_results[0][1] > 0.92:
            metadata = cache_results[0][0].metadata
            return metadata.get("response"), 0, "YES (Cached)", metadata.get("context", "No cached context")

        # 2. Execution Flow
        final_docs = traced_retrieval(query)
        if not final_docs:
            return "No documents found", 0, "NO_DATA", "Context empty"
            
        ctx = "\n\n".join([d.page_content for d in final_docs])
        res = traced_generation(query, final_docs)
        tokens = llm.get_num_tokens(ctx + query)
        
        # 3. Audit Flow
        verd = traced_evaluation(llm, ctx, res)

        # 4. Persistence
        cache_db.add_texts(
            texts=[query], 
            metadatas=[{"response": res, "context": ctx}]
        )
        return res, tokens, verd, ctx

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"--- INTERNAL PIPELINE CRASH ---")
        print(error_details)
        print(f"-------------------------------")
        
        # Return the SPECIFIC traceback so the Judge (and you) can see it
        return f"CRASH in pipeline: {str(e)}", 0, "ERROR", "No context"

if __name__ == "__main__":
    q = "give me recipes with > 20g protein and < 500 calories"
    ans, tokens, verdict, context = run_with_cache_and_eval(q)
    
    print("\n" + "="*50)
    print("ARCHITECT'S OPS REPORT (Traced via LangSmith)")
    print("="*50)
    print(f"VERDICT: {verdict} | TOKENS: {tokens}")
    print(f"RESULT: {ans[:100]}...") # Printing first 100 chars
    print(f"CONTEXT: {context[:100]}...") # Printing first 100 chars
    print("="*50)
