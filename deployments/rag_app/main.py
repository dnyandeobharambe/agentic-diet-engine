import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

# --- SECURITY ---
API_KEY_NAME = "X-API-KEY"
MASTER_KEY = os.getenv("INTERNAL_API_KEY") # This is our "Manual Gate"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(header_value: str = Security(api_key_header)):
    # If key is missing from environment or doesn't match, block access
    if not MASTER_KEY or header_value != MASTER_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, 
            detail="System Locked: Maintenance Mode or Invalid Key"
        )
    return header_value

app = FastAPI(title="Sovereign Nutrition API")

class UserQuery(BaseModel):
    text: str

class RAGResponse(BaseModel):
    answer: str
    tokens: int
    faithfulness: str
    sources: List[str]

@app.post("/ask", response_model=RAGResponse)
async def process_question(query: UserQuery, authenticated: str = Depends(get_api_key)):
    # Import logic here to ensure it only runs when authenticated
    from advanced_rag_ops_langsmith import run_with_cache_and_eval
    
    answer, tokens, verdict, docs = run_with_cache_and_eval(query.text)
    context_snippets = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs]
    
    return RAGResponse(answer=answer, tokens=tokens, faithfulness=verdict, sources=context_snippets)
