import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "./chroma_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 1. Load and Split as before
loader = PyPDFLoader("data/cookbook.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_documents(docs)
print(f" Total chunks to index: {len(splits)}")

# 2. Manual Batching (The 100 RPM Bypass)
# We initialize an empty vector store first
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

batch_size = 50  # Stay well under the 100 limit
for i in range(0, len(splits), batch_size):
    batch = splits[i : i + batch_size]
    print(f" Indexing batch {i//batch_size + 1} ({len(batch)} chunks)...")
    
    vectorstore.add_documents(batch)
    
    # Architect's Cooldown: Wait 60 seconds after 2 batches (100 requests)
    if (i + batch_size) < len(splits):
        print(" Rate limit cooldown... waiting 65 seconds.")
        time.sleep(65)

print(f" Final verification: {vectorstore._collection.count()} items in DB.")