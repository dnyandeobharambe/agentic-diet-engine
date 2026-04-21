import os
from dotenv import load_dotenv

# 1. THE IMPORTS (Updated for 2026 compatibility)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# --- 2. THE INFRASTRUCTURE (Loading, not Building) ---
CHROMA_PATH = "./chroma_db"

# Ensure we use the exact model string from your dashboard
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

if os.path.exists(CHROMA_PATH):
    print("Loading existing vector store from disk... (Fast Mode)")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
else:
    print(" Error: ./chroma_db not found. Please run your reindex script first.")
    exit()
# semantic search with k=3 to get the top 3 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 3. THE TOOL (The 'Eyes' of the Chef) ---
@tool
def search_cookbook(query: str):
    """Consult the cookbook for recipes, ingredients, and protein content."""
    # Use 'invoke' to fetch the most relevant 3 chunks
    docs = retriever.invoke(query)
    formatted_context = "\n\n".join([f"Source: {d.metadata.get('source', 'Book')}\nContent: {d.page_content}" for d in docs])
    return formatted_context

# --- 4. THE AGENT (The 'Brain' of the Chef) ---
# We use Gemini 2.5 Flash as per your dashboard quota
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [search_cookbook]

# Architect Tip: create_react_agent is the standard for autonomous tool-use
app = create_react_agent(llm, tools)

if __name__ == "__main__":
    print(" Chef is ready. Type your question or 'exit' to quit.")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        inputs = {"messages": [("user", user_input)]}
        
        # Stream the thought process so you can see the 'Tool Call' in action
        for chunk in app.stream(inputs, stream_mode="values"):
            # Print the last message in the thread (The Agent's final answer)
            message = chunk["messages"][-1]
            if hasattr(message, 'content') and message.content:
                message.pretty_print()