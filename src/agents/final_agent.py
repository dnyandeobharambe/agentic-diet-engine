import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Import your optimized logic from previous pieces
from src.retrieval.reranker import get_reranked_docs 

load_dotenv()

# --- 1. Define the Data Contract ---
class RecipeResponse(BaseModel):
    """Schema to force-strip metadata and signatures."""
    answer: str = Field(description="The clean text answer containing recipes and protein counts.")

def generate_answer(query, context_docs):
    # Initialize Gemini 3 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    
    # --- 2. Initialize the Parser ---
    parser = PydanticOutputParser(pydantic_object=RecipeResponse)
    
    # 3. Enhanced Template with Format Instructions
    template = """
    You are a professional Nutrition Assistant. Use the provided context to answer the question.
    
    {format_instructions}
    
    Context:
    {context}
    
    Question: {question}
    
    Instructions:
    - Only use the provided context.
    - Provide the Recipe Name and the specific Protein amount.
    """
    
    prompt = ChatPromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Prepare context
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    # --- 4. The Clean Chain ---
    # The parser at the end of the chain is what wipes out the 'extras' dictionary
    chain = prompt | llm | parser
    
    response = chain.invoke({"context": context_text, "question": query})
    
    # Return ONLY the string from our Pydantic object
    return response.answer

if __name__ == "__main__":
    query = "give me protein recipes which can give me 25g protein"
    
    print("Agent is processing...")
    final_docs = get_reranked_docs(query) 
    
    if final_docs:
        result = generate_answer(query, final_docs)
        print("\n--- AGENT ANSWER ---")
        print(result) # Now prints ONLY the validated text
    else:
        print("No relevant documents found.")
