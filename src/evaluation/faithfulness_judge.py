import os
import json
from google import genai

def grade_faithfulness(question, context, answer):
    # The new client automatically looks for GEMINI_API_KEY or GOOGLE_API_KEY
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    prompt = f"""
    You are an SRE Quality Auditor. 
    Context: {context}
    Question: {question}
    AI Answer: {answer}

    Verify if the AI Answer is strictly derived from the Context. 
    Return ONLY a JSON object: {{"score": float, "reason": "string"}}
    """
    
    try:
        # Modern 2026 syntax: client.models.generate_content
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=prompt
        )
        
        # New SDK returns a cleaner object, but we still handle potential markdown
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return data.get("score", 0.0), data.get("reason", "No reason provided")
    except Exception as e:
        return 0.0, f"Judge Error: {str(e)}"
    