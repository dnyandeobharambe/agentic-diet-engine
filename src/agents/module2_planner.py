import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class WeddingPlan(BaseModel):
    date: str = Field(..., description="The date of the wedding")
    location: str = Field(..., description="The location of the wedding")
    number_of_guests: int = Field(..., description="The number of guests attending the wedding")
    budget: float = Field(..., description="The budget for the wedding in USD")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
planner_llm = llm.with_structured_output(WeddingPlan)

query = "Plan a beach wedding for 50 people with a $15,000 budget at Las Vegas on April 20,2026."
result = planner_llm.invoke(query)

print(f"Plan created: {result.location} | Budget: ${result.budget} | Guests: {result.number_of_guests} | Date: {result.date}")
