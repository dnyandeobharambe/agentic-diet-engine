import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent # Keep this import

load_dotenv()

@tool
def get_fridge_inventory() -> str:
    """Returns a list of ingredients currently in the user's fridge."""
    return "3 Eggs, 1 Spinach bunch, Feta cheese, 2 Tomatoes."

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [get_fridge_inventory]

# CHANGE: 'state_modifier' is now 'prompt' in the latest prebuilt agent
chef_agent = create_react_agent(
    llm, 
    tools, 
    prompt="You are a Michelin-star chef. Always check the fridge inventory before suggesting a meal."
)

if __name__ == "__main__":
    print(" Running Agent...")
    
    # Standard input for ReAct agents
    inputs = {"messages": [HumanMessage(content="What high-protein snack can I make?")]}
    
    # Streaming the response to see the 'thought' process
    for event in chef_agent.stream(inputs, stream_mode="values"):
        message = event["messages"][-1]
        if hasattr(message, "content") and message.content:
            print(f"\n[CHEF]: {message.content}")