import os
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Memory is handled by passing a list of messages back and forth
chat_history = [
    HumanMessage(content="My name is Dnyan."),
    AIMessage(content="Nice to meet you, Dnyan! How can I assist with your emails?")
]

# The agent now 'remembers' the context
follow_up = "What is my name and can you draft a thank you note?"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = llm.invoke(chat_history + [HumanMessage(content=follow_up)])

print(response.content) 
# The agent will correctly identify your name from the history