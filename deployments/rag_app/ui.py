import streamlit as st
import requests
import os
from dotenv import load_dotenv

# 1. Load local environment variables (Only for local dev)
load_dotenv()

# --- CONFIGURATION ---
st.set_page_config(page_title="Nutrition Architect AI", layout="wide", page_icon="🥗")
BACKEND_URL = "http://localhost:8000/ask" # Locally, this is our FastAPI address

# --- 2. SYSTEM KILL-SWITCH (The Manual Gate) ---
# This variable must be set in your .env or Hugging Face Secrets
MASTER_KEY = os.getenv("INTERNAL_API_KEY")

# If the key is missing entirely, show Maintenance Mode
if not MASTER_KEY or MASTER_KEY.strip() == "":
    st.title(" Nutrition Architect AI")
    st.header("System Status: Offline")
    st.warning("This demo is currently in Maintenance Mode to protect API credits.")
    st.info("The backend is powered down. For a scheduled demonstration, please contact the Architect to activate the environment.")
    st.stop()

# --- 3. AUTHENTICATION SESSION STATE ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. LOGIN INTERFACE ---
if not st.session_state.authenticated:
    st.title("Portfolio Access Control")
    st.write("Welcome. This is a private Agentic AI demo. Please enter the access key provided in the project documentation.")
    
    user_input_key = st.text_input("Access Key", type="password", placeholder="Enter your demo passcode...")
    
    if st.button("Unlock System"):
        if user_input_key == MASTER_KEY:
            st.session_state.authenticated = True
            st.success("Access Granted. Initializing RAG Engine...")
            st.rerun() # Refresh to show the chat interface
        else:
            st.error("Invalid access key. Please check your credentials.")
    st.stop()

# --- 5. THE ACTIVE CHAT INTERFACE ---
# This part of the code is only reached if authenticated is True
st.title("Nutrition RAG Assistant")
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a nutrition or meal planning question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the FastAPI Backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*Architecting response...*")
        
        try:
            # We pass the MASTER_KEY as the X-API-KEY header to talk to our FastAPI
            headers = {"X-API-KEY": MASTER_KEY}
            response = requests.post(
                BACKEND_URL, 
                json={"text": prompt}, 
                headers=headers,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                full_response = data.get("answer", "No response received.")
                
                # Format the response with metadata
                formatted_response = f"{full_response}\n\n---\n**Metadata:**"
                formatted_response += f"\n* 💡 Tokens Used: {data.get('tokens')}"
                formatted_response += f"\n* Faithfulness: {data.get('faithfulness')}"
                
                message_placeholder.markdown(formatted_response)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                message_placeholder.error(error_msg)
                
        except Exception as e:
            message_placeholder.error(f"Connection Error: {str(e)}")

# Sidebar for logout/status
with st.sidebar:
    st.header("System Status")
    st.success("Backend: Connected")
    st.info(f"Mode: Local / Sovereign")
    if st.button("Log Out / Lock System"):
        st.session_state.authenticated = False
        st.rerun()
