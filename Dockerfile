# --- PRODUCTION DOCKERFILE (FAST BOOT) ---
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install dependencies (Cached Layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy everything (Uses .hfignore to skip .env and caches)
COPY . .

# 3. Setup Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 4. Runtime Execution
# We skip the heavy ingestion script and go straight to the API and UI
CMD (python -m uvicorn deployments.rag_app.main:app --host 0.0.0.0 --port 8000 & \
     streamlit run deployments/rag_app/ui.py --server.port 7860 --server.address 0.0.0.0)
     