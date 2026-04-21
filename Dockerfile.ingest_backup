FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy everything to /app
# In your HF repo, 'src' and 'deployments' will now be in '/app/src' and '/app/deployments'
COPY . .

# 3. Setup Environment
ENV PYTHONUNBUFFERED=1
# Tell Python that /app is the root for all imports (so 'from src... ' works)
ENV PYTHONPATH=/app

# 4. Runtime Execution
# Path updated to match your HF repo structure exactly
CMD python src/utils/Safe-Batching.py && \
    (python -m uvicorn deployments.rag_app.main:app --host 0.0.0.0 --port 8000 & \
     streamlit run deployments/rag_app/ui.py --server.port 7860 --server.address 0.0.0.0)
     