# ============================================================================
# KRONECTOR — Hugging Face Spaces Dockerfile
# ============================================================================
# HF Spaces runs a single container, so we run FastAPI directly.
# HF exposes port 7860 by default.
#
# Deploy:
#   1. Create a Space on huggingface.co (SDK: Docker)
#   2. Push this repo to the Space
#   3. Set GROQ_API_KEY as a Space Secret
# ============================================================================

FROM python:3.11-slim

WORKDIR /app

# LightGBM requires libgomp (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agents/ ./agents/
COPY api/ ./api/
COPY ml/ ./ml/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY ui/ ./ui/
COPY run_api.py run_all.py ./

# Copy pre-trained model artifacts and data
COPY data_output/ ./data_output/
COPY mlruns/ ./mlruns/

# HF Spaces expects port 7860
EXPOSE 7860

# Default model run ID
ENV KRONECTOR_MODEL_RUN_ID=8d8d20f14dce44d991c5fccfdc090a68

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
