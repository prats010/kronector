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

COPY --chown=1000:1000 requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=1000:1000 agents/ ./agents/
COPY --chown=1000:1000 api/ ./api/
COPY --chown=1000:1000 ml/ ./ml/
COPY --chown=1000:1000 data/ ./data/
COPY --chown=1000:1000 scripts/ ./scripts/
COPY --chown=1000:1000 run_api.py run_all.py ./

# Copy pre-trained model artifacts and data
COPY --chown=1000:1000 data_output/ ./data_output/
COPY --chown=1000:1000 mlruns/ ./mlruns/

# HF Spaces expects port 7860
EXPOSE 7860

# Default model run ID
ENV KRONECTOR_MODEL_RUN_ID=8d8d20f14dce44d991c5fccfdc090a68
ENV MLFLOW_ALLOW_FILE_STORE=true

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
