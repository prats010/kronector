"""Startup script that loads .env and starts the FastAPI app."""
import os
import sys
from pathlib import Path

# Load .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    print(f"Loading {env_file}")
    from dotenv import load_dotenv
    load_dotenv(env_file)
    print(f"  KRONECTOR_MODEL_RUN_ID = {os.getenv('KRONECTOR_MODEL_RUN_ID')}")
else:
    print(f"Warning: {env_file} not found")

# Start uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    print(f"  Starting on http://127.0.0.1:{port}")
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=port,
        reload=False,
    )

