"""
Vercel serverless function entry point for FastAPI backend
"""
import sys
import os
from pathlib import Path

# Add the project root and src directories to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
backend_path = project_root / "backend"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(backend_path))

try:
    # Import the FastAPI app from backend
    from backend.main import app

    # Export the app for Vercel
    handler = app
except ImportError as e:
    # Fallback minimal API if full backend can't be imported
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="AlgoTrading API - Minimal", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health_check():
        return {"status": "ok", "message": "Minimal API running on Vercel"}

    @app.get("/")
    async def root():
        return {"message": "AlgoTrading API is running on Vercel"}

    handler = app
