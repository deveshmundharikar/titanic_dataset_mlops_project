"""FastAPI application for training service."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add parent directory to path to allow imports from src and pipeline
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from pipeline.train_stage import run_training_pipeline as main
except ImportError:
    main = None

app = FastAPI(
    title="Training Service",
    description="API for managing model training pipeline",
    version="0.1.0"
)

def load_config() -> Dict[str, Any]:
    """Load configuration from yaml file with fallback."""
    config_path = BASE_DIR / "config" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    return {"training": {"model_name": "default_model"}}

CONFIG = load_config()

class TrainingRequest(BaseModel):
    dataset_path: str
    model_name: str = Field(default_factory=lambda: CONFIG.get("training", {}).get("model_name", "default_model"))

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    service: str = "training_service"
    version: str = "0.1.0"

@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Training Service",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }

@app.post("/train")
async def trigger_training(request: TrainingRequest):
    """
    Train a new model with the provided dataset.
    """
    if main is None:
        raise HTTPException(
            status_code=500, 
            detail="Training pipeline module not found. Check pipeline/train_stage.py"
        )

    try:
        # Run the training pipeline
        # Note: run_training_pipeline returns (model, metrics)
        _, metrics = main(dataset_path=request.dataset_path)
        
        # Extract meaningful metrics to return
        test_metrics = metrics.get('test_metrics', {}) if isinstance(metrics, dict) else {}
        
        return {
            "status": "success",
            "message": "Model training completed successfully",
            "dataset_path": request.dataset_path,
            "model_name": request.model_name,
            "metrics": test_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

