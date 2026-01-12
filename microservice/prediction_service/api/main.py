
# prediction_service/app/main.py
import sys
from pathlib import Path
from fastapi import FastAPI
# Ensure the project root is in the system path for module resolution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from src.schema import TitanicRequest
from src.predictor import predict

app = FastAPI(title="Prediction Service", version="0.1.0")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Prediction Service",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "endpoint": "/predict"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "prediction_service"}

@app.post("/predict")
def predict_api(payload: TitanicRequest):
    """
    Make a survival prediction for a Titanic passenger.
    
    Returns:
        - prediction: 0 (did not survive) or 1 (survived)
        - probability: Probability of survival (if available)
    """
    # use model_dump() for Pydantic v2 compatibility
    return predict(payload.model_dump())
