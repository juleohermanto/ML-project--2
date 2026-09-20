from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List

# Import our custom utilities
from app.model_utils import load_models, predict_pipeline

# --------------------------
# 1. Schema Definitions
# --------------------------
class ReviewRequest(BaseModel):
    text: str

class ReviewResponse(BaseModel):
    sentiment: str
    confidence: float
    aspects: List[str]

# --------------------------
# 2. Application Setup
# --------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs right before the server starts accepting requests
    load_models()
    yield
    # We could put cleanup code here (if any) when the server shuts down

app = FastAPI(
    title="E-Commerce Review Categorizer API",
    description="FastAPI Backend for Sentiment Analysis and Aspect Classification",
    version="1.0.0",
    lifespan=lifespan
)

# --------------------------
# 3. API Endpoints
# --------------------------
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the NLP Inference API!",
        "docs_url": "/docs"
    }

@app.post("/predict", response_model=ReviewResponse)
def predict_review(request: ReviewRequest):
    """
    Takes a raw review string, cleans it, and returns the predicted sentiment and aspects.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    try:
        # Run inference using our loaded models
        result = predict_pipeline(request.text)
        return result
    except RuntimeError as e:
        # Happens if models aren't physically present on disk
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Catch-all for other inference bugs
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Trigger Reload
