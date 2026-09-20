from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List

from app.model_utils import load_models, predict_pipeline

class ReviewRequest(BaseModel):
    text: str

class ReviewResponse(BaseModel):
    sentiment: str
    confidence: float
    aspects: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(
    title="E-Commerce Review Categorizer API",
    description="FastAPI Backend for Sentiment Analysis and Aspect Classification",
    version="1.0.0",
    lifespan=lifespan
)

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
        result = predict_pipeline(request.text)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
