"""FastAPI REST API server for FakeLenseV2"""

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging

from code.inference import InferenceEngine
from code.utils.config import get_default_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FakeLenseV2 API",
    description="AI-Powered Fake News Detection System",
    version="2.0.0"
)

# Global inference engine
inference_engine: Optional[InferenceEngine] = None


class PredictionRequest(BaseModel):
    """Request model for prediction"""
    text: str = Field(..., description="Article text content", min_length=10)
    source: Optional[str] = Field(None, description="News source name")
    social_reactions: Optional[float] = Field(0, description="Number of social media reactions", ge=0)


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: int = Field(..., description="Predicted class (0: Fake, 1: Suspicious, 2: Real)")
    label: str = Field(..., description="Human-readable label")
    confidence: Optional[float] = Field(None, description="Prediction confidence (if available)")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup"""
    global inference_engine
    try:
        config = get_default_config()
        model_path = config.get("model_save_path", "./models/best_model.pth")
        inference_engine = InferenceEngine(model_path, config)
        logger.info("Inference engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load inference engine: {e}")
        # Continue startup even if model fails to load


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info"""
    return {
        "status": "online",
        "model_loaded": inference_engine is not None,
        "version": "2.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if inference_engine is not None else "unhealthy",
        "model_loaded": inference_engine is not None,
        "version": "2.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict if a news article is fake, suspicious, or real.

    Args:
        request: PredictionRequest containing article details

    Returns:
        PredictionResponse with prediction and label
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Make prediction
        prediction = inference_engine.predict(
            text=request.text,
            source=request.source or "Unknown",
            social_reactions=request.social_reactions or 0
        )

        # Map prediction to label
        label_map = {0: "Fake News", 1: "Suspicious News", 2: "Real News"}
        label = label_map.get(prediction, "Unknown")

        return {
            "prediction": prediction,
            "label": label,
            "confidence": None  # Can be enhanced to return actual confidence
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(articles: list[PredictionRequest]):
    """
    Predict multiple articles at once.

    Args:
        articles: List of PredictionRequest objects

    Returns:
        List of predictions
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(articles) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 articles per batch")

    try:
        results = []
        for article in articles:
            prediction = inference_engine.predict(
                text=article.text,
                source=article.source or "Unknown",
                social_reactions=article.social_reactions or 0
            )
            label_map = {0: "Fake News", 1: "Suspicious News", 2: "Real News"}
            results.append({
                "prediction": prediction,
                "label": label_map.get(prediction, "Unknown"),
                "confidence": None
            })
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


def main():
    """Run the API server"""
    uvicorn.run(
        "code.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
