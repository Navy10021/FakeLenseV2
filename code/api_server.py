"""FastAPI REST API server for FakeLenseV2"""

import time
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

from code.inference import InferenceEngine
from code.utils.config import get_default_config
from code.utils.logging_utils import StructuredLogger, RequestIDMiddleware
from code.utils.validators import ValidationError

# Setup structured logging
logger = StructuredLogger("fakelense-api")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# Initialize FastAPI app
app = FastAPI(
    title="FakeLenseV2 API",
    description="AI-Powered Fake News Detection System with Confidence Scoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine: Optional[InferenceEngine] = None


class PredictionRequest(BaseModel):
    """Request model for prediction"""

    text: str = Field(
        ..., description="Article text content", min_length=10, max_length=10000
    )
    source: Optional[str] = Field(None, description="News source name")
    social_reactions: Optional[float] = Field(
        0, description="Number of social media reactions", ge=0
    )


class PredictionResponse(BaseModel):
    """Response model for prediction"""

    prediction: int = Field(
        ..., description="Predicted class (0: Fake, 1: Suspicious, 2: Real)"
    )
    label: str = Field(..., description="Human-readable label")
    confidence: float = Field(
        ..., description="Prediction confidence score (0-1)", ge=0, le=1
    )
    all_probabilities: Dict[str, float] = Field(
        ..., description="Probabilities for all classes"
    )
    request_id: str = Field(..., description="Unique request identifier")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""

    articles: List[PredictionRequest] = Field(
        ..., description="List of articles to predict", max_items=100
    )


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str
    model_loaded: bool
    version: str
    uptime_seconds: Optional[float] = None


# Store startup time for uptime calculation
startup_time = time.time()


# Request tracking middleware
@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    """Add request ID to all requests and log API calls"""
    request_id = RequestIDMiddleware.generate_request_id()
    request.state.request_id = request_id

    start_time = time.time()

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Log API request
        logger.log_api_request(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms,
            client_ip=request.client.host if request.client else None,
        )

        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.log_api_request(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            status_code=500,
            duration_ms=duration_ms,
            client_ip=request.client.host if request.client else None,
            error=str(e),
        )
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup"""
    global inference_engine
    start_time = time.time()

    try:
        config = get_default_config()
        model_path = config.get("model_save_path", "./models/best_model.pth")
        inference_engine = InferenceEngine(model_path, config)

        duration_ms = (time.time() - start_time) * 1000
        logger.log_model_load(
            model_path=model_path, duration_ms=duration_ms, success=True
        )
        logger.info("Inference engine loaded successfully", model_path=model_path)
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.log_model_load(
            model_path="./models/best_model.pth",
            duration_ms=duration_ms,
            success=False,
            error=str(e),
        )
        logger.error("Failed to load inference engine", error=str(e))
        # Continue startup even if model fails to load


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info"""
    uptime = time.time() - startup_time
    return {
        "status": "online",
        "model_loaded": inference_engine is not None,
        "version": "2.0.0",
        "uptime_seconds": uptime,
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    uptime = time.time() - startup_time
    return {
        "status": "healthy" if inference_engine is not None else "unhealthy",
        "model_loaded": inference_engine is not None,
        "version": "2.0.0",
        "uptime_seconds": uptime,
    }


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("100/minute")
async def predict(req: Request, request: PredictionRequest):
    """
    Predict if a news article is fake, suspicious, or real with confidence scores.

    Args:
        request: PredictionRequest containing article details

    Returns:
        PredictionResponse with prediction, label, confidence, and probabilities

    Rate Limit:
        100 requests per minute per IP
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = req.state.request_id
    start_time = time.time()

    try:
        # Make prediction with confidence
        result = inference_engine.predict_with_confidence(
            text=request.text,
            source=request.source or "Unknown",
            social_reactions=request.social_reactions or 0,
        )

        duration_ms = (time.time() - start_time) * 1000

        # Log prediction
        logger.log_prediction(
            request_id=request_id,
            text_length=len(request.text),
            source=request.source,
            prediction=result["prediction"],
            label=result["label"],
            confidence=result["confidence"],
            duration_ms=duration_ms,
        )

        return {
            "prediction": result["prediction"],
            "label": result["label"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"],
            "request_id": request_id,
        }
    except ValidationError as e:
        # Validation errors return 400 Bad Request
        duration_ms = (time.time() - start_time) * 1000
        logger.log_prediction(
            request_id=request_id,
            text_length=len(request.text) if hasattr(request, "text") else 0,
            source=request.source,
            prediction=-1,
            label="ValidationError",
            confidence=0.0,
            duration_ms=duration_ms,
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except FileNotFoundError as e:
        # Model file not found
        duration_ms = (time.time() - start_time) * 1000
        logger.error("Model file not found", error=str(e), request_id=request_id)
        raise HTTPException(status_code=503, detail="Model files not accessible")
    except RuntimeError as e:
        # Runtime errors (e.g., CUDA errors, model inference issues)
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "Runtime error during prediction", error=str(e), request_id=request_id
        )
        raise HTTPException(status_code=500, detail="Model inference error occurred")
    except Exception as e:
        # Catch-all for unexpected errors
        duration_ms = (time.time() - start_time) * 1000
        logger.log_prediction(
            request_id=request_id,
            text_length=len(request.text),
            source=request.source,
            prediction=-1,
            label="Error",
            confidence=0.0,
            duration_ms=duration_ms,
            error=str(e),
        )
        logger.error(
            "Unexpected error during prediction", error=str(e), request_id=request_id
        )
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during prediction"
        )


@app.post("/batch_predict")
@limiter.limit("20/minute")
async def batch_predict(req: Request, batch_request: BatchPredictionRequest):
    """
    Predict multiple articles at once with confidence scores.

    Args:
        batch_request: BatchPredictionRequest containing list of articles

    Returns:
        List of prediction results with confidence scores

    Rate Limit:
        20 requests per minute per IP
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = req.state.request_id
    articles = batch_request.articles
    batch_size = len(articles)

    if batch_size > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 articles per batch")

    start_time = time.time()
    success_count = 0
    error_count = 0
    results = []

    try:
        for article in articles:
            try:
                result = inference_engine.predict_with_confidence(
                    text=article.text,
                    source=article.source or "Unknown",
                    social_reactions=article.social_reactions or 0,
                )
                results.append(
                    {
                        "prediction": result["prediction"],
                        "label": result["label"],
                        "confidence": result["confidence"],
                        "all_probabilities": result["all_probabilities"],
                    }
                )
                success_count += 1
            except ValidationError as e:
                results.append(
                    {
                        "prediction": -1,
                        "label": "ValidationError",
                        "confidence": 0.0,
                        "all_probabilities": {
                            "fake": 0.0,
                            "suspicious": 0.0,
                            "real": 0.0,
                        },
                        "error": f"Validation error: {str(e)}",
                    }
                )
                error_count += 1
            except RuntimeError as e:
                results.append(
                    {
                        "prediction": -1,
                        "label": "RuntimeError",
                        "confidence": 0.0,
                        "all_probabilities": {
                            "fake": 0.0,
                            "suspicious": 0.0,
                            "real": 0.0,
                        },
                        "error": "Model inference error",
                    }
                )
                error_count += 1
                logger.error(
                    "Runtime error in batch prediction",
                    error=str(e),
                    request_id=request_id,
                )
            except Exception as e:
                results.append(
                    {
                        "prediction": -1,
                        "label": "Error",
                        "confidence": 0.0,
                        "all_probabilities": {
                            "fake": 0.0,
                            "suspicious": 0.0,
                            "real": 0.0,
                        },
                        "error": "Unexpected error occurred",
                    }
                )
                error_count += 1
                logger.error(
                    "Unexpected error in batch prediction",
                    error=str(e),
                    request_id=request_id,
                )

        duration_ms = (time.time() - start_time) * 1000

        # Log batch prediction
        logger.log_batch_prediction(
            request_id=request_id,
            batch_size=batch_size,
            duration_ms=duration_ms,
            success_count=success_count,
            error_count=error_count,
        )

        return {
            "predictions": results,
            "request_id": request_id,
            "total": batch_size,
            "success": success_count,
            "errors": error_count,
        }
    except ValidationError as e:
        # Validation error at batch level
        duration_ms = (time.time() - start_time) * 1000
        logger.log_batch_prediction(
            request_id=request_id,
            batch_size=batch_size,
            duration_ms=duration_ms,
            success_count=0,
            error_count=batch_size,
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=f"Batch validation error: {str(e)}")
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.log_batch_prediction(
            request_id=request_id,
            batch_size=batch_size,
            duration_ms=duration_ms,
            success_count=success_count,
            error_count=batch_size,
            error=str(e),
        )
        logger.error(
            "Critical error in batch prediction", error=str(e), request_id=request_id
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during batch prediction",
        )


def main():
    """Run the API server"""
    uvicorn.run(
        "code.api_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )


if __name__ == "__main__":
    main()
