"""Tests for the FastAPI server"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import torch

# Note: We'll mock the inference engine to avoid loading the actual model
from code.api_server import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_inference_engine():
    """Mock inference engine"""
    mock_engine = Mock()
    mock_engine.predict.return_value = 2  # Real News
    mock_engine.predict_with_confidence.return_value = {
        "prediction": 2,
        "label": "Real News",
        "confidence": 0.95,
        "all_probabilities": {
            "fake": 0.02,
            "suspicious": 0.03,
            "real": 0.95
        }
    }
    return mock_engine


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data
    assert data["version"] == "2.0.0"


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data
    assert "uptime_seconds" in data


@patch('code.api_server.inference_engine')
def test_predict_endpoint_success(mock_engine, client):
    """Test successful prediction"""
    mock_engine.predict_with_confidence.return_value = {
        "prediction": 2,
        "label": "Real News",
        "confidence": 0.95,
        "all_probabilities": {
            "fake": 0.02,
            "suspicious": 0.03,
            "real": 0.95
        }
    }

    response = client.post("/predict", json={
        "text": "This is a test article with enough length to pass validation checks.",
        "source": "Reuters",
        "social_reactions": 1000
    })

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "label" in data
    assert "confidence" in data
    assert "all_probabilities" in data
    assert "request_id" in data
    assert data["confidence"] >= 0 and data["confidence"] <= 1
    assert "X-Request-ID" in response.headers


@patch('code.api_server.inference_engine')
def test_predict_endpoint_without_optional_fields(mock_engine, client):
    """Test prediction without optional fields"""
    mock_engine.predict_with_confidence.return_value = {
        "prediction": 0,
        "label": "Fake News",
        "confidence": 0.88,
        "all_probabilities": {
            "fake": 0.88,
            "suspicious": 0.10,
            "real": 0.02
        }
    }

    response = client.post("/predict", json={
        "text": "This is a test article without optional fields but with sufficient length."
    })

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["label"] == "Fake News"


def test_predict_endpoint_text_too_short(client):
    """Test prediction with text too short"""
    response = client.post("/predict", json={
        "text": "Short",
        "source": "Test"
    })

    assert response.status_code == 422  # Validation error


def test_predict_endpoint_no_model(client):
    """Test prediction when model is not loaded"""
    with patch('code.api_server.inference_engine', None):
        response = client.post("/predict", json={
            "text": "This is a test article with enough length.",
            "source": "Test"
        })

        assert response.status_code == 503


@patch('code.api_server.inference_engine')
def test_predict_endpoint_inference_error(mock_engine, client):
    """Test prediction when inference fails"""
    mock_engine.predict_with_confidence.side_effect = Exception("Inference failed")

    response = client.post("/predict", json={
        "text": "This is a test article that will cause an error in inference.",
        "source": "Test"
    })

    assert response.status_code == 500


@patch('code.api_server.inference_engine')
def test_batch_predict_endpoint_success(mock_engine, client):
    """Test successful batch prediction"""
    mock_engine.predict_with_confidence.return_value = {
        "prediction": 2,
        "label": "Real News",
        "confidence": 0.95,
        "all_probabilities": {
            "fake": 0.02,
            "suspicious": 0.03,
            "real": 0.95
        }
    }

    response = client.post("/batch_predict", json={
        "articles": [
            {
                "text": "Article 1 with sufficient length for validation checks.",
                "source": "Reuters"
            },
            {
                "text": "Article 2 with sufficient length for validation checks.",
                "source": "BBC"
            }
        ]
    })

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "request_id" in data
    assert "total" in data
    assert "success" in data
    assert "errors" in data
    assert len(data["predictions"]) == 2
    assert data["total"] == 2


@patch('code.api_server.inference_engine')
def test_batch_predict_too_many_articles(mock_engine, client):
    """Test batch prediction with too many articles"""
    articles = [
        {
            "text": f"Article {i} with sufficient length.",
            "source": "Test"
        }
        for i in range(101)
    ]

    response = client.post("/batch_predict", json={"articles": articles})
    assert response.status_code == 422  # Validation error - max 100 items


def test_batch_predict_no_model(client):
    """Test batch prediction when model is not loaded"""
    with patch('code.api_server.inference_engine', None):
        response = client.post("/batch_predict", json={
            "articles": [
                {"text": "Test article with enough length.", "source": "Test"}
            ]
        })

        assert response.status_code == 503


@patch('code.api_server.inference_engine')
def test_batch_predict_partial_failure(mock_engine, client):
    """Test batch prediction with some failures"""
    def side_effect(*args, **kwargs):
        # First call succeeds, second fails, third succeeds
        if not hasattr(side_effect, 'call_count'):
            side_effect.call_count = 0
        side_effect.call_count += 1

        if side_effect.call_count == 2:
            raise Exception("Inference failed for this article")

        return {
            "prediction": 2,
            "label": "Real News",
            "confidence": 0.95,
            "all_probabilities": {
                "fake": 0.02,
                "suspicious": 0.03,
                "real": 0.95
            }
        }

    mock_engine.predict_with_confidence.side_effect = side_effect

    response = client.post("/batch_predict", json={
        "articles": [
            {"text": "Article 1 with sufficient length.", "source": "Test"},
            {"text": "Article 2 with sufficient length.", "source": "Test"},
            {"text": "Article 3 with sufficient length.", "source": "Test"}
        ]
    })

    assert response.status_code == 200
    data = response.json()
    assert data["success"] == 2
    assert data["errors"] == 1


def test_request_id_in_headers(client):
    """Test that request ID is added to response headers"""
    response = client.get("/")
    assert "X-Request-ID" in response.headers


def test_cors_headers(client):
    """Test CORS headers are present"""
    response = client.options("/", headers={
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "POST"
    })
    # CORS middleware should handle OPTIONS requests


@patch('code.api_server.inference_engine')
def test_confidence_scores_range(mock_engine, client):
    """Test that confidence scores are in valid range [0, 1]"""
    mock_engine.predict_with_confidence.return_value = {
        "prediction": 1,
        "label": "Suspicious News",
        "confidence": 0.65,
        "all_probabilities": {
            "fake": 0.20,
            "suspicious": 0.65,
            "real": 0.15
        }
    }

    response = client.post("/predict", json={
        "text": "Test article with sufficient length for validation.",
        "source": "Test"
    })

    data = response.json()
    assert 0 <= data["confidence"] <= 1

    # Check all probabilities sum to approximately 1
    probs = data["all_probabilities"]
    total_prob = sum(probs.values())
    assert 0.99 <= total_prob <= 1.01  # Allow small floating point errors


@patch('code.api_server.inference_engine')
def test_all_prediction_classes(mock_engine, client):
    """Test all three prediction classes (fake, suspicious, real)"""
    test_cases = [
        (0, "Fake News", 0.92),
        (1, "Suspicious News", 0.73),
        (2, "Real News", 0.88)
    ]

    for prediction, label, confidence in test_cases:
        mock_engine.predict_with_confidence.return_value = {
            "prediction": prediction,
            "label": label,
            "confidence": confidence,
            "all_probabilities": {
                "fake": 0.33,
                "suspicious": 0.33,
                "real": 0.34
            }
        }

        response = client.post("/predict", json={
            "text": f"Test article for {label} class with sufficient length.",
            "source": "Test"
        })

        data = response.json()
        assert data["prediction"] == prediction
        assert data["label"] == label
        assert data["confidence"] == confidence


@patch('code.api_server.inference_engine')
def test_social_reactions_validation(mock_engine, client):
    """Test social reactions field validation"""
    mock_engine.predict_with_confidence.return_value = {
        "prediction": 2,
        "label": "Real News",
        "confidence": 0.95,
        "all_probabilities": {
            "fake": 0.02,
            "suspicious": 0.03,
            "real": 0.95
        }
    }

    # Valid social reactions
    response = client.post("/predict", json={
        "text": "Test article with valid social reactions field and sufficient length.",
        "source": "Test",
        "social_reactions": 5000
    })
    assert response.status_code == 200

    # Negative social reactions should fail validation
    response = client.post("/predict", json={
        "text": "Test article with negative social reactions field.",
        "source": "Test",
        "social_reactions": -100
    })
    assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
