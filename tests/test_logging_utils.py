"""Tests for structured logging utilities"""

import pytest
import json
import logging
from code.utils.logging_utils import (
    StructuredLogger,
    RequestIDMiddleware,
    log_duration
)


@pytest.fixture
def structured_logger():
    """Create a structured logger for testing"""
    return StructuredLogger("test-logger", level=logging.DEBUG)


def test_structured_logger_initialization(structured_logger):
    """Test logger initialization"""
    assert structured_logger.logger is not None
    assert structured_logger.logger.name == "test-logger"


def test_log_prediction(structured_logger, caplog):
    """Test prediction logging"""
    with caplog.at_level(logging.INFO):
        structured_logger.log_prediction(
            request_id="test-123",
            text_length=100,
            source="Reuters",
            prediction=2,
            label="Real News",
            confidence=0.95,
            duration_ms=50.5
        )

    assert len(caplog.records) == 1
    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["event"] == "prediction"
    assert log_entry["level"] == "INFO"
    assert log_entry["request_id"] == "test-123"
    assert log_entry["text_length"] == 100
    assert log_entry["source"] == "Reuters"
    assert log_entry["prediction"] == 2
    assert log_entry["label"] == "Real News"
    assert log_entry["confidence"] == 0.95
    assert log_entry["duration_ms"] == 50.5
    assert "timestamp" in log_entry


def test_log_prediction_with_error(structured_logger, caplog):
    """Test prediction logging with error"""
    with caplog.at_level(logging.ERROR):
        structured_logger.log_prediction(
            request_id="test-456",
            text_length=50,
            source="Unknown",
            prediction=-1,
            label="Error",
            confidence=0.0,
            duration_ms=10.0,
            error="Model inference failed"
        )

    assert len(caplog.records) == 1
    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["level"] == "ERROR"
    assert log_entry["error"] == "Model inference failed"


def test_log_batch_prediction(structured_logger, caplog):
    """Test batch prediction logging"""
    with caplog.at_level(logging.INFO):
        structured_logger.log_batch_prediction(
            request_id="batch-789",
            batch_size=10,
            duration_ms=500.0,
            success_count=9,
            error_count=1
        )

    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["event"] == "batch_prediction"
    assert log_entry["batch_size"] == 10
    assert log_entry["success_count"] == 9
    assert log_entry["error_count"] == 1


def test_log_api_request(structured_logger, caplog):
    """Test API request logging"""
    with caplog.at_level(logging.INFO):
        structured_logger.log_api_request(
            request_id="api-001",
            method="POST",
            path="/predict",
            status_code=200,
            duration_ms=75.5,
            client_ip="192.168.1.1"
        )

    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["event"] == "api_request"
    assert log_entry["method"] == "POST"
    assert log_entry["path"] == "/predict"
    assert log_entry["status_code"] == 200
    assert log_entry["client_ip"] == "192.168.1.1"


def test_log_api_request_error(structured_logger, caplog):
    """Test API request logging with error status"""
    with caplog.at_level(logging.ERROR):
        structured_logger.log_api_request(
            request_id="api-002",
            method="POST",
            path="/predict",
            status_code=500,
            duration_ms=20.0,
            error="Internal server error"
        )

    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["level"] == "ERROR"
    assert log_entry["status_code"] == 500
    assert log_entry["error"] == "Internal server error"


def test_log_model_load_success(structured_logger, caplog):
    """Test model loading success logging"""
    with caplog.at_level(logging.INFO):
        structured_logger.log_model_load(
            model_path="/path/to/model.pth",
            duration_ms=1500.0,
            success=True
        )

    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["event"] == "model_load"
    assert log_entry["level"] == "INFO"
    assert log_entry["model_path"] == "/path/to/model.pth"
    assert log_entry["success"] is True


def test_log_model_load_failure(structured_logger, caplog):
    """Test model loading failure logging"""
    with caplog.at_level(logging.ERROR):
        structured_logger.log_model_load(
            model_path="/path/to/model.pth",
            duration_ms=100.0,
            success=False,
            error="File not found"
        )

    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["level"] == "ERROR"
    assert log_entry["success"] is False
    assert log_entry["error"] == "File not found"


def test_structured_logger_info(structured_logger, caplog):
    """Test info logging"""
    with caplog.at_level(logging.INFO):
        structured_logger.info("Test message", extra_field="extra_value")

    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["event"] == "info"
    assert log_entry["level"] == "INFO"
    assert log_entry["message"] == "Test message"
    assert log_entry["extra_field"] == "extra_value"


def test_structured_logger_warning(structured_logger, caplog):
    """Test warning logging"""
    with caplog.at_level(logging.WARNING):
        structured_logger.warning("Warning message", code=1001)

    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["event"] == "warning"
    assert log_entry["level"] == "WARNING"
    assert log_entry["message"] == "Warning message"
    assert log_entry["code"] == 1001


def test_structured_logger_error(structured_logger, caplog):
    """Test error logging"""
    with caplog.at_level(logging.ERROR):
        structured_logger.error("Error message", error_code=500)

    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["event"] == "error"
    assert log_entry["level"] == "ERROR"
    assert log_entry["message"] == "Error message"
    assert log_entry["error_code"] == 500


def test_request_id_generation():
    """Test request ID generation"""
    request_id_1 = RequestIDMiddleware.generate_request_id()
    request_id_2 = RequestIDMiddleware.generate_request_id()

    # Check that request IDs are strings
    assert isinstance(request_id_1, str)
    assert isinstance(request_id_2, str)

    # Check that request IDs are unique
    assert request_id_1 != request_id_2

    # Check that request IDs are valid UUIDs (36 characters with hyphens)
    assert len(request_id_1) == 36
    assert len(request_id_2) == 36


def test_log_duration_success(structured_logger, caplog):
    """Test log_duration context manager with success"""
    with caplog.at_level(logging.INFO):
        with log_duration(structured_logger, "test_operation", operation_id=123):
            # Simulate some work
            pass

    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["event"] == "test_operation"
    assert log_entry["level"] == "INFO"
    assert log_entry["operation_id"] == 123
    assert "duration_ms" in log_entry
    assert log_entry["duration_ms"] >= 0


def test_log_duration_with_error(structured_logger, caplog):
    """Test log_duration context manager with error"""
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            with log_duration(structured_logger, "failing_operation"):
                raise ValueError("Test error")

    log_entry = json.loads(caplog.records[0].message)

    assert log_entry["event"] == "failing_operation"
    assert log_entry["level"] == "ERROR"
    assert log_entry["error"] == "Test error"
    assert "duration_ms" in log_entry


def test_timestamp_format(structured_logger, caplog):
    """Test that timestamps are in ISO format with Z suffix"""
    with caplog.at_level(logging.INFO):
        structured_logger.info("Test timestamp")

    log_entry = json.loads(caplog.records[0].message)

    # Check timestamp format (ISO 8601 with Z)
    assert "timestamp" in log_entry
    assert log_entry["timestamp"].endswith("Z")
    assert "T" in log_entry["timestamp"]  # ISO format includes T separator


def test_log_entry_json_serializable(structured_logger, caplog):
    """Test that all log entries are valid JSON"""
    with caplog.at_level(logging.INFO):
        structured_logger.log_prediction(
            request_id="test",
            text_length=100,
            source="Test",
            prediction=0,
            label="Fake",
            confidence=0.9,
            duration_ms=50.0
        )

    # This should not raise an exception
    log_entry = json.loads(caplog.records[0].message)
    assert isinstance(log_entry, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
