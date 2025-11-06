"""Structured logging utilities for FakeLenseV2"""

import logging
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import contextmanager
import time


class StructuredLogger:
    """
    Structured logger for API requests and predictions.
    Logs in JSON format for easy parsing and analysis.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize the structured logger.

        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []

        # Create console handler with JSON formatter
        handler = logging.StreamHandler()
        handler.setLevel(level)
        self.logger.addHandler(handler)

    def _create_log_entry(
        self,
        event: str,
        level: str,
        message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a structured log entry.

        Args:
            event: Event type
            level: Log level
            message: Log message
            **kwargs: Additional fields

        Returns:
            Dictionary containing log entry
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "level": level,
        }

        if message:
            log_entry["message"] = message

        # Add all additional fields
        log_entry.update(kwargs)

        return log_entry

    def log_prediction(
        self,
        request_id: str,
        text_length: int,
        source: Optional[str],
        prediction: int,
        label: str,
        confidence: float,
        duration_ms: float,
        error: Optional[str] = None
    ):
        """
        Log a prediction event.

        Args:
            request_id: Unique request identifier
            text_length: Length of input text
            source: News source
            prediction: Predicted class
            label: Human-readable label
            confidence: Prediction confidence
            duration_ms: Processing duration in milliseconds
            error: Error message if any
        """
        log_entry = self._create_log_entry(
            event="prediction",
            level="INFO" if not error else "ERROR",
            request_id=request_id,
            text_length=text_length,
            source=source,
            prediction=prediction,
            label=label,
            confidence=confidence,
            duration_ms=duration_ms,
            error=error
        )

        if error:
            self.logger.error(json.dumps(log_entry))
        else:
            self.logger.info(json.dumps(log_entry))

    def log_batch_prediction(
        self,
        request_id: str,
        batch_size: int,
        duration_ms: float,
        success_count: int,
        error_count: int,
        error: Optional[str] = None
    ):
        """
        Log a batch prediction event.

        Args:
            request_id: Unique request identifier
            batch_size: Number of articles in batch
            duration_ms: Processing duration in milliseconds
            success_count: Number of successful predictions
            error_count: Number of failed predictions
            error: Error message if any
        """
        log_entry = self._create_log_entry(
            event="batch_prediction",
            level="INFO" if not error else "ERROR",
            request_id=request_id,
            batch_size=batch_size,
            duration_ms=duration_ms,
            success_count=success_count,
            error_count=error_count,
            error=error
        )

        if error:
            self.logger.error(json.dumps(log_entry))
        else:
            self.logger.info(json.dumps(log_entry))

    def log_api_request(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        client_ip: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Log an API request.

        Args:
            request_id: Unique request identifier
            method: HTTP method
            path: Request path
            status_code: HTTP status code
            duration_ms: Processing duration in milliseconds
            client_ip: Client IP address
            error: Error message if any
        """
        log_entry = self._create_log_entry(
            event="api_request",
            level="INFO" if status_code < 400 else "ERROR",
            request_id=request_id,
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            client_ip=client_ip,
            error=error
        )

        if status_code >= 400:
            self.logger.error(json.dumps(log_entry))
        else:
            self.logger.info(json.dumps(log_entry))

    def log_model_load(
        self,
        model_path: str,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """
        Log a model loading event.

        Args:
            model_path: Path to model file
            duration_ms: Loading duration in milliseconds
            success: Whether loading was successful
            error: Error message if any
        """
        log_entry = self._create_log_entry(
            event="model_load",
            level="INFO" if success else "ERROR",
            model_path=model_path,
            duration_ms=duration_ms,
            success=success,
            error=error
        )

        if success:
            self.logger.info(json.dumps(log_entry))
        else:
            self.logger.error(json.dumps(log_entry))

    def info(self, message: str, **kwargs):
        """Log info message"""
        log_entry = self._create_log_entry("info", "INFO", message, **kwargs)
        self.logger.info(json.dumps(log_entry))

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        log_entry = self._create_log_entry("warning", "WARNING", message, **kwargs)
        self.logger.warning(json.dumps(log_entry))

    def error(self, message: str, **kwargs):
        """Log error message"""
        log_entry = self._create_log_entry("error", "ERROR", message, **kwargs)
        self.logger.error(json.dumps(log_entry))


class RequestIDMiddleware:
    """
    Middleware to generate and track request IDs.
    """

    @staticmethod
    def generate_request_id() -> str:
        """Generate a unique request ID"""
        return str(uuid.uuid4())


@contextmanager
def log_duration(logger: StructuredLogger, event_name: str, **kwargs):
    """
    Context manager to log event duration.

    Usage:
        with log_duration(logger, "prediction", text_length=100):
            # Your code here
            result = model.predict(...)

    Args:
        logger: StructuredLogger instance
        event_name: Name of the event
        **kwargs: Additional fields to log
    """
    start_time = time.time()
    error = None

    try:
        yield
    except Exception as e:
        error = str(e)
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        log_entry = logger._create_log_entry(
            event=event_name,
            level="INFO" if not error else "ERROR",
            duration_ms=duration_ms,
            error=error,
            **kwargs
        )

        if error:
            logger.logger.error(json.dumps(log_entry))
        else:
            logger.logger.info(json.dumps(log_entry))
