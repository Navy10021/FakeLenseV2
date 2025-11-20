"""Data validation utilities for FakeLenseV2"""

from typing import Any, Dict, List
import json
import logging

from code.utils.config import (
    TEXT_MIN_LENGTH,
    TEXT_MAX_LENGTH,
    SOCIAL_REACTIONS_MAX,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class DataValidator:
    """Validator for input data"""

    @staticmethod
    def validate_article_text(
        text: str,
        min_length: int = TEXT_MIN_LENGTH,
        max_length: int = TEXT_MAX_LENGTH,
    ) -> bool:
        """
        Validate article text.

        Args:
            text: Article text to validate
            min_length: Minimum acceptable text length (default from config)
            max_length: Maximum acceptable text length (default from config)

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(text, str):
            raise ValidationError(f"Text must be a string, got {type(text)}")

        if len(text.strip()) < min_length:
            raise ValidationError(f"Text too short (minimum {min_length} characters)")

        if len(text) > max_length:
            raise ValidationError(f"Text too long (maximum {max_length} characters)")

        return True

    @staticmethod
    def validate_source(source: str) -> bool:
        """
        Validate news source name.

        Args:
            source: Source name to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(source, str):
            raise ValidationError(f"Source must be a string, got {type(source)}")

        if len(source.strip()) == 0:
            logger.warning("Empty source provided, will use default reliability")

        return True

    @staticmethod
    def validate_social_reactions(reactions: float, max_value: float = SOCIAL_REACTIONS_MAX) -> bool:
        """
        Validate social reactions count.

        Args:
            reactions: Number of social reactions
            max_value: Maximum allowed value (default from config)

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(reactions, (int, float)):
            raise ValidationError(f"Social reactions must be numeric, got {type(reactions)}")

        if reactions < 0:
            raise ValidationError("Social reactions cannot be negative")

        if reactions > max_value:
            raise ValidationError(f"Social reactions value too large (max: {max_value})")

        return True

    @staticmethod
    def validate_training_sample(sample: Dict[str, Any]) -> bool:
        """
        Validate a training data sample.

        Args:
            sample: Training sample dictionary

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        required_fields = ["text", "source_reliability", "social_reactions", "label"]

        # Check required fields
        for field in required_fields:
            if field not in sample:
                raise ValidationError(f"Missing required field: {field}")

        # Validate text
        DataValidator.validate_article_text(sample["text"])

        # Validate source
        DataValidator.validate_source(sample["source_reliability"])

        # Validate social reactions
        DataValidator.validate_social_reactions(sample["social_reactions"])

        # Validate label
        if sample["label"] not in [0, 1, 2]:
            raise ValidationError(f"Invalid label: {sample['label']} (must be 0, 1, or 2)")

        return True

    @staticmethod
    def validate_training_data(data: List[Dict[str, Any]]) -> bool:
        """
        Validate entire training dataset.

        Args:
            data: List of training samples

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, list):
            raise ValidationError(f"Training data must be a list, got {type(data)}")

        if len(data) == 0:
            raise ValidationError("Training data is empty")

        # Validate each sample
        for i, sample in enumerate(data):
            try:
                DataValidator.validate_training_sample(sample)
            except ValidationError as e:
                raise ValidationError(f"Invalid sample at index {i}: {str(e)}")

        logger.info(f"Validated {len(data)} training samples successfully")
        return True

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        required_params = [
            "state_size", "action_size", "learning_rate",
            "batch_size", "gamma", "epsilon"
        ]

        for param in required_params:
            if param not in config:
                raise ValidationError(f"Missing required config parameter: {param}")

        # Validate ranges
        if config["learning_rate"] <= 0 or config["learning_rate"] > 1:
            raise ValidationError("Learning rate must be between 0 and 1")

        if config["batch_size"] <= 0:
            raise ValidationError("Batch size must be positive")

        if config["gamma"] < 0 or config["gamma"] > 1:
            raise ValidationError("Gamma must be between 0 and 1")

        if config["epsilon"] < 0 or config["epsilon"] > 1:
            raise ValidationError("Epsilon must be between 0 and 1")

        logger.info("Configuration validated successfully")
        return True


def validate_json_file(file_path: str, data_type: str = "training") -> bool:
    """
    Validate a JSON data file.

    Args:
        file_path: Path to JSON file
        data_type: Type of data ("training", "test", or "config")

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValidationError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {file_path}: {str(e)}")

    if data_type in ["training", "test"]:
        return DataValidator.validate_training_data(data)
    elif data_type == "config":
        return DataValidator.validate_config(data)
    else:
        raise ValidationError(f"Unknown data type: {data_type}")
