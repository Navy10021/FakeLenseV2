"""Custom exceptions for FakeLenseV2"""


class FakeLenseError(Exception):
    """Base exception for FakeLenseV2"""

    pass


class ModelLoadError(FakeLenseError):
    """Raised when model fails to load"""

    pass


class DataLoadError(FakeLenseError):
    """Raised when data fails to load"""

    pass


class ValidationError(FakeLenseError):
    """Raised when data validation fails"""

    pass


class TrainingError(FakeLenseError):
    """Raised when training fails"""

    pass


class InferenceError(FakeLenseError):
    """Raised when inference fails"""

    pass


class ConfigurationError(FakeLenseError):
    """Raised when configuration is invalid"""

    pass
