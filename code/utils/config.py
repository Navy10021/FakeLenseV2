"""Configuration settings for FakeLenseV2"""

from typing import Any, Dict

# Source reliability mapping for news outlets
SOURCE_RELIABILITY_MAPPING = {
    "The New York Times": 0.90,
    "The Washington Post": 0.85,
    "CNN": 0.80,
    "BBC": 0.85,
    "NPR": 0.90,
    "Reuters": 0.90,
    "The Wall Street Journal": 0.85,
    "USA Today": 0.75,
    "Fox News": 0.60,
    "Bloomberg": 0.85,
    "The Guardian": 0.80,
    "Los Angeles Times": 0.80,
    "New York Post": 0.60,
    "HuffPost": 0.70,
    "Associated Press": 0.90,
}

# Default reliability score for unknown sources
DEFAULT_RELIABILITY = 0.50


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the FakeNewsAgent.

    Returns:
        Dictionary containing default configuration parameters
    """
    return {
        # Memory and replay buffer
        "memory_size": 2000,

        # Learning parameters
        "learning_rate": 0.0005,
        "batch_size": 64,
        "gamma": 0.99,  # Discount factor

        # Exploration parameters
        "epsilon": 1.0,  # Initial exploration rate
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,

        # Target network parameters
        "update_target_freq": 10,  # Update target network every N steps
        "tau": 0.005,  # Soft update parameter

        # Model architecture
        "use_residual": True,  # Use DQNResidual instead of basic DQN
        "dropout": 0.2,

        # Training parameters
        "num_episodes": 500,
        "patience": 15,  # Early stopping patience

        # Feature dimensions
        "state_size": 768 + 2,  # BERT embedding + 2 metadata features
        "action_size": 3,  # Fake (0), Suspicious (1), Real (2)

        # Model paths
        "model_save_path": "./models/best_model.pth",

        # Data paths
        "train_data_path": "./data/train_data.json",
        "test_data_path": "./data/test_data.json",

        # Vectorizer settings
        "model_name": "bert-base-uncased",
        "max_seq_length": 512,
    }


def get_training_config() -> Dict[str, Any]:
    """
    Get configuration optimized for training.

    Returns:
        Dictionary containing training configuration parameters
    """
    config = get_default_config()
    config.update({
        "num_episodes": 1000,
        "patience": 20,
        "learning_rate": 0.0003,
    })
    return config


def get_inference_config() -> Dict[str, Any]:
    """
    Get configuration optimized for inference.

    Returns:
        Dictionary containing inference configuration parameters
    """
    config = get_default_config()
    config.update({
        "epsilon": 0.0,  # No exploration during inference
    })
    return config
