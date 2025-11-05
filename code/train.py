"""Training module for the FakeNewsAgent"""

import os
import json
import logging
from typing import List, Dict, Any

import matplotlib.pyplot as plt

from code.agents.fake_news_agent import FakeNewsAgent
from code.utils.feature_extraction import FeatureExtractor
from code.utils.config import get_default_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Trainer:
    """
    Trainer class for managing the training process of the FakeNewsAgent.
    """

    def __init__(
        self,
        agent: FakeNewsAgent,
        feature_extractor: FeatureExtractor,
        config: Dict[str, Any]
    ):
        """
        Initialize the trainer.

        Args:
            agent: FakeNewsAgent instance
            feature_extractor: FeatureExtractor instance
            config: Configuration dictionary
        """
        self.agent = agent
        self.feature_extractor = feature_extractor
        self.config = config
        self.reward_history = []

    def train(
        self,
        train_data: List[Dict[str, Any]],
        num_episodes: int = None,
        patience: int = None
    ) -> None:
        """
        Train the agent on the training data.

        Args:
            train_data: List of training samples
            num_episodes: Number of training episodes (defaults to config value)
            patience: Early stopping patience (defaults to config value)
        """
        num_episodes = num_episodes or self.config.get("num_episodes", 500)
        patience = patience or self.config.get("patience", 15)

        best_reward = -float("inf")
        no_improvement = 0
        max_possible_reward = len(train_data)

        logging.info(f"Starting training for {num_episodes} episodes")
        logging.info(f"Training samples: {len(train_data)}")
        logging.info(f"Early stopping patience: {patience}")

        for episode in range(num_episodes):
            total_reward = 0

            for sample in train_data:
                # Extract features
                features = self.feature_extractor.extract_features(
                    sample["text"],
                    sample["source_reliability"],
                    sample["social_reactions"]
                )

                state = features
                action = self.agent.act(state)

                # Compute reward
                reward = 1 if action == sample["label"] else -1

                # Store in memory and train
                self.agent.remember(state, action, reward, state, False)
                self.agent.replay()

                total_reward += reward

            # Normalize reward to 100-point scale
            normalized_reward = (total_reward / max_possible_reward) * 100
            self.reward_history.append(normalized_reward)

            logging.info(
                f"Episode {episode + 1:03d}/{num_episodes} - "
                f"Total Reward: {total_reward} ({normalized_reward:.2f}/100) - "
                f"Epsilon: {self.agent.epsilon:.4f}"
            )

            # Check for improvement
            if total_reward > best_reward:
                best_reward = total_reward
                no_improvement = 0

                # Save best model
                model_path = self.config.get("model_save_path", "./models/best_model.pth")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.agent.save(model_path)
                logging.info(f"New best model saved to {model_path}")
            else:
                no_improvement += 1

            # Early stopping
            if no_improvement >= patience:
                logging.info(f"Early stopping triggered after {episode + 1} episodes")
                break

        logging.info("Training completed!")
        logging.info(f"Best reward: {best_reward} ({(best_reward / max_possible_reward) * 100:.2f}/100)")

        # Plot training curve
        self._plot_training_curve()

    def _plot_training_curve(self) -> None:
        """
        Visualize the training reward curve.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.reward_history, label="Normalized Reward", linewidth=2)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Normalized Reward (%)", fontsize=12)
        plt.title("Training Reward Curve", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = "./models/training_curve.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        logging.info(f"Training curve saved to {plot_path}")

        plt.show()


def train_from_config(config_path: str = None) -> None:
    """
    Train the agent using configuration from file or defaults.

    Args:
        config_path: Path to JSON configuration file (optional)
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
    else:
        config = get_default_config()
        logging.info("Using default configuration")

    # Load training data
    train_data_path = config.get("train_data_path", "./data/train_data.json")
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    logging.info(f"Loaded {len(train_data)} training samples from {train_data_path}")

    # Initialize components
    from code.models.vectorizer import BaseVectorizer

    vectorizer = BaseVectorizer(model_name=config.get("model_name", "bert-base-uncased"))
    feature_extractor = FeatureExtractor(vectorizer=vectorizer)

    # Initialize agent
    state_size = config.get("state_size", 770)
    action_size = config.get("action_size", 3)
    agent = FakeNewsAgent(state_size, action_size, config)

    # Initialize trainer
    trainer = Trainer(agent, feature_extractor, config)

    # Train
    trainer.train(train_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train FakeLenseV2 model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file"
    )
    args = parser.parse_args()

    train_from_config(args.config)
