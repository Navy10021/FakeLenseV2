"""Inference and evaluation module for FakeNewsAgent"""

import os
import json
import logging
from typing import List, Dict, Any, Union

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from code.agents.fake_news_agent import FakeNewsAgent
from code.utils.feature_extraction import FeatureExtractor
from code.utils.config import get_default_config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class InferenceEngine:
    """
    Optimized inference engine for fake news detection.
    Loads the model once and reuses it for multiple predictions.
    """

    def __init__(self, model_path: str, config: Dict[str, Any] = None) -> None:
        """
        Initialize the inference engine.

        Args:
            model_path: Path to the trained model file (.pth)
            config: Configuration dictionary with model settings (optional, uses defaults if not provided)

        Raises:
            FileNotFoundError: If model_path does not exist
            RuntimeError: If model loading fails

        Example:
            >>> engine = InferenceEngine("./models/best_model.pth")
            >>> result = engine.predict_with_confidence("Breaking news...", "CNN", 1000)
        """
        self.config = config or get_default_config()
        self.model_path = model_path

        # Initialize feature extractor
        from code.models.vectorizer import BaseVectorizer

        vectorizer = BaseVectorizer(
            model_name=self.config.get("model_name", "bert-base-uncased")
        )
        self.feature_extractor = FeatureExtractor(vectorizer=vectorizer)

        # Initialize and load agent
        state_size = self.config.get("state_size", 770)
        action_size = self.config.get("action_size", 3)
        self.agent = FakeNewsAgent(state_size, action_size, self.config)
        self.agent.load(model_path)
        self.agent.epsilon = 0.0  # No exploration during inference

        logging.info(f"Loaded model from {model_path}")

    def predict(self, text: str, source: str, social_reactions: float) -> int:
        """
        Predict the class of a single news article.

        Args:
            text: Article text content
            source: News source name
            social_reactions: Number of social media reactions

        Returns:
            Predicted class (0: Fake, 1: Suspicious, 2: Real)
        """
        features = self.feature_extractor.extract_features(
            text, source, social_reactions
        )
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.agent.device)

        with torch.no_grad():
            action = self.agent.act(features_tensor)

        return action

    def predict_with_confidence(
        self, text: str, source: str, social_reactions: float
    ) -> Dict[str, Any]:
        """
        Predict the class of a single news article with confidence scores.

        Args:
            text: Article text content
            source: News source name
            social_reactions: Number of social media reactions

        Returns:
            Dictionary containing:
                - prediction: Predicted class (0: Fake, 1: Suspicious, 2: Real)
                - confidence: Confidence score for the prediction (0-1)
                - all_probabilities: Dictionary with probabilities for all classes
                - label: Human-readable label
        """
        features = self.feature_extractor.extract_features(
            text, source, social_reactions
        )
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.agent.device)

        with torch.no_grad():
            # Get Q-values from the model
            q_values = self.agent.model(features_tensor)

            # Convert Q-values to probabilities using softmax
            probabilities = torch.softmax(q_values, dim=1)

            # Get prediction and confidence
            prediction = torch.argmax(q_values, dim=1).item()
            confidence = probabilities[0][prediction].item()

            # Get all class probabilities
            all_probs = {
                "fake": float(probabilities[0][0]),
                "suspicious": float(probabilities[0][1]),
                "real": float(probabilities[0][2]),
            }

        # Map prediction to label
        label_map = {0: "Fake News", 1: "Suspicious News", 2: "Real News"}

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "all_probabilities": all_probs,
            "label": label_map[prediction],
        }

    def predict_batch(
        self, articles: List[Dict[str, Any]], include_confidence: bool = False
    ) -> Union[List[int], List[Dict[str, Any]]]:
        """
        Predict the class of multiple news articles.

        Args:
            articles: List of dictionaries with keys: text, source_reliability, social_reactions
            include_confidence: If True, return full prediction dictionaries with confidence scores

        Returns:
            List of predicted classes (if include_confidence=False)
            or List of prediction dictionaries (if include_confidence=True)
        """
        predictions = []
        for article in articles:
            if include_confidence:
                pred = self.predict_with_confidence(
                    article["text"],
                    article.get("source_reliability", "Unknown"),
                    article.get("social_reactions", 0),
                )
            else:
                pred = self.predict(
                    article["text"],
                    article.get("source_reliability", "Unknown"),
                    article.get("social_reactions", 0),
                )
            predictions.append(pred)
        return predictions


class Evaluator:
    """
    Evaluation class for computing metrics and visualizations.
    """

    def __init__(self, inference_engine: InferenceEngine):
        """
        Initialize the evaluator.

        Args:
            inference_engine: InferenceEngine instance
        """
        self.engine = inference_engine

    def evaluate(
        self, test_data: List[Dict[str, Any]], verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_data: List of test samples with labels
            verbose: Whether to print detailed results

        Returns:
            Dictionary containing evaluation metrics
        """
        y_true = []
        y_pred = []

        if verbose:
            logging.info("\n" + "=" * 50)
            logging.info("EVALUATING ON TEST DATASET")
            logging.info("=" * 50)

        for sample in test_data:
            prediction = self.engine.predict(
                sample["text"],
                sample.get("source_reliability", "Unknown"),
                sample.get("social_reactions", 0),
            )

            y_true.append(sample["label"])
            y_pred.append(prediction)

            if verbose:
                label_map = {0: "Fake News", 1: "Suspicious News", 2: "Real News"}
                logging.info(f"Article: {sample['text'][:80]}...")
                logging.info(f"Prediction: {label_map[prediction]}")
                logging.info(f"Ground Truth: {label_map[sample['label']]}")
                logging.info("-" * 50)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

        if verbose:
            logging.info("\n" + "=" * 50)
            logging.info("PERFORMANCE METRICS")
            logging.info("=" * 50)
            logging.info(f"Accuracy:  {metrics['accuracy']:.4f}")
            logging.info(f"Precision: {metrics['precision']:.4f}")
            logging.info(f"Recall:    {metrics['recall']:.4f}")
            logging.info(f"F1-Score:  {metrics['f1_score']:.4f}")
            logging.info("\n" + "=" * 50)
            logging.info("CLASSIFICATION REPORT")
            logging.info("=" * 50)
            print(
                classification_report(
                    y_true,
                    y_pred,
                    target_names=["Fake News", "Suspicious News", "Real News"],
                )
            )

        # Visualize confusion matrix
        self._plot_confusion_matrix(y_true, y_pred)

        return metrics

    def _plot_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> None:
        """
        Plot and save confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fake News", "Suspicious News", "Real News"],
            yticklabels=["Fake News", "Suspicious News", "Real News"],
            cbar_kws={"label": "Count"},
        )
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save plot
        plot_path = "./models/confusion_matrix.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        logging.info(f"Confusion matrix saved to {plot_path}")

        plt.show()


def evaluate_from_config(
    config_path: str = None, model_path: str = None
) -> Dict[str, float]:
    """
    Evaluate the model using configuration.

    Args:
        config_path: Path to JSON configuration file (optional)
        model_path: Path to trained model (optional)

    Returns:
        Dictionary containing evaluation metrics
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
    else:
        config = get_default_config()
        logging.info("Using default configuration")

    # Determine model path
    model_path = model_path or config.get("model_save_path", "./models/best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load test data
    test_data_path = config.get("test_data_path", "./data/test_data.json")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logging.info(f"Loaded {len(test_data)} test samples from {test_data_path}")

    # Initialize inference engine and evaluator
    engine = InferenceEngine(model_path, config)
    evaluator = Evaluator(engine)

    # Evaluate
    metrics = evaluator.evaluate(test_data, verbose=True)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FakeLenseV2 model")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to trained model file"
    )
    args = parser.parse_args()

    evaluate_from_config(args.config, args.model)
