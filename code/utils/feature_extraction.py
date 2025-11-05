"""Feature extraction for fake news detection"""

import numpy as np
from typing import Optional

from code.models.vectorizer import BaseVectorizer
from code.utils.config import SOURCE_RELIABILITY_MAPPING, DEFAULT_RELIABILITY


class FeatureExtractor:
    """
    Extracts and combines features for fake news detection.

    Combines:
        - Text embeddings from BERT/RoBERTa
        - Source reliability scores
        - Social reaction metrics
    """

    def __init__(self, vectorizer: Optional[BaseVectorizer] = None):
        """
        Initialize the feature extractor.

        Args:
            vectorizer: Text vectorizer instance (defaults to BaseVectorizer)
        """
        self.vectorizer = vectorizer if vectorizer is not None else BaseVectorizer()

    @staticmethod
    def convert_source_reliability(source: str) -> float:
        """
        Convert media source name to a reliability score.

        Args:
            source: Name of the news source

        Returns:
            Reliability score between 0.0 and 1.0
        """
        return SOURCE_RELIABILITY_MAPPING.get(source, DEFAULT_RELIABILITY)

    def extract_features(
        self,
        text: str,
        source: str,
        social_reactions: float
    ) -> np.ndarray:
        """
        Extract combined features from article data.

        Args:
            text: Article text content
            source: News source name
            social_reactions: Number of social media reactions (shares, likes, etc.)

        Returns:
            Combined feature vector as numpy array
        """
        # Get reliability score
        reliability_score = self.convert_source_reliability(source)

        # Get text embedding
        text_vector = self.vectorizer.vectorize(text)

        # Normalize social reactions (divide by 10,000 for scaling)
        normalized_social = social_reactions / 10000.0

        # Combine all features
        combined_features = np.concatenate([
            text_vector,
            [reliability_score],
            [normalized_social]
        ], axis=0)

        return combined_features
