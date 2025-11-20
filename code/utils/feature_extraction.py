"""Feature extraction for fake news detection"""

import numpy as np
from typing import Optional

from code.models.vectorizer import BaseVectorizer
from code.utils.config import (
    SOURCE_RELIABILITY_MAPPING,
    DEFAULT_RELIABILITY,
    SOCIAL_REACTIONS_NORMALIZER,
)


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
        self, text: str, source: str, social_reactions: float
    ) -> np.ndarray:
        """
        Extract combined features from article data.

        Creates a comprehensive feature vector combining:
        - Text embeddings (768-dimensional BERT/RoBERTa representation)
        - Source reliability score (0.0-1.0, based on known source mapping)
        - Normalized social reactions (scaled by configured normalizer)

        Args:
            text: Article text content (will be truncated to max_seq_length)
            source: News source name (e.g., "CNN", "BBC")
            social_reactions: Number of social media reactions (shares, likes, comments, etc.)

        Returns:
            Combined feature vector as numpy array with shape (770,):
            - [0:768]: Text embedding from transformer model
            - [768]: Source reliability score
            - [769]: Normalized social reactions count

        Example:
            >>> extractor = FeatureExtractor()
            >>> features = extractor.extract_features(
            ...     "Breaking news article text...",
            ...     "The New York Times",
            ...     5000
            ... )
            >>> features.shape
            (770,)
        """
        # Get reliability score
        reliability_score = self.convert_source_reliability(source)

        # Get text embedding
        text_vector = self.vectorizer.vectorize(text)

        # Normalize social reactions using configured normalizer
        normalized_social = social_reactions / SOCIAL_REACTIONS_NORMALIZER

        # Combine all features
        combined_features = np.concatenate(
            [text_vector, [reliability_score], [normalized_social]], axis=0
        )

        return combined_features
