"""Unit tests for inference module"""

import pytest
from code.utils.feature_extraction import FeatureExtractor
from code.utils.config import SOURCE_RELIABILITY_MAPPING


class TestFeatureExtractor:
    """Tests for feature extraction"""

    def test_convert_source_reliability_known(self):
        """Test converting known source to reliability score"""
        score = FeatureExtractor.convert_source_reliability("Reuters")
        assert score == 0.90

    def test_convert_source_reliability_unknown(self):
        """Test converting unknown source to default score"""
        score = FeatureExtractor.convert_source_reliability("UnknownSource")
        assert score == 0.50

    @pytest.mark.skip(reason="Requires BERT model download")
    def test_extract_features(self):
        """Test feature extraction"""
        extractor = FeatureExtractor()
        features = extractor.extract_features(
            text="Test article about news.",
            source="Reuters",
            social_reactions=5000
        )
        # 768 (BERT) + 1 (reliability) + 1 (social)
        assert features.shape == (770,)
        assert features[-2] == 0.90  # Reuters reliability
        assert features[-1] == 0.5  # 5000 / 10000
