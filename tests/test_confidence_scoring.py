"""Tests for confidence scoring in inference engine"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from code.inference import InferenceEngine


@pytest.fixture
def mock_config():
    """Mock configuration"""
    return {
        "state_size": 770,
        "action_size": 3,
        "model_name": "bert-base-uncased",
        "epsilon": 0.0
    }


@pytest.fixture
def mock_vectorizer():
    """Mock BERT vectorizer"""
    mock = Mock()
    # Return 768-dimensional embedding
    mock.vectorize.return_value = torch.randn(768).numpy()
    return mock


@pytest.fixture
def mock_feature_extractor(mock_vectorizer):
    """Mock feature extractor"""
    mock = Mock()
    # Return 770-dimensional feature vector (768 + 1 + 1)
    mock.extract_features.return_value = torch.randn(770).numpy()
    return mock


@pytest.fixture
def mock_agent():
    """Mock fake news agent with model"""
    agent = Mock()
    agent.device = torch.device("cpu")
    agent.epsilon = 0.0

    # Mock model that returns Q-values
    mock_model = Mock()
    # Return Q-values for 3 classes
    mock_model.return_value = torch.tensor([[2.0, 1.0, 3.5]])  # Highest Q-value for class 2

    agent.model = mock_model
    agent.act = Mock(return_value=2)  # Returns class 2

    return agent


@patch('code.inference.FakeNewsAgent')
@patch('code.inference.FeatureExtractor')
@patch('code.models.vectorizer.BaseVectorizer')
def test_predict_with_confidence_returns_dict(
    mock_vectorizer_class,
    mock_feature_extractor_class,
    mock_agent_class,
    mock_config,
    mock_feature_extractor,
    mock_agent
):
    """Test that predict_with_confidence returns a dictionary with required fields"""
    mock_vectorizer_class.return_value = Mock()
    mock_feature_extractor_class.return_value = mock_feature_extractor
    mock_agent_class.return_value = mock_agent

    # Create inference engine
    engine = InferenceEngine("/fake/model/path.pth", mock_config)

    # Make prediction
    result = engine.predict_with_confidence(
        text="This is a test article",
        source="Reuters",
        social_reactions=1000
    )

    # Check result structure
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "confidence" in result
    assert "all_probabilities" in result
    assert "label" in result


@patch('code.inference.FakeNewsAgent')
@patch('code.inference.FeatureExtractor')
@patch('code.models.vectorizer.BaseVectorizer')
def test_confidence_score_range(
    mock_vectorizer_class,
    mock_feature_extractor_class,
    mock_agent_class,
    mock_config,
    mock_feature_extractor,
    mock_agent
):
    """Test that confidence scores are in valid range [0, 1]"""
    mock_vectorizer_class.return_value = Mock()
    mock_feature_extractor_class.return_value = mock_feature_extractor
    mock_agent_class.return_value = mock_agent

    engine = InferenceEngine("/fake/model/path.pth", mock_config)
    result = engine.predict_with_confidence(
        text="Test article",
        source="BBC",
        social_reactions=500
    )

    # Check confidence is in valid range
    assert 0 <= result["confidence"] <= 1

    # Check all probabilities are in valid range
    for prob in result["all_probabilities"].values():
        assert 0 <= prob <= 1


@patch('code.inference.FakeNewsAgent')
@patch('code.inference.FeatureExtractor')
@patch('code.models.vectorizer.BaseVectorizer')
def test_probabilities_sum_to_one(
    mock_vectorizer_class,
    mock_feature_extractor_class,
    mock_agent_class,
    mock_config,
    mock_feature_extractor,
    mock_agent
):
    """Test that all probabilities sum to approximately 1"""
    mock_vectorizer_class.return_value = Mock()
    mock_feature_extractor_class.return_value = mock_feature_extractor
    mock_agent_class.return_value = mock_agent

    engine = InferenceEngine("/fake/model/path.pth", mock_config)
    result = engine.predict_with_confidence(
        text="Test article",
        source="Reuters",
        social_reactions=1000
    )

    # Sum all probabilities
    prob_sum = sum(result["all_probabilities"].values())

    # Should be approximately 1 (allow for floating point errors)
    assert 0.99 <= prob_sum <= 1.01


@patch('code.inference.FakeNewsAgent')
@patch('code.inference.FeatureExtractor')
@patch('code.models.vectorizer.BaseVectorizer')
def test_all_probability_classes_present(
    mock_vectorizer_class,
    mock_feature_extractor_class,
    mock_agent_class,
    mock_config,
    mock_feature_extractor,
    mock_agent
):
    """Test that all probability classes are present in result"""
    mock_vectorizer_class.return_value = Mock()
    mock_feature_extractor_class.return_value = mock_feature_extractor
    mock_agent_class.return_value = mock_agent

    engine = InferenceEngine("/fake/model/path.pth", mock_config)
    result = engine.predict_with_confidence(
        text="Test article",
        source="Reuters",
        social_reactions=1000
    )

    # Check all classes are present
    assert "fake" in result["all_probabilities"]
    assert "suspicious" in result["all_probabilities"]
    assert "real" in result["all_probabilities"]


@patch('code.inference.FakeNewsAgent')
@patch('code.inference.FeatureExtractor')
@patch('code.models.vectorizer.BaseVectorizer')
def test_label_mapping(
    mock_vectorizer_class,
    mock_feature_extractor_class,
    mock_agent_class,
    mock_config,
    mock_feature_extractor
):
    """Test that prediction labels are correctly mapped"""
    mock_vectorizer_class.return_value = Mock()
    mock_feature_extractor_class.return_value = mock_feature_extractor

    test_cases = [
        (0, "Fake News"),
        (1, "Suspicious News"),
        (2, "Real News")
    ]

    for prediction_class, expected_label in test_cases:
        # Mock agent for this prediction class
        agent = Mock()
        agent.device = torch.device("cpu")
        agent.epsilon = 0.0

        mock_model = Mock()
        # Create Q-values where the target class has highest value
        q_values = torch.tensor([[1.0, 1.0, 1.0]])
        q_values[0][prediction_class] = 5.0
        mock_model.return_value = q_values

        agent.model = mock_model
        mock_agent_class.return_value = agent

        engine = InferenceEngine("/fake/model/path.pth", mock_config)
        result = engine.predict_with_confidence(
            text="Test article",
            source="Test",
            social_reactions=0
        )

        assert result["prediction"] == prediction_class
        assert result["label"] == expected_label


@patch('code.inference.FakeNewsAgent')
@patch('code.inference.FeatureExtractor')
@patch('code.models.vectorizer.BaseVectorizer')
def test_confidence_matches_max_probability(
    mock_vectorizer_class,
    mock_feature_extractor_class,
    mock_agent_class,
    mock_config,
    mock_feature_extractor,
    mock_agent
):
    """Test that confidence matches the maximum probability"""
    mock_vectorizer_class.return_value = Mock()
    mock_feature_extractor_class.return_value = mock_feature_extractor
    mock_agent_class.return_value = mock_agent

    engine = InferenceEngine("/fake/model/path.pth", mock_config)
    result = engine.predict_with_confidence(
        text="Test article",
        source="Reuters",
        social_reactions=1000
    )

    # Confidence should equal the maximum probability
    max_prob = max(result["all_probabilities"].values())
    assert abs(result["confidence"] - max_prob) < 0.001  # Allow small floating point diff


@patch('code.inference.FakeNewsAgent')
@patch('code.inference.FeatureExtractor')
@patch('code.models.vectorizer.BaseVectorizer')
def test_predict_batch_with_confidence(
    mock_vectorizer_class,
    mock_feature_extractor_class,
    mock_agent_class,
    mock_config,
    mock_feature_extractor,
    mock_agent
):
    """Test batch prediction with confidence scores"""
    mock_vectorizer_class.return_value = Mock()
    mock_feature_extractor_class.return_value = mock_feature_extractor
    mock_agent_class.return_value = mock_agent

    engine = InferenceEngine("/fake/model/path.pth", mock_config)

    articles = [
        {"text": "Article 1", "source_reliability": "Reuters", "social_reactions": 100},
        {"text": "Article 2", "source_reliability": "BBC", "social_reactions": 200},
    ]

    # Test with confidence
    results = engine.predict_batch(articles, include_confidence=True)

    assert len(results) == 2
    for result in results:
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "confidence" in result
        assert "all_probabilities" in result
        assert "label" in result


@patch('code.inference.FakeNewsAgent')
@patch('code.inference.FeatureExtractor')
@patch('code.models.vectorizer.BaseVectorizer')
def test_predict_batch_without_confidence(
    mock_vectorizer_class,
    mock_feature_extractor_class,
    mock_agent_class,
    mock_config,
    mock_feature_extractor,
    mock_agent
):
    """Test batch prediction without confidence scores"""
    mock_vectorizer_class.return_value = Mock()
    mock_feature_extractor_class.return_value = mock_feature_extractor

    # Mock agent that returns simple predictions
    agent = Mock()
    agent.device = torch.device("cpu")
    agent.epsilon = 0.0
    agent.model = Mock()
    agent.act = Mock(return_value=2)
    mock_agent_class.return_value = agent

    engine = InferenceEngine("/fake/model/path.pth", mock_config)

    articles = [
        {"text": "Article 1", "source_reliability": "Reuters", "social_reactions": 100},
        {"text": "Article 2", "source_reliability": "BBC", "social_reactions": 200},
    ]

    # Test without confidence
    results = engine.predict_batch(articles, include_confidence=False)

    assert len(results) == 2
    for result in results:
        assert isinstance(result, int)  # Should return just the class number


@patch('code.inference.FakeNewsAgent')
@patch('code.inference.FeatureExtractor')
@patch('code.models.vectorizer.BaseVectorizer')
def test_softmax_conversion(
    mock_vectorizer_class,
    mock_feature_extractor_class,
    mock_agent_class,
    mock_config,
    mock_feature_extractor
):
    """Test that Q-values are correctly converted to probabilities using softmax"""
    mock_vectorizer_class.return_value = Mock()
    mock_feature_extractor_class.return_value = mock_feature_extractor

    # Create agent with known Q-values
    agent = Mock()
    agent.device = torch.device("cpu")
    agent.epsilon = 0.0

    mock_model = Mock()
    # Use simple Q-values for easy manual calculation
    q_values = torch.tensor([[1.0, 2.0, 3.0]])
    mock_model.return_value = q_values

    agent.model = mock_model
    mock_agent_class.return_value = agent

    engine = InferenceEngine("/fake/model/path.pth", mock_config)
    result = engine.predict_with_confidence(
        text="Test article",
        source="Test",
        social_reactions=0
    )

    # Manually calculate expected softmax probabilities
    exp_values = torch.exp(q_values)
    expected_probs = exp_values / exp_values.sum()

    # Check that probabilities match expected softmax output
    probs = result["all_probabilities"]
    assert abs(probs["fake"] - expected_probs[0][0].item()) < 0.001
    assert abs(probs["suspicious"] - expected_probs[0][1].item()) < 0.001
    assert abs(probs["real"] - expected_probs[0][2].item()) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
