"""Unit tests for model modules"""

import pytest
import torch
import numpy as np

from code.models.dqn import DQN, DQNResidual
from code.models.vectorizer import BaseVectorizer


class TestDQN:
    """Tests for DQN model"""

    def test_dqn_initialization(self):
        """Test DQN model initialization"""
        model = DQN(input_dim=770, output_dim=3)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_dqn_forward(self):
        """Test DQN forward pass"""
        model = DQN(input_dim=770, output_dim=3)
        input_tensor = torch.randn(1, 770)
        output = model(input_tensor)
        assert output.shape == (1, 3)

    def test_dqn_residual_initialization(self):
        """Test DQNResidual model initialization"""
        model = DQNResidual(input_dim=770, output_dim=3, dropout=0.2)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_dqn_residual_forward(self):
        """Test DQNResidual forward pass"""
        model = DQNResidual(input_dim=770, output_dim=3, dropout=0.2)
        input_tensor = torch.randn(1, 770)
        output = model(input_tensor)
        assert output.shape == (1, 3)

    def test_dqn_residual_single_input(self):
        """Test DQNResidual with single input (no batch dimension)"""
        model = DQNResidual(input_dim=770, output_dim=3)
        input_tensor = torch.randn(770)
        output = model(input_tensor)
        assert output.shape[1] == 3


class TestVectorizer:
    """Tests for text vectorizer"""

    @pytest.mark.skip(reason="Requires BERT model download")
    def test_vectorizer_initialization(self):
        """Test vectorizer initialization"""
        vectorizer = BaseVectorizer()
        assert vectorizer is not None

    @pytest.mark.skip(reason="Requires BERT model download")
    def test_vectorize_single_text(self):
        """Test vectorizing a single text"""
        vectorizer = BaseVectorizer()
        text = "This is a test article about fake news."
        vector = vectorizer.vectorize(text)
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (768,)

    @pytest.mark.skip(reason="Requires BERT model download")
    def test_vectorize_batch(self):
        """Test vectorizing multiple texts"""
        vectorizer = BaseVectorizer()
        texts = ["First test article", "Second test article"]
        vectors = vectorizer.vectorize(texts)
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (2, 768)
