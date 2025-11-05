"""Deep Q-Network (DQN) models for reinforcement learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Basic Deep Q-Network (DQN) model.

    Architecture:
        - Input layer -> Hidden layer (128) -> Hidden layer (64) -> Output layer
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the DQN model.

        Args:
            input_dim: Dimension of input features
            output_dim: Number of possible actions
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Q-values for each action
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNResidual(nn.Module):
    """
    Enhanced DQN model with residual connections, layer normalization, and dropout.

    Architecture:
        - Input layer -> Hidden layer (256) -> Hidden layer (128) -> Hidden layer (64) -> Output layer
        - Includes residual connection from input to the third hidden layer
        - Layer normalization after first two layers
        - Dropout for regularization
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        """
        Initialize the DQN Residual model.

        Args:
            input_dim: Dimension of input features
            output_dim: Number of possible actions
            dropout: Dropout rate for regularization (default: 0.2)
        """
        super(DQNResidual, self).__init__()

        # First layer
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(dropout)

        # Second layer
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(dropout)

        # Third layer
        self.fc3 = nn.Linear(128, 64)

        # Output layer
        self.fc4 = nn.Linear(64, output_dim)

        # Residual connection layer
        self.residual_fc = nn.Linear(input_dim, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network with residual connection.

        Args:
            x: Input tensor

        Returns:
            Q-values for each action
        """
        # Ensure input has batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Store for residual connection
        residual = self.residual_fc(x)

        # Forward pass through main layers
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)

        # Apply residual connection
        x = F.relu(self.fc3(x) + residual)

        return self.fc4(x)
