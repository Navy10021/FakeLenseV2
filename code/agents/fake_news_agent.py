"""Reinforcement learning agent for fake news detection"""

import random
from collections import deque
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from code.models.dqn import DQN, DQNResidual


class FakeNewsAgent:
    """
    Deep Q-Learning agent for fake news detection.

    Implements:
        - Double DQN to reduce Q-value overestimation
        - Target network with soft updates
        - Experience replay buffer
        - Epsilon-greedy exploration
        - Reward shaping for better learning
    """

    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any]):
        """
        Initialize the FakeNewsAgent.

        Args:
            state_size: Dimension of the state space (feature vector size)
            action_size: Number of possible actions (3: Fake, Suspicious, Real)
            config: Configuration dictionary with hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Experience replay memory
        self.memory = deque(maxlen=config.get("memory_size", 2000))

        # Initialize networks
        use_residual = config.get("use_residual", False)
        dropout = config.get("dropout", 0.2)

        if use_residual:
            self.model = DQNResidual(state_size, action_size, dropout).to(self.device)
            self.target_model = DQNResidual(state_size, action_size, dropout).to(self.device)
        else:
            self.model = DQN(state_size, action_size).to(self.device)
            self.target_model = DQN(state_size, action_size).to(self.device)

        # Copy weights to target model
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 0.0005)
        )
        self.criterion = nn.MSELoss()

        # Training hyperparameters
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.update_target_freq = config.get("update_target_freq", 10)
        self.tau = config.get("tau", 0.005)

        # Reward shaping parameters
        self.reward_penalty_confident_wrong = config.get("reward_penalty_confident_wrong", 2.0)
        self.reward_penalty_correct = config.get("reward_penalty_correct", 0.5)

        # Training state
        self.step_count = 0

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store a transition in the replay memory.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: Union[np.ndarray, torch.Tensor]) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action (0: Fake, 1: Suspicious, 2: Real)
        """
        # Exploration: random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation: best action according to Q-network
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)

        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state)

        self.model.train()
        return int(torch.argmax(q_values).item())

    def replay(self) -> None:
        """
        Train the model using experience replay with Double DQN.

        Implements:
            - Random sampling from replay buffer
            - Double DQN target computation
            - Confidence-based reward shaping
            - Gradient descent optimization
            - Epsilon decay
            - Soft target network updates

        Notes:
            - Requires at least batch_size samples in memory
            - Updates target network every update_target_freq steps
            - Applies configurable reward penalties based on prediction confidence
        """
        # Not enough samples in memory
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        actions = list(actions)

        # Current Q-values
        q_values = self.model(states)

        # Double DQN: Use current model to select action, target model to evaluate
        with torch.no_grad():
            next_q_values = self.model(next_states)
            next_q_target = self.target_model(next_states)

        # Compute target Q-values
        q_target = q_values.clone()
        for i in range(self.batch_size):
            if dones[i]:
                q_target[i, actions[i]] = rewards[i]
            else:
                # Double DQN update
                best_action = torch.argmax(next_q_values[i]).item()
                q_target[i, actions[i]] = rewards[i] + self.gamma * next_q_target[i, best_action]

        # Reward shaping: Apply confidence-based penalty
        for i in range(self.batch_size):
            confidence = torch.max(q_values[i]).item()
            if not dones[i]:
                if torch.argmax(q_values[i]).item() != actions[i]:
                    # Higher penalty for confident wrong predictions
                    reward_penalty = self.reward_penalty_confident_wrong * confidence
                else:
                    # Smaller penalty for correct predictions
                    reward_penalty = self.reward_penalty_correct * confidence
                q_target[i, actions[i]] -= reward_penalty

        # Compute loss and update
        loss = self.criterion(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self._soft_update()

    def _soft_update(self) -> None:
        """
        Soft update of target network parameters.

        θ_target = τ * θ_model + (1 - τ) * θ_target
        """
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str) -> None:
        """
        Save the model weights.

        Args:
            path: Path to save the model
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load the model weights.

        Args:
            path: Path to load the model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
