"""Unit tests for agent modules"""

import pytest
import numpy as np

from code.agents.fake_news_agent import FakeNewsAgent
from code.utils.config import get_default_config


class TestFakeNewsAgent:
    """Tests for FakeNewsAgent"""

    def test_agent_initialization(self):
        """Test agent initialization"""
        config = get_default_config()
        agent = FakeNewsAgent(state_size=770, action_size=3, config=config)
        assert agent is not None
        assert agent.state_size == 770
        assert agent.action_size == 3

    def test_remember(self):
        """Test storing transitions in memory"""
        config = get_default_config()
        agent = FakeNewsAgent(state_size=770, action_size=3, config=config)

        state = np.random.randn(770)
        action = 1
        reward = 1.0
        next_state = np.random.randn(770)
        done = False

        agent.remember(state, action, reward, next_state, done)
        assert len(agent.memory) == 1

    def test_act_exploration(self):
        """Test action selection during exploration"""
        config = get_default_config()
        config["epsilon"] = 1.0  # Always explore
        agent = FakeNewsAgent(state_size=770, action_size=3, config=config)

        state = np.random.randn(770)
        action = agent.act(state)
        assert action in [0, 1, 2]

    def test_act_exploitation(self):
        """Test action selection during exploitation"""
        config = get_default_config()
        config["epsilon"] = 0.0  # Never explore
        agent = FakeNewsAgent(state_size=770, action_size=3, config=config)

        state = np.random.randn(770)
        action = agent.act(state)
        assert action in [0, 1, 2]

    def test_replay_insufficient_memory(self):
        """Test replay with insufficient memory samples"""
        config = get_default_config()
        config["batch_size"] = 64
        agent = FakeNewsAgent(state_size=770, action_size=3, config=config)

        # Add only a few samples (less than batch size)
        for _ in range(10):
            state = np.random.randn(770)
            agent.remember(state, 1, 1.0, state, False)

        # Should not raise error
        agent.replay()
