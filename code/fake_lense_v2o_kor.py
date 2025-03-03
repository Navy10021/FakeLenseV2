import os
import json
import random
import logging
from collections import deque
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =====================================
# STEP 0: 텍스트 벡터화 (BERT/RoBERTa)
# =====================================
class BaseVectorizer:
    """
    벡터화 클래스의 기본 템플릿.
    BERT나 RoBERTa와 같은 사전 학습된 모델을 이용해 입력 텍스트를 벡터로 변환한다.
    """
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)

    def vectorize(self, text: Union[str, List[str]], pooling: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        주어진 텍스트를 벡터화한다.
        :param text: 단일 문자열 또는 문자열 리스트
        :param pooling: 마지막 은닉 상태의 평균 풀링 여부
        :return: 텍스트 벡터 (numpy array 또는 torch.Tensor)
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if pooling:
            return outputs.last_hidden_state.mean(dim=1).cpu()
        # [CLS] 토큰의 벡터
        result = outputs.last_hidden_state[:, 0, :]
        if isinstance(text, list):
            return result.cpu().numpy()
        else:
            return result.squeeze().cpu().numpy()


# ======================================
# STEP 1: 특징 추출 (Feature Extraction)
# ======================================
class FeatureExtractor:
    """
    텍스트, 미디어 신뢰도, 소셜 반응과 같은 메타데이터를 결합하여 특징 벡터를 생성.
    """
    def __init__(self, vectorizer: BaseVectorizer = None):
        # 기본적으로 BERT 벡터라이저 사용 (필요시 RoBERTa 등으로 교체 가능)
        self.vectorizer = vectorizer if vectorizer is not None else BaseVectorizer()

    @staticmethod
    def convert_source_reliability(source_reliability: str) -> float:
        """
        미디어 신뢰도를 실제 수치(float)로 변환.
        신뢰도 매핑: 미디어별 실제 신뢰도 수치 기준.
        """
        reliability_mapping = {
            "The New York Times": 0.90,
            "The Washington Post": 0.85,
            "CNN": 0.80,
            "BBC": 0.85,
            "NPR": 0.90,
            "Reuters": 0.90,
            "The Wall Street Journal": 0.85,
            "USA Today": 0.75,
            "Fox News": 0.60,
            "Bloomberg": 0.85,
            "The Guardian": 0.80,
            "Los Angeles Times": 0.80,
            "New York Post": 0.60,
            "HuffPost": 0.70,
            "Associated Press": 0.90
        }
        return reliability_mapping.get(source_reliability, 0.50)

    def extract_features(self, text: str, source_reliability: str, social_reactions: float) -> np.ndarray:
        """
        텍스트 벡터와 추가 메타데이터(신뢰도, 소셜 반응)를 결합.
        :param text: 기사 본문
        :param source_reliability: 기사 출처
        :param social_reactions: 소셜 미디어 반응 수
        :return: 결합된 특징 벡터 (numpy array)
        """
        reliability_score = self.convert_source_reliability(source_reliability)
        text_vector = self.vectorizer.vectorize(text)
        # 소셜 반응을 정규화 (예: 10000으로 나누기)
        return np.concatenate([text_vector, [reliability_score], [social_reactions / 10000]], axis=0)


# ==================================================
# STEP 2: DQN 및 개선된 모델 (Neural Network Models)
# ==================================================
class DQN(nn.Module):
    """
    기본 DQN 모델.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNResidual(nn.Module):
    """
    개선된 DQN 모델 (Residual connection, Layer Normalization, Dropout 포함).
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super(DQNResidual, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        # Residual 연결을 위한 선형 레이어 (입력 차원 -> 64)
        self.residual_fc = nn.Linear(input_dim, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_fc(x if x.dim() == 2 else x.unsqueeze(0))
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x) + residual)  # Residual 연결
        return self.fc4(x)


# ===========================================
# STEP 3: 강화학습 에이전트 (Fake News Agent)
# ===========================================
class FakeNewsAgent:
    """
    강화학습 기반 가짜뉴스 탐지 에이전트.
    Double DQN, Target Network Smoothing, Reward Shaping 등을 포함.
    """
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any]):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=config.get("memory_size", 2000))
        # 옵션에 따라 기본 DQN 또는 Residual DQN 선택
        use_residual = config.get("use_residual", False)
        if use_residual:
            self.model = DQNResidual(state_size, action_size).to(self.device)
            self.target_model = DQNResidual(state_size, action_size).to(self.device)
        else:
            self.model = DQN(state_size, action_size).to(self.device)
            self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get("learning_rate", 0.0005))
        self.criterion = nn.MSELoss()
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.update_target_freq = config.get("update_target_freq", 10)
        self.step_count = 0
        self.tau = config.get("tau", 0.005)  # Target Network Smoothing

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: Union[np.ndarray, torch.Tensor]) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state if state.dim() == 2 else state.unsqueeze(0))
        return int(torch.argmax(q_values).item())

    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Double DQN 업데이트
        q_values = self.model(states)
        next_q_values = self.model(next_states)
        next_q_target = self.target_model(next_states).detach()

        q_target = q_values.clone()
        for i in range(self.batch_size):
            if dones[i]:
                q_target[i, actions[i]] = rewards[i]
            else:
                best_action = torch.argmax(next_q_values[i]).item()
                q_target[i, actions[i]] = rewards[i] + self.gamma * next_q_target[i, best_action]

        # Reward Shaping: 에이전트의 예측 확신도를 기반으로 페널티 부여
        for i in range(self.batch_size):
            confidence = torch.max(q_values[i]).item()
            if not dones[i]:
                if torch.argmax(q_values[i]).item() != actions[i]:
                    reward_penalty = 2 * confidence
                else:
                    reward_penalty = 0.5 * confidence
                q_target[i, actions[i]] -= reward_penalty

        loss = self.criterion(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 탐험-활용 균형을 위한 epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 타깃 네트워크 소프트 업데이트
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.soft_update(self.model, self.target_model)

    def soft_update(self, model: nn.Module, target_model: nn.Module) -> None:
        """
        타깃 네트워크의 파라미터를 소프트 업데이트.
        """
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


# ==================================================
# STEP 4: 트레이너 클래스 (Training and Evaluation)
# ==================================================
class Trainer:
    """
    에이전트의 학습, 검증, 평가를 총괄하는 클래스.
    """
    def __init__(self, agent: FakeNewsAgent, feature_extractor: FeatureExtractor, config: Dict[str, Any]):
        self.agent = agent
        self.feature_extractor = feature_extractor
        self.config = config

    def train(self, data: List[Dict[str, Any]], num_episodes: int = 500, patience: int = 15) -> None:
        best_reward = -float("inf")
        no_improvement = 0
        max_possible_reward = len(data)
        reward_history = []

        for episode in range(num_episodes):
            total_reward = 0
            for sample in data:
                features = self.feature_extractor.extract_features(
                    sample["text"],
                    sample["source_reliability"],
                    sample["social_reactions"]
                )
                state = features
                action = self.agent.act(state)
                reward = 1 if action == sample["label"] else -1
                self.agent.remember(state, action, reward, state, False)
                self.agent.replay()
                total_reward += reward

            normalized_reward = (total_reward / max_possible_reward) * 100
            reward_history.append(normalized_reward)
            logging.info(f"Episode {episode + 1:03d} - Total Reward: {total_reward} ({normalized_reward:.2f}/100)")

            if total_reward > best_reward:
                best_reward = total_reward
                no_improvement = 0
                if not os.path.exists('models'):
                    os.makedirs('models')
                torch.save(self.agent.model.state_dict(), "./models/best_model.pth")
            else:
                no_improvement += 1
            if no_improvement >= patience:
                logging.info("Early stopping triggered.")
                break

        # 학습 곡선 시각화
        plt.figure(figsize=(8, 5))
        plt.plot(reward_history, label="Normalized Reward")
        plt.xlabel("Episode")
        plt.ylabel("Normalized Reward")
        plt.title("Training Reward Curve")
        plt.legend()
        plt.show()

    def infer(self, text: str, source_reliability: str, social_reactions: float) -> int:
        """
        단일 샘플에 대해 추론을 수행.
        """
        features = self.feature_extractor.extract_features(text, source_reliability, social_reactions)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.agent.device)
        self.agent.model.load_state_dict(torch.load('./models/best_model.pth', map_location=self.agent.device))
        self.agent.model.eval()
        with torch.no_grad():
            action = self.agent.act(features_tensor)
        return action

    def evaluate(self, data: List[Dict[str, Any]]) -> None:
        """
        테스트 데이터셋에 대해 에이전트를 평가하고 성능 지표와 혼동 행렬을 출력.
        """
        y_true = []
        y_pred = []

        logging.info("\n========== ON TEST DATASET ==========")
        for sample in data:
            prediction = self.infer(
                sample["text"],
                sample["source_reliability"],
                sample["social_reactions"]
            )
            y_true.append(sample["label"])
            y_pred.append(prediction)
            pred_str = {0: "Fake News", 1: "Suspicious News", 2: "Real News"}.get(prediction, "Unknown")
            logging.info(f"Article: {sample['text']}\nPrediction: {pred_str}\n")

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        logging.info("\n========== Performance Metrics ==========")
        logging.info(f"Accuracy: {acc:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        print("\nClassification Report:\n", classification_report(y_true, y_pred))

        # 혼동 행렬 시각화
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Fake News", "Suspicious News", "Real News"],
                    yticklabels=["Fake News", "Suspicious News", "Real News"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()


# ====================================
# STEP 5: 메인 함수 (실행 파이프라인)
# ====================================
def main():
    """
    학습 및 평가 데이터의 형식 예시:
    {
        "text": "The federal government announces new regulations to ensure AI ethics in law enforcement.",
        "source_reliability": "The New York Times",
        "social_reactions": 6200,
        "label": 1
    }
    """
    # 구성 설정
    config = {
        "memory_size": 2000,
        "learning_rate": 0.0005,
        "batch_size": 64,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "update_target_freq": 10,
        "tau": 0.005,
        "use_residual": True  # Residual 연결 모델 사용 여부
    }

    # 데이터 불러오기
    with open("./data/train_data.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("./data/test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 상태 및 액션 차원 설정
    state_size = 768 + 2  # BERT 벡터 차원 + (source reliability, social reactions)
    action_size = 3       # Fake, Suspicious, Real

    # 구성된 벡터화 및 특징 추출기 생성
    vectorizer = BaseVectorizer(model_name="bert-base-uncased")
    feature_extractor = FeatureExtractor(vectorizer=vectorizer)

    # 에이전트 초기화
    agent = FakeNewsAgent(state_size, action_size, config)

    # 트레이너 객체 생성
    trainer = Trainer(agent, feature_extractor, config)

    # 학습 진행
    trainer.train(train_data, num_episodes=500, patience=15)

    # 평가 진행
    trainer.evaluate(test_data)


if __name__ == "__main__":
    main()
