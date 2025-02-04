![header](https://capsule-render.vercel.app/api?type=waving&color=0:00008B,50:4169E1,100:1E90FF&height=300&section=header&text=FakeLenseV2&fontColor=00008B&fontSize=110&fontAlignY=40&fontAlign=50&animation=fadeIn&fontStyle=stroke)

<div align="center">

# 📰🔍 FakeLense V2: Fake News Detection Model Integrating NLP and Reinforcement Learning
</div>

## 1. Introduction
Fake news and disinformation have become pervasive threats to societies, shaping public opinion, influencing political discourse, and eroding trust in credible information sources. The rapid evolution of misinformation tactics necessitates **adaptive and robust detection mechanisms** that go beyond traditional machine learning approaches. While conventional supervised models have demonstrated effectiveness, they often struggle with **generalizing to unseen misinformation patterns** and lack the ability to dynamically adapt to new forms of deceptive content.

To address these challenges, **FakeLenseV2** introduces an AI-powered fake news detection framework that integrates Natural Language Processing (NLP) and Deep Reinforcement Learning (DRL) to enhance classification accuracy, adaptability, and robustness. Unlike static classifiers, FakeLenseV2 iteratively refines its decision-making process, ensuring superior resilience against evolving misinformation strategies.

**FakeLenseV1** was an **NLP-driven fake news detection model** designed to identify and prevent the proliferation of disinformation. By leveraging **BERT for deep text comprehension** and **GPT for generative insights**, it achieved high accuracy in detecting deceptive content while dynamically adapting to shifting misinformation trends.

🔗 **FakeLenseV1 GitHub Repository:** [FakeLenseV1](https://github.com/Navy10021/FakeLense)

Building on this foundation, **FakeLenseV2** integrates **reinforcement learning (RL)** methodologies to **further enhance detection capabilities**. Unlike its predecessor, FakeLenseV2 employs:
  - ✅ **BERT-based embeddings** for robust text representation.
  - ✅ **Deep Q-Networks (DQN)** with **residual learning** to refine classification strategies dynamically.
  - ✅ **An adaptive reward mechanism** to improve long-term learning efficiency.

By leveraging RL-driven adaptive learning, **FakeLenseV2** dynamically adjusts to new disinformation patterns, optimizes its classification policy, and strengthens its resilience against adversarial manipulation. This results in higher accuracy, improved generalization, and enhanced real-time misinformation detection—marking a significant leap forward in AI-driven fact-checking technologies.

## 2. Key Features
### 2.1 NLP-Based Text Representation
  - Leverages **BERT** and **RoBERTa** to generate contextual embeddings for robust feature extraction.
  - Incorporates **semantic understanding** of news content, surpassing traditional bag-of-words or TF-IDF methods.
  - Enables **fine-grained text analysis**, capturing nuanced language used in deceptive content.
    
### 2.2 Deep Reinforcement Learning for Adaptive Classification
  - Implements a **Deep Q-Network (DQN)** with residual connections to improve training stability.
  - Uses **Double DQN (DDQN)** to mitigate the overestimation bias in Q-learning.
  - Employs **target network smoothing**, reducing volatility in the learning process.
  - **Reward shaping** mechanism incentivizes correct classifications and penalizes overconfidence in incorrect predictions.
    
### 2.3 Source Credibility and Social Reactions Analysis
  - Assigns a **source reliability score** based on trust-worthiness rankings of news agencies.
  - Integrates **social engagement metrics** (e.g., number of shares, likes) to quantify public reception of news articles.
  - Enhances classification performance by considering external credibility factors.
    
### 2.4 Real-Time Inference and Scalability
  - Optimized for **GPU acceleration**, allowing **efficient real-time detection**.
  - Supports batch inference, making it suitable for large-scale misinformation monitoring.
  - Can be deployed as an API for integration into social media platforms and fact-checking systems.

## 3. System Architecture
FakeLenseV2 consists of three primary components:
### 3.1 Feature Extraction
  - **Text Tokenization & Embedding**: Uses BERT or RoBERTa to encode textual content into a dense vector representation. Supports pooling mechanisms to derive sentence-level embeddings.
  - **Source Credibility Encoding**: Maps news sources to predefined reliability scores based on media bias ratings. Assigns higher trust scores to reputable sources (e.g., Reuters, BBC) and lower scores to sensationalist outlets.
  - **Social Reaction Normalization**: Converts engagement metrics (likes, shares, comments) into a normalized scale.
    
### 3.2 Reinforcement Learning-based Classifier
  - **Deep Q-Network (DQN) with Residual Learning**:
    1) Uses a three-layer neural network with residual connections to prevent information loss.
    2) Implements Layer Normalization to stabilize training.
  - **Action Space & Rewards**:
    1) Action Space: Assigns labels {Real (2), Suspicious (1), Fake (0)}.
    2) Reward Mechanism:
        Correct classification → +1 reward
        Incorrect classification → Negative penalty proportional to confidence
        High-confidence incorrect prediction → Larger penalty to discourage overfitting
  - **Double DQN & Target Network Smoothing**:
    1) Uses a target Q-network updated via soft target updates (τ = 0.005).
    2) Reduces fluctuations in Q-values, ensuring smooth convergence.
       
### 3.3 Evaluation & Performance Metrics
  - Standard Classification Metrics: Accuracy, Precision, Recall, F1-score
  - Confusion Matrix Analysis: Provides insights into misclassification patterns.
  - Ablation Studies: Evaluates the impact of BERT embeddings, residual learning, and reward shaping.
    
## 4. Installation & Dependencies
### 4.1 Prerequisites
  - Python 3.8+
  - PyTorch 1.12+
  - Hugging Face Transformers
  - NumPy, Matplotlib, Seaborn, Scikit-learn
    
### 4.2 Setup
 ```bash
   git clone https://github.com/Navy10021/FakeLenseV2.git
   cd FakeLenseV2
   ```

## 5. Usage
### 5.1 Training the Model
 ```python
from model import train_agent
train_agent(num_episodes=500, patience=15)

   ```

### 5.2 Performing Inference
 ```python
from model import infer

text = "Breaking news: Scientists discover a new planet with signs of life."
source = "CNN"
social_reactions = 12000  # Example engagement count

result = infer(text, source, social_reactions)
print(f"Prediction: {result}")  # Output: {2: Real, 1: Suspicious, 0: Fake}

   ```

## 6. Evaluation
### 6.1 Performance Metrics
 ```python
from model import eval_agent
eval_agent()

   ```

### 6.2 Benchmarking Results

### 6.3 Fake News Detection Performance Evaluation Results
The experimental results demonstrated a **high detection accuracy of over 97%**, proving the tool's effectiveness in identifying fake news. **FakeLenseV2** is expected to serve as an innovative "cognitive warfare" tool, capable of addressing misinformation across various channels and contributing to national interests.

## 👨‍💻 Contributors
- **Seoul National University Graduate School of Data Science (SNU GSDS)**
- Under the guidance of ***Navy Lee***

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
