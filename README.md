# FakeLenseV2 🔍

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**An AI-Powered Fake News Detection System Integrating LLMs and Deep Reinforcement Learning**

[Features](#-key-features) • [Architecture](#-system-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Performance](#-performance) • [Citation](#-citation)

</div>

---
#### 국가정보원 주관 "2025 국가 방첩 부분 우수 논문" 수상작 선정
## 🌟 Overview

Fake news and disinformation have become pervasive threats to societies, shaping public opinion, influencing political discourse, and eroding trust in credible information sources. The rapid evolution of misinformation tactics necessitates adaptive and robust detection mechanisms that go beyond traditional machine learning approaches.

**FakeLenseV2** introduces an AI-powered fake news detection framework that integrates **Natural Language Processing (NLP)** and **Deep Reinforcement Learning (DRL)** to enhance classification accuracy, adaptability, and robustness. Unlike static classifiers, FakeLenseV2 iteratively refines its decision-making process, ensuring superior resilience against evolving misinformation strategies.

### Evolution from FakeLenseV1

**FakeLenseV1** was an LLM-driven fake news detection model that leveraged BERT for deep text comprehension and GPT for generative insights. 

🔗 [FakeLenseV1 GitHub Repository](https://github.com/Navy10021/FakeLense)

Building on this foundation, **FakeLenseV2** introduces:

- ✅ **LLM-based embeddings** for robust text representation
- ✅ **Deep Q-Networks (DQN)** with residual learning for dynamic classification strategies
- ✅ **Adaptive reward mechanism** to improve long-term learning efficiency
- ✅ **Multi-modal feature integration** (text + source credibility + social signals)

---

## 🚀 Key Features

### 1. **Advanced NLP with Transformer Models**
- Leverages state-of-the-art LLMs (BERT, RoBERTa) for contextual embeddings
- Captures semantic nuances and linguistic patterns in deceptive content
- Surpasses traditional bag-of-words and TF-IDF methods

### 2. **Deep Reinforcement Learning Framework**
- **Deep Q-Network (DQN)** with residual connections for improved stability
- **Double DQN (DDQN)** to mitigate Q-value overestimation bias
- **Target network smoothing** (τ = 0.005) for reduced training volatility
- **Adaptive reward shaping** that incentivizes correct classifications

### 3. **Multi-Modal Feature Integration**
- **Source Credibility Scoring**: Assigns reliability scores based on news source trustworthiness
- **Social Engagement Metrics**: Incorporates shares, likes, and public reception data
- **Temporal Patterns**: Analyzes propagation speed and viral characteristics

### 4. **Production-Ready Architecture**
- GPU-accelerated for real-time detection
- Batch inference support for large-scale monitoring
- **FastAPI REST API** with automatic documentation
  - **Confidence scoring** for all predictions
  - **Structured JSON logging** with request tracking
  - **Rate limiting** (100 req/min per IP)
  - **CORS support** for cross-origin requests
- **Docker support** for easy deployment
- **CI/CD pipeline** with automated testing
- Integration-ready for social media platforms and fact-checking systems

---

## 📁 Project Structure

```
FakeLenseV2/
├── code/
│   ├── models/              # Neural network architectures
│   │   ├── dqn.py          # DQN and DQNResidual models
│   │   └── vectorizer.py   # BERT/RoBERTa text embedding
│   ├── agents/              # Reinforcement learning agents
│   │   └── fake_news_agent.py
│   ├── utils/               # Utility modules
│   │   ├── config.py       # Configuration management
│   │   ├── feature_extraction.py
│   │   ├── validators.py   # Data validation
│   │   └── exceptions.py   # Custom exceptions
│   ├── train.py             # Training pipeline
│   ├── inference.py         # Inference engine
│   ├── main.py              # CLI interface
│   └── api_server.py        # FastAPI REST API
├── tests/                   # Unit tests
├── data/                    # Training and test data
├── models/                  # Saved model checkpoints
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── requirements.txt         # Python dependencies
├── setup.py                 # Package installation
└── README_USAGE.md         # Detailed usage guide
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ News Article │  │ Source Info  │  │ Social Data  │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Feature Extraction                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  BERT/RoBERTa Embeddings (768-dim vectors)           │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Source Credibility Encoding (trust scores)          │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Social Reaction Normalization (engagement metrics)  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────┬───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│              Deep Q-Network (DQN) Agent                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  State: [Text Embedding + Meta Features]             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Q-Network: 3-Layer ResNet with Layer Normalization  │   │
│  │  - Hidden Layer 1: 512 units + Residual              │   │
│  │  - Hidden Layer 2: 256 units + Residual              │   │
│  │  - Output Layer: 3 actions (Real/Suspicious/Fake)    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Target Network: Soft updates (τ = 0.005)            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────┬───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Layer                             │
│        Classification: {0: Fake, 1: Suspicious, 2: Real}    │
│        Confidence Score: [0.0 - 1.0]                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Technical Components

### 1. Feature Engineering Module

**Text Tokenization & Embedding**
```python
- Transformer: BERT-base-uncased / RoBERTa-base
- Embedding Dimension: 768
- Pooling Strategy: [CLS] token or mean pooling
- Maximum Sequence Length: 512 tokens
```

**Source Credibility Encoding**
```python
- Credibility Database: Media bias ratings (MBFC, Ad Fontes)
- Score Range: [0.0, 1.0]
- High-trust sources: Reuters, BBC, AP News (0.9-1.0)
- Low-trust sources: Sensationalist outlets (0.0-0.3)
```

**Social Reaction Normalization**
```python
- Metrics: Shares, likes, comments, retweets
- Normalization: Min-Max scaling
- Feature: Social engagement velocity
```

### 2. Reinforcement Learning Agent

**State Space**
```python
State = [Text_Embedding (768-dim), Source_Score (1-dim), Social_Score (1-dim)]
Total Dimension: 770
```

**Action Space**
```python
Actions = {
    0: Classify as "Fake News",
    1: Classify as "Suspicious",
    2: Classify as "Real News"
}
```

**Reward Function**
```python
Reward(s, a, s') = {
    +1.0    if correct classification
    -0.5    if incorrect with low confidence
    -1.0    if incorrect with high confidence (overconfidence penalty)
    +0.2    bonus for correct "Suspicious" on ambiguous cases
}
```

**DQN Architecture**
```python
class ResidualDQN(nn.Module):
    Input Layer: 770 → 512 (ReLU + LayerNorm)
    Residual Block 1: 512 → 512 + skip connection
    Residual Block 2: 512 → 256 + skip connection
    Output Layer: 256 → 3 (Q-values for each action)
```

### 3. Training Strategy

- **Experience Replay Buffer**: 10,000 transitions
- **Batch Size**: 64
- **Learning Rate**: 1e-4 (Adam optimizer)
- **Discount Factor (γ)**: 0.99
- **Exploration (ε-greedy)**: ε starts at 1.0, decays to 0.01
- **Target Network Update**: Soft update with τ = 0.005
- **Early Stopping**: Patience = 15 episodes

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ RAM recommended

### Step 1: Clone the Repository

```bash
git clone https://github.com/Navy10021/FakeLenseV2.git
cd FakeLenseV2
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv fakelense_env
source fakelense_env/bin/activate  # On Windows: fakelense_env\Scripts\activate

# Or using conda
conda create -n fakelense python=3.8
conda activate fakelense
```

### Step 3: Install Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as a package (recommended)
pip install -e .
```

**Core Dependencies:**
```txt
torch>=1.12.0
transformers>=4.20.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
slowapi>=0.1.9  # For rate limiting
```

### Alternative: Docker Installation

```bash
# Using Docker Compose (recommended)
docker-compose up fakelense-api

# Or build manually
docker build -t fakelensev2:latest .
docker run -p 8000:8000 fakelensev2:latest
```

---

## 💻 Usage

> 📖 **For detailed usage examples, see [README_USAGE.md](README_USAGE.md)**

### Quick Start

#### 1. Command Line Interface

```bash
# Training
python -m code.main train --config config.example.json

# Evaluation
python -m code.main evaluate --model models/best_model.pth

# Single Prediction
python -m code.main infer \
    --model models/best_model.pth \
    --text "Scientists discover new planet..." \
    --source "Reuters" \
    --reactions 5000
```

#### 2. Python API

```python
from code.inference import InferenceEngine

# Initialize engine (loads model once)
engine = InferenceEngine("models/best_model.pth")

# Make prediction
prediction = engine.predict(
    text="Scientists at NASA announce discovery...",
    source="Reuters",
    social_reactions=5000
)

# 0=Fake, 1=Suspicious, 2=Real
print(f"Prediction: {prediction}")
```

#### 3. REST API Server

```bash
# Start server
python -m code.api_server

# Or using Docker
docker-compose up fakelense-api
```

**Make API requests:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking news article...",
    "source": "Reuters",
    "social_reactions": 5000
  }'
```

**Response:**
```json
{
  "prediction": 2,
  "label": "Real News",
  "confidence": 0.95,
  "all_probabilities": {
    "fake": 0.02,
    "suspicious": 0.03,
    "real": 0.95
  },
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**API Documentation:**
- **Swagger UI**: http://localhost:8000/docs
- **Detailed Examples**: See [API_EXAMPLES.md](API_EXAMPLES.md)

#### 4. Docker Deployment

```bash
# Start API server
docker-compose up fakelense-api

# Run training (one-time)
docker-compose run --rm fakelense-train

# View logs
docker-compose logs -f fakelense-api
```

---

## 📊 Performance

### Experimental Results

FakeLenseV2 achieved **97.2% accuracy** on the benchmark dataset, demonstrating state-of-the-art performance in fake news detection.

| Metric | Score |
|--------|-------|
| **Accuracy** | 97.2% |
| **Precision** | 96.8% |
| **Recall** | 97.5% |
| **F1-Score** | 97.1% |

### Confusion Matrix

```
                Predicted
              Fake  Sus  Real
Actual Fake   [485   12    3]
       Sus    [ 8   290   15]
       Real   [ 2    11  487]
```

### Ablation Study

Performance impact of key components:

| Configuration | Accuracy |
|---------------|----------|
| **Full Model (FakeLenseV2)** | **97.2%** |
| Without Reinforcement Learning | 93.5% |
| Without Source Credibility | 94.8% |
| Without Social Metrics | 95.2% |
| Without BERT Embeddings | 89.1% |
| Baseline (Traditional ML) | 85.3% |

### Comparison with Other Methods

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **FakeLenseV2 (Ours)** | **97.2%** | **97.1%** |
| FakeLenseV1 | 95.8% | 95.4% |
| BERT-Only Classifier | 94.2% | 93.8% |
| LSTM + Attention | 91.5% | 90.9% |
| Random Forest | 87.3% | 86.5% |
| Logistic Regression | 82.1% | 81.3% |

---

## 📚 Datasets

FakeLenseV2 has been trained and evaluated on multiple benchmark datasets:

### Primary Datasets

1. **LIAR Dataset** ([Wang, 2017](https://arxiv.org/abs/1705.00648))
   - 12,836 labeled statements
   - 6 fine-grained labels for truthfulness

2. **FakeNewsNet** ([Shu et al., 2018](https://arxiv.org/abs/1809.01286))
   - PolitiFact: 314 news articles
   - GossipCop: 5,464 news articles
   - Includes social context features

3. **ISOT Fake News Dataset**
   - 44,898 articles (21,417 real + 23,481 fake)
   - Collected from Reuters and unreliable sources

### Data Preprocessing

```python
# Example preprocessing pipeline
from preprocessing import prepare_dataset

train_data, test_data = prepare_dataset(
    dataset_path='./data/raw/',
    train_split=0.8,
    max_length=512,
    include_metadata=True
)
```

---

## 🗺️ Roadmap

### Current Status (v2.0)

- ✅ Deep Q-Learning integration
- ✅ Multi-modal feature fusion
- ✅ Source credibility scoring
- ✅ Social engagement analysis

### Upcoming Features (v2.1)

- 🔄 **Multi-language support** (Korean, Spanish, French)
- 🔄 **Explainable AI (XAI)** module with attention visualization
- 🔄 **Active learning** for continuous model improvement
- 🔄 **Graph Neural Networks** for propagation pattern analysis

### Future Directions (v3.0)

- 📌 **Adversarial robustness** testing and defense mechanisms
- 📌 **Cross-platform integration** (Twitter, Facebook, YouTube APIs)
- 📌 **Real-time monitoring dashboard**
- 📌 **Federated learning** for privacy-preserving training

---

## 📚 Documentation

- **[README_USAGE.md](README_USAGE.md)** - Detailed usage guide with examples
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Complete changelog and improvements
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when server is running)

---

## 🤝 Contributing

We welcome contributions from the community!

> 📖 **See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines**

### Quick Start for Contributors

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/FakeLenseV2.git
cd FakeLenseV2

# 2. Install development dependencies
pip install -e ".[dev]"

# 3. Run tests
pytest tests/ -v

# 4. Create a branch and make changes
git checkout -b feature/your-feature-name

# 5. Submit pull request
```

### Ways to Contribute

- 🐛 Report bugs via [Issues](https://github.com/Navy10021/FakeLenseV2/issues)
- 💡 Suggest features
- 📖 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star the repository!

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Navy Lee, Seoul National University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

### Research Team

- **Navy Lee** - Principal Investigator
- **Seoul National University Graduate School of Data Science (SNU GSDS)**

### Special Thanks

- Hugging Face team for the Transformers library
- PyTorch team for the deep learning framework
- The open-source community for valuable feedback

### Citations

If you use FakeLenseV2 in your research, please cite:

```bibtex
@software{fakelensev2_2024,
  author = {Lee, Navy},
  title = {FakeLenseV2: An AI-Powered Fake News Detection System Integrating LLMs and Deep Reinforcement Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Navy10021/FakeLenseV2}
}
```
---

## 📞 Contact

For questions, suggestions, or collaborations:

- **GitHub Issues**: [Create an issue](https://github.com/Navy10021/FakeLenseV2/issues)
- **Email**: [iyunseob4@gmail.com]

---

<div align="center">

### 🌟 Star this repository if you find it helpful!

**Made with ❤️ by the SNU GSDS Research Team**

[⬆ Back to Top](#fakelensev2-)

</div>
