# FakeLenseV2 - Quick Start Guide

This guide covers the new modular architecture and usage patterns.

## 📦 Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/Navy10021/FakeLenseV2.git
cd FakeLenseV2

# Create virtual environment (recommended)
python -m venv fakelense_env
source fakelense_env/bin/activate  # On Windows: fakelense_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Docker Installation

```bash
# Build and run API server
docker-compose up fakelense-api

# Run training (one-time)
docker-compose run --rm fakelense-train
```

## 🚀 Quick Start

### 1. Training a Model

#### Command Line Interface
```bash
# Using default configuration
python -m code.main train

# Using custom configuration
python -m code.main train --config config.example.json
```

#### Python API
```python
from code.train import train_from_config

# Train with default settings
train_from_config()

# Train with custom config
train_from_config(config_path="my_config.json")
```

### 2. Making Predictions

#### Command Line Interface
```bash
python -m code.main infer \
    --model models/best_model.pth \
    --text "Scientists at NASA announce discovery of Earth-like planet." \
    --source "Reuters" \
    --reactions 5000
```

#### Python API
```python
from code.inference import InferenceEngine

# Initialize engine (loads model once)
engine = InferenceEngine("models/best_model.pth")

# Single prediction
prediction = engine.predict(
    text="Scientists at NASA announce discovery...",
    source="Reuters",
    social_reactions=5000
)

# prediction = 2 (Real News)
print(f"Prediction: {prediction}")  # 0=Fake, 1=Suspicious, 2=Real
```

#### Batch Predictions
```python
articles = [
    {"text": "Article 1...", "source_reliability": "CNN", "social_reactions": 1000},
    {"text": "Article 2...", "source_reliability": "Reuters", "social_reactions": 5000}
]

predictions = engine.predict_batch(articles)
```

### 3. Evaluation

#### Command Line
```bash
python -m code.main evaluate \
    --model models/best_model.pth \
    --config config.example.json
```

#### Python API
```python
from code.inference import evaluate_from_config

# Evaluate on test set
metrics = evaluate_from_config(
    model_path="models/best_model.pth"
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

## 🌐 REST API Server

### Starting the Server

#### Using Python
```bash
python -m code.api_server
# Server runs on http://localhost:8000
```

#### Using Docker
```bash
docker-compose up fakelense-api
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Scientists discover new planet with potential for life.",
    "source": "Reuters",
    "social_reactions": 5000
  }'
```

Response:
```json
{
  "prediction": 2,
  "label": "Real News",
  "confidence": null
}
```

#### Batch Prediction
```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "text": "Article 1...",
      "source": "CNN",
      "social_reactions": 1000
    },
    {
      "text": "Article 2...",
      "source": "Reuters",
      "social_reactions": 5000
    }
  ]'
```

#### API Documentation
Visit http://localhost:8000/docs for interactive API documentation (Swagger UI).

## ⚙️ Configuration

### Creating a Configuration File

Copy the example configuration:
```bash
cp config.example.json my_config.json
```

Edit `my_config.json`:
```json
{
  "memory_size": 2000,
  "learning_rate": 0.0005,
  "batch_size": 64,
  "gamma": 0.99,
  "epsilon": 1.0,
  "epsilon_decay": 0.995,
  "epsilon_min": 0.01,
  "num_episodes": 500,
  "patience": 15,
  "model_save_path": "./models/best_model.pth",
  "train_data_path": "./data/train_data.json",
  "test_data_path": "./data/test_data.json"
}
```

### Using Configuration in Code

```python
from code.utils.config import get_default_config
import json

# Load custom config
with open("my_config.json") as f:
    config = json.load(f)

# Or use defaults
config = get_default_config()
```

## 📊 Data Format

### Training Data Format
```json
[
  {
    "text": "The federal government announces new AI ethics regulations.",
    "source_reliability": "The New York Times",
    "social_reactions": 6200,
    "label": 2
  },
  {
    "text": "New evidence suggests the moon is actually a hologram.",
    "source_reliability": "Unknown Blog",
    "social_reactions": 12000,
    "label": 0
  }
]
```

Labels:
- `0`: Fake News
- `1`: Suspicious News
- `2`: Real News

### Supported News Sources

The system recognizes these sources with pre-defined reliability scores:
- The New York Times (0.90)
- Reuters (0.90)
- BBC (0.85)
- CNN (0.80)
- Fox News (0.60)
- [See full list in `code/utils/config.py`]

Unknown sources default to 0.50 reliability.

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=code --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## 🐳 Docker Usage

### Build Image
```bash
docker build -t fakelensev2:latest .
```

### Run API Server
```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  fakelensev2:latest
```

### Run Training
```bash
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  fakelensev2:latest \
  python -m code.main train
```

## 📚 Code Examples

### Custom Training Loop
```python
from code.agents.fake_news_agent import FakeNewsAgent
from code.utils.feature_extraction import FeatureExtractor
from code.utils.config import get_default_config
from code.models.vectorizer import BaseVectorizer
import json

# Load data
with open("data/train_data.json") as f:
    train_data = json.load(f)

# Setup
config = get_default_config()
config["num_episodes"] = 1000  # Custom value

vectorizer = BaseVectorizer()
feature_extractor = FeatureExtractor(vectorizer=vectorizer)

# Create agent
agent = FakeNewsAgent(
    state_size=770,
    action_size=3,
    config=config
)

# Training loop
for episode in range(config["num_episodes"]):
    total_reward = 0
    for sample in train_data:
        features = feature_extractor.extract_features(
            sample["text"],
            sample["source_reliability"],
            sample["social_reactions"]
        )

        action = agent.act(features)
        reward = 1 if action == sample["label"] else -1

        agent.remember(features, action, reward, features, False)
        agent.replay()
        total_reward += reward

    print(f"Episode {episode + 1}: Reward = {total_reward}")

# Save model
agent.save("models/my_model.pth")
```

### Custom Inference
```python
from code.inference import InferenceEngine

# Load model
engine = InferenceEngine("models/best_model.pth")

# Analyze article
article = """
Breaking: Scientists at CERN announce discovery of new particle
that could revolutionize our understanding of physics.
"""

prediction = engine.predict(
    text=article,
    source="Reuters",
    social_reactions=15000
)

labels = {0: "Fake", 1: "Suspicious", 2: "Real"}
print(f"This article is likely: {labels[prediction]}")
```

## 🔧 Troubleshooting

### Model Not Found
```python
# Error: FileNotFoundError: Model not found
# Solution: Check model path
import os
model_path = "models/best_model.pth"
if not os.path.exists(model_path):
    print("Please train a model first!")
    # Run training
```

### CUDA Out of Memory
```python
# Reduce batch size in config
config["batch_size"] = 32  # Instead of 64
```

### Import Errors
```bash
# Make sure you're in the project root
export PYTHONPATH=/path/to/FakeLenseV2:$PYTHONPATH

# Or install as package
pip install -e .
```

## 📖 Further Reading

- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Detailed changelog
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when server is running)
- [Original README](README.md) - Project overview and research details

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## 📞 Support

- GitHub Issues: https://github.com/Navy10021/FakeLenseV2/issues
- Email: iyunseob4@gmail.com
