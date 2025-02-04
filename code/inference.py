# Inference
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from fake_news_agent import FakeNewsAgent
from feature_extraction import FeatureExtractor

# Inference Process
def infer(agent, text, source_reliability, social_reactions):
    features = FeatureExtractor().extract_features(text, source_reliability, social_reactions)
    features = torch.FloatTensor(features).unsqueeze(0).to(agent.device)
    agent.model.load_state_dict(torch.load('./models/best_model_v2.pth', map_location=agent.device))
    agent.model.eval()
    with torch.no_grad():
        action = agent.act(features)
    return action  

    # A function that quantifies the trust level of media companies
def convert_source_reliability(source_reliability: str) -> float:
    """Converts source reliability to a float based on the US news outlet."""
    # Mapping trustworthiness with actual media outlets
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

def eval_agent(agent, data):
    # Perform inference and collect predictions
    y_true = []
    y_pred = []

    print("\n========== ON TEST DATASET ==========")
    for sample in test_data:
        text = sample["text"]
        source_reliability = convert_source_reliability(sample["source_reliability"])
        social_reactions = sample["social_reactions"]
        label = sample["label"]

        prediction = infer(agent, text, source_reliability, social_reactions)
        y_true.append(label)
        y_pred.append(prediction)
        print(f"Article: {text}\nPrediction: {'Real News' if prediction == 2 else 'Suspicious News' if prediction == 1 else 'Fake News'}\n")

    # Compute performance metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')

    print("\n========== Performance Metrics ==========")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake News", "Suspicious News", "Real News"], yticklabels=["Fake News", "Suspicious News", "Real News"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
# Load test data
file_path = "./data/test_data.json"
with open(file_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define State and Action Sizes
state_size = 768 + 2    # BERT vector + source reliability + social reactions
action_size = 3         # Fake or Real or Suspicious

# Initialize the reinforcement learning agent
agent = FakeNewsAgent(state_size, action_size)

# Eval the agent
eval_agent(agent, test_data)