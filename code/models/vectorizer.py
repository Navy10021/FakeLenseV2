"""Text vectorization using pre-trained language models (BERT/RoBERTa)"""

from typing import List, Union
import numpy as np
import torch
from transformers import BertTokenizer, BertModel


class BaseVectorizer:
    """
    Base vectorizer class for converting text into embeddings.
    Uses pre-trained models like BERT or RoBERTa.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the vectorizer with a pre-trained model.

        Args:
            model_name: Name of the pre-trained model (default: bert-base-uncased)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode

    def vectorize(
        self,
        text: Union[str, List[str]],
        pooling: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert text into vector embeddings.

        Args:
            text: A single string or list of strings to vectorize
            pooling: Whether to apply mean pooling over hidden states

        Returns:
            Text embedding as numpy array or torch.Tensor
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if pooling:
            return outputs.last_hidden_state.mean(dim=1).cpu()

        # Use the [CLS] token's vector representation
        result = outputs.last_hidden_state[:, 0, :]

        if isinstance(text, list):
            return result.cpu().numpy()
        else:
            return result.squeeze().cpu().numpy()
