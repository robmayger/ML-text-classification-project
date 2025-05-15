import torch
import torch.nn as nn

from .BaseModel import PytorchBaseModel


class MeanEmbeddingClassifier(PytorchBaseModel):
    """
    A simple classifier that averages word embeddings and feeds them to a linear layer.

    Attributes:
        embedding (nn.Embedding): Embedding layer that maps token indices to vectors.
        fc (nn.Linear): Fully connected layer that outputs class logits.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int) -> None:
        """
        Initializes the MeanEmbeddingClassifier.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of each word embedding vector.
            num_classes (int): Number of target classes for classification.
        """

        super(MeanEmbeddingClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
                              containing token indices.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes) for each input sample.
        """
        embedded = self.embedding(x)
        mean_embedded = embedded.mean(dim=1)
        output = self.fc(mean_embedded)
        return output