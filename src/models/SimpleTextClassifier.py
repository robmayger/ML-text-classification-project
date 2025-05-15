import torch
import torch.nn.functional as F

from .BaseClassifier import BaseClassifier
from .neural_nets.MeanEmbeddingClassifier import MeanEmbeddingClassifier


class SimpleTextClassifier(BaseClassifier):
    """
    A simple text classification model using mean word embeddings.

    Attributes:
        model (MeanEmbeddingClassifier): The underlying model that performs classification.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int) -> None:
        """
        Initializes the SimpleTextClassifier.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the embedding vectors.
            num_classes (int): Number of output classes for classification.
        """
        super(SimpleTextClassifier, self).__init__()
        self.model = MeanEmbeddingClassifier(vocab_size, embed_dim, num_classes)

    def forward(self, x) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (Tensor): Input tensor containing token indices.

        Returns:
            Tensor: Output logits from the classifier.
        """

        return self.model.forward(x)
    
    def training_step(self, batch) -> torch.Tensor:
        """
        Computes the loss on a training batch.

        Args:
            batch (Tuple[Tensor, Tensor]): A tuple of (inputs, targets).

        Returns:
            Tensor: Cross-entropy loss computed on the batch.
        """
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        return loss

    def validation_step(self, batch) -> dict:
        """
        Computes the loss and accuracy on a validation batch.

        Args:
            batch (Tuple[Tensor, Tensor]): A tuple of (inputs, targets).

        Returns:
            dict: Dictionary containing validation loss and accuracy.
        """
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        return {"val_loss": loss.item(), "val_acc": acc.item()}

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer used during training.
        """
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)