import torch
import torch.nn.functional as F

from .neural_nets import BERTClassifier
from .BaseClassifier import BaseClassifier


class BERTTextClassifier(BaseClassifier):
    """
    BERT-based text classification model with training and validation steps.

    Args:
        vocab_size (int): Vocabulary size.
        num_classes (int): Number of classes for classification.
        embed_size (int): Size of token embeddings.
        heads (int): Number of attention heads.
        depth (int): Number of Transformer blocks.
        ff_hidden (int): Hidden size in feed-forward layers.
        max_len (int): Maximum sequence length.
        lr (float): Learning rate for the optimizer.
    """
    def __init__(self, vocab_size, num_classes=20, embed_size=256, heads=2, depth=6, ff_hidden=128, max_len=200, lr=3e-4):
        super(BERTTextClassifier, self).__init__()
        self.model = BERTClassifier(
            vocab_size=vocab_size,
            embed_size=embed_size,
            heads=heads,
            depth=depth,
            ff_hidden=ff_hidden,
            max_len=max_len,
            num_classes=num_classes
        )
        self.lr = lr

    def forward(self, x):
        """
        Forward pass for the text classifier.

        Args:
            x (Tensor): Input tensor of token indices.

        Returns:
            Tensor: Classification logits.
        """
        return self.model(x)

    def training_step(self, batch):
        """
        Compute training loss for a batch.

        Args:
            batch (tuple): A tuple (x, y) of inputs and labels.

        Returns:
            Tensor: Cross-entropy loss.
        """
        x, y = batch
        logits = self.forward(x)
        return F.cross_entropy(logits, y)

    def validation_step(self, batch):
        """
        Compute validation loss and accuracy for a batch.

        Args:
            batch (tuple): A tuple (x, y) of inputs and labels.

        Returns:
            dict: Dictionary containing validation loss and accuracy.
        """
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        return {'val_loss': loss.item(), 'val_acc': acc.item()}

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            torch.optim.Optimizer: The Adam optimizer instance.
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

