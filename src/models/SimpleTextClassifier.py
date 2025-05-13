import torch
import torch.nn.functional as F

from .BaseClassifier import BaseClassifier
from models.neural_nets.SimpleTextClassifier import SimpleTextClassifier


class SimpleTextClassifier(BaseClassifier):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int) -> None:
        super(SimpleTextClassifier, self).__init__()
        self.model = SimpleTextClassifier(vocab_size, embed_dim, num_classes)

    def forward(self, x):
        return self.model.forward(x)
    
    def training_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        return {"val_loss": loss.item(), "val_acc": acc.item()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)