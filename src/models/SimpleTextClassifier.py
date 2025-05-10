import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import BaseModel


class SimpleTextClassifier(BaseModel, nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        mean_embedded = embedded.mean(dim=1)
        output = self.fc(mean_embedded)
        return output
    
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
        return torch.optim.Adam(self.parameters(), lr=1e-3)