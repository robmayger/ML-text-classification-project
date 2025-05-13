import torch
import torch.nn as nn

from .BaseModel import PytorchBaseModel


class MeanEmbeddingClassifier(PytorchBaseModel):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(MeanEmbeddingClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.tensor):
        embedded = self.embedding(x)
        mean_embedded = embedded.mean(dim=1)
        output = self.fc(mean_embedded)
        return output