import torch
import torch.nn as nn

from .BaseModel import PytorchBaseModel


class BertEmbedding(PytorchBaseModel):
    """
    Embedding layer that combines token and positional embeddings for BERT.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_size (int): Dimensionality of the embeddings.
        max_len (int): Maximum length of the input sequences.
    """
    def __init__(self, vocab_size: int, embed_size: int, max_len: int = 512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_len, embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward pass for the embedding layer.

        Args:
            x (Tensor): Input token indices of shape [batch_size, seq_len].

        Returns:
            Tensor: Embedded input of shape [batch_size, seq_len, embed_size].
        """
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # shape: [1, seq_len]
        token_embeddings = self.token_embed(x)
        position_embeddings = self.pos_embed(pos)
        return self.dropout(token_embeddings + position_embeddings)