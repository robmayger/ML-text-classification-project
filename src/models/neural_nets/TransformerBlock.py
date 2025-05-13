import torch.nn as nn

from .BaseModel import PytorchBaseModel
from .MultiHeadSelfAttention import MultiHeadSelfAttention


class TransformerBlock(PytorchBaseModel):
    """
    Transformer block containing multi-head self-attention and a feed-forward network.

    Args:
        embed_size (int): Embedding size.
        heads (int): Number of attention heads.
        ff_hidden_size (int): Hidden layer size in the feed-forward network.
        dropout (float): Dropout rate.
    """
    def __init__(self, embed_size, heads, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass through the Transformer block.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, embed_size].
            mask (Tensor, optional): Attention mask.

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        attn = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x