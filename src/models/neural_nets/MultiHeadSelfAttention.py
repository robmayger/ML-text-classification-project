import torch
import torch.nn as nn
import math

from .BaseModel import PytorchBaseModel


class MultiHeadSelfAttention(PytorchBaseModel):
    """
    Multi-head self-attention mechanism.

    Args:
        embed_size (int): Total embedding size.
        heads (int): Number of attention heads.
    """
    def __init__(self, embed_size: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "embed size must be divisible by heads"

        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        """
        Forward pass for the multi-head self-attention layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, embed_size].
            mask (Tensor, optional): Attention mask of shape [batch_size, 1, 1, seq_len].

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, embed_size].
        """
        B, T, C = x.size()
        qkv = self.qkv(x)  # [B, T, 3C]
        qkv = qkv.reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, heads, T, head_dim]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, heads, T, T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # [B, heads, T, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.fc_out(out)