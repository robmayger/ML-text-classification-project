import torch
import torch.nn as nn

from .BaseModel import PytorchBaseModel
from .BertEmbedding import BertEmbedding
from .TransformerBlock import TransformerBlock


class BERTClassifier(PytorchBaseModel):
    """
    BERT-style classifier composed of embedding and Transformer blocks.

    Args:
        vocab_size (int): Vocabulary size.
        embed_size (int): Embedding size.
        heads (int): Number of attention heads.
        depth (int): Number of Transformer blocks.
        ff_hidden (int): Hidden size of the feed-forward layers.
        max_len (int): Maximum sequence length.
        num_classes (int): Number of output classes for classification.
    """
    def __init__(self, vocab_size, embed_size=256, heads=8, depth=6, ff_hidden=512, max_len=512, num_classes=20):
        super().__init__()
        self.embedding = BertEmbedding(vocab_size, embed_size, max_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, heads, ff_hidden) for _ in range(depth)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, x, mask=None):
        """
        Forward pass for the BERT encoder.

        Args:
            x (Tensor): Input token indices of shape [batch_size, seq_len].
            mask (Tensor, optional): Attention mask.

        Returns:
            Tensor: Logits of shape [batch_size, num_classes].
        """
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed]
        x = self.embedding(x)
        x = torch.cat((cls_tokens, x), dim=1)  # prepend CLS
        for block in self.transformer_blocks:
            x = block(x, mask)
        cls_out = x[:, 0, :]  # CLS token
        return self.classifier(cls_out)
