import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.BaseModel import BaseModel


class BertEmbedding(nn.Module):
    """
    Embedding layer that combines token and positional embeddings for BERT.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_size (int): Dimensionality of the embeddings.
        max_len (int): Maximum length of the input sequences.
    """
    def __init__(self, vocab_size, embed_size, max_len=512):
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


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        embed_size (int): Total embedding size.
        heads (int): Number of attention heads.
    """
    def __init__(self, embed_size, heads):
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


class TransformerBlock(nn.Module):
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


class BERTEncoder(nn.Module):
    """
    BERT-style encoder composed of embedding and Transformer blocks.

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


class BERTTextClassifier(BaseModel):
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
        self.model = BERTEncoder(
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

