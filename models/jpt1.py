import torch
import torch.nn as nn
from typing import Optional
from helpers.training import check_memory_usage


class JPT1(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        hypertoken_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.average_memory_usage = 0
        # Use nn.Embedding for learnable positional encodings
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

        # Use PyTorch's TransformerEncoder -- since we are only trying to predict the next sequence after the final input sequence we can just use the encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.initial_expand = nn.Sequential(
            nn.Linear(hypertoken_size, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.ln_final = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hypertoken_size = x.shape

        # Embeddings
        token_embed = self.initial_expand(x)  # expand the hypertoken into the embed_dim
        position_ids = (
            torch.arange(seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        x = token_embed + self.position_embedding(position_ids)

        # Transformer blocks
        x = self.transformer(x)

        check_memory_usage(self)

        x = self.ln_final(x)
        return self.fc_out(x)

        # return x  # we output the raw embeddings that will flow into the decoder model
