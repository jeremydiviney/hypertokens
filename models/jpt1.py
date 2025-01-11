import torch
import torch.nn as nn
from typing import Optional


class JPT1(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Use nn.Embedding for learnable positional encodings
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

        # Use PyTorch's TransformerEncoder
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)

        self.ln_final = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def get_causal_mask(self, size: int) -> torch.Tensor:
        # Create causal mask for self-attention
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hypertoken_size = x.shape

        # Create dummy memory tensor with minimal sequence length
        dummy_memory = torch.zeros(batch_size, 1, self.embed_dim, device=x.device)

        # Embeddings
        token_embed = x  # the hypertoken is the embedding
        position_ids = (
            torch.arange(seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        x = token_embed + self.position_embedding(position_ids)

        # Transformer blocks
        x = self.transformer(x, memory=dummy_memory)

        # x = self.ln_final(x)
        # return self.fc_out(x)
        return x  # we output the raw embeddings that will flow into the decoder model
