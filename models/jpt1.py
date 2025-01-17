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
        dropout: float,
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
        # self.transformer = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)

        # self.initial_expand = nn.Sequential(
        #     nn.Linear(hypertoken_size, embed_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.LayerNorm(embed_dim // 2),
        #     nn.Linear(embed_dim // 2, embed_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.LayerNorm(embed_dim),
        # )

        self.ln_final = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, hypertoken_size)

    def generate_square_subsequent_mask(self, sz):
        """Generate a causal mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)  # Upper triangular matrix
        mask = mask.masked_fill(
            mask == 1, float("-inf")
        )  # Mask future tokens with -inf
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hypertoken_size = x.shape

        # Create causal mask to prevent attending to future tokens
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        # Embeddings
        # token_embed = self.initial_expand(x)  # expand the hypertoken into the embed_dim

        expand_factor = self.embed_dim // hypertoken_size
        token_embed = x.unsqueeze(-1).expand(-1, -1, -1, expand_factor).reshape(
            batch_size, seq_len, self.embed_dim
        ) * (1.0 / expand_factor)

        position_ids = (
            torch.arange(seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        x = token_embed + self.position_embedding(position_ids)

        # Transformer blocks
        # For TransformerDecoder, we need memory (encoder output) and target (decoder input)
        # Using a zero memory tensor of the same size as input
        memory = torch.zeros_like(x)

        # Transformer blocks - note we're passing memory as the first argument
        x = self.transformer(x, mask=causal_mask)
        # x = self.transformer(tgt=x, memory=memory, tgt_mask=causal_mask)

        check_memory_usage(self)

        x = self.ln_final(x)
        x = self.fc_out(x)
        return x

        # return x  # we output the raw embeddings that will flow into the decoder model
