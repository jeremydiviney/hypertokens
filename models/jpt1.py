from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpandMethod(Enum):
    LINEAR = "linear"
    REPEAT = "repeat"
    TILED = "tiled"
    ZERO_PAD = "zero_pad"


class JPT1(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        token_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        hypertoken_size: int,
        dropout: float,
        expand_method: ExpandMethod,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.expand_method = expand_method
        # Use nn.Embedding for learnable positional encodings
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.vocab_size = vocab_size
        self.token_len = token_len

        # Use PyTorch's TransformerEncoder -- since we are only trying to predict the next sequence after the final input sequence we can just use the encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )

        self.expand = nn.Sequential(
            nn.Linear(hypertoken_size, embed_dim // 2),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.transformer = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)

        self.ln_final = nn.LayerNorm(embed_dim)
        # self.fc_out = nn.Linear(embed_dim, hypertoken_size)
        self.fc_out = nn.Linear(embed_dim, token_len * vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """Generate a causal mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)  # Upper triangular matrix
        mask = mask.masked_fill(mask == 1, float("-inf"))  # Mask future tokens with -inf
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hypertoken_size = x.shape

        # Create causal mask to prevent attending to future tokens
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        expand_factor = self.embed_dim // hypertoken_size

        if self.expand_method == ExpandMethod.LINEAR:
            token_embed = self.expand(x)
        elif self.expand_method == ExpandMethod.REPEAT:
            token_embed = x.unsqueeze(-1).expand(-1, -1, -1, expand_factor).reshape(batch_size, seq_len, self.embed_dim)
        elif self.expand_method == ExpandMethod.TILED:
            token_embed = x.repeat(1, 1, expand_factor)
        elif self.expand_method == ExpandMethod.ZERO_PAD:
            token_embed = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
            token_embed[:, :, :hypertoken_size] = x
        else:
            raise ValueError(f"Invalid expand method: {self.expand_method}")

        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        embedded = token_embed + self.position_embedding(position_ids)

        # For TransformerDecoder, we need memory (encoder output) and target (decoder input)
        # Using a zero memory tensor of the same size as input
        # memory = torch.zeros_like(x)

        # Transformer blocks - note we're passing memory as the first argument
        x = self.transformer(embedded, mask=causal_mask)
        # x = self.transformer(tgt=x, memory=memory, tgt_mask=causal_mask)

        # Add residual connection from embeddings

        x = self.ln_final(x)

        x = self.fc_out(x)  # Shape: [batch_size, seq_len, hypertoken_size * vocab_size]
        x = x.reshape(batch_size, seq_len, self.token_len, self.vocab_size)

        # x = F.normalize(x, p=2, dim=2)
        return x
