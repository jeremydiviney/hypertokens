from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from sklearn.neighbors import KDTree


class TokenCodebook(nn.Module):
    def __init__(self, token_list, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(len(token_list), embed_dim)

        # Create mappings
        self.token_to_idx = {token: i for i, token in enumerate(token_list)}
        self.idx_to_token = {i: token for i, token in enumerate(token_list)}

        self.token_array = np.array(token_list)
        self.tree = KDTree((self.embeddings.weight).detach().cpu().numpy())

        # Store raw tokens
        self.token_array = np.array(token_list)

    def forward(self, token_indices):
        """Get embeddings for token indices"""
        return self.embeddings(token_indices)

    def update_tree(self):
        """Update the KDTree with current embeddings"""
        self.tree = KDTree((self.embeddings.weight).detach().cpu().numpy())

    def get_nearest_tokens(self, projections):
        """Find nearest tokens for given projections"""
        projections_np = projections.detach().cpu().numpy()
        _, indices = self.tree.query(projections_np, k=1)
        return [self.idx_to_token[idx] for idx in indices.flatten()]

    def get_token_indices(self, text_tokens):
        """Get embeddings for text tokens"""
        device = self.embeddings.weight.device
        indices = [self.token_to_idx[token] for token in text_tokens]
        return indices

    def get_embeddings_for_indeces(self, indices: torch.Tensor) -> torch.Tensor:
        # Handle batched input of shape [batch_size, sequence_length]

        batch_size, seq_length = indices.shape
        # Reshape to 1D for conversion
        flat_indices = indices.reshape(-1)

        # Get embeddings
        embeddings = self.embeddings(flat_indices)
        # Reshape back to [batch_size, sequence_length, embed_dim]
        return embeddings.reshape(batch_size, seq_length, self.embed_dim)


class ExpandMethod(Enum):
    LINEAR = "linear"
    REPEAT = "repeat"
    TILED = "tiled"
    ZERO_PAD = "zero_pad"


class JPT1Quantized(nn.Module):
    def __init__(
        self,
        seq_len: int,
        token_len: int,
        embed_dim: int,
        token_space_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        codebook: TokenCodebook,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        # Use nn.Embedding for learnable positional encodings
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.token_len = token_len
        self.codebook = codebook
        self.token_space_dim = token_space_dim

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

        # self.ln_final = nn.LayerNorm(embed_dim)
        # self.fc_out = nn.Linear(embed_dim, hypertoken_size)
        self.fc_out = nn.Linear(embed_dim, token_space_dim)

    def generate_square_subsequent_mask(self, sz):
        """Generate a causal mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)  # Upper triangular matrix
        mask = mask.masked_fill(mask == 1, float("-inf"))  # Mask future tokens with -inf
        return mask

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        # Detach for nearest neighbor search
        projected_np = x.detach().cpu().numpy()

        # Find nearest neighbors
        _, indices = self.codebook.tree.query(projected_np, k=1)
        indices = indices.squeeze(-1)

        # Get tokens and embeddings
        tokens = self.codebook.token_array[indices]
        quantized_embeddings = torch.tensor(self.codebook.embeddings[indices], device=x.device)

        # Straight-through estimator
        quantized_embeddings = x + (quantized_embeddings - x).detach()

        return tokens, quantized_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hypertoken_size = x.shape

        # Create causal mask to prevent attending to future tokens
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        embedded = x + self.position_embedding(position_ids)

        # For TransformerDecoder, we need memory (encoder output) and target (decoder input)
        # Using a zero memory tensor of the same size as input
        # memory = torch.zeros_like(x)

        # Transformer blocks - note we're passing memory as the first argument
        x = self.transformer(embedded, mask=causal_mask)
        # x = self.transformer(tgt=x, memory=memory, tgt_mask=causal_mask)

        # Add residual connection from embeddings

        x = self.fc_out(x)  # Shape: [batch_size, seq_len, output_dim]

        return x
