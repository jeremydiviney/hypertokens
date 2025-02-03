import time
import os
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tokenizers import Tokenizer


class TokenCodebook(nn.Module):
    def __init__(self, embed_dim, tokenizer: Tokenizer):
        super().__init__()
        self.embed_dim = embed_dim
        self.tokenizer = tokenizer

        self.token_list = tokenizer.get_vocab()

        self.text_token_to_idx = {token: self.token_list[token] for token in self.token_list}

        self.lookup_embeddings = nn.Embedding(len(self.token_list), embed_dim)

    def get_nearest_token_indices_cossim(self, projections: torch.Tensor, top_k: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """
        Return the nearest token indices based on cosine similarity.

        Args:
            projections (torch.Tensor): [B, S, E] tensor of projected embeddings.
            top_k (int): Number of candidates to sample from. If 1, returns argmax.
            temperature (float): Softmax temperature for sampling from the top_k candidates.

        Returns:
            torch.Tensor: [B, S] tensor of token indices.
        """
        # Normalize the projections and codebook embeddings.
        proj_norm = F.normalize(projections, p=2, dim=-1)  # [B, S, E]
        emb_norm = F.normalize(self.lookup_embeddings.weight, p=2, dim=-1)  # [V, E]

        # Compute cosine similarity using matmul.
        similarity = torch.matmul(proj_norm, emb_norm.T)  # [B, S, V]

        if top_k == 1:
            return similarity.argmax(dim=-1)  # [B, S]

        # For top_k > 1, retrieve top_k candidates.
        topk_values, topk_indices = similarity.topk(k=top_k, dim=-1)  # Both are [B, S, top_k]

        # Convert the top_k scores to probabilities using softmax.
        probs = F.softmax(topk_values / temperature, dim=-1)  # [B, S, top_k]

        # Flatten batch and sequence dimensions for sampling.
        bsz, seq_len, k = probs.shape
        probs_flat = probs.reshape(-1, k)  # [B*S, top_k]

        # Sample one candidate index per position.
        chosen = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # [B*S]

        # Map the sampled indices back to the original token indices.
        topk_indices_flat = topk_indices.reshape(-1, k)  # [B*S, top_k]
        final_indices = torch.gather(topk_indices_flat, dim=1, index=chosen.unsqueeze(-1)).view(bsz, seq_len)

        return final_indices

    def get_token_indices(self, text_tokens: list[str]) -> list[int]:
        """
        Convert a list of text tokens to their corresponding indices.
        """
        return [self.text_token_to_idx[token] for token in text_tokens]

    def get_text_token_from_indices(self, indices: np.ndarray) -> np.ndarray:
        shape = indices.shape
        # reshape so we have have a giant batch of 1 token each so the decode_batch will return a bit array as we don't just want a blob of text yet.
        indices = indices.reshape(-1, 1)
        decoded_tokens = self.tokenizer.decode_batch(indices)
        decoded_tokens = np.array(decoded_tokens)
        return decoded_tokens.reshape(shape)


class JPT1QuantModelType(Enum):
    COS_SIM = "cossim"
    STANDARD = "standard"


class JPT1Quantized(nn.Module):
    def __init__(
        self,
        seq_len: int,
        token_len: int,
        embed_dim: int,
        token_space_dim: int,
        num_heads: int,
        num_layers: int,
        num_experts: int,
        dropout: float,
        codebook: TokenCodebook,
        modelType: JPT1QuantModelType,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        # Use nn.Embedding for learnable positional encodings
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.modelType = modelType
        self.num_experts = num_experts

        # self.embeddings = nn.Embedding(len(codebook.token_list), embed_dim)
        self.embeddings = codebook.lookup_embeddings
        self.lookup_embeddings = codebook.lookup_embeddings

        self.token_len = token_len
        self.codebook = codebook
        self.token_space_dim = self.lookup_embeddings.weight.shape[1]

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

        if modelType == JPT1QuantModelType.COS_SIM:
            # self.fc_out = nn.Linear(embed_dim, len(self.codebook.token_list))
            self.fc_out_experts = nn.Linear(embed_dim, token_space_dim * num_experts)
            self.gate = nn.Linear(embed_dim, num_experts)
        else:
            self.fc_out = nn.Linear(embed_dim, len(self.codebook.token_list))

        self.temperature = nn.Parameter(torch.tensor(1.0))

        # self.fc_out = nn.Linear(embed_dim, len(self.codebook.tokens))

    def generate_square_subsequent_mask(self, sz):
        """Generate a causal mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)  # Upper triangular matrix
        mask = mask.masked_fill(mask == 1, float("-inf"))  # Mask future tokens with -inf
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        embedded = self.embeddings(x)
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        embedded = embedded + self.position_embedding(position_ids)

        x = self.transformer(embedded, mask=causal_mask)  # [batch, seq_len, embed_dim]

        expert_outputs = self.fc_out_experts(x).view(batch_size, seq_len, self.token_space_dim, self.num_experts)

        # Compute gating weights as [B, S, N]
        gate_weights = F.softmax(self.gate(x), dim=-1).unsqueeze(2)  # [B, S, 1, N]

        # Weighted sum across experts => [B, S, token_space_dim]
        output = (expert_outputs * gate_weights).sum(dim=-1)
        return output, gate_weights
