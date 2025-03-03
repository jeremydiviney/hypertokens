import time
import os
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tokenizers import Tokenizer


class JPT1QuantModelType(Enum):
    COS_SIM = "cossim"
    STANDARD = "standard"


class JPT1Quantized(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        token_space_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        tokenizer: Tokenizer,
        model_type: JPT1QuantModelType,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        # Use nn.Embedding for learnable positional encodings
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.model_type = model_type
        self.tokenizer = tokenizer

        self.token_list = tokenizer.get_vocab()
        self.vocab_size = len(self.token_list)

        self.embeddings = nn.Embedding(self.vocab_size, embed_dim)

        self.lookup_embeddings = self.embeddings

        self.text_token_to_idx = {token: self.token_list[token] for token in self.token_list}

        self.token_space_dim = self.lookup_embeddings.weight.shape[1]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
            activation=nn.GELU(),
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.transformer = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)

        # self.ln_final = nn.LayerNorm(embed_dim)

        if model_type == JPT1QuantModelType.COS_SIM:
            self.fc_out = nn.Linear(embed_dim, self.token_space_dim)
            # self.fc_out_experts = nn.Linear(embed_dim, token_space_dim * num_experts)
            # self.gate = nn.Linear(embed_dim, num_experts)
        else:
            self.fc_out = nn.Linear(embed_dim, self.vocab_size)

        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.extra_temperature = nn.Parameter(torch.tensor(1.0))

    def generate_square_subsequent_mask(self, sz, device):
        """Generate a causal mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)  # Upper triangular matrix
        mask = mask.masked_fill(mask == 1, float("-inf"))  # Mask future tokens with -inf
        return mask

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     batch_size, seq_len = x.shape
    #     embedded = self.embeddings(x)
    #     causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
    #     position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
    #     embedded = embedded + self.position_embedding(position_ids)

    #     x = self.transformer(embedded, mask=causal_mask)  # [batch, seq_len, embed_dim]

    #     expert_outputs = self.fc_out_experts(x).view(batch_size, seq_len, self.token_space_dim, self.num_experts)

    #     # Compute gating weights as [B, S, N]
    #     gate_weights = F.softmax(self.gate(x), dim=-1).unsqueeze(2)  # [B, S, 1, N]

    #     # Weighted sum across experts => [B, S, token_space_dim]
    #     output = (expert_outputs * gate_weights).sum(dim=-1)
    #     return output, gate_weights
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        embedded = self.embeddings(x)

        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        embedded = embedded + self.position_embedding(position_ids)

        causal_mask = self.generate_square_subsequent_mask(seq_len, x.device)

        x = self.transformer(embedded, mask=causal_mask)  # [B, S, embed_dim]

        output = self.fc_out(x)
        return output  # , gate_weights

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
        # Normalize the projections and embeddings.
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


def grouped_batch_infoNCE_loss(model, hidden_states, target_indices, group_size=8):
    """
    InfoNCE loss computed by grouping batches into sets of batches per iteration.
    Uses in-batch tokens for comparison with unique targets calculated per group.

    Args:
        model: Model
        hidden_states: Hidden states tensor of shape [batch_size, seq_length, hidden_dim]
        target_indices: Target token indices of shape [batch_size, seq_length]
        group_size: Number of batches to process together in each iteration

    Returns:
        Average InfoNCE loss
    """
    batch_size = hidden_states.shape[0]
    seq_length = hidden_states.shape[1]
    total_loss = 0
    num_groups = (batch_size + group_size - 1) // group_size  # Ceiling division

    # Normalize all hidden states at once
    hidden_states_norm = F.normalize(hidden_states, p=2, dim=2)

    # Process batches in groups
    for group_idx in range(num_groups):
        # Determine the start and end indices for this group
        group_start = group_idx * group_size
        group_end = min(group_start + group_size, batch_size)
        current_group_size = group_end - group_start

        # Get all hidden states and targets for this group
        group_hidden_norm = hidden_states_norm[group_start:group_end]  # [current_group_size, seq_length, hidden_dim]
        group_targets = target_indices[group_start:group_end]  # [current_group_size, seq_length]

        # Reshape to combine all sequences in the group
        # From [current_group_size, seq_length, hidden_dim] to [current_group_size*seq_length, hidden_dim]
        group_hidden_flat = group_hidden_norm.reshape(-1, group_hidden_norm.size(-1))
        group_targets_flat = group_targets.reshape(-1)  # [current_group_size*seq_length]

        # Find unique targets in this group
        group_unique_targets, group_inverse = torch.unique(group_targets_flat, return_inverse=True)

        # Get embeddings for unique targets and normalize
        group_unique_embeds = model.lookup_embeddings(group_unique_targets)
        group_unique_embeds_norm = F.normalize(group_unique_embeds, p=2, dim=1)

        # Compute similarities for all tokens in the group against the group's unique embeddings
        similarities = torch.matmul(group_hidden_flat, group_unique_embeds_norm.t()) / model.temperature

        # Compute loss for the entire group
        group_loss = F.cross_entropy(similarities, group_inverse)

        # Weight the group loss by the number of batches in this group
        # (to maintain proper averaging when not all groups have the same size)
        total_loss += group_loss * current_group_size

    # Return average loss
    return total_loss / batch_size
