import time
import os
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from tokenizers import Tokenizer


class JPT1QuantModelType(Enum):
    COS_SIM = "cossim"
    L2_SIM = "l2sim"
    STANDARD = "standard"


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        assert d_model % num_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # output projection
        self.c_proj = nn.Linear(d_model, d_model)
        self.c_proj.SCALE_RESIDUAL = True  # Mark for scaling

        # regularization
        self.n_head = num_head
        self.n_embd = d_model

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, d_model, dim_feedforward, activation):
        super().__init__()
        self.c_fc = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.c_proj = nn.Linear(dim_feedforward, d_model)
        self.c_proj.SCALE_RESIDUAL = True  # Mark for scaling

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x


class TransformerDecoderLayerCustom(nn.Module):

    def __init__(self, d_model, num_head, dropout, dim_feedforward, activation):
        super().__init__()
        self.ln_1 = LlamaRMSNorm(d_model, eps=1e-6)
        self.attn = CausalSelfAttention(d_model, num_head, dropout)
        self.ln_2 = LlamaRMSNorm(d_model, eps=1e-6)
        self.mlp = MLP(d_model, dim_feedforward, activation)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerDecoderCustom(nn.Module):

    def __init__(self, seq_len, d_model, dim_feedforward, num_head, num_layers, dropout, activation):
        super().__init__()
        self.seq_len = seq_len
        self.t_layers = nn.ModuleList(
            [TransformerDecoderLayerCustom(d_model, num_head, dropout, dim_feedforward, activation) for _ in range(num_layers)]
        )

    def forward(self, x):
        # idx is of shape (B, T, C)
        B, T, C = x.size()
        assert T <= self.seq_len, f"Cannot forward sequence of length {T}, sequence length is only {self.seq_len}"

        # forward the blocks of the transformer
        for block in self.t_layers:
            x = block(x)
        return x


class JPT1Quantized(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        token_space_dim: int,
        num_head: int,
        num_layers: int,
        dropout: float,
        tokenizer: Tokenizer,
        model_type: JPT1QuantModelType,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
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

        self.transformer = TransformerDecoderCustom(
            d_model=embed_dim,
            num_head=num_head,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation=nn.GELU(),
            num_layers=num_layers,
            seq_len=seq_len,
        )

        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.transformer = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)

        self.fc_ln = LlamaRMSNorm(embed_dim, eps=1e-6)

        if model_type == JPT1QuantModelType.COS_SIM:
            self.fc_out = nn.Linear(embed_dim, self.token_space_dim)
        elif model_type == JPT1QuantModelType.L2_SIM:
            self.fc_out = nn.Linear(embed_dim, self.token_space_dim)
        else:
            self.fc_out = nn.Linear(embed_dim, self.vocab_size)
            # Tie weights - share the embedding matrix with the output projection
            self.embeddings.weight = self.fc_out.weight

        self.temperature = nn.Parameter(torch.tensor(0.07))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            if hasattr(module, "SCALE_RESIDUAL") and module.SCALE_RESIDUAL:
                # Scale by depth for residual path projections
                std = 0.02 * (2 * self.num_layers) ** -0.5

            # Use normal initialization with the calculated std
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Small normal initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Standard init for LayerNorm
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, LlamaRMSNorm) or "LlamaRMSNorm" in module.__class__.__name__:
            # RMSNorm only has weight parameter (no bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        embedded = self.embeddings(x)

        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        embedded = embedded + self.position_embedding(position_ids)

        x = self.transformer(embedded)  # [B, S, embed_dim]

        x = self.fc_ln(x)
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

    def get_nearest_token_indices_l2sim(self, projections: torch.Tensor, top_k: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """
        Return the nearest token indices based on L2 distance similarity.
        Memory-optimized version without loops.

        Args:
            projections (torch.Tensor): [B, S, E] tensor of projected embeddings.
            top_k (int): Number of candidates to sample from. If 1, returns argmin.
            temperature (float): Temperature for sampling from the top_k candidates.

        Returns:
            torch.Tensor: [B, S] tensor of token indices.
        """
        batch_size, seq_len = projections.shape[:2]

        # Compute L2 distances more efficiently
        # Instead of explicitly calculating squared norms and dot products separately,
        # we can use torch's cdist function which is optimized for this purpose

        # Reshape projections to 2D for cdist
        proj_flat = projections.reshape(-1, projections.size(-1))  # [B*S, E]

        # Calculate pairwise distances efficiently
        # We use squared L2 distance (p=2, squared=True)
        distances = torch.cdist(proj_flat, self.lookup_embeddings.weight, p=2).pow(2)  # [B*S, V]

        # Convert distances to similarities (negative distances)
        similarity = -distances

        if top_k == 1:
            # For top_k=1, just return the argmax directly
            indices_flat = similarity.argmax(dim=-1)  # [B*S]
            return indices_flat.view(batch_size, seq_len)

        # For top_k > 1, get top-k values and indices
        topk_values, topk_indices = similarity.topk(k=top_k, dim=-1)  # [B*S, top_k]

        # Apply temperature scaling and softmax
        probs = F.softmax(topk_values / temperature, dim=-1)  # [B*S, top_k]

        # Sample one index per position
        chosen = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B*S]

        # Map back to original indices and reshape
        final_indices = torch.gather(topk_indices, dim=1, index=chosen.unsqueeze(-1)).squeeze(-1)  # [B*S]

        return final_indices.view(batch_size, seq_len)

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
