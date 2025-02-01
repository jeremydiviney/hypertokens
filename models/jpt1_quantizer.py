import time
import os
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tokenizers import Tokenizer


def compute_logits(model, hidden_states):
    """
    Convert hidden states to similarity scores over vocab.
    hidden_states shape: [batch_size, seq_len, embed_dim]
    Returns logits of shape [batch_size, seq_len, vocab_size]
    """
    # Normalize
    hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
    output_norm = F.normalize(model.codebook.lookup_embeddings.weight, p=2, dim=-1)  # [vocab_size, embed_dim]

    # Compute cosine similarities (batch x seq x vocab)
    # (hidden_norm: bsz, seq, embed) x (output_norm: vocab, embed) -> bsz, seq, vocab
    logits = torch.einsum("bse,ve->bsv", hidden_norm, output_norm)
    logits = logits / model.temperature
    return logits


class TokenCodebook(nn.Module):
    def __init__(self, embed_dim, tokenizer: Tokenizer):
        super().__init__()
        self.embed_dim = embed_dim
        self.tokenizer = tokenizer

        self.token_list = tokenizer.get_vocab()

        self.text_token_to_idx = {token: self.token_list[token] for token in self.token_list}

        self.lookup_embeddings = nn.Embedding(len(self.token_list), embed_dim)

    def get_nearest_token_indices(self, projections: torch.Tensor, top_k: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """
        Find the nearest tokens using pure PyTorch cosine similarity.
        projections shape: [batch_size, seq_len, embed_dim]
        lookup_embeddings shape: [vocab_size, embed_dim]
        Returns a [batch_size, seq_len] tensor of token indices.
        """

        # 1. Normalize the projections
        proj_norm = F.normalize(projections, p=2, dim=-1)  # [B, S, E]

        # 2. Normalize the codebook embeddings
        emb_norm = F.normalize(self.lookup_embeddings.weight, p=2, dim=-1)  # [V, E]

        # 3. Compute pairwise similarity: shape [B, S, V]
        similarity = torch.einsum("bse,ve->bsv", proj_norm, emb_norm)

        if top_k == 1:
            # do_sample doesn't really apply when top_k=1, so we just pick argmax
            indices = similarity.argmax(dim=-1)  # [B, S]
            return indices

        # -- Otherwise, select top_k candidates along the vocab dimension --
        # topk_values, topk_indices both => [B, S, top_k]
        topk_values, topk_indices = similarity.topk(k=top_k, dim=-1)

        # Sample from a softmax distribution over the top_k
        # 1) Convert topk_values => probabilities via softmax
        probs = F.softmax(topk_values / temperature, dim=-1)  # [B, S, top_k]

        # 2) Flatten batch and sequence dims for multinomial
        bsz, seq_len, k = probs.shape
        probs_flat = probs.view(-1, k)  # [B*S, top_k]

        # 3) Sample an index in [0..top_k-1] for each position in the batch
        chosen_in_topk = torch.multinomial(probs_flat, 1).squeeze(-1)  # [B*S]

        # 4) Map chosen_in_topk back to the actual token IDs
        topk_indices_flat = topk_indices.view(-1, k)  # [B*S, top_k]
        final_indices_flat = torch.gather(topk_indices_flat, dim=1, index=chosen_in_topk.unsqueeze(-1)).squeeze(-1)  # [B*S]

        # 5) Reshape to [B, S]
        final_indices = final_indices_flat.view(bsz, seq_len)
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

        self.embeddings = nn.Embedding(len(codebook.token_list), embed_dim)

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

        if modelType == JPT1QuantModelType.COS_SIM:
            self.fc_out = nn.Linear(embed_dim, token_space_dim)
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

        # Create causal mask to prevent attending to future tokens
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        embedded = embedded + self.position_embedding(position_ids)

        # For TransformerDecoder, we need memory (encoder output) and target (decoder input)
        # Using a zero memory tensor of the same size as input
        # memory = torch.zeros_like(x)

        # Transformer blocks - note we're passing memory as the first argument
        x = self.transformer(embedded, mask=causal_mask)
        # x = self.transformer(tgt=x, memory=memory, tgt_mask=causal_mask)

        # Add residual connection from embeddings

        x = self.fc_out(x)  # Shape: [batch_size, seq_len, output_dim]

        return x
