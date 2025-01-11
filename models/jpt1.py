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
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        
        # Use PyTorch's TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.ln_final = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
        
    def get_causal_mask(self, size: int) -> torch.Tensor:
        # Create causal mask for self-attention
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        
        # Create causal mask
        mask = self.get_causal_mask(seq_len).to(x.device)
        
        # Embeddings
        token_embed = self.token_embedding(x)
        x = token_embed + self.position_embedding[:, :seq_len, :]
        
        # Transformer blocks
        x = self.transformer(x, src_mask=mask)
            
        x = self.ln_final(x)
        return self.fc_out(x) 