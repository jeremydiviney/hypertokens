import torch
from torch import nn
from torch.nn import functional as F


class HyperTokenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.embed_dim = embed_dim
        self.hypertoken_size = hypertoken_size

        # Embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim).to(torch.bfloat16)
        self.pos_embed = nn.Embedding(token_len, embed_dim).to(torch.bfloat16)

        self.register_buffer("positions", torch.arange(token_len, dtype=torch.long).unsqueeze(0))

        compress_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=embed_dim // head_size,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dtype=torch.bfloat16,
            dropout=0.15,
            norm_first=True,
        )

        self.compression_layer = nn.TransformerEncoder(compress_layer, num_layers=n_layers)

        # self.linear_compression_layer = nn.Linear(embed_dim, 2 * (hypertoken_size // token_len))

        # self.fc_out = nn.Linear(2 * (hypertoken_size // token_len) * token_len, hypertoken_size)
        self.fc_out = nn.Sequential(nn.Linear(embed_dim, hypertoken_size), nn.LayerNorm(hypertoken_size), nn.LayerNorm(hypertoken_size))

        self.to(torch.bfloat16)

    @typechecked
    def forward(self, x: TensorType["batch_size", "token_len"]) -> TensorType["batch_size", "hypertoken_size"]:
        batch_size, token_len = x.size()

        embedded = self.embed(x) + self.pos_embed(self.positions)

        compressed = self.compression_layer(embedded)

        # compressed = self.linear_compression_layer(compressed)

        # compressed = compressed.view(batch_size, -1)
        # compressed = compressed[:, -1, :]  # only take cls char/token on the end

        compressed = compressed.mean(dim=1)

        compressed = self.fc_out(compressed)

        return compressed


class HyperTokenDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.embed_dim = embed_dim

        self.hypertoken = None
        self.vocab_size = vocab_size

        self.fc_out = nn.Linear(embed_dim, vocab_size)

        # Positional embedding for static tokens
        self.pos_embed = nn.Embedding(token_len, embed_dim)
        self.register_buffer("positions", torch.arange(token_len))

        t_alt_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=embed_dim // head_size,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dtype=torch.bfloat16,
            dropout=0.15,
            norm_first=True,
        )

        self.initial_expansion_layer = nn.Sequential(
            nn.Linear(hypertoken_size, embed_dim), nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        )
        # self.linear_expansion_layer = nn.Linear(2 * (hypertoken_size // token_len), embed_dim)

        self.expansion_layer = nn.TransformerEncoder(t_alt_layer, num_layers=n_layers)

        self._alt_expansion_layer = nn.Sequential(
            nn.Linear(hypertoken_size, embed_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2),
            nn.Linear(embed_dim // 2, vocab_size * token_len // 4),
            nn.GELU(),
            nn.LayerNorm(vocab_size * token_len // 4),
            nn.Linear(vocab_size * token_len // 4, vocab_size * token_len // 2),
            nn.GELU(),
            nn.LayerNorm(vocab_size * token_len // 2),
            nn.Linear(vocab_size * token_len // 2, vocab_size * token_len),
            nn.GELU(),
            nn.LayerNorm(vocab_size * token_len),
        )

        self.to(torch.bfloat16)

    @typechecked
    def forward(self, x: TensorType["batch_size", "hypertoken_size"]) -> TensorType["batch_size", "token_len", "vocab_size"]:
        device = x.device
        batch_size = x.size(0)

        # expanded = self.initial_expansion_layer(x)

        # # Create static tokens with positional info
        # static_tokens = torch.ones(batch_size, self.token_len - 1, self.embed_dim, dtype=torch.bfloat16, device=device)
        # static_tokens = static_tokens + self.pos_embed(self.positions[:-1])

        # # Combine first token with static tokens
        # expanded = torch.cat([expanded.unsqueeze(1), static_tokens], dim=1)

        # expanded = self.expansion_layer(expanded)

        expanded = self._alt_expansion_layer(x)

        fc_out = expanded.view(batch_size, self.token_len, self.vocab_size)

        # fc_out = self.fc_out(expanded)
        return fc_out


class HyperTokenAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
    ) -> None:
        super().__init__()

        self.average_memory_usage = 0

        self.encoder = HyperTokenEncoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=embed_dim,
            head_size=head_size,
            n_layers=n_layers,
            hypertoken_size=hypertoken_size,
        )

        self.decoder = HyperTokenDecoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=embed_dim,
            head_size=head_size,
            n_layers=n_layers,
            hypertoken_size=hypertoken_size,
        )

        self.to(torch.bfloat16)

    @typechecked
    def forward(self, x: TensorType["batch_size", "token_len"]) -> TensorType["batch_size", "vocab_size"]:
        hypertoken = self.encoder(x)
        self.hypertoken = hypertoken  # used for some loss stuff later
        decoded = self.decoder(hypertoken)

        return decoded
