import torch
from torch import nn
from torch.nn import functional as F
from typeguard import typechecked
from torchtyping import TensorType


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        # BatchNorm1d works better for conv layers since it normalizes across batch and spatial dimensions
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        # No need to transpose since BatchNorm1d expects [N, C, L]
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.norm2(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual


class ConvHyperTokenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        hypertoken_size: int,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.embed_dim = embed_dim
        self.hypertoken_size = hypertoken_size

        self.hypertoken = None

        # Embeddings with larger embedding dimension
        self.embed = nn.Embedding(vocab_size, embed_dim).to(torch.bfloat16)
        self.pos_embed = nn.Embedding(token_len, embed_dim).to(torch.bfloat16)

        # Add embedding dropout
        self.embed_dropout = nn.Dropout(0.1)

        self.register_buffer("positions", torch.arange(token_len, dtype=torch.long).unsqueeze(0))

        # Initial projection to increase channel dimension
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2), nn.GELU(), nn.Dropout(0.1)
        )

        # Residual blocks at each resolution
        self.res_blocks1 = nn.Sequential(ResidualConvBlock(embed_dim * 2), ResidualConvBlock(embed_dim * 2))

        self.downsample1 = nn.Sequential(
            nn.BatchNorm1d(embed_dim * 2),
            nn.Conv1d(embed_dim * 2, embed_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )

        self.res_blocks2 = nn.Sequential(ResidualConvBlock(embed_dim * 4), ResidualConvBlock(embed_dim * 4))

        self.downsample2 = nn.Sequential(
            nn.BatchNorm1d(embed_dim * 4),
            nn.Conv1d(embed_dim * 4, embed_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )

        self.res_blocks3 = nn.Sequential(ResidualConvBlock(embed_dim * 8), ResidualConvBlock(embed_dim * 8))

        # Global context mixing
        self.global_pool = nn.Sequential(nn.BatchNorm1d(embed_dim * 8), nn.AdaptiveAvgPool1d(1), nn.Flatten())

        # Final projection with skip connection
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim * 8, hypertoken_size * 2),
            nn.LayerNorm(hypertoken_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hypertoken_size * 2, hypertoken_size),
            nn.LayerNorm(hypertoken_size),
        )

        self.to(torch.bfloat16)

    @typechecked
    def forward(self, x: TensorType["batch_size", "token_len"]) -> TensorType["batch_size", "hypertoken_size"]:
        batch_size, token_len = x.size()

        # Embedding and positional encoding with dropout
        embedded = self.embed(x) + self.pos_embed(self.positions)
        embedded = self.embed_dropout(embedded)

        # Initial projection
        x = self.input_proj(embedded)
        x = x.transpose(1, 2)  # B, C, L

        # Multi-scale processing with residual blocks
        x = self.res_blocks1(x)
        x = self.downsample1(x)

        x = self.res_blocks2(x)
        x = self.downsample2(x)

        x = self.res_blocks3(x)

        # Global pooling and final projection
        x = self.global_pool(x)
        x = self.fc_out(x)

        return x


class ConvHyperTokenDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        hypertoken_size: int,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Initial projection with wider network
        self.initial_projection = nn.Sequential(
            nn.Linear(hypertoken_size, embed_dim * 8), nn.LayerNorm(embed_dim * 8), nn.GELU(), nn.Dropout(0.1)
        )

        # Residual blocks and upsampling
        self.res_blocks1 = nn.Sequential(ResidualConvBlock(embed_dim * 8), ResidualConvBlock(embed_dim * 8))

        self.upsample1 = nn.Sequential(
            nn.BatchNorm1d(embed_dim * 8),
            nn.ConvTranspose1d(embed_dim * 8, embed_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )

        self.res_blocks2 = nn.Sequential(ResidualConvBlock(embed_dim * 4), ResidualConvBlock(embed_dim * 4))

        self.upsample2 = nn.Sequential(
            nn.BatchNorm1d(embed_dim * 4),
            nn.ConvTranspose1d(embed_dim * 4, embed_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )

        self.res_blocks3 = nn.Sequential(ResidualConvBlock(embed_dim * 2), ResidualConvBlock(embed_dim * 2))

        # Final projection to vocab size with skip connection
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, vocab_size),
        )

        self.to(torch.bfloat16)

    @typechecked
    def forward(
        self, x: TensorType["batch_size", "hypertoken_size"]
    ) -> TensorType["batch_size", "token_len", "vocab_size"]:
        batch_size = x.size(0)

        # Initial projection and reshape
        x = self.initial_projection(x)
        x = x.unsqueeze(-1)  # Add sequence dimension

        # Multi-scale processing with residual blocks
        x = self.res_blocks1(x)
        x = self.upsample1(x)

        x = self.res_blocks2(x)
        x = self.upsample2(x)

        x = self.res_blocks3(x)

        # Ensure correct sequence length and final projection
        x = F.interpolate(x, size=self.token_len, mode="linear", align_corners=False)
        x = x.transpose(1, 2)  # B, L, C
        x = self.final_proj(x)

        return x


class ConvHyperTokenAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        hypertoken_size: int,
    ) -> None:
        super().__init__()

        self.encoder = ConvHyperTokenEncoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=embed_dim,
            hypertoken_size=hypertoken_size,
        )

        self.decoder = ConvHyperTokenDecoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=embed_dim,
            hypertoken_size=hypertoken_size,
        )

        self.to(torch.bfloat16)

    @typechecked
    def forward(self, x: TensorType["batch_size", "token_len"]) -> TensorType["batch_size", "token_len", "vocab_size"]:
        self.hypertoken = self.encoder(x)
        decoded = self.decoder(self.hypertoken)
        return decoded
