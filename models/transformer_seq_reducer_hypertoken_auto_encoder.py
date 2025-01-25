import torch
from torch import nn
from torch.nn import functional as F
from typeguard import typechecked
from torchtyping import TensorType


class TransformerSequenceReduceHyperTokenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
        compress_factor: int,
        target_seq_len: int = 2,  # New parameter for target sequence length
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.embed_dim = embed_dim
        self.hypertoken_size = hypertoken_size
        self.target_seq_len = target_seq_len  # Store target sequence length

        # Embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim).to(torch.bfloat16)
        self.pos_embed = nn.Embedding(token_len, embed_dim).to(torch.bfloat16)

        self.compress_factor = compress_factor  # Sequence compress factor
        self.min_seq_len = target_seq_len  # Target sequence length for compression

        self.register_buffer("positions", torch.arange(token_len).unsqueeze(0))

        compression_sizes = []
        current_size = self.token_len
        while (current_size // self.compress_factor) >= self.min_seq_len:
            compression_sizes.append(current_size)
            current_size //= self.compress_factor

        if compression_sizes[-1] != hypertoken_size // self.token_len:
            compression_sizes.append(self.min_seq_len)

        self.compression_sizes = list(zip(compression_sizes[:-1], compression_sizes[1:]))
        self.compression_layers = nn.ModuleList([])

        nh = max(2, self.embed_dim // head_size)

        t_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,  # Constant embed_dim
            nhead=nh,
            dim_feedforward=self.embed_dim * 4,
            batch_first=True,
            dtype=torch.bfloat16,
            dropout=0.00,
            norm_first=True,
        )

        for in_dim, out_dim in self.compression_sizes:
            self.compression_layers.append(nn.TransformerEncoder(t_layer, num_layers=n_layers))

        self.compression_layer = nn.TransformerEncoder(t_layer, num_layers=n_layers)

        # Final compression to hypertoken size after sequence reduction
        self.final_compress = nn.Sequential(
            nn.Linear(self.min_seq_len * embed_dim, hypertoken_size * 2),
            nn.LayerNorm(hypertoken_size * 2),
            nn.GELU(),
            nn.Linear(hypertoken_size * 2, hypertoken_size),
        )

        self.to(torch.bfloat16)

    @typechecked
    def forward(self, x: TensorType["batch_size", "token_len"]) -> TensorType["batch_size", "hypertoken_size"]:
        batch_size, seq_len = x.size()

        embedded = self.embed(x) + self.pos_embed(self.positions)
        compressed = embedded

        current_seq_len = self.token_len

        for i, (in_len, out_len) in enumerate(self.compression_sizes):
            compressed = self.compression_layers[i](compressed)
            compress_factor = in_len // out_len
            compressed = compressed.reshape(batch_size, out_len, compress_factor, self.embed_dim)
            compressed = compressed.mean(dim=2)

            current_seq_len = out_len

        # Flatten sequence and embed dims for final compression
        compressed = compressed.reshape(batch_size, self.min_seq_len * self.embed_dim)

        compressed = self.final_compress(compressed)  # Project to hypertoken size

        return compressed


class TransformerSequenceExpandHyperTokenDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
        compress_factor: int,
        target_seq_len: int = 2,  # Match encoder's target_seq_len
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.embed_dim = embed_dim
        self.head_size = head_size
        self.hypertoken_size = hypertoken_size
        self.target_seq_len = target_seq_len  # Match encoder's target

        self.compress_factor = compress_factor  # Sequence compress factor
        self.min_seq_len = target_seq_len  # Target sequence length

        self.hypertoken = None
        self.fc_out = nn.Linear(embed_dim, vocab_size).to(torch.bfloat16)

        expansion_sizes = []
        current_size = self.min_seq_len
        while (current_size * self.compress_factor) <= self.token_len:
            expansion_sizes.append(current_size)
            current_size *= self.compress_factor

        if expansion_sizes[-1] != self.token_len:
            expansion_sizes.append(self.token_len)

        self.expansion_sizes = list(zip(expansion_sizes[:-1], expansion_sizes[1:]))
        self.expansion_layers = nn.ModuleList([])

        # Initial expansion from hypertoken to sequence of length MIN_SEQ_LEN
        self.initial_expand = nn.Sequential(
            nn.Linear(hypertoken_size, hypertoken_size * 2),
            nn.LayerNorm(hypertoken_size * 2),
            nn.GELU(),
            nn.Linear(hypertoken_size * 2, self.min_seq_len * embed_dim),
        )

        nh = max(2, self.embed_dim // head_size)

        t_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,  # Constant embed_dim
            nhead=nh,
            dim_feedforward=self.embed_dim * 4,
            batch_first=True,
            dtype=torch.bfloat16,
            dropout=0.00,
            norm_first=True,
        )

        for in_dim, out_dim in self.expansion_sizes:
            self.expansion_layers.append(nn.TransformerEncoder(t_layer, num_layers=n_layers))

        self.compression_layer = nn.TransformerEncoder(t_layer, num_layers=n_layers)

        self.to(torch.bfloat16)

    @typechecked
    def forward(
        self, x: TensorType["batch_size", "hypertoken_size"]
    ) -> TensorType["batch_size", "token_len", "vocab_size"]:
        batch_size = x.size(0)

        expanded = self.initial_expand(x)  # Project from hypertoken
        expanded = expanded.reshape(batch_size, self.min_seq_len, self.embed_dim)  # Reshape to initial sequence length

        for i, (in_len, out_len) in enumerate(self.expansion_sizes):
            in_len, out_len = self.expansion_sizes[i]
            expand_factor = out_len // in_len

            expanded = expanded.repeat_interleave(expand_factor, dim=1)
            expanded = expanded / expand_factor

            expanded = self.expansion_layers[i](expanded)

        fc_out = self.fc_out(expanded)
        return fc_out


class TransformerSequenceReduceHyperTokenAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
        compress_factor: int,
        target_seq_len: int = 2,  # Pass target_seq_len to autoencoder
    ) -> None:
        super().__init__()

        self.average_memory_usage = 0

        self.encoder = TransformerSequenceReduceHyperTokenEncoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=embed_dim,
            head_size=head_size,
            n_layers=n_layers,
            hypertoken_size=hypertoken_size,
            compress_factor=compress_factor,
            target_seq_len=target_seq_len,  # Pass to encoder
        )

        self.decoder = TransformerSequenceExpandHyperTokenDecoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=embed_dim,
            head_size=head_size,
            n_layers=n_layers,
            hypertoken_size=hypertoken_size,
            compress_factor=compress_factor,
            target_seq_len=target_seq_len,  # Pass to decoder
        )

        self.to(torch.bfloat16)

    @typechecked
    def forward(
        self, x: TensorType["batch_size", "token_len"]
    ) -> TensorType["batch_size", "seq_len", "vocab_size"]:  # Corrected output shape
        hypertoken = self.encoder(x)
        self.hypertoken = hypertoken  # used for some loss stuff later
        decoded = self.decoder(hypertoken)

        return decoded
