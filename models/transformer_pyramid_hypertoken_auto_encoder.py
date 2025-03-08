import torch
from torch import nn
from torch.nn import functional as F


class TransformerPyramidHyperTokenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
        compress_factor: int,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.embed_dim = embed_dim
        self.hypertoken_size = hypertoken_size

        # Embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(token_len, embed_dim)

        self.COMPRESS_FACTOR = compress_factor
        self.MIN_EMBED_DIM = hypertoken_size // self.token_len

        self.register_buffer("positions", torch.arange(token_len).unsqueeze(0))

        compression_sizes = []
        current_size = self.embed_dim
        while current_size > self.MIN_EMBED_DIM:
            compression_sizes.append(current_size)
            current_size //= self.COMPRESS_FACTOR

        if compression_sizes[-1] != hypertoken_size // self.token_len:
            compression_sizes.append(self.MIN_EMBED_DIM)

        self.compression_sizes = list(zip(compression_sizes[:-1], compression_sizes[1:]))
        self.compression_layers = nn.ModuleList([])

        for in_dim, out_dim in self.compression_sizes:
            nh = out_dim // head_size
            nh = max(2, nh) if nh >= 2 else 1

            t_layer = nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=nh,
                dim_feedforward=in_dim * 4,
                batch_first=True,
                dropout=0.000,
                norm_first=True,
            )

            self.compression_layers.append(nn.TransformerEncoder(t_layer, num_layers=n_layers))

        self.compression_residuals = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in self.compression_sizes])

        self.compression_res_scales = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in self.compression_sizes])

        self.input_res_scale = nn.Parameter(torch.zeros(1))

        self.input_residual = nn.Sequential(
            nn.Linear(embed_dim, hypertoken_size // self.token_len),
        )

    @typechecked
    def forward(self, x: TensorType["batch_size", "token_len"]) -> TensorType["batch_size", "hypertoken_size"]:
        batch_size, token_len = x.size()

        embedded = self.embed(x) + self.pos_embed(self.positions)
        orig_embedded = embedded  # Save for final residual

        compressed = embedded

        for i, (dim_in, dim_out) in enumerate(self.compression_sizes):
            res_input = compressed
            compressed = self.compression_layers[i](compressed)
            compress_factor = dim_in // dim_out

            compressed = compressed.reshape(batch_size, token_len, -1, compress_factor).mean(dim=-1)
            compressed += self.compression_residuals[i](res_input) * torch.sigmoid(self.compression_res_scales[i])

        compressed = compressed.flatten(start_dim=1)
        compressed += self.input_residual(orig_embedded).flatten(start_dim=1) * torch.sigmoid(self.input_res_scale)
        # compressed = F.normalize(compressed, p=2, dim=1, eps=1e-2)

        return compressed


class TransformerPyramidHyperTokenDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
        compress_factor: int,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.embed_dim = embed_dim

        self.COMPRESS_FACTOR = compress_factor
        self.MIN_EMBED_DIM = hypertoken_size // token_len

        self.hypertoken = None

        expansion_sizes = []
        current_size = self.MIN_EMBED_DIM
        while current_size < self.embed_dim:
            expansion_sizes.append(current_size)
            current_size *= self.COMPRESS_FACTOR

        if expansion_sizes[-1] != self.embed_dim:
            expansion_sizes.append(self.embed_dim)

        self.expansion_sizes = list(zip(expansion_sizes[:-1], expansion_sizes[1:]))
        self.expansion_layers = nn.ModuleList([])

        for in_dim, out_dim in self.expansion_sizes:
            nh = max(2, out_dim // head_size)

            t_layer = nn.TransformerEncoderLayer(
                d_model=out_dim,
                nhead=nh,
                dim_feedforward=out_dim * 4,
                batch_first=True,
                dropout=0.00,
                norm_first=True,
            )

            self.expansion_layers.append(nn.TransformerEncoder(t_layer, num_layers=n_layers))

        # Intra-expansion residuals
        self.expansion_residuals = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in self.expansion_sizes])
        self.expansion_res_scales = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in self.expansion_sizes])

        # Hypertoken-to-output residual
        self.output_residual = nn.Linear(hypertoken_size, token_len * vocab_size)
        self.output_res_scale = nn.Parameter(torch.zeros(1))

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    @typechecked
    def forward(self, x: TensorType["batch_size", "hypertoken_size"]) -> TensorType["batch_size", "token_len", "vocab_size"]:
        batch_size = x.size(0)
        hypertoken = x
        expanded = x.reshape(batch_size, self.token_len, -1)
        # expanded = F.normalize(expanded, p=2, dim=2)

        for i, (dim_in, dim_out) in enumerate(self.expansion_sizes):
            res_input = expanded  # Preserve pre-expansion state
            expand_factor = dim_out // dim_in
            expanded = expanded.unsqueeze(-1).expand(-1, -1, -1, expand_factor).reshape(batch_size, self.token_len, dim_out) * (
                1.0 / expand_factor
            )

            expanded = self.expansion_layers[i](expanded)
            expanded += self.expansion_residuals[i](res_input) * torch.sigmoid(self.expansion_res_scales[i])

        expanded = self.fc_out(expanded)
        expanded += self.output_residual(hypertoken).view_as(expanded) * torch.sigmoid(self.output_res_scale)

        return expanded


class TransformerPyramidHyperTokenAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
        compress_factor: int,
    ) -> None:
        super().__init__()

        self.average_memory_usage = 0

        self.encoder = TransformerPyramidHyperTokenEncoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=embed_dim,
            head_size=head_size,
            n_layers=n_layers,
            hypertoken_size=hypertoken_size,
            compress_factor=compress_factor,
        )

        self.decoder = TransformerPyramidHyperTokenDecoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=embed_dim,
            head_size=head_size,
            n_layers=n_layers,
            hypertoken_size=hypertoken_size,
            compress_factor=compress_factor,
        )

    @typechecked
    def forward(self, x: TensorType["batch_size", "token_len"]) -> TensorType["batch_size", "vocab_size"]:
        hypertoken = self.encoder(x)
        hypertoken = hypertoken + torch.randn_like(hypertoken) * 0.02
        self.hypertoken = hypertoken  # used for some loss stuff later
        # add some noise to the hypertoken

        decoded = self.decoder(hypertoken)

        return decoded
