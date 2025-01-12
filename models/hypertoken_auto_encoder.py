import torch
import torch.nn as nn


class HyperTokenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encode_last_n_length: int,
        embed_dim: int,
        seq_len: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
        compress_factor: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.encode_last_n_length = encode_last_n_length
        self.embed_dim = embed_dim
        self.hypertoken_size = hypertoken_size

        # Embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim).to(torch.bfloat16)
        self.pos_embed = nn.Embedding(seq_len, embed_dim).to(torch.bfloat16)

        self.COMPRESS_FACTOR = compress_factor
        self.MIN_EMBED_DIM = hypertoken_size // self.seq_len

        self.register_buffer("positions", torch.arange(seq_len).unsqueeze(0))

        compression_sizes = []
        current_size = self.embed_dim
        while current_size > self.MIN_EMBED_DIM:
            compression_sizes.append(current_size)
            current_size //= self.COMPRESS_FACTOR

        if compression_sizes[-1] != hypertoken_size // self.seq_len:
            compression_sizes.append(self.MIN_EMBED_DIM)

        self.compression_sizes = list(
            zip(compression_sizes[:-1], compression_sizes[1:])
        )
        self.compression_layers = nn.ModuleList([])

        for in_dim, out_dim in self.compression_sizes:
            nh = out_dim // head_size
            nh = max(2, nh) if nh >= 2 else 1

            t_layer = nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=nh,
                dim_feedforward=in_dim * 4,
                batch_first=True,
                dtype=torch.bfloat16,
                dropout=0.025,
                norm_first=True,
            )

            self.compression_layers.append(
                nn.TransformerEncoder(t_layer, num_layers=n_layers)
            )

        self.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()

        embedded = self.embed(x) + self.pos_embed(self.positions)
        compressed = embedded

        for i, (dim_in, dim_out) in enumerate(self.compression_sizes):
            compressed = self.compression_layers[i](compressed)
            compress_factor = dim_in // dim_out
            compressed = compressed.reshape(
                batch_size, seq_len, -1, compress_factor
            ).sum(dim=-1)

        return compressed.flatten(start_dim=1)


class HyperTokenDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encode_last_n_length: int,
        embed_dim: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
        compress_factor: int,
    ) -> None:
        super().__init__()
        self.encode_last_n_length = encode_last_n_length
        self.embed_dim = embed_dim

        self.COMPRESS_FACTOR = compress_factor
        self.MIN_EMBED_DIM = hypertoken_size // encode_last_n_length

        self.fc_out = nn.Linear(embed_dim, vocab_size)

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
                dtype=torch.bfloat16,
                dropout=0.025,
                norm_first=True,
            )

            self.expansion_layers.append(
                nn.TransformerEncoder(t_layer, num_layers=n_layers)
            )

        self.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        expanded = x.reshape(batch_size, self.encode_last_n_length, -1)

        for sub_layer_index, (dim_in, dim_out) in enumerate(self.expansion_sizes):
            expand_factor = dim_out // dim_in
            expanded = expanded.unsqueeze(-1).expand(-1, -1, -1, expand_factor).reshape(
                batch_size, self.encode_last_n_length, dim_out
            ) * (1.0 / expand_factor)

            expanded = self.expansion_layers[sub_layer_index](expanded)

        return self.fc_out(expanded)


class HyperTokenAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encode_last_n_length: int,
        embed_dim: int,
        seq_len: int,
        head_size: int,
        n_layers: int,
        hypertoken_size: int,
        compress_factor: int,
    ) -> None:
        super().__init__()

        self.average_memory_usage = 0

        self.encoder = HyperTokenEncoder(
            vocab_size=vocab_size,
            encode_last_n_length=encode_last_n_length,
            embed_dim=embed_dim,
            seq_len=seq_len,
            head_size=head_size,
            n_layers=n_layers,
            hypertoken_size=hypertoken_size,
            compress_factor=compress_factor,
        )

        self.decoder = HyperTokenDecoder(
            vocab_size=vocab_size,
            encode_last_n_length=encode_last_n_length,
            embed_dim=embed_dim,
            head_size=head_size,
            n_layers=n_layers,
            hypertoken_size=hypertoken_size,
            compress_factor=compress_factor,
        )

        self.to(torch.bfloat16)

    def check_memory_usage(self):
        if torch.cuda.is_available():
            current_mem = float(torch.cuda.memory_allocated()) / 1e9
            # max_mem = float(torch.cuda.max_memory_allocated())/1e9
            # print("Current memory: {:.2f}GB".format(current_mem))
            # print("Max memory: {:.2f}GB".format(max_mem))
            self.average_memory_usage = (self.average_memory_usage + current_mem) / 2
            # print("Average memory usage: {:.2f}GB".format(self.average_memory_usage))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hypertoken = self.encoder(x)
        decoded = self.decoder(hypertoken)
        self.check_memory_usage()
        return decoded
