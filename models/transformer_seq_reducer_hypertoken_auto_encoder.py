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

        self.COMPRESS_FACTOR = compress_factor  # Sequence compress factor
        self.MIN_SEQ_LEN = target_seq_len  # Target sequence length for compression

        self.register_buffer("positions", torch.arange(token_len).unsqueeze(0))

        self.reduction_steps = []
        current_seq_len = self.token_len
        while current_seq_len > self.MIN_SEQ_LEN:
            self.reduction_steps.append(current_seq_len)
            current_seq_len //= self.COMPRESS_FACTOR  # Integer division for sequence length

        if self.reduction_steps and self.reduction_steps[-1] != self.MIN_SEQ_LEN:
            self.reduction_steps.append(self.MIN_SEQ_LEN * self.COMPRESS_FACTOR)  # Add a step to get closer if needed

        self.reduction_seq_lens = sorted(list(set(self.reduction_steps)), reverse=True)  # Ensure unique and ordered
        self.compression_layers = nn.ModuleList([])

        prev_seq_len = self.token_len  # Keep track of previous sequence length
        for seq_len in self.reduction_seq_lens:
            nh = max(2, embed_dim // head_size)
            nh = max(2, nh) if nh >= 2 else 1

            t_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,  # Constant embed_dim
                nhead=nh,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
                dtype=torch.bfloat16,
                dropout=0.025,
                norm_first=True,
            )
            self.compression_layers.append(
                nn.ModuleList(
                    [
                        nn.TransformerEncoder(t_layer, num_layers=n_layers),
                        nn.Linear(prev_seq_len * embed_dim, seq_len * embed_dim).to(
                            torch.bfloat16
                        ),  # Linear for sequence length change if needed
                    ]
                )
            )
            prev_seq_len = seq_len  # Update previous seq len

        # Final compression to hypertoken size after sequence reduction
        self.final_compress = nn.Linear(self.MIN_SEQ_LEN * embed_dim, hypertoken_size).to(torch.bfloat16)

        self.to(torch.bfloat16)

    @typechecked
    def forward(self, x: TensorType["batch_size", "token_len"]) -> TensorType["batch_size", "hypertoken_size"]:
        batch_size, seq_len = x.size()

        embedded = self.embed(x) + self.pos_embed(self.positions)
        compressed = embedded

        current_seq_len = self.token_len
        for i in range(len(self.reduction_seq_lens)):
            transformer_layer, linear_resize = self.compression_layers[i]
            compressed = transformer_layer(compressed)

            target_seq_len = self.reduction_seq_lens[i]
            if current_seq_len != target_seq_len:  # Only resize if lengths are different
                compressed = compressed.reshape(
                    batch_size, current_seq_len * self.embed_dim
                )  # Flatten for linear layer
                compressed = linear_resize(compressed)  # Linear layer to change sequence length
                compressed = compressed.reshape(batch_size, target_seq_len, self.embed_dim)  # Reshape back

            current_seq_len = target_seq_len

        compressed = compressed.reshape(
            batch_size, self.MIN_SEQ_LEN * self.embed_dim
        )  # Flatten sequence and embed dims for final compression
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
        self.hypertoken_size = hypertoken_size
        self.target_seq_len = target_seq_len  # Match encoder's target

        self.COMPRESS_FACTOR = compress_factor  # Sequence compress factor
        self.MIN_SEQ_LEN = target_seq_len  # Target sequence length

        self.hypertoken = None
        self.fc_out = nn.Linear(embed_dim, vocab_size).to(torch.bfloat16)

        self.expansion_steps = []
        current_seq_len = self.MIN_SEQ_LEN
        while current_seq_len < self.token_len:
            self.expansion_steps.append(current_seq_len)
            current_seq_len *= self.COMPRESS_FACTOR

        if self.expansion_steps and self.expansion_steps[-1] != self.token_len:
            self.expansion_steps.append(self.token_len)  # Ensure final step is token_len

        self.expansion_seq_lens = sorted(list(set(self.expansion_steps)))  # Ensure unique and ordered
        self.expansion_layers = nn.ModuleList([])

        # Initial expansion from hypertoken to sequence of length MIN_SEQ_LEN
        self.initial_expand = nn.Linear(hypertoken_size, self.MIN_SEQ_LEN * embed_dim).to(torch.bfloat16)

        prev_seq_len = self.MIN_SEQ_LEN  # Track previous sequence length
        for seq_len in self.expansion_seq_lens:
            nh = max(2, embed_dim // head_size)

            t_layer = nn.TransformerEncoderLayer(  # Using EncoderLayer in Decoder is intentional as per original design
                d_model=embed_dim,
                nhead=nh,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
                dtype=torch.bfloat16,
                dropout=0.025,
                norm_first=True,
            )
            self.expansion_layers.append(
                nn.ModuleList(
                    [
                        nn.TransformerEncoder(t_layer, num_layers=n_layers),
                        nn.Linear(prev_seq_len * embed_dim, seq_len * embed_dim).to(
                            torch.bfloat16
                        ),  # Linear for sequence length change
                    ]
                )
            )
            prev_seq_len = seq_len  # Update previous sequence length

        self.pos_embed = nn.Embedding(self.token_len, embed_dim).to(
            torch.bfloat16
        )  # Positional embedding for full token_len
        self.register_buffer("positions", torch.arange(self.token_len).unsqueeze(0))

        self.to(torch.bfloat16)

    @typechecked
    def forward(
        self, x: TensorType["batch_size", "hypertoken_size"]
    ) -> TensorType["batch_size", "token_len", "vocab_size"]:
        batch_size = x.size(0)

        expanded = self.initial_expand(x)  # Project from hypertoken
        expanded = expanded.reshape(batch_size, self.MIN_SEQ_LEN, self.embed_dim)  # Reshape to initial sequence length

        current_seq_len = self.MIN_SEQ_LEN
        for i in range(len(self.expansion_seq_lens)):
            transformer_layer, linear_resize = self.expansion_layers[i]
            expanded = transformer_layer(expanded)

            target_seq_len = self.expansion_seq_lens[i]
            if current_seq_len != target_seq_len:  # Only resize if lengths are different
                expanded = expanded.reshape(batch_size, current_seq_len * self.embed_dim)  # Flatten for linear layer
                expanded = linear_resize(expanded)  # Linear layer to change sequence length
                expanded = expanded.reshape(batch_size, target_seq_len, self.embed_dim)  # Reshape back
            current_seq_len = target_seq_len

        # Add positional embeddings at the very end, for the full token_len sequence
        expanded = expanded + self.pos_embed(self.positions[:, : self.token_len])  # Ensure correct position indices

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
