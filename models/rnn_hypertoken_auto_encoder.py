import torch
from torch import nn
from torch.nn import functional as F


class RNNHyperTokenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        hypertoken_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.embed_dim = embed_dim
        self.hypertoken_size = hypertoken_size
        self.num_layers = num_layers

        # Embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim).to(torch.bfloat16)
        self.pos_embed = nn.Embedding(token_len, embed_dim).to(torch.bfloat16)
        self.embed_dropout = nn.Dropout(dropout)

        self.register_buffer("positions", torch.arange(token_len, dtype=torch.long).unsqueeze(0))

        # Simple RNN with larger hidden size
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=embed_dim * 2,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            nonlinearity="relu",  # ReLU for better gradient flow
        ).to(torch.bfloat16)

        # Compression layers
        self.compress = nn.Sequential(
            nn.Linear(embed_dim * 4, hypertoken_size * 2),
            nn.LayerNorm(hypertoken_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hypertoken_size * 2, hypertoken_size),
            nn.LayerNorm(hypertoken_size),
        ).to(torch.bfloat16)

    @typechecked
    def forward(self, x: TensorType["batch_size", "token_len"]) -> TensorType["batch_size", "hypertoken_size"]:
        batch_size, token_len = x.size()

        # Embedding with positional information
        embedded = self.embed(x) + self.pos_embed(self.positions)
        embedded = self.embed_dropout(embedded)

        # Process through RNN
        output, hidden = self.rnn(embedded)

        # Combine bidirectional hidden states
        hidden = hidden.view(self.num_layers, 2, batch_size, -1)
        final_hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=-1)

        # Compress to hypertoken size
        compressed = self.compress(final_hidden)
        return compressed


class RNNHyperTokenDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        hypertoken_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Initial projection from hypertoken
        self.expand = nn.Sequential(
            nn.Linear(hypertoken_size, embed_dim * 2),  # Reduced to match RNN hidden size
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        ).to(torch.bfloat16)

        # RNN layers
        self.hidden_size = embed_dim * 2  # Corrected hidden size to match expanded hypertoken dimension
        self.rnn = nn.RNN(
            input_size=embed_dim * 2,
            hidden_size=self.hidden_size,  # This will double with bidirectional
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
            nonlinearity="relu",
        ).to(torch.bfloat16)

        # Output projection
        self.project = nn.Sequential(
            nn.Linear(self.hidden_size * 2, embed_dim),  # Corrected input dimension to match bidirectional hidden output
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, vocab_size),
        ).to(torch.bfloat16)

        # Position embeddings for output sequence
        self.pos_embed = nn.Embedding(token_len, embed_dim * 2).to(torch.bfloat16)
        self.register_buffer("positions", torch.arange(token_len, dtype=torch.long))

    @typechecked
    def forward(self, x: TensorType["batch_size", "hypertoken_size"]) -> TensorType["batch_size", "token_len", "vocab_size"]:
        batch_size = x.size(0)

        # Expand hypertoken
        expanded = self.expand(x)  # [batch_size, embed_dim * 2]

        # Initialize hidden state for all layers and directions
        # We need [num_layers * num_directions, batch_size, hidden_size]
        h0 = expanded.unsqueeze(0)  # [1, batch_size, embed_dim * 2]
        h0 = h0.repeat(self.num_layers * 2, 1, 1)  # [num_layers * 2, batch_size, embed_dim * 2]
        # Split the embed_dim * 2 dimension to match RNN's hidden size
        # h0 = h0.view(self.num_layers * 2, batch_size, self.hidden_size) # No need to reshape as hidden_size is now embed_dim * 2

        # Create sequence input by repeating
        sequence_input = expanded.view(batch_size, 1, -1)  # [batch_size, 1, embed_dim * 2]
        sequence_input = sequence_input.repeat(1, self.token_len, 1)  # [batch_size, token_len, embed_dim * 2]

        # Add positional embeddings
        pos_embedded = self.pos_embed(self.positions)  # [token_len, embed_dim * 2]
        sequence_input = sequence_input + pos_embedded.unsqueeze(0)  # [batch_size, token_len, embed_dim * 2]

        # Process through RNN
        output, _ = self.rnn(sequence_input, h0)  # output: [batch_size, token_len, hidden_size * 2]

        # Project to vocabulary size
        logits = self.project(output)  # [batch_size, token_len, vocab_size]

        # Project to vocabulary size
        logits = self.project(output)
        return logits


class RNNHyperTokenAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_len: int,
        embed_dim: int,
        hypertoken_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = RNNHyperTokenEncoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=embed_dim,
            hypertoken_size=hypertoken_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = RNNHyperTokenDecoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=embed_dim,
            hypertoken_size=hypertoken_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.to(torch.bfloat16)

    @typechecked
    def forward(self, x: TensorType["batch_size", "token_len"]) -> TensorType["batch_size", "token_len", "vocab_size"]:
        self.hypertoken = self.encoder(x)
        decoded = self.decoder(self.hypertoken)
        return decoded
