from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch


# --------------------------------------------------
# 1. Data Preparation
# --------------------------------------------------
class TinyShakespeareDataset(Dataset):
    def __init__(self, seq_len: int = 128):
        # Load TinyShakespeare from Hugging Face
        dataset = load_dataset("tiny_shakespeare")
        text = dataset["train"]["text"][0]  # Get the text content

        # Build vocabulary
        chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

        # Add padding token
        self.pad_token = len(chars)
        self.char2idx["[PAD]"] = self.pad_token
        self.idx2char[self.pad_token] = "[PAD]"

        # Convert text to indices
        self.data = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - 101  # Ensure we have at least one char after position 100

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate random sequence length between 1 and seq_len
        rand_len = torch.randint(1, min(self.seq_len + 1, len(self.data) - idx), (1,)).item()

        # Create padded sequence of max length
        x = torch.full((self.seq_len,), self.pad_token, dtype=torch.long)

        sequence = self.data[idx : idx + rand_len]

        # Place sequence in padded tensor
        if len(sequence) > 28:
            overflow = len(sequence) - 28
            x[100 - overflow : 100 + len(sequence) - overflow] = sequence
        else:
            x[100 : 100 + len(sequence)] = sequence
        y = x.clone()

        return x, y


# --------------------------------------------------
# 2. Model Definition
# --------------------------------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, seq_len=128, n_heads=2, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Basic positional encoding
        self.pos_embed = nn.Embedding(seq_len, embed_dim)

        # Transformer: N layers, multi-head self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=64,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Additional layers to reshape transformer output to [batch_size, 28, vocab_size]
        self.fc1 = nn.Linear(embed_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, vocab_size)

        # To map the sequence length from 128 to 28, use Adaptive Average Pooling
        self.pool = nn.AdaptiveAvgPool1d(28)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]

        # Sum char embedding + positional embedding
        x_emb = self.embed(x) + self.pos_embed(positions)  # [batch_size, seq_len, embed_dim]

        # Transformer forward
        transformer_out = self.transformer(x_emb)  # [batch_size, seq_len, embed_dim]

        residual = x_emb
        transformer_out = transformer_out + residual

        # Permute for pooling: [batch_size, embed_dim, seq_len]
        out = transformer_out.permute(0, 2, 1)

        # Pool to reduce sequence length from 128 to 28
        out = self.pool(out)  # [batch_size, embed_dim, 28]

        # Permute back to [batch_size, 28, embed_dim]
        out = out.permute(0, 2, 1)

        # Apply additional linear layers
        out = self.fc1(out)  # [batch_size, 28, 64]
        out = self.relu(out)  # [batch_size, 28, 64]
        logits = self.fc2(out)  # [batch_size, 28, vocab_size]

        return logits


# --------------------------------------------------
# 3. Training Loop
# --------------------------------------------------
def train_model(epochs=3, batch_size=128, lr=1e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TinyShakespeareDataset(seq_len=128)
    vocab_size = len(dataset.char2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerModel(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    current_lr = lr

    for epoch in range(epochs):

        optimizer.param_groups[0]["lr"] = current_lr
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Forward
            logits = model(x)

            # Convert input tensor to text
            input_indexes = x[0]
            preBoundary = "".join([dataset.idx2char[idx.item()] for idx in input_indexes[:100]])
            postBoundary = "".join([dataset.idx2char[idx.item()] for idx in input_indexes[100:]])

            # Get model predictions and convert to text
            predictions = torch.argmax(logits[0], dim=-1)
            output_text = "".join([dataset.idx2char[idx.item()] for idx in predictions])

            print("\nInput text:")
            print(preBoundary + "<|>" + postBoundary)
            print("\nModel output:")
            print(output_text)
            print("-" * 80)

            loss_per_pos = criterion(logits.reshape(-1, vocab_size), y[:, -28:].reshape(-1))

            loss = loss_per_pos.mean()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - LR: {current_lr:.6f}")

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - LR: {current_lr:.6f}")
        current_lr = current_lr * 0.1
    return model


if __name__ == "__main__":
    # Example usage
    trained_model = train_model()

    # Inference example (greedy sampling for demonstration)
    test_input = "that didn't really seem to "
    # Convert to indices
    # ... etc. (omitted to keep script concise)
    print("Training complete.")
