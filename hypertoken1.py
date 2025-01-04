from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch

# --------------------------------------------------
# 1. Data Preparation
# --------------------------------------------------
class TinyShakespeareDataset(Dataset):
    def __init__(self, encode_last_n_length: int, seq_len: int = 128):
        # Load TinyShakespeare from Hugging Face
        dataset = load_dataset("tiny_shakespeare")
        text = dataset['train']['text'][0]  # Get the text content
        
        # Build vocabulary
        chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        
        # Add padding token
        self.pad_token = len(chars)
        self.char2idx['<PAD>'] = self.pad_token
        self.idx2char[self.pad_token] = '<PAD>'
        
        self.encode_last_n_length = encode_last_n_length

        # Convert text to indices
        self.data = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)
        self.seq_len = seq_len
        
    def __len__(self) -> int:
        return len(self.data) - 1  # Ensure we have at least one char after position 100

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate random sequence length between 1 and seq_len
        rand_len = torch.randint(1, min(self.seq_len + 1, len(self.data) - idx), (1,)).item()
        
        # Create padded sequence of max length
        x = torch.full((self.seq_len,), self.pad_token, dtype=torch.long)
        
        sequence = self.data[idx:idx + rand_len]

        boundary_index = self.seq_len - self.encode_last_n_length

        # Place sequence in padded tensor
        if len(sequence) > self.encode_last_n_length:
            overflow = len(sequence) - self.encode_last_n_length
            x[boundary_index-overflow:boundary_index+len(sequence)-overflow] = sequence
        else:
            x[boundary_index:boundary_index + len(sequence)] = sequence
        y = x.clone()
        
        return x, y

# --------------------------------------------------
# 2. Model Definition
# --------------------------------------------------
class TransformerAutoencoder(nn.Module):
    def __init__(
        self, 
        vocab_size,
        encode_last_n_length,
        embed_dim=16,
        seq_len=128,
        
        n_heads=2,
        n_layers=2,
        hypertoken_size=32
    ):
        super().__init__()
        self.seq_len = seq_len
        self.decoder_seq_len = encode_last_n_length  # Fixed sequence length for decoder output
        
        # --- Embedding + Positional Encoding ---
        self.embed_enc = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed_enc = nn.Embedding(seq_len, embed_dim)

        # --- Encoder Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # --- Compression to hypertoken ---
        # Remove pooling and use a linear layer instead
        self.fc_compress = nn.Linear(seq_len * embed_dim, hypertoken_size)
        
        # --- Decoder Positional Encoding (encode_last_n_length tokens) ---
        self.pos_embed_dec = nn.Embedding(self.decoder_seq_len, embed_dim)

        # --- Decoder Expansion ---
        self.fc_expand = nn.Linear(hypertoken_size, self.decoder_seq_len * embed_dim)
        
        # --- Decoder Transformer ---
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=64,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        
        # --- Final Projection to vocab_size ---
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        returns: [batch_size, encode_last_n_length, vocab_size]
        """
        batch_size, seq_len = x.size()
        
        # ---------------------
        # 1) Encoder
        # ---------------------
        positions_enc = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        x_enc = self.embed_enc(x) + self.pos_embed_enc(positions_enc)        # [batch_size, seq_len, embed_dim]
        enc_out = self.encoder(x_enc)                                        # [batch_size, seq_len, embed_dim]
        
        # Reshape encoder output for compression
        enc_out_flat = enc_out.reshape(batch_size, -1)                      # [batch_size, seq_len * embed_dim]
        hypertoken = self.fc_compress(enc_out_flat)                          # [batch_size, hypertoken_size]
        
        # ---------------------
        # 2) Decoder
        # ---------------------
        # Expand hypertoken back to [batch_size, encode_last_n_length * embed_dim]
        dec_expanded = self.fc_expand(hypertoken)                            # [batch_size, encode_last_n_length * embed_dim]
        dec_expanded = dec_expanded.view(batch_size, self.decoder_seq_len, -1)  # [batch_size, encode_last_n_length, embed_dim]
        
        # Add positional encoding for decoder
        #positions_dec = torch.arange(self.decoder_seq_len, device=x.device).unsqueeze(0)  # [1, encode_last_n_length]
        dec_in = dec_expanded #+ self.pos_embed_dec(positions_dec)                        # [batch_size, encode_last_n_length, embed_dim]
        
        # Pass through decoder Transformer
        dec_out = self.decoder(dec_in)                                                 # [batch_size, encode_last_n_length, embed_dim]
        
        # Final projection to vocab size
        logits = self.fc_out(dec_out)                                                  # [batch_size, encode_last_n_length, vocab_size]
        
        return logits




def count_parameters(model: nn.Module) -> tuple[int, dict]:
    """
    Count total trainable parameters and parameters per layer
    Returns: (total_params, params_by_layer)
    """
    params_by_layer = {
        name: p.numel() for name, p in model.named_parameters() if p.requires_grad
    }
    total_params = sum(params_by_layer.values())
    
    # Format large numbers with commas
    formatted_layers = {
        name: f"{count:,}" for name, count in params_by_layer.items()
    }
    
    print(f"\nTotal trainable parameters: {total_params:,}")
    # print("\nParameters per layer:")
    # for name, count in formatted_layers.items():
    #     print(f"{name}: {count}")
        
    return total_params, params_by_layer




# --------------------------------------------------
# 3. Training Loop
# --------------------------------------------------
def train_model(epochs=3, batch_size=512, encode_last_n_length=4,seq_len=128, hypertoken_size=16, n_heads=2, n_layers=2, embed_dim=16, lr=.0025):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    dataset = TinyShakespeareDataset(encode_last_n_length,seq_len=seq_len)
    vocab_size = len(dataset.char2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TransformerAutoencoder(vocab_size=vocab_size, seq_len=seq_len, encode_last_n_length=encode_last_n_length, hypertoken_size=hypertoken_size, n_heads=n_heads, n_layers=n_layers, embed_dim=embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
        

    count_parameters(model)

    current_lr = lr

    for epoch in range(epochs):
    
        batch_count = 0
        optimizer.param_groups[0]['lr'] = current_lr
        for x, y in dataloader:
            batch_count += 1
            x, y = x.to(device), y.to(device)
            
            # Forward
            logits = model(x)

            boundary_index = x.shape[1] - encode_last_n_length
            
            # Convert input tensor to text
            input_indexes = x[0] 
            preBoundary = ''.join([dataset.idx2char[idx.item()] for idx in input_indexes[:boundary_index]])
            postBoundary = ''.join([dataset.idx2char[idx.item()] for idx in input_indexes[boundary_index:]])

            
            # Get model predictions and convert to text
            predictions = torch.argmax(logits[0], dim=-1)
            output_text = ''.join([dataset.idx2char[idx.item()] for idx in predictions])
       
             
            loss_per_pos = criterion(
                logits.reshape(-1, vocab_size),
                 y[:, -encode_last_n_length:].reshape(-1)
            )

            loss = loss_per_pos.mean()

            if batch_count % 1000 == 0:
                print("\nInput text:")
                print(preBoundary + "<|>" + postBoundary)
                print("\nTarget text:" + postBoundary)
                print("\nModel output:" + output_text)
                print("-" * 80)
                print(f"Epoch {epoch+1}/{epochs}/Batch Count: {batch_count} - Loss: {loss.item():.4f} - LR: {current_lr:.6f}")

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - LR: {current_lr:.6f}")
        current_lr = current_lr * .1
    return model


if __name__ == "__main__":

    train_model_params = {
        'seq_len':32,
        'encode_last_n_length':32,
        'hypertoken_size':64,
        'epochs':2,
        'batch_size':512,
        'lr':.0001,
        'n_heads':1,
        'n_layers':4,
        'embed_dim':256
    }
  
    trained_model = train_model(**train_model_params)

    train_model_params['n_heads'] = 2


    trained_model = train_model(**train_model_params)

    train_model_params['n_heads'] = 4
    

    trained_model = train_model(**train_model_params)

    train_model_params['n_heads'] = 8

    trained_model = train_model(**train_model_params)





    
    # Inference example (greedy sampling for demonstration)
    test_input = "that didn't really seem to "
    # Convert to indices
    # ... etc. (omitted to keep script concise)
    print("Training complete.")
