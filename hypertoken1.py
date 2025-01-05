from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
import math
import random
from helpers.experiments import run_experiment, ExperimentConfig, get_memory_gb
# --------------------------------------------------
# 1. Data Preparation
# --------------------------------------------------
class TinyShakespeareDataset(Dataset):
    def __init__(self, encode_last_n_length: int,segments:int, seq_len: int = 128,type:str = "train"):
        # Load TinyShakespeare from Hugging Face
        dataset = load_dataset("tiny_shakespeare")

        train_text = dataset['train']['text'][0]
        val_text = dataset['validation']['text'][0]
        test_text = dataset['test']['text'][0]

        all_text = train_text + val_text + test_text

        text = train_text if type == "train" else val_text if type == "validation" else test_text
        
        # Build vocabulary
        chars = sorted(list(set(all_text)))
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
        self.segments = segments
        self.type = type
    def __len__(self) -> int:


        if self.type == "train":
            return math.floor(len(self.data)/self.segments)
        else:
            return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        final_index = 0

        if self.type == "train":
            final_index = random.randint(0,len(self.data)-1)
        else:
            final_index = idx

        # Calculate random sequence length between 1 and seq_len
        rand_len = random.randint(1, min(self.seq_len, len(self.data) - final_index))
        
        # Create padded sequence of max length
        x = torch.full((self.seq_len,), self.pad_token, dtype=torch.long)
        
        if final_index < 0:
            print("final_index < 0")

        if final_index + rand_len > len(self.data):
            print("final_index + rand_len > len(self.data)")

        sequence = self.data[final_index:final_index + rand_len]

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
        self.embed_dim = embed_dim
        
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
        
        # --- Multiscale Convolutions ---
        # Parallel convolutions with different kernel sizes
        self.conv_k3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        
        # Optional second stage of strided convolution
        # (comment out if you don’t need extra downsampling)
        self.conv_down = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        
        # --- Linear compression to hypertoken ---
        # After 2 stride=2 ops, sequence length ~ seq_len / 4
        self.fc_compress = nn.Linear(embed_dim * (seq_len // 4), hypertoken_size)
        
        # --- Smaller Expand (hypertoken_size -> decoder_seq_len * embed_dim) ---
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
        
        # Prepare for conv: (bsz, channels=embed_dim, seq_len)
        enc_out = enc_out.permute(0, 2, 1)

        try:

            #xmean = torch.mean(x_enc.flatten(1),dim=1)

            # --- Multiscale Convolutions (parallel) ---
            conv_out = self.conv_k3(enc_out)

            # Optional second downsampling
            downsampled = self.conv_down(conv_out)  # (bsz, embed_dim, seq_len/4)

            # Flatten for linear compression
            downsampled = downsampled.view(batch_size, -1)   # (bsz, embed_dim * seq_len/4)
            
            # --- Compress to hypertoken ---
            hypertoken = self.fc_compress(downsampled) 
            

            #hypertoken_mean = torch.mean(hypertoken,dim=1)

            # --- Expand + Decoder Input ---
            dec_in = self.fc_expand(hypertoken) #+ hypertoken_mean.unsqueeze(1)    # (batch, decoder_seq_len * embed_dim)
            dec_in = dec_in.view(-1, self.decoder_seq_len, self.embed_dim)
            
            # --- Decode & Project ---
            dec_out = self.decoder(dec_in) #+ hypertoken_mean.unsqueeze(1).unsqueeze(2)      # (batch, decoder_seq_len, embed_dim)
            logits = self.fc_out(dec_out)  #+ hypertoken_mean.unsqueeze(1).unsqueeze(2)        # (batch, decoder_seq_len, vocab_size)                              
            
            return logits
        except Exception as e:
            print(e)
            return None





# --------------------------------------------------
# 3. Training Loop
# --------------------------------------------------
def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str) -> float:
    """Evaluate model on given dataloader"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(
                logits.reshape(-1, len(dataloader.dataset.char2idx)),
                y[:, -dataloader.dataset.encode_last_n_length:].reshape(-1)
            ).mean()
            total_loss += loss.item()
            batch_count += 1
    
    model.train()
    return total_loss / batch_count



def train_model(wandb, epochs=3, batch_size=512, encode_last_n_length=4, seq_len=128, 
                hypertoken_size=16, n_heads=2, n_layers=2, embed_dim=16, lr=.0025):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    segments = 10
    
    dataset = TinyShakespeareDataset(encode_last_n_length,segments=segments,seq_len=seq_len)
    vocab_size = len(dataset.char2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TinyShakespeareDataset(encode_last_n_length,segments=segments,seq_len=seq_len,type="validation")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = TransformerAutoencoder(vocab_size=vocab_size, seq_len=seq_len, encode_last_n_length=encode_last_n_length, hypertoken_size=hypertoken_size, n_heads=n_heads, n_layers=n_layers, embed_dim=embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    current_lr = lr

    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        optimizer.param_groups[0]['lr'] = current_lr

        for segment in range(dataset.segments):
            segment_loss = 0
            segment_batch_count = 0
            
            for x, y in dataloader:
                batch_count += 1
                segment_batch_count += 1
                x, y = x.to(device), y.to(device)
                
                # Forward pass and loss calculation
                logits = model(x)
                loss_per_pos = criterion(
                    logits.reshape(-1, vocab_size),
                    y[:, -encode_last_n_length:].reshape(-1)
                )
                loss = loss_per_pos.mean()

                # Update metrics
                segment_loss += loss.item()
                epoch_loss += loss.item()

                # Log batch metrics
                if batch_count % 10 == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "learning_rate": current_lr,
                        "epoch": epoch,
                    })

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Log segment metrics
            avg_segment_loss = segment_loss / segment_batch_count

        # Log epoch metrics
        val_loss = evaluate_model(model, val_dataloader, criterion, device)

        wandb.log({
                    "epoch_loss": epoch_loss/batch_count,
                    "val_loss": val_loss,
                })

        print(f"Epoch {epoch} train_loss: {avg_segment_loss:.4f}, val_loss: {val_loss:.4f}")

        current_lr = current_lr * .1


    return model


if __name__ == "__main__":

    # Define experiments
    experiments: list[ExperimentConfig] = [
        {
            "seq_len": 32,
            "encode_last_n_length": 32,
            "hypertoken_size": 64,
            "epochs": 2,
            "batch_size": 512,
            "lr": 0.0001,
            "n_heads": 4,
            "n_layers": layers,
            "embed_dim": 128
        }
        for layers in [1, 2, 3, 4]  # Varying n_heads
    ]
  
  
    for experiment in experiments:
        run_experiment("HyperTokens",train_model,experiment)
    
    
    # Inference example (greedy sampling for demonstration)
    test_input = "that didn't really seem to "
    # Convert to indices
    # ... etc. (omitted to keep script concise)
    print("Training complete.")
