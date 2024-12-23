import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import string
import random
from datasets import load_dataset

###########################################
# Configuration
###########################################
L_MAX = 20               # Maximum chunk length
LATENT_DIM = 256         # Dimension of the latent vector
EMB_DIM = 128            # Embedding dimension
NUM_ENCODER_LAYERS = 2   # Number of Transformer encoder layers
HIDDEN_DIM = 256         # Hidden dimension in feedforward layers
NHEAD = 4                # Number of attention heads
SEQ_LEN = 50             # Length of input character sequence
BATCH_SIZE = 16          # Batch size
EPOCHS = 10              # Number of training epochs
LR = 1e-3                # Learning rate

###########################################
# Load and Prepare Tiny Shakespeare Data
###########################################
def load_tiny_shakespeare():
    """
    Loads the Tiny Shakespeare dataset from Hugging Face's datasets library.
    
    Returns:
        str: Concatenated text from the dataset.
    """
    print("Loading Tiny Shakespeare dataset...")
    dataset = load_dataset("tiny_shakespeare", split="train")
    text_data = dataset["text"]
    # Concatenate all text into one string
    text_data = "\n".join(text_data)
    print(f"Total characters before filtering: {len(text_data)}")
    
    # Define allowed characters: lowercase letters, space, and basic punctuation
    allowed_chars = string.ascii_lowercase + " ,.!?;:'\"-\n"
    
    # Filter and clean text
    print("Filtering and cleaning text...")
    filtered_text = ''.join(c.lower() if c.lower() in allowed_chars else ' ' for c in text_data)
    filtered_text = ' '.join(filtered_text.split())  # Collapse multiple spaces
    print(f"Total characters after filtering: {len(filtered_text)}")
    
    return filtered_text

# Load the data
filtered_text = load_tiny_shakespeare()

###########################################
# Create Character Mappings
###########################################
def create_char_mappings(text):
    """
    Creates mappings from characters to indices and vice versa.
    
    Args:
        text (str): The text to create mappings from.
    
    Returns:
        dict, dict, int: char_to_idx, idx_to_char, vocabulary size.
    """
    unique_chars = sorted(list(set(text)))
    char_to_idx = {c: i+1 for i, c in enumerate(unique_chars)}  # 1-based indexing
    idx_to_char = {i+1: c for c, i in char_to_idx.items()}
    vocab_size = len(char_to_idx) + 1  # Include padding (0)
    print(f"Vocabulary size (including padding): {vocab_size}")
    return char_to_idx, idx_to_char, vocab_size

char_to_idx, idx_to_char, V = create_char_mappings(filtered_text)

###########################################
# Encode the Corpus
###########################################
def encode_corpus(text, char_to_idx):
    """
    Encodes the text into a list of integer indices.
    
    Args:
        text (str): The text to encode.
        char_to_idx (dict): Mapping from characters to indices.
    
    Returns:
        list: Encoded text as a list of integers.
    """
    return [char_to_idx.get(c, 1) for c in text]  # Unknowns map to 1

encoded_corpus = encode_corpus(filtered_text, char_to_idx)
print(f"Encoded corpus length: {len(encoded_corpus)}")

###########################################
# Generate Training Examples
###########################################
def get_random_example(seq_len=SEQ_LEN, l_max=L_MAX):
    """
    Generates a random training example from the encoded corpus.
    
    Args:
        seq_len (int): Length of the input sequence.
        l_max (int): Maximum length of the next chunk.
    
    Returns:
        torch.Tensor, torch.Tensor, int: input sequence, next chunk (padded), true length.
    """
    if len(encoded_corpus) < seq_len + l_max:
        raise ValueError("Corpus not long enough for given seq_len and l_max.")
    start = random.randint(0, len(encoded_corpus) - seq_len - l_max)
    input_seq = encoded_corpus[start:start + seq_len]
    true_len = random.randint(1, l_max)
    next_chunk = encoded_corpus[start + seq_len:start + seq_len + true_len]
    padded_chunk = next_chunk + [0] * (l_max - true_len)  # Pad with 0s
    return torch.tensor(input_seq, dtype=torch.long), torch.tensor(padded_chunk, dtype=torch.long), true_len

def get_batch(batch_size=BATCH_SIZE):
    """
    Generates a batch of training examples.
    
    Args:
        batch_size (int): Number of examples in the batch.
    
    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor: input sequences, next chunks, lengths.
    """
    inputs = []
    targets = []
    lengths = []
    for _ in range(batch_size):
        inp, tgt, l = get_random_example()
        inputs.append(inp)
        targets.append(tgt)
        lengths.append(l)
    return torch.stack(inputs), torch.stack(targets), torch.tensor(lengths, dtype=torch.long)

###########################################
# Model Definitions
###########################################
class EncoderModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, latent_dim, nhead, hidden_dim, num_layers, max_len):
        super(EncoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Average pooling over sequence length
        self.latent_proj = nn.Linear(emb_dim, latent_dim)
        self.size_predictor = nn.Linear(latent_dim, max_len)  # Predict chunk length
    
    def forward(self, x):
        """
        Forward pass of the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len].
        
        Returns:
            torch.Tensor, torch.Tensor: Latent vector [batch, latent_dim], size logits [batch, L_MAX].
        """
        emb = self.embedding(x)            # [batch, seq_len, emb_dim]
        emb = emb.permute(1, 0, 2)         # [seq_len, batch, emb_dim]
        enc_out = self.encoder(emb)        # [seq_len, batch, emb_dim]
        enc_out = enc_out.permute(1, 2, 0) # [batch, emb_dim, seq_len]
        pooled = self.pool(enc_out).squeeze(-1)  # [batch, emb_dim]
        z = self.latent_proj(pooled)       # [batch, latent_dim]
        size_logits = self.size_predictor(z)  # [batch, L_MAX]
        return z, size_logits

class DecoderModel(nn.Module):
    def __init__(self, vocab_size, latent_dim, emb_dim, hidden_dim, L_max):
        super(DecoderModel, self).__init__()
        self.latent_to_emb = nn.Linear(latent_dim, emb_dim)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, L_max * vocab_size)
        )
        self.vocab_size = vocab_size
        self.L_max = L_max
    
    def forward(self, z):
        """
        Forward pass of the decoder.
        
        Args:
            z (torch.Tensor): Latent vector [batch, latent_dim].
        
        Returns:
            torch.Tensor: Decoder output [batch, L_max, vocab_size].
        """
        h = self.latent_to_emb(z)         # [batch, emb_dim]
        out = self.decoder(h)             # [batch, L_max * vocab_size]
        out = out.view(-1, self.L_max, self.vocab_size)  # [batch, L_max, vocab_size]
        return out

###########################################
# Initialize Models and Optimizer
###########################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_encoder = EncoderModel(V, EMB_DIM, LATENT_DIM, NHEAD, HIDDEN_DIM, NUM_ENCODER_LAYERS, L_MAX).to(device)
model_decoder = DecoderModel(V, LATENT_DIM, EMB_DIM, HIDDEN_DIM, L_MAX).to(device)

# Combine parameters of both models
params = list(model_encoder.parameters()) + list(model_decoder.parameters())
optimizer = optim.Adam(params, lr=LR)

# Define loss functions
ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none')  # Ignore padding
length_ce = nn.CrossEntropyLoss()  # For size prediction

###########################################
# Training Loop
###########################################
for epoch in range(EPOCHS):
    model_encoder.train()
    model_decoder.train()
    
    total_losses = []
    reconstruction_losses = []
    size_losses = []
    incentive_losses = []
    
    for step in range(1, 1 + 1000):  # Adjust number of steps per epoch as needed
        # Generate a batch
        input_seq, next_token, next_token_len = get_batch(BATCH_SIZE)
        input_seq = input_seq.to(device)            # [batch, seq_len]
        next_token = next_token.to(device)          # [batch, L_MAX]
        next_token_len = next_token_len.to(device)  # [batch]
        
        optimizer.zero_grad()
        
        # Forward pass through encoder and decoder
        z, size_logits = model_encoder(input_seq)    # z: [batch, latent_dim], size_logits: [batch, L_MAX]
        dec_out = model_decoder(z)                   # [batch, L_max, V]
        
        # Compute Reconstruction Loss
        recon_loss = 0.0
        for i in range(BATCH_SIZE):
            L = next_token_len[i].item()
            if L > 0:
                # Decoder output for the first L characters
                pred_chars = dec_out[i, :L, :]     # [L, V]
                gold_chars = next_token[i, :L]     # [L]
                # Compute loss for each character
                loss_i = ce_loss(pred_chars, gold_chars)  # [L]
                recon_loss += loss_i.sum()
        
        # Average reconstruction loss over the batch
        recon_loss = recon_loss / BATCH_SIZE
        
        # Compute Size Prediction Loss
        # size_logits: [batch, L_MAX], target_length: [batch]
        # Targets should be in [0, L_MAX-1] representing lengths [1, L_MAX]
        size_targets = next_token_len - 1
        size_loss = length_ce(size_logits, size_targets)
        
        # Compute Length Incentive Loss
        # Encourage larger chunks if reconstruction is successful
        # Penalize choosing larger L if reconstruction fails
        # Here, we approximate 'successful' by having low reconstruction loss
        threshold = 0.1
        avg_L = next_token_len.float().mean()
        if recon_loss.item() < threshold:
            length_incentive = -0.001 * avg_L
        else:
            length_incentive = 0.001 * recon_loss * avg_L
        
        # Total Loss
        total_loss = recon_loss + size_loss + length_incentive
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses for reporting
        total_losses.append(total_loss.item())
        reconstruction_losses.append(recon_loss.item())
        size_losses.append(size_loss.item())
        incen
