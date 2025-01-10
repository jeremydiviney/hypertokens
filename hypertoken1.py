from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
import math
import random
from helpers.experiments import run_experiment, ExperimentConfig, get_memory_gb, count_parameters
from torch.amp import autocast, GradScaler
from lion_pytorch import Lion
from torch.utils.checkpoint import checkpoint

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


# 2. Enable torch.backends optimizations
def enable_torch_optimizations():
    if torch.cuda.is_available():
        # Enable TF32 for faster matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def setup_flash_attention():
    # Enable Flash Attention if available
    if torch.cuda.is_available():
        flash_available = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
        print(f"Flash Attention available and enabled: {flash_available}")
        # Enable Flash Attention
        torch.backends.cuda.enable_flash_sdp(True)
        # Enable Math Flash Attention (more efficient math operations)
        torch.backends.cuda.enable_math_sdp(True)
        # Enable Memory Efficient Attention
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        return flash_available
    print("CUDA not available, Flash Attention disabled")
    return False       




# --------------------------------------------------
# 2. Model Definition
# --------------------------------------------------

class TransformerAutoencoder(nn.Module):
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
        mode: str = "autoencoder"
    ) -> None:
        super().__init__()
        self.mode = mode
        self.seq_len = seq_len
        self.encode_last_n_length = encode_last_n_length
        self.embed_dim = embed_dim
        self.hypertoken_size = hypertoken_size

        self.average_memory_usage = 0
       
        # Add checkpointing flag
        self.use_checkpointing = False
        # Verify BF16 is supported
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            print("Warning: BF16 not supported on this device, falling back to FP32")
            return

        # Embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)
  
        self.COMPRESS_FACTOR = compress_factor
        self.MIN_EMBED_DIM = hypertoken_size//self.seq_len
              
        # Register a buffer so it moves to the right device (CPU/GPU),
        # but won't be updated by the optimizer.
        self.register_buffer("positions", torch.arange(seq_len).unsqueeze(0))

        # self.use_checkpointing = True  # Add flag for checkpointing

        if self.mode in {"encoder", "autoencoder"}:

            compression_sizes = []
            current_size = self.embed_dim
            while current_size > self.MIN_EMBED_DIM:
                compression_sizes.append(current_size)
                current_size //= self.COMPRESS_FACTOR

            if compression_sizes[-1] != hypertoken_size//self.seq_len:
                compression_sizes.append(self.MIN_EMBED_DIM)
            
            self.compression_sizes = list(zip(compression_sizes[:-1], compression_sizes[1:]))

            # Create progressive transformer layers
            self.compression_layers = nn.ModuleList([])

            for in_dim, out_dim in self.compression_sizes:
                nh = out_dim // head_size
                nh = max(2, nh) if nh >= 2 else 1

                t_layer = nn.TransformerEncoderLayer(
                    d_model=in_dim,
                    nhead=nh,  # Ensure reasonable number of heads
                    dim_feedforward=in_dim * 4,
                    batch_first=True,
                    dtype=torch.bfloat16,
                    dropout=0.025,
                    norm_first=True
                )

                self.compression_layers.append(
                    nn.TransformerEncoder(t_layer, num_layers=n_layers)
                )

        if self.mode in {"decoder", "autoencoder"}:
             
            self.fc_out = nn.Linear(embed_dim, vocab_size)

            expanssion_sizes = []
            current_size = self.compression_sizes[-1][1]
            while current_size < self.embed_dim:
                expanssion_sizes.append(current_size)
                current_size *= self.COMPRESS_FACTOR
            expanssion_sizes.append(self.embed_dim)

            if expanssion_sizes[-1] != self.embed_dim:
                expanssion_sizes.append(self.embed_dim)
            
            self.expansion_sizes = list(zip(expanssion_sizes[:-1], expanssion_sizes[1:]))
            # Create progressive transformer layers
            self.expansion_layers = nn.ModuleList([])
         

            for in_dim, out_dim in self.expansion_sizes:
                
                nh = max(2, out_dim // head_size)

                t_layer = nn.TransformerEncoderLayer(
                    d_model=out_dim,
                    nhead=nh,  # Ensure reasonable number of heads
                    dim_feedforward=out_dim * 4,
                    batch_first=True,
                    dtype=torch.bfloat16,
                    dropout=0.025,
                    norm_first=True
                )

                self.expansion_layers.append(
                    nn.TransformerEncoder(t_layer, num_layers=n_layers)
                )

    # def forward(self,x):
    #     if self.use_checkpointing and self.training:
    #         return checkpoint(self._forward, x)

    #     return self._forward(x)

    def check_memory_usage(self):
        if torch.cuda.is_available():
            current_mem = float(torch.cuda.memory_allocated())/1e9
            #max_mem = float(torch.cuda.max_memory_allocated())/1e9
            #print("Current memory: {:.2f}GB".format(current_mem))
            #print("Max memory: {:.2f}GB".format(max_mem))
            self.average_memory_usage = (self.average_memory_usage + current_mem)/2
            #print("Average memory usage: {:.2f}GB".format(self.average_memory_usage))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        
        #print("Starting forward pass")
        #self.print_memory()

        if self.mode in {"encoder", "autoencoder"}:

            #positions = self.positions[:, :seq_len]  # handle seq_len if needed
            embedded = self.embed(x) + self.pos_embed(self.positions)

            compressed = embedded

            # Pass through progressive compression transformer layers
            for i, (dim_in,dim_out) in enumerate(self.compression_sizes):
                # Apply checkpointing to transformer layers
                if self.use_checkpointing and self.training and dim_in >= 128:
                    compressed = checkpoint(self.compression_layers[i], compressed)
                else:
                    compressed = self.compression_layers[i](compressed)


                compress_factor = dim_in//dim_out
                compressed = compressed.reshape(batch_size, seq_len, -1, compress_factor).sum(dim=-1)
   
   
            hypertoken = compressed.flatten(start_dim=1)
        
            if self.mode == "encoder":
                return hypertoken
        
        if self.mode in {"decoder", "autoencoder"}:

            if self.mode == "decoder":
                hypertoken = x


            expanded = hypertoken

            # Start with hypertoken expanded across sequence length
            expanded = expanded.reshape(batch_size, self.encode_last_n_length, -1)
            
            # Pass through progressive expansion layers
            for sub_layer_index, (dim_in,dim_out) in enumerate(self.expansion_sizes):

                expand_factor = dim_out//dim_in
                expanded = (
                    expanded
                    .unsqueeze(-1)
                    .expand(-1, -1, -1, expand_factor)
                    .reshape(batch_size, self.encode_last_n_length, dim_out)
                    * (1.0 / expand_factor)
                )

                # Apply checkpointing to transformer layers
                if self.use_checkpointing and self.training and dim_out >= 128:
                    expanded = checkpoint(self.expansion_layers[sub_layer_index], expanded)
                else:
                    expanded = self.expansion_layers[sub_layer_index](expanded)

                 # Mirror compression logic but expand instead
   
                # expanded = self.expansion_layers[sub_layer_index](expanded) #linear layer
                # sub_layer_index += 1
                
            self.check_memory_usage()

            logits = self.fc_out(expanded)

            return logits
        
     

def batch_tensor_to_text(batch_tensor: torch.Tensor, idx2char: dict) -> list[str]:
    """Convert batch of tensors to text efficiently by moving data to CPU once"""
    # Move entire tensor to CPU at once and convert to numpy
    sequences = batch_tensor.cpu().numpy()
    pad_token = len(idx2char) - 1
    
    # Process all sequences at once
    ret = [
        ''.join(idx2char[idx] for idx in seq if idx != pad_token)
        for seq in sequences
    ]
    
    return ret


# --------------------------------------------------
# 3. Training Loop
# --------------------------------------------------
def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str) -> dict:
    """Evaluate model on given dataloader"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    total_loss = 0
    batch_count = 0
    exact_matches = 0
    total_samples = 0
    # Add character-level tracking
    matching_chars = 0
    total_chars = 0
     
    with torch.inference_mode(), autocast("cuda", dtype=torch.bfloat16):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(
                logits.reshape(-1, len(dataloader.dataset.char2idx)),
                y[:, -dataloader.dataset.encode_last_n_length:].reshape(-1)
            ).mean()
            total_loss += loss.item()
            batch_count += 1

            pred_logits = logits[:, -dataloader.dataset.encode_last_n_length:]
            pred_indices = torch.argmax(pred_logits, dim=-1)
            
            target_seqs = y[:, -dataloader.dataset.encode_last_n_length:]
            target_texts = batch_tensor_to_text(target_seqs, dataloader.dataset.idx2char)
            pred_texts = batch_tensor_to_text(pred_indices, dataloader.dataset.idx2char)
            
            # Count exact matches
            exact_matches += sum(1 for pred, target in zip(pred_texts, target_texts) if pred == target)
            total_samples += len(pred_texts)

            # Add character-level accuracy calculation
            for pred, target in zip(pred_texts, target_texts):
                total_chars += len(target)
                matching_chars += sum(p == t for p, t in zip(pred, target))

            if batch_count % 10 == 0:    
                print(f"\nSample {batch_count}:")
                print(f"Target: {target_texts[0]}")
                print(f"Pred:   {pred_texts[0]}")
                print(f"Current sequence accuracy: {exact_matches/total_samples:.2%}")
                print(f"Current character accuracy: {matching_chars/total_chars:.2%}")
    
    model.train()
    return {
        "val_loss": total_loss / batch_count,
        "val_sequence_accuracy": exact_matches/total_samples,
        "val_char_accuracy": matching_chars/total_chars
    }



def train_model(wandb, epochs=3, batch_size=512, encode_last_n_length=4, seq_len=128, 
                hypertoken_size=16, head_size=32, compress_factor=2, n_layers=2, embed_dim=16, lr=.0025):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    segments = 10
    
    dataset = TinyShakespeareDataset(encode_last_n_length,segments=segments,seq_len=seq_len)
    vocab_size = len(dataset.char2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TinyShakespeareDataset(encode_last_n_length,segments=segments,seq_len=seq_len,type="validation")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = TransformerAutoencoder(
                vocab_size=vocab_size, seq_len=seq_len, encode_last_n_length=encode_last_n_length, 
                hypertoken_size=hypertoken_size, head_size=head_size, compress_factor=compress_factor, 
                n_layers=n_layers, embed_dim=embed_dim
            ).to(device)

    model = torch.compile(model)

    count_parameters(model)


    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # optimizer = Lion(
    #     model.parameters(),
    #     lr=lr,  # Usually needs 3-10x smaller learning rate than Adam
    #     weight_decay=1e-2  # Lion typically works better with higher weight decay
    # )
    
    criterion = nn.CrossEntropyLoss()

    total_steps = epochs * len(dataloader) * segments

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.2,  # Use 30% of steps for warmup
        anneal_strategy='cos',
        cycle_momentum=False
    )


    current_lr = lr

    #evaluate_model(model, val_dataloader, criterion, device)

    low_loss = 10000

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

                with autocast(device_type='cuda', dtype=torch.bfloat16):

                    logits = model(x)
                    loss_per_pos = criterion(
                        logits.reshape(-1, vocab_size),
                        y[:, -encode_last_n_length:].reshape(-1)
                    )
                    loss = loss_per_pos.mean()

                # Update metrics
                segment_loss += loss.item()
                epoch_loss += loss.item()

                if loss.item() < low_loss:
                    low_loss = loss.item()
                    print(f"New low loss: {low_loss:.7f}")

                # Log batch metrics
                if batch_count % 10 == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "epoch": epoch,
                    })

                 # Standard backward pass (no scaling needed for BF16)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                optimizer.step()
                
                scheduler.step()

            # Log segment metrics
            avg_segment_loss = segment_loss / segment_batch_count

        # Log epoch metrics
        eval_results = evaluate_model(model, val_dataloader, criterion, device)
        val_loss = eval_results["val_loss"]
        val_sequence_accuracy = eval_results["val_sequence_accuracy"]
        val_char_accuracy = eval_results["val_char_accuracy"]


        wandb.log({
                    "epoch_loss": epoch_loss/batch_count,
                    "val_loss": val_loss,
                    "val_sequence_accuracy": val_sequence_accuracy,
                    "val_char_accuracy": val_char_accuracy
                })

        print(f"Epoch {epoch} train_loss: {avg_segment_loss:.4f}, val_loss: {val_loss:.4f}, val_sequence_accuracy: {val_sequence_accuracy:.2%}, val_char_accuracy: {val_char_accuracy:.2%}")

    return model


def verify_hyperparameters(hs,ed,n_layers,head_size,lr,seq_len,hypertoken_size,compress_factor,encode_last_n_length):

    print(f"Verifying hyperparameters \n\
            hypertoken_size: {hypertoken_size}, \n\
            seq_len: {seq_len}, \n\
            encode_last_n_length: {encode_last_n_length}")

    if hypertoken_size < seq_len:
        raise ValueError("hypertoken_size must be greater than or equal to seq_len")

    if hypertoken_size < encode_last_n_length:
        raise ValueError("encode_last_n_length must be greater than or equal to hypertoken_size")

    # Add check for embed_dim being multiple of seq_len
    if hypertoken_size % seq_len != 0:
        raise ValueError(f"hypertoken_size must be a multiple of seq_len")

    if hypertoken_size % encode_last_n_length != 0:
        raise ValueError(f"hypertoken_size must be a multiple of encode_last_n_length")

    # Check if embed_dim is a power of compress_factor
    # if not (math.log(ed, compress_factor).is_integer()):
    #     raise ValueError(f"Embed_dim ({ed}) must be a power of compress_factor ({compress_factor})")

    if encode_last_n_length > seq_len:
        raise ValueError("encode_last_n_length must be less than or equal to seq_len")


if __name__ == "__main__":

    # Define experiments
    experiments: list[ExperimentConfig] = [
        {
            "seq_len": 128,
            "encode_last_n_length": 128,
            "hypertoken_size": hs,
            "epochs": 1,
            "batch_size": 512,
            "lr": lr,
            "head_size": head_size,
            "n_layers": n_layers,
            "embed_dim": ed,
            "compress_factor": cf
        }
        for hs in [512]  # Varying hypertoken_size
        for ed in [512]  # Varying embed_dim
        for n_layers in [1]  # Varying n_layers
        for head_size in [64]  # Varying head_size
        for lr in [0.001]
        for cf in [4]
        
    ]

    for experiment in experiments:
        verify_hyperparameters(
            experiment["hypertoken_size"],experiment["embed_dim"],
            experiment["n_layers"],experiment["head_size"],experiment["lr"],
            experiment["seq_len"],experiment["hypertoken_size"],
            experiment["compress_factor"],experiment["encode_last_n_length"]
        )

    enable_torch_optimizations()
    setup_flash_attention()
  
    for experiment in experiments:
        run_experiment("HyperTokens",train_model,experiment)
    

#TODO: fix gpu memory reporting

