import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch

from helpers.experiments import run_experiment, ExperimentConfig
from torch.amp import autocast, GradScaler
from lion_pytorch import Lion
from torch.utils.checkpoint import checkpoint
import os
from typing import Optional
from models.hypertoken_auto_encoder import HyperTokenEncoder, HyperTokenDecoder, HyperTokenAutoencoder
from datetime import datetime
from data.tinyshakespeare import TinyShakespeareDataset
from helpers.training import train_model


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
# def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str) -> dict:
#     """Evaluate model on given dataloader"""
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     model.eval()
#     total_loss = 0
#     batch_count = 0
#     exact_matches = 0
#     total_samples = 0
#     # Add character-level tracking
#     matching_chars = 0
#     total_chars = 0
     
#     with torch.inference_mode(), autocast("cuda", dtype=torch.bfloat16):
#         for x, y in dataloader:
#             x = x.to(device)
#             y = y.to(device)
#             logits = model(x)
#             loss = criterion(
#                 logits.reshape(-1, len(dataloader.dataset.char2idx)),
#                 y[:, -dataloader.dataset.encode_last_n_length:].reshape(-1)
#             ).mean()
#             total_loss += loss.item()
#             batch_count += 1

#             pred_logits = logits[:, -dataloader.dataset.encode_last_n_length:]
#             pred_indices = torch.argmax(pred_logits, dim=-1)
            
#             target_seqs = y[:, -dataloader.dataset.encode_last_n_length:]
#             target_texts = batch_tensor_to_text(target_seqs, dataloader.dataset.idx2char)
#             pred_texts = batch_tensor_to_text(pred_indices, dataloader.dataset.idx2char)
            
#             # Count exact matches
#             exact_matches += sum(1 for pred, target in zip(pred_texts, target_texts) if pred == target)
#             total_samples += len(pred_texts)

#             # Add character-level accuracy calculation
#             for pred, target in zip(pred_texts, target_texts):
#                 total_chars += len(target)
#                 matching_chars += sum(p == t for p, t in zip(pred, target))

#             if batch_count % 10 == 0:    
#                 print(f"\nSample {batch_count}:")
#                 print(f"Target: {target_texts[0]}")
#                 print(f"Pred:   {pred_texts[0]}")
#                 print(f"Current sequence accuracy: {exact_matches/total_samples:.2%}")
#                 print(f"Current character accuracy: {matching_chars/total_chars:.2%}")
    
#     model.train()
#     return {
#         "val_loss": total_loss / batch_count,
#         "val_sequence_accuracy": exact_matches/total_samples,
#         "val_char_accuracy": matching_chars/total_chars
#     }




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


def save_model(
    model: HyperTokenAutoencoder,
    save_dir: str,
    model_name: str,
    save_separate: bool = True
) -> None:
    """
    Save the model state. Optionally save encoder and decoder separately.
    
    Args:
        model: The model to save
        save_dir: Directory to save the model(s) in
        model_name: Base name for the saved model files
        save_separate: If True, save encoder and decoder separately
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save full model
    full_model_path = os.path.join(save_dir, f"{model_name}_full.pt")
    torch.save(model.state_dict(), full_model_path)
    
    if save_separate:
        # Save encoder
        encoder_path = os.path.join(save_dir, f"{model_name}_encoder.pt")
        torch.save(model.encoder.state_dict(), encoder_path)
        
        # Save decoder
        decoder_path = os.path.join(save_dir, f"{model_name}_decoder.pt")
        torch.save(model.decoder.state_dict(), decoder_path)

def load_model(
    model: HyperTokenAutoencoder,
    load_dir: str,
    model_name: str,
    device: Optional[str] = None,
    encoder_only: bool = False,
    decoder_only: bool = False
) -> HyperTokenAutoencoder:
    """
    Load a saved model state.
    
    Args:
        model: The model to load state into
        load_dir: Directory containing the saved model(s)
        model_name: Base name of the saved model files
        device: Device to load the model to
        encoder_only: Only load the encoder part
        decoder_only: Only load the decoder part
    
    Returns:
        The model with loaded state
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if encoder_only and decoder_only:
        raise ValueError("Cannot specify both encoder_only and decoder_only")
    
    def clean_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        return new_state_dict
    
    if encoder_only:
        encoder_path = os.path.join(load_dir, f"{model_name}_encoder.pt")
        state_dict = torch.load(encoder_path, map_location=device)
        model.encoder.load_state_dict(clean_state_dict(state_dict))
    elif decoder_only:
        decoder_path = os.path.join(load_dir, f"{model_name}_decoder.pt")
        state_dict = torch.load(decoder_path, map_location=device)
        model.decoder.load_state_dict(clean_state_dict(state_dict))
    else:
        full_model_path = os.path.join(load_dir, f"{model_name}_full.pt")
        state_dict = torch.load(full_model_path, map_location=device)
        model.load_state_dict(clean_state_dict(state_dict))
    
    return model.to(device)

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


    enable_torch_optimizations()
    setup_flash_attention()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for experiment in experiments:
        seq_len = experiment["seq_len"]
        encode_last_n_length = experiment["encode_last_n_length"]
        hypertoken_size = experiment["hypertoken_size"]
        batch_size = experiment["batch_size"]
        compress_factor = experiment["compress_factor"]
        n_layers = experiment["n_layers"]
        head_size = experiment["head_size"]
        embed_dim = experiment["embed_dim"]
        segments = 10

        #model = load_model(model, "saved_models", "hypertoken_2025-01-10T00:21:59.914619_encode_last_n_length128_hypertoken_size512")

        dataset = TinyShakespeareDataset(encode_last_n_length,segments=segments,seq_len=seq_len)
        vocab_size = len(dataset.char2idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TinyShakespeareDataset(encode_last_n_length,segments=segments,seq_len=seq_len,type="validation")
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        verify_hyperparameters(
            experiment["hypertoken_size"],experiment["embed_dim"],
            experiment["n_layers"],experiment["head_size"],experiment["lr"],
            experiment["seq_len"],experiment["hypertoken_size"],
            experiment["compress_factor"],experiment["encode_last_n_length"]
        )

        model = HyperTokenAutoencoder(
            vocab_size=vocab_size, 
            seq_len=seq_len, 
            encode_last_n_length=encode_last_n_length, 
            hypertoken_size=hypertoken_size, 
            head_size=head_size, 
            compress_factor=compress_factor, 
            n_layers=n_layers, 
            embed_dim=embed_dim,
        ).to(device)

        run_experiment("HyperTokens",model,train_model, dataloader, val_dataloader, experiment)
    

#TODO: fix gpu memory reporting

