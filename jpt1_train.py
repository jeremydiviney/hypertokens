import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
from helpers.experiments import run_experiment, count_parameters
from torch.amp import autocast, GradScaler
from lion_pytorch import Lion
from torch.utils.checkpoint import checkpoint
import os
from typing import Optional
from models.hypertoken_auto_encoder import HyperTokenEncoder, HyperTokenDecoder, HyperTokenAutoencoder
from datetime import datetime
from data.tinyshakespeare import TinyShakespeareDataset
from helpers.training import save_model, enable_torch_optimizations, setup_flash_attention
import sys
from transformers import get_linear_schedule_with_warmup
from models.jpt1 import JPT1
from data.tinyshakespeare import HyperTokenTinyShakespeareDataset
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



def evaluate_model(model: nn.Module, decoder: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str) -> dict:
    """Evaluate model on given dataloader"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    decoder.eval()
    total_loss = 0
    batch_count = 0
    exact_matches = 0
    total_samples = 0
    matching_chars = 0
    total_chars = 0
     
    with torch.inference_mode(), autocast("cuda", dtype=torch.bfloat16):
        for batch in dataloader:
            encoded_seq = batch["encoded"].to(device)
            target_chars = batch["target_chars"].to(device)
            
            # Forward through JPT1
            jpt_output = model(encoded_seq)
            
            # Decode each embedding in the sequence
            decoded_outputs = []
            for i in range(jpt_output.size(1)):
                decoded = decoder(jpt_output[:, i])
                decoded_outputs.append(decoded)
            
            # Stack decoded outputs
            decoded = torch.stack(decoded_outputs, dim=1)
            
            loss = criterion(
                decoded.reshape(-1, len(dataloader.dataset.char2idx)),
                target_chars.reshape(-1)
            ).mean()
            
            total_loss += loss.item()
            batch_count += 1

            pred_indices = torch.argmax(decoded, dim=-1)
            target_texts = batch_tensor_to_text(target_chars, dataloader.dataset.idx2char)
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
    decoder.train()
    return {
        "val_loss": total_loss / batch_count,
        "val_sequence_accuracy": exact_matches/total_samples,
        "val_char_accuracy": matching_chars/total_chars
    }

def train_model(wandb, model, dataloader, val_dataloader, config: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_segments = config["segments"]
   
    # Load encoder and decoder
    encoder = HyperTokenEncoder(
        vocab_size=config["vocab_size"],
        seq_len=config["h_seq_len"],
        embed_dim=config["embed_dim"],
        head_size=config["head_size"],
        n_layers=config["n_layers"]
    )
    
    decoder = HyperTokenDecoder(
        vocab_size=config["vocab_size"],
        seq_len=config["encode_last_n_length"],
        embed_dim=config["embed_dim"],
        head_size=config["head_size"],
        n_layers=config["n_layers"]
    ).to(device)
    
    # Load weights
    encoder = load_model(encoder, "saved_models", config["encoder_model_name"], encoder_only=True)
    encoder.eval()  # Set to eval mode since we're not training it
    
    decoder = load_model(decoder, "saved_models", config["decoder_model_name"], decoder_only=True)
    decoder.train()  # Set to train mode since we're training it end-to-end
    
    # Create datasets with encoder
    train_dataset = HyperTokenTinyShakespeareDataset(
        encoder=encoder,
        encode_last_n_length=config["encode_last_n_length"],
        segments=config["segments"],
        h_seq_len=config["h_seq_len"],
        jpt_seq_len=config["jpt_seq_len"],
        type="train"
    )
    
    val_dataset = HyperTokenTinyShakespeareDataset(
        encoder=encoder,
        encode_last_n_length=config["encode_last_n_length"],
        segments=config["segments"],
        h_seq_len=config["h_seq_len"],
        jpt_seq_len=config["jpt_seq_len"],
        type="validation"
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    #only if not debugging
    if sys.gettrace() is None:  # No debugger attached
        model = torch.compile(model)
        decoder = torch.compile(decoder)

    count_parameters(model)
    count_parameters(decoder)

    # Create optimizer for both models
    optimizer = optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    total_steps = config["epochs"] * len(train_dataloader) * data_segments

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy='cos',
        cycle_momentum=False
    )

    current_lr = config["lr"]
    low_loss = 10000
    vocab_size = len(train_dataset.char2idx)

    for epoch in range(config["epochs"]):
        epoch_loss = 0
        batch_count = 0
        optimizer.param_groups[0]['lr'] = current_lr

        for segment in range(data_segments):
            segment_loss = 0
            segment_batch_count = 0
            
            for batch in train_dataloader:
                batch_count += 1
                segment_batch_count += 1
                
                encoded_seq = batch["encoded"].to(device)
                target_chars = batch["target_chars"].to(device)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Forward through JPT1
                    jpt_output = model(encoded_seq)
                    
                    # Decode each embedding in the sequence
                    decoded_outputs = []
                    for i in range(jpt_output.size(1)):
                        decoded = decoder(jpt_output[:, i])
                        decoded_outputs.append(decoded)
                    
                    # Stack decoded outputs
                    decoded = torch.stack(decoded_outputs, dim=1)
                    
                    loss = criterion(
                        decoded.reshape(-1, vocab_size),
                        target_chars.reshape(-1)
                    ).mean()

                # Update metrics
                segment_loss += loss.item()
                epoch_loss += loss.item()

                if loss.item() < low_loss:
                    low_loss = loss.item()
                    print(f"New low loss: {low_loss:.7f}")

                # Log batch metrics
                if batch_count % 10 == 0:
                    # Get predictions for logging
                    with torch.inference_mode():
                        pred_indices = torch.argmax(decoded, dim=-1)
                        target_texts = batch_tensor_to_text(target_chars, train_dataset.idx2char)
                        pred_texts = batch_tensor_to_text(pred_indices, train_dataset.idx2char)
                        
                        print(f"\nSample {batch_count}:")
                        print(f"Target: {target_texts[0]}")
                        print(f"Pred:   {pred_texts[0]}")
                    
                    wandb.log({
                        "batch_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "epoch": epoch,
                    })

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Log segment metrics
            avg_segment_loss = segment_loss / segment_batch_count

        # Evaluation
 
        eval_results = evaluate_model(model, decoder, val_dataloader, criterion, device)

        
        val_loss = eval_results["val_loss"]
        val_sequence_accuracy = eval_results["val_sequence_accuracy"]
        val_char_accuracy = eval_results["val_char_accuracy"]

        wandb.log({
            "epoch_loss": epoch_loss/batch_count,
            "val_loss": val_loss,
            "val_sequence_accuracy": val_sequence_accuracy,
            "val_char_accuracy": val_char_accuracy,
            "epoch": epoch,
        })

        print(f"Epoch {epoch} train_loss: {avg_segment_loss:.4f}, val_loss: {val_loss:.4f}, "
              f"val_sequence_accuracy: {val_sequence_accuracy:.2%}, val_char_accuracy: {val_char_accuracy:.2%}")

    # Save both models
    save_dir = "saved_models"
    timestamp = datetime.now().isoformat()
    model_name = f"jpt1_{timestamp}_encode_last_n_length{config['encode_last_n_length']}_h_seq_len{config['h_seq_len']}"
    
    save_model(model, save_dir, f"{model_name}_jpt1")
    save_model(decoder, save_dir, f"{model_name}_decoder")

    return model, decoder

def verify_model_params(hs,ed,n_layers,head_size,lr,seq_len,hypertoken_size,compress_factor,encode_last_n_length):
    return
    # print(f"Verifying hyperparameters \n\
    #         hypertoken_size: {hypertoken_size}, \n\
    #         seq_len: {seq_len}, \n\
    #         encode_last_n_length: {encode_last_n_length}")

    # if hypertoken_size < seq_len:
    #     raise ValueError("hypertoken_size must be greater than or equal to seq_len")

    # if hypertoken_size < encode_last_n_length:
    #     raise ValueError("encode_last_n_length must be greater than or equal to hypertoken_size")

    # # Add check for embed_dim being multiple of seq_len
    # if hypertoken_size % seq_len != 0:
    #     raise ValueError(f"hypertoken_size must be a multiple of seq_len")

    # if hypertoken_size % encode_last_n_length != 0:
    #     raise ValueError(f"hypertoken_size must be a multiple of encode_last_n_length")

    # # Check if embed_dim is a power of compress_factor
    # # if not (math.log(ed, compress_factor).is_integer()):
    # #     raise ValueError(f"Embed_dim ({ed}) must be a power of compress_factor ({compress_factor})")

    # if encode_last_n_length > seq_len:
    #     raise ValueError("encode_last_n_length must be less than or equal to seq_len")


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
    experiments: list[dict] = [
        {
            "seq_len": 6,
            "hypertoken_seq_len": 128,
            "hypertoken_size": 512,
            "epochs": 1,
            "batch_size": 512,
            "lr": lr,
            "head_size": head_size,
            "n_layers": n_layers,
            "embed_dim": ed,
        }
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
        hypertoken_seq_len = experiment["hypertoken_seq_len"]
        hypertoken_size = experiment["hypertoken_size"]
        batch_size = experiment["batch_size"]
        n_layers = experiment["n_layers"]
        head_size = experiment["head_size"]
        embed_dim = experiment["embed_dim"]
        segments = 10

        #model = load_model(model, "saved_models", "hypertoken_2025-01-10T00:21:59.914619_encode_last_n_length128_hypertoken_size512")

        model = load_model(model, "saved_models", "hypertoken_2025-01-10T00:21:59.914619_encode_last_n_length128_hypertoken_size512")

        dataset = HyperTokenTinyShakespeareDataset(hypertoken_seq_len=hypertoken_seq_len,segments=segments,seq_len=seq_len)
        vocab_size = len(dataset.char2idx)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,  # Number of batches loaded in advance per worker)
        ) 

        val_dataset = HyperTokenTinyShakespeareDataset(hypertoken_seq_len,segments=segments,seq_len=seq_len,type="validation")
        val_dataloader = DataLoader(val_dataset,  
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,  # Number of batches loaded in advance per worker)
        )

        verify_model_params(
            experiment["hypertoken_size"],experiment["embed_dim"],
            experiment["n_layers"],experiment["head_size"],experiment["lr"],
            experiment["seq_len"],experiment["hypertoken_size"],
            experiment["compress_factor"],experiment["encode_last_n_length"]
        )

        gptModel = JPT1(
            vocab_size=vocab_size, 
            seq_len=seq_len, 
            embed_dim=embed_dim,
            num_heads=head_size,
            num_layers=n_layers,
            ff_dim=embed_dim*4,
            dropout=0.1
        ).to(device)

        h_decoder_model = HyperTokenDecoder(
            vocab_size=vocab_size, 
            seq_len=seq_len, 
            encode_last_n_length=hypertoken_seq_len, 
            hypertoken_size=hypertoken_size, 
            head_size=head_size, 
            compress_factor=1, 
            n_layers=n_layers, 
            embed_dim=embed_dim,
        ).to(device)

        h_encoder_model = HyperTokenEncoder(
            vocab_size=vocab_size, 
            seq_len=seq_len, 
            encode_last_n_length=hypertoken_seq_len, 
            hypertoken_size=hypertoken_size, 
            head_size=head_size, 
            compress_factor=1, 
            n_layers=n_layers, 
            embed_dim=embed_dim,
        ).to(device)

       

        run_experiment("HyperTokens",model,train_model,dataloader, val_dataloader, experiment)
    

#TODO: fix gpu memory reporting

