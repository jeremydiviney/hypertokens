from typing import Callable, TypeVar, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from datetime import datetime
import os
import torch.optim as optim
from torch.amp import autocast


Model = TypeVar('Model', bound=nn.Module)

def save_model(
    model: Any,
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
    model: Model,
    load_dir: str,
    model_name: str,
    device: Optional[str] = None
) -> Model:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def clean_state_dict(state_dict: dict) -> dict:
        return {
            key.replace('_orig_mod.', ''): value 
            for key, value in state_dict.items()
        }
    
    full_model_path = os.path.join(load_dir, f"{model_name}_full.pt")
    state_dict = torch.load(full_model_path, map_location=device)
    model.load_state_dict(clean_state_dict(state_dict))
    return model.to(device)

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
