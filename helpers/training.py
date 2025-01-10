from typing import Callable, TypeVar, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from datetime import datetime
import os
from helpers.experiments import count_parameters
import torch.optim as optim
from helpers.experiments import ExperimentConfig
from torch.amp import autocast
import sys

Model = TypeVar('Model', bound=nn.Module)

def save_model(
    model: Model,
    save_dir: str,
    model_name: str,
    save_separate: bool = False
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    full_model_path = os.path.join(save_dir, f"{model_name}_full.pt")
    torch.save(model.state_dict(), full_model_path)

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

def evaluate_model(
    model: Model, 
    dataloader: DataLoader, 
    criterion: nn.Module,
    process_batch: Callable[[Model, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    post_evaluation: Callable[[dict], dict] = lambda x: x
) -> dict:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    total_loss = 0
    batch_count = 0
    exact_matches = 0
    total_samples = 0
    matching_chars = 0
    total_chars = 0
     
    with torch.inference_mode(), autocast(device_type='cuda', dtype=torch.bfloat16):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, targets = process_batch(model, x, y)
            loss = criterion(logits, targets).mean()
            total_loss += loss.item()
            batch_count += 1

            # Get predictions
            pred_indices = torch.argmax(logits, dim=-1)
            
            # Calculate accuracy metrics
            exact_matches += (pred_indices == targets).all(dim=-1).sum().item()
            total_samples += targets.size(0)
            
            # Character-level accuracy
            matching_chars += (pred_indices == targets).sum().item()
            total_chars += targets.numel()

            if batch_count % 10 == 0:
                print(f"\nBatch {batch_count} metrics:")
                print(f"Current sequence accuracy: {exact_matches/total_samples:.2%}")
                print(f"Current character accuracy: {matching_chars/total_chars:.2%}")

    metrics = {
        "val_loss": total_loss / batch_count,
        "val_sequence_accuracy": exact_matches/total_samples,
        "val_char_accuracy": matching_chars/total_chars
    }
    return post_evaluation(metrics)

def train_model(wandb, model, dataloader, val_dataloader, config: ExperimentConfig):
   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    data_segments = dataloader.dataset.segments
        
    lr = config["lr"]
    epochs = config["epochs"]
    encode_last_n_length = config["encode_last_n_length"]


    #only if not debugging
    if sys.gettrace() is None:  # No debugger attached
        model = torch.compile(model)

    count_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # optimizer = Lion(
    #     model.parameters(),
    #     lr=lr,  # Usually needs 3-10x smaller learning rate than Adam
    #     weight_decay=1e-2  # Lion typically works better with higher weight decay
    # )

    
    criterion = nn.CrossEntropyLoss()

    total_steps = epochs * len(dataloader) * data_segments

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.2,  # Use 30% of steps for warmup
        anneal_strategy='cos',
        cycle_momentum=False
    )

    current_lr = lr

    low_loss = 10000

    vocab_size = len(dataloader.dataset.char2idx)

    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        optimizer.param_groups[0]['lr'] = current_lr

        for segment in range(data_segments):
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
                    "val_char_accuracy": val_char_accuracy,
                    "epoch": epoch,
                })

        print(f"Epoch {epoch} train_loss: {avg_segment_loss:.4f}, val_loss: {val_loss:.4f}, val_sequence_accuracy: {val_sequence_accuracy:.2%}, val_char_accuracy: {val_char_accuracy:.2%}")

    # After training loop ends, save the model
    save_dir = "saved_models"
    timestamp = datetime.now().isoformat()
    model_name = f"hypertoken_{timestamp}_encode_last_n_length{encode_last_n_length}_hypertoken_size{config['hypertoken_size']}"
    save_model(model, save_dir, model_name)

    return model
