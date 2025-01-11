from helpers.training import train_model, evaluate_model
from models.jpt1 import JPT1
import torch.optim as optim
import torch.nn as nn

def process_batch_jpt1(model: JPT1, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(x)
    return logits.view(-1, logits.size(-1)), y.view(-1)

def main():
    # Setup model and training parameters
    model = JPT1(
        vocab_size=256,  # adjust based on your vocabulary
        seq_len=128,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        ff_dim=2048
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Setup your dataloaders here
    train_dataloader = ...
    val_dataloader = ...
    
    # Create scheduler if needed
    scheduler = optim.lr_scheduler.OneCycleLR(...)
    
    # Train the model
    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        process_batch=process_batch_jpt1,
        wandb=wandb,
        epochs=10,
        save_dir="saved_models",
        model_name="jpt1"
    )

if __name__ == "__main__":
    main() 