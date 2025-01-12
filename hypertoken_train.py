import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
from helpers.experiments import run_experiment, count_parameters
from lion_pytorch import Lion
from torch.utils.checkpoint import checkpoint
import os
from typing import Optional
from models.hypertoken_auto_encoder import (
    HyperTokenEncoder,
    HyperTokenDecoder,
    HyperTokenAutoencoder,
)
from datetime import datetime
from datasources.tinyshakespeare import TinyShakespeareDataset
from helpers.training import (
    save_model,
    enable_torch_optimizations,
    setup_flash_attention,
)
import sys
from transformers import get_linear_schedule_with_warmup
from helpers.training import batch_tensor_to_text


# test test tes

# --------------------------------------------------
# 2. Model Definition
# --------------------------------------------------


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str
) -> dict:
    """Evaluate model on given dataloader"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    total_loss = 0
    batch_count = 0
    exact_matches = 0
    total_samples = 0
    # Add character-level tracking
    matching_chars = 0
    total_chars = 0

    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device)  # x is in int (char index)
            y = y.to(device)  # y is in int (char index)
            logits = model(x)
            loss = criterion(
                logits.reshape(-1, len(dataloader.dataset.char2idx)),
                y[:, -dataloader.dataset.encode_last_n_length :].reshape(-1),
            ).mean()
            total_loss += loss.item()
            batch_count += 1

            pred_logits = logits[:, -dataloader.dataset.encode_last_n_length :]
            pred_indices = torch.argmax(pred_logits, dim=-1)

            target_seqs = y[:, -dataloader.dataset.encode_last_n_length :]
            target_texts = batch_tensor_to_text(
                target_seqs, dataloader.dataset.idx2char
            )
            pred_texts = batch_tensor_to_text(pred_indices, dataloader.dataset.idx2char)

            # Count exact matches
            exact_matches += sum(
                1 for pred, target in zip(pred_texts, target_texts) if pred == target
            )
            total_samples += len(pred_texts)

            # Add character-level accuracy calculation
            for pred, target in zip(pred_texts, target_texts):
                total_chars += len(target)
                matching_chars += sum(p == t for p, t in zip(pred, target))

            if batch_count % 10 == 0:
                print(f"\nSample {batch_count}:")
                print(f"Target: {target_texts[0]}")
                print(f"Pred: {pred_texts[0]}")
                print(f"Current sequence accuracy: {exact_matches/total_samples:.2%}")
                print(f"Current character accuracy: {matching_chars/total_chars:.2%}")

    model.train()
    return {
        "val_loss": total_loss / batch_count,
        "val_sequence_accuracy": exact_matches / total_samples,
        "val_char_accuracy": matching_chars / total_chars,
    }


def train_model(wandb, model, dataloader, val_dataloader, config: dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_segments = dataloader.dataset.segments

    lr = config["lr"]
    epochs = config["epochs"]
    encode_last_n_length = config["encode_last_n_length"]

    # only if not debugging
    if sys.gettrace() is None:  # No debugger attached
        model = torch.compile(model)

    count_parameters(model)

    # optimizer = optim.Adam(model.parameters(), lr=lr)

    optimizer = Lion(
        model.parameters(),
        lr=lr,  # Usually needs 3-10x smaller learning rate than Adam
        weight_decay=1e-2,  # Lion typically works better with higher weight decay
    )

    criterion = nn.CrossEntropyLoss()

    total_steps = epochs * len(dataloader) * data_segments

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy="cos",
        cycle_momentum=False,
    )

    current_lr = lr

    low_loss = 10000

    vocab_size = len(dataloader.dataset.char2idx)

    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        optimizer.param_groups[0]["lr"] = current_lr

        for segment in range(data_segments):
            segment_loss = 0
            segment_batch_count = 0

            for x, y in dataloader:

                batch_count += 1
                segment_batch_count += 1
                x = x.to(device)  # x is in int (char index)
                y = y.to(device)  # y is in int (char index)

                logits = model(x)
                loss_per_pos = criterion(
                    logits.reshape(-1, vocab_size),
                    y[:, -encode_last_n_length:].reshape(-1),
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
                    wandb.log(
                        {
                            "batch_loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                        }
                    )

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

        wandb.log(
            {
                "epoch_loss": epoch_loss / batch_count,
                "val_loss": val_loss,
                "val_sequence_accuracy": val_sequence_accuracy,
                "val_char_accuracy": val_char_accuracy,
                "epoch": epoch,
            }
        )

        print(
            f"Epoch {epoch} train_loss: {avg_segment_loss:.4f}, val_loss: {val_loss:.4f}, val_sequence_accuracy: {val_sequence_accuracy:.2%}, val_char_accuracy: {val_char_accuracy:.2%}"
        )

    # After training loop ends, save the model
    save_dir = "saved_models"
    timestamp = datetime.now().isoformat()
    model_name = f"hypertoken_{timestamp}_encode_last_n_length{encode_last_n_length}_hypertoken_size{config['hypertoken_size']}"
    save_model(model, save_dir, model_name)

    return model


def verify_model_params(
    hs,
    ed,
    n_layers,
    head_size,
    lr,
    seq_len,
    hypertoken_size,
    compress_factor,
    encode_last_n_length,
):

    print(
        f"Verifying hyperparameters \n\
            hypertoken_size: {hypertoken_size}, \n\
            seq_len: {seq_len}, \n\
            encode_last_n_length: {encode_last_n_length}"
    )

    if hypertoken_size < seq_len:
        raise ValueError("hypertoken_size must be greater than or equal to seq_len")

    if hypertoken_size < encode_last_n_length:
        raise ValueError(
            "encode_last_n_length must be greater than or equal to hypertoken_size"
        )

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


def load_model(
    model: HyperTokenAutoencoder,
    load_dir: str,
    model_name: str,
    device: Optional[str] = None,
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
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def clean_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        return new_state_dict

    full_model_path = os.path.join(load_dir, f"{model_name}.pt")
    state_dict = torch.load(full_model_path, map_location=device)
    model.load_state_dict(clean_state_dict(state_dict))

    return model.to(device)


if __name__ == "__main__":

    # Define experiments
    experiments: list[dict] = [
        {
            "seq_len": 128,
            "encode_last_n_length": 128,
            "hypertoken_size": hs,
            "epochs": 3,
            "batch_size": 512,
            "lr": lr,
            "head_size": head_size,
            "n_layers": n_layers,
            "embed_dim": ed,
            "compress_factor": cf,
        }
        for hs in [512]  # Varying hypertoken_size
        for ed in [512]  # Varying embed_dim
        for n_layers in [1]  # Varying n_layers
        for head_size in [16]  # Varying head_size
        for lr in [0.0001]
        for cf in [4]
    ]

    enable_torch_optimizations()
    setup_flash_attention()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

        # model = load_model(model, "saved_models", "hypertoken_2025-01-10T00:21:59.914619_encode_last_n_length128_hypertoken_size512")

        dataset = TinyShakespeareDataset(
            encode_last_n_length, segments=segments, seq_len=seq_len
        )
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

        val_dataset = TinyShakespeareDataset(
            encode_last_n_length, segments=segments, seq_len=seq_len, type="validation"
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,  # Number of batches loaded in advance per worker)
        )

        verify_model_params(
            experiment["hypertoken_size"],
            experiment["embed_dim"],
            experiment["n_layers"],
            experiment["head_size"],
            experiment["lr"],
            experiment["seq_len"],
            experiment["hypertoken_size"],
            experiment["compress_factor"],
            experiment["encode_last_n_length"],
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

        # create wrapper function for train_model
        def train_model_lambda(wandb):
            return train_model(wandb, model, dataloader, val_dataloader, experiment)

        project_name = "HyperTokens"
        exp_name = f"{project_name}-sl:{experiment['seq_len']}-elnl:{experiment['encode_last_n_length']}-hts:{experiment['hypertoken_size']}-e:{experiment['epochs']}-bs:{experiment['batch_size']}-lr:{experiment['lr']}-hs:{experiment['head_size']}-nl:{experiment['n_layers']}-ed:{experiment['embed_dim']}-cf:{experiment['compress_factor']}"

        run_experiment(project_name, train_model_lambda, exp_name, experiment)

    print("Training complete")
    sys.exit()

# TODO: fix gpu memory reporting
