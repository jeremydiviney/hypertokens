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
from models.hypertoken_auto_encoder import (
    HyperTokenEncoder,
    HyperTokenDecoder,
    HyperTokenAutoencoder,
)
from datetime import datetime
from data.tinyshakespeare import TinyShakespeareDataset
from helpers.training import (
    save_model,
    enable_torch_optimizations,
    setup_flash_attention,
)
import sys
from transformers import get_linear_schedule_with_warmup
from models.jpt1 import JPT1
from data.tinyshakespeare import HyperTokenTinyShakespeareDataset
from helpers.training import batch_tensor_to_text

# --------------------------------------------------
# 2. Model Definition
# --------------------------------------------------


def evaluate_model(
    model: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict:
    """Evaluate model on given dataloader"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
                target_chars.reshape(-1),
            ).mean()

            total_loss += loss.item()
            batch_count += 1

            pred_indices = torch.argmax(decoded, dim=-1)
            target_texts = batch_tensor_to_text(
                target_chars, dataloader.dataset.idx2char
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
                print(f"Pred:   {pred_texts[0]}")
                print(f"Current sequence accuracy: {exact_matches/total_samples:.2%}")
                print(f"Current character accuracy: {matching_chars/total_chars:.2%}")

    model.train()
    decoder.train()
    return {
        "val_loss": total_loss / batch_count,
        "val_sequence_accuracy": exact_matches / total_samples,
        "val_char_accuracy": matching_chars / total_chars,
    }


def train_model(
    wandb, model, decoder_model, train_dataloader, val_dataloader, config: dict
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_segments = config["data_segments"]

    train_dataset = train_dataloader.dataset

    # only if not debugging
    if sys.gettrace() is None:  # No debugger attached
        model = torch.compile(model)
        decoder_model = torch.compile(decoder_model)

    count_parameters(model)
    count_parameters(decoder_model)

    # Create optimizer for both models
    optimizer = optim.AdamW(
        list(model.parameters()) + list(decoder_model.parameters()), lr=config["lr"]
    )
    criterion = nn.CrossEntropyLoss()

    total_steps = config["epochs"] * len(train_dataloader) * data_segments

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy="cos",
        cycle_momentum=False,
    )

    current_lr = config["lr"]
    low_loss = 10000
    vocab_size = len(train_dataset.char2idx)

    for epoch in range(config["epochs"]):
        epoch_loss = 0
        batch_count = 0
        optimizer.param_groups[0]["lr"] = current_lr

        for segment in range(data_segments):
            segment_loss = 0
            segment_batch_count = 0

            for x, y in train_dataloader:
                batch_count += 1
                segment_batch_count += 1

                encoded_seq = x.to(device)
                target_chars = y.to(device)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Forward through JPT1
                    jpt_output = model(encoded_seq)

                    # Decode each embedding in the sequence
                    decoded_outputs = []
                    for i in range(jpt_output.size(1)):
                        decoded = decoder_model(jpt_output[:, i])
                        decoded_outputs.append(decoded)

                    # Stack decoded outputs
                    decoded = torch.stack(decoded_outputs, dim=1)

                    loss = criterion(
                        decoded.reshape(-1, vocab_size), target_chars.reshape(-1)
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
                        target_texts = batch_tensor_to_text(
                            target_chars, train_dataset.idx2char
                        )
                        pred_texts = batch_tensor_to_text(
                            pred_indices, train_dataset.idx2char
                        )

                        print(f"\nSample {batch_count}:")
                        print(f"Target: {target_texts[0]}")
                        print(f"Pred:   {pred_texts[0]}")

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

        # Evaluation

        eval_results = evaluate_model(
            model, decoder_model, val_dataloader, criterion, device
        )

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
            f"Epoch {epoch} train_loss: {avg_segment_loss:.4f}, val_loss: {val_loss:.4f}, "
            f"val_sequence_accuracy: {val_sequence_accuracy:.2%}, val_char_accuracy: {val_char_accuracy:.2%}"
        )

    # Save both models
    save_dir = "saved_models"
    timestamp = datetime.now().isoformat()
    model_name = f"jpt1_{timestamp}_encode_last_n_length{config['encode_last_n_length']}_h_seq_len{config['h_seq_len']}"

    save_model(model, save_dir, f"{model_name}_jpt1")
    save_model(decoder_model, save_dir, f"{model_name}_decoder")

    return model, decoder_model


def verify_model_params():
    return


def load_model(
    model: any,
    load_dir: str,
    model_name: str,
    device: Optional[str] = None,
    encoder_only: bool = False,
    decoder_only: bool = False,
) -> nn.Module:
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

    if encoder_only and decoder_only:
        raise ValueError("Cannot specify both encoder_only and decoder_only")

    def clean_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        return new_state_dict

    if encoder_only:
        encoder_path = os.path.join(load_dir, f"{model_name}_encoder.pt")
        state_dict = torch.load(encoder_path, map_location=device)
        model.load_state_dict(clean_state_dict(state_dict))
    elif decoder_only:
        decoder_path = os.path.join(load_dir, f"{model_name}_decoder.pt")
        state_dict = torch.load(decoder_path, map_location=device)
        model.load_state_dict(clean_state_dict(state_dict))
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
            "hypertoken_embed_dim": 512,
            "epochs": 1,
            "batch_size": 512,
            "lr": lr,
            "head_size": head_size,
            "n_layers": n_layers,
            "hypertoken_embed_dim": 512,
            "jpt_embed_dim": 512,
            "compress_factor": 4,
            "data_segments": 10,
        }
        for n_layers in [1]  # Varying n_layers
        for head_size in [64]  # Varying head_size
        for lr in [0.001]
    ]

    enable_torch_optimizations()
    setup_flash_attention()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_size = 0

    is_debugging = sys.gettrace() is not None

    for experiment in experiments:
        seq_len = experiment["seq_len"]
        hypertoken_seq_len = experiment["hypertoken_seq_len"]
        hypertoken_size = experiment["hypertoken_size"]
        hypertoken_embed_dim = experiment["hypertoken_embed_dim"]
        batch_size = experiment["batch_size"]
        n_layers = experiment["n_layers"]
        head_size = experiment["head_size"]
        jpt_embed_dim = experiment["jpt_embed_dim"]
        hypertoeken_compress_factor = experiment["compress_factor"]
        data_segments = experiment["data_segments"]

        # load this just to get the vocab size
        if vocab_size == 0:
            tmp_dset = TinyShakespeareDataset(
                seq_len=experiment["seq_len"],
                encode_last_n_length=experiment["hypertoken_seq_len"],
                segments=data_segments,
            )
            vocab_size = len(tmp_dset.char2idx)
            del tmp_dset

        # delete the tmp dataset

        h_decoder_model = HyperTokenDecoder(
            vocab_size=vocab_size,
            encode_last_n_length=hypertoken_seq_len,
            hypertoken_size=hypertoken_size,
            head_size=head_size,
            compress_factor=hypertoeken_compress_factor,
            n_layers=n_layers,
            embed_dim=hypertoken_embed_dim,
        ).to(device)

        h_encoder_model = HyperTokenEncoder(
            vocab_size=vocab_size,
            seq_len=hypertoken_seq_len,
            encode_last_n_length=hypertoken_seq_len,
            hypertoken_size=hypertoken_size,
            head_size=head_size,
            compress_factor=hypertoeken_compress_factor,
            n_layers=n_layers,
            embed_dim=hypertoken_embed_dim,
        ).to(device)

        h_encoder_model = load_model(
            h_encoder_model,
            "saved_models",
            "hypertoken_2025-01-10T17:44:02.397086_encode_last_n_length128_hypertoken_size512",
            encoder_only=True,
        )
        h_decoder_model = load_model(
            h_decoder_model,
            "saved_models",
            "hypertoken_2025-01-10T17:44:02.397086_encode_last_n_length128_hypertoken_size512",
            decoder_only=True,
        )

        dataset = HyperTokenTinyShakespeareDataset(
            h_encoder_model,
            hypertoken_seq_len=hypertoken_seq_len,
            segments=data_segments,
            seq_len=seq_len,
        )
        vocab_size = len(dataset.char2idx)

        gptModel = JPT1(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=jpt_embed_dim,
            num_heads=head_size,
            num_layers=n_layers,
            ff_dim=jpt_embed_dim * 4,
            dropout=0.1,
        ).to(device)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 if not is_debugging else 0,  # Parallel data loading
            pin_memory=(
                True if not is_debugging else False
            ),  # Faster data transfer to GPU
            persistent_workers=(
                True if not is_debugging else False
            ),  # Keep workers alive between epochs
            prefetch_factor=(
                4 if not is_debugging else None
            ),  # Number of batches loaded in advance per worker)
        )

        val_dataset = HyperTokenTinyShakespeareDataset(
            h_encoder_model,
            hypertoken_seq_len=hypertoken_seq_len,
            segments=data_segments,
            seq_len=seq_len,
            type="validation",
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if not is_debugging else 0,  # Parallel data loading
            pin_memory=(
                True if not is_debugging else False
            ),  # Faster data transfer to GPU
            persistent_workers=(
                True if not is_debugging else False
            ),  # Keep workers alive between epochs
            prefetch_factor=(
                4 if not is_debugging else None
            ),  # Number of batches loaded in advance per worker)
        )

        verify_model_params()

        # create wrapper function for train_model
        def train_model_lambda(wandb):
            model = train_model(
                wandb, gptModel, h_decoder_model, dataloader, val_dataloader, experiment
            )
            return model[0]

        project_name = "jpt1"
        exp_name = f"{project_name}-sl:{experiment['seq_len']}-htsl:{experiment['hypertoken_seq_len']}-hts:{experiment['hypertoken_size']}-e:{experiment['epochs']}-bs:{experiment['batch_size']}-lr:{experiment['lr']}-hs:{experiment['head_size']}-nl:{experiment['n_layers']}-ed:{experiment['jpt_embed_dim']}"

        run_experiment(project_name, train_model_lambda, exp_name, experiment)


# TODO: fix gpu memory reporting
