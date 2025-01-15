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
from datasources.tinyshakespeare import TinyShakespeareDataset
from helpers.training import (
    save_model,
    enable_torch_optimizations,
    setup_flash_attention,
)
import sys
from transformers import get_linear_schedule_with_warmup
from models.jpt1 import JPT1
from datasources.tinyshakespeare import HyperTokenTinyShakespeareDataset
from helpers.training import batch_tensor_to_text
import time
import random

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
    model.eval()
    decoder.eval()
    total_loss = 0
    batch_count = 0
    exact_matches = 0
    total_samples = 0
    matching_chars = 0
    total_chars = 0

    with torch.inference_mode(), autocast(device_type="cuda", dtype=torch.bfloat16):
        for x, y in dataloader:
            encoded_seq = x.to(device)
            target_chars = y.to(device)

            jpt_output = model(encoded_seq)

            loss = calculate_loss(jpt_output, target_chars, criterion)

            total_loss += loss.item()
            batch_count += 1

            final_embedding = jpt_output[:, -1]
            final_target_char = target_chars[:, -1]

            pred_indices = torch.argmax(final_embedding, dim=-1).unsqueeze(1)
            target_texts = batch_tensor_to_text(
                final_target_char, dataloader.dataset.idx2char
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
                # for p, t in zip(pred, target):
                #     print(f"pred: {p}, target: {t}, {"!" if p == t else " "}")

            if batch_count % 2 == 0:
                print(f"\nSample {batch_count}:")
                # print(f"Target: {target_texts[0]}")
                # print(f"Pred:   {pred_texts[0]}")
                print(f"Current sequence accuracy: {exact_matches/total_samples:.2%}")
                print(f"Current character accuracy: {matching_chars/total_chars:.2%}")

    model.train()
    decoder.train()
    return {
        "val_loss": total_loss / batch_count,
        "val_sequence_accuracy": exact_matches / total_samples,
        "val_char_accuracy": matching_chars / total_chars,
    }


def calculate_loss(model_output, target_chars, criterion):
    all_embeddings = model_output
    cur_batch_size = all_embeddings.size(0)
    final_embedding = all_embeddings.view(cur_batch_size, -1, vocab_size)
    target_chars = target_chars.view(cur_batch_size, -1)

    # Split the predictions into final and non-final timesteps
    final_pred = final_embedding[:, -1:]  # Get last timestep
    other_preds = final_embedding[:, :-1]  # Get all other timesteps

    final_target = target_chars[:, -1:]
    other_targets = target_chars[:, :-1]

    # Calculate losses separately
    final_loss = criterion(
        final_pred.transpose(1, 2), final_target
    ).mean()  # Multiply final loss by weight (e.g. 10x)

    other_loss = criterion(other_preds.transpose(1, 2), other_targets).mean()

    # Combine losses
    loss = other_loss / 10 + final_loss
    return loss


def train_model(
    wandb, model, decoder_model, train_dataloader, val_dataloader, config: dict
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_segments = config["data_segments"]

    train_dataset = train_dataloader.dataset

    # Freeze decoder parameters to prevent updates
    # for param in decoder_model.parameters():
    #     param.requires_grad = True
    # decoder_model.eval()  # Set decoder to evaluation mode

    # only if not debugging
    if sys.gettrace() is None:  # No debugger attached
        model = torch.compile(model)
        decoder_model = torch.compile(decoder_model)

    count_parameters(model)
    count_parameters(decoder_model)

    # Create optimizer for both JPT1 and decoder model parameters
    optimizer = optim.Adam(
        # list(model.parameters()) + list(decoder_model.parameters()),
        model.parameters(),
        lr=config["lr"],
    )

    # optimizer = Lion(
    #     model.parameters(),
    #     lr=config["lr"],  # Usually needs 3-10x smaller learning rate than Adam
    #     weight_decay=1e-2,  # Lion typically works better with higher weight decay
    # )

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

    evaluate_model(model, decoder_model, val_dataloader, criterion, device)

    current_lr = config["lr"]
    low_loss = 10000
    vocab_size = len(train_dataset.char2idx)

    train_time_start = time.time()
    total_training_examples = 0

    for epoch in range(config["epochs"]):
        epoch_loss = 0
        batch_count = 0
        optimizer.param_groups[0]["lr"] = current_lr

        for segment in range(data_segments):
            segment_loss = 0
            segment_batch_count = 0
            fetch_start_time = time.time()
            train_step_start = fetch_start_time
            for x, y in train_dataloader:

                fetch_end_time = time.time()

                batch_count += 1
                segment_batch_count += 1
                total_training_examples += x.shape[0]

                encoded_seq = x.to(device)
                target_chars = y.to(device)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Forward through JPT1
                    jpt_output = model(encoded_seq)

                    loss = calculate_loss(jpt_output, target_chars, criterion)

                # Update metrics
                segment_loss += loss.item()
                epoch_loss += loss.item()

                new_low = False

                if loss.item() < low_loss:
                    low_loss = loss.item()
                    print(f"New low loss: {low_loss:.7f}")
                    new_low = True
                # Log batch metrics

                if batch_count % 10 == 0:
                    # Get predictions for logging
                    # with torch.inference_mode():
                    # pred_indices = torch.argmax(decoded, dim=-1)
                    # target_texts = batch_tensor_to_text(
                    #     target_chars, train_dataset.idx2char
                    # )
                    # pred_texts = batch_tensor_to_text(
                    #     pred_indices, train_dataset.idx2char
                    # )

                    # if new_low:
                    #     print(f"\nSample {batch_count}:")
                    #     print(f"Target: {target_texts[0]}")
                    #     print(f"Pred:   {pred_texts[0]}")

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

                train_step_end = time.time()

                train_step_start = train_step_end
                fetch_start_time = train_step_start

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

    train_time_end = time.time()

    total_time = train_time_end - train_time_start
    examples_per_second = total_training_examples / total_time
    time_per_million = 1_000_000 / examples_per_second

    wandb.log({"time_per_million": time_per_million})

    print(f"Training time: {train_time_end - train_time_start:.4f} seconds")

    # Save both models
    save_dir = "saved_models"
    timestamp = datetime.now().isoformat()
    model_name = f"jpt1_{timestamp}"

    save_model(model, save_dir, f"{model_name}_jpt1")
    save_model(decoder_model, save_dir, f"{model_name}_decoder")

    return model, decoder_model


def verify_model_params(config: dict):
    """
    Verify model parameters meet requirements
    """
    if config["jpt_embed_dim"] <= config["hypertoken_size"]:
        raise ValueError(
            f"embed_dim ({config['jpt_embed_dim']}) must be greater than hypertoken_size ({config['hypertoken_size']})"
        )

    if config["jpt_embed_dim"] % config["hypertoken_size"] != 0:
        raise ValueError(
            f"embed_dim ({config['jpt_embed_dim']}) must be a multiple of hypertoken_size ({config['hypertoken_size']})"
        )

    multiple = config["jpt_embed_dim"] // config["hypertoken_size"]
    if multiple <= 1:
        raise ValueError(
            f"embed_dim must be at least 2x hypertoken_size. Current ratio: {multiple}"
        )


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


def validate_hypertoken_models(
    encoder: nn.Module,
    decoder: nn.Module,
    dataset: TinyShakespeareDataset,
    hypertoken_seq_len: int,
    num_samples: int = 10,
    device: str = "cuda",
) -> None:
    """
    Validate hypertoken encoder and decoder models by comparing input/output text

    Args:
        encoder: HyperToken encoder model
        decoder: HyperToken decoder model
        dataset: TinyShakespeare dataset instance
        num_samples: Number of samples to validate
        device: Device to run validation on
    """
    encoder.eval()
    decoder.eval()

    print("\nValidating HyperToken Encoder-Decoder Pipeline...")
    print("-" * 50)

    good_count = 0

    with torch.inference_mode():
        for i in range(num_samples):
            input_chunk = dataset.get_batch_item(
                i, chunk_count=1, chunk_size=hypertoken_seq_len
            )
            input_chunk = input_chunk.view(-1).to(device)

            # Convert to text for display
            input_text = "".join([dataset.idx2char[idx.item()] for idx in input_chunk])

            # Encode
            encoded = encoder(input_chunk.unsqueeze(0))

            # Decode
            decoded = decoder(encoded)

            # Get predicted text
            pred_indices = torch.argmax(decoded, dim=-1)[0]
            output_text = "".join(
                [dataset.idx2char[idx.item()] for idx in pred_indices]
            )

            # Calculate character accuracy
            good = output_text == input_text

            if good:
                good_count += 1

            accuracy = good_count / (i + 1)

            print(f"\nSample {i+1}:")
            print(f"Input:  '{input_text}'")
            print(f"Output: '{output_text}'")
            print(f"Character Accuracy: {accuracy:.2%}")

    print("\nValidation Complete")


def generate_text(
    jpt_model: nn.Module,
    encoder_model: nn.Module,
    prompt: str,
    max_new_chars: int,
    hypertoken_seq_len: int,
    dataset: HyperTokenTinyShakespeareDataset,
    temperature: float = 1.0,
    device: str = "cuda",
) -> str:
    # Set models to eval mode
    jpt_model.eval()
    encoder_model.eval()

    print(f"\nPrompt: {prompt}\n")
    print("Generating...\n")

    # Initialize result with prompt
    result = list(prompt)

    with torch.inference_mode(), autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(max_new_chars):
            # Get the last hypertoken_seq_len * seq_len characters
            input_text = result[-(hypertoken_seq_len * jpt_model.seq_len) :]

            # Pad if we don't have enough characters
            if len(input_text) < hypertoken_seq_len * jpt_model.seq_len:
                padding = ["<PAD>"] * (
                    hypertoken_seq_len * jpt_model.seq_len - len(input_text)
                )
                input_text = padding + input_text

            # Convert to character indices and chunk into hypertoken_seq_len sized pieces
            char_indices = [
                dataset.char2idx.get(c, dataset.pad_token) for c in input_text
            ]
            char_chunks = torch.tensor(char_indices, dtype=torch.long).view(
                -1, hypertoken_seq_len
            )

            # Ensure we have the right shape
            if char_chunks.size(0) != jpt_model.seq_len:
                padding_chunks = torch.full(
                    (jpt_model.seq_len - char_chunks.size(0), hypertoken_seq_len),
                    dataset.pad_token,
                    dtype=torch.long,
                )
                char_chunks = torch.cat([padding_chunks, char_chunks])

            # Process through encoder
            char_chunks = char_chunks.to(device)
            encoded = encoder_model(char_chunks.view(-1, hypertoken_seq_len))
            encoded = encoded.view(1, jpt_model.seq_len, -1)

            # Get prediction from JPT1
            output = jpt_model(encoded)
            logits = output[0, -1] / temperature

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Sample next character
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = dataset.idx2char[next_char_idx]

            # Print the generated character
            print(f"[{i+1}/{max_new_chars}] Generated: '{next_char}'")

            result.append(next_char)

    final_text = "".join(result)
    print(f"\nFinal text:\n{final_text}")
    return final_text


if __name__ == "__main__":

    # Define experiments
    experiments: list[dict] = [
        {
            "seq_len": sl,
            "hypertoken_seq_len": 1,
            "hypertoken_size": 32,
            "epochs": epochs,
            "batch_size": 64,
            "lr": lr,
            "head_size": head_size,
            "n_layers": n_layers,
            "hypertoken_embed_dim": 128,
            "jpt_embed_dim": jed,
            "compress_factor": 2,
            "data_segments": 10,
        }
        for n_layers in [6]  # Varying n_layers
        for head_size in [64]  # Varying head_size
        for jed in [384]
        for lr in [0.0003]
        for sl in [128]
        for epochs in [3]
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
            head_size=16,
            compress_factor=hypertoeken_compress_factor,
            n_layers=1,
            embed_dim=hypertoken_embed_dim,
        ).to(device)

        h_encoder_model = HyperTokenEncoder(
            vocab_size=vocab_size,
            seq_len=hypertoken_seq_len,
            encode_last_n_length=hypertoken_seq_len,
            hypertoken_size=hypertoken_size,
            head_size=16,
            compress_factor=hypertoeken_compress_factor,
            n_layers=1,
            embed_dim=hypertoken_embed_dim,
        )

        h_encoder_model = load_model(
            h_encoder_model,
            "saved_models",
            "hypertoken_2025-01-14T02:13:00.639557_encode_last_n_length1_hypertoken_size32",
            encoder_only=True,
        )

        h_decoder_model = load_model(
            h_decoder_model,
            "saved_models",
            "hypertoken_2025-01-14T02:13:00.639557_encode_last_n_length1_hypertoken_size32",
            decoder_only=True,
        )

        dataset = HyperTokenTinyShakespeareDataset(
            h_encoder_model,
            hypertoken_seq_len=hypertoken_seq_len,
            segments=data_segments,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        vocab_size = len(dataset.char2idx)

        gptModel = JPT1(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=jpt_embed_dim,
            num_heads=head_size,
            num_layers=n_layers,
            dropout=0.1,
            hypertoken_size=hypertoken_size,
        ).to(device)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            # num_workers=4 if not is_debugging else 0,  # Parallel data loading
            # pin_memory=False,  # Faster data transfer to GPU
            # persistent_workers=(
            #     True if not is_debugging else False
            # ),  # Keep workers alive between epochs
            # prefetch_factor=(
            #     4 if not is_debugging else None
            # ),  # Number of batches loaded in advance per worker)
        )

        val_dataset = HyperTokenTinyShakespeareDataset(
            h_encoder_model,
            hypertoken_seq_len=hypertoken_seq_len,
            segments=data_segments,
            seq_len=seq_len,
            batch_size=batch_size,
            type="validation",
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            # num_workers=4 if not is_debugging else 0,  # Parallel data loading
            # pin_memory=False,
            # persistent_workers=(
            #     True if not is_debugging else False
            # ),  # Keep workers alive between epochs
            # prefetch_factor=(
            #     4 if not is_debugging else None
            # ),  # Number of batches loaded in advance per worker)
        )

        verify_model_params(experiment)

        validate_hypertoken_models(
            h_encoder_model, h_decoder_model, dataset, hypertoken_seq_len
        )

        # create wrapper function for train_model
        def train_model_lambda(wandb):
            model = train_model(
                wandb, gptModel, h_decoder_model, dataloader, val_dataloader, experiment
            )
            return model[0]

        project_name = "jpt1"
        exp_name = f"{project_name}-sl:{experiment['seq_len']}-htsl:{experiment['hypertoken_seq_len']}-hts:{experiment['hypertoken_size']}-e:{experiment['epochs']}-bs:{experiment['batch_size']}-lr:{experiment['lr']}-hs:{experiment['head_size']}-nl:{experiment['n_layers']}-ed:{experiment['jpt_embed_dim']}"

        run_experiment(project_name, train_model_lambda, exp_name, experiment)

        # generate_text(
        #     gptModel, h_decoder_model, "Hello, world!", 100, hypertoken_seq_len, dataset
        # )


print("Training Complete")

# TODO: fix gpu memory reporting
