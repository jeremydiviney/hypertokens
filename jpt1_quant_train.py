import os
from typing import Optional
from datetime import datetime
import time
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from tokenizers import Tokenizer

from sklearn.neighbors import KDTree

from models.jpt1_quantizer import JPT1Quantized, TokenCodebook, compute_logits

from datasources.booksum import BooksumDataset, get_or_train_tokenizer
from helpers.experiments import run_experiment, count_parameters
from helpers.training import (
    save_model,
    enable_torch_optimizations,
    setup_flash_attention,
)

from models.jpt1_quantizer import JPT1QuantModelType

from helpers.utilities import calculate_token_accuracy

# --------------------------------------------------
# 2. Model Definition
# --------------------------------------------------


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict:

    model.eval()

    dataset = dataloader.dataset

    total_loss = 0
    batch_count = 0

    char_matches_total = 0
    token_matches_total = 0
    char_total = 0
    token_total = 0

    for x, y in dataloader:

        x = x.to(device)
        y = y.to(device)

        current_batch_size = x.shape[0]

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            jpt_output, loss = inference_and_loss_step(dataset, model, x, y)

            total_loss += loss.item()
            batch_count += 1

            if model.modelType == JPT1QuantModelType.COS_SIM:
                pred_token_indices = dataset.codebook.get_nearest_token_indices(jpt_output)
            else:
                pred_token_indices = jpt_output.argmax(dim=-1)

            pred_token_indices = pred_token_indices.detach().cpu().numpy()

        pred_tokens = dataset.codebook.get_text_token_from_indices(pred_token_indices)

        target_tokens = dataset.codebook.get_text_token_from_indices(y.detach().cpu().numpy())

        accuracy_metrics = calculate_token_accuracy(pred_tokens, target_tokens, dataset.codebook.token_list["[PAD]"])

        token_matches_total += accuracy_metrics["token_matches"]
        token_total += accuracy_metrics["token_count"]

        # print(f"Time taken to do evaluation step: {end_time - start_time:.4f} seconds")

        print(f"\nSample {batch_count}:")
        # print(f"Target: {target_texts[0]}")
        # print(f"Pred:   {pred_texts[0]}")
        print(f"Current token accuracy: {token_matches_total/token_total:.2%}")

    generate_text(model, "The", 250, dataloader.dataset)

    result = {
        "val_loss": total_loss / batch_count,
        "val_token_accuracy": token_matches_total / token_total,
    }

    model.train()
    return result


def inference_step(model, x):

    model_output = model(x)
    return model_output


def analyze_embedding_clustering(model):
    with torch.no_grad():
        embeddings = model.codebook.lookup_embeddings.weight
        normalized_emb = F.normalize(embeddings, p=2, dim=1)

        # Get pairwise similarities
        sim_matrix = torch.matmul(normalized_emb, normalized_emb.t())

        # For each embedding, count how many others are "close"
        similarity_thresholds = [0.5, 0.7, 0.9]
        for thresh in similarity_thresholds:
            close_count = (sim_matrix > thresh).sum(dim=1)
            print(f"\nTokens with similarity > {thresh}:")
            print(f"Mean neighbors: {close_count.float().mean():.2f}")
            print(f"Max neighbors: {close_count.max().item()}")

            # Find most clustered embeddings
            most_clustered = torch.topk(close_count, 5)
            print(f"Most clustered token indices: {most_clustered.indices.tolist()}")


def inference_and_loss_step(dataset, model, x, y):

    # Forward pass to get output embeddings
    model_output = inference_step(model, x)  # [batch_size, seq_len, embed_dim]

    if model.modelType == JPT1QuantModelType.COS_SIM:

        lookup_embeddings = model.codebook.lookup_embeddings.weight

        logits = compute_logits(model, model_output)

        # Calculate cross entropy loss
        loss_fn = nn.CrossEntropyLoss()
        ce_loss = loss_fn(logits.view(-1, lookup_embeddings.size(0)), y.view(-1))

        loss = ce_loss

        return model_output, loss

    else:
        # For standard model types, compute logits normally and apply cross entropy over the vocab.
        logits = model_output
        loss_fn = nn.CrossEntropyLoss()
        # logits shape is assumed to be [batch, seq_len, vocab_size]
        ce_loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = ce_loss
        return model_output, loss


def train_model(
    wandb,
    model,
    train_dataloader,
    val_dataloader,
    config: dict,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_segments = config["data_segments"]

    count_parameters(model)

    # Create optimizer for both JPT1 and decoder model parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        fused=True,
    )

    # Create optimizer with Lion instead of AdamW
    # optimizer = Lion(
    #     model.parameters(),
    #     lr=config["lr"],
    #     weight_decay=0.0001,  # Lion typically works well with higher weight decay
    # )

    total_steps = config["epochs"] * len(train_dataloader) * data_segments

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy="cos",
        cycle_momentum=False,
    )

    dataset = train_dataloader.dataset

    current_lr = config["lr"]
    low_loss = 10000

    train_time_start = time.time()
    total_training_examples = 0

    loss_history = []

    eval_every_n_tokens = 1000000 * 15
    tokens_since_eval = 0

    batches_per_epoch = 100000 // config["batch_size"]

    wandb.watch(model, log_freq=batches_per_epoch, log="all")

    # eval_results = evaluate_model(
    #     model,
    #     encoder_model,
    #     decoder_model,
    #     val_dataloader,
    #     device,
    #     from_hypertoken=hypertoken_output,
    # )

    for epoch in range(config["epochs"]):
        epoch_loss = 0
        batch_count = 0
        optimizer.param_groups[0]["lr"] = current_lr

        train_epoch_start = time.time()
        tokens_processed = 0

        for segment in range(data_segments):
            segment_loss = 0
            segment_batch_count = 0

            train_step_start = time.time()

            for x, y in train_dataloader:

                x = x.to(device)
                y = y.to(device)

                tokens_since_eval += x.shape[0] * x.shape[1]

                if tokens_since_eval >= eval_every_n_tokens:
                    eval_results = evaluate_model(model, val_dataloader, device)
                    tokens_since_eval = 0
                    wandb.log(
                        {
                            "val_loss": eval_results["val_loss"],
                            "val_token_accuracy": eval_results["val_token_accuracy"],
                            "epoch": epoch,
                        }
                    )

                batch_count += 1
                segment_batch_count += 1

                total_training_examples += x.shape[0]

                start_time = time.time()

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    jpt_output, loss = inference_and_loss_step(dataset, model, x, y)

                end_time = time.time()
                # print(f"Time taken to do inference and loss step: {end_time - start_time:.4f} seconds")

                # Update metrics
                segment_loss += loss.item()
                epoch_loss += loss.item()

                loss_history.append(loss.item())

                if len(loss_history) > 20:
                    loss_history.pop(0)

                current_mean_loss = sum(loss_history) / len(loss_history)

                if current_mean_loss < low_loss:
                    low_loss = current_mean_loss
                    print(f"New low loss: {low_loss:.7f}")

                # Log batch metrics

                start_time = time.time()

                optimizer.zero_grad(set_to_none=True)

                max_grad_norm = 0.1
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                loss.backward()
                optimizer.step()
                # scheduler.step()
                end_time = time.time()
                # print(f"Time taken to do optimizer step: {end_time - start_time:.4f} seconds")

                tokens_processed += x.shape[0] * x.shape[1]  # x.shape[0] is batch size, x.shape[1] is sequence length

                train_step_end = time.time()
                # print(f"Time taken to do train step: {train_step_end - train_step_start:.4f} seconds")

                if batch_count % 10 == 0:
                    train_epoch_time = time.time() - train_epoch_start
                    tokens_per_second = tokens_processed / train_epoch_time
                    wandb.log(
                        {
                            "batch_loss": current_mean_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                            "tokens_per_second": tokens_per_second,
                        }
                    )

                train_step_start = time.time()

            avg_segment_loss = segment_loss / segment_batch_count

        eval_results = evaluate_model(model, val_dataloader, device)

        val_loss = eval_results["val_loss"]
        val_token_accuracy = eval_results["val_token_accuracy"]

        wandb.log(
            {
                "epoch_loss": epoch_loss / batch_count,
                "epoch": epoch,
            }
        )

        print(
            f"Epoch {epoch} train_loss: {avg_segment_loss:.4f}, val_loss: {val_loss:.4f}, " f"val_token_accuracy: {val_token_accuracy:.2%}"
        )

    # Final Evaluation
    eval_results = evaluate_model(
        model,
        val_dataloader,
        device,
    )

    wandb.log(
        {
            "epoch_loss": epoch_loss / batch_count,
            "val_loss": eval_results["val_loss"],
            "val_token_accuracy": eval_results["val_token_accuracy"],
            "epoch": epoch,
        }
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

    return model


def verify_model_params(config: dict):
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


def get_text_token_from_prediction_text(prediction_text: str) -> str:
    pred_text_chars = []
    for char in prediction_text:
        if char == "<EOT>":
            break
        pred_text_chars.append("" if char == "[PAD]" else char)

    return "".join(pred_text_chars)


def get_codebook(tokenizer: Tokenizer, embed_dim: int):

    codebook = TokenCodebook(tokenizer=tokenizer, embed_dim=embed_dim)

    print(f"Populated codebook with {len(codebook.token_list)} unique tokens")

    return codebook


def generate_text(
    jpt_model: nn.Module,
    prompt: str,
    max_new_tokens: int,
    dataset: BooksumDataset,
    temperature: float = 0.5,
    device: str = "cuda",
) -> str:
    # Set models to eval mode
    jpt_model.eval()

    print("Generating...\n")

    print(f"\nPrompt: {prompt}\n", end="", flush=True)

    if len(prompt) == 0:
        raise ValueError("Prompt must be at least one character long")

    result: [str] = [prompt]

    for _ in range(max_new_tokens):

        current_context = "".join(result)
        # make sure the context is not empty
        current_context = " " if current_context == "" else current_context

        tokens = dataset.codebook.tokenizer.encode(current_context).tokens
        tokens = tokens[-jpt_model.seq_len :]

        x = torch.tensor(dataset.codebook.get_token_indices(tokens)).to(device)
        x = x.unsqueeze(0)

        jpt_output = inference_step(jpt_model, x)

        cur_batch_size = jpt_output.shape[0]
        cur_seq_len = jpt_output.shape[1]

        # Print the generated character

        last_token = jpt_output[0:1, -1:, :]

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            if jpt_model.modelType == JPT1QuantModelType.COS_SIM:
                pred_token_indices = dataset.codebook.get_nearest_token_indices(last_token, top_k=1, temperature=0.1)
            else:
                pred_token_indices = last_token.argmax(dim=-1)

        next_token = dataset.codebook.get_text_token_from_indices(pred_token_indices.cpu().numpy())
        next_token = next_token.item()

        next_token = "" if next_token == "[UNK]" or next_token == "[PAD]" else next_token

        print(next_token, end="", flush=True)

        result.append(next_token)

        if len(result) > jpt_model.seq_len:
            result.pop(0)

    final_text = "".join(result)
    # print(f"\nFinal text:\n{final_text}")
    return final_text


if __name__ == "__main__":

    # Define experiments
    experiments: list[dict] = [
        {
            "seq_len": sl,
            "token_len": 8,
            "token_space_dim": token_space_dim,
            "epochs": epochs,
            "batch_size": 32,
            "lr": lr,
            "head_size": head_size,
            "n_layers": n_layers,
            "jpt_embed_dim": jed,
            "data_segments": 10,
            "dropout": dropout,
        }
        for n_layers in [6]  # Varying n_layers
        for head_size in [32]  # Varying head_size
        for jed in [384]
        for lr in [0.0007]
        for sl in [256]
        for epochs in [10]
        for dropout in [0.1]
        for token_space_dim in [8, 16, 32, 64, 128, 256, 384]
    ]

    enable_torch_optimizations()
    setup_flash_attention()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    is_debugging = sys.gettrace() is not None

    for experiment in experiments:
        seq_len = experiment["seq_len"]
        token_len = experiment["token_len"]
        batch_size = experiment["batch_size"]
        n_layers = experiment["n_layers"]
        head_size = experiment["head_size"]
        jpt_embed_dim = experiment["jpt_embed_dim"]
        data_segments = experiment["data_segments"]
        dropout = experiment["dropout"]

        token_space_dim = experiment["token_space_dim"]
        # load this just to get the vocab size

        dataset_all = BooksumDataset(
            token_len=experiment["token_len"],
            seq_len=seq_len,
            segments=data_segments,
            type="all",
            codebook=None,
            data_stride=1,
            tokenizer=None,
        )

        tokenizer = get_or_train_tokenizer(dataset_all.all_text, 30000, "tokenizer_cache/booksum_tokenizer.json")

        codebook = get_codebook(tokenizer, token_space_dim)

        dataset_train = BooksumDataset(
            token_len=experiment["token_len"],
            seq_len=seq_len,
            segments=data_segments,
            type="train",
            codebook=None,
            data_stride=1,
            tokenizer=tokenizer,
        )

        dataset_train.codebook = codebook.to(DEVICE)

        vocab_size = len(dataset_train.codebook.token_list)

        gpt_model = JPT1Quantized(
            token_space_dim=token_space_dim,
            seq_len=seq_len,
            token_len=token_len,
            embed_dim=jpt_embed_dim,
            num_heads=head_size,
            num_layers=n_layers,
            dropout=dropout,
            codebook=codebook,
            modelType=JPT1QuantModelType.COS_SIM,
        ).to(DEVICE)

        dataloader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
        )

        val_dataset = BooksumDataset(
            token_len=token_len,
            seq_len=seq_len,
            segments=data_segments,
            type="validation",
            codebook=codebook,
            data_stride=seq_len,
            tokenizer=tokenizer,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
        )

        # only if not debugging
        if sys.gettrace() is None:  # No debugger attached
            print("Compiling models...")
            gpt_model = torch.compile(gpt_model)
            codebook = torch.compile(codebook)

            print("Models compiled!")

        verify_model_params(experiment)

        # create wrapper function for train_model
        def train_model_lambda(wandb):
            model = train_model(
                wandb,
                gpt_model,
                dataloader,
                val_dataloader,
                experiment,
            )
            return model

        project_name = "jpt1"
        exp_name = f"{project_name}-sl:{experiment['seq_len']}-tl:{experiment['token_len']}-e:{experiment['epochs']}-bs:{experiment['batch_size']}-lr:{experiment['lr']}-hs:{experiment['head_size']}-nl:{experiment['n_layers']}-ed:{experiment['jpt_embed_dim']}-ts:{experiment['token_space_dim']}"

        run_experiment(project_name, train_model_lambda, exp_name, experiment)

        # generate_text(
        #     gptModel, h_decoder_model, "Hello, world!", 100, hypertoken_seq_len, dataset
        # )


print("Training Complete")
