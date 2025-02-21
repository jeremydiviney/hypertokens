import os
from typing import Optional
from datetime import datetime
import time
import sys
import random

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

from models.jpt1_quantizer import JPT1Quantized, TokenCodebook

from datasources.fineweb10B import get_or_train_tokenizer, Fineweb10BDataset
from helpers.experiments import run_experiment, count_parameters
from helpers.training import (
    save_model,
    enable_torch_optimizations,
    setup_flash_attention,
)


from datasources.fineweb10B import load_hf_dataset, Fineweb10BDataset

from models.jpt1_quantizer import JPT1QuantModelType
from models.schedulers.empiriclaLRScheduler import EmpiricalLRScheduler
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

    token_matches_total = 0

    token_total = 0

    for x, y in dataloader:

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            jpt_output, loss = inference_and_loss_step(dataset, model, x, y)

            total_loss += loss.item()
            batch_count += 1

            if model.model_type == JPT1QuantModelType.COS_SIM:
                pred_token_indices = dataset.codebook.get_nearest_token_indices_cossim(jpt_output)
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


def compute_logits_with_extras(model, hidden_states, target_indices, extra_negatives):
    # Flatten target indices and get unique targets.
    target_flat = target_indices.view(-1)
    unique_targets = target_flat.unique()

    lookup_embeddings = model.codebook.lookup_embeddings

    if extra_negatives > 0:
        vocab_size = lookup_embeddings.weight.size(0)
        # Create a mask to exclude unique targets.
        mask = torch.ones(vocab_size, dtype=torch.bool, device=hidden_states.device)
        mask[unique_targets] = False
        non_target_tokens = mask.nonzero(as_tuple=True)[0]
        perm = torch.randperm(non_target_tokens.size(0), device=hidden_states.device)
        extra_candidates = non_target_tokens[perm[:extra_negatives]]
        candidate_set = torch.sort(torch.cat([unique_targets, extra_candidates]))[0]
    else:
        candidate_set = unique_targets

    # Retrieve and normalize candidate embeddings.
    candidate_embeds = lookup_embeddings(candidate_set)
    candidate_embeds = F.normalize(candidate_embeds, p=2, dim=1)

    # Flatten and normalize hidden states.
    N, D = target_flat.shape[0], hidden_states.shape[-1]
    hidden_norm = F.normalize(hidden_states.view(N, D), p=2, dim=1)

    # Compute logits as cosine similarities scaled by temperature.
    logits = torch.mm(hidden_norm, candidate_embeds.t()) / model.temperature

    # Remap each original target to its position in candidate_set using a mapping tensor.
    mapping = torch.full((lookup_embeddings.weight.size(0),), -1, dtype=torch.long, device=candidate_set.device)
    mapping[candidate_set] = torch.arange(candidate_set.size(0), device=candidate_set.device)
    new_targets = mapping[target_flat]

    return logits, new_targets


def unique_batch_cosine_ce_loss(model: nn.Module, pred: torch.Tensor, target_indices: torch.Tensor, extra_negatives=16000) -> torch.Tensor:
    """
    Computes cross entropy loss where logits are based on negative MSE distances.
    """

    logits, new_targets = compute_logits_with_extras(model, pred, target_indices, extra_negatives)

    loss = F.cross_entropy(logits, new_targets)

    return loss


def compute_gate_loss(model: nn.Module, gate_weights: torch.Tensor, alpha: float = 2) -> torch.Tensor:

    num_experts = gate_weights.shape[-1]

    expert_usage = gate_weights.sum(dim=(0, 1))  # shape [num_experts]
    total = expert_usage.sum()  # total "token mass"
    usage_dist = expert_usage / total  # shape [num_experts]

    alpha = num_experts / 3

    uniform_dist = torch.full_like(usage_dist, 1.0 / num_experts)
    balancing_loss = alpha * F.mse_loss(usage_dist, uniform_dist)
    return balancing_loss


def inference_and_loss_step(dataset, model, x, y):

    # Forward pass to get output embeddings

    model_output = inference_step(model, x)  # [batch_size, seq_len, embed_dim]

    if model.model_type == JPT1QuantModelType.COS_SIM:

        loss = unique_batch_cosine_ce_loss(model, model_output, y)

        # gate_loss = compute_gate_loss(model, gate_weights)
        # norm_loss = compute_norm_loss(model_output)
        # loss += gate_loss  # + norm_loss
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

    count_parameters(model)

    # Create optimizer for both JPT1 and decoder model parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        fused=True,
    )

    step_count = 0
    step_size = 1_000_000

    batch_tokens = config["batch_size"] * config["seq_len"]

    total_steps = (config["epochs"] * train_dataloader.dataset.token_count) // step_size
    scheduler_steps = 1 + (config["epochs"] * train_dataloader.dataset.token_count) // batch_tokens

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        total_steps=scheduler_steps,
        pct_start=0.01,
        anneal_strategy="cos",
        cycle_momentum=False,
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=3,  # Min lr = initial_lr/10 = max_lr/(25*10)
    )

    # scheduler = EmpiricalLRScheduler(
    #     optimizer,
    #     alpha=0.025,
    #     n=9,
    # )

    dataset = train_dataloader.dataset

    current_lr = config["lr"]
    low_loss = 10e10

    train_time_start = time.time()

    loss_history = []

    tokens_since_step = 0

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

    tokens_processed = 0
    train_time_accumulated = 0

    for epoch in range(config["epochs"]):
        batch_count = 0

        train_step_start = time.time()

        for x, y in train_dataloader:
            # start_time = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            tokens_since_step += x.shape[0] * x.shape[1]

            batch_count += 1

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                jpt_output, loss = inference_and_loss_step(dataset, model, x, y)

            # Update metrics

            current_loss = loss.item()

            loss_history.append(current_loss)

            if len(loss_history) > 50:
                loss_history.pop(0)

            current_mean_loss = sum(loss_history) / len(loss_history)

            if current_mean_loss < low_loss:
                low_loss = current_mean_loss
                print(f"\nNew low loss: {low_loss:.7f},")

            # Log batch metrics

            optimizer.zero_grad(set_to_none=True)

            max_grad_norm = 1
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            loss.backward()
            optimizer.step()
            scheduler.step()

            tokens_processed += x.shape[0] * x.shape[1]  # x.shape[0] is batch size, x.shape[1] is sequence length

            # end_time = time.time()
            # print(f"Train step time: {end_time - start_time:.4f} seconds")

            if tokens_since_step >= step_size:
                tokens_since_step = 0
                step_count += 1

                step_time = time.time() - train_step_start
                train_time_accumulated += step_time
                tokens_per_second = tokens_processed / train_time_accumulated

                wandb.log(
                    {
                        "loss": current_mean_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "tokens_per_second": tokens_per_second,
                    }
                )

                if step_count % 100 == 0:
                    eval_results = evaluate_model(model, val_dataloader, device)
                    val_loss = eval_results["val_loss"]
                    val_token_accuracy = eval_results["val_token_accuracy"]
                    wandb.log(
                        {
                            "val_loss": val_loss,
                            "val_token_accuracy": eval_results["val_token_accuracy"],
                            "epoch": epoch,
                        }
                    )
                    print(
                        f"\nEpoch {epoch} train_loss: {current_mean_loss:.4f}, val_loss: {val_loss:.4f}, "
                        f"val_token_accuracy: {val_token_accuracy:.2%}"
                        f"tokens_per_second: {tokens_per_second:.2f}"
                    )

                train_step_start = time.time()

            # print(
            #     f"\nEpoch {epoch} train_loss: {current_mean_loss:.4f}, val_loss: {val_loss:.4f}, "
            #     f"val_token_accuracy: {val_token_accuracy:.2%}"
            # )
        # Final Evaluation
        eval_results = evaluate_model(
            model,
            val_dataloader,
            device,
        )

        wandb.log(
            {
                "val_loss": eval_results["val_loss"],
                "val_token_accuracy": eval_results["val_token_accuracy"],
                "epoch": epoch,
            }
        )
        train_time_end = time.time()

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
    dataset: Fineweb10BDataset,
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
            if jpt_model.model_type == JPT1QuantModelType.COS_SIM:
                pred_token_indices = dataset.codebook.get_nearest_token_indices_cossim(last_token, top_k=1, temperature=0.1)
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
            "token_space_dim": token_space_dim,
            "epochs": epochs,
            "batch_size": 12,
            "lr": lr,
            "head_size": head_size,
            "n_layers": n_layers,
            "jpt_embed_dim": jed,
            "dropout": dropout,
            "vocab_size": vocab_size,
            "output_type": output_type,
        }
        for n_layers in [12]  # Varying n_layers
        for jed in [768]
        for head_size in [64]  # Varying head_size
        for lr in [0.0009]
        for sl in [1024]
        for epochs in [1]
        for dropout in [0.0]
        for token_space_dim in [jed]
        for vocab_size in [50304]
        for output_type in [JPT1QuantModelType.COS_SIM]
    ]

    enable_torch_optimizations()
    setup_flash_attention()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    is_debugging = sys.gettrace() is not None

    for experiment in experiments:
        seq_len = experiment["seq_len"]
        batch_size = experiment["batch_size"]
        n_layers = experiment["n_layers"]
        head_size = experiment["head_size"]
        jpt_embed_dim = experiment["jpt_embed_dim"]
        dropout = experiment["dropout"]
        vocab_size = experiment["vocab_size"]
        output_type = experiment["output_type"]
        token_space_dim = experiment["token_space_dim"]
        # load this just to get the vocab size

        dataset_name = "fineweb-10BT"

        hf_dataset = load_hf_dataset()

        text_corpus_iterator = (item["text"] for item in hf_dataset["train"])
        tokenizer = get_or_train_tokenizer(text_corpus_iterator, vocab_size, f"tokenizer_cache/{dataset_name}_tokenizer_{vocab_size}.json")

        codebook = get_codebook(tokenizer, token_space_dim).to(DEVICE)

        dataset_train = Fineweb10BDataset(
            seq_len=seq_len,
            type="train",
            codebook=codebook,
            data_stride=seq_len,
            tokenizer=tokenizer,
            hf_dataset=hf_dataset,
        )

        vocab_size = len(tokenizer.get_vocab())

        gpt_model = JPT1Quantized(
            token_space_dim=token_space_dim,
            seq_len=seq_len,
            embed_dim=jpt_embed_dim,
            num_heads=head_size,
            num_layers=n_layers,
            dropout=dropout,
            codebook=codebook,
            model_type=output_type,
        ).to(DEVICE)

        dataloader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=8,
        )

        val_dataset = Fineweb10BDataset(
            seq_len=seq_len,
            type="validation",
            codebook=codebook,
            data_stride=seq_len,
            tokenizer=tokenizer,
            hf_dataset=hf_dataset,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=8,
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
        exp_name = f"{project_name}-sl:{experiment['seq_len']}-e:{experiment['epochs']}-bs:{experiment['batch_size']}-lr:{experiment['lr']}-hs:{experiment['head_size']}-nl:{experiment['n_layers']}-ed:{experiment['jpt_embed_dim']}-ts:{experiment['token_space_dim']}"

        run_experiment(project_name, train_model_lambda, exp_name, experiment)

        # generate_text(
        #     gptModel, h_decoder_model, "Hello, world!", 100, hypertoken_seq_len, dataset
        # )


print("Training Complete")
