import os
from typing import Optional
from datetime import datetime
import time
import sys
import random
import math

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

from models.jpt1_quantizer import JPT1Quantized

from datasources.fineweb10B import get_or_train_tokenizer, Fineweb10BDataset
from helpers.experiments import run_experiment, count_parameters, create_experiments
from helpers.training import (
    save_model,
    enable_torch_optimizations,
    setup_flash_attention,
)


from datasources.fineweb10B import load_hf_dataset, Fineweb10BDataset

from models.jpt1_quantizer import JPT1QuantModelType
from models.schedulers.empiriclaLRScheduler import EmpiricalLRScheduler
from schedulers.oscillatingOneCycleLR import OscillatingOneCycleLR
from helpers.utilities import calculate_token_accuracy


# --------------------------------------------------
# 2. Model Definition
# --------------------------------------------------


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    loss_fn: torch.nn.Module,
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
            jpt_output, loss = inference_and_loss_step(dataset, model, x, y, loss_fn)

            total_loss += loss.item()
            batch_count += 1

            if model.model_type == JPT1QuantModelType.COS_SIM:
                pred_token_indices = model.get_nearest_token_indices_cossim(jpt_output)
            elif model.model_type == JPT1QuantModelType.L2_SIM:
                pred_token_indices = model.get_nearest_token_indices_l2sim(jpt_output)
            else:
                pred_token_indices = jpt_output.argmax(dim=-1)

            pred_token_indices = pred_token_indices.detach().cpu().numpy()

        pred_tokens = model.get_text_token_from_indices(pred_token_indices)

        target_tokens = model.get_text_token_from_indices(y.detach().cpu().numpy())

        accuracy_metrics = calculate_token_accuracy(pred_tokens, target_tokens, dataset.token_list["[PAD]"])

        token_matches_total += accuracy_metrics["token_matches"]
        token_total += accuracy_metrics["token_count"]

        # print(f"Time taken to do evaluation step: {end_time - start_time:.4f} seconds")

        print(f"\nSample {batch_count}:")
        # print(f"Target: {target_texts[0]}")
        # print(f"Pred:   {pred_texts[0]}")
        print(f"Current token accuracy: {token_matches_total/token_total:.2%}")

    print("Generating text...")
    for _ in range(20):
        generate_text(model, "Hello, I'm a language model,", 50, dataloader.dataset)

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
        embeddings = model.lookup_embeddings.weight
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


def compute_logits_with_extras(model, hidden_states, target_indices):

    target_flat = target_indices.reshape(-1)
    unique_targets = target_flat.unique()
    lookup_embeddings = model.lookup_embeddings

    # Calculate number of extra negatives needed
    total_tokens = target_flat.size(0)  # original number of tokens (seq_len * batch_size)
    num_unique_targets = unique_targets.size(0)
    extra_negatives = max(0, total_tokens - num_unique_targets)  # ensure we don't go negative

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


class CustomLossCosSim(torch.nn.Module):
    def __init__(self, ignore_index: int, total_compare_tokens: int):
        super().__init__()
        self.ignore_index = ignore_index
        self.total_compare_tokens = total_compare_tokens

    def forward(self, model, hidden_states, target_indices):
        batch_size, seq_length, hidden_dim = hidden_states.shape
        vocab_size = model.lookup_embeddings.weight.size(0)

        # Flatten tensors
        flat_hidden = hidden_states.reshape(-1, hidden_dim)  # [batch_size*seq_length, hidden_dim]
        flat_targets = target_indices.reshape(-1)  # [batch_size*seq_length]

        # Ignore padding tokens
        ignore_mask = flat_targets == self.ignore_index
        flat_hidden = flat_hidden[~ignore_mask]
        flat_targets = flat_targets[~ignore_mask]

        # Normalize embeddings and hidden states
        flat_hidden = F.normalize(flat_hidden, p=2, dim=1)
        all_embeddings = F.normalize(model.lookup_embeddings.weight, p=2, dim=1)

        # Get unique targets from the batch
        unique_targets = torch.unique(flat_targets)

        # Sample additional negative indices
        # Target number of negative samples
        num_batch_uniques = unique_targets.shape[0]
        num_extra_samples = max(0, self.total_compare_tokens - num_batch_uniques)

        # Sample from non-batch tokens
        sampling_mask = torch.ones(vocab_size, device=flat_hidden.device, dtype=torch.bool)
        sampling_mask[unique_targets] = False
        valid_indices = torch.nonzero(sampling_mask, as_tuple=True)[0]

        # Get extra negative indices
        if num_extra_samples > 0 and len(valid_indices) > 0:
            perm = torch.randperm(len(valid_indices), device=valid_indices.device)
            num_to_sample = min(num_extra_samples, len(valid_indices))
            extra_neg_indices = valid_indices[perm[:num_to_sample]]
        else:
            extra_neg_indices = torch.tensor([], device=flat_hidden.device, dtype=torch.long)

        # Combine all indices for comparisons - positives first, then uniques, then extras
        all_indices = torch.cat([unique_targets, extra_neg_indices])  # Include all unique targets (which includes all positives)

        # Get embeddings for all indices
        comparison_embeddings = all_embeddings[all_indices]  # [num_comparisons, hidden_dim]

        # Compute all similarities at once
        all_similarities = torch.matmul(flat_hidden, comparison_embeddings.t()) / model.temperature  # [batch*seq, num_comparisons]

        # Create targets tensor - map each flat_target to its position in all_indices
        # First create a mapping from token IDs to their positions in all_indices
        indices_map = torch.zeros(vocab_size, dtype=torch.long, device=flat_hidden.device)
        indices_map[all_indices] = torch.arange(len(all_indices), device=flat_hidden.device)

        # Use the mapping to get the target positions
        targets = indices_map[flat_targets]

        # Apply cross entropy loss
        loss = F.cross_entropy(all_similarities, targets)

        return loss


class CustomLossL2Sim(torch.nn.Module):
    def __init__(self, ignore_index: int, total_compare_tokens: int):
        super().__init__()
        self.ignore_index = ignore_index
        self.total_compare_tokens = total_compare_tokens

    def forward(self, model, hidden_states, target_indices):
        batch_size, seq_length, hidden_dim = hidden_states.shape
        vocab_size = model.lookup_embeddings.weight.size(0)

        # Flatten tensors
        flat_hidden = hidden_states.reshape(-1, hidden_dim)  # [batch_size*seq_length, hidden_dim]
        flat_targets = target_indices.reshape(-1)  # [batch_size*seq_length]

        # Ignore padding tokens
        ignore_mask = flat_targets == self.ignore_index
        flat_hidden = flat_hidden[~ignore_mask]
        flat_targets = flat_targets[~ignore_mask]

        # Get unique targets from the batch
        unique_targets = torch.unique(flat_targets)

        # Sample additional negative indices
        num_batch_uniques = unique_targets.shape[0]
        num_extra_samples = max(0, self.total_compare_tokens - num_batch_uniques)

        # Sample from non-batch tokens
        sampling_mask = torch.ones(vocab_size, device=flat_hidden.device, dtype=torch.bool)
        sampling_mask[unique_targets] = False
        valid_indices = torch.nonzero(sampling_mask, as_tuple=True)[0]

        # Get extra negative indices
        if num_extra_samples > 0 and len(valid_indices) > 0:
            perm = torch.randperm(len(valid_indices), device=valid_indices.device)
            num_to_sample = min(num_extra_samples, len(valid_indices))
            extra_neg_indices = valid_indices[perm[:num_to_sample]]
        else:
            extra_neg_indices = torch.tensor([], device=flat_hidden.device, dtype=torch.long)

        # Combine all indices for comparisons
        all_indices = torch.cat([unique_targets, extra_neg_indices])

        # Get embeddings for all indices
        all_embeddings = model.lookup_embeddings.weight
        comparison_embeddings = all_embeddings[all_indices]  # [num_comparisons, hidden_dim]

        # Compute L2 distances - we need to compute pairwise distances efficiently
        # Using the formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>

        # Compute squared norms
        hidden_norm_squared = torch.sum(flat_hidden**2, dim=1, keepdim=True)  # [batch*seq, 1]
        embed_norm_squared = torch.sum(comparison_embeddings**2, dim=1, keepdim=True).t()  # [1, num_comparisons]

        # Compute dot products
        dot_products = torch.matmul(flat_hidden, comparison_embeddings.t())  # [batch*seq, num_comparisons]

        # Compute squared L2 distances
        l2_squared = hidden_norm_squared + embed_norm_squared - 2 * dot_products  # [batch*seq, num_comparisons]

        # Convert distances to similarities (negative distances scaled by temperature)
        # Lower distance = higher similarity
        all_similarities = -l2_squared / model.temperature  # [batch*seq, num_comparisons]

        # Create targets tensor - map each flat_target to its position in all_indices
        indices_map = torch.zeros(vocab_size, dtype=torch.long, device=flat_hidden.device)
        indices_map[all_indices] = torch.arange(len(all_indices), device=flat_hidden.device)

        # Use the mapping to get the target positions
        targets = indices_map[flat_targets]

        # Apply cross entropy loss
        loss = F.cross_entropy(all_similarities, targets)

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


def inference_and_loss_step(dataset, model, x, y, loss_fn):

    # Forward pass to get output embeddings
    # start_time = time.time()
    model_output = inference_step(model, x)  # [batch_size, seq_len, embed_dim]
    # end_time = time.time()
    # print(f"Inference step time: {end_time - start_time:.4f} seconds")

    if model.model_type == JPT1QuantModelType.COS_SIM or model.model_type == JPT1QuantModelType.L2_SIM:
        loss = loss_fn(model, model_output, y)
        return model_output, loss
    else:
        # For standard model types, compute logits normally and apply cross entropy over the vocab.
        logits = model_output
        # logits shape is assumed to be [batch, seq_len, vocab_size]
        ce_loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = ce_loss
        return model_output, loss


def get_grad_accum_size(completion_percentage: float, batch_tokens: int, grad_accum_size: int, hit_max_at: float) -> int:

    max_facter = 1 / hit_max_at

    return max(batch_tokens, min(grad_accum_size, max_facter * completion_percentage * grad_accum_size))


def train_model(
    wandb,
    model,
    train_dataloader,
    val_dataloader,
    config: dict,
    loss_fn: torch.nn.Module,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    count_parameters(model)

    # Create optimizer for both JPT1 and decoder model parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        fused=True,
        betas=(0.9, 0.95),
    )

    seq_len = config["seq_len"]
    batch_size = config["batch_size"]

    batch_tokens = batch_size * seq_len

    log_step_count = 0
    grad_accum_size = config["grad_accum_size"]
    log_step_size = config["log_step_size"]

    completion_percentage = 0.0

    early_end_pct = config["early_end_pct"]

    grad_accum_max_at = config["grad_accum_max_at"]

    logging_steps = 1 + (config["epochs"] * train_dataloader.dataset.token_count) // log_step_size

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        total_steps=logging_steps,
        pct_start=config["warmup_pct"],
        anneal_strategy="cos",
        cycle_momentum=False,
        div_factor=10,
        final_div_factor=1,
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
    tokens_since_grad_accum = 0

    for epoch in range(config["epochs"]):
        batch_count = 0

        loss_accum = 0

        train_step_start = time.time()

        # Grad accum size is a function of the completion percentage we reach 100% at 50% completion
        current_grad_accum_size = get_grad_accum_size(completion_percentage, batch_tokens, grad_accum_size, grad_accum_max_at)
        grad_accum_step_count = math.ceil(current_grad_accum_size / batch_tokens)

        for x, y in train_dataloader:
            # start_time = time.time()
            x = x.to(device)
            y = y.to(device)

            tokens_processed = x.shape[0] * x.shape[1]
            tokens_since_step += tokens_processed
            tokens_since_grad_accum += tokens_processed

            batch_count += 1

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                jpt_output, loss = inference_and_loss_step(dataset, model, x, y, loss_fn)
                # logits = model(x, y)
                # loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            loss = loss / grad_accum_step_count
            loss_accum += loss.detach()
            loss.backward()

            if tokens_since_grad_accum >= current_grad_accum_size:

                # Add gradient clipping
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    4.0 if model.model_type == JPT1QuantModelType.COS_SIM or model.model_type == JPT1QuantModelType.L2_SIM else 1,
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                torch.cuda.synchronize()

                # loss and timing stuff
                current_loss = loss_accum
                loss_accum = 0
                loss_history.append(current_loss)
                if len(loss_history) > 50:
                    loss_history.pop(0)
                current_mean_loss = sum(loss_history) / len(loss_history)
                tokens_per_second = tokens_since_grad_accum / (time.time() - train_step_start)

                tokens_since_grad_accum = 0

                if current_mean_loss < low_loss:
                    low_loss = current_mean_loss
                    print(
                        f"\nNew low loss: {low_loss:.7f}, Batch Time: {time.time() - train_step_start:.2f},Grad Norm: {norm:.2f}, Tokens per second: {tokens_per_second:.2f}"
                    )

                if tokens_since_step >= log_step_size:
                    tokens_since_step = 0
                    log_step_count += 1

                    wandb.log(
                        {
                            "loss": current_mean_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                            "tokens_per_second": tokens_per_second,
                            "current_grad_accum_size": current_grad_accum_size,
                            "grad_norm": norm,
                        }
                    )

                    if log_step_count % 50 == 0:
                        eval_results = evaluate_model(model, val_dataloader, device, loss_fn)
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

                    completion_percentage = log_step_count / logging_steps
                    scheduler.step()

                # Grad accum size is a function of the completion percentage we reach 100% at grad_accum_max_at completion
                current_grad_accum_size = get_grad_accum_size(completion_percentage, batch_tokens, grad_accum_size, grad_accum_max_at)
                grad_accum_step_count = math.ceil(current_grad_accum_size / batch_tokens)

                torch.cuda.synchronize()
                train_step_start = time.time()

                if early_end_pct is not None and completion_percentage > early_end_pct:
                    break

            # print(
            #     f"\nEpoch {epoch} train_loss: {current_mean_loss:.4f}, val_loss: {val_loss:.4f}, "
            #     f"val_token_accuracy: {val_token_accuracy:.2%}"
            # )
        # Final Evaluation
        eval_results = evaluate_model(
            model,
            val_dataloader,
            device,
            loss_fn,
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


def generate_text(
    model: nn.Module,
    prompt: str,
    max_new_tokens: int,
    dataset: Fineweb10BDataset,
    temperature: float = 0.5,
    device: str = "cuda",
) -> str:
    # Set models to eval mode
    model.eval()

    print(f"\n\nPrompt: {prompt}", end="", flush=True)

    if len(prompt) == 0:
        raise ValueError("Prompt must be at least one character long")

    result: [str] = [prompt]

    for _ in range(max_new_tokens):

        current_context = "".join(result)
        # make sure the context is not empty
        current_context = " " if current_context == "" else current_context

        tokens = dataset.tokenizer.encode(current_context).tokens
        tokens = tokens[-model.seq_len :]

        x = torch.tensor(model.get_token_indices(tokens)).to(device)
        x = x.unsqueeze(0)

        jpt_output = inference_step(model, x)

        cur_batch_size = jpt_output.shape[0]
        cur_seq_len = jpt_output.shape[1]

        # Print the generated character

        last_token = jpt_output[0:1, -1:, :]

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            if model.model_type == JPT1QuantModelType.COS_SIM:
                pred_token_indices = model.get_nearest_token_indices_cossim(last_token, top_k=50, temperature=temperature)
            elif model.model_type == JPT1QuantModelType.L2_SIM:
                pred_token_indices = model.get_nearest_token_indices_l2sim(last_token, top_k=50, temperature=temperature)
            else:
                # Apply temperature and sample using top-k
                logits = last_token.squeeze() / temperature

                probs = torch.softmax(logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, 50)

                # Sample from the filtered distribution
                pred_token_indices = torch.multinomial(top_k_probs, num_samples=1).unsqueeze(0)

                # Map back to original indices
                pred_token_indices = top_k_indices[pred_token_indices]

        next_token = model.get_text_token_from_indices(pred_token_indices.cpu().numpy())
        next_token = next_token.item()

        next_token = "" if next_token == "[UNK]" or next_token == "[PAD]" else next_token

        print(next_token, end="", flush=True)

        result.append(next_token)

        if len(result) > model.seq_len:
            result.pop(0)

    final_text = "".join(result)
    # print(f"\nFinal text:\n{final_text}")
    return final_text


if __name__ == "__main__":

    bs = 24
    # Define experiments
    experiments: list[dict] = {
        "seq_len": [1024],
        "token_space_dim": [768],
        "epochs": [1],
        "batch_size": [bs],
        "lr": [0.00015, 0.0004, 0.001],
        "num_head": [12],
        "n_layers": [12],
        "jpt_embed_dim": [768],
        "dropout": [0.0],
        "vocab_size": [50304],
        "output_type": [
            JPT1QuantModelType.COS_SIM,
        ],
        "grad_accum_size": [bs * 1024, bs * 1024, bs * 1024 * 20],
        "log_step_size": [bs * 1024 * 20 * 2],
        "dset_ratio": [1],
        "warmup_pct": [0.01, 0.05, 0.1],
        "grad_accum_max_at": [0.1],
        "early_end_pct": [0.6],
        "total_compare_tokens": [12 * 1024],
    }

    experiments = create_experiments(mode="paired", **experiments)

    enable_torch_optimizations()
    setup_flash_attention()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    is_debugging = sys.gettrace() is not None

    for experiment in experiments:
        seq_len = experiment["seq_len"]
        batch_size = experiment["batch_size"]
        n_layers = experiment["n_layers"]
        num_head = experiment["num_head"]
        jpt_embed_dim = experiment["jpt_embed_dim"]
        dropout = experiment["dropout"]
        vocab_size = experiment["vocab_size"]
        output_type = experiment["output_type"]
        token_space_dim = experiment["token_space_dim"]
        grad_accum_size = experiment["grad_accum_size"]
        log_step_size = experiment["log_step_size"]
        dset_ratio = experiment["dset_ratio"]
        total_compare_tokens = experiment["total_compare_tokens"]
        # load this just to get the vocab size

        dataset_name = "fineweb-10BT-edu"

        hf_dataset = load_hf_dataset(dataset_name)

        text_corpus_iterator = (item["text"] for item in hf_dataset["train"])
        tokenizer = get_or_train_tokenizer(text_corpus_iterator, vocab_size, f"tokenizer_cache/{dataset_name}_tokenizer_{vocab_size}.json")

        loss_fn = None
        if output_type == JPT1QuantModelType.COS_SIM:
            loss_fn = CustomLossCosSim(ignore_index=tokenizer.token_to_id("[PAD]"), total_compare_tokens=total_compare_tokens)
        elif output_type == JPT1QuantModelType.L2_SIM:
            loss_fn = CustomLossL2Sim(ignore_index=tokenizer.token_to_id("[PAD]"), total_compare_tokens=total_compare_tokens)
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

        dataset_train = Fineweb10BDataset(
            seq_len=seq_len,
            type="train",
            data_stride=seq_len,
            tokenizer=tokenizer,
            hf_dataset=hf_dataset,
            dataset_name=dataset_name,
            dset_ratio=dset_ratio,
        )

        vocab_size = len(tokenizer.get_vocab())

        gpt_model = JPT1Quantized(
            token_space_dim=token_space_dim,
            seq_len=seq_len,
            embed_dim=jpt_embed_dim,
            num_head=num_head,
            num_layers=n_layers,
            dropout=dropout,
            tokenizer=tokenizer,
            model_type=output_type,
        ).to(DEVICE)

        # only if not debugging
        if sys.gettrace() is None:  # No debugger attached
            print("Compiling models...")
            gpt_model = torch.compile(gpt_model)
            loss_fn = torch.compile(loss_fn)

            print("Models compiled!")

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
            data_stride=seq_len,
            tokenizer=tokenizer,
            hf_dataset=hf_dataset,
            dataset_name=dataset_name,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=8,
        )

        verify_model_params(experiment)

        # create wrapper function for train_model
        def train_model_lambda(wandb):
            model = train_model(
                wandb,
                gpt_model,
                dataloader,
                val_dataloader,
                experiment,
                loss_fn,
            )
            return model

        project_name = "jpt1"
        exp_name = f"{project_name}-sl:{experiment['seq_len']}-e:{experiment['epochs']}-bs:{experiment['batch_size']}-lr:{experiment['lr']}-hs:{experiment['num_head']}-nl:{experiment['n_layers']}-ed:{experiment['jpt_embed_dim']}-ts:{experiment['token_space_dim']}"

        run_experiment(project_name, train_model_lambda, exp_name, experiment)

        # generate_text(
        #     gptModel, h_decoder_model, "Hello, world!", 100, hypertoken_seq_len, dataset
        # )


print("Training Complete")
