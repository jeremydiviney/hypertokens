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

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from models.transformer_pyramid_hypertoken_auto_encoder import (
    TransformerPyramidHyperTokenEncoder,
    TransformerPyramidHyperTokenDecoder,
)

from models.jpt1 import JPT1, ExpandMethod
from datasources.tinyshakespeare import HyperTokenTinyShakespeareDataset, TinyShakespeareDataset, decode_indices_to_text
from helpers.experiments import run_experiment, count_parameters
from helpers.training import (
    save_model,
    enable_torch_optimizations,
    setup_flash_attention,
)

from helpers.utilities import calculate_text_accuracy

# --------------------------------------------------
# 2. Model Definition
# --------------------------------------------------


@typechecked
def decode_text_from_hypertoken(
    dataset: Dataset,
    hypertoken: TensorType["batch_size", "seq_len", "hypertoken_size"],
    decoder: nn.Module,
    token_len: int,
    temperature: float = 0,
) -> np.ndarray[str]:  # --->["batch_size", "seq_len", "character"]:

    decoder.eval()

    batch_size, seq_len, hypertoken_size = hypertoken.shape

    with torch.inference_mode(), autocast(device_type="cuda", dtype=torch.bfloat16):
        hypertoken = hypertoken.reshape(-1, hypertoken_size)
        decoder_output = decoder(hypertoken)  # [batch_size, seq_len, vocab_size]
        decoder_output = decoder_output.reshape(batch_size, seq_len, token_len, -1)

    # Get predicted indices for each position in the sequence
    if temperature > 0:
        # Apply temperature scaling
        decoder_output = decoder_output / temperature

        # Apply softmax - handle 3D tensor
        probs = F.softmax(decoder_output, dim=-1).cpu()  # Remove seq_len dimension

        # Apply top-k sampling
        k = 12  # Adjust k value as needed
        top_k = min(k, probs.size(-1))

        # Zero out probabilities below top k
        top_k_values, _ = torch.topk(probs, top_k, dim=-1)
        indices_to_remove = probs < top_k_values[..., -1, None]
        probs[indices_to_remove] = 0

        # Renormalize probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample from the filtered distribution
        pred_indices = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(
            probs.size(0), probs.size(1), probs.size(2)
        )
    else:
        pred_indices = torch.argmax(decoder_output, dim=-1).cpu().unsqueeze(-1)

    # Convert indices to text for each item in the batch

    # start_time = time.time()

    # for indices in pred_indices:
    #     text = "".join([dataset.idx2char[idx.item()] for idx in indices])
    #     batch_texts.append(text)

    batch_texts = decode_indices_to_text(
        pred_indices.view(batch_size, seq_len, token_len), dataset.idx2char, dataset.pad_token
    )

    # end_time = time.time()
    # print(f"Time taken to decode: {end_time - start_time:.4f} seconds")

    return batch_texts


def evaluate_model(
    model: nn.Module,
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    dataloader: DataLoader,
    device: str,
    from_hypertoken: bool,
) -> dict:
    encoder_model.eval()
    model.eval()
    decoder_model.eval()

    dataset = dataloader.dataset

    total_loss = 0
    batch_count = 0

    char_matches_total = 0
    token_matches_total = 0
    char_total = 0
    token_total = 0

    for x, y in dataloader:
        x_tokens = x.to(device)
        y_tokens = y.to(device)

        start_time = time.time()

        current_batch_size = x_tokens.shape[0]

        jpt_output, decoder_output, loss = inference_and_loss_step(
            dataset, model, encoder_model, decoder_model, x_tokens, y_tokens
        )

        decoder_output = decoder_output.reshape(current_batch_size, seq_len, token_len, -1)

        total_loss += loss.item()
        batch_count += 1

        target_texts = decode_indices_to_text(y_tokens, dataloader.dataset.idx2char, dataloader.dataset.pad_token)

        if from_hypertoken:

            # we are actually dealing with hypertokens not characters and character probabilities
            final_hypertoken_pred = jpt_output
            pred_texts = decode_text_from_hypertoken(
                dataloader.dataset, final_hypertoken_pred, decoder_model, decoder_model.token_len
            )

        else:

            pred_indices = torch.argmax(decoder_output, dim=-1)

            pred_texts = decode_indices_to_text(pred_indices, dataloader.dataset.idx2char, dataloader.dataset.pad_token)

        accuracy_metrics = calculate_text_accuracy(pred_texts, target_texts, dataset.idx2char[dataset.pad_token])

        char_matches_total += accuracy_metrics["char_matches"]
        token_matches_total += accuracy_metrics["token_matches"]
        char_total += accuracy_metrics["char_count"]
        token_total += accuracy_metrics["token_count"]

        end_time = time.time()
        # print(f"Time taken to do evaluation step: {end_time - start_time:.4f} seconds")

        print(f"\nSample {batch_count}:")
        # print(f"Target: {target_texts[0]}")
        # print(f"Pred:   {pred_texts[0]}")
        print(f"Current token accuracy: {token_matches_total/token_total:.2%}")
        print(f"Current character accuracy: {char_matches_total/char_total:.2%}")

    generate_text(
        model, encoder_model, decoder_model, "KING:", 250, dataloader.dataset, from_hypertoken=from_hypertoken
    )

    result = {
        "val_loss": total_loss / batch_count,
        "val_token_accuracy": token_matches_total / token_total,
        "val_char_accuracy": char_matches_total / char_total,
    }

    model.train()
    decoder_model.train()
    return result


def calculate_hypertoken_loss(model_output, target_hypertokens):

    # cos_similarity = nn.CosineSimilarity(dim=-1)
    # cos_loss = 1 - cos_similarity(model_output, target_hypertokens).mean()

    hypertoken_norms = torch.norm(model_output, dim=-1)

    norm_loss = F.l1_loss(hypertoken_norms, torch.ones_like(hypertoken_norms))

    return norm_loss
    # return cos_loss + norm_loss

    # return huber_loss


def calculate_loss(model_output, target_chars, pad_token: int):
    # model_output is [batch_size, seq_len, token_len, vocab_size]
    # target_chars is [batch_size, seq_len, token_len]

    batch_size, seq_len, token_len, vocab_size = model_output.shape

    mask = target_chars != pad_token

    target_chars = target_chars[mask]
    model_output = model_output[mask]

    loss = F.cross_entropy(model_output, target_chars).mean()

    return loss


def inference_and_loss_step(dataset, model, encoder_model, decoder_model, x_tokens, y_tokens):

    decoder_device = next(decoder_model.parameters()).device

    seq_len = x_tokens.shape[1]
    token_len = x_tokens.shape[2]

    with autocast(device_type="cuda", dtype=torch.bfloat16):

        all_tokens = torch.cat([x_tokens, y_tokens[:, -1:]], dim=1)

        hypertokens = dataset.encode_to_hypertokens(encoder_model, all_tokens, token_len)
        hypertokens = torch.stack(hypertokens)
        x_hypertokens = hypertokens[:, :-1]
        y_hypertokens = hypertokens[:, 1:]

        # Forward through JPT1
        jpt_output = model(x_hypertokens)  # [batch_size, seq_len, hypertoken_size]

        # Get decoder output for each token
        batch_size, seq_len, token_len, vocab_size = jpt_output.shape

        # decoder_output = decoder_model(jpt_output.reshape(-1, jpt_output.shape[-1]))

        # loss_ht = calculate_hypertoken_loss(jpt_output, y_hypertokens)
        decoder_output = jpt_output
        loss_ce = calculate_loss(
            decoder_output.reshape(batch_size, seq_len, token_len, -1), y_tokens, dataset.pad_token
        )

        # Maybe scale cross entropy loss down
        # loss = loss_ht + loss_ce
        loss = loss_ce

    return jpt_output, decoder_output, loss


def train_model(
    wandb,
    model,
    encoder_model,
    decoder_model,
    train_dataloader,
    val_dataloader,
    config: dict,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_segments = config["data_segments"]

    # Freeze decoder parameters to prevent updates
    for param in decoder_model.parameters():
        param.requires_grad = True
    decoder_model.train()  # Set decoder to evaluation mode

    count_parameters(model)
    count_parameters(decoder_model)

    # Create optimizer for both JPT1 and decoder model parameters
    optimizer = optim.AdamW(
        list(model.parameters()) + list(decoder_model.parameters()),
        # model.parameters(),
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

    # set hypertoken output mode
    hypertoken_output = False

    dataset = train_dataloader.dataset

    current_lr = config["lr"]
    low_loss = 10000

    train_time_start = time.time()
    total_training_examples = 0

    loss_history = []

    eval_every_n_tokens = 1000000
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

                x_tokens = x.to(device)
                y_tokens = y.to(device)

                tokens_since_eval += x.shape[0] * x.shape[1]

                if tokens_since_eval >= eval_every_n_tokens:
                    eval_results = evaluate_model(
                        model, encoder_model, decoder_model, val_dataloader, device, hypertoken_output
                    )
                    tokens_since_eval = 0
                    wandb.log(
                        {
                            "val_loss": eval_results["val_loss"],
                            "val_token_accuracy": eval_results["val_token_accuracy"],
                            "val_char_accuracy": eval_results["val_char_accuracy"],
                            "epoch": epoch,
                        }
                    )

                batch_count += 1
                segment_batch_count += 1

                total_training_examples += x.shape[0]

                start_time = time.time()

                jpt_output, decoder_output, loss = inference_and_loss_step(
                    dataset, model, encoder_model, decoder_model, x_tokens, y_tokens
                )

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

        eval_results = evaluate_model(model, encoder_model, decoder_model, val_dataloader, device, hypertoken_output)

        val_loss = eval_results["val_loss"]
        val_token_accuracy = eval_results["val_token_accuracy"]
        val_char_accuracy = eval_results["val_char_accuracy"]

        wandb.log(
            {
                "epoch_loss": epoch_loss / batch_count,
                "epoch": epoch,
            }
        )

        print(
            f"Epoch {epoch} train_loss: {avg_segment_loss:.4f}, val_loss: {val_loss:.4f}, "
            f"val_token_accuracy: {val_token_accuracy:.2%}, val_char_accuracy: {val_char_accuracy:.2%}"
        )

    # Final Evaluation
    eval_results = evaluate_model(
        model,
        encoder_model,
        decoder_model,
        val_dataloader,
        device,
        from_hypertoken=hypertoken_output,
    )

    decoder_model.train()

    wandb.log(
        {
            "epoch_loss": epoch_loss / batch_count,
            "val_loss": eval_results["val_loss"],
            "val_token": eval_results["val_token_accuracy"],
            "val_char_accuracy": eval_results["val_char_accuracy"],
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
        raise ValueError(f"embed_dim must be at least 2x hypertoken_size. Current ratio: {multiple}")

    # because we tile in the hypertoken into the jpt_embed_dim
    if config["jpt_embed_dim"] % config["hypertoken_size"] != 0:
        raise ValueError(
            f"embed_dim ({config['jpt_embed_dim']}) must be a multiple of hypertoken_size ({config['hypertoken_size']})"
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

    for i in range(num_samples):
        input_chunk = dataset.get_batch_item(i, seq_len=1)
        input_chunk = torch.tensor(input_chunk).view(-1).to(device)

        # Convert to text for display
        input_text = [dataset.idx2char[idx.item()] for idx in input_chunk]

        with torch.inference_mode():
            # Encode
            encoded = encoder(input_chunk.unsqueeze(0))
            # Decode
            decoded = decoder(encoded)

        # Get predicted text
        pred_indices = torch.argmax(decoded, dim=-1)[0]
        output_text = [dataset.idx2char[idx.item()] for idx in pred_indices]

        output_text = get_text_token_from_prediction_text(output_text)
        input_text = get_text_token_from_prediction_text(input_text)

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


def get_text_token_from_prediction_text(prediction_text: str) -> str:
    pred_text_chars = []
    for char in prediction_text:
        if char == "<EOT>":
            break
        pred_text_chars.append("" if char == "[PAD]" else char)

    return "".join(pred_text_chars)


def generate_text(
    jpt_model: nn.Module,
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    prompt: str,
    max_new_tokens: int,
    dataset: HyperTokenTinyShakespeareDataset,
    temperature: float = 0.5,
    from_hypertoken: bool = True,
    device: str = "cuda",
) -> str:
    # Set models to eval mode
    jpt_model.eval()
    decoder_model.eval()

    print("Generating...\n")

    print(f"\nPrompt: {prompt}\n", end="", flush=True)

    if len(prompt) == 0:
        raise ValueError("Prompt must be at least one character long")

    result: [str] = list(prompt)

    for i in range(max_new_tokens):

        current_context = "".join(result)
        # make sure the context is not empty
        current_context = " " if current_context == "" else current_context

        encoded_list = dataset.encode_to_hypertokens_from_text(encoder_model, current_context, jpt_model.seq_len)
        encoded = torch.stack(encoded_list).to(device)

        jpt_output = jpt_model(encoded)

        cur_batch_size = jpt_output.shape[0]
        cur_seq_len = jpt_output.shape[1]

        if from_hypertoken:
            hypertoken = jpt_output[0:1, -1:, :]
            pred_texts = decode_text_from_hypertoken(dataset, hypertoken, decoder_model, encoder_model.token_len, 0)

        else:

            # decoder_output = decoder_model(jpt_output.reshape(-1, jpt_output.shape[-1]))

            # decoder_output = decoder_output.reshape(cur_batch_size, cur_seq_len, token_len, -1)

            decoder_output = jpt_output

            pred_indices = torch.argmax(decoder_output, dim=-1)

            pred_texts = decode_indices_to_text(pred_indices, dataset.idx2char, dataset.pad_token)

        # Print the generated character

        next_token = get_text_token_from_prediction_text(pred_texts[0][-1])

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
            "token_len": 16,
            "hypertoken_size": 64,
            "autoencoder_head_size": 32,
            "autoencoder_n_layers": 1,
            "compress_factor": 4,
            "epochs": epochs,
            "batch_size": 32,
            "lr": lr,
            "head_size": head_size,
            "n_layers": n_layers,
            "hypertoken_embed_dim": 256,
            "jpt_embed_dim": jed,
            "data_segments": 10,
            "dropout": dropout,
            "expand_method": expand_method,
        }
        for n_layers in [4]  # Varying n_layers
        for head_size in [32]  # Varying head_size
        for jed in [512]
        for lr in [0.003]
        for sl in [128]
        for epochs in [3]
        for dropout in [0.2]
        for expand_method in [
            ExpandMethod.TILED,
        ]
    ]

    enable_torch_optimizations()
    setup_flash_attention()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 0

    is_debugging = sys.gettrace() is not None

    for experiment in experiments:
        seq_len = experiment["seq_len"]
        token_len = experiment["token_len"]
        hypertoken_size = experiment["hypertoken_size"]
        autoencoder_head_size = experiment["autoencoder_head_size"]
        autoencoder_n_layers = experiment["autoencoder_n_layers"]
        compress_factor = experiment["compress_factor"]
        hypertoken_embed_dim = experiment["hypertoken_embed_dim"]
        batch_size = experiment["batch_size"]
        n_layers = experiment["n_layers"]
        head_size = experiment["head_size"]
        jpt_embed_dim = experiment["jpt_embed_dim"]
        data_segments = experiment["data_segments"]
        dropout = experiment["dropout"]
        expand_method = experiment["expand_method"]
        # load this just to get the vocab size
        if vocab_size == 0:
            tmp_dset = TinyShakespeareDataset(
                token_len=experiment["token_len"],
                segments=data_segments,
            )
            vocab_size = len(tmp_dset.char2idx)
            del tmp_dset

        # delete the tmp dataset

        h_encoder_model = TransformerPyramidHyperTokenEncoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=hypertoken_embed_dim,
            head_size=autoencoder_head_size,
            n_layers=autoencoder_n_layers,
            hypertoken_size=hypertoken_size,
            compress_factor=compress_factor,
        ).to(DEVICE)

        h_decoder_model = TransformerPyramidHyperTokenDecoder(
            vocab_size=vocab_size,
            token_len=token_len,
            embed_dim=hypertoken_embed_dim,
            head_size=autoencoder_head_size,
            n_layers=autoencoder_n_layers,
            hypertoken_size=hypertoken_size,
            compress_factor=compress_factor,
        ).to(DEVICE)

        autoencoder_model_name = "hypertoken_2025-01-27T14:14:46.508086_token_len_16_hypertoken_size64"

        h_encoder_model = load_model(
            h_encoder_model,
            "saved_models",
            autoencoder_model_name,
            encoder_only=True,
        )

        h_model = load_model(
            h_decoder_model,
            "saved_models",
            autoencoder_model_name,
            decoder_only=True,
        )

        dataset = HyperTokenTinyShakespeareDataset(
            token_len=token_len,
            segments=data_segments,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        vocab_size = len(dataset.char2idx)

        gpt_model = JPT1(
            vocab_size=vocab_size,
            seq_len=seq_len,
            token_len=token_len,
            embed_dim=jpt_embed_dim,
            num_heads=head_size,
            num_layers=n_layers,
            dropout=dropout,
            hypertoken_size=hypertoken_size,
            expand_method=expand_method,
        ).to(DEVICE)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        val_dataset = HyperTokenTinyShakespeareDataset(
            token_len=token_len,
            segments=data_segments,
            seq_len=seq_len,
            batch_size=batch_size,
            type="validation",
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
        )

        # only if not debugging
        if sys.gettrace() is None:  # No debugger attached
            print("Compiling models...")
            gpt_model = torch.compile(gpt_model)
            h_decoder_model = torch.compile(h_decoder_model)
            h_encoder_model = torch.compile(h_encoder_model)
            print("Models compiled!")

        verify_model_params(experiment)

        validate_hypertoken_models(h_encoder_model, h_decoder_model, dataset, token_len)

        # create wrapper function for train_model
        def train_model_lambda(wandb):
            model = train_model(
                wandb,
                gpt_model,
                h_encoder_model,
                h_decoder_model,
                dataloader,
                val_dataloader,
                experiment,
            )
            return model[0]

        project_name = "jpt1"
        exp_name = f"{project_name}-sl:{experiment['seq_len']}-tl:{experiment['token_len']}-hts:{experiment['hypertoken_size']}-e:{experiment['epochs']}-bs:{experiment['batch_size']}-lr:{experiment['lr']}-hs:{experiment['head_size']}-nl:{experiment['n_layers']}-ed:{experiment['jpt_embed_dim']}"

        run_experiment(project_name, train_model_lambda, exp_name, experiment)

        # generate_text(
        #     gptModel, h_decoder_model, "Hello, world!", 100, hypertoken_seq_len, dataset
        # )


print("Training Complete")
