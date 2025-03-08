import random
import math
import re
import os
import json

from typing import List, Tuple


import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from torch.utils.data.dataset import Dataset

from datasets import load_dataset

from models.jpt1_quantizer import TokenCodebook

TOKEN_PATTERN = re.compile(r"(\s+|\w+|[^\w\s])")


def process_character_sequence(dataset: Dataset, sequence: torch.Tensor) -> str:
    # try and take whole words
    # test test tes
    sequence_str = "".join([dataset.idx2char[idx.item()] for idx in sequence])
    words = re.split(r"(\s)", sequence_str)
    words = [w for w in words if w]  # Remove empty strings

    # Only process if we have at least 3 words (to remove first and last)
    # strip partial words in front and back, also randomly leave the first or last whitespace
    if len(words) >= 3:
        words = words[1:-1]

        if len(words) >= 1 and words[0] in [" ", "\n"] and random.random() < 0.5:
            words = words[1:]

        if len(words) >= 1 and words[-1] in [" ", "\n"] and random.random() < 0.5:
            words = words[:-1]

    if len(words) > 0:
        sequence_str = "".join(words)


def decode_indices_to_text(
    char_sequence: TensorType["batch_size", "seq_len", "token_char_len"], idx2char: [int], pad_token_idx: int
) -> np.ndarray:
    # Convert indices to numpy array once

    indices_array = char_sequence if isinstance(char_sequence, np.ndarray) else char_sequence.cpu().numpy()
    # Create mask for non-pad tokens
    mask = indices_array == pad_token_idx
    # Create a lookup array with the character mapping
    char_lookup = np.array([idx2char[i] for i in range(len(idx2char))])
    # Use numpy vectorized lookup
    chars_array = char_lookup[indices_array]
    chars_array[mask] = ""
    return chars_array


def tokenize(text: str, token_len: int, silent: bool = True) -> List[str]:
    # Split text while preserving whitespace groups

    pre_tokens = re.findall(TOKEN_PATTERN, text)

    # Split long tokens into chunks of maximum size token_len
    tokens = []
    for token in pre_tokens:
        if len(token) <= token_len:
            tokens.append(token)
        else:
            # Split into chunks of token_len characters
            for i in range(0, len(token), token_len):
                tokens.append(token[i : i + token_len])

    # Calculate statistics

    if not silent:
        # Convert to numpy array for analysis
        token_lengths = np.array([len(token) for token in tokens])
        unique_tokens = np.unique(tokens)

        stats = {
            "min_len": np.min(token_lengths),
            "max_len": np.max(token_lengths),
            "mean_len": np.mean(token_lengths),
            "median_len": np.median(token_lengths),
            "total_tokens": len(tokens),
            "unique_tokens": len(unique_tokens),
        }

        # Create histogram data
        length_counts = np.bincount(token_lengths)

        print("\nToken Statistics:")
        print(f"Min length: {stats['min_len']}")
        print(f"Max length: {stats['max_len']}")
        print(f"Mean length: {stats['mean_len']:.2f}")
        print(f"Median length: {stats['median_len']:.1f}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"Unique tokens: {stats['unique_tokens']:,}")

        print("\nLength Distribution:")

        for length, count in enumerate(length_counts):
            if count > 0:
                print(f"{length}: {'#' * (count // 1000)} ({count:,})")

        # Find and print longest tokens
        max_len = stats["max_len"]
        longest_tokens = [t for t in tokens if len(t) == max_len]
        unique_longest = np.unique(longest_tokens)[:5]  # Get up to 5 unique examples
        print(f"\nSample of longest tokens (length {max_len}):")

        for token in unique_longest:
            print(f"'{token}'")

    return tokens


def finalize_tokens(text_tokens: List[str], char2idx: dict, token_len: int, pad_token: int, eot_token: int) -> np.ndarray:
    tokens = []

    try:

        for token in text_tokens:
            # Create array of character indices with an extra space for EOT
            char_indices = [char2idx[ch] for ch in token]
            # Append EOT token
            char_indices.append(eot_token)
            char_indices = char_indices[:token_len]
            tokens.append(np.array(char_indices, dtype=np.int64))

        padded_tokens = [np.pad(token, (0, token_len - len(token)), constant_values=pad_token) for token in tokens]
        tokens = np.stack(padded_tokens)

        return tokens

    except Exception as e:
        print(f"Error finalizing tokens: {e}")
        raise e


def load_cached_dataset():
    cache_path = "data_cache/tiny_shakespeare"
    try:
        # Try to load from cache first
        dataset = load_dataset("tiny_shakespeare", cache_dir=cache_path)
    except Exception as e:
        print(f"Warning: Could not load dataset from Hugging Face: {e}")
        # Fallback to local backup if it exists
        try:
            dataset = load_dataset(
                "json",
                data_files={
                    "train": f"{cache_path}/train.json",
                    "validation": f"{cache_path}/validation.json",
                    "test": f"{cache_path}/test.json",
                },
            )
        except Exception as backup_e:
            print(f"Error: Could not load from backup: {backup_e}")
            raise RuntimeError("Failed to load dataset from both HF and local cache")

    # Save a local backup after successful HF download
    try:
        os.makedirs(cache_path, exist_ok=True)
        for split in ["train", "validation", "test"]:
            with open(f"{cache_path}/{split}.json", "w") as f:
                json.dump({"text": dataset[split]["text"]}, f)
    except Exception as e:
        print(f"Warning: Could not save backup: {e}")

    return dataset


# --------------------------------------------------
# --------------------------------------------------
# 1. Data Preparation
# --------------------------------------------------
class TinyShakespeareDataset(Dataset):
    def __init__(
        self,
        token_len: int,
        type: str = "train",
    ):
        # Load TinyShakespeare with caching
        dataset = load_cached_dataset()

        train_text = dataset["train"]["text"][0]
        val_text = dataset["validation"]["text"][0]
        test_text = dataset["test"]["text"][0]

        all_text = train_text + val_text + test_text
        val_text = val_text + test_text

        if type == "train":
            text = train_text
        elif type == "all":
            text = all_text
        else:
            text = val_text

        self.token_len = token_len

        # Build vocabulary
        chars = sorted(list(set(all_text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

        # Add padding token
        self.pad_token = len(chars)
        self.char2idx["[PAD]"] = self.pad_token
        self.idx2char[self.pad_token] = "[PAD]"

        # Add end of text token
        self.eot_token = len(chars) + 1
        self.char2idx["<EOT>"] = self.eot_token
        self.idx2char[self.eot_token] = "<EOT>"

        # Create numpy versions of the mappings
        self.idx2char_np = np.array([self.idx2char[i] for i in range(len(self.idx2char))])

        self.text_tokens = tokenize(text, self.token_len, False)

        self.tokens = finalize_tokens(self.text_tokens, self.char2idx, self.token_len, self.pad_token, self.eot_token)

        self.type = type

    def __len__(self) -> int:

        if self.type == "train":
            return math.floor(len(self.tokens))
        else:
            return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        final_index = 0

        if self.type == "train":
            final_index = random.randint(0, len(self.tokens) - 1)
        else:
            final_index = idx

        token = self.tokens[final_index]

        y = token
        x = token
        return x, y


class HyperTokenTinyShakespeareDataset(Dataset):
    def __init__(
        self,
        token_len: int,
        seq_len: int,
        batch_size: int,
        type: str = "train",
    ):
        self.token_len = token_len  # hypertoken sequence length
        self.seq_len = seq_len  # JPT1 sequence length
        self.type = type
        self.batch_size = batch_size

        self.validation_stride = self.seq_len // 8

        self.data_tile_length = 16

        # Load TinyShakespeare from Hugging Face
        dataset = load_cached_dataset()

        train_text = dataset["train"]["text"][0]
        val_text = dataset["validation"]["text"][0]
        test_text = dataset["test"]["text"][0]

        all_text = train_text + val_text + test_text

        val_text = val_text + test_text

        text = train_text if type == "train" else val_text

        # Build vocabulary
        chars = sorted(list(set(all_text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

        # Add padding token
        self.pad_token = len(chars)
        self.char2idx["[PAD]"] = self.pad_token
        self.idx2char[self.pad_token] = "[PAD]"

        # Add end of text token
        self.eot_token = len(chars) + 1
        self.char2idx["<EOT>"] = self.eot_token
        self.idx2char[self.eot_token] = "<EOT>"

        self.batch_item_buffer_x = []
        self.batch_item_buffer_y = []

        self.text_tokens = tokenize(text, self.token_len, False)

        self.tokens = finalize_tokens(self.text_tokens, self.char2idx, self.token_len, self.pad_token, self.eot_token)

    def __len__(self) -> int:

        if self.type == "train":
            return math.floor(len(self.tokens))
        else:
            return len(self.tokens) // self.validation_stride

    def get_batch_item(self, idx: int, seq_len: int = None) -> np.ndarray:
        """Get input-target sequence pair from tokens array without padding."""
        # Get sequence length parameters from instance

        final_seq_len = None

        if seq_len is None:
            final_seq_len = self.seq_len + 1
        else:
            final_seq_len = seq_len

        if self.type == "train":
            # Random sampling with available context
            start_idx = random.randint(0, len(self.tokens) - 1)
            end_idx = min(start_idx + final_seq_len, len(self.tokens))
        else:
            # Sequential validation access
            start_idx = idx * self.validation_stride
            end_idx = min(start_idx + final_seq_len, len(self.tokens))

        # Get sequence of up to seq_len+1 tokens (or whatever remains)
        token_sequence = self.tokens[start_idx:end_idx]

        padding_needed = final_seq_len - len(token_sequence)

        if padding_needed > 0:
            # Create padding array filled with pad_token
            padding = np.full((padding_needed, self.token_len), self.pad_token, dtype=np.int64)
            # Concatenate the token sequence with padding
            token_sequence = np.concatenate([token_sequence, padding], axis=0)

        return token_sequence

    def encode_to_hypertokens_from_text(self, encoder: nn.Module, text: str, max_seq_len: int) -> torch.Tensor:
        device = next(encoder.parameters()).device

        try:
            # Convert text to character indices
            text_tokens = tokenize(text, self.token_len)

            text_tokens = text_tokens[-max_seq_len:]

            tokens = finalize_tokens(text_tokens, self.char2idx, self.token_len, self.pad_token, self.eot_token)
            tokens = torch.tensor(tokens).to(device)

            return self.encode_to_hypertokens(encoder, tokens.reshape(1, -1, self.token_len), self.token_len)
        except Exception as e:
            print(f"Error encoding: {e}")
            print(f"Text: {text}")
            print(f"Max seq len: {max_seq_len}")
            raise e

    def encode_characters_to_indexes(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)

    def encode_to_hypertokens(
        self,
        encoder: nn.Module,
        sequence_batch: torch.Tensor,
        token_len,
    ) -> List[torch.Tensor]:
        # Calculate batch sizes to process
        device = next(encoder.parameters()).device

        actual_batch_size = sequence_batch.size(0)
        cur_seq_len = sequence_batch.size(1)

        batches_per_iter = 4096 // cur_seq_len

        remaining = actual_batch_size

        batch_sizes = []

        while remaining > 0:
            current_batch = min(batches_per_iter, remaining)
            batch_sizes.append(current_batch)
            remaining -= current_batch

        # Process each batch size

        encoded_list = []
        start_idx = 0
        for size in batch_sizes:
            end_idx = start_idx + size

            current_input = sequence_batch[start_idx:end_idx, :, :].view(-1, token_len)
            cur_batch_slice_size = end_idx - start_idx

            # Encode current batch
            encoded = encoder(current_input.to(device))

            encoded = encoded.view(cur_batch_slice_size, cur_seq_len, -1)
            encoded_list.extend(encoded)

            start_idx = end_idx

        return encoded_list

    def __getitems__(self, batch_indices: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:

        pre_batch_items = [self.get_batch_item(idx) for idx in batch_indices]

        all_chunks = np.stack(pre_batch_items)

        batch_items: List[Tuple[torch.Tensor, torch.Tensor]] = []

        target_chars = all_chunks[:, 1:]
        input_chars = all_chunks[:, :-1]

        # Zip encoded inputs with targets and extend batch_items
        batch_items = list(zip(input_chars, target_chars))

        return batch_items


class TinyShakespeareDatasetQuantized(Dataset):
    def __init__(
        self,
        token_len: int,
        seq_len: int,
        codebook: TokenCodebook,
        data_stride: int,
        type: str = "train",
    ):
        # Load TinyShakespeare from Hugging Face
        dataset = load_cached_dataset()
        self.seq_len = seq_len
        self.codebook = codebook
        self.data_stride = data_stride
        train_text = dataset["train"]["text"][0]
        val_text = dataset["validation"]["text"][0]
        test_text = dataset["test"]["text"][0]

        all_text = train_text + val_text + test_text
        val_text = val_text + test_text

        if type == "train":
            text = train_text
        elif type == "all":
            text = all_text
        else:
            text = val_text

        self.token_len = token_len

        self.text_tokens = np.array(tokenize(text, self.token_len, False))

        self.type = type

    def __len__(self) -> int:

        if self.type == "train":
            return math.floor(len(self.codebook.tokens))
        else:
            return len(self.text_tokens) // self.data_stride

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        final_index = 0

        if self.type == "train":
            final_index = random.randint(0, len(self.text_tokens) - 1)
        else:
            final_index = idx * self.data_stride

        # Get sequence of tokens up to seq_len
        token_sequence = self.text_tokens[final_index : (final_index + self.seq_len + 1)]

        # Calculate padding needed
        padding_needed = (self.seq_len + 1) - len(token_sequence)

        if padding_needed > 0:
            # Create padding array filled with pad_token
            padding = np.full((padding_needed), "[PAD]")
            # Concatenate the token sequence with padding
            token_sequence = np.concatenate([padding, token_sequence], axis=0)

        x = token_sequence[:-1]
        y = token_sequence[1:]

        x = np.array(self.codebook.get_token_indices(x))
        y = np.array(self.codebook.get_token_indices(y))

        return x, y
