import random
import math
import re

from typing import List, Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from torch.utils.data.dataset import Dataset

from datasets import load_dataset


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


def tokenize(text: str) -> List[str]:
    # Split text while preserving whitespace groups
    pattern = r"(\s+|\w+|[^\w\s])"
    tokens = re.findall(pattern, text)

    # Convert to numpy array for analysis
    token_lengths = np.array([len(token) for token in tokens])
    unique_tokens = np.unique(tokens)

    # Calculate statistics
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


# --------------------------------------------------
# --------------------------------------------------
# 1. Data Preparation
# --------------------------------------------------
class TinyShakespeareDataset(Dataset):
    def __init__(
        self,
        segments: int,
        token_len: int,
        type: str = "train",
    ):
        # Load TinyShakespeare from Hugging Face
        dataset = load_dataset("tiny_shakespeare")

        train_text = dataset["train"]["text"][0]
        val_text = dataset["validation"]["text"][0]
        test_text = dataset["test"]["text"][0]

        all_text = train_text + val_text + test_text
        val_text = val_text + test_text

        text = train_text if type == "train" else val_text

        self.token_len = token_len

        # Build vocabulary
        chars = sorted(list(set(all_text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

        # Add padding token
        self.pad_token = len(chars)
        self.char2idx["<PAD>"] = self.pad_token
        self.idx2char[self.pad_token] = "<PAD>"

        # Create numpy versions of the mappings
        self.idx2char_np = np.array([self.idx2char[i] for i in range(len(self.idx2char))])

        self.text_tokens = tokenize(text)

        self.tokens = []

        for token in self.text_tokens:
            self.tokens.append(np.array([self.char2idx[ch] for ch in token], dtype=np.int64))

        padded_tokens = [
            np.pad(token, (0, self.token_len - len(token)), constant_values=self.pad_token) for token in self.tokens
        ]
        self.tokens = np.stack(padded_tokens)

        self.segments = segments
        self.type = type

    def __len__(self) -> int:

        if self.type == "train":
            return math.floor(len(self.tokens) / self.segments)
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
        segments: int,
        token_len: int,
        seq_len: int,
        batch_size: int,
        type: str = "train",
    ):
        self.token_len = token_len  # hypertoken sequence length
        self.seq_len = seq_len  # JPT1 sequence length
        self.segments = segments
        self.type = type
        self.batch_size = batch_size

        self.validation_stride = self.seq_len // 8

        self.data_tile_length = 16

        # Load TinyShakespeare from Hugging Face
        dataset = load_dataset("tiny_shakespeare")

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
        self.char2idx["<PAD>"] = self.pad_token
        self.idx2char[self.pad_token] = "<PAD>"

        self.batch_item_buffer_x = []
        self.batch_item_buffer_y = []

        self.text_tokens = tokenize(text)

        self.tokens = []

        for token in self.text_tokens:
            self.tokens.append(np.array([self.char2idx[ch] for ch in token], dtype=np.int64))

        padded_tokens = [
            np.pad(token, (0, self.token_len - len(token)), constant_values=self.pad_token) for token in self.tokens
        ]
        self.tokens = np.stack(padded_tokens)

    def __len__(self) -> int:

        if self.type == "train":
            return math.floor(len(self.tokens) / self.segments)
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

    def encode_to_hypertokens_from_text(self, encoder: nn.Module, text: str) -> torch.Tensor:
        device = next(encoder.parameters()).device

        # Convert text to character indices
        char_sequence_indexes = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)

        # Calculate padding needed to make length a multiple of hypertoken_seq_len
        remainder = len(char_sequence_indexes) % self.token_len

        # Create left padding tensor
        padding = torch.full((remainder,), self.pad_token, dtype=torch.long)

        # Concatenate padding with character sequence
        padded_sequence = torch.cat([padding, char_sequence_indexes]).to(device)

        return self.encode_to_hypertokens(encoder, padded_sequence.reshape(1, -1, self.seq_len), self.token_len)

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
