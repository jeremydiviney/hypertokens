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


def decode_indices_to_text(char_sequence: TensorType["batch_size", "seq_len", "token_char_len"], idx2char: [int], pad_token_idx: int) -> np.ndarray:
    # Convert indices to numpy array once
    indices_array = char_sequence.cpu().numpy()
    # Create mask for non-pad tokens
    mask = indices_array == pad_token_idx
    # Create a lookup array with the character mapping
    char_lookup = np.array([idx2char[i] for i in range(len(idx2char))])
    # Use numpy vectorized lookup
    chars_array = char_lookup[indices_array]
    chars_array[mask] = ""
    return chars_array


# --------------------------------------------------
# --------------------------------------------------
# 1. Data Preparation
# --------------------------------------------------
class TinyShakespeareDataset(Dataset):
    def __init__(
        self,
        encode_last_n_length: int,
        segments: int,
        seq_len: int = 128,
        type: str = "train",
    ):
        # Load TinyShakespeare from Hugging Face
        dataset = load_dataset("tiny_shakespeare")

        train_text = dataset["train"]["text"][0]
        val_text = dataset["validation"]["text"][0]
        test_text = dataset["test"]["text"][0]

        all_text = train_text + val_text + test_text

        text = train_text if type == "train" else val_text if type == "validation" else test_text

        # Build vocabulary
        chars = sorted(list(set(all_text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

        # Add padding token
        self.pad_token = len(chars)
        self.char2idx["<PAD>"] = self.pad_token
        self.idx2char[self.pad_token] = "<PAD>"

        self.encode_last_n_length = encode_last_n_length

        # Convert text to indices
        self.data = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)
        self.seq_len = seq_len
        self.segments = segments
        self.type = type

    def __len__(self) -> int:

        if self.type == "train":
            return math.floor(len(self.data) / self.segments)
        else:
            return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        final_index = 0

        if self.type == "train":
            final_index = random.randint(0, len(self.data) - 1)
        else:
            final_index = idx

        # Calculate random sequence length between 1 and seq_len
        rand_len = random.randint(1, min(self.seq_len, len(self.data) - final_index))

        # Create padded sequence of max length
        x = torch.full((self.seq_len,), self.pad_token, dtype=torch.long)

        if final_index < 0:
            print("final_index < 0")

        if final_index + rand_len > len(self.data):
            print("final_index + rand_len > len(self.data)")

        sequence = self.data[final_index : final_index + rand_len]

        boundary_index = self.seq_len - self.encode_last_n_length

        # Place sequence in padded tensor
        if len(sequence) > self.encode_last_n_length:
            overflow = len(sequence) - self.encode_last_n_length
            x[boundary_index - overflow : boundary_index + len(sequence) - overflow] = sequence
        else:
            x[boundary_index : boundary_index + len(sequence)] = sequence

        y = x.clone()
        return x, y


class HyperTokenTinyShakespeareDataset(Dataset):
    def __init__(
        self,
        encoder: nn.Module,
        segments: int,
        hypertoken_seq_len: int,
        seq_len: int,
        batch_size: int,
        type: str = "train",
    ):

        encoder.eval()

        self.encoder = encoder

        self.encoder = encoder
        self.hypertoken_seq_len = hypertoken_seq_len  # hypertoken sequence length
        self.seq_len = seq_len  # JPT1 sequence length
        self.segments = segments
        self.type = type
        self.batch_size = batch_size

        # Load TinyShakespeare from Hugging Face
        dataset = load_dataset("tiny_shakespeare")

        train_text = dataset["train"]["text"][0]
        val_text = dataset["validation"]["text"][0]
        test_text = dataset["test"]["text"][0]

        all_text = train_text + val_text + test_text

        text = train_text if type == "train" else val_text if type == "validation" else test_text

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
        # Convert text to indices
        self.data = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)

    def __len__(self) -> int:

        if self.type == "train":
            return math.floor(len(self.data) / self.segments)
        else:
            return len(self.data)

    def get_batch_item(self, idx: int, chunk_count: int = None, chunk_size: int = None) -> tuple[torch.Tensor, torch.Tensor]:

        if chunk_count is None:
            chunk_count = self.seq_len + 1

        if chunk_size is None:
            chunk_size = self.hypertoken_seq_len

        if self.type == "train":
            idx = random.randint(0, len(self.data) - 1)

        # Get sequence that will fit jpt_seq_len hypertokens
        total_chars_needed = chunk_size * (chunk_count)

        # Ensure we have enough characters
        if total_chars_needed >= len(self.data) - idx:
            idx = len(self.data) - total_chars_needed

        # Get the full sequence
        char_sequence = self.data[idx : idx + total_chars_needed]

        # Pad if needed
        if len(char_sequence) < total_chars_needed:
            padding = torch.full(
                (total_chars_needed - len(char_sequence),),
                self.pad_token,
            )
            char_sequence = torch.cat([padding, char_sequence])

        # Reshape to get hypertoken chunks
        char_chunks = char_sequence.view(chunk_count, chunk_size)

        if char_chunks.shape[0] != chunk_count:
            print("char_chunks.device.shape[0] != chunk_count")
            raise Exception("char_chunks.device.shape[0] != chunk_count")

        if char_chunks.shape[1] != chunk_size:
            print("char_chunks.device.shape[1] != chunk_size")
            raise Exception("char_chunks.device.shape[1] != chunk_size")

        return char_chunks

    def encode_to_hypertokens_from_text(self, text: str) -> torch.Tensor:
        device = next(self.encoder.parameters()).device

        char_sequence_indexes = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long).to(device)

        return self.encode_to_hypertokens(char_sequence_indexes.reshape(1, -1, 1))

    def encode_characters_to_indexes(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)

    def encode_to_hypertokens(self, char_sequence: torch.Tensor) -> List[torch.Tensor]:
        # Calculate batch sizes to process
        device = next(self.encoder.parameters()).device

        cur_seq_len = char_sequence.size(1)

        # Take up to sequence length and pad if needed
        if char_sequence.size(1) < cur_seq_len:
            padding = torch.full(
                (
                    char_sequence.size(0),
                    cur_seq_len - char_sequence.size(1),
                    char_sequence.size(1),
                ),
                self.pad_token,
                device=device,
            )
            char_sequence = torch.cat([padding, char_sequence], dim=1)
        else:
            char_sequence = char_sequence[:, -cur_seq_len:]

        chunks_per_item = cur_seq_len
        items_per_batch = 4096 // chunks_per_item

        actual_batch_size = char_sequence.size(0)
        remaining = actual_batch_size

        batch_sizes = []

        while remaining > 0:
            current_batch = min(items_per_batch, remaining)
            batch_sizes.append(current_batch)
            remaining -= current_batch

        # Process each batch size

        encoded_list = []
        start_idx = 0
        for size in batch_sizes:
            end_idx = start_idx + size
            batch_slice = slice(start_idx, end_idx)

            current_input = char_sequence[batch_slice]

            current_input_flat = current_input.reshape(-1, self.hypertoken_seq_len)
            # Encode current batch
            encoded = self.encoder(current_input_flat.to(device))

            encoded = encoded.reshape(current_input.size(0), cur_seq_len, -1)
            encoded_list.extend(encoded)

            start_idx = end_idx

        return encoded_list

    def __getitems__(self, batch_indices: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:

        pre_batch_items = [self.get_batch_item(idx) for idx in batch_indices]

        all_chunks = torch.stack(pre_batch_items)

        batch_items: List[Tuple[torch.Tensor, torch.Tensor]] = []

        target_chars = all_chunks[:, 1:]
        input_chars = all_chunks[:, :-1]

        # Zip encoded inputs with targets and extend batch_items
        batch_items = list(zip(input_chars, target_chars))

        return batch_items
