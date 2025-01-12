import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import random
import math
import re
from typing import TypedDict
import torch.nn as nn
from helpers.training import batch_tensor_to_text
import time


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

        text = (
            train_text
            if type == "train"
            else val_text if type == "validation" else test_text
        )

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
            x[boundary_index - overflow : boundary_index + len(sequence) - overflow] = (
                sequence
            )
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

        text = (
            train_text
            if type == "train"
            else val_text if type == "validation" else test_text
        )

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

    def get_batch_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.encoder.parameters()).device

        if self.type == "train":
            idx = random.randint(0, len(self.data) - 1)

        # Get sequence that will fit jpt_seq_len hypertokens
        total_chars_needed = self.hypertoken_seq_len * (self.seq_len + 1)

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
        char_chunks = char_sequence.view(self.seq_len + 1, self.hypertoken_seq_len)
        return char_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.encoder.parameters()).device

        if self.type == "train":
            idx = random.randint(0, len(self.data) - 1)

        pre_batch_items = []

        for i in range(self.batch_size):
            item = self.get_batch_item(idx)
            pre_batch_items.append(item)

        # Stack all items into a single tensor
        all_chunks = torch.stack(
            pre_batch_items
        )  # Shape: [batch_size, seq_len + 1, hypertoken_seq_len]

        # Calculate how many chunks we can process at once
        # Each item has seq_len chunks to encode
        chunks_per_item = self.seq_len
        items_per_batch = 2048 // chunks_per_item

        # Prepare input chunks (all except last sequence for each item)
        input_chunks = all_chunks[
            :, :-1
        ]  # Shape: [batch_size, seq_len, hypertoken_seq_len]
        target_chunks = all_chunks[:, -1]  # Shape: [batch_size, hypertoken_seq_len]

        # Process in smaller batches if needed
        encoded_chunks_list = []
        for i in range(0, self.batch_size, items_per_batch):
            batch_slice = slice(i, min(i + items_per_batch, self.batch_size))
            current_input = input_chunks[batch_slice].to(device)

            # Reshape to encode all sequences at once
            batch_size_current = current_input.size(0)
            current_input = current_input.view(-1, self.hypertoken_seq_len)

            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16
            ):
                encoded = self.encoder(current_input)
                # Reshape back to [batch_size, seq_len, encoder_output_dim]
                encoded = encoded.view(batch_size_current, self.seq_len, -1)
                encoded_chunks_list.append(encoded)

        # Combine all encoded chunks
        x = torch.cat(encoded_chunks_list, dim=0)
        y = target_chunks

        return x, y
