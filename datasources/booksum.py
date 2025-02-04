import random
import math
import re
import os
import json

from typing import List, Tuple, Iterable
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from torch.utils.data.dataset import Dataset

from datasets import load_dataset

from models.jpt1_quantizer import TokenCodebook

TOKEN_PATTERN = re.compile(r"(\s+|\w+|[^\w\s])")

TOKEN_CORPUS_PATTERN = re.compile(r"(\n+|\w+|[^\w\s])")


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


# def tokenize(text: str, token_len: int, silent: bool = True) -> List[str]:
#     # Split text while preserving whitespace groups

#     pre_tokens = re.findall(TOKEN_PATTERN, text)

#     # Split long tokens into chunks of maximum size token_len
#     tokens = []
#     for token in pre_tokens:
#         if len(token) <= token_len:
#             tokens.append(token)
#         else:
#             # Split into chunks of token_len characters
#             for i in range(0, len(token), token_len):
#                 tokens.append(token[i : i + token_len])

#     # Calculate statistics

#     if not silent:
#         # Convert to numpy array for analysis
#         token_lengths = np.array([len(token) for token in tokens])
#         unique_tokens = np.unique(tokens)

#         stats = {
#             "min_len": np.min(token_lengths),
#             "max_len": np.max(token_lengths),
#             "mean_len": np.mean(token_lengths),
#             "median_len": np.median(token_lengths),
#             "total_tokens": len(tokens),
#             "unique_tokens": len(unique_tokens),
#         }

#         # Create histogram data
#         length_counts = np.bincount(token_lengths)

#         print("\nToken Statistics:")
#         print(f"Min length: {stats['min_len']}")
#         print(f"Max length: {stats['max_len']}")
#         print(f"Mean length: {stats['mean_len']:.2f}")
#         print(f"Median length: {stats['median_len']:.1f}")
#         print(f"Total tokens: {stats['total_tokens']:,}")
#         print(f"Unique tokens: {stats['unique_tokens']:,}")

#         print("\nLength Distribution:")

#         for length, count in enumerate(length_counts):
#             if count > 0:
#                 print(f"{length}: {'#' * (count // 1000)} ({count:,})")

#         # Find and print longest tokens
#         max_len = stats["max_len"]
#         longest_tokens = [t for t in tokens if len(t) == max_len]
#         unique_longest = np.unique(longest_tokens)[:5]  # Get up to 5 unique examples
#         print(f"\nSample of longest tokens (length {max_len}):")

#         for token in unique_longest:
#             print(f"'{token}'")

#     return tokens


def get_or_train_tokenizer(text_corpus: str, vocab_size: int, tokenizer_path: str):
    """
    Train a BPE tokenizer on the given corpus or load it from disk if already saved.

    Args:
        corpus (iterable of str): Text corpus for training.
        vocab_size (int): Desired vocabulary size.
        tokenizer_path (str): File path for saving/loading the tokenizer.

    Returns:
        Tokenizer: A Hugging Face Tokenizers object.
    """
    if os.path.exists(tokenizer_path):
        # Load the tokenizer if it exists
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
    else:

        corpus = re.split(TOKEN_CORPUS_PATTERN, text_corpus)
        corpus = [w for w in corpus if w]  # Remove empty strings

        # Create a new BPE tokenizer with an unknown token
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        # Use ByteLevel pre-tokenizer so that spaces and even cross-boundary merges can be learned
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        # Define special tokens; these will be added to the vocabulary
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=1)

        # Train the tokenizer on the corpus iterator
        tokenizer.train_from_iterator(corpus, trainer=trainer)

        # Set a decoder to handle the byte-level encoding (this will reassemble tokens correctly)
        tokenizer.decoder = decoders.ByteLevel()
        # Optionally, add a post-processor if you need specific handling (for example, GPT-2 style)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Save the trained tokenizer to file for future use
        tokenizer.save(tokenizer_path)
        print(f"Trained and saved tokenizer to {tokenizer_path}")
    return tokenizer


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
    cache_path = "data_cache/booksum-complete-cleaned"
    try:
        # Try to load from cache first
        dataset = load_dataset("ubaada/booksum-complete-cleaned", "books", cache_dir=cache_path)
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


class BooksumDataset(Dataset):
    def __init__(
        self,
        token_len: int,
        seq_len: int,
        codebook: TokenCodebook,
        data_stride: int,
        tokenizer: Tokenizer | None,
        type: str = "train",
    ):
        # Load TinyShakespeare from Hugging Face
        dataset = load_cached_dataset()
        self.seq_len = seq_len
        self.codebook = codebook
        self.data_stride = data_stride
        train_text = "\n".join([dataset["train"][i]["text"] for i in range(75)])
        val_text = "\n".join([dataset["validation"][i]["text"] for i in range(3)])
        test_text = "\n".join([dataset["test"][i]["text"] for i in range(3)])

        all_text = train_text + val_text + test_text
        val_text = val_text + test_text

        if type == "train":
            text = train_text
        elif type == "all":
            text = all_text
        else:
            text = val_text

        self.all_text = all_text

        self.token_len = token_len

        if tokenizer:
            self.text_tokens = tokenizer.encode(text)
            self.text_tokens = self.text_tokens.tokens
        else:
            self.text_tokens = None

        self.type = type

    def __len__(self) -> int:

        if self.type == "train":
            return len(self.text_tokens) // self.data_stride
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
