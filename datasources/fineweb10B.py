import random
import math
import re
import os
import json
import pickle
from bisect import bisect_left

from multiprocessing import Pool

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


def chunked(iterator, chunk_size):
    """Yield lists of up to chunk_size items from the iterator."""
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def build_selection_table(data, tokenizer):
    vocab_size = len(tokenizer.get_vocab())
    cache_filename = f"meta_cache/jpt1/selection_table_vocab_{vocab_size}.pkl"

    if os.path.exists(cache_filename):
        with open(cache_filename, "rb") as f:
            lookup_table = pickle.load(f)
        return lookup_table

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_filename), exist_ok=True)

    lookup_table = []
    total = 0
    cnt = 0

    for d in data:
        # tokens = tokenizer.encode(d["text"]).tokens
        text = d["text"]
        total += len(text) / 5  # rought token count
        lookup_table.append(total)
        cnt += 1
        if cnt % 10000 == 0:
            print(f"build_selection_table: Processed {cnt} items")

    with open(cache_filename, "wb") as f:
        pickle.dump(lookup_table, f)

    return lookup_table


def get_or_train_tokenizer(text_corpus: str | Iterable[str], vocab_size: int, tokenizer_path: str):
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

        if isinstance(text_corpus, str):
            corpus = re.split(TOKEN_CORPUS_PATTERN, text_corpus)
            corpus = [w for w in corpus if w]  # Remove empty strings
        else:
            corpus = text_corpus

        # Create a new BPE tokenizer with an unknown token
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        # Use ByteLevel pre-tokenizer so that spaces and even cross-boundary merges can be learned
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        # Define special tokens; these will be added to the vocabulary
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=2)

        # Train the tokenizer on the corpus iterator
        tokenizer.train_from_iterator(corpus, trainer=trainer)

        # Set a decoder to handle the byte-level encoding (this will reassemble tokens correctly)
        tokenizer.decoder = decoders.ByteLevel()
        # Optionally, add a post-processor if you need specific handling (for example, GPT-2 style)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Save the trained tokenizer to file for future use
        tokenizer.save(tokenizer_path)
        print(f"Trained and saved tokenizer to {tokenizer_path}")
        os._exit(0)
    return tokenizer


def load_hf_dataset():
    cache_path = "data_cache/fineweb-10BT"
    try:
        # Try to load from cache first
        dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", cache_dir=cache_path)
    except Exception as e:
        print(f"Warning: Could not load dataset from Hugging Face: {e}")
        # Fallback to local backup if it exists

    return dataset


class Fineweb10BDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        codebook: TokenCodebook,
        data_stride: int,
        hf_dataset: Dataset,
        tokenizer: Tokenizer | None,
        type: str = "train",
    ):
        # Load TinyShakespeare from Hugging Face
        self.hf_dataset = hf_dataset
        self.seq_len = seq_len
        self.codebook = codebook
        self.data_stride = data_stride
        self.tokenizer = tokenizer
        self.train_ratio = 1000
        self.type = type

        self.selection_table = build_selection_table(self.hf_dataset["train"], self.tokenizer)
        self.selection_table_train = self.selection_table[:: self.train_ratio]

        self.token_count = math.floor(self.selection_table[-1])
        if self.type == "validation":
            self.token_count = self.token_count // self.train_ratio

    def get_data_chunk(self, idx: int):

        adjusted_idx_value = idx * self.data_stride

        if self.type == "validation":
            adjusted_idx_value *= self.train_ratio

        if self.type == "train":
            data_row_idx = bisect_left(self.selection_table, adjusted_idx_value)
            if data_row_idx % self.train_ratio == 0:
                data_row_idx = data_row_idx + 1 if data_row_idx < len(self.selection_table) else data_row_idx - 1

        else:
            data_row_idx = bisect_left(self.selection_table_train, adjusted_idx_value)

        # Get the full text from the dataset
        full_text = self.hf_dataset["train"][data_row_idx]["text"]

        full_text_tokens = self.tokenizer.encode(full_text).tokens

        data_row_chunks = len(full_text_tokens) // self.data_stride

        chuck_selection_idx = random.randint(0, data_row_chunks - 1)

        # Calculate chunk size based on sequence length
        chunk_size = self.seq_len + 1  # +1 for the target token

        # Calculate start and end positions for the chunk
        start_pos = chuck_selection_idx * chunk_size
        end_pos = start_pos + chunk_size

        # Extract the chunk, handling potential end of text
        text_chunk = full_text_tokens[start_pos:end_pos]
        return text_chunk

    def __len__(self) -> int:

        if self.type == "train":
            return math.floor(self.selection_table[-1] / self.data_stride)
        else:
            return math.floor(self.selection_table[-1] / (self.data_stride * self.train_ratio))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        token_sequence = self.get_data_chunk(idx)

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
