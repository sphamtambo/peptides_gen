#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the dataset class and the dataloader for the text dataset.
"""

import os
import random

import numpy as np
import torch

from argument_parser import get_args

args = get_args()

torch.manual_seed(args.random_seed)
torch.backends.cudnn.deterministic = True


def reset_random_seeds(seed):
    """
    reset random seeds for reproducibility
    :param seed: {int} random seed
    Returns:
        -None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


reset_random_seeds(args.random_seed)


def load_text(file):
    """
    load text from a file
    :param file: {str} file of the text file to be read
    Returns:
        -text : {str} text
    """

    with open(file, "r", encoding="utf8") as f:
        text = f.read()
        # uppercase all text
        text = text.upper()
    return text


ROOT_DIR = f"data/{args.dataset}"
text = load_text(ROOT_DIR)

chars = sorted(list(set(text)))
vocab_size = len(chars)
vocab = "".join(chars)


def string_to_int(text):
    """
    encode a given text/char into integers (pytorch tensors)
    :param text: {str} text to be encoded
    Returns:
        -tensor: {torch.tensor} encoded text
    """
    chars = sorted(list(set(text)))
    stoi = {cha: i for i, cha in enumerate(chars)}
    encode = [stoi[cha] for cha in text]
    tensor = torch.tensor(encode).long()
    return tensor


split_idx = int(0.8 * len(text))
train_data = text[:split_idx]
val_data = text[split_idx:]


def text_chunks(text):
    """
    create text chunks consisting of [text length] number of character
    each. They will then be used to construct input and targert text, both
    with [text length] number of elements.
    param: text: {str} text to be chunked
    Returns:
        -text_chunks: {list} list of text length
    """
    block_size = args.block_size + 1
    encoded_text = string_to_int(text)
    text_chunks = [
        encoded_text[i : i + block_size]
        for i in range(len(encoded_text) - block_size + 1)
    ]
    random.shuffle(text_chunks)
    return text_chunks


train_chunks = text_chunks(train_data)
val_chunks = text_chunks(val_data)


class TextDataset(torch.utils.data.Dataset):
    """
    create a dataset of text
    Attributes: chunks: {list} list of text chunks
    Returns:
        -inpt: {torch.tensor} input text
        -target: {torch.tensor} target text
    """

    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunks = self.chunks[idx]
        inpt = chunks[:-1].long()
        target = chunks[1:].long()
        return inpt, target


train_dataset = TextDataset(train_chunks)
val_dataset = TextDataset(val_chunks)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)
