import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.utils as utils
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
import datetime
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"


def create_dictionaries(letter_list):
    """
    Create dictionaries for letter2index and index2letter transformations
    based on LETTER_LIST

    Args:
        letter_list: LETTER_LIST

    Return:
        letter2index: Dictionary mapping from letters to indices
        index2letter: Dictionary mapping from indices to letters
    """
    letter2index = dict()
    index2letter = dict()
    for i in range(len(letter_list)):
        index2letter[i] = letter_list[i]
        letter2index[letter_list[i]] = i
    return letter2index, index2letter


def transform_index_to_letter(y, i2l, strip_start_and_end=True):
    """
    Transforms numerical index input to string output by converting each index 
    to its corresponding letter from LETTER_LIST

    Args:
        batch_indices: List of indices from LETTER_LIST with the shape of (N, )
    
    Return:
        transcripts: List of converted string transcripts. This would be a list with a length of N
    """
    transcripts = []
    lengths, B = y.shape
    for i in range(B):
        target_str = "".join(i2l[int(x)] for x in y[:, i])
        target_str = target_str.split("<eos>")[0]

        if not strip_start_and_end:
            target_str = "<sos>" + target_str + "<eos>"
        transcripts.append(target_str)
    return transcripts


class LibriSamples(torch.utils.data.Dataset):
    """
    Custom Dataset meant to handle a modified version of the LibriSpeech Dataset
    """

    def __init__(self, data_path, letter2index, partition="train", shuffle=True):
        self.x_dir = data_path + "/" + partition + "/mfcc"
        self.y_dir = data_path + "/" + partition + "/transcript"
        self.x_files = os.listdir(data_path + "/" + partition + "/mfcc")
        self.y_files = os.listdir(data_path + "/" + partition + "/transcript")
        self.files = [x for x in zip(self.x_files, self.y_files)]
        if shuffle:
            random.shuffle(self.files)
        self.letter2index = letter2index

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, ind):
        xdir, ydir = self.files[ind]
        x = torch.tensor(np.load(self.x_dir + "/" + xdir))
        y_s = np.load(self.y_dir + "/" + ydir)
        y_s = y_s[1:]
        y = torch.tensor([self.letter2index[x] for x in y_s])

        return x, y

    def collate_fn(self, batch):
        x_batch = [x for x, y in batch]
        y_batch = [y for x, y in batch]
        batch_x_pad = pad_sequence(x_batch, batch_first=True)
        lengths_x = [len(x) for x in x_batch]
        batch_y_pad = pad_sequence(y_batch, batch_first=True)
        lengths_y = [len(y) for y in y_batch]
        return (
            batch_x_pad,
            batch_y_pad,
            torch.tensor(lengths_x),
            torch.tensor(lengths_y),
        )


class LibriSamplesTest(torch.utils.data.Dataset):
    def __init__(self, data_path, test_order):
        with open(data_path + "/test/" + test_order, newline="") as f:
            reader = csv.reader(f)
            test_order_list = list(reader)
        self.X = [
            torch.tensor(np.load(data_path + "/test/mfcc/" + X_path[0]))
            for X_path in test_order_list[1:]
        ]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        return self.X[ind]

    def collate_fn(self, batch):
        batch_x = [x for x in batch]
        batch_x_pad = pad_sequence(batch_x, batch_first=True)
        lengths_x = [len(x) for x in batch_x]

        return batch_x_pad, torch.tensor(lengths_x)


def test_dataloaders(root, LETTER_LIST):
    bs = 64
    train_loader, val_loader, test_loader = get_dataloader(root, batch_size=bs)
    l2i, i2l = create_dictionaries(LETTER_LIST)
    train_data = LibriSamples(root, l2i, "train")
    val_data = LibriSamples(root, l2i, "dev")
    test_data = LibriSamplesTest(root, "test_order.csv")
    print("Batch size: ", bs)
    print(
        "Train dataset samples = {}, batches = {}".format(
            train_data.__len__(), len(train_loader)
        )
    )
    print(
        "Val dataset samples = {}, batches = {}".format(
            val_data.__len__(), len(val_loader)
        )
    )
    print(
        "Test dataset samples = {}, batches = {}".format(
            test_data.__len__(), len(test_loader)
        )
    )


def get_dataloader(root, batch_size=64, num_workers=4):
    """
    Returns dataloaders used for model training
    
    Args:
        root = path to the dataset
        batch_size = batch size of the data-loader
        num_workers = number of CPU workers used in loading data
    
    Returns:
        train_loader = dataloader for training data
        val_loader = dataloader for validation data
        test_loader = dataloader for test data
    """
    LETTER_LIST = [
        "<sos>",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "'",
        " ",
        "<eos>",
    ]
    l2i, i2l = create_dictionaries(LETTER_LIST)
    train_data = LibriSamples(root, l2i, "train")
    val_data = LibriSamples(root, l2i, "dev")
    test_data = LibriSamplesTest(root, "test_order.csv")
    train_loader = DataLoader(
        train_data,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=train_data.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=val_data.collate_fn,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_data,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=test_data.collate_fn,
    )
    return train_loader, val_loader, test_loader


class LibriSamplesSimple:
    """
    Custom Dataset to handle a much smaller version of the LibriSpeech Dataset used for debugging
    """

    def __init__(self, data_path, letter2index, partition="train", shuffle=True):
        self.X = np.load(data_path + "/" + partition + ".npy", allow_pickle=True)
        self.Y = np.load(
            data_path + "/" + partition + "_transcripts.npy", allow_pickle=True
        )
        self.letter2index = letter2index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        x = torch.tensor(self.X[ind])
        y_s = self.Y[ind]
        y_s = y_s[1:]
        y = torch.tensor([self.letter2index[x] for x in y_s])
        return x, y

    def collate_fn(self, batch):
        x_batch = [x for x, y in batch]
        y_batch = [y for x, y in batch]
        batch_x_pad = pad_sequence(x_batch, batch_first=True)
        lengths_x = [x.shape[0] for x in x_batch]

        batch_y_pad = pad_sequence(y_batch, batch_first=True)
        lengths_y = [y.shape[0] for y in y_batch]

        return (
            batch_x_pad,
            batch_y_pad,
            torch.tensor(lengths_x),
            torch.tensor(lengths_y),
        )


def get_simple_dataloader(root, batch_size, num_workers=4):
    LETTER_LIST = [
        "<sos>",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "'",
        " ",
        "<eos>",
    ]
    l2i, i2l = create_dictionaries(LETTER_LIST)
    train_data = LibriSamplesSimple(data_path=root, letter2index=l2i)
    val_data = LibriSamplesSimple(data_path=root, letter2index=l2i, partition="dev")
    train_loader = DataLoader(
        train_data,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=train_data.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=val_data.collate_fn,
        drop_last=True,
    )
    return train_loader, val_loader


def test_simple_dataloader(root, LETTER_LIST):
    bs = 64
    train_loader, val_loader = get_simple_dataloader(root, batch_size=bs)
    l2i, i2l = create_dictionaries(LETTER_LIST)
    train_data = LibriSamplesSimple(root, l2i, "train")
    val_data = LibriSamplesSimple(root, l2i, "dev")
    print("Batch size: ", bs)
    print(
        "Train dataset samples = {}, batches = {}".format(
            train_data.__len__(), len(train_loader)
        )
    )
    print(
        "Val dataset samples = {}, batches = {}".format(
            val_data.__len__(), len(val_loader)
        )
    )
    pass


def testi2l(root, LETTER_LIST):
    bs = 2
    train_loader, val_loader, test_loader = get_dataloader(root, batch_size=bs)
    l2i, i2l = create_dictionaries(LETTER_LIST)
    x, y, len_x, len_y = next(iter(val_loader))
    ts = transform_index_to_letter(y, i2l)
    mask = generate_mask(len_y)
    print(ts)


def generate_mask(lens):
    """
    Generates a mask, used by attention module
    """
    lens = torch.tensor(lens).to(device)
    max_len = torch.max(lens)
    masks = []
    for i in range(lens.shape[0]):
        mask_temp = (torch.arange(0, max_len).to(device) < lens[i]).int()
        masks.append(mask_temp)
    masks = torch.stack(masks, dim=0)
    return masks


if __name__ == "__main__":
    LETTER_LIST = [
        "<sos>",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "'",
        " ",
        "<eos>",
    ]
    root = "LAS-Dataset/complete"
    simple = "LAS-Dataset/simple"
    test_dataloaders(root, LETTER_LIST)
    test_simple_dataloader(simple, LETTER_LIST)
    testi2l(root, LETTER_LIST)
