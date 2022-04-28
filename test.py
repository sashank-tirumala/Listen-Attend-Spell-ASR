import os
import sys
import pandas as pd
import numpy as np
# import Levenshtein as lev
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
from dataloader import get_dataloader, create_dictionaries, get_simple_dataloader, generate_mask
from LAS_model import Seq2Seq
import wandb
import argparse
from Levenshtein import distance as lev
from tqdm import tqdm
LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
         'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']
class WeiSimple(torch.utils.data.Dataset):

    def __init__(self, data_path, partition= "train"):

        self.X_files = data_path + "/" + partition + ".npy"
        self.Y_files = data_path + "/" + partition + "_transcripts.npy"

        self.X_dataset = np.load(self.X_files, allow_pickle=True)# TODO: Load the mfcc npy file at the specified index ind in the directory
        self.Y_dataset = np.load(self.Y_files, allow_pickle=True)# TODO: Load the corresponding transcripts

        self.LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
         'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']

    def __len__(self):
        # print("get length")
        return len(self.X_dataset)

    def __getitem__(self, ind):
        X = np.array(self.X_dataset[ind])
        Y = np.array(self.Y_dataset[ind])

        # TODO: Convert sequence of  phonemes into sequence of Long tensors
        Y_crop = Y[1:] # Remove <SOS>
        YY = np.vectorize(letter2index.get)(Y_crop)

        X = torch.tensor(X)
        YY = torch.tensor(YY)

        return X, YY
    
    def collate_fn(self, batch):

        # print("call collate_fn in Library samples batch: ", type(batch), np.array(batch).shape, np.array(batch[0][0]).shape, np.array(batch[0][1]).shape)
        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]

        batch_x_pad = pad_sequence(batch_x, batch_first=True)# TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [x.shape[0] for x in batch_x]# TODO: Get original lengths of the sequence before padding
        batch_y_pad = pad_sequence(batch_y, batch_first=True)# TODO: pad the sequence with pad_sequence (already imported)
        lengths_y = [y.shape[0] for y in batch_y]# TODO: Get original lengths of the sequence before padding

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)

def testing_the_2_dataloaders():
    #TESTING DATALOADERS
    root = '/home/sashank/Courses/11785_HW4_P2/hw4p2_simple/hw4p2_simple'
    bs=2
    letter2index, index2letter = create_dictionaries(LETTER_LIST)

    train_loader, val_loader = get_simple_dataloader(root, batch_size=bs)
    x,y,len_x, len_y = next(iter(train_loader))
    print("mine: ", x[:5,:,:],y[:5,:],len_x, len_y, x.shape, y.shape)
    train_data = WeiSimple(root, 'train')
    val_data = WeiSimple(root, 'dev')
    train_loader = DataLoader(train_data, batch_size=bs, collate_fn=train_data.collate_fn, shuffle=True, drop_last=False, num_workers=2)# TODO: Define the train loader. Remember to pass in a parameter (function) for the collate_fn argument 
    val_loader = DataLoader(val_data, batch_size=bs, collate_fn=val_data.collate_fn, shuffle=True, drop_last=False, num_workers=1)# TODO: Define the val loader. Remember to pass in a parameter (function) for the collate_fn argument 
    x,y,len_x, len_y = next(iter(train_loader))
    print("Wei: ", x[:,:5,:],y[:,:5],len_x, len_y, x.shape, y.shape)
if(__name__ == "__main__"):
    
    pass
