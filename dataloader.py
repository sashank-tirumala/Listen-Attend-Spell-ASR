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
def create_dictionaries(letter_list):
    '''
    Create dictionaries for letter2index and index2letter transformations
    based on LETTER_LIST

    Args:
        letter_list: LETTER_LIST

    Return:
        letter2index: Dictionary mapping from letters to indices
        index2letter: Dictionary mapping from indices to letters
    '''
    letter2index = dict()
    index2letter = dict()
    for i in range(len(letter_list)):
        index2letter[i] = letter_list[i]
        letter2index[letter_list[i]] = i 
    return letter2index, index2letter
    
#Don't know what he wants right  now and why, will figure out
def transform_index_to_letter(batch_indices, lindex2letter):
    '''
    Transforms numerical index input to string output by converting each index 
    to its corresponding letter from LETTER_LIST

    Args:
        batch_indices: List of indices from LETTER_LIST with the shape of (N, )
    
    Return:
        transcripts: List of converted string transcripts. This would be a list with a length of N
    '''
    transcripts = []
    # TODO
    return transcripts
        
# Create the letter2index and index2letter dictionary
# letter2index, index2letter = create_dictionaries(LETTER_LIST)
class LibriSamples(torch.utils.data.Dataset):

    def __init__(self, data_path, letter2index, partition= "train", shuffle=True):
        self.x_dir = data_path+"/"+partition+"/mfcc"
        self.y_dir = data_path+"/"+partition+"/transcript"
        self.x_files = os.listdir(data_path+"/"+partition+"/mfcc")
        self.y_files = os.listdir(data_path+"/"+partition+"/transcript")
        self.files = [x for x in zip(self.x_files, self.y_files)]
        if(shuffle):
            random.shuffle(self.files)
        self.letter2index = letter2index
        


    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, ind):
        xdir ,ydir = self.files[ind]
        x = torch.tensor(np.load(self.x_dir+"/"+xdir))
        y_s = np.load(self.y_dir+"/"+ydir)
        y = torch.tensor([self.letter2index[x] for x in y_s])
        return x,y

    def collate_fn(batch):
        x_batch = [x for x,y in batch]
        y_batch = [y for x,y in batch]
        batch_x_pad = pad_sequence(x_batch)
        lengths_x = [len(x) for x in x_batch]

        batch_y_pad = pad_sequence(y_batch) 
        lengths_y = [len(y) for y in y_batch]

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)

class LibriSamplesTest(torch.utils.data.Dataset):

    def __init__(self, data_path, test_order): # test_order is the csv similar to what you used in hw1
        with open(data_path + '/test/'+test_order, newline='') as f:
          reader = csv.reader(f)
          test_order_list = list(reader)
        self.X = [torch.tensor(np.load(data_path + '/test/mfcc/' + X_path[0])) for X_path in test_order_list[1:]] # TODO: Load the npy files from test_order.csv and append into a list
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ind):
        return self.X[ind]
    
    def collate_fn(self,batch):
        batch_x = [x for x in batch]
        batch_x_pad = pad_sequence(batch_x)
        lengths_x = [len(x) for x in batch_x]

        return batch_x_pad, torch.tensor(lengths_x)

def test_dataloaders():
    root = 'hw4p2_student_data/hw4p2_student_data'
    bs=64
    train_loader, val_loader, test_loader = get_dataloader(root, batch_size=bs)
    LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
         'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']
    l2i, i2l = create_dictionaries(LETTER_LIST)
    train_data = LibriSamples(root, l2i, 'train')
    val_data = LibriSamples(root, l2i, 'dev')
    test_data = LibriSamplesTest(root, "test_order.csv")
    print("Batch size: ", bs)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

def get_dataloader(root, batch_size=64, num_workers=4):
    LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
         'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']
    l2i, i2l = create_dictionaries(LETTER_LIST)
    train_data = LibriSamples(root, l2i, 'train')
    val_data = LibriSamples(root, l2i, 'dev')
    test_data = LibriSamplesTest(root, "test_order.csv")
    train_loader = DataLoader(train_data, num_workers=num_workers, batch_size = batch_size)
    val_loader = DataLoader(val_data, num_workers=num_workers, batch_size = batch_size)
    test_loader = DataLoader(test_data, num_workers=num_workers, batch_size = batch_size)
    return train_loader, val_loader, test_loader
if(__name__ == "__main__"):
    test_dataloaders()