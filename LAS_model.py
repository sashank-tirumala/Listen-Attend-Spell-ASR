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

class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    Read paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed
    2. Truncate the input length dimension by concatenating feature dimension
        (i) How should  you deal with odd/even length input? 
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers=1, bidirectional=True)


    def forward(self, x, len_x):
        # from IPython import embed; embed()
        if(x.shape[0]%2 == 1):
            x = x[:-1, :, :]
            len_x[len_x.argmax()] -= 1
        len_x = torch.minimum(len_x.max()/2, len_x)
        x = x.permute(1,0,2)
        x = x.reshape((x.shape[0], int(x.shape[1]/2), x.shape[2]*2))
        from IPython import embed; embed()
        x = x.permute(1,0,2)
        packed_input = pack_padded_sequence(x,len_x, enforce_sorted=False)
        del x, len_x
        out1, (out2, out3) = self.blstm(packed_input)
        del out2, out3
        out, lengths  = pad_packed_sequence(out1)
        del out1
        return out, lengths

def test_pBLSTM():
    from dataloader import get_dataloader
    root = 'hw4p2_student_data/hw4p2_student_data'
    train_loader, val_loader, test_loader = get_dataloader(root, batch_size=2)
    x,y,len_x, len_y = next(iter(train_loader))
    print(x.shape)
    pyr = pBLSTM(int(x.shape[-1]*2), 512)
    y = pyr(x, len_x)
    pass

if(__name__ == "__main__"):
    test_pBLSTM()