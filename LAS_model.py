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
		self.blstm = nn.LSTM(input_size = input_dim*2, hidden_size = hidden_dim, num_layers=1, bidirectional=True)


	def forward(self, inp):
		# from IPython import embed; embed()
		x, len_x  = pad_packed_sequence(inp)
		if(x.shape[0]%2 == 1):
			x = x[:-1, :, :]
			len_x[len_x.argmax()] -= 1
		for i in range(len_x.shape[0]):
			if(len_x[i]%2 == 1):
				len_x[i] +=1
		len_x = len_x/2
		x = x.permute(1,0,2)
		x = x.reshape((x.shape[0], int(x.shape[1]/2), x.shape[2]*2))
		# from IPython import embed; embed()
		x = x.permute(1,0,2)
		packed_input = pack_padded_sequence(x,len_x, enforce_sorted=False)
		del x, len_x
		out1, (out2, out3) = self.blstm(packed_input)
		return out1

class Encoder(nn.Module):
	'''
	Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.

	'''
	def __init__(self, input_dim, encoder_hidden_dim, key_value_size=128, num_layers=4):
		super(Encoder, self).__init__()
		self.lstm = nn.LSTM(input_size = input_dim, hidden_size = encoder_hidden_dim, bidirectional=True, num_layers=1)
		#TODO add DropOut Below, maybe add more layers
		self.pBLSTMs = nn.ModuleList([pBLSTM(input_dim = encoder_hidden_dim*2, hidden_dim = encoder_hidden_dim)]*num_layers)
		self.pBLSTMs = nn.Sequential(*self.pBLSTMs)
		self.key_network = nn.Linear(in_features = encoder_hidden_dim*2, out_features = 128)
		self.value_network = nn.Linear(in_features = encoder_hidden_dim*2,out_features = 128)

	def forward(self, x, len_x):
		"""
        1. Pack your input and pass it through the first LSTM layer (no truncation)
        2. Pass it through the pyramidal LSTM layer
        3. Pad your input back to (B, T, *) or (T, B, *) shape
        4. Output Key, Value, and truncated input lens

        Key and value could be
            (i) Concatenated hidden vectors from all time steps (key == value).
            (ii) Linear projections of the output from the last pBLSTM network.
                If you choose this way, you can use the final output of
                your pBLSTM network.
        """
		packed_input = pack_padded_sequence(x,len_x, enforce_sorted=False)
		out1, (out2, out3) = self.lstm(packed_input)
		del out2, out3
		out1 = self.pBLSTMs(out1)
		out, lengths = pad_packed_sequence(out1)
		print(out.shape)
		out = out.permute(1,0,2)
		key = self.key_network(out)
		value = self.value_network(out)
		return key, value, lengths

	



def test_pBLSTM():
	from dataloader import get_dataloader
	root = 'hw4p2_student_data/hw4p2_student_data'
	train_loader, val_loader, test_loader = get_dataloader(root, batch_size=2)
	x,y,len_x, len_y = next(iter(train_loader))
	pyr = pBLSTM(int(x.shape[-1]), 512)
	print(x.shape)
	y, len_y = pyr(x, len_x)
	print(y.shape)
	pass

def test_encoder():
	from dataloader import get_dataloader
	root = 'hw4p2_student_data/hw4p2_student_data'
	train_loader, val_loader, test_loader = get_dataloader(root, batch_size=2)
	x,y,len_x, len_y = next(iter(train_loader))
	pyr = Encoder(int(x.shape[-1]), 256)
	print(x.shape)
	key, value, len_y = pyr(x, len_x)
	print(key.shape, value.shape)




if(__name__ == "__main__"):
	# test_pBLSTM()
	test_encoder()