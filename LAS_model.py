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
	def __init__(self, input_dim, encoder_hidden_dim, key_value_size=128, num_layers=4):
		super(Encoder, self).__init__()
		self.lstm = nn.LSTM(input_size = input_dim, hidden_size = encoder_hidden_dim, bidirectional=True, num_layers=1)
		#TODO add DropOut Below, maybe add more layers
		self.pBLSTMs = nn.ModuleList([pBLSTM(input_dim = encoder_hidden_dim*2, hidden_dim = encoder_hidden_dim)]*num_layers)
		self.pBLSTMs = nn.Sequential(*self.pBLSTMs)
		self.key_network = nn.Linear(in_features = encoder_hidden_dim*2, out_features = 128)
		self.value_network = nn.Linear(in_features = encoder_hidden_dim*2,out_features = 128)

	def forward(self, x, len_x):
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

class bmmAttention(nn.Module):
	def __init__(self, normalize = False):
		super(bmmAttention, self).__init__()
		# Optional: dropout

	def forward(self, query, key, value, mask):
		#TODO add normalize option
		query=query.reshape(query.shape[0],-1,1)
		print("query: ", query.shape)
		energy  = torch.bmm(key, query) + 1e-9
		print("energy: ", energy.shape)
		energy = torch.mul(energy, mask)
		attention = F.softmax(energy, dim=1) #Pretty sure it is right, but noting anyway
		print("attention: ",attention.shape)
		print("value: ",value.shape)
		value = value.permute(0,2,1)
		context = torch.bmm(value, attention)
		print("context: ", context.shape)
		return context, attention


class Decoder(nn.Module):
	def __init__(self, vocab_size, decoder_hidden_dim, embed_dim ,key_value_size=128):
		super(Decoder, self).__init__()
		#Be careful with padding_idx
		self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_dim, padding_idx = 29)
		#Might want to concatenate context here
		self.lstm1 = nn.LSTMCell(embed_dim, hidden_size = decoder_hidden_dim, bias=True)
		self.lstm2 = nn.LSTMCell(decoder_hidden_dim, decoder_hidden_dim, bias=True)

		self.attention = bmmAttention()
		self.query_linear = nn.Linear(in_features = decoder_hidden_dim, out_features = key_value_size)
		self.vocab_size = vocab_size
		
		#Optional Weight Tying
		self.character_prob = nn.Linear(in_features = key_value_size, out_features = vocab_size, bias=True)
		self.key_value_size = key_value_size

		#Weight Tying
		self.character_prob.weight = self.embedding.weight

	def forward(self, key , value, encoder_len, y=None, mode='train'):
		B, key_seq_max_len, key_value_size = key.shape
		if mode == 'train':
			y = y.permute(1,0)
			max_len = y.shape[1]
			char_embeddings = self.embedding(y)
		else:
			y = y.permute(1,0)
			max_len=600
		
		#Creating the attention mask here:
		mask = torch.zeros(key.shape[0],key.shape[1],1)
		from IPython import embed; embed()




		pass


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

def test_bmmAttention():
	# from dataloader import get_dataloader
	# root = 'hw4p2_student_data/hw4p2_student_data'
	# train_loader, val_loader, test_loader = get_dataloader(root, batch_size=2)
	# x,y,len_x, len_y = next(iter(train_loader))
	key = torch.rand(32, 80, 128)
	value = torch.rand(32, 80, 128)
	query = torch.rand(32, 128)
	mask = torch.zeros(32, 80, 1)
	mask[:,40,:] = 1.
	att = bmmAttention()
	ctxt, att= att(query, key, value, mask)
	print(ctxt.shape, att.shape)

def test_decoder():
	from dataloader import get_dataloader
	root = 'hw4p2_student_data/hw4p2_student_data'
	train_loader, val_loader, test_loader = get_dataloader(root, batch_size=2)
	x,y,len_x, len_y = next(iter(train_loader))
	pyr = Encoder(int(x.shape[-1]), 256)
	key, value, len_y = pyr(x, len_x)
	dec = Decoder(vocab_size = 30, decoder_hidden_dim=128, embed_dim=128, key_value_size=128)
	dec(key, value, len_y, y=y, mode='train')





if(__name__ == "__main__"):
	# test_pBLSTM()
	#test_encoder()
	# test_bmmAttention()
	test_decoder()