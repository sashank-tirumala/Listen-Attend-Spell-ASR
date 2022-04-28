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

class WeiAttention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # Optional: dropout


    def forward(self, query, key, value, mask):
        query = query.unsqueeze(2) #(batch_size, d_q, 1)
        energy = torch.bmm(key, query).squeeze(2) #(batch_size, seq_len, 1) -> (batch_size, seq_len)
        energy = energy.masked_fill(mask, -1e-9)  #mask out according to paper

        attention = F.softmax(energy, dim=1)  #(batch_size, seq_len)
        
        atthetion_unsqueeze = attention.unsqueeze(1) #(batch_size, 1, seq_len)
        context = torch.bmm(atthetion_unsqueeze, value) # (batch_size,1, seq_len) @ (b, seq_len, d_v) -> (B, 1, d_v)
        context = context.squeeze(1) # (batch_size, d_v)
        
        return context, attention

class WeiEncoder(nn.Module):
    
    def __init__(self, input_dim, encoder_hidden_dim, key_value_size=128):
        super(Encoder, self).__init__()
        # The first LSTM layer at the bottom
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=encoder_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        # The linear transformations for producing Key and Value for attention
        # Hint: Dimensions when bidirectional lstm? 
        self.key_network = nn.Linear(encoder_hidden_dim*2, key_value_size)
        self.value_network = nn.Linear(encoder_hidden_dim*2, key_value_size)

    def forward(self, x, x_len):
        
        # Pack input
        x_packed = pack_padded_sequence(x, x_len, enforce_sorted=False, batch_first=True)

        # Pass it through first LSTM
        out, hidden = self.lstm(x_packed)

        # Pad(unpack) input back to (B, T, *)
        out_unpack, lengths = pad_packed_sequence(out, batch_first=True)  # lengths is original signal length

        key = self.key_network(out_unpack)
        value = self.value_network(out_unpack)

        return key, value, lengths
class WeiDecoder(nn.Module):
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Decoder, self).__init__()
        # Hint: Be careful with the padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # padding idx should be <sos>
        self.lstm1 = nn.LSTMCell(input_size=embed_dim + key_value_size, hidden_size=key_value_size)
    
        self.attention = Attention()     
        self.vocab_size = vocab_size
        self.character_prob = nn.Linear(key_value_size*2, vocab_size) #: d_v -> vocab_size
        self.key_value_size = key_value_size
        
        # Weight tying
        self.character_prob.weight = self.embedding.weight  # embed_dim should equal to charachter_prob size for weight tyning

    def forward(self, key, value, encoder_len, y=None, mode='train', teaching_rate=0.0):
        B, key_seq_max_len, key_value_size = key.shape

        if mode == 'train':
            max_len =  y.shape[1]                # (B, S+2) or (B, text_len) y alrealy been padded, len are same
            char_embeddings = self.embedding(y)  #(B, S+2, embed_dim) or (B, text_len, embed_dim)
        else:
            max_len = 600
        mask = torch.arange(encoder_len.max()).unsqueeze(0) >= encoder_len.unsqueeze(1)  # encoder_len original len for seq, (B, T)
        mask = mask.to(device)
        predictions = []
        prediction = torch.full((B,1), fill_value=0, device=device)
        hidden_states = [None, None] 
        context = value[:,0,:]
        context = context.to(device)
        attention_plot = [] 
        for i in range(max_len):
            if mode == 'train':
                if random.random() < teaching_rate:
                    if i == 0:
                      char_embed = torch.full((B, char_embeddings.shape[2]), fill_value=0, device=device) 
                    else:
                      char_embed = char_embeddings[:, i-1, :] 
                else:
                  char_embed = self.embedding(prediction.argmax(dim=-1))    
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1)) # embedding of the previous prediction
            y_context = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(y_context, hidden_states[0])
            query = hidden_states[0][0]# fill this out
            context, attention = self.attention(query, key, value, mask)
            attention_plot.append(attention[0].detach().cpu())
            output_context = torch.cat([query, context], dim=1)
            prediction = self.character_prob(output_context)
            predictions.append(prediction.unsqueeze(1))
        attentions = torch.stack(attention_plot, dim=0)
        predictions = torch.cat(predictions, dim=1)

        return predictions, attentions
if(__name__ == "__main__"):
    
    pass
