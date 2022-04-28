import os
import sys
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size = input_dim*2, hidden_size = hidden_dim, num_layers=1, bidirectional=True, batch_first=True)


    def forward(self, inp):
        # from IPython import embed; embed()
        x, len_x  = pad_packed_sequence(inp)
        if(x.shape[0]%2 == 1):
            x = x[:-1, :, :]
        for i in range(len_x.shape[0]):
            if(len_x[i]%2 == 1):
                len_x[i] -=1
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
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = encoder_hidden_dim, bidirectional=True, num_layers=1, batch_first=True)
        #TODO add DropOut Below, maybe add more layers
        self.pBLSTMs = nn.ModuleList([pBLSTM(input_dim = encoder_hidden_dim*2, hidden_dim = encoder_hidden_dim)]*num_layers)
        self.pBLSTMs = nn.Sequential(*self.pBLSTMs)
        self.key_network = nn.Linear(in_features = encoder_hidden_dim*2, out_features = 128)
        self.value_network = nn.Linear(in_features = encoder_hidden_dim*2,out_features = 128)

    def forward(self, x, len_x):
        packed_input = pack_padded_sequence(x,len_x, enforce_sorted=False, batch_first=True)
        out1, (out2, out3) = self.lstm(packed_input)
        # out1 = self.pBLSTMs(out1)
        out, lengths = pad_packed_sequence(out1,  batch_first=True)
        key = self.key_network(out)
        value = self.value_network(out)
        return key, value, lengths

class bmmAttention(nn.Module):
    def __init__(self, normalize = False):
        super(bmmAttention, self).__init__()
        # Optional: dropout

    def forward(self, query, key, value, mask):
        # from IPython import embed; embed()
        #TODO add normalize option
        # print("query: ", query.shape)
        # from IPython import embed; embed()
        # size = torch.tensor(key.shape[1], dtype=torch.float32).to(device)
        #TODO Mask Energy -- Mask might be wrong, mostly correct though
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        # energy.masked_fill_(mask, torch.tensor(float("-inf")))
        # print("energy: ", energy.shape)
        attention = F.softmax(energy, dim= - 1) #Pretty sure it is right, but noting anyway
        # print("attention: ",attention.shape)
        # print("value: ",value.shape)
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
        return context, attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim ,key_value_size=128):
        super(Decoder, self).__init__()
        #Be careful with padding_idx
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_dim, padding_idx = 0)
        #Might want to concatenate context here
        self.lstm1 = nn.LSTMCell(embed_dim+key_value_size, hidden_size = key_value_size)
        # self.lstm2 = nn.LSTMCell(decoder_hidden_dim, decoder_hidden_dim, bias=True) TODO UNCOMMENT when basic attention done

        self.attention = bmmAttention()
        # self.query_linear = nn.Linear(in_features = decoder_hidden_dim, out_features = key_value_size) #Might not be needed
        self.vocab_size = vocab_size
        
        self.character_prob = nn.Linear(in_features = key_value_size*2, out_features = vocab_size)
        self.key_value_size = key_value_size

        #Optional Weight Tying
        print(self.character_prob.weight.shape, self.embedding.weight.shape)
        self.character_prob.weight = self.embedding.weight #

    def forward(self, key , value, encoder_len, y=None, mode='train', teacher_forcing=True):
        B, key_seq_max_len, key_value_size = key.shape
        if mode == 'train':
            # y = y.permute(1,0)
            max_len = y.shape[1]
            char_embeddings = self.embedding(y)
        else:
            # y = y.permute(1,0)
            max_len=600
        
        #Creating the attention mask here:
        mask = torch.arange(encoder_len.max()).unsqueeze(0) >= encoder_len.unsqueeze(1)  # encoder_len original len for seq, (B, T)
        mask = mask.to(device)
        #TODO Initialize the context
        predictions = []
        prediction = torch.full((B,1), fill_value=0, device=device)
        hidden_states = [None, None]
        prediction = torch.zeros(B, 1).to(device)
        context = value[:, 0, :] #initializing context with the first value
        context = context.to(device)
        attention_plot=[]
        # from IPython import embed; embed()
        for i in range(max_len):
            if mode == 'train':
                if teacher_forcing:
                    if i == 0:
                        char_embed = torch.full((B, char_embeddings.shape[2]), fill_value=0, device=device) # For timestamp 0, input should be <SOS>, which is 0
                    else:
                        char_embed = char_embeddings[:, i-1, :]
                else:
                    char_embed = self.embedding(softmax(prediction).argmax(dim=-1))
            else:
                char_embed = self.embedding(softmax(prediction).argmax(dim=-1))
            y_context = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(y_context, hidden_states[0])
            # hidden_states[1] = self.lstm2(hidden_states[0][0], hidden_states[1])

            query =  hidden_states[0][0] #TODO change with multilayers
            context, attention = self.attention(query, key, value, mask)
            # print(i, attention.shape)
            attention_plot.append(attention[0].detach().cpu())
            output_context = torch.cat([query, context], dim=1)
            # del query
            prediction = self.character_prob(output_context)
            #Checking if any major errors were made
            predictions.append(prediction.unsqueeze(1))		
        attentions = torch.stack(attention_plot, dim=0)
        predictions = torch.cat(predictions, dim=1)
        return predictions, attentions

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, key_value_size=128, num_layers=4):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim, num_layers = num_layers)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim ,key_value_size=128)
    
    def forward(self, x, x_len, y=None, mode='train', teacher_forcing=True):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions, attentions = self.decoder(key, value, encoder_len, y=y, mode=mode, teacher_forcing=teacher_forcing)
        return predictions, attentions

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
    train_loader, val_loader, test_loader = get_dataloader(root, batch_size=64)
    x,y,len_x, len_y = next(iter(train_loader))
    x = x.cuda()
    y = y.cuda()
    pyr = Encoder(int(x.shape[-1]), 256).to(device)
    print(x.shape)
    key, value, len_y = pyr(x, len_x)
    print(key.shape, value.shape)

def test_bmmAttention():
    from dataloader import get_dataloader
    root = 'hw4p2_student_data/hw4p2_student_data'
    train_loader, val_loader, test_loader = get_dataloader(root, batch_size=2)
    x,y,len_x, len_y = next(iter(train_loader))
    x = x.cuda()
    y = y.cuda()
    pyr = Encoder(int(x.shape[-1]), 256).to(device)
    key, value, len_y = pyr(x, len_x)
    att = bmmAttention()
    ctxt, att= att(query, key, value, mask)
    print(ctxt.shape, att.shape)

def test_decoder():
    from dataloader import get_dataloader
    root = 'hw4p2_student_data/hw4p2_student_data'
    train_loader, val_loader, test_loader = get_dataloader(root, batch_size=2)
    x,y,len_x, len_y = next(iter(train_loader))
    x = x.cuda()
    y = y.cuda()
    # len_x = len_x.cuda()
    # len_y = len_y.cuda()
    pyr = Encoder(int(x.shape[-1]), 256).to(device)
    print(x.shape[-1])
    key, value, len_y = pyr(x, len_x)
    print(key.dtype, value.dtype, len_y.dtype)
    dec = Decoder(vocab_size = 30, decoder_hidden_dim=128, embed_dim=128, key_value_size=128).to(device)
    pred, att = dec(key, value, len_y, y=y, mode='train')
    print("predictions: ", pred)
    print("attention: ",att)

def test_seq2seq():
    from dataloader import get_dataloader
    root = 'hw4p2_student_data/hw4p2_student_data'
    train_loader, val_loader, test_loader = get_dataloader(root, batch_size=2)
    x,y,len_x, len_y = next(iter(val_loader))
    x = x.cuda()
    y = y.cuda()
    print(x.shape)
    print(x.shape[-1])
    net = Seq2Seq(input_dim = x.shape[-1], encoder_hidden_dim = 256, decoder_hidden_dim = 256, vocab_size=30, embed_dim=128, key_value_size=128).to(device)
    pred, att= net(x,len_x, y, "train")
    print("Predictions: ",pred.shape)
    print("Attention: ",att.shape)



if(__name__ == "__main__"):
    # test_pBLSTM()
    test_encoder()
    # test_bmmAttention()
    # test_decoder()
    # test_seq2seq()