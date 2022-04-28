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
from torch.autograd import Variable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LockedDropout(nn.Module):
    def __init__(self, dropout = 0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or not self.dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'

class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size = input_dim*2, hidden_size = hidden_dim, num_layers=1, bidirectional=True, batch_first=True)


    def forward(self, inp):
        x, len_x  = pad_packed_sequence(inp, batch_first=True)
        if(x.shape[1]%2 == 1):
            x = x[:, :-1, :]
        len_x = torch.div(len_x, 2, rounding_mode='floor')
        x = x.view(x.shape[0], int(x.shape[1]/2), x.shape[2]*2)
        packed_input = pack_padded_sequence(x,len_x, enforce_sorted=False, batch_first=True)
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
        self.key_network = nn.Linear(in_features = encoder_hidden_dim*2, out_features = key_value_size)
        self.value_network = nn.Linear(in_features = encoder_hidden_dim*2,out_features = key_value_size)

    def forward(self, x, len_x):
        packed_input = pack_padded_sequence(x,len_x, enforce_sorted=False, batch_first=True)
        out1, (out2, out3) = self.lstm(packed_input)
        out1 = self.pBLSTMs(out1)
        out, lengths = pad_packed_sequence(out1,  batch_first=True)
        key = self.key_network(out)
        value = self.value_network(out)
        return key, value, lengths

class bmmAttention(nn.Module):
    def __init__(self, normalize = False):
        super(bmmAttention, self).__init__()
        # Optional: dropout

    def forward(self, query, key, value, mask):
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)/torch.sqrt(torch.tensor(key.shape[-1]).to(device))
        energy.masked_fill_(mask, torch.tensor(float("-inf")))
        attention = F.softmax(energy, dim= 1) #Pretty sure it is right, but noting anyway
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
        return context, attention

class multiheadAttention(nn.Module):
    def __init__(self, normalize = False):
        super(bmmAttention, self).__init__()
        # Optional: dropout

    def forward(self, query, key, value, mask):
        
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)/torch.sqrt(torch.tensor(key.shape[-1]).to(device))
        energy.masked_fill_(mask, torch.tensor(float("-inf")))
        # print("energy: ", energy.shape)
        attention = F.softmax(energy, dim= 1) #Pretty sure it is right, but noting anyway
        # print("attention: ",attention.shape)
        # print("value: ",value.shape)
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
        return context, attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim ,key_value_size=128, num_decoder_layers=2, attention_type="single"):
        super(Decoder, self).__init__()
        #Be careful with padding_idx
        embed_dim = 2*key_value_size #Needed for weight tying
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_dim, padding_idx = 0)
        #Might want to concatenate context here
        self.lstms = nn.ModuleList([])
        self.lstms.append(nn.LSTMCell(embed_dim+key_value_size, hidden_size = decoder_hidden_dim))
        for i in range(0, num_decoder_layers-2):
            self.lstms.append(nn.LSTMCell(decoder_hidden_dim, hidden_size = decoder_hidden_dim))
        self.lstms.append(nn.LSTMCell(decoder_hidden_dim, hidden_size = key_value_size))
        if(attention_type=="single"):
            self.attention = bmmAttention()
        self.vocab_size = vocab_size
        
        self.character_prob = nn.Linear(in_features = key_value_size*2, out_features = vocab_size)
        self.key_value_size = key_value_size
        self.num_layers = num_decoder_layers

        #Optional Weight Tying
        self.character_prob.weight = self.embedding.weight #

    def forward(self, key , value, encoder_len, y=None, mode='train', teacher_forcing=True):
        B, key_seq_max_len, key_value_size = key.shape
        if mode == 'train':
            max_len = y.shape[1]
            char_embeddings = self.embedding(y)
        else:
            max_len=600
        
        #Creating the attention mask here:
        mask = torch.arange(encoder_len.max()).unsqueeze(0) >= encoder_len.unsqueeze(1)  # encoder_len original len for seq, (B, T)
        mask = mask.to(device)
        #TODO Initialize the context
        predictions = []
        prediction = torch.full((B,1), fill_value=0, device=device)
        hidden_states = [None]*self.num_layers
        prediction = torch.zeros(B, 1).to(device)
        context = value[:, 0, :] #initializing context with the first value
        context = context.to(device)
        attention_plot=[]
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
            hidden_states[0] = self.lstms[0](y_context, hidden_states[0])
            for i in range(1, self.num_layers):
                hidden_states[i] = self.lstms[i](hidden_states[i-1][i-1], hidden_states[i])
            query =  hidden_states[-1][0]
            context, attention = self.attention(query, key, value, mask)
            attention_plot.append(attention[encoder_len.argmax()].detach().cpu())
            output_context = torch.cat([query, context], dim=1)
            prediction = self.character_prob(output_context)
            predictions.append(prediction.unsqueeze(1))		
        attentions = torch.stack(attention_plot, dim=0)
        predictions = torch.cat(predictions, dim=1)
        return predictions, attentions

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, key_value_size=128, num_layers=4, num_decoder_layers=2, attention="single"):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim, num_layers = num_layers)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim ,key_value_size=128, num_decoder_layers=num_decoder_layers, attention_type="single")
    
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

def test_locked_dropout():
    from dataloader import get_dataloader
    root = 'hw4p2_student_data/hw4p2_student_data'
    train_loader, val_loader, test_loader = get_dataloader(root, batch_size=2)
    x,y,len_x, len_y = next(iter(val_loader))
    dp = LockedDropout(dropout = 0.99)
    print(x)
    x = dp(x)
    print(x)
    pass


if(__name__ == "__main__"):
    # test_pBLSTM()
    # test_encoder()
    # test_bmmAttention()
    # test_decoder()
    # test_seq2seq()
    test_locked_dropout()