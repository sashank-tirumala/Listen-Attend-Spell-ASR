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
from dataloader import get_dataloader
from LAS_model import Seq2Seq
import wandb
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(cfg):
    optimizer = optim.Adam(model.parameters(), lr = cfg["lr"], weight_decay=cfg["w_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs'], eta_min=1e-6, last_epoch=- 1, verbose=False)
    criterion = nn.CrossEntropyLoss(reduction='none')
    model = Seq2Seq(input_dim = 13,  encoder_hidden_dim = cfg["encoder_dim"], decoder_hidden_dim = cfg["decoder_dim"], vocab_size=30, embed_dim=cfg["embed_dim"], key_value_size=cfg["key_value_size"], num_layers = cfg["num_layers_encoder"]).to(device)
	root = cfg["datapath"]
	train_loader, val_loader, test_loader = get_dataloader(root, batch_size=cfg["batch_size"])
    n_epochs = cfg["epochs"]
    running_loss = 0
    for epoch in range(n_epochs):
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, desc='Train')
        total_loss = 0
        for i, data in enumerate(trian_loader):
            scaler = torch.cuda.amp.GradScaler()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  
                x, y, lx, ly = data
                x = x.cuda()
                y = y.cuda()
                y = y.transpose(0,1)
                predictions, attentions = model(x, x_len, y, mode=mode)
                lx = lx.cpu()
                loss = criterion(out,y,h,ly)
                del x,y,out,h
                torch.cuda.empty_cache()
            total_loss += float(loss)
            wandb.log( { 'loss_step' : float(total_loss / (i + 1)) } )
            wandb.log({'lr_step': float(optimizer.param_groups[0]['lr']) })
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 
            scheduler.step() # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.
            
            batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
            batch_bar.update() # Update tqdm bar


    
if(__name__ == "__main__"):
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr','--lr', type=float, help='learning rate', default = 1e-3) # required=True ,
    parser.add_argument('-wd','--w_decay', type=float, help='weight decay (regularization)', default=0) #required=True ,
    # parser.add_argument('-m','--momentum', type=float, help='momentum (Adam)',  default=0)#required=True ,
    parser.add_argument('-bs','--batch_size', type=int, help='Description for bar argument', default=8)#required=True ,
    parser.add_argument('-e','--epochs', type=int, help='Description for bar argument', default=50)#required=True ,
    parser.add_argument('-dp','--datapath', type=str, help='Description for bar argument', default="/home/sashank/deepl_project/data/dataset/test/")#required=True ,
    parser.add_argument('-rp','--runspath', type=str, help='Description for bar argument', default="/home/sashank/deepl_project/cloth-segmentation/train_runs")#required=True ,
    parser.add_argument('-t','--transform', type=bool, help='Description for bar argument', default=True)#required=True ,
    parser.add_argument('-nl','--num_layers_encoder', type=int, help='Number of layers in encoder only', default=3)#required=True ,
    parser.add_argument('-ed','--encoder_dim', type=int, help='Number of layers in encoder only', default=256)#required=True ,
    parser.add_argument('-dd','--decoder_dim', type=int, help='Number of layers in encoder only', default=256)#required=True ,
    parser.add_argument('-ebd','--embed_dim', type=int, help='Number of layers in encoder only', default=128)#required=True ,
    parser.add_argument('-kvs','--key_value_size', type=int, help='Number of layers in encoder only', default=128)#required=True ,

    args = vars(parser.parse_args())
    wandb.init(project="11785_HW4P2", entity="stirumal", config=args)
    