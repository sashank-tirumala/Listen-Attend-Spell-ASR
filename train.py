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
from dataloader import get_dataloader, create_dictionaries
from LAS_model import Seq2Seq
import wandb
import argparse
from Levenshtein import distance as lev
LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
         'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']
l2i, i2l = create_dictionaries(LETTER_LIST)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_attention(attention):
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()

def train(cfg):
	model = Seq2Seq(input_dim = 13,  encoder_hidden_dim = cfg["encoder_dim"], decoder_hidden_dim = cfg["decoder_dim"], vocab_size=30, embed_dim=cfg["embed_dim"], key_value_size=cfg["key_value_size"], num_layers = cfg["num_layers_encoder"]).to(device)
	optimizer = optim.Adam(model.parameters(), lr = cfg["lr"], weight_decay=cfg["w_decay"])
	root = cfg["datapath"]
	train_loader, val_loader, test_loader = get_dataloader(root, batch_size=cfg["batch_size"])
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs']*len(train_loader), eta_min=1e-6, last_epoch=- 1, verbose=False)
	criterion = nn.CrossEntropyLoss(reduction='none')
	n_epochs = cfg["epochs"]
	running_loss = 0
	for epoch in range(n_epochs):
		# batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, desc='Train')
		total_loss = 0
		for i, data in enumerate(train_loader):
			scaler = torch.cuda.amp.GradScaler()
			optimizer.zero_grad()
			with torch.cuda.amp.autocast():  
				x, y, lx, ly = data
				x = x.cuda()
				y = y.cuda()
				predictions, attentions = model(x, lx, y, mode="train")
				mask = []
				for i in range(len(ly)):
					mask_temp = torch.arange(0, predictions.shape[1], dtype=torch.float32)
					mask_temp = mask_temp > ly[i]*30
					mask.append(mask_temp)
				mask = torch.stack(mask).to(device)
				predictions = predictions.masked_fill(mask, 0.0)
				loss = criterion(predictions.view(-1, 30), y.view(-1))
				loss = torch.sum(loss)/mask.sum()
				torch.cuda.empty_cache()

			total_loss += float(loss)
			# wandb.log( { 'loss_step' : float(total_loss / (i + 1)) } )
			# wandb.log({'lr_step': float(optimizer.param_groups[0]['lr']) })
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update() 
			scheduler.step() # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.
			break
		plot_attention(attentions)
		val(model, val_loader)
		break

def val(model, val_loader):
	model.eval()
	l2i, i2l = create_dictionaries(LETTER_LIST)
	dists = []
	for i, data in enumerate(val_loader):
		x, y, lx, ly = data
		x = x.cuda()
		y = y.cuda()
		predictions, attentions = model(x, lx, y, mode="train")
		mask = []
		for i in range(len(ly)):
			mask_temp = torch.arange(0, predictions.shape[1], dtype=torch.float32)
			mask_temp = mask_temp > ly[i]*30
			mask.append(mask_temp)
		mask = torch.stack(mask).to(device)
		predictions = predictions.masked_fill(mask, 0.0)
		predictions = predictions.view(predictions.shape[0],-1,30)

		greedy_pred = torch.max(predictions, dim=2)[1]
		dists.append(get_dist(greedy_pred, y))
	dists = np.array(dists)
	return np.mean(dists), np.mean()


def get_dist(greedy_pred, y):
	dist = 0
	for b in range(y.shape[1]):
		target_str = "".join(i2l[int(x)] for x in y[1:,b])
		pred_str = "".join(i2l[int(x)] for x in greedy_pred[b,:])
		target_str = target_str.split('<eos>')[0][:]
		pred_str = pred_str.split('<eos>')[0][:]
		dist = dist + lev(target_str, pred_str)
	from IPython import embed; embed()
	return dist/y.shape[1]

if(__name__ == "__main__"):
	torch.manual_seed(11785)
	torch.cuda.manual_seed(11785)
	np.random.seed(11785)
	random.seed(11785)
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-lr','--lr', type=float, help='learning rate', default = 1e-3) # required=True ,
	parser.add_argument('-wd','--w_decay', type=float, help='weight decay (regularization)', default=0) #required=True ,
	# parser.add_argument('-m','--momentum', type=float, help='momentum (Adam)',  default=0)#required=True ,
	parser.add_argument('-bs','--batch_size', type=int, help='Description for bar argument', default=8)#required=True ,
	parser.add_argument('-e','--epochs', type=int, help='Description for bar argument', default=50)#required=True ,
	parser.add_argument('-dp','--datapath', type=str, help='Description for bar argument', default="hw4p2_student_data/hw4p2_student_data")#required=True ,
	parser.add_argument('-rp','--runspath', type=str, help='Description for bar argument', default="")#required=True ,
	parser.add_argument('-t','--transform', type=bool, help='Description for bar argument', default=True)#required=True ,
	parser.add_argument('-nl','--num_layers_encoder', type=int, help='Number of layers in encoder only', default=3)#required=True ,
	parser.add_argument('-ed','--encoder_dim', type=int, help='Number of layers in encoder only', default=256)#required=True ,
	parser.add_argument('-dd','--decoder_dim', type=int, help='Number of layers in encoder only', default=256)#required=True ,
	parser.add_argument('-ebd','--embed_dim', type=int, help='Number of layers in encoder only', default=128)#required=True ,
	parser.add_argument('-kvs','--key_value_size', type=int, help='Number of layers in encoder only', default=128)#required=True ,
	
	args = vars(parser.parse_args())
	# wandb.init(project="11785_HW4P2", entity="stirumal", config=args)
	train(args)
	