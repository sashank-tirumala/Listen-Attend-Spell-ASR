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
l2i, i2l = create_dictionaries(LETTER_LIST)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_attention(attention):
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.savefig("attention.png")

def train(model, criterion, train_loader, optimizer, i_ini, scheduler, using_wandb=False, tf=True):
	model.train()
	if not using_wandb:
		batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, desc='Train')
	total_loss = 0
	for i, (x,y,lx,ly) in enumerate(train_loader):
		x = x.to(device)
		y = y.to(device)
		predictions, attentions = model.forward(x, lx, y, mode="train", teacher_forcing=tf)
		mask = torch.arange(ly.max()).unsqueeze(0) >= ly.unsqueeze(1)
		mask = mask.view(-1).to(device)
		loss = criterion(predictions.view(-1, len(LETTER_LIST)), y.view(-1))
		loss = loss.masked_fill_(mask, 0)
		loss = torch.sum(loss)/mask.sum()
		total_loss += loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if(using_wandb):
			wandb.log({"loss":float(total_loss / (i + 1)), "step":int(i_ini), 'lr': float(optimizer.param_groups[0]['lr'])})
		else:
			batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))))
			batch_bar.update()
		if(scheduler is not None):
			scheduler.step()
		i_ini += 1
	if(using_wandb):
		plot_attention(attentions)
		wandb.log("attention: ", wandb.Image("attention.png"))
	else:
		plot_attention(attentions)
		batch_bar.close()
	return i_ini, float(total_loss / (i + 1)

def get_model(cfg):
	model = Seq2Seq(input_dim = x.shape[-1], 
	encoder_hidden_dim = cfg["encoder_dim"], 
	decoder_hidden_dim = cfg["decoder_dim"], 
	vocab_size=len(LETTER_LIST), 
	embed_dim=cfg["embed_dim"],
	key_value_size=cfg["key_value_size"],
	num_layers=cfg["num_layers_encoder"], 
	num_decoder_layers=cfg["num_layers_decoder"], 
	attention=cfg["attention_type"]
	).to(device)
	return model

def dataloader(cfg):
	f(cfg["simple"]):
		train_loader, val_loader = get_simple_dataloader(cfg["datapath"], batch_size = cfg["batch_size"])
		return train_loader, val_loader, None
	else:
		train_loader, val_loader, test_loader = get_dataloader(cfg["datapath"], batch_size=cfg["batch_size"])
		return train_loader, val_loader, test_loader
def training(cfg):
	model = get_model(cfg)
	optimizer = optim.Adam(model.parameters(), lr = cfg["lr"], weight_decay=cfg["w_decay"])
	train_loader, val_loader, test_loader = dataloader(cfg)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs']*len(train_loader), eta_min=1e-6, last_epoch=- 1, verbose=False)
	criterion = nn.CrossEntropyLoss(reduction='none')
	n_epochs = cfg["epochs"]
	i_ini = 0
	for epoch in range(n_epochs):
		i_ini, loss = train(model, criterion, train_loader, optimizer, i_ini, scheduler=None, using_wandb = cfg["wandb"], tf = get_teacher_forcing(epoch))
		val(model, val_loader, using_wandb = cfg["wandb"], epoch)
		save_model(model, optimizer, scheduler, loss,  cfg)

def save_model(model, optimizer, scheduler, loss,  cfg):
	torch.save({'epoch': epoch, 
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'scheduler_state_dict':scheduler.state_dict(),
				'loss':loss,
				'cfg':cfg
				}, cfg["runspath"]+"/"+"ckpt")

def load_model(path):
	checkpoint = torch.load(path):
	model = get_model(checkpoint["cfg"])
	optimizer = optim.Adam(model.parameters(), lr = cfg["lr"], weight_decay=cfg["w_decay"])
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs']*len(train_loader), eta_min=1e-6, last_epoch=- 1, verbose=False)
	model.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
	return model, optimizer, scheduler
def val(model, val_loader, using_wandb, epoch):
	model.eval()
	l2i, i2l = create_dictionaries(LETTER_LIST)
	dists = []
	print("len: ", len(val_loader))
	total_loss = 0
	for i, data in enumerate(val_loader):
		x, y, lx, ly = data
		x = x.to(device)
		y = y.to(device)
		predictions, attentions = model(x, lx, y, mode="train")
		mask = torch.arange(ly.max()).unsqueeze(0) >= ly.unsqueeze(1)
		mask = mask.view(-1).to(device)
		loss = criterion(predictions.view(-1, len(LETTER_LIST)), y.view(-1))
		loss = loss.masked_fill_(mask, 0)
		loss = torch.sum(loss)/mask.sum()
		total_loss += loss
		greedy_pred = torch.max(predictions, dim=2)[1]
		dists.append(get_dist(greedy_pred, y))
	dists = np.array(dists)
	lev_distance = np.mean(dists)
	if(using_wandb):
		wandb.log({"val_loss": total_loss, "epoch":epoch, "lev_distance":lev_distance})
	else:
		print("lev_distance: ",lev_distance )


def get_dist(greedy_pred, y):
	dist = 0
	for b in range(y.shape[1]):
		target_str = "".join(i2l[int(x)] for x in y[:,b])
		pred_str = "".join(i2l[int(x)] for x in greedy_pred[b,:])
		target_str = target_str.split('<eos>')[0][:]
		pred_str = pred_str.split('<eos>')[0][:]
		dist = dist + lev(target_str, pred_str)
	return dist/y.shape[1]

if(__name__ == "__main__"):
	torch.manual_seed(11785)
	torch.cuda.manual_seed(11785)
	np.random.seed(11785)
	random.seed(11785)
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-lr','--lr', type=float, help='learning rate', default = 1e-3) 
	parser.add_argument('-wd','--w_decay', type=float, help='weight decay (regularization)', default=0) 
	parser.add_argument('-bs','--batch_size', type=int, help='Description for bar argument', default=8)
	parser.add_argument('-e','--epochs', type=int, help='Description for bar argument', default=50)
	parser.add_argument('-dp','--datapath', type=str, help='Description for bar argument', default="hw4p2_student_data/hw4p2_student_data")
	parser.add_argument('-rp','--runspath', type=str, help='Description for bar argument', default="")
	parser.add_argument('-t','--transform', type=bool, help='Description for bar argument', default=True)
	parser.add_argument('-nl','--num_layers_encoder', type=int, help='Number of layers in encoder only', default=3)
	parser.add_argument('-nld','--num_layers_decoder', type=int, help='Number of layers in decoder only', default=2)
	parser.add_argument('-ap','--attention_type', type=str, help='type of attention used', default="single")
	parser.add_argument('-ed','--encoder_dim', type=int, help='Dimensionality of encoder only', default=256)
	parser.add_argument('-dd','--decoder_dim', type=int, help='Dimensionality of decoder only', default=256)
	parser.add_argument('-ebd','--embed_dim', type=int, help='Number of layers in encoder only', default=128),
	parser.add_argument('-kvs','--key_value_size', type=int, help='Number of layers in encoder only', default=128)
	parser.add_argument('-sim','--simple', type=bool, help='use simple dataset', default=False)
	parser.add_argument('-w','--wandb', type=bool, help='determines if Wandb is to be used', default=False)

	args = vars(parser.parse_args())
	import os 
	if not os.path.isdir(args["runspath"]):
		os.makedirs(args["runspath"])

	wandb.init(project="11785_HW4P2", entity="stirumal", config=args)
	train(args)
	