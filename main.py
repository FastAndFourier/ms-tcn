#!/usr/bin/python2.7

import torch
from ms_tcn.model import Trainer
from torch.utils.data import Dataset, DataLoader
# from batch_gen import BatchGenerator
import os
import argparse
import random
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--split', default='1')

args = parser.parse_args()

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1
lr = 0.0005
num_epochs = 50

# use the full temporal resolution @ 100 Hz
sample_rate = 1


model_dir = "./models/ms_tcn/split_"+args.split
results_dir = "./results/ms_tcn/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


actions_dict = {"Background":0, "Action":1}

num_classes = len(actions_dict)

trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)
X = np.load("data_binary.npy")
y = np.load("label_binary.npy")



# if args.action == "train":
#     batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
#     batch_gen.read_data(vid_list_file)
#     trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

# if args.action == "predict":
#     trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
