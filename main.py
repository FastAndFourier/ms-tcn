#!/usr/bin/python2.7

import torch
from model import Trainer
from torch.utils.data import TensorDataset, DataLoader
import os
import argparse
import random
import numpy as np
import wandb

import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sr', default=100, type=int, help="Sampling rate (if < 100, signal will be downsampled)")
parser.add_argument('--classes', choices=['binary', 'full', 'tri'], 
                    help="Binary: Background (including Other) and Action / Full: all labels / Tri: Background, Action, and Other")
parser.add_argument('--duration', default=10, type=int, help="Window duration in seconds")
parser.add_argument('--overlap', default=0, type=int, help="Overlap between windows (if 0, no sliding window)")
parser.add_argument('--modality', choices=["imu", "pressure", "both"], help="Input modalities")


data_path = "/media/isir/PHD/code/data_processing/Xml2Py/data/grasping"

sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "best_val_acc"},
    "parameters": {
        "lr": {"max": 0.001, "min": 0.0001},
        "num_stages":{"min":2, "max":5},
        "num_layers":{"min":2, "max":8},
        "num_f_maps":{"values":[16,32,64,128]},
        "lambda_":{"values":list(np.arange(0,2,0.2))},
        "duration":{"values":[30,60]},
        "sr":{"values":[50,100]}
    },
}

#sweep_id = wandb.sweep(sweep=sweep_configuration, project="ms-tcn")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


    run = wandb.init(project="ms-tcn")
    

    

    args = parser.parse_args()


    num_stages = 3
    num_layers = 6
    num_f_maps = 64
    lambda_ = 1
    sr = 100
    duration = 30
    overlap = 0 
    modality = "both" 
    classes = "tri" 
    lr = 0.0005

    # num_stages = wandb.config.num_stages
    # num_layers = wandb.config.num_layers
    # num_f_maps = wandb.config.num_f_maps
    # lambda_ = wandb.config.lambda_
    # sr = wandb.config.sr 
    # duration = wandb.config.duration
    # lr = wandb.config.lr
    # overlap = 0 
    # modality = "both" 
    # classes = "binary" 

    if modality == "both":
        features_dim = 7
    elif modality == "imu":
        features_dim = 6
    else:
        features_dim = 1

    bz = 4
    
    num_epochs = 100



    model_dir = "./models/ms_tcn/"
    model_id = np.random.randint(100000)
    run.name = f"{model_id}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    if classes == "binary":
        actions_dict = {"Background":0, "Action":1}
    elif classes == "full":
        actions_dict = {"Background":0, "GB":1, "GU":2, "Hand":3, "Hit":4, "Mouth":5, "Feet":6, "Other":7}
    elif classes == "tri":
        actions_dict = {"Background":0, "Action":1, "Other":2}




    num_classes = len(actions_dict)

    data_fname = f'{classes}_{sr}_{duration}_{overlap}_{modality}'


    X = np.load(f"{data_path}/data_{data_fname}.npy")
    y = np.load(f"{data_path}/label_{data_fname}.npy")
    split = np.load(f"{data_path}/split_{data_fname}.npy", allow_pickle=True)[()]


    X_train, y_train = X[split['train']], y[split['train']]
    X_test, y_test = X[split['test']], y[split['test']]
    X_val, y_val = X[split['val']], y[split['val']]


    trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes, model_id)
    trainer.model.to("cuda")

    dataset_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
    train_loader = DataLoader(dataset_train, batch_size=bz, shuffle=True, num_workers=4)

    dataset_test = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test))
    test_loader = DataLoader(dataset_train, batch_size=bz, shuffle=True, num_workers=4)

    dataset_val = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val))
    valid_loader = DataLoader(dataset_val, batch_size=bz, shuffle=True, num_workers=4)

    trainer.train(model_dir, train_loader, valid_loader, num_epochs, lr, device, lambda_)



main()

# wandb.agent(sweep_id, function=main, count=50)


