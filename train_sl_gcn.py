#-*- coding:utf-8 -*-

from datas.sl_dataset import DistributeSynthesisDataset
from models.sl.gcn_model import ResidualGatedGCNModelForCND
from torch.utils.data import DataLoader
from fastprogress import master_bar, progress_bar
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import argparse
import random
import torch 
import wandb
import json
import os 

EXPORT_DIR = "weights"

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed for model and dataset. (default: 42)')
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('--dataset_path', type=str, default="dataset/synth/")
parser.add_argument('--lr', type=float, default=0.001)
opt = parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Seeds set.")
seed_everything(opt.seed)

def main():
    device = 'cuda'
    dataset = DistributeSynthesisDataset(
        folder_path=opt.dataset_path
    )

    dataset_iter = iter(dataset)
    print("Dataset Size:", len(dataset))
    X, Y, A = dataset[0]
    print("X:", X.shape)
    print("A:", A.shape)
    print("Y:", Y.shape)

    model = ResidualGatedGCNModelForCND(
        node_dim=5
    ).to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        lr=opt.lr, 
        weight_decay=1e-6
    )
    print("Model Loaded!")

    batch_steps = len(dataset) // opt.batch_size
    model.train()

    run = wandb.init(project = 'cnd_sl_gcn')
    config = run.config
    config.epochs = opt.epochs
    config.batch_size = opt.batch_size
    config.dataset_size = len(dataset)

    dataset_indexes = list(range(len(dataset)))

    for epoch in range(opt.epochs):
        random.shuffle(dataset_indexes)
        dataset_count = 0
        pbar = tqdm(total=batch_steps)
        for batch_step_num in range(batch_steps):
            batch_loss = []

            optimizer.zero_grad()
            for batch_num in range(opt.batch_size):
                X, Y, A = dataset[dataset_indexes[dataset_count]]
                dataset_count += 1
                X = X.unsqueeze(0).to(device)
                Y = Y.unsqueeze(0).to(device)
                A = A.unsqueeze(0).to(device)

                Y_pred = model(X, A).squeeze(2)
                loss = F.binary_cross_entropy(Y_pred, Y)
                batch_loss.append(loss)
            
            losses = torch.mean(torch.stack(batch_loss))
            losses.backward()
            optimizer.step()
            pbar.update(1)
            run.log({
                "train_batch_loss": losses.cpu().detach().numpy()
            })
        pbar.close()

    save_path =  os.path.join(EXPORT_DIR, 'sl_gcn_model_' + f'epoch{opt.epochs}.pth')
    torch.save(model.state_dict(), save_path)
    print("Done!")

if __name__ == '__main__':
    main()
