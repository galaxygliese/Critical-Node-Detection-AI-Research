#-*- coding:utf-8 -*-

from datas.dataset import TerroristNetworkDataset, FullRaryTreeDataset
from datas.dataset import train_val_test_split

from models.maskgae.model import GNNEncoder, DegreeDecoder, EdgeDecoder, MaskGAE
from models.maskgae.mask import MaskPath, MaskEdge

from typing import Literal
from torch_geometric.data import Data
from tqdm import tqdm
import torch_geometric.transforms as T
import numpy as np
import argparse
import random 
import torch 
import wandb
import os 

DatasetTypes = Literal[
    'terrorist',
    'tree',
]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="tree", type=str, help="Datasets.")
parser.add_argument('--seed', type=int, default=42, help='Random seed for model and dataset. (default: 42)')
parser.add_argument('--dataset_path', type=str)

parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('-b', '--batch_size', type=int, default=2**16, help='Number of batch size. (default: 2**16)')
parser.add_argument('-e', '--epochs', type=int, default=500, help='Number of training epochs. (default: 300)')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=128, help='Channels of hidden representation. (default: 128)')
parser.add_argument('--decoder_channels', type=int, default=64, help='Channels of decoder. (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers of encoder. (default: 1)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.7)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.3)')
parser.add_argument('--alpha', type=float, default=0.003, help='loss weight for degree prediction. (default: 2e-3)')

parser.add_argument("--start", nargs="?", default="edge", help="Which Type to sample starting nodes for random walks, (default: edge)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument("--device", type=int, default=0)
opt = parser.parse_args()

EXPORT_DIR = "weights"

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

def train(model:MaskGAE, data:Data, opt, device="cpu"):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )
    
    for epoch in tqdm(range(1, 1 + opt.epochs)):
        loss = model.train_step(
            data, 
            optimizer,
            alpha=opt.alpha, 
            batch_size=opt.batch_size
        )

        # print("Loss >>", loss)
    
    print("Last loss>>", loss)
    save_path = os.path.join(EXPORT_DIR, 'maskgae_' + opt.dataset + f'_epoch{epoch}.pth')
    torch.save(model.state_dict(), save_path)



def main():
    device = opt.device
    dataset_type = opt.dataset 
    dataset_path = opt.dataset_path 

    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ]) 

    if dataset_type == 'terrorist':
        dataset = TerroristNetworkDataset(folder_path=dataset_path, transform=transform)

    elif dataset_type == 'tree':
        node_num:int = 62
        dataset = FullRaryTreeDataset(node_num=node_num, transform=transform)

    else:
        raise NotImplementedError
    
    data = dataset[0]

    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0, 
        num_test=0.1,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False)(data)
    
    print("Train Datas>", train_data.x.shape, train_data.edge_index.shape)
    # print("Test Datas>", test_data.edge_index)

    # mask = MaskPath(
    #     p=opt.p, 
    #     num_nodes=data.num_nodes, 
    #     start=opt.start,
    #     walk_length=opt.encoder_layers+1
    # )
    mask = MaskEdge(p=opt.p)

    encoder = GNNEncoder(
        data.num_features, 
        opt.encoder_channels, 
        opt.hidden_channels,
        num_layers=opt.encoder_layers, 
        dropout=opt.encoder_dropout,
        bn=opt.bn, 
        layer=opt.layer, 
        activation=opt.encoder_activation
    )

    edge_decoder = EdgeDecoder(
        opt.hidden_channels, 
        opt.decoder_channels,
        num_layers=opt.decoder_layers, 
        dropout=opt.decoder_dropout
    )

    degree_decoder = DegreeDecoder(
        opt.hidden_channels, 
        opt.decoder_channels,
        num_layers=opt.decoder_layers, 
        dropout=opt.decoder_dropout
    )

    model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)
    print("Model Loaded!")

    train(model, train_data, opt, device=device)
    print("Done!")


if __name__ == '__main__':
    main()
