#-*- coding:utf-8 -*-

from datas.dataset import TerroristNetworkDataset, FullRaryTreeDataset

from models.maskgae.model import GNNEncoder, DegreeDecoder, EdgeDecoder, MaskGAE
from models.maskgae.mask import MaskPath, MaskEdge

from torch_geometric.data import Data
from typing import List
from tqdm import tqdm
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import numpy as np
import argparse
import random 
import torch 
import wandb
import os 


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="tree", type=str, help="Datasets.")
parser.add_argument('--seed', type=int, default=42, help='Random seed for model and dataset. (default: 42)')
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--weight_path', type=str)
parser.add_argument('--k', type=int, default=4, help='Number of Critical Nodes')

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

def save_critical_node_graph(
        G:nx.Graph, 
        critical_nodes:List[int]
    ):
    color_map = []
    for node_id in G.nodes:
        if node_id in critical_nodes:
            color_map.append("red")
        else:
            color_map.append("blue")
    plt.cla()
    nx.draw(
        G, 
        node_color=color_map,
        with_labels=True,
        node_size=[v * 10 for v in dict(G.degree).values()]
    )
    plt.savefig(f"results/output_critical_nodes_graphmae_{opt.dataset}.jpg")

def cosine_similarity(v1:np.ndarray, v2:np.ndarray):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    weight_path = opt.weight_path
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
    model.load_state_dict(torch.load(weight_path))
    print("Model Loaded!")

    edge_index = data.edge_index # (2, N_edge)
    x = data.x.float() # (N_node, D_node)

    z = model.encoder.get_embedding(x, edge_index, mode='last') # (N_node, 128)
    degrees_sorted = sorted(dataset.G.degree, key=lambda x: x[1], reverse=True)
    node_num = z.shape[0]
    print("Emb>>", z.shape)
    print("Degrees>>", degrees_sorted)

    embeddings = z.cpu().data.numpy()
    key_node_id = degrees_sorted[0][0]
    key_embedding = embeddings[key_node_id, ...]
    
    k = opt.k
    similarities = np.array([cosine_similarity(key_embedding, embeddings[i]) for i in range(node_num)])
    sorted_nodes = similarities.argsort()[-k:][::-1]
    print(sorted_nodes)

    save_critical_node_graph(dataset.G, critical_nodes=sorted_nodes)

if __name__ == '__main__':
    main()
