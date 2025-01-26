#-*- coding:utf-8 -*-

from datas.sl_dataset import SynthesisGraphDataset
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
from typing import List
from methods.bicnd import bicnp
from methods.cnp1 import cnp1
from tqdm import tqdm 
from glob import glob
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import random
import torch 
import json
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed for model and dataset. (default: 42)')
parser.add_argument('-n', '--dataset_num', type=int, default=1000)
parser.add_argument('--solver', type=str, default="cnp1")
parser.add_argument('--min_node_num', type=int, default=20)
parser.add_argument('--max_node_num', type=int, default=60)
parser.add_argument('--max_remove_num', type=int, default=15)
opt = parser.parse_args()

EXPORT_DIR = "dataset/synth"

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
    dataset_num = opt.dataset_num
    dataset = SynthesisGraphDataset(
        min_node_num=opt.min_node_num, 
        max_node_num=opt.max_node_num, 
        max_remove_node_num=opt.max_remove_num,
        transform=None
    )
    print("Selected solver:", opt.solver)

    data_seeds = list(range(dataset_num))
    

    for i in tqdm(range(dataset_num)):
        seed = data_seeds[i]
        data, G, K = dataset.get_all_data(seed)

        if opt.solver == 'cnp1':
            critical_nodes = cnp1(G, K=K)
        else:
            critical_nodes = bicnp(G, K=K)

        y = [0] * len(G.nodes)
        for node_id in critical_nodes:
            y[node_id] += 1
        
        G_with_label = dataset.update_graph(G, data.x, y)
        graph_name = str(i) + ".gpickle"
        nx.write_gpickle(G_with_label, os.path.join(EXPORT_DIR, graph_name))

        # break

    print("Done!")

def plot_critical_node_graph(G:nx.Graph, critical_nodes:List[int]):
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
    plt.show()

def check():
    graph_files = glob(os.path.join(EXPORT_DIR, '*.gpickle'))
    for graph_file in graph_files:
        G = nx.read_gpickle(graph_file)
        break
    
    data = from_networkx(G)
    print("X>>", data.x)
    print("Y>>", data.y)
    critical_nodes = [i for i, label in enumerate(data.y) if label == 1]

    plot_critical_node_graph(G, critical_nodes)

if __name__ == '__main__':
    main()
    # check()
