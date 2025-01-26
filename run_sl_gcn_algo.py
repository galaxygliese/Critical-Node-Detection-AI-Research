#-*- coding:utf-8 -*-

from datas.sample_from_csv import SampleGraphFromCSV
from models.sl.gcn_model import ResidualGatedGCNModelForCND
from typing import List
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random 
import torch 
import time
import sys 
import os 

seed = 42
dataset_type = sys.argv[1]

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Seeds set.")
seed_everything(seed)

def create_node_vector(G:nx.Graph, K:int):
    X = []
    nodes = list(G.nodes)
    node_degrees = dict(G.degree())
    node_degree_centralities = dict(nx.degree_centrality(G))
    node_centralities = dict(nx.eigenvector_centrality(G, max_iter=400))
    node_pageranks = dict(nx.pagerank(G, alpha=0.85, max_iter=500))
    for node_id in nodes:
        x = [node_degrees[node_id], node_degree_centralities[node_id], node_centralities[node_id], node_pageranks[node_id], K]
        X.append(x)
    return X

def save_critical_node_graph(G:nx.Graph, critical_nodes:List[int]):
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
    plt.savefig(f"results/output_critical_nodes_{dataset_type}_sl_gcn.jpg")

def main():
    K = 7
    if dataset_type == 'terrorist':
        sampler = SampleGraphFromCSV("dataset/9-11_terrorists")
        sampler.save_graph()
        G = sampler.G

    elif dataset_type == 'erdos_renyi':
        node_num:int = 30#62 
        p:float = 0.3
        G = nx.erdos_renyi_graph(n=node_num, p=p, seed=seed)

    elif dataset_type == 'watts_strogatz':
        node_num:int = 30#62 
        k:int = 5 # mean_degree
        beta:float = 0.1
        G = nx.watts_strogatz_graph(n=node_num, k=k, p=beta, seed=seed)
    device = 'cuda'
    weight_path = 'weights/sl_gcn_model_epoch100.pth'
    model = ResidualGatedGCNModelForCND(
        node_dim=5
    ).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    print("Model Loaded!")

    X = create_node_vector(G, K)
    A = nx.adjacency_matrix(G).toarray()
    

    X_tensor = torch.Tensor(X).unsqueeze(0).to(device)
    A_tensor = torch.Tensor(A).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        pred = model(X_tensor, A_tensor).squeeze(2).squeeze(0)
        print(pred)
        pred_topk = pred.topk(k=K).indices
        critical_nodes = pred_topk.cpu().tolist()
    end_time = time.time()
    print("Supervised GCN Time (sec): ", end_time - start_time)
    print(critical_nodes)
    save_critical_node_graph(G, critical_nodes)
    

if __name__ == '__main__':
    main()

