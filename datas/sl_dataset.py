#-*- coding:utf-8 -*-

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils.convert import from_networkx
from torch.utils.data import Dataset
from typing import Any, List, Tuple, Optional
from glob import glob 
from tqdm import tqdm
import networkx as nx 
import numpy as np
import random
import torch
import os

class SynthesisGraphDataset(InMemoryDataset):
    def __init__(self, 
            max_node_num:int = 150,
            min_node_num:int = 50,
            max_remove_node_num:int = 30,
            transform = None
        ):
        super().__init__()
        self.max_node_num = max_node_num
        self.min_node_num = min_node_num
        self.max_remove_node_num = max_remove_node_num
        self.graph_types: List[str] = ["erdos_renyi", "watts_strogatz", "barabasi_albert"]
        self.transform = transform

    def generate_graph(self, graph_type:str, node_num:int, seed:int) -> nx.Graph:
        if graph_type == 'erdos_renyi':
            p = random.random() * (0.5 - 0.1) + 0.1
            G = nx.erdos_renyi_graph(n=node_num, p=p, seed=seed)
            return G
        elif graph_type == 'watts_strogatz':
            p = random.random() * (0.3 - 0.1) + 0.1
            mean_degree = max(int(p*node_num), 2)
            G = nx.watts_strogatz_graph(n=node_num, k=mean_degree, p=p, seed=seed)
            return G
        elif graph_type == 'barabasi_albert':
            m = random.randint(1, node_num - 1)
            G = nx.barabasi_albert_graph(n=node_num, m=m, seed=seed)
            return G
        else:
            raise NotImplementedError
    
    def update_graph(self, G:nx.Graph, X:List[List[int]], y:Optional[List[int]] = None) -> nx.Graph:
        if y is None:
            y = [0] * len(G.nodes)
        G_ = nx.Graph()
        G_.add_nodes_from([
            (node_id, {'y': y[node_id], 'x':X[node_id]})
            for node_id in G.nodes]
        )
        
        G_.add_edges_from(G.edges)
        return G_

    def get_data(self, seed:int) -> Data:
        node_num = random.randint(self.min_node_num, self.max_node_num)
        if seed % 3 == 0:
            graph_type = self.graph_types[0]
        elif seed % 3 == 1:
            graph_type = self.graph_types[1]
        else:
            graph_type = self.graph_types[2]
        
        G = self.generate_graph(graph_type, node_num, seed=seed)
        X: List[List[float]] = []

        nodes = list(G.nodes)
        node_degrees = dict(G.degree())
        node_degree_centralities = dict(nx.degree_centrality(G))
        node_centralities = dict(nx.eigenvector_centrality(G, max_iter=400))
        node_pageranks = dict(nx.pagerank(G, alpha=0.85, max_iter=500))
        for node_id in nodes:
            x = [node_degrees[node_id], node_degree_centralities[node_id], node_centralities[node_id], node_pageranks[node_id]]
            X.append(x)
        
        G = self.update_graph(G, X)
        data : Data = from_networkx(G)
        
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def get_all_data(self, seed:int) -> Tuple[Data, nx.Graph, int]:
        node_num = random.randint(self.min_node_num, self.max_node_num)
        K = random.randint(1, self.max_remove_node_num)

        if seed % 3 == 0:
            graph_type = self.graph_types[0]
        elif seed % 3 == 1:
            graph_type = self.graph_types[1]
        else:
            graph_type = self.graph_types[2]
        
        G = self.generate_graph(graph_type, node_num, seed=seed)
        X: List[List[float]] = []

        nodes = list(G.nodes)
        node_degrees = dict(G.degree())
        node_degree_centralities = dict(nx.degree_centrality(G))
        node_centralities = dict(nx.eigenvector_centrality(G, max_iter=400))
        node_pageranks = dict(nx.pagerank(G, alpha=0.85, max_iter=500))
        for node_id in nodes:
            x = [node_degrees[node_id], node_degree_centralities[node_id], node_centralities[node_id], node_pageranks[node_id]]
            X.append(x)
        
        G = self.update_graph(G, X)
        data : Data = from_networkx(G)
        
        if self.transform is not None:
            data = self.transform(data)
        return data, G, K

class DistributeSynthesisDataset(Dataset):
    def __init__(self, folder_path:str, append_k:bool=True):
        super().__init__()
        self.folder_path = folder_path 
        self.append_k = append_k
        self.datas = self.load_datas()

    def load_datas(self):
        graph_files = np.sort(glob(os.path.join(self.folder_path, '*.gpickle')))
        samples = []
        for graph_file in tqdm(graph_files):
            G = nx.read_gpickle(graph_file)
            data = from_networkx(G)
            X = data.x.tolist()
            Y = data.y.tolist()
            K = np.sum(Y) # remove number
            A = nx.adjacency_matrix(G).toarray()
            sample = {
                'X':X,
                'Y':Y,
                'A':A,
                'K':K
            }
            samples.append(sample)
        return samples 
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index) -> Any:
        data = self.datas[index]
        X = data['X']
        Y = data['Y']
        A = data['A']
        K = data['K']
        if self.append_k:
            X_ = []
            for x in X:
                x_ = x.copy()
                x_.append(K)
                X_.append(x_)
            X = X_
        
        X_tensor = torch.Tensor(X)
        Y_tensor = torch.Tensor(Y) #.long()
        A_tensor = torch.Tensor(A)
        return X_tensor, Y_tensor, A_tensor



if __name__ == '__main__':
    from torch_geometric.utils import to_networkx
    import torch_geometric.transforms as T
    import matplotlib.pyplot as plt 

    device = 'cpu'
    seed = 0

    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ]) 
    dataset = SynthesisGraphDataset(min_node_num=6, max_node_num=12, transform=transform)
    data = dataset.get_data(seed=seed)

    x = data.x 
    y = data.y

    print("X >>", x.shape)
    print("Edges >>", data.edge_index.shape)

    G = to_networkx(data, to_undirected=True)
    nx.draw(G)
    plt.show()