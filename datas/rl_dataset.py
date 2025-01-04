#-*- coding:utf-8 -*-

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils.convert import from_networkx
from typing import Any, List, Literal
import networkx as nx 
import random

class SynthesisGraphDataset(InMemoryDataset):
    def __init__(self, 
            max_node_num:int = 150,
            min_node_num:int = 50,
            transform = None
        ):
        super().__init__()
        self.max_node_num = max_node_num
        self.min_node_num = min_node_num
        self.graph_types: List[str] = ["erdos_renyi", "watts_strogatz", "barabasi_albert"]
        self.transform = transform

    def generate_graph(self, graph_type:str, node_num:int, seed:int) -> nx.Graph:
        if graph_type == 'erdos_renyi':
            p = random.random() * (0.3 - 0.1) + 0.1
            G = nx.erdos_renyi_graph(n=node_num, p=p, seed=seed)
            return G
        elif graph_type == 'watts_strogatz':
            p = random.random() * (0.3 - 0.1) + 0.1
            mean_degree = int(p*node_num)
            G = nx.watts_strogatz_graph(n=node_num, k=mean_degree, p=p, seed=seed)
            return G
        elif graph_type == 'barabasi_albert':
            m = random.randint(1, node_num - 1)
            G = nx.barabasi_albert_graph(n=node_num, m=m, seed=seed)
            return G
        else:
            raise NotImplementedError
    
    def update_graph(self, G:nx.Graph, X:List[List[int]]) -> nx.Graph:
        G_ = nx.Graph()
        G_.add_nodes_from([
            (node_id, {'y': [0], 'x':X[node_id]})
            for node_id in G.nodes]
        )
        
        G_.add_edges_from(G.edges)
        return G_

    def get_data(self, seed:int) -> Data:
        node_num = random.randint(self.min_node_num, self.max_node_num)
        graph_type = random.choice(self.graph_types)
        
        G = self.generate_graph(graph_type, node_num, seed=seed)
        X: List[List[float]] = []

        nodes = list(G.nodes)
        node_degrees = dict(G.degree())
        node_degree_centralities = dict(nx.degree_centrality(G))
        node_centralities = dict(nx.eigenvector_centrality(G))
        node_pageranks = dict(nx.pagerank(G, alpha=0.85))
        for node_id in nodes:
            x = [node_degrees[node_id], node_degree_centralities[node_id], node_centralities[node_id], node_pageranks[node_id]]
            X.append(x)
        
        G = self.update_graph(G, X)
        data : Data = from_networkx(G)
        
        if self.transform is not None:
            data = self.transform(data)
        return data

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
    dataset = SynthesisGraphDataset(min_node_num=4, max_node_num=10, transform=transform)
    data = dataset.get_data(seed=seed)

    x = data.x 
    y = data.y

    print("X >>", x.shape)
    print("Edges >>", data.edge_index.shape)

    G = to_networkx(data, to_undirected=True)
    nx.draw(G)
    plt.show()