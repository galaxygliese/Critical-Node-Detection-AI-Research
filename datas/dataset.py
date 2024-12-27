#-*- coding:utf-8 -*-

from scipy.spatial import distance
from torch_geometric.data import Data, InMemoryDataset
from typing import Any, List
import networkx as nx 
import pandas as pd
import numpy as np
import torch
import csv
import os

def train_val_test_split(
        data:Data, 
        val_ratio: float = 0.15,
        test_ratio: float = 0.15):
    rnd = torch.rand(len(data.x))
    train_mask = [False if (x > val_ratio + test_ratio) else True for x in rnd]
    val_mask = [False if (val_ratio + test_ratio >= x) and (x > test_ratio) else True for x in rnd]
    test_mask = [False if (test_ratio >= x) else True for x in rnd]
    return torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask)

class JapanCitiesDataset(InMemoryDataset):
    def __init__(self, df_path:str, transform = None):
        super().__init__('.', transform)
        self.df = pd.read_csv(df_path, header=None, sep="\t")
        self.embeddings = torch.tensor(self.df.iloc[:, [-1, -2]].values, dtype=torch.float) 
        self.ys = self.get_labels() 
        self.edges = []
        self.edge_attr = []
        self.get_edges()

        data = Data(
            x=self.embeddings, # (N_node, D)
            edge_index=self.edges, # (N_edge, 2)
            y=self.ys, # (N_node, )
            edge_attr=self.edge_attr # (N_edge, D_edge)
        )
        self.data, self.slices = self.collate([data])
        print("Dataset Loaded!")

    def get_edges(self, top:int = 8):
        self.edges = []
        self.edge_attr = []
        adj_matrix = []
        location = self.df.iloc[:, -2:].values
        dist_matrix = distance.cdist(location, location, metric='euclidean')
        for idxs in np.argsort(dist_matrix)[:, :top]:
            adj_matrix.append([1 if i in idxs else 0 for i in range(len(dist_matrix))])

        adj_matrix = np.array(adj_matrix)

        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if i == j:
                    continue
                elif adj_matrix[i][j] == 1:
                    self.edges.append([i, j])
                    self.edge_attr.append(dist_matrix[i][j])

    def get_labels(self):
        name2id = {}
        id2name = {}
        for v in self.df.iloc[:, 1].values:
            if v not in name2id.keys():
                name2id[v] = len(name2id.keys())
                id2name[name2id[v]] = v
        ys = torch.tensor([name2id[v] for v in self.df.iloc[:, 1].values], dtype=torch.long)
        return ys
    

class TerroristNetworkDataset(InMemoryDataset):
    def __init__(self, 
                 folder_path:str, 
                 embeddings:Any = None, # [WIP]
                 labels: Any = None, # [WIP]
                 transform = None
        ):
        super().__init__('.', transform)
        self.folder_path = folder_path
        self.G = self.load_graph()

        if embeddings is None:
            self.embeddings = torch.nn.functional.one_hot(torch.arange(0, len(self.G.nodes)), num_classes=len(self.G.nodes))
        else:
            self.embeddings = embeddings

        if labels is None:
            self.ys = torch.zeros(len(self.G.nodes))
        else:
            self.ys = labels
        self.edges = []
        self.edge_attr = []

        data = Data(
            x=self.embeddings, # (N_node, D)
            edge_index=self.edges, # (N_edge, 2)
            y=self.ys, # (N_node, )
            edge_attr=self.edge_attr # (N_edge, D_edge)
        )
        self.data, self.slices = self.collate([data])
        print("Dataset Loaded!")

    def load_graph(self) -> nx.Graph:
        nodes_file = os.path.join(self.folder_path, 'nodes.csv')
        edge_file = os.path.join(self.folder_path, 'edges.csv')

        node_ids = []
        G = nx.Graph()
        with open(nodes_file) as file:
            reader = csv.reader(file)
            for i,row in enumerate(reader):
                if i > 0:
                    node_ids.append(int(row[0]))
        G.add_nodes_from(node_ids)

        edges = []
        with open(edge_file) as file:
            reader = csv.reader(file)
            for i,row in enumerate(reader):
                if i > 0:
                    edge = row
                    edges.append((int(edge[0]), int(edge[1])))
        G.add_edges_from(edges)
        return G


if __name__ == '__main__':
    # dataset = JapanCitiesDataset(df_path="dataset/japan_cities/japan_cities.csv")
    dataset = TerroristNetworkDataset(folder_path="dataset/9-11_terrorists")
    data = dataset[0]
    print("Length>", len(dataset))
    x = data.x 
    y = data.y

    print("X >>", x.shape)
    print("Y >>", y.shape)
    print(x[0])