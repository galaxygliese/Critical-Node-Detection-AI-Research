#-*- coding:utf-8 -*-

from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
import torch 


class FeatureExtractorGAT(nn.Module):
    def __init__(
            self, 
            in_channels:int = 4, 
            hidden_channels:int = 60, 
            out_channels:int = 96, 
            num_heads:int = 8,
            dropout = 0.3
        ) -> None:
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels, hidden_channels, num_heads, dropout=dropout)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1,
                             concat=True, dropout=dropout)
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x # (node_num, out_channels)

class QNetwork(nn.Module):
    def __init__(
            self, 
            # action_dim:int, 
            output_dim:int,
            state_dim:int = 96, 
            hidden_dim:int = 96
        ):
        super().__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc_3 = nn.Linear(hidden_dim, action_dim)
        self.fc_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1        


if __name__ == '__main__':
    from torch_geometric.utils import to_networkx, from_networkx
    import matplotlib.pyplot as plt
    import networkx as nx


    node_num = 62
    mean_degree = 5
    beta = 0.3
    seed = 0
    max_distance = 2

    extractor_model = FeatureExtractorGAT()
    # qmodel = QNetwork(node_num)
    qmodel = QNetwork(1)

    def update_graph(G:nx.Graph, X) -> nx.Graph:
        G_ = nx.Graph()
        G_.add_nodes_from([
            (node_id, {'y': [0], 'x':X[node_id]})
            for node_id in G.nodes]
        )
        
        G_.add_edges_from(G.edges)
        return G_

    G = nx.watts_strogatz_graph(
        n=node_num, 
        k=mean_degree, 
        p=beta, 
        seed=seed
    )

    X = []

    nodes = list(G.nodes)
    node_degrees = dict(G.degree())
    node_degree_centralities = dict(nx.degree_centrality(G))
    node_centralities = dict(nx.eigenvector_centrality(G))
    node_pageranks = dict(nx.pagerank(G, alpha=0.85))
    for node_id in nodes:
        x = [node_degrees[node_id], node_degree_centralities[node_id], node_centralities[node_id], node_pageranks[node_id]]
        X.append(x)
        
    G = update_graph(G, X)

    data = from_networkx(G)
    x = data.x 
    edge_index = data.edge_index 

    h = extractor_model(x, edge_index)
    print("H >>", h.shape)

    # h = h.unsqueeze(0)
    # q = qmodel(h[:, :10, ...])
    q = qmodel(h).squeeze(1)
    print("Q >>", q.shape)