#-*- coding:utf-8 -*-

from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
from typing import List, Tuple, Dict
import networkx as nx
import random
import torch 


def get_subgraph(graph_data:Data, max_distance:int=2) -> nx.Graph:
    G:nx.Graph = to_networkx(graph_data)
    node_num = len(G.nodes) 
    neighbors:List[int] = []
    root_node:int = 0

    # To avoid len(neighbors) == 0 condition.
    while len(neighbors) == 0:
        root_node = random.choice(list(G.nodes))
        neighbors = list(G.neighbors(root_node))
    
    sub_G = get_subgraph_by_distance(G, root_node, distance=max_distance)
    return sub_G

def get_subgraph_by_distance(G: nx.Graph, root_node: int, distance: int) -> nx.Graph:
    """
    Creates a subgraph of G containing nodes within a certain distance from a root node.

    Args:
        G: The input graph.
        root_node: The starting node for distance calculation.
        distance: The maximum distance from the root node.

    Returns:
        A subgraph of G containing nodes within the specified distance, or an empty graph if the root node is not in G.
        Returns None if the input graph is not a networkx Graph object.
    """

    if not isinstance(G, nx.Graph):
        return None

    if root_node not in G:
        return nx.Graph()  # Return an empty graph if the root node is not found

    nodes_in_subgraph = set()
    nodes_to_explore = {root_node}
    explored_nodes = set()

    for _ in range(distance + 1):  # Iterate up to the maximum distance + 1 to include nodes at exactly the distance
        new_nodes_to_explore = set()
        for node in nodes_to_explore:
            if node not in explored_nodes:
                nodes_in_subgraph.add(node)
                neighbors = G.neighbors(node)
                new_nodes_to_explore.update(neighbors)
                explored_nodes.add(node)
        nodes_to_explore = new_nodes_to_explore

    sub_G = G.subgraph(nodes_in_subgraph).copy().to_undirected() # Use .copy() to create an independent subgraph
    return sub_G

def get_rearrenged_graph(sub_G:nx.Graph, plot:bool=False) -> Tuple[nx.Graph, Dict[int, int]]:
    node_ids = list(sub_G.nodes)
    new_node_ids = list(range(len(node_ids)))
    old2new = {i:j for i,j in zip(node_ids, new_node_ids)}

    new_sub_G = nx.Graph()
    new_sub_G.add_nodes_from([
        (old2new[node_id])
        for node_id in sub_G.nodes]
    )
    
    new_edges = []
    for edge in sub_G.edges:    
        new_edges.append((old2new[edge[0]], old2new[edge[1]]))    
    new_sub_G.add_edges_from(new_edges)

    if plot:
        print("Is isometric>>", nx.is_isomorphic(sub_G, new_sub_G))
        plt.subplot(1,2,1)
        plt.cla()
        plt.title("Sub Graph")
        nx.draw_networkx(sub_G)
        plt.subplot(1,2,2)
        plt.cla()
        plt.title("Re-Arrenged")
        nx.draw_networkx(new_sub_G)
        plt.show()
    return new_sub_G, old2new

def get_sub_graph_data(
        G:nx.Graph, # new_sub_G
        z:torch.Tensor, # (node_num, hidden_dim)
        y:torch.Tensor, # (node_num)
        old2new_nodes:dict) -> Data:

    new_G = nx.Graph()
    new_G.add_nodes_from([
        (new_node_id, {'y': y[new_node_id].tolist(), 'x':z[node_id].tolist()})
        for node_id, new_node_id in old2new_nodes.items()]
    )
    new_G.add_edges_from(G.edges)
    data = from_networkx(new_G)
    return data

def critical_nodes_to_binary_tensor(critical_nodes:List[int], node_num:int) -> torch.Tensor:
    y = torch.zeros(node_num)
    for critical_node in critical_nodes:
        y[critical_node] += 1
    return y

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    node_num = 62
    mean_degree = 5
    beta = 0.3
    seed = 0
    max_distance = 2

    G = nx.watts_strogatz_graph(
        n=node_num, 
        k=mean_degree, 
        p=beta, 
        seed=seed
    )

    data = from_networkx(G)
    sub_G = get_subgraph(graph_data=data, max_distance=max_distance)

    print("Sub Nodes>>", sub_G.nodes, len(sub_G.nodes))
    print("Edges>>", sub_G.edges)
    print("Adj sub graph>>", nx.adj_matrix(sub_G).toarray().shape)

    # plt.cla()
    # nx.draw_networkx(sub_G)
    # plt.show()

    sub_G, old2new_nodes = sub_data = get_rearrenged_graph(sub_G, plot=True)

    z = torch.zeros((node_num, 128))
    y = torch.zeros((node_num,))
    sub_data = get_sub_graph_data(sub_G, z, y, old2new_nodes)

    print(">>", sub_data.x.shape, sub_data.y.shape)

    recon_G = to_networkx(sub_data).to_undirected()
    plt.title("Reconstructed")
    nx.draw_networkx(recon_G)
    plt.show()


