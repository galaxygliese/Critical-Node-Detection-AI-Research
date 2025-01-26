#-*- coding:utf-8 -*-

from typing import List, Set
import networkx as nx
import torch 

def calc_param_connected_components(G:nx.Graph) -> float:
    C:List[Set[int]] = [c for c in nx.connected_components(G)]
    connected_components = 0
    for c in C:
        sigma = 0.5 * len(c) * (len(c) - 1)
        connected_components += sigma 
    return connected_components

def critical_nodes_reward(G_init:nx.Graph, G_current:nx.Graph) -> float:
    C0 = calc_param_connected_components(G_init)
    Ct = calc_param_connected_components(G_current)
    r = C0 - Ct
    return r

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # G = nx.path_graph(4)
    # Gt = G.copy()
    # nx.add_path(G, [10, 11, 12])
    G = nx.watts_strogatz_graph(n=10, k=5, p=0.2, seed=0)
    Gt = G.copy()
    nx.add_path(G, [10, 11, 12, 13])
    r = critical_nodes_reward(G, Gt)
    print(r)

    plt.subplot(1,2,1)
    nx.draw(G)

    plt.subplot(1,2,2)
    nx.draw(Gt)
    plt.show()
    
