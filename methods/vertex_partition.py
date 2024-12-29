#-*- coding:utf-8 -*-
# Paper: https://www.researchgate.net/figure/Python-Based-Pseudo-code-for-Linear-Time-Vertex-Partitioning-Algorithm_fig1_317086968

import networkx as nx

def vertex_partition(G:nx.Graph, K:int):
    node_num = len(G.nodes)
    L = [{} for i in range(node_num)]
    S = []
    for i in range(node_num):
        deg = G.degree[i]
        L[deg][i] = True

    current = node_num - 1
    while len(S) < K:
        while len(L[current].keys()) == 0: 
            current -= 1
        keys = list(L[current].keys())
        while len(keys) > 0:
            choice = keys[0]
            S.append(choice)
            neighbors = G.neighbors(choice)
            for neigh in neighbors:
                ndegree = G.degree[neigh]
                del L[ndegree][neigh]
                L[ndegree - 1][neigh] = True 
            G.remove_node(choice)
            del L[current][choice]
            if len(S) == K:
                break 
            keys = list(L[current].keys())
    return S
