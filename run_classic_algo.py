#-*- coding:utf-8 -*-

from datas.sample_generator import SampleGraphGenerator
from datas.sample_from_csv import SampleGraphFromCSV
from methods.heuristic_cnd import heuristic_cricital_node2, heuristic_critical_node_detection_gemini_mis
from methods.copilot import calc_critical_node_detection
from methods.vertex_partition import vertex_partition
from methods.cnp1 import cnp1
from typing import List
import matplotlib.pyplot as plt 
import networkx as nx
import numpy as np
import random 
import time 
import sys 
import os

dataset_type = sys.argv[1]
seed = 42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    print("Seeds set.")
seed_everything(seed)

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
        plt.savefig(f"results/output_critical_nodes_{dataset_type}.jpg")

def main():
    K = 7
    if dataset_type == 'tree':
        # graph_type = "barbell_graph"
        graph_type = "full_rary_tree"
        # graph_type = "star_graph"
        # graph_type = "lollipop_graph"

        node_num:int = 62
        sampler = SampleGraphGenerator(graph_type_name=graph_type, node_num=node_num)
        G = sampler.get(5)
        # G = sampler.get_isolated_graphs()

        sampler.save_graph(G)
        # sampler.plot_graph(G)

    elif dataset_type == 'terrorist':
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

    print("Calculation Started!")
    print("Node number:", len(G.nodes))
    print("Edge number:", len(G.edges))
    start_time = time.time()
    # critical_nodes = calc_critical_node_detection(G, k=K)
    critical_nodes = cnp1(G, K=K)
    end_time = time.time()
    # critical_nodes = heuristic_critical_node_detection_gemini_mis(G, num_iterations=100, top_n=3)
    # print("Critical Nodes:")
    # print(critical_nodes)
    print("CNP-1 Time (sec): ", end_time - start_time)

    save_critical_node_graph(G, critical_nodes)
    print("Done!")

    start_time = time.time()
    critical_nodes2 = vertex_partition(G, K=K)
    end_time = time.time()
    print("Vertex-Partion Time (sec): ", end_time - start_time)

    print("Sol CNP-1:", sorted(critical_nodes))
    print("Sol Vertex-Partion:", sorted(critical_nodes2))

if __name__ == '__main__':
    main()