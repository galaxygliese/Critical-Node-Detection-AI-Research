#-*- coding:utf-8 -*-

from typing import List, Set, Optional, Literal
import matplotlib.pyplot as plt
import networkx as nx
import csv
import os

class SampleGraphFromCSV(object):
    def __init__(self, 
                 folder_path:str,
        ):
        self.folder_path = folder_path
        self.G = self.load_graph()

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

    def plot_graph(self):
        plt.cla()
        nx.draw_networkx(self.G)
        plt.show()

    def save_graph(self):
        plt.cla()
        nx.draw_networkx(self.G)
        plt.savefig("results/input_sample_graph_csv.jpg")

    def save_critical_node_graph(self, G:nx.Graph, critical_nodes:List[int]):
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
        plt.savefig("results/output_critical_nodes_csv.jpg")
    
if __name__ == '__main__':
    sampler = SampleGraphFromCSV("dataset/9-11_terrorists")
    sampler.plot_graph()