#-*- coding:utf-8 -*-

from typing import List, Set, Optional, Literal
import matplotlib.pyplot as plt
import networkx as nx

GraphTypes = Literal["barbell_graph", "full_rary_tree", "lollipop_graph", "star_graph"]

class SampleGraphGenerator(object):
    def __init__(self, 
                 graph_type_name:GraphTypes,
                 node_num:int
        ):
        self.graph_type_name = graph_type_name
        self.node_num = node_num

    def get(self, param:Optional[int] = None) -> nx.Graph:
        if self.graph_type_name == 'barbell_graph':
            if param is None:
                param = 0
            return nx.barbell_graph(self.node_num, param)
        elif self.graph_type_name == 'full_rary_tree':
            if param is None:
                param = 0
            return nx.full_rary_tree(param, self.node_num, create_using=None)
        elif self.graph_type_name == 'lollipop_graph':
            if param is None:
                param = 0
            return nx.lollipop_graph(param, self.node_num)
        elif self.graph_type_name == 'star_graph':
            return nx.star_graph(self.node_num, create_using=None)
        return 
    
    def get_isolated_graphs(self) -> nx.Graph:
        edge_list = [
            (1, 2), (2, 3), (3, 1),
            (4, 5), (5, 6), (6, 7), (7, 8), (8, 4),
            (9, 10)
        ]

        G = nx.Graph()
        G.add_edges_from(edge_list)
        return G

    def plot_graph(self, G:nx.Graph):
        plt.cla()
        nx.draw_networkx(G)
        plt.show()
        
    def save_graph(elf, G:nx.Graph):
        plt.cla()
        nx.draw_networkx(G)
        plt.savefig("sample_graph.jpg")