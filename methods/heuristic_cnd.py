#-*- coding:utf-8 -*-

from typing import List, Set, Optional
import networkx as nx 
import numpy as np
import random

def heuristic_cricital_node(G:nx.Graph, k:int, seed:Optional[int] = None) -> Set[int]:
    MIS : Set[int] = set(nx.maximal_independent_set(G, seed=seed))
    V : Set[int] = set(G.nodes)

    count = 0
    while len(MIS) != len(V) - k:
        M : List[Set[int]] = [m for m in nx.connected_components(G)]
        V_MIS = V - MIS
        M_j : Optional[Set[int]] = None
        for j in V_MIS:
            for _M_j in M:
                if j in _M_j:
                    M_j = _M_j 
        
        i = np.argmin([(len(nx.node_connected_component(G, h)) - 1) * len(nx.node_connected_component(G, h)) / 2 for h in M_j])
        node_id = list(M_j)[i]
        MIS.add(node_id)
        count += 1
        if count > 10:
            break
    return V - MIS

def heuristic_cricital_node2(G:nx.Graph, k:int, seed:Optional[int] = None) -> Set[int]:
    num_component = nx.number_connected_components(G)
    components = [0] * num_component
    sizes = [0] * num_component
    
    mis : List[int] = nx.maximal_independent_set(G, seed=seed)
    component_id = 0 
    forbidden_count = 0 
    for i in range(num_component):
        print(">>", mis, i)
    return 


def heuristic_critical_node_detection_gemini(graph:nx.Graph, num_iterations:int=100, top_n:int=5):
    """
    Heuristically identifies critical nodes in a graph based on iterative removal
    and largest connected component size.

    Args:
        graph: A NetworkX graph object.
        num_iterations: The number of iterations to perform. Higher values increase accuracy but also computation time.
        top_n: The number of top critical nodes to return.

    Returns:
        A list of the top_n most critical nodes, or an empty list if the graph is empty.
        Returns a list of tuples, where each tuple contains the node and its "criticality score" (number of times it caused a large component split).
    """

    if not graph:
        return []

    num_nodes = graph.number_of_nodes()
    critical_nodes = {}

    for _ in range(num_iterations):
        # Create a copy of the graph to avoid modifying the original
        temp_graph = graph.copy()

        # Choose a random node to remove
        if temp_graph.number_of_nodes() > 0: # Check if there are still nodes
            node_to_remove = random.choice(list(temp_graph.nodes()))
            temp_graph.remove_node(node_to_remove)

            # Check the size of the largest connected component
            largest_cc_size_after_removal = 0
            if temp_graph.number_of_nodes() > 0:
                largest_cc_size_after_removal = len(max(nx.connected_components(temp_graph), key=len)) if nx.number_connected_components(temp_graph) > 0 else 0

            # Check the size of the largest connected component of the original graph
            largest_cc_size_before_removal = len(max(nx.connected_components(graph), key=len)) if nx.number_connected_components(graph) > 0 else 0

            # If removing the node significantly reduced the largest connected component, consider it critical
            if largest_cc_size_after_removal < largest_cc_size_before_removal * 0.8: # Example threshold: 80% of original size. Tune as needed.
                critical_nodes[node_to_remove] = critical_nodes.get(node_to_remove, 0) + 1

    # Sort nodes by criticality score
    sorted_critical_nodes = sorted(critical_nodes.items(), key=lambda item: item[1], reverse=True)

    return sorted_critical_nodes[:top_n]

def heuristic_critical_node_detection_gemini_mis(graph, num_iterations=100, top_n=5):
    """
    Heuristically identifies critical nodes in a graph based on iterative removal
    and the impact on the Maximal Independent Set (MIS).

    Args:
        graph: A NetworkX graph object.
        num_iterations: The number of iterations to perform.
        top_n: The number of top critical node IDs to return.

    Returns:
        A list of the top_n most critical node IDs (integers), or an empty list if the graph is empty.
        Returns a list of tuples, where each tuple contains the node ID and its "criticality score".
    """

    if not graph:
        return []

    critical_nodes = {}
    print("OK1")

    for _ in range(num_iterations):
        temp_graph = graph.copy()

        if temp_graph.number_of_nodes() > 0:
            node_to_remove = random.choice(list(temp_graph.nodes()))
            temp_graph.remove_node(node_to_remove)

            # Calculate MIS before and after removal
            mis_before = len(nx.maximal_independent_set(graph))
            mis_after = len(nx.maximal_independent_set(temp_graph)) if temp_graph.number_of_nodes() > 0 else 0

            # If removing the node significantly increased the MIS, consider it critical
            if mis_after > mis_before * 1.2:  # Example threshold: 20% increase. Tune as needed.
                critical_nodes[node_to_remove] = critical_nodes.get(node_to_remove, 0) + 1

    sorted_critical_nodes = sorted(critical_nodes.items(), key=lambda item: item[1], reverse=True)

    #Extract only the node IDs
    top_critical_node_ids = [(node_id, score) for node_id, score in sorted_critical_nodes[:top_n]]

    return top_critical_node_ids