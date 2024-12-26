import networkx as nx
import random

def calc_critical_node_detection(G: nx.Graph, k: int) -> list[int]:
    """
    Detects critical nodes in a graph using a simple heuristic approach.

    Args:
        G: The input graph (nx.Graph).
        k: The number of critical nodes to find.

    Returns:
        A list of critical node indices.
        Returns an empty list if k is invalid or the graph is empty.
    """

    if not G or k <= 0 or k > len(G):
        return []

    # 1. Calculate node degrees
    node_degrees = dict(G.degree())

    # 2. Sort nodes by degree in descending order
    sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)

    # 3. Select the top k nodes as critical nodes
    critical_nodes = sorted_nodes[:k]

    return critical_nodes


def calc_critical_node_detection_betweenness(G: nx.Graph, k: int) -> list[int]:
    """
    Detects critical nodes using betweenness centrality.

    Args:
        G: The input graph (nx.Graph).
        k: The number of critical nodes to find.

    Returns:
        A list of critical node indices.
        Returns an empty list if k is invalid or the graph is empty.
    """
    if not G or k <= 0 or k > len(G):
        return []

    betweenness_centrality = nx.betweenness_centrality(G)
    sorted_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
    critical_nodes = sorted_nodes[:k]
    return critical_nodes


def calc_critical_node_detection_random(G: nx.Graph, k: int) -> list[int]:
    """
    Detects critical nodes using random selection.
    This serves as a baseline for comparison.

    Args:
        G: The input graph (nx.Graph).
        k: The number of critical nodes to find.

    Returns:
        A list of critical node indices.
        Returns an empty list if k is invalid or the graph is empty.
    """
    if not G or k <= 0 or k > len(G):
        return []
        
    nodes = list(G.nodes())
    if k > len(nodes):
        k = len(nodes)
    critical_nodes = random.sample(nodes, k)
    return critical_nodes



# Example usage:
if __name__ == "__main__":
    # Create a sample graph
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 5), (4,6), (5,6)])

    k_value = 2

    # Degree centrality method
    critical_nodes_degree = calc_critical_node_detection(graph, k_value)
    print(f"Critical nodes (Degree Centrality, k={k_value}): {critical_nodes_degree}")

    # Betweenness centrality method
    critical_nodes_betweenness = calc_critical_node_detection_betweenness(graph, k_value)
    print(f"Critical nodes (Betweenness Centrality, k={k_value}): {critical_nodes_betweenness}")

    # Random selection method
    critical_nodes_random = calc_critical_node_detection_random(graph, k_value)
    print(f"Critical nodes (Random Selection, k={k_value}): {critical_nodes_random}")

    k_value = 3
    # Degree centrality method
    critical_nodes_degree = calc_critical_node_detection(graph, k_value)
    print(f"Critical nodes (Degree Centrality, k={k_value}): {critical_nodes_degree}")

    # Betweenness centrality method
    critical_nodes_betweenness = calc_critical_node_detection_betweenness(graph, k_value)
    print(f"Critical nodes (Betweenness Centrality, k={k_value}): {critical_nodes_betweenness}")

    # Random selection method
    critical_nodes_random = calc_critical_node_detection_random(graph, k_value)
    print(f"Critical nodes (Random Selection, k={k_value}): {critical_nodes_random}")