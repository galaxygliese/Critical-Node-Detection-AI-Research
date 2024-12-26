import networkx as nx
from itertools import combinations

def calc_critical_node_detection(G, k):
    """
    Function to detect critical nodes in a non-directional graph using a heuristic algorithm.
    
    Parameters:
    G (nx.Graph): Non-directional graph
    k (int): Critical node size
    
    Returns:
    List[int]: List of critical node indexes
    """
    
    def calculate_graph_disruption(graph, nodes_to_remove):
        """
        Calculate the disruption in the graph by removing the given nodes.
        
        Parameters:
        graph (nx.Graph): The graph
        nodes_to_remove (List[int]): List of nodes to remove
        
        Returns:
        int: The number of connected components after removal
        """
        temp_graph = graph.copy()
        temp_graph.remove_nodes_from(nodes_to_remove)
        return nx.number_connected_components(temp_graph)
    
    # Initialize the best set of critical nodes and the maximum disruption
    best_critical_nodes = []
    max_disruption = 0
    
    # Iterate over all combinations of k nodes
    for nodes in combinations(G.nodes(), k): # nodes: Tuple[int]
        disruption = calculate_graph_disruption(G, nodes)
        
        # Update the best set of critical nodes if the current disruption is greater
        if disruption > max_disruption:
            best_critical_nodes = nodes
            max_disruption = disruption
    
    return list(best_critical_nodes)



# Example usage:
if __name__ == "__main__":
    # Create a sample graph
    import matplotlib.pyplot as plt

    G = nx.erdos_renyi_graph(10, 0.5, seed=42)
    plt.cla()
    nx.draw_networkx(G)
    plt.savefig("sample_graph.jpg")

    # Set the critical node size
    k = 2

    # Calculate the critical nodes
    critical_nodes = calc_critical_node_detection(G, k)

    print(f"The critical nodes are: {critical_nodes}")