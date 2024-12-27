#-*- coding:utf-8 -*-

from datas.sample_generator import SampleGraphGenerator
from datas.sample_from_csv import SampleGraphFromCSV
from methods.heuristic_cnd import heuristic_cricital_node2, heuristic_critical_node_detection_gemini_mis
from methods.copilot import calc_critical_node_detection
import time 

# MODE = 'EXAMPLE' 
MODE = 'CSV' 

def main():
    K = 4
    if MODE == 'EXAMPLE':
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

    elif MODE == 'CSV':
        sampler = SampleGraphFromCSV("dataset/9-11_terrorists")
        sampler.save_graph()
        G = sampler.G

    print("Calculation Started!")
    print("Node number:", len(G.nodes))
    print("Edge number:", len(G.edges))
    start_time = time.time()
    critical_nodes = calc_critical_node_detection(G, k=K)
    end_time = time.time()
    # critical_nodes = heuristic_critical_node_detection_gemini_mis(G, num_iterations=100, top_n=3)
    print("Critical Nodes:")
    print(critical_nodes)
    print("Time (sec): ", end_time - start_time)

    sampler.save_critical_node_graph(G, critical_nodes)
    print("Done!")

if __name__ == '__main__':
    main()