#-*- coding:utf-8 -*-

from datas.sample_generator import SampleGraphGenerator
from methods.heuristic_cnd import heuristic_cricital_node2, heuristic_critical_node_detection_gemini_mis
from methods.copilot import calc_critical_node_detection

def main():
    graph_type = "barbell_graph"
    # graph_type = "full_rary_tree"
    # graph_type = "star_graph"
    # graph_type = "lollipop_graph"
    node_num:int = 10
    sampler = SampleGraphGenerator(graph_type_name=graph_type, node_num=node_num)
    G = sampler.get(5)
    # G = sampler.get_isolated_graphs()
    sampler.save_graph(G)

    # sampler.plot_graph(G)

    # cricital_nodes = heuristic_cricital_node2(G, k=3)
    critical_nodes = calc_critical_node_detection(G, k=3)
    # critical_nodes = heuristic_critical_node_detection_gemini_mis(G, num_iterations=100, top_n=3)
    print("Critical Nodes:")
    print(critical_nodes)

if __name__ == '__main__':
    main()