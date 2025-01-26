#-*- coding:utf-8 -*-

from typing import List 
import networkx as nx
import cplex
import time

def cnp1(G:nx.Graph, K:int):
    node_num:int = len(G.nodes)
    critical_node_num:int = K

    solver = cplex.Cplex()
    solver.parameters.timelimit.set(150)

    # Define: target function
    v = list(solver.variables.add(
        obj = [0] * node_num,
        lb = [0] * node_num,
        ub = [1] * node_num,
        types = ['B'] * node_num,
        names = ["v[" +str(i)+"]" for i in range(node_num)]
    ))

    u = [[] for i in range(node_num)]
    for i in range(node_num):
        weight_values = [1.0 if j < i else 0.0 for j in range(node_num)]
        u[i] = list(solver.variables.add(
            obj = weight_values,
            lb = [0]*node_num,
            ub = [1]*node_num,
            types = ["B"]*node_num,
            names = ["u[" + str(j) + "][" + str(i) +"]" for j in range(node_num)]
        ))

    # Define: s.t.
    for k, (node_i, node_j) in enumerate(G.edges):
        [node_i, node_j] = [node_i, node_j] if node_i < node_j else [node_j, node_i]
        solver.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=[u[node_i][node_j], v[node_i], v[node_j]], val=[1, 1, 1])],
            senses = ["G"],
            rhs = [1.0]
        )

    for i in range(node_num):
        for j in range(node_num):
            if i == j: 
                continue

            for k in range(node_num):
                if j == k or i == k: 
                    continue

                solver.linear_constraints.add(
                    lin_expr = [cplex.SparsePair(ind=[u[i][j],u[j][k], u[k][i]], val=[1, 1, -1])],
                    senses = ["L"],
                    rhs = [1.0]
                )
                solver.linear_constraints.add(
                    lin_expr = [cplex.SparsePair(ind=[u[i][j],u[j][k], u[k][i]], val=[1, -1, 1])],
                    senses = ["L"],
                    rhs = [1.0]
                )
                solver.linear_constraints.add(
                    lin_expr = [cplex.SparsePair(ind=[u[i][j],u[j][k], u[k][i]], val=[-1, 1, 1])],
                    senses = ["L"],
                    rhs = [1.0]
                )
    
    b = [1] * node_num
    solver.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(ind=v, val=b)
        ],
        senses = ["L"],
        rhs = [critical_node_num]
    )

    print("Problem Defined!")
    solver.objective.set_sense(solver.objective.sense.minimize)

    start = time.time() 
    solver.solve()
    duration = time.time() - start

    solution = solver.solution
    # print("Duration (sec):", duration)
    # print("Solution:", solution)
    solved_nodes = []
    for node_id, flag in enumerate(solution.get_values(v)):
        if flag > 0:
            solved_nodes.append(node_id)
    return solved_nodes
