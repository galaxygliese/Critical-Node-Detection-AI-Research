#-*- coding:utf-8 -*-

from cplex.callbacks import IncumbentCallback, BranchCallback, UserCutCallback, HeuristicCallback

from typing import List
import networkx as nx
import numpy as np
import pickle
import cplex
import time
import math

class CustomizedIncumbent(IncumbentCallback):
    eps = 1e-4

    def __call__(self):
        x_values = list(map(lambda v: 1 if v > 1-self.eps else 0, self.get_values(self.x)))
        y_values = list(map(lambda v: 1 if v > 1-self.eps else 0, self.get_values(self.y)))

        incval =  self.get_objective_value()
        incval1 = self.get_incumbent_objective_value()

        cost = list(map(lambda i: self.c[i] + self.c_[i]*y_values[i], range(self.node_num)))
        x_ = self.x_ 
        u_ = self.u_

        self.solver2.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=x_, val=cost)],
            senses = ["L"],
            rhs = [self.L],
            names = ["newcon"]
        )

        self.solver2.solve()
        solver2_obj = self.solver2.solution.get_objective_value()

        if solver2_obj < incval - self.eps:
            if solver2_obj > incval1 + self.eps:
                solution = self.solver2.solution
                u_values = []
                for i in range(self.node_num):
                    u_values.append([])
                    u_values[i] = list(map(lambda x: 1 if x > 1-self.eps else 0, solution.get_values(self.u_[i])))

                xsol = list(map(lambda x: 1 if x > self.eps else 0, solution.get_values(self.x_)))
                innerobj = solver2_obj
                self.set_node_data([xsol, innerobj, incval, y_values, u_values])
            self.solver2.linear_constraints.delete("newcon")
            self.reject()
        else:
            self.solver2.linear_constraints.delete("newcon")
            # self.accept() [?]

class CustomizedHeuristic(HeuristicCallback):
    def __call__(self):
        dbug = self.dbug
        
        node_data = self.get_node_data()
        if self.get_node_ID() % 50 != 0: return
        incum_value = self.get_incumbent_objective_value()
        if dbug: print ("Entering heuristic callback", self.get_node_ID(), self.get_current_node_depth(), incum_value)

        if node_data is None:
            
            lp_yvals = np.array(self.get_values(self.yvars))
            
            subsidy_cost = 0
            
            yvals = [0 for i in range(self.N)]
            
            for i in range(self.N):
                if np.random.random() > lp_yvals[i]: continue #do a randomised rounding on the LP value
                if subsidy_cost + self.cU[i] > self.bU: continue
                yvals[i] = 1
                subsidy_cost += self.cU[i]
            
            cost = list(map(lambda x: self.cL[x] + self.ciL[x]*yvals[x], range(self.N)))#subsidised cost of items - list
            cost_new = cost.copy()
            x_inc = self.x_inc
            u_inc = self.u_inc
            
            self.incmodel.linear_constraints.add(lin_expr = [cplex.SparsePair(ind=x_inc, val=cost)],
                                       senses = ["L"],
                                        rhs = [self.bL], names = ["newcon"])
            for i in range(self.N):
                cost_new[i] += self.bL+1 - cost[i]
                self.incmodel.linear_constraints.add(lin_expr = [cplex.SparsePair(ind= x_inc, val = cost_new)],
                                                     senses = ["G"],
                                                     rhs = [self.bL+1 - cost[i]], names = ["maxcon"+str(i)])
                cost_new[i] += self.bL+1 - cost[i]
            #todo remove this constraint after solve()
            if not dbug: self.incmodel.parameters.mip.display.set(0)
            if not dbug: self.incmodel.set_results_stream(None)
            #if dbug: self.incmodel.write("innerprobHeur.lp")
            self.incmodel.solve()
            if dbug: print ("Solved LL problem", self.incmodel.solution.get_status())
            #self.incmodel.linear_constraints.delete("newCon")
            #todo do early termination to exceed incumbent  value
            
            incmodelObj = self.incmodel.solution.get_objective_value()
            
            if incmodelObj < incum_value - 1e-4: return
            
            #get u values, x values from LL problem and y values from upper level problem
            uvals = []
            solution = self.incmodel.solution
            for i in range(self.N):
                uvals.append([])
                uvals[i] = list(map(lambda x: 1 if x > 1-1e-4 else 0, solution.get_values(self.u_inc[i])))
            
            xvals = list(map(lambda x: 1 if x > 1e-4 else 0, solution.get_values(self.x_inc)))
            zvals = list(map(lambda x: 1 if xvals[x] > 1 - 1e-4 and yvals[x] > 1 - 1e-4 else 0, range(self.N)))
            index = self.xvars + self.yvars + self.zvars
            vals = xvals + yvals + zvals
            for i in range(self.N):
                index += self.uvars[i]
                vals += list(uvals[i])
            
            self.incmodel.linear_constraints.delete("newcon")
            for i in range(self.N):
                self.incmodel.linear_constraints.delete("maxcon"+str(i))
            self.set_solution([index, vals], objective_value = incmodelObj)
        else:
            
            xvals, innerobj, incval, yvals, uvals = node_data
            
            if innerobj <= incum_value - 1e-4: return
            zvals = list(map(lambda x: 1 if xvals[x] > 1 - 1e-4 and yvals[x] > 1 - 1e-4 else 0, range(self.N)))
            #zvals = xvals*yvals
            #
            index = self.xvars + self.yvars + self.zvars
            vals = xvals + yvals + zvals
            
            for i in range(self.N):
                index += self.uvars[i]
                vals += list(uvals[i])
                
            
            self.set_solution([index, vals])

class CustomizedBranch(BranchCallback):
    def __call__(self):
        dbug = self.dbug
        if dbug: print ("Entering branching callback", self.get_node_ID(), self.get_current_node_depth())
        node_data = self.get_node_data()
        if node_data is not None:
            xsol, innerobj, incval, yvals, uvals = node_data
        else: 
            return
            
        if dbug: print ("Making branches in branch callback...", xsol, innerobj, incval)

        rhs = self.bL
        ind = []
        val = []
        for x in range(self.N):
            if xsol[x] < 1e-4: continue
            ind += [self.yvars[x]]
            val += [self.ciL[x]*xsol[x]]
            rhs -= self.cL[x]*xsol[x]
           
        index = []
        value = []
        
        #values = [1.0 for i in range(self.N)]
        for x in range(self.N):
            index += self.uvars[x]
            #value += values
            value += self.uobj[x]
        
        #left branch
        #TODO: add the corresponding covercuts here for the left branch
        con = [(cplex.SparsePair(ind = ind, val = val), "G", rhs+1)] #budget based cut
        
        con += [(cplex.SparsePair(ind = index, val = value), "L", incval)] #UB based on rejected incumbent's value
        #Add Nogood cut
        con += [(cplex.SparsePair(ind = self.xvars, 
                                  val = list(map(lambda x: 1.0 if x > 1 - 1e-4 else -1.0, xsol))), "L", sum(xsol) - 1) ]
        #upperbound the objective by incval
        con += [(cplex.SparsePair(ind = index, val= value), "L", incval)]

        self.make_branch(incval, constraints = con)	
        
        #rightbranch
        con = [(cplex.SparsePair(ind = ind, val = val), "L", rhs)]
        
        con += [(cplex.SparsePair(ind = index, val= value), "L", innerobj)]
        self.make_branch(innerobj, constraints = con)	

class CustomizedCut(UserCutCallback):
    def __call__(self):
        dbug = self.dbug 
        current_node_depth = self.get_current_node_depth()
        if dbug: print ("Entering user cut callback", self.get_node_ID(), current_node_depth)
        if current_node_depth % 10 != 0: return
        G = self.G
        N = self.N
        save_total = self.totcuts
        save_nodedepth = self.nodedepth
        startnode = self.startnode
        numcuts = math.ceil(math.log(current_node_depth) if current_node_depth != 0 else 0)*30
        if save_nodedepth == current_node_depth and save_total >= 200:
            return
        if save_nodedepth != current_node_depth:
            self.nodedepth = current_node_depth
            self.totcuts = 0
            
        #if startnode != 0: print ("Startnode in usercut", startnode,numcuts, current_node_depth)
        attr = {}
        rvals = self.get_values(self.rvars)
        for k,(i,j) in enumerate(G.edges):
            attr[(i,j)] = {'capacity':rvals[k]}
        nx.set_edge_attributes(G, attr)
        totCut = 0
        nodelist = list(range(startnode, N)) + list(range(startnode))
        for t in nodelist:
            uvals = self.get_values(self.uvars[t])
            if totCut > numcuts: break
            for s in range(t+1, N):
                if dbug: print("Running min cut for", s, t)
                cut = nx.minimum_cut(G, s,t)
                if cut[0] < uvals[s] - 1e-4:
                    if dbug: print("cut violation, adding cuts:", cut[0], s,t, uvals[s])
                    nodecutset1 = []
                    nodecutset2 = []
                    nodeval1 = []
                    nodeval2 = []
                    cutset = [self.uvars[s][t]]
                    vals = [-1.0]
                    for k, (i,j) in enumerate(G.edges):
                        if i in cut[1][0] and j in cut[1][1]:
                            cutset += [self.rvars[k]]
                            vals += [1.0]
                            nodecutset1 += [self.xvars[i]]
                            nodecutset2 += [self.xvars[j]]
                            nodeval1 += [1.0]
                            nodeval2 += [1.0]
                        if j in cut[1][0] and i in cut[1][1]:
                            cutset += [self.rvars[k]]
                            vals += [1.0] 
                            nodecutset1 += [self.xvars[j]]
                            nodecutset2 += [self.xvars[i]]
                            nodeval1 += [1.0]
                            nodeval2 += [1.0]
                    if dbug: print ("cutset found", cutset)       
                    self.add(cut=cplex.SparsePair(ind=cutset, val= vals), 
                             sense= "G",
                             rhs = 0)
                    nodeval1 += [1]
                    nodeval2 += [1]
                    for p in cut[1][0]: 
                        for q in cut[1][1]:
                            nodecutset1 += [self.uvars[p][q]]
                            nodecutset2 += [self.uvars[p][q]]
                            self.add(cut=cplex.SparsePair(ind=nodecutset1, val= nodeval1), 
                                     sense= "L",
                                     rhs = len(nodeval1)-1)
                            self.add(cut=cplex.SparsePair(ind=nodecutset2, val= nodeval2), 
                                     sense= "L",
                                     rhs = len(nodeval2)-1)
                            nodecutset1.pop()
                            nodecutset2.pop()
                    totCut += 1
        
        self.startnode = t
        self.totcuts += totCut
            
        if totCut > 0 and dbug: print("Added ", totCut, "cutset cuts")

def bicnp(G:nx.Graph, K:int):
    # K : int = 24 # budget_upper
    L : int = K # budget_lower
    node_num : int = len(G.nodes)
    upper_variable_cost_value : int = 1
    lower_variable_costs_value : int = 1 
    lower_variable_costs_increased_value : int = L

    b : List[int] = [upper_variable_cost_value] * node_num # upper_variable_costs
    c : List[int] = [lower_variable_costs_value] * node_num # lower_variable_costs
    c_ : List[int] = [lower_variable_costs_increased_value] * node_num # lower_variable_costs_increased

    # edge_density : float = 0.2 
    # item_cost : int = 1
    # budget_prop : float = 0.6

    # G : nx.Graph = pickle.load(open('graph_sample.pickle', 'rb'))
    edge_num = len(G.edges)
    print("Graph Loaded!")

    solver1 = cplex.Cplex()
    solver2 = cplex.Cplex()

    # Define: target function
    x = list(solver1.variables.add(
        obj = [0] * node_num,
        lb = [0] * node_num,
        ub = [1] * node_num,
        types = ['B'] * node_num,
        names = ["x[" +str(i)+"]" for i in range(node_num)]
    ))

    y = list(solver1.variables.add(
        obj = [0] * node_num,
        lb = [0] * node_num,
        ub = [1] * node_num,
        types = ['B'] * node_num,
        names = ["y[" +str(i)+"]" for i in range(node_num)]
    ))

    z = list(solver1.variables.add(
        obj = [0] * node_num,
        lb = [0] * node_num,
        ub = [1] * node_num,
        types = ['B'] * node_num,
        names = ["z[" +str(i)+"]" for i in range(node_num)]
    ))

    x_ = list(solver2.variables.add(
        obj = [0] * node_num,
        lb = [0] * node_num,
        ub = [1] * node_num,
        types = ['B'] * node_num,
        names = ["x[" +str(i)+"]" for i in range(node_num)]
    ))

    u = [[] for i in range(node_num)]
    u_ = [[] for i in range(node_num)]
    u_sol = [0 for i in range(node_num)]
    u_sols = []
    u_mask = []

    for i in range(node_num):
        u_sols += [u_sol]
        weight_values = [1.0 if j < i else 0.0 for j in range(node_num)]
        u_mask.append(weight_values)

        u[i] = list(solver1.variables.add(
            obj = weight_values,
            lb = [0]*node_num,
            ub = [1]*node_num,
            types = ["B"]*node_num,
            names = ["u[" + str(j) + "][" + str(i) +"]" for j in range(node_num)]
        ))

        u_[i] = list(solver2.variables.add(
            obj = weight_values,
            lb = [0]*node_num,
            ub = [1]*node_num,
            types = ["B"]*node_num,
            names = ["u[" + str(j) + "][" + str(i) +"]" for j in range(node_num)]
        ))

    
    r = list(solver1.variables.add(
        obj = [0] * edge_num,
        lb = [0] * edge_num,
        ub = [1] * edge_num,
        types = ['B'] * edge_num,
        names = ["r[" +str(i)+"]" for i in range(edge_num)]
    ))


    # Define: s.t.
    solver1.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(ind=y, val=b) # <= upper_variable_costs
        ],
        senses = ["L"],
        rhs = [K]
    )

    solver1.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(ind=[*x, *z], val = [*list(c), *list(c_)]) # <= lower_variable_costs
        ],
        senses = ["L"],
        rhs = [L]
    )

    c_buff = c.copy()
    for i in range(node_num):
        c_buff[i] += L + 1 - c[i]
        solver1.linear_constraints.add(
            lin_expr = [
                cplex.SparsePair(ind=[*x,*z, y[i]], val = [*c_buff, *list(c_), c_[i]])
            ],
            senses = ["G"], # =>
            rhs = [L + 1 - c[i]]
        )
        c_buff[i] -= L + 1 - c[i]

    for i in range(node_num):
        solver1.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind = [z[i],y[i]], val = [1.0, -1.0])],
            senses = ["L"], # <=
            rhs = [0.0]
        )
        solver1.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind = [z[i],x[i]], val = [1.0, -1.0])],
            senses = ["L"],
            rhs = [0.0]
        )
        solver1.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind = [z[i],x[i],y[i]], val = [-1.0, 1.0, 1.0])],
            senses = ["L"],
            rhs = [1.0]
        )

    for k, (node_i, node_j) in enumerate(G.edges):
        [node_i, node_j] = [node_i, node_j] if node_i < node_j else [node_j, node_i]
        solver1.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=[u[node_i][node_j], x[node_i], x[node_j]], val=[1, 1, 1])],
            senses = ["G"],
            rhs = [1.0]
        )
        solver1.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=[r[k], x[node_i]], val=[1, 1])],
            senses = ["L"],
            rhs = [1.0]
        )   
        solver1.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=[r[k], x[node_j]], val=[1, 1])],
            senses = ["L"],
            rhs = [1.0]
        )
        solver1.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=[u[i][node_j], x[i]], val=[1, 1])],
            senses = ["L"],
            rhs = [1.0]
        )
        solver1.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=[u[i][node_j], x[node_j]], val=[1, 1])],
            senses = ["L"],
            rhs = [1.0]
        )
        solver2.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=[u_[node_i][node_j], x_[node_i], x_[node_j]], val=[1, 1, 1])],
            senses = ["G"],
            rhs = [1.0]
        )
        solver2.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=[u_[node_i][node_j], x_[node_i]], val=[1, 1])],
            senses = ["L"],
            rhs = [1.0]
        )
        solver2.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind=[u_[node_i][node_j], x_[node_j]], val=[1, 1])],
            senses = ["L"],
            rhs = [1.0]
        )

    for i in range(node_num):
        for j in range(node_num):
            if i == j: 
                continue

            solver1.linear_constraints.add(
                lin_expr = [cplex.SparsePair(ind=[u[i][j],u[j][i]], val=[1, - 1])],
                senses = ["E"],
                rhs = [0]
            )
            solver2.linear_constraints.add(
                lin_expr = [cplex.SparsePair(ind=[u_[i][j],u_[j][i]], val=[1, - 1])],
                senses = ["E"],
                rhs = [0]
            )
            for k in range(node_num):
                if j == k or i == k: 
                    continue

                solver1.linear_constraints.add(
                    lin_expr = [cplex.SparsePair(ind=[u[i][j],u[j][k], u[k][i]], val=[1, 1, -1])],
                    senses = ["L"],
                    rhs = [1.0]
                )
                solver1.linear_constraints.add(
                    lin_expr = [cplex.SparsePair(ind=[u[i][j],u[j][k], u[k][i]], val=[1, -1, 1])],
                    senses = ["L"],
                    rhs = [1.0]
                )
                solver1.linear_constraints.add(
                    lin_expr = [cplex.SparsePair(ind=[u[i][j],u[j][k], u[k][i]], val=[-1, 1, 1])],
                    senses = ["L"],
                    rhs = [1.0]
                )

                solver2.linear_constraints.add(
                    lin_expr = [cplex.SparsePair(ind=[u_[i][j],u_[j][k], u_[k][i]], val=[1, 1, -1])],
                    senses = ["L"],
                    rhs = [1.0]
                )
                solver2.linear_constraints.add(
                    lin_expr = [cplex.SparsePair(ind=[u_[i][j],u_[j][k],u_[k][i]], val=[1, -1, 1])],
                    senses = ["L"],
                    rhs = [1.0]
                )
                solver2.linear_constraints.add(
                    lin_expr = [cplex.SparsePair(ind=[u_[i][j],u_[j][k], u_[k][i]], val=[-1, 1, 1])],
                    senses = ["L"],
                    rhs = [1.0]
                )
    
    print("Problem Defined!")
    solver1.objective.set_sense(solver1.objective.sense.maximize)

    # Simulation settings
    solver1.parameters.timelimit.set(3600)
    solver1.parameters.mip.tolerances.mipgap.set(0.01)

    x_sol = list(map(lambda x: 0, range(node_num)))

    inccb = solver1.register_callback(CustomizedIncumbent)
    inccb.solver2 = solver2
    inccb.x_ = x_
    inccb.u_ = u_
    inccb.c_ = c_
    inccb.c = c
    inccb.L = L
    inccb.node_num = node_num
    inccb.x = x
    inccb.y = y
    inccb.xsol = x_sol
    inccb.usol = u_sols
    inccb.innerobj = 0

    brcb = solver1.register_callback(CustomizedBranch)
    brcb.dbug = False
    brcb.yvars = y
    brcb.xvars = x
    brcb.uvars = u
    brcb.ciL = c_
    brcb.cL = c
    brcb.bL = L
    brcb.N  = node_num
    brcb.uobj = u_mask

    heurcb = solver1.register_callback(CustomizedHeuristic)
    heurcb.xvars = x
    heurcb.dbug = False
    heurcb.yvars = y
    heurcb.zvars = z
    heurcb.uvars = u
    heurcb.incmodel = solver2
    heurcb.x_inc = x_
    heurcb.u_inc = u_
    heurcb.N = node_num
    heurcb.ciL = c_
    heurcb.cL = c

    heurcb.bL = L
    heurcb.N = node_num
    heurcb.cU = b
    heurcb.bU = K

    # Solve
    start = time.time() 
    solver1.solve()
    duration = time.time() - start

    solution = solver1.solution
    mip = solution.MIP
    x_values = solution.get_values(x)
    y_values = solution.get_values(y)
    z_values = solution.get_values(z)

    obj = solver1.solution.get_objective_value()

    print("Time:", duration)
    print ("### Total variables picked:")
    print("X:")
    print(x_values)

    print("Y:")
    print(y_values)

    print("Z:")
    print(z_values)


    print("### Objectives:")
    print(obj)
    print(mip.get_mip_relative_gap())

    print("SUM of X:", sum(x_values))
    return x_values


if __name__ == '__main__':
    bicnp()