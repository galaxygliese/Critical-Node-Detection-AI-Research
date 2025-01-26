#-*- coding:utf-8 -*-

from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
from typing import List, Tuple
import torch.nn as nn
import networkx as nx
import collections
import numpy as np
import random
import torch 

def calc_q_values(
        embedding_vector:torch.Tensor, # state
        current_action_model:nn.Module, 
        current_value_model:nn.Module, 
    ) -> torch.Tensor:
    state = embedding_vector
    V = current_value_model(state) # (action_dim, )
    A = current_action_model(state) # (action_dim, )

    A_ = torch.mean(A)
    Q = V + A - A_
    return Q # (action_dim, )

def remove_node(G:nx.Graph, node_id:int) -> nx.Graph:
    G.remove_node(node_id)
    return G


def update_graph(G:nx.Graph, X:List[List[int]]) -> nx.Graph:
    G_ = nx.Graph()
    G_.add_nodes_from([
        (new_node_id, {'y': [0], 'x':X[new_node_id]})
        for new_node_id, old_node_id in enumerate(G.nodes)]
    )

    old2new = {old_node_id:new_node_id for new_node_id, old_node_id in enumerate(G.nodes)}
    new_edges = []
    for edge in G.edges:    
        new_edges.append((old2new[edge[0]], old2new[edge[1]]))    
    G_.add_edges_from(new_edges)
    return G_

def get_removed_data(G:nx.Graph, remove_node_id:int, transform=None) -> Tuple[Data, nx.Graph, int]:
    G = remove_node(G, remove_node_id)
    X: List[List[float]] = []
    node_num = len(G.nodes)

    nodes = list(G.nodes)
    node_degrees = dict(G.degree())
    node_degree_centralities = dict(nx.degree_centrality(G))
    node_centralities = dict(nx.eigenvector_centrality(G))
    node_pageranks = dict(nx.pagerank(G, alpha=0.85, max_iter=600))
    for node_id in nodes:
        x = [node_degrees[node_id], node_degree_centralities[node_id], node_centralities[node_id], node_pageranks[node_id]]
        X.append(x)
    
    G = update_graph(G, X)
    data : Data = from_networkx(G)
    if transform is not None:
        data = transform(data)
    return data, G, node_num

class Memory:
    def __init__(
            self, 
            dataset_seeds:List[int],
            length:int = 2, 
            device:str='cuda'
        ):
        self.rewards = collections.deque(maxlen=length)
        self.state = collections.deque(maxlen=length)
        self.action = collections.deque(maxlen=length)
        self.is_done = collections.deque(maxlen=length)
        self.dataset_seeds = dataset_seeds
        self.device = device
        self.all_rewards : List[float] = []

    def update(
            self, 
            state:nx.Graph, 
            action, 
            reward, 
            done
        ):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size:int):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        device = self.device

        seeds = random.sample(self.dataset_seeds)
        n = len(self.is_done)
        idx = random.sample(range(0, n-1), batch_size)

        state = np.array(self.state)
        action = np.array(self.action)
        return torch.Tensor(state)[idx].to(device), torch.LongTensor(action)[idx].to(device), \
               torch.Tensor(state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()

# Copy current weights into EMA model
def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def select_action(model, env, state, eps, device:str='cuda'):
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = np.argmax(values.cpu().numpy())

    return action
