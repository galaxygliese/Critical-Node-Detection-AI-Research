#-*- coding:utf-8 -*-

from models.rl.nn import FeatureExtractorGAT, QNetwork
from models.rl.reward import critical_nodes_reward
from models.rl.q_learning import *
from datas.rl_dataset import SynthesisGraphDataset
from typing import List
from tqdm import tqdm

import torch_geometric.transforms as T
import torch.nn as nn
import numpy as np
import argparse
import random 
import torch 
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed for model and dataset. (default: 42)')
parser.add_argument('-e', '--episodes', type=int, default=6000)
parser.add_argument('--ema_epochs', type=int, default=300)
parser.add_argument('-b', '--batch_size', type=int, default=20)
parser.add_argument('--min_node_num', type=int, default=50)
parser.add_argument('--max_node_num', type=int, default=150)
parser.add_argument('--max_remove_num', type=int, default=30)
parser.add_argument('--gamma', type=float, default=0.98, help='discount rate')
parser.add_argument('--hidden_dim', type=int, default=96)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
parser.add_argument("--device", type=int, default=0)
opt = parser.parse_args()

EXPORT_DIR = "weights"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Seeds set.")
seed_everything(opt.seed)

def train_batch(
        batch_size:int, 
        dataset:SynthesisGraphDataset,
        extractor_model:FeatureExtractorGAT,
        current_action_model:nn.Module, 
        target_action_model:nn.Module, 
        current_value_model:nn.Module, 
        target_value_model:nn.Module, 
        dataset_seeds:List[int],
        optim:torch.optim, 
        gamma:float,
        eps:float = 1e-2,
        device:str='cuda'
    ):

    seeds = random.sample(dataset_seeds, batch_size)
    batch_loss = []
    batch_rewards = []
    final_loss = 0
    batch_count = 0

    for seed in seeds:
        try:
            K = random.randint(1, opt.max_remove_num)

            remove_count = 0
            data, G, node_num = dataset.get_all_data(seed)
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            G_init = G.copy()

            h = extractor_model(x, edge_index)
            q_values = []
            while remove_count <= K:
                Q = calc_q_values(
                    embedding_vector=h,
                    current_action_model=current_action_model,
                    current_value_model=current_value_model,
                )
                q_values.append(Q)

                if random.random() < eps:
                    remove_node_id = random.choice(list(G.nodes))
                else:
                    remove_node_id = torch.argmax(Q)
                    remove_node_id = int(remove_node_id.cpu().detach())
                
                data_next, G_next, node_num_next = get_removed_data(G, remove_node_id, transform=dataset.transform)

                x_next = data_next.x.to(device)
                edge_index_next = data_next.edge_index.to(device)

                h_next = extractor_model(x_next, edge_index_next)

                G = G_next
                h = h_next
                remove_count += 1

            reward = critical_nodes_reward(G_init, G)

            with torch.no_grad():
                Q_next_target = calc_q_values(
                    embedding_vector=h_next,
                    current_action_model=target_action_model,
                    current_value_model=target_value_model,
                )

            Q_expected = reward + gamma * torch.max(Q_next_target.detach())
            Q = torch.max(q_values[0]) 

            loss = (Q - Q_expected.detach()).pow(2).mean()
            final_loss += loss
            batch_loss.append(loss)
            batch_rewards.append(reward)
            batch_count += 1
        except Exception as e:
            print("Error occured during the training.")
            print(e)
        

    final_loss = final_loss / batch_count
    optim.zero_grad()
    final_loss.backward()
    optim.step()

    batch_reward = np.mean(batch_rewards)
    return final_loss.detach(), batch_reward


def main():
    device = opt.device 
    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ]) 
    dataset = SynthesisGraphDataset(
        min_node_num=opt.min_node_num, 
        max_node_num=opt.max_node_num, 
        transform=transform
    )
    num_episodes = opt.episodes
    data_seeds = list(range(num_episodes))

    feature_extracting_model = FeatureExtractorGAT(
        out_channels=opt.hidden_dim,
    ).to(device)
    q_action_network = QNetwork(state_dim=opt.hidden_dim, output_dim=1).to(device)
    q_value_network = QNetwork(state_dim=opt.hidden_dim, output_dim=1).to(device)

    q_action_target = QNetwork(state_dim=opt.hidden_dim, output_dim=1).to(device)
    q_value_target = QNetwork(state_dim=opt.hidden_dim, output_dim=1).to(device)

    update_parameters(q_action_network, q_action_target)
    update_parameters(q_value_network, q_value_target)

    for param in q_action_target.parameters():
        param.requires_grad = False

    for param in q_value_target.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        list(feature_extracting_model.parameters()) + list(q_action_network.parameters()) + list(q_value_network.parameters()), 
        lr=opt.lr
    )

    print("Models Loaded!")

    run = wandb.init(project = 'cnd_rl_ddqn')
    config = run.config
    config.episode = opt.episodes
    config.batch_size = opt.batch_size

    for episode in tqdm(range(opt.episodes)):
        batch_loss, batch_reward = train_batch(
            batch_size=opt.batch_size,
            dataset=dataset,
            extractor_model=feature_extracting_model,
            current_action_model=q_action_network, 
            target_action_model=q_action_target, 
            current_value_model=q_value_network, 
            target_value_model=q_value_target, 
            dataset_seeds=data_seeds,
            optim=optimizer, 
            gamma=opt.gamma,
            eps=1e-2,
            device=device
        )

        if episode % opt.ema_epochs == 0:
            update_parameters(q_action_network, q_action_target)
            update_parameters(q_value_network, q_value_target)

        run.log({
            "train_batch_loss": batch_loss,
            "train_batch_reward": batch_reward,
        })

    save_path = os.path.join(EXPORT_DIR, 'rl_ddqn_model_' + f'_episode{opt.episodes}.pth')
    torch.save(q_action_target.state_dict(), save_path)
    print("Done!")



if __name__ == '__main__':
    main()