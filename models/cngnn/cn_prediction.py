#-*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn

def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")

class CriticalNodePredictor(nn.Module):
    """Simple MLP Model"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu',
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, x, edge_index):

        for i, mlp in enumerate(self.mlps[:-1]):
            x = mlp(x)
            x = self.dropout(x)
            x = self.activation(x)

        x = self.mlps[-1](x)
        x = self.activation(x)

        return x.squeeze(1)#.sigmoid()
    
if __name__ == '__main__':
    import random 
    K = 4
    node_num = 20
    hidden_channels = 128
    decoder_channels = 64
    decoder_dropout = 0.2
    model = CriticalNodePredictor(
        in_channels=hidden_channels,
        hidden_channels=decoder_channels,
        out_channels=1,
        num_layers=2,
        dropout=decoder_dropout,
    )

    criterion = torch.nn.BCELoss()
    
    z = torch.randn(node_num, hidden_channels)
    o = model(z).squeeze(1).sigmoid()
    print(">>", o.shape)

    critical_node_labels = random.sample(list(range(node_num)), K)
    y = torch.zeros(node_num)
    for critical_node in critical_node_labels:
        y[critical_node] += 1


    print("Model output >>", o)    
    print("Critical nodes >>", y)

    loss = criterion(o, y)
    print("BCELoss >>", loss)
