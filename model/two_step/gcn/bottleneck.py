import torch.nn as nn
from torch_geometric.nn import GCNConv


class GResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, in_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs, edges):
        x = self.relu(self.conv1(inputs, edges))
        x = self.relu(self.conv2(x, edges))

        return (inputs + x) * 0.5


class GBottleneck(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, block_num: int = 6):
        super().__init__()
        resblock_layers = [GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim) for _ in range(block_num)]
        self.blocks = nn.Sequential(*resblock_layers)

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs, edges):
        x = self.relu(self.conv1(inputs, edges))
        x_hidden = self.blocks(x, edges)
        x_out = self.conv2(x_hidden, edges)

        return x_out, x_hidden
