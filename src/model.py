import torch
from torch.nn import Linear, Dropout, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        self.dropout = Dropout(p=0.2)
        self.relu = ReLU()

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)
