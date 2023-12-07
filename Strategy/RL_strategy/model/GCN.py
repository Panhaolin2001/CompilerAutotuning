import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F
torch.manual_seed(1234)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.line1 = torch.nn.Linear(128, 128)
        self.line2 = torch.nn.Linear(128, 64)
        self.line3 = torch.nn.Linear(64, output_dim)

        self.act = torch.nn.ReLU()

    def forward(self, state):
        x, edge_index, batch = state.x, state.edge_index, state.batch
        x = F.relu(self.conv1(x, edge_index))
        x1 = gap(x, batch)
        x = F.relu(self.conv2(x, edge_index))
        x2 = gap(x, batch)
        x = F.relu(self.conv3(x, edge_index))
        x3 = gap(x, batch)
        x = x1 + x2 + x3
        x = self.line1(x)
        x = self.act(x)
        x = self.line2(x)
        x = self.act(x)
        x = self.line3(x)
        return x