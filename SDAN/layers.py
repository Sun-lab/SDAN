import torch
from torch import nn
from torch_geometric.nn import DenseGraphConv, dense_mincut_pool
import torch.nn.functional as F
from SDAN.utils import to_dense_normalized_adj


class PoolSuper(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, n_comp, out_channels):
        super(PoolSuper, self).__init__()
        self.conv1 = DenseGraphConv(in_channels, hidden_channels1)
        self.conv2 = DenseGraphConv(hidden_channels1, hidden_channels2)
        self.pool = nn.Linear(hidden_channels2, n_comp)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_comp, hidden_channels2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_channels2, hidden_channels2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_channels2, hidden_channels2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_channels2, out_channels)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        adj = to_dense_normalized_adj(edge_index=edge_index, max_num_nodes=data.num_nodes)
        x_GCN = F.relu(self.conv1(x, adj))
        x_GCN = F.relu(self.conv2(x_GCN, adj))
        x_GCN = x_GCN.squeeze()
        s = self.pool(x_GCN)
        _, _, mc, o = dense_mincut_pool(x, adj, s)
        s = torch.softmax(s, dim=-1)
        x = torch.matmul(x.t(), s)
        x = self.linear_relu_stack(x)
        return x, s, mc, o