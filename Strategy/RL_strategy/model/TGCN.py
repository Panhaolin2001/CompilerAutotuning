import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F
torch.manual_seed(1234)

class TGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=128):
        super(TGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.gcn = GCNConv(num_node_features, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, graph_data_list):
        batch = 0

        if isinstance(graph_data_list, tuple):
            all_out = []
            batch = graph_data_list[0][0].batch
            for sub_list in graph_data_list:
                graph_features = [F.relu(self.gcn(data.x, data.edge_index)) for data in sub_list]
                graph_features_sequence = torch.stack(graph_features, dim=1)  # (batch_size, seq_len, hidden_dim)
                gru_out, _ = self.gru(graph_features_sequence)
                last_step_out = gru_out[:, -1, :]  # (batch_size, hidden_dim)

                out = self.fc(last_step_out)
                out = gap(out, batch)
                all_out.append(out)
            
            all_out = torch.stack(all_out, dim=1)
            return all_out

        else:
            batch = graph_data_list[0].batch
            graph_features = [F.relu(self.gcn(data.x, data.edge_index)) for data in graph_data_list]
            graph_features_sequence = torch.stack(graph_features, dim=1)  # (batch_size, seq_len, hidden_dim)
            gru_out, _ = self.gru(graph_features_sequence)
            last_step_out = gru_out[:, -1, :]  # (batch_size, hidden_dim)

            out = self.fc(last_step_out)
            out = gap(out,batch)

            return out