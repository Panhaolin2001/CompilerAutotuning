import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel

torch.manual_seed(1234)

class GCN(DQNTorchModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **customized_model_kwargs):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs, model_config, name, **customized_model_kwargs)
        torch.nn.Module.__init__(self)

        self.advantage_module = torch.nn.Sequential()
        self.value_module = torch.nn.Sequential()

        self.databatch = None

        self.input_dim = model_config['custom_model_config']['input_dim']
        self.output_dim = model_config['custom_model_config']['output_dim']

        self.conv1 = GCNConv(self.input_dim, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.line1 = torch.nn.Linear(128, 128)
        self.line2 = torch.nn.Linear(128, 64)
        self.line3 = torch.nn.Linear(64, self.output_dim)
        self.act = torch.nn.ReLU()

    def forward(self, input_dict, state, seq_lens):
        
        self.databatch = self.convert_simplebatch_to_geometric_batch(input_dict)

        x, edge_index, batch = self.databatch.x, self.databatch.edge_index, self.databatch.batch
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

        return x, state

    def value_function(self):
        return torch.zeros([1])
    
    def convert_simplebatch_to_geometric_batch(self, input_dict):
        dataset = []
        for node, edge in zip(input_dict["obs"]["nodes_features"], input_dict["obs"]["edge_index"]):
            dataset.append(Data(x=node, edge_index=edge.to(torch.int64)))
        return Batch.from_data_list(dataset)
