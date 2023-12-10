import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from ..envUtility.common import feature_change_due_to_pass, get_pass_feature_internal
from ..actionspace.llvm16.actions import Actions_LLVM_16
from ..actionspace.llvm14.actions import Actions_LLVM_14
from ..actionspace.llvm10.actions import Actions_LLVM_10
from ..actionspace.CompilerGymLLVMv0.actions import Actions_LLVM_10_0_0

def one_hot(index_list, class_num):

    if type(index_list) == torch.Tensor:
        index_list = index_list.detach().numpy()
    indexes = torch.LongTensor(index_list).view(-1, 1)
    out = torch.zeros(len(index_list), class_num)
    out = out.scatter_(dim=1,index=indexes,value=1)
    return out

def GetFeature(ll_file, obs_type="P2VInstCount", action_space="llvm-16.x"):

    if obs_type == "P2VInstCount":
        Actions = 0
        match action_space:
            case "llvm-16.x":
                Actions = Actions_LLVM_16
            case "llvm-14.x":
                Actions = Actions_LLVM_14
            case "llvm-10.x":
                Actions = Actions_LLVM_10
            case "llvm-10.0.0":
                Actions = Actions_LLVM_10_0_0
            case _:
                raise ValueError(f"Unknown action space: {action_space}, please choose 'llvm-16.x','llvm-14.x','llvm-10.x','llvm-10.0.0' ")
            
        pass_features = {}
        baseline_counts = get_pass_feature_internal(ll_file, "-O0", obs_type=obs_type)  # Get the counts for no optimizations
        for action in Actions:
            if action_space != "llvm-10.0.0" and action_space != "llvm-10.x":
                pass_features[action.name] = feature_change_due_to_pass(ll_file, "--enable-new-pm=0 " + action.value, baseline_counts, obs_type=obs_type)
            else:
                pass_features[action.name] = feature_change_due_to_pass(ll_file, action.value, baseline_counts=baseline_counts, obs_type=obs_type)

        original_keys = list(pass_features.keys())
        original_sub_keys = list(pass_features[original_keys[0]].keys())

        pass_features_values = np.array([list(d.values()) for d in pass_features.values()])
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # scaled_features = scaler.fit_transform(pass_features_values)

        scaled_pass_features = {name: dict(zip(original_sub_keys, features)) for name, features in zip(original_keys, pass_features_values)}

        return scaled_pass_features

class CustomDataset(Dataset):
    def __init__(self, obs_list, actions, rewards, next_obs_list, done_list):
        self.obs_list = obs_list
        self.actions = actions
        self.rewards = rewards
        self.next_obs_list = next_obs_list
        self.done_list = done_list

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        obs = self.obs_list[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obs_list[idx]
        done = self.done_list[idx]
        return obs, action, reward, next_obs, done

def GNN_collate_fn(data_list):
    graphs, actions, rewards, next_graphs, dones = zip(*data_list)
    batched_graphs = Batch.from_data_list(graphs)
    batched_next_graphs = Batch.from_data_list(next_graphs)
    actions = torch.tensor(actions, dtype=torch.float)
    rewards = torch.tensor(rewards, dtype=torch.float)
    dones = torch.tensor(dones, dtype=torch.float)
    return batched_graphs, actions, rewards, batched_next_graphs, dones

def Transformer_collate_fn(data_list):
    obs_list, actions, rewards, next_obs_list, done_list = zip(*data_list)
    batched_obs = torch.stack(obs_list)
    batched_next_obs = torch.stack(next_obs_list)
    actions = torch.tensor(actions, dtype=torch.float)
    rewards = torch.tensor(rewards, dtype=torch.float)
    dones = torch.tensor(done_list, dtype=torch.float)

    return batched_obs, actions, rewards, batched_next_obs, dones

def TGCN_collate_fn(batch):
    nested_obs_list, actions, rewards, nested_next_obs_list, done_list = zip(*batch)

    batched_obs = nested_obs_list
    batched_next_obs = nested_next_obs_list
    actions = torch.tensor(actions, dtype=torch.float)
    rewards = torch.tensor(rewards, dtype=torch.float)
    done_list = torch.tensor(done_list, dtype=torch.float)

    return batched_obs, actions, rewards, batched_next_obs, done_list