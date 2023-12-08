from ..common import get_instrcount, Actions, get_codesize, get_runtime_internal
from .utility.torchUtils import GetFeature
from torch_geometric.data import Data
import torch
import copy

class CompilerEnv:
    def __init__(self, ll_file, max_steps=20, obs_model='MLP', reward_type="InstCount", obs_type="pass2vec"):
        self.ll_file = ll_file
        self.reward_type = reward_type
        self.obs_type = obs_type
        self.baseline_perf = 0

        if self.reward_type == "InstCount":
            self.baseline_perf = get_instrcount(ll_file, "-Oz")
        elif self.reward_type == "CodeSize":
            self.baseline_perf = get_codesize(ll_file, "-Oz")
        elif self.reward_type == "RunTime":
            self.baseline_perf = get_runtime_internal(ll_file, "-O3")

        self.epsilon = 0
        self.obs_model = obs_model
        self.max_steps = max_steps
        self.pass_features = GetFeature(ll_file, obs_type=self.obs_type)
        self.feature_dim = len(self.pass_features[next(iter(self.pass_features))]) + 1
        self.n_act = len(Actions)
        self.state = None
        self.list = []

    def update_graph_state(self, action):
        '''
        更新状态
            1. GCN
                获取每个pass的特征向量 -> 造新的特征值new_value并添加到特征向量中
                -> 更新图的节点向量特征 -> 更新图的边特征
            
            2.  MLP
                获取每个pass的特征向量 -> 造新的特征值new_value并添加到特征向量中
                -> 更新节点向量特征(求均值)
        '''
        action_idx = list(Actions).index(action)
        features = self.pass_features[action.name]
        features_vector = torch.tensor([value for value in features.values() if isinstance(value, (int, float))], dtype=torch.float)
        new_value = torch.tensor([(self.steps) / self.max_steps], dtype=torch.float)
        features_vector = torch.cat((features_vector, new_value))

        if self.obs_model == "GCN":
            self.state.x[self.steps] = features_vector
            if self.steps >= 2:
                new_edge = torch.tensor([[self.steps - 1], [self.steps]], dtype=torch.long)
                self.state.edge_index = torch.cat([self.state.edge_index, new_edge], dim=1)

        elif self.obs_model == "MLP":
            self.state += features_vector
            self.state /= torch.tensor([self.steps], dtype=torch.float)

        elif self.obs_model == "GRNN":
            data = copy.deepcopy(self.state[-1])
            data.x[self.steps] = features_vector
            if self.steps >= 1:
                new_edge = torch.tensor([[self.steps - 1], [self.steps]], dtype=torch.long)
                data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
            self.datalist.append(data)
            
            # data = copy.deepcopy(self.state[-1])
            # data.x = torch.cat([data.x, features_vector.unsqueeze(0)], dim=0)
            # if self.steps >= 1:
            #     new_edge = torch.tensor([[self.steps - 1], [self.steps]], dtype=torch.long)
            #     data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
            # self.datalist.append(data)
        
        elif self.obs_model == "Transformer":
            self.state[self.steps] = features_vector

        elif self.obs_model == "T-GCN":
            # self.state.x[self.steps] = features_vector
            # if self.steps >= 2:
            #     new_edge = torch.tensor([[self.steps - 1], [self.steps]], dtype=torch.long)
            #     self.state.edge_index = torch.cat([self.state.edge_index, new_edge], dim=1)
            data = copy.deepcopy(self.state[-1])
            data.x[self.steps] = features_vector
            if self.steps >= 1:
                new_edge = torch.tensor([[self.steps - 1], [self.steps]], dtype=torch.long)
                data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
            self.datalist.append(data)
            
    def step(self, action):
        '''
        1. applied_passes_set用于不要让后续的pass和前面的pass重复
        2. reward设置为此次状态下的性能比baseline多了多少百分比
        3. 设置done的结束条件为达到最大step数
        '''
        self.steps += 1
        self.applied_passes.append(action)
        self.applied_passes_set.add(action)
        self.update_graph_state(action)

        optimization_flags = "--enable-new-pm=0 " + " ".join([act.value for act in self.applied_passes])

        if self.reward_type == "InstCount":
            current_perf = get_instrcount(self.ll_file, optimization_flags)
        elif self.reward_type == "CodeSize":
            current_perf = get_codesize(self.ll_file, optimization_flags)
        elif self.reward_type == "RunTime":
            current_perf = get_runtime_internal(self.ll_file, optimization_flags)

        # self.reward = (self.baseline_perf / (current_perf + self.epsilon)) - 1
        self.reward = (self.current_perf - current_perf) / self.baseline_perf
        self.current_perf = current_perf

        done = self.steps >= self.max_steps - 1 
        return self.reward, copy.deepcopy(self.state), done

    def reset(self):
        '''
        重置参数和状态
        '''
        self.reward = 0
        self.steps = -1
        self.current_perf = self.baseline_perf
        self.applied_passes = []
        self.datalist = []
        self.applied_passes_set = set()
        self.state = self.init_state()

        return self.state

    def init_state(self):
        '''
        初始化状态
        '''
        if self.obs_model == "GCN":
            x = torch.zeros((self.max_steps, self.feature_dim), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index)
        
        elif self.obs_model == "MLP":
            return torch.zeros((self.feature_dim), dtype=torch.float)
        
        elif self.obs_model == "GRNN":
            x = torch.zeros((self.max_steps, self.feature_dim), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            self.datalist.append(data)
            return self.datalist

        elif self.obs_model == "Transformer":
            return torch.zeros((self.max_steps, self.feature_dim), dtype=torch.float)
        
        elif self.obs_model == "T-GCN":
            # x = torch.zeros((self.max_steps, self.feature_dim), dtype=torch.float)
            # edge_index = torch.empty((2, 0), dtype=torch.long)
            # return Data(x=x, edge_index=edge_index)
        
            x = torch.zeros((self.max_steps, self.feature_dim), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            self.datalist.append(data)
            return self.datalist