from ..common import get_instrcount, get_codesize, get_runtime_internal, compile_cpp_to_ll
from .actionspace.llvm16.actions import Actions_LLVM_16
from .actionspace.llvm14.actions import Actions_LLVM_14
from .actionspace.llvm10.actions import Actions_LLVM_10
from .actionspace.CompilerGymLLVMv0.actions import Actions_LLVM_10_0_0
from .obsUtility.InstCount import get_inst_count_obs
from .utility.torchUtils import GetNodeFeature
from torch_geometric.data import Data
import torch
import copy

class CompilerEnv:
    def __init__(self,source_file,is_wafer=False,wafer_lower_pass_options=None,max_steps=20,agent_type="DQN",obs_model='MLP',reward_type="IRInstCount",obs_type="P2VInstCount",action_space="llvm-16.x"):
        self.ll_file = compile_cpp_to_ll(source_file, ll_file_dir=None, is_wafer=is_wafer,wafer_lower_pass_options=wafer_lower_pass_options)
        self.reward_type = reward_type
        self.obs_type = obs_type
        self.agent_type = agent_type
        self.action_space = action_space
        self.Actions = 0
        self.baseline_perf = 0
        self.epsilon = 0
        self.obs_model = obs_model
        self.optimization_flags = None
        self.max_steps = max_steps
        self.pass_features = GetNodeFeature(self.ll_file, obs_type=self.obs_type, action_space=self.action_space)
        self.feature_dim = len(self.pass_features[next(iter(self.pass_features))]) + 1
        self.state = None
        self.list = []

        print(f"-- Using Source File: {source_file}")
        print(f"-- Using Wafer: {is_wafer}")
        print(f"-- Using Wafer Lower Pass Options: {wafer_lower_pass_options}")
        print(f"-- Max Steps: {max_steps}")
        print(f"-- Using Agent: {agent_type}")
        print(f"-- Using Model: {obs_model}")
        print(f"-- Using Reward Type: {reward_type}")
        print(f"-- Using Obs Type: {obs_type}")
        print(f"-- Using Action Space: {action_space}")
        print()

        match self.action_space:
            case "llvm-16.x":
                self.Actions = Actions_LLVM_16
            case "llvm-14.x":
                self.Actions = Actions_LLVM_14
            case "llvm-10.x":
                self.Actions = Actions_LLVM_10
            case "llvm-10.0.0":
                self.Actions = Actions_LLVM_10_0_0
            case _:
                raise ValueError(f"Unknown action space: {self.action_space}, please choose 'llvm-16.x','llvm-14.x','llvm-10.x','llvm-10.0.0' ")

        self.n_act = len(self.Actions)

        match self.reward_type:
            case "IRInstCount":
                self.baseline_perf = get_instrcount(self.ll_file, "-Oz")
            case "CodeSize":
                self.baseline_perf = get_codesize(self.ll_file, "-Oz")
            case "RunTime":
                self.baseline_perf = get_runtime_internal(self.ll_file, "-O3")
            case _:
                raise ValueError(f"Unknown reward type: {self.reward_type}, please choose 'IRInstCount','CodeSize','RunTime'")

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
        features = self.pass_features[action.name]
        features_vector = torch.tensor([value for value in features.values()], dtype=torch.float)
        new_value = torch.tensor([self.steps], dtype=torch.float)
        features_vector = torch.cat((features_vector, new_value))

        match self.obs_model:
            case "GCN":
                self.state.x[self.steps + 1] = features_vector
                if self.steps >= 1:
                    new_edge = torch.tensor([[self.steps], [self.steps + 1]], dtype=torch.long)
                    self.state.edge_index = torch.cat([self.state.edge_index, new_edge], dim=1)

            case "MLP":
                self.state += features_vector
                self.state /= torch.tensor([self.steps], dtype=torch.float)

            case "Transformer":
                self.state[self.steps] = features_vector

            case "T-GCN" | "GRNN" :
                data = copy.deepcopy(self.state[-1])
                data.x[self.steps] = features_vector
                if self.steps >= 1:
                    new_edge = torch.tensor([[self.steps - 1], [self.steps]], dtype=torch.long)
                    data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
                self.datalist.append(data)

            case _:
                raise ValueError(f"Unknown obs model: {self.obs_model}, please choose 'GCN', 'MLP', 'Transformer', 'T-GCN', 'GRNN' ")
            
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

        if self.action_space != "llvm-10.0.0" and self.action_space != "llvm-10.x":
            self.optimization_flags = "--enable-new-pm=0 " + " ".join([act.value for act in self.applied_passes])
        else:
            self.optimization_flags = " ".join([act.value for act in self.applied_passes])

        match self.reward_type:
            case "IRInstCount":
                current_perf = get_instrcount(self.ll_file, self.optimization_flags)
            case "CodeSize":
                current_perf = get_codesize(self.ll_file, self.optimization_flags)
            case "RunTime":
                current_perf = get_runtime_internal(self.ll_file, self.optimization_flags)
            case _:
                raise ValueError(f"Unknown reward type: {self.reward_type}, please choose 'IRInstCount','CodeSize','RunTime'")

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
        match self.obs_model:
            case "GCN":
                x = torch.zeros((self.max_steps + 1, self.feature_dim), dtype=torch.float) # add 1 means adding a program global feature vector.
                features_vector = torch.tensor([value for value in get_inst_count_obs(self.ll_file, self.action_space).values()], dtype=torch.float)
                new_value = torch.tensor([-1], dtype=torch.float)
                features_vector = torch.cat((features_vector, new_value))
                x[0] = features_vector
                edge_index = torch.empty((2, 0), dtype=torch.long)
                return Data(x=x, edge_index=edge_index)
            
            case "MLP":
                return torch.zeros((self.feature_dim), dtype=torch.float)

            case "Transformer":
                return torch.zeros((self.max_steps, self.feature_dim), dtype=torch.float)

            case "T-GCN" | "GRNN":
                x = torch.zeros((self.max_steps, self.feature_dim), dtype=torch.float)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                data = Data(x=x, edge_index=edge_index)
                self.datalist.append(data)
                return self.datalist

            case _:
                raise ValueError(f"Unknown action space: {self.obs_model}, please choose 'GCN', 'MLP', 'Transformer', 'T-GCN', 'GRNN' ")
            