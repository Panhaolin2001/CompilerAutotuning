from Strategy.RL_strategy.utility.torchUtils import GetNodeFeature
from Strategy.RL_strategy.obsUtility.IR2Vec import get_ir2vec_fa_obs
from Strategy.RL_strategy.obsUtility.IR2Vec import get_ir2vec_sym_obs
from Strategy.RL_strategy.obsUtility.Autophase import get_autophase_obs
from Strategy.RL_strategy.obsUtility.InstCount import get_inst_count_obs
from Strategy.RL_strategy.actionspace.llvm16.actions import Actions_LLVM_16
from Strategy.RL_strategy.actionspace.llvm14.actions import Actions_LLVM_14
from Strategy.RL_strategy.actionspace.llvm10.actions import Actions_LLVM_10
from Strategy.RL_strategy.actionspace.CompilerGymLLVMv0.actions import Actions_LLVM_10_0_0
from Strategy.common import get_instrcount, get_codesize, get_runtime_internal, compile_cpp_to_ll

from gymnasium.spaces import Discrete, Box, Dict
from torch_geometric.data import Data
import gymnasium as gym
import numpy as np
import copy
import torch

class CompilerEnv(gym.Env):
    def __init__(self, config):
        super(CompilerEnv, self).__init__()
        self._config = config
        self._ll_file = compile_cpp_to_ll(self._config['source_file'], ll_file_dir=None, is_wafer=self._config['is_wafer'],wafer_lower_pass_options=self._config['wafer_lower_pass_options'], 
                                          llvm_tools_path=self._config['llvm_tools_path'], wafer_tools_path=self._config['wafer_tools_path'] )
        self._llvm_tools_path = self._config['llvm_tools_path']
        self._reward_type = self._config['reward_type']
        self._obs_type = self._config['obs_type']
        self.action_space = Discrete(self.get_output_dim())
        self._Actions = self._select_actions()
        self._baseline_perf = self._calculate_baseline_perf()
        self._obs_model = self._config['obs_model']
        self._max_steps = self._config['max_steps']
        self._pass_features = GetNodeFeature(self._ll_file,obs_type=self._obs_type,action_space=self._config['action_space'],llvm_tools_path=self._config['llvm_tools_path'])
        self._feature_dim = self._get_node_feature_dim()
        self.observation_space = self._get_observation_space()
        self._state = None

    def step(self, action_idx):
        self._steps += 1
        self._applied_passes.append(self._get_action_name(action_idx))
        self._update_graph_state(action_idx)

        allowed_versions = ["llvm-10.0.0", "llvm-10.x"]
        optimization_flags = (
            "--enable-new-pm=0 " + " ".join([act.value for act in self._applied_passes])
            if self._config['action_space'] not in allowed_versions
            else " ".join([act.value for act in self._applied_passes])
        )

        current_perf = self._calculate_current_perf(optimization_flags)

        self._reward = (self._current_perf - current_perf) / self._baseline_perf
        self._current_perf = current_perf

        terminated = self._steps >= self._max_steps - 1 
        return self._state, self._reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        self._reward = 0
        self._steps = -1
        self._current_perf = self._baseline_perf
        self._applied_passes = []
        self._datalist = []
        self._state = self._init_state()

        return self._state, {}
    
    def get_input_dim(self):
        return self._get_node_feature_dim()
    
    def get_output_dim(self):
        return len(self._get_actions())

    def _get_observation_space(self):
        observation_space_mappings = {
            "GCN": Dict({"nodes_features": Box(low=float('-inf'), high=float('inf'), shape=(self._max_steps + 1, self._feature_dim), dtype=np.float32),
                                       "edge_index": Box(low=0, high=self._max_steps, shape=(2,self._max_steps), dtype=np.int64)}),  # TODO: add edge feature.

            "MLP": None,         # TODO: add MLP observation space.
            "Transformer": None, # TODO: add Transformer observation space.
            "T-GCN": None,       # TODO: add T-GCN observation space.
            "GRNN": None,        # TODO: add GRNN observation space.
        }
        selected_observation_space = observation_space_mappings.get(self._config['obs_model'])
        if selected_observation_space is None:
            raise ValueError(f"Unknown obs model: {selected_observation_space}")
        return selected_observation_space

    def _get_node_feature_dim(self):
        return len(self._pass_features[next(iter(self._pass_features))]) + 1

    def _select_actions(self):
        action_space_mappings = {
            "llvm-16.x": Actions_LLVM_16,
            "llvm-14.x": Actions_LLVM_14,
            "llvm-10.x": Actions_LLVM_10,
            "llvm-10.0.0": Actions_LLVM_10_0_0
        }
        selected_actions = action_space_mappings.get(self._config['action_space'])
        if selected_actions is None:
            raise ValueError(f"Unknown action space: {self._config['action_space']}, please choose 'llvm-16.x', 'llvm-14.x', 'llvm-10.x', 'llvm-10.0.0' ")
        return selected_actions

    def _calculate_baseline_perf(self):
        reward_functions = {
            "IRInstCount": lambda: get_instrcount(self._ll_file, "-Oz", llvm_tools_path=self._llvm_tools_path),
            "CodeSize": lambda: get_codesize(self._ll_file, "-Oz", llvm_tools_path=self._llvm_tools_path),
            "RunTime": lambda: get_runtime_internal(self._ll_file, "-O3", llvm_tools_path=self._llvm_tools_path)
        }

        reward_function = reward_functions.get(self._config['reward_type'])
        if reward_function is None:
            raise ValueError(f"Unknown reward type: {self._config['reward_type']}, please choose 'IRInstCount', 'CodeSize', 'RunTime'")

        baseline_perf = reward_function()
        return baseline_perf

    def _calculate_current_perf(self, optimization_flags):
        perf_functions = {
            "IRInstCount": lambda: get_instrcount(self._ll_file, optimization_flags, llvm_tools_path=self._llvm_tools_path),
            "CodeSize": lambda: get_codesize(self._ll_file, optimization_flags, llvm_tools_path=self._llvm_tools_path),
            "RunTime": lambda: get_runtime_internal(self._ll_file, optimization_flags, llvm_tools_path=self._llvm_tools_path)
        }

        perf_function = perf_functions.get(self._config['reward_type'])
        if perf_function is None:
            raise ValueError(f"Unknown reward type: {self._config['reward_type']}, please choose 'IRInstCount', 'CodeSize', 'RunTime'")

        current_perf = perf_function()
        return current_perf

    def _update_graph_state(self, action_idx):
        action = self._get_action_name(action_idx)
        features = self._pass_features[action.name]
        features_vector = torch.tensor([value for value in features.values()], dtype=torch.float)
        new_value = torch.tensor([self._steps], dtype=torch.float)
        features_vector = torch.cat((features_vector, new_value))
        features_vector_np = features_vector.numpy()

        self._process_obs_model(self._state, self._steps, features_vector_np, self._datalist)    

    def _process_obs_model(self, state, steps, features_vector, datalist):
        obs_model_functions = {
            "GCN": lambda: self._process_gcn(state, steps, features_vector),
            "MLP": lambda: self._process_mlp(state, steps, features_vector),
            "Transformer": lambda: self._process_transformer(state, steps, features_vector),
            "T-GCN": lambda: self._process_tgcn(state, steps, features_vector, datalist),
            "GRNN": lambda: self._process_grnn(state, steps, features_vector, datalist),
        }

        obs_model_function = obs_model_functions.get(self._obs_model)
        if obs_model_function is None:
            raise ValueError(f"Unknown obs model: {self._obs_model}, please choose 'GCN', 'MLP', 'Transformer', 'T-GCN', 'GRNN' ")

        obs_model_function()

    def _process_gcn(self, state, steps, features_vector):
        state["nodes_features"][steps + 1] = features_vector
        if steps >= 1:
            new_edge = torch.tensor([[steps], [steps + 1]], dtype=torch.long).numpy()
            state["edge_index"][:, steps] = new_edge[:, 0]

    def _process_mlp(self, state, steps, features_vector):
        state += features_vector
        state /= torch.tensor([steps], dtype=torch.float)

    def _process_transformer(self, state, steps, features_vector):
        state[steps] = features_vector

    def _process_tgcn(self, state, steps, features_vector, datalist):
        data = copy.deepcopy(state[-1])
        data.x[steps] = features_vector
        if steps >= 1:
            new_edge = torch.tensor([[steps - 1], [steps]], dtype=torch.long)
            data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
        datalist.append(data)

    def _process_grnn(self, state, steps, features_vector, datalist):
        data = copy.deepcopy(state[-1])
        data.x[steps] = features_vector
        if steps >= 1:
            new_edge = torch.tensor([[steps - 1], [steps]], dtype=torch.long)
            data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
        datalist.append(data)

    def _get_action_name(self, action_idx):
        return list(self._Actions)[action_idx]

    def _get_actions(self):
        return {
            "llvm-16.x": list(Actions_LLVM_16),
            "llvm-14.x": list(Actions_LLVM_14),
            "llvm-10.x": list(Actions_LLVM_10),
            "llvm-10.0.0": list(Actions_LLVM_10_0_0),
        }.get(self._config['action_space'], [])

    def _get_node_feature_type(self):
        return {
            "P2VInstCount": lambda: get_inst_count_obs(self._ll_file, self._config['action_space']),
            "P2VAutoPhase": lambda: get_autophase_obs(self._ll_file, self._config['action_space']),
            "P2VIR2VFa": lambda: get_ir2vec_fa_obs(self._ll_file),
            "P2VIR2VSym": lambda: get_ir2vec_sym_obs(self._ll_file)
        }.get(self._config['obs_type'], [])()

    def _init_state(self):
        return self._get_init_state(self._obs_model, self._max_steps, self._feature_dim, self._ll_file, self._config['action_space'], self._datalist)
            
    def _get_init_state(self, obs_model, max_steps, feature_dim, ll_file, action_space, datalist):
        init_state_functions = {
            "GCN": lambda: self._init_gcn_state(max_steps, feature_dim, ll_file, action_space),
            "MLP": lambda: self._init_mlp_state(feature_dim),
            "Transformer": lambda: self._init_transformer_state(max_steps, feature_dim),
            "T-GCN": lambda: self._init_tgcn_state(max_steps, feature_dim, datalist),
            "GRNN": lambda: self._init_grnn_state(max_steps, feature_dim, datalist),
        }

        init_state_function = init_state_functions.get(obs_model)
        if init_state_function is None:
            raise ValueError(f"Unknown obs model: {obs_model}, please choose 'GCN', 'MLP', 'Transformer', 'T-GCN', 'GRNN' ")

        return init_state_function()

    def _init_gcn_state(self, max_steps, feature_dim, ll_file, action_space):
        x = np.zeros((max_steps + 1, feature_dim), dtype=np.float32)
        features_np = np.array([value for value in self._get_node_feature_type().values()], dtype=np.float32)
        features_np = np.append(features_np, -1)
        x[0] = features_np
        edge_index = np.zeros((2, max_steps), dtype=np.int64)
        return {"nodes_features": x, "edge_index": edge_index}
    
    def _init_mlp_state(self, feature_dim):
        return torch.zeros((feature_dim,), dtype=torch.float)

    def _init_transformer_state(self, max_steps, feature_dim):
        return torch.zeros((max_steps, feature_dim), dtype=torch.float)
        
    def _init_tgcn_state(self, max_steps, feature_dim, datalist):
        x = torch.zeros((max_steps, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        datalist.append(data)
        return datalist

    def _init_grnn_state(self, max_steps, feature_dim, datalist):
        x = torch.zeros((max_steps, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        datalist.append(data)
        return datalist