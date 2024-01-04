from Strategy.RL_strategy.utility.torchUtils import GetNodeFeature
from gymnasium.spaces import Box, Dict
from .LLVMEnv import LLVMEnv
import gymnasium as gym
import numpy as np
import torch

class GNNLLVMEnv(LLVMEnv, gym.Env):
    def __init__(self, config):
        super(GNNLLVMEnv, self).__init__(config)

        self._pass_features = GetNodeFeature(self._ll_code,obs_type=self._obs_type,action_space=self._llvm_version,llvm_tools_path=self._llvm_tools_path)
        
    def _get_input_dim(self, ll_code):
        return len(np.array([value for value in self._get_features(self._ll_code).values()], dtype=np.float32)) + 1
    
    def reset(self, *, seed=None, options=None):
        self._reward = 0
        self._steps = -1
        self._current_perf = self.baseline_perf
        self._applied_passes = []
        self._optimization_flags = ""
        self._state = self.init_state()

        return self._state, {}

    def update_state(self, action_idx):
        action = self._get_action_name(action_idx)
        features = self._pass_features[action.name]
        features_vector = torch.tensor([value for value in features.values()], dtype=torch.float)
        new_value = torch.tensor([self._steps], dtype=torch.float)
        features_vector = torch.cat((features_vector, new_value))
        features_vector_np = features_vector.numpy()
        self._process_obs_model(self._state, self._steps, features_vector_np)

    def _process_obs_model(self, state, steps, features_vector):
        state["nodes_features"][steps + 1] = features_vector
        if steps >= 0:
            new_edge = torch.tensor([[steps], [steps + 1]], dtype=torch.long).numpy()
            state["edge_index"][:, steps] = new_edge[:, 0]

    def is_terminated(self, steps, max_steps):
        return steps >= max_steps - 1

    def init_state(self):
        return self._get_init_state(self._max_steps, self.feature_dim, self._original_ll_code)
    
    def _get_init_state(self, max_steps, feature_dim, ll_code):
        x = np.zeros((max_steps + 1, feature_dim), dtype=np.float32)
        features_np = np.array([value for value in self._get_features(ll_code).values()], dtype=np.float32)
        features_np = np.append(features_np, -1)
        x[0] = features_np
        edge_index = np.zeros((2, max_steps), dtype=np.int64)
        return {"nodes_features": x, "edge_index": edge_index}

    def _get_observation_space(self):
        return Dict({"nodes_features": Box(low=float('-inf'), high=float('inf'), shape=(self._max_steps + 1, self.feature_dim), dtype=np.float32),
                                       "edge_index": Box(low=0, high=self._max_steps, shape=(2,self._max_steps), dtype=np.int64)})  # TODO: add edge feature
        