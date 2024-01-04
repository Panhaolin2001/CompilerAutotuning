from Strategy.RL_strategy.obsUtility.IR2Vec import get_ir2vec_fa_obs
from Strategy.RL_strategy.obsUtility.IR2Vec import get_ir2vec_sym_obs
from Strategy.RL_strategy.obsUtility.Autophase import get_autophase_obs
from Strategy.RL_strategy.obsUtility.InstCount import get_inst_count_obs
from Strategy.RL_strategy.actionspace.llvm16.actions import Actions_LLVM_16
from Strategy.RL_strategy.actionspace.llvm14.actions import Actions_LLVM_14
from Strategy.RL_strategy.actionspace.llvm10.actions import Actions_LLVM_10
from Strategy.RL_strategy.actionspace.CompilerGymLLVMv0.actions import Actions_LLVM_10_0_0
from Strategy.common import get_instrcount, get_codesize, get_runtime_internal, compile_cpp_to_ll, GenerateOptimizedLLCode

from gymnasium.spaces import Discrete, Box
import gymnasium as gym
import numpy as np
import shlex

class LLVMEnv(gym.Env):
    def __init__(self, config):
        super(LLVMEnv, self).__init__()
        self._config = config
        self._llvm_tools_path = self._config['llvm_tools_path']
        self._llvm_version = self._config['llvm_version']
        self._reward_type = self._config['reward_space']
        self._reward_baseline = self._config['reward_baseline']
        self._obs_type = self._config['observation_type']
        self._max_steps = self._config['max_steps']
        self._ll_code = self.benchmark(self._config['source_file'])
        self._original_ll_code = self._ll_code

        self.old_pm_llvm_versions = ["llvm-10.0.0", "llvm-10.x"]
        self.output_dim = self._get_output_dim(self._llvm_version)
        self.action_space = Discrete(self.output_dim)
        self.Actions = self._select_actions(self._llvm_version)
        self.baseline_perf = self.calculate_baseline_perf(self._original_ll_code, self._reward_baseline)
        self.feature_dim = self._get_input_dim(self._original_ll_code)
        self.observation_space = self._get_observation_space()
        
    def step(self, action_idx):
        self._steps += 1
        self._applied_passes.append(self._get_action_name(action_idx))

        self._optimization_flags = (
            "--enable-new-pm=0 " + " ".join([act.value for act in self._applied_passes])
            if self._llvm_version not in self.old_pm_llvm_versions
            else " ".join([act.value for act in self._applied_passes])
        )

        self.update_state(action_idx)

        current_perf = self.calculate_current_perf(self._original_ll_code, self._reward_type, self._optimization_flags)
        self._reward = (self._current_perf - current_perf) / self.baseline_perf
        self._current_perf = current_perf

        terminated = self.is_terminated(self._steps, self._max_steps)
        return self._state, self._reward, terminated, False, {}

    def is_terminated(self, steps, max_steps):
        return steps >= max_steps

    def benchmark(self, source_file):
        benchmark = compile_cpp_to_ll(source_file, llvm_tools_path=self._llvm_tools_path)
        return benchmark

    def reset(self, *, seed=None, options=None):
        self._reward = 0
        self._steps = 0
        self._current_perf = self.baseline_perf
        self._applied_passes = []
        self._datalist = []
        self._optimization_flags = ""
        self._state = self.init_state()

        return self._state, {}
    
    def get_input_dim(self):
        return self.feature_dim
    
    def _get_input_dim(self, ll_code):
        return len(np.array([value for value in self._get_features(ll_code).values()], dtype=np.float32))
    
    def get_output_dim(self):
        return self.output_dim

    def _get_output_dim(self, llvm_version):
        return len({
            "llvm-16.x": list(Actions_LLVM_16),
            "llvm-14.x": list(Actions_LLVM_14),
            "llvm-10.x": list(Actions_LLVM_10),
            "llvm-10.0.0": list(Actions_LLVM_10_0_0),
        }.get(llvm_version, []))

    def _get_observation_space(self):
        return Box(low=float('-inf'), high=float('inf'), shape=(self.feature_dim,), dtype=np.float32)

    def _select_actions(self, llvm_version):
        action_space_mappings = {
            "llvm-16.x": Actions_LLVM_16,
            "llvm-14.x": Actions_LLVM_14,
            "llvm-10.x": Actions_LLVM_10,
            "llvm-10.0.0": Actions_LLVM_10_0_0
        }
        selected_actions = action_space_mappings.get(llvm_version)
        if selected_actions is None:
            raise ValueError(f"Unknown action space: {llvm_version}, please choose 'llvm-16.x', 'llvm-14.x', 'llvm-10.x', 'llvm-10.0.0' ")
        return selected_actions

    def calculate_baseline_perf(self, ll_code, reward_baseline):
        reward_functions = {
            "IRInstCountOz": lambda: get_instrcount(ll_code, "-Oz", llvm_tools_path=self._llvm_tools_path),
            "IRInstCountO3": lambda: get_instrcount(ll_code, "-O3", llvm_tools_path=self._llvm_tools_path),
            "IRInstCountO2": lambda: get_instrcount(ll_code, "-O2", llvm_tools_path=self._llvm_tools_path),
            "IRInstCountO1": lambda: get_instrcount(ll_code, "-O1", llvm_tools_path=self._llvm_tools_path),
            "IRInstCountO0": lambda: get_instrcount(ll_code, "-O0", llvm_tools_path=self._llvm_tools_path),
            "CodeSizeOz": lambda: get_codesize(ll_code, "-Oz", llvm_tools_path=self._llvm_tools_path),
            "CodeSizeO3": lambda: get_codesize(ll_code, "-O3", llvm_tools_path=self._llvm_tools_path),
            "CodeSizeO2": lambda: get_codesize(ll_code, "-O2", llvm_tools_path=self._llvm_tools_path),
            "CodeSizeO1": lambda: get_codesize(ll_code, "-O1", llvm_tools_path=self._llvm_tools_path),
            "CodeSizeO0": lambda: get_codesize(ll_code, "-O0", llvm_tools_path=self._llvm_tools_path),
            "RunTimeOz": lambda: get_runtime_internal(ll_code, "-Oz", llvm_tools_path=self._llvm_tools_path),
            "RunTimeO3": lambda: get_runtime_internal(ll_code, "-O3", llvm_tools_path=self._llvm_tools_path),
            "RunTimeO2": lambda: get_runtime_internal(ll_code, "-O2", llvm_tools_path=self._llvm_tools_path),
            "RunTimeO1": lambda: get_runtime_internal(ll_code, "-O1", llvm_tools_path=self._llvm_tools_path),
            "RunTimeO0": lambda: get_runtime_internal(ll_code, "-O0", llvm_tools_path=self._llvm_tools_path),
        }

        reward_function = reward_functions.get(reward_baseline)
        if reward_function is None:
            raise ValueError(f"Unknown reward_baseline: {reward_baseline}, please choose 'IRInstCountOz', 'IRInstCountO3', \
                             'IRInstCountO2','IRInstCountO1','IRInstCountO0','CodeSizeOz', 'CodeSizeO3','CodeSizeO2', \
                             'CodeSizeO1','CodeSizeO0','RunTimeOz','RunTimeO3','RunTimeO2','RunTimeO1','RunTimeO0'")

        baseline_perf = reward_function()
        return baseline_perf

    def calculate_current_perf(self, ll_code, reward_type, optimization_flags):
        perf_functions = {
            "IRInstCount": lambda: get_instrcount(ll_code, optimization_flags, llvm_tools_path=self._llvm_tools_path),
            "CodeSize": lambda: get_codesize(ll_code, optimization_flags, llvm_tools_path=self._llvm_tools_path),
            "RunTime": lambda: get_runtime_internal(ll_code, optimization_flags, llvm_tools_path=self._llvm_tools_path)
        }

        perf_function = perf_functions.get(reward_type)
        if perf_function is None:
            raise ValueError(f"Unknown reward type: {reward_type}, please choose 'IRInstCount', 'CodeSize', 'RunTime'")

        current_perf = perf_function()
        return current_perf

    def update_state(self, action_idx):
        self._ll_code = GenerateOptimizedLLCode(self._original_ll_code, shlex.split(self._optimization_flags), self._llvm_tools_path)
        self._state = np.array([value for value in self._get_features(self._ll_code).values()], dtype=np.float32)

    def _get_action_name(self, action_idx):
        return list(self.Actions)[action_idx]

    def _get_features(self, ll_code):
        return {
            "InstCount": lambda: get_inst_count_obs(ll_code, self._llvm_version),
            "AutoPhase": lambda: get_autophase_obs(ll_code, self._llvm_version),
            "IR2VecFa": lambda: get_ir2vec_fa_obs(ll_code),
            "IR2VecSym": lambda: get_ir2vec_sym_obs(ll_code)
        }.get(self._obs_type, [])()

    def init_state(self):
        return np.array([value for value in self._get_features(self._ll_code).values()], dtype=np.float32)
