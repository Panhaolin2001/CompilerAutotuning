from Strategy.RL_strategy.CompilerEnv import CompilerEnv
from Strategy.RL_strategy.train import TrainManager
from Strategy.common import set_llvm_tools_path

# MLP/GCN/GRNN perform good

if __name__ == '__main__':
    set_llvm_tools_path(bin_file="/wafer/phl/project/wafer-compiler/3rdparty/llvm/build-08d094a/bin/")
    env = CompilerEnv(ll_file="/wafer/phl/project/Compiler_autotuning/benchmark/src/result.ll", 
                      max_steps=10, 
                      state_type="GRNN")
    train_manager = TrainManager(env=env, episodes=3000, e_greed=0.1)
    train_manager.train()