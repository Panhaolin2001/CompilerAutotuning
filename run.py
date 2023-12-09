from Strategy.RL_strategy.CompilerEnv import CompilerEnv
from Strategy.RL_strategy.train import TrainManager
from Strategy.common import set_llvm_tools_path, set_wafer_tools_path

# MLP/GCN/GRNN perform good

if __name__ == '__main__':
    set_llvm_tools_path(bin_file="/wafer/phl/project/wafer-compiler/3rdparty/llvm/build-16.x/bin/")
    set_wafer_tools_path(bin_file="/wafer/phl/project/wafer-compiler/build/bin")
    env = CompilerEnv(source_file="/wafer/phl/project/wafer-compiler/tools/waferfrontend/test/foo.cpp",
                      is_wafer=True,
                      wafer_lower_pass_options=["--lower-crypto-multi-x86", "--lower-crypto-to-x86"],
                      max_steps=10,
                      obs_model="GRNN",
                      reward_type="RunTime",
                      obs_type="pass2vec",
                      action_space="llvm-16.x")
    train_manager = TrainManager(env=env, episodes=3000, e_greed=0.1)
    train_manager.train()