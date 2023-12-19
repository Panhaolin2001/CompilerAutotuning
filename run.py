from Strategy.RL_strategy.CompilerEnv import CompilerEnv
from Strategy.RL_strategy.train import TrainManager
from Strategy.common import set_llvm_tools_path, set_wafer_tools_path

# MLP/GCN/GRNN perform good
# wafer only adapts llvm-16.x

'''
TODO: 
    1. Benchmark Set : CBench; 
    2. Other Obs Types: IR2Vec, AutoPhase;
    3. Pass2Vec: 解决泛化性的问题
    4. Preprocess Stage: 比如聚类等
    5. codesize,instcount,runtime的计算方式需要和compilergym一致才可比较
    6. Pass2Vec: 目前只支持llvm10.0.0,因为内部使用的是compilergym的obs,
                 需要支持其他的llvm版本(具体应该实现一下内部的compilergym的obs)
    7. 移植ray的agent
'''

if __name__ == '__main__':
    env = CompilerEnv(source_file="/PATH/ll_file",
                      is_wafer=False,
                      wafer_lower_pass_options=["-I", "/usr/lib/gcc/x86_64-linux-gnu/11/include",
                                                "--JsonFilePath", "/PATH/design.json",
                                                "--lower-crypto-multi-x86", "--lower-crypto-to-x86"],
                      max_steps=10,
                      obs_model="GRNN",
                      reward_type="CodeSize",
                      obs_type="P2VAutoPhase",
                      action_space="llvm-16.x",
                      llvm_tools_path="/PATH/llvm-16.X/build/bin/",
                      )
    train_manager = TrainManager(env=env, episodes=3000, e_greed=0.1)
    train_manager.train()