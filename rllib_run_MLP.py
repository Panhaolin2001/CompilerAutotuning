import ray
from ray.rllib.models import ModelCatalog
from rllib_example.env import CompilerEnv
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray import tune
from ray import air
from ray.tune.registry import register_env

def env_creator(env_config):
     return CompilerEnv(env_config)

if __name__ == "__main__":

    ray.init(include_dashboard=True, ignore_reinit_error=True)

    env_name = 'CompilerGYM'
    env_config={
            "source_file": "/wafer/phl/project/compiler_autotuning/ll_file/14.x/a.cpp",
            "is_wafer": False,
            "wafer_tools_path": "/wafer/phl/project/wafer-compiler/build/bin",
            "wafer_lower_pass_options": ["-I", "/usr/lib/gcc/x86_64-linux-gnu/11/include",
                                                "--JsonFilePath", "/wafer/phl/project/wafer-compiler/design.json",
                                                "--lower-crypto-multi-x86", "--lower-cryptosha1-without-SHA1ISA"],
            "max_steps": 10,
            "obs_model": "MLP",
            "reward_type": "IRInstCount",
            "obs_type": "P2VIR2VSym",
            "action_space": "llvm-14.x",
            "llvm_tools_path": "/wafer/phl/project/wafer-compiler/3rdparty/llvm/build-14.x/bin/",
            "isPass2Vec": True,
    }

    register_env('CompilerGYM', lambda config: env_creator(env_config))

    env = CompilerEnv(env_config)
    
    replay_config = {
            "capacity": 2000,
            "prioritized_replay_alpha": 0.5,
            "prioritized_replay_beta": 0.5,
            "prioritized_replay_eps": 3e-6,
        }

    explore_config = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.5,
            "final_epsilon": 0.01,
        }

    algo = DQNConfig()\
        .environment(env=env_name, disable_env_checking=True)\
        .framework("torch")\
        .training(
            replay_buffer_config=replay_config,
            _enable_learner_api=False,
            lr=0.001,
            gamma=0.9,
            # dueling=False,
        )\
        .rollouts(num_rollout_workers=0)\
        .rl_module( _enable_rl_module_api=False)\
        .exploration(exploration_config=explore_config)


    tune.Tuner(  
            "DQN",
            run_config=air.RunConfig(stop={
                #  "episode_reward_mean": 0.2, 
                 "episodes_total": 1000
                 }),
            param_space=algo
        ).fit()
