import ray
from ray.rllib.models import ModelCatalog
from rllib_example.env import CompilerEnv
from rllib_example.GCN_PPO import GCN
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray import tune
from ray import train
from ray import air
from ray.tune.registry import register_env

def env_creator(env_config):
     return CompilerEnv(env_config)

if __name__ == "__main__":

    ray.init(include_dashboard=True, ignore_reinit_error=True)

    env_name = 'CompilerGYM'
    env_config={
            "source_file": "/wafer/phl/project/wafer-compiler/tools/waferfrontend/test/foo.cpp",
            "is_wafer": True,
            "wafer_tools_path": "/wafer/phl/project/wafer-compiler/build/bin",
            "wafer_lower_pass_options": ["-I", "/usr/lib/gcc/x86_64-linux-gnu/11/include",
                                                "--JsonFilePath", "/wafer/phl/project/wafer-compiler/design.json",
                                                "--lower-crypto-multi-x86", "--lower-cryptosha1-without-SHA1ISA"],
            "max_steps": 10,
            "obs_model": "GCN",
            "reward_type": "CodeSize",
            "obs_type": "P2VInstCount",
            "action_space": "llvm-16.x",
            "llvm_tools_path": "/wafer/phl/project/wafer-compiler/3rdparty/llvm/build-16.x/bin/",
    }

    register_env('CompilerGYM', lambda config: env_creator(env_config))

    env = CompilerEnv(env_config)

    ModelCatalog.register_custom_model("gcn_model", GCN)

    algo = ppo.PPOConfig()\
        .environment(env=env_name, disable_env_checking=True)\
        .framework("torch")\
        .training(
            model={
                "custom_model": "gcn_model",
                "custom_model_config": {"input_dim": env.get_input_dim(), "output_dim": env.get_output_dim()},
            },
            _enable_learner_api=False
        )\
        .rollouts(num_rollout_workers=16, create_env_on_local_worker=True)\
        .rl_module( _enable_rl_module_api=False)\
    
    stop = {
        "episode_reward_mean": 0.2,
        # "episodes_total": 500
        # "timesteps_total": 10000,
        # "training_iteration": 5
    }

    tuner = tune.Tuner(
            "PPO",
            param_space=algo,
            run_config=air.RunConfig(stop=stop),
        )
    
    results = tuner.fit()