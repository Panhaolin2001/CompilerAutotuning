import ray
from ray.rllib.models import ModelCatalog
from AdaptRLlib.env.LLVMEnv.env import CompilerEnv
from AdaptRLlib.custom_model.GCN_PPO import GCN
from ray.rllib.algorithms import ppo
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
            "source_file": "/wafer/phl/project/compiler_autotuning/ll_file/16.x/out.ll",
            "max_steps": 10,
            "reward_space": "IRInstCount",
            "reward_baseline": "IRInstCountOz",
            "observation_type": "P2VInstCount",
            "observation_model": "MLP",
            "llvm_version": "llvm-16.x",
            "llvm_tools_path": "/wafer/phl/project/wafer-compiler/3rdparty/llvm/build-16.x/bin/",
            "isPass2VecObs": False,
    }

    register_env('CompilerGYM', lambda config: env_creator(env_config))

    env = CompilerEnv(env_config)

    ModelCatalog.register_custom_model("gcn_model", GCN)

    algo = ppo.PPOConfig()\
        .environment(env=env_name, disable_env_checking=True)\
        .framework("torch")\
        .training(
            # model={
            #     "custom_model": "gcn_model",
            #     "custom_model_config": {"input_dim": env.get_input_dim(), "output_dim": env.get_output_dim()},
            # },
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