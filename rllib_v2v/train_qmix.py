from ray.rllib.algorithms.qmix import QMixConfig
from env import V2V_Env_ray_OMNET_qmix
from ray.rllib.models import ModelCatalog
from torch_q_model import TorchQModel
import os
import yaml
import ray
from ray import tune
from ray.tune import run_experiments, register_env
from gym.spaces import Tuple


def load_config():
    with open(os.path.join('config', 'config_5agent.yaml'), 'r') as f:
        return yaml.load(f, Loader=yaml.loader.SafeLoader)

if __name__ == '__main__':

    # Register model
    ModelCatalog.register_custom_model(
        "new_model", TorchQModel
    )

    configs = load_config()
    # config = QMixConfig()
    # config.model = {"custom_model": "new_model"}
    # config.environment(env=V2V_Env_ray, env_config=configs['environment'])
    # print(config.to_dict())
    # # Build an Algorithm object from the config and run 1 training iteration.
    # algo = config.build(env=V2V_Env_ray)
    # algo.train()
    def env_creator(cfs):
        env = V2V_Env_ray_OMNET_qmix(cfs)
        agent_list = list(env._agent_ids)
        grouping = {
            "group_1": agent_list,
        }
        print(agent_list)
        obs_space = Tuple([env.observation_space[i] for i in agent_list])
        act_space = Tuple([env.action_space[i] for i in agent_list])

        print(obs_space)
        return env.with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space
        )


    ray.init()
    register_env("rllib_v2v", env_creator)

    run_experiments(
        {
            "rllib_v2v": {
                "run": "QMIX",
                "env": "rllib_v2v",
                "stop": {
                    "training_iteration": configs['train']['stop_timesteps'],
                },
                "config": {
                    "num_workers": 1,
                    "env_config": configs['environment'],
                    "disable_env_checking": True,
                    "model": {
                        "custom_model": "new_model",
                        "custom_model_config": {"hidden_dim": 32,
                                                "n_agents": 5}
                    }
                },
            },
        }
    )

