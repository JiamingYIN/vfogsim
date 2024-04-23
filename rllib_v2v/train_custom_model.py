import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import ppo, PPO, PPOTorchPolicy
from ray.tune.registry import register_env
from env import V2V_Env_ray
from ray.rllib.models import ModelCatalog
import yaml
import os
from ray.rllib.policy.policy import PolicySpec
from model import TorchCustomModel
from ray.tune.logger import pretty_print


def load_config():
    with open(os.path.join('config', 'config.yaml'), 'r') as f:
        return yaml.load(f, Loader=yaml.loader.SafeLoader)


def gen_policy(i):
    config = {
        "model": {
            "custom_model": "torch_model",
        },
    }
    return PolicySpec(config=config)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # DO NOT name the agent id with chars
    # i = int(float(agent_id)) % 2
    i = 0
    # return "policies_{}".format(i)
    return "ppo_policy"

def select_policy(algorithm, framework):
    if algorithm == "PPO":
        if framework == "torch":
            return PPOTorchPolicy


if __name__ == '__main__':
    configs = load_config()
    ray.init()

    # Register model
    ModelCatalog.register_custom_model(
        "torch_model", TorchCustomModel
    )
    env = V2V_Env_ray(configs['environment'])

    policies = {
        "ppo_policy": (
            PPOTorchPolicy,
            env.observation_space[env._agent_ids[0]],
            env.action_space[env._agent_ids[0]],
            {},
        )
    }
    print(policies)
    # policies = {
    #     "custom_policy": PolicySpec(policy_class=)
    # }
    # policies = {"policy_{}".format(i): gen_policy(i) for i in range(configs['train']['num_policies'])}
    policies_ids = list(policies.keys())

    # No tune, train with PPO
    ppo_config = configs['PPO_config']
    ppo_config['env_config'] = configs['environment']
    ppo_config['multiagent'] = {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        # "policies_to_train": ["ppo_policy"]
    }
    ppo_config['model'] = {"custom_model": "torch_model"}
    ppo_config["disable_env_checking"] = True
    trainer = ppo.PPO(config=ppo_config, env=V2V_Env_ray)

    # run manual training loop and print results after each iteration
    for _ in range(configs['train']['stop_iters']):
        result = trainer.train()
        # print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if (
                result["timesteps_total"] >= configs['train']['stop_timesteps']
                or result["episode_reward_mean"] >= configs['train']['stop_reward']
        ):
            break

    ray.shutdown()

