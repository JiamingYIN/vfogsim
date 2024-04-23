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
import numpy as np
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

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

class CentralizedValueMixin:
    def __init(self):
        self.compute_central_vf = self.model.central_value_function


def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):
    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")) or (
        not pytorch and policy.loss_initialized()
    ):
        assert other_agent_batches is not None
        [(_, opponent_batch)] = list(other_agent_batches.values())
        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch[SampleBatch.CUR_OBS], policy.device
                ),
                convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
                convert_to_torch_tensor(
                    sample_batch[OPPONENT_ACTION], policy.device
                ),
            )
            .cpu()
            .detach()
            .numpy()
        )
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch

# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss

def central_vf_stats(policy, train_batch):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out
        )
    }

class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return centralized_critic_postprocessing(
            self, sample_batch, other_agent_batches, episode
        )


class CentralizedCritic(PPO):
    @override(PPO)
    def get_default_policy_class(self, config):
        print('CentralizedCritic!')
        return CCPPOTorchPolicy


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
            None,
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
    trainer = CentralizedCritic(config=ppo_config, env=V2V_Env_ray)

    # run manual training loop and print results after each iteration
    for _ in range(configs['train']['stop_iters']):
        result = trainer.train()
        print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if (
                result["timesteps_total"] >= configs['train']['stop_timesteps']
                or result["episode_reward_mean"] >= configs['train']['stop_reward']
        ):
            break

    ray.shutdown()

