import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import ppo, PPO, PPOTorchPolicy
from ray.tune.registry import register_env
from env import V2V_Env_ray_OMNET_2hop_obs
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
NUM_OPP = 4

def load_config():
    with open(os.path.join('config', 'config_5agent.yaml'), 'r') as f:
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


def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):

    if hasattr(policy, "compute_central_vf"):
        assert other_agent_batches is not None
        print(list(other_agent_batches.values()))
        # [opponent_batch1, opponent_batch2] = list(other_agent_batches.values())
        # also record the opponent obs and actions in the trajectory
        # sample_batch[OPPONENT_OBS] = np.concatenate((opponent_batch1[1][SampleBatch.CUR_OBS],
        #                                             opponent_batch2[1][SampleBatch.CUR_OBS]), axis=1)
        # todo: test this function
        opponent_batches = []
        for ab in list(other_agent_batches.values()):
            opponent_batches.append(ab[1][SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_OBS] = np.concatenate(tuple(opponent_batches), axis=1)

        #sample_batch[OPPONENT_ACTION] = opponent_batch1[SampleBatch.ACTIONS]
        print(sample_batch[OPPONENT_OBS].shape)
        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch[SampleBatch.CUR_OBS], policy.device
                ),
                convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
                #convert_to_torch_tensor(
                #    sample_batch[OPPONENT_ACTION], policy.device
                #),
            )
            .cpu()
            .detach()
            .numpy()
        )
    else:
        # Policy hasn't been initialized yet, use zeros.
        #sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        opp_batch = []
        for _ in range(NUM_OPP):
            opp_batch.append(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_OBS] = np.zeros_like(np.concatenate(tuple(opp_batch), axis=1))
        # sample_batch[OPPONENT_OBS] = np.zeros_like(np.concatenate((sample_batch[SampleBatch.CUR_OBS],
        #                                                            sample_batch[SampleBatch.CUR_OBS]), axis=1))
        #sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
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
        #train_batch[OPPONENT_ACTION],
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

class CCPPOTorchPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        self.compute_central_vf = self.model.central_value_function

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
    env = V2V_Env_ray_OMNET_2hop_obs(configs['environment'])

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
    trainer = CentralizedCritic(config=ppo_config, env=V2V_Env_ray_OMNET_2hop_obs)

    # run manual training loop and print results after each iteration
    for i in range(configs['train']['stop_iters']):
        result = trainer.train()
        #print(pretty_print(result))

        # Save model
        if i % configs['train']['save_iters'] == 0:
            checkpoint_path = trainer.save()
            print(checkpoint_path)

        # stop training of the target train steps or reward are reached
        if (
                result["timesteps_total"] >= configs['train']['stop_timesteps']
                or result["episode_reward_mean"] >= configs['train']['stop_reward']
        ):
            break

    ray.shutdown()

