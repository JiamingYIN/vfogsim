from gym.spaces import Box
import numpy as np
import random

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights

class CustomPolicy(Policy):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.config.get("ignore_action_bounds", False) and isinstance(
				self.action_space, Box
		):
			self.action_space_for_sampling = Box(
				-float("inf"),
				float("inf"),
				shape=self.action_space.shape,
				dtype=self.action_space.dtype,
			)
		else:
			self.action_space_for_sampling = self.action_space

	@override(Policy)
	def compute_actions(
			self,
			obs_batch,
			state_batches=None,
			prev_action_batch=None,
			prev_reward_batch=None,
			**kwargs
	):
		pass

	@override(Policy)
	def learn_on_batch(self, samples):
		"""No learning."""
		return {}

	@override(Policy)
	def compute_log_likelihoods(
			self,
			actions,
			obs_batch,
			state_batches=None,
			prev_action_batch=None,
			prev_reward_batch=None,
	):
		return np.array([random.random()] * len(obs_batch))

	@override(Policy)
	def get_weights(self) -> ModelWeights:
		"""No weights to save."""
		return {}

	@override(Policy)
	def set_weights(self, weights: ModelWeights) -> None:
		"""No weights to set."""
		pass

