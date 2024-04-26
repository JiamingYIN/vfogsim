from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.modelv2 import ModelV2
from gym.spaces import Dict
class TorchCustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        print(model_config)
        orig_space = getattr(obs_space, "original_space", obs_space)

        self.torch_sub_model = TorchFC(
            orig_space['env_obs'],
            action_space,
            num_outputs,
            model_config,
            name
        )

        # input_size = 25 + 25 + 5
        # input_size = 30 + (30) * 4
        input_size = 120 + 120 * 4
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 128, activation_fn=nn.Tanh),
            SlimFC(128, 1),
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"]["env_obs"] = input_dict["obs"]["env_obs"].float()
        action_mask = input_dict["obs"]["mask"]

        # Compute the unmasked logits.
        logits, _ = self.torch_sub_model({"obs": input_dict["obs"]["env_obs"]})

        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits, []


    #def central_value_function(self, obs, opponent_obs, opponent_actions):
    def central_value_function(self, obs, opponent_obs):
        print("Observation (central): ", obs.size())
        print('Central_value_function', opponent_obs.size())
        input_ = torch.cat(
            [
                obs,
                opponent_obs,
                # obs['env_obs'],
                # opponent_obs['env_obs'],
                #torch.nn.functional.one_hot(opponent_actions.long(), 5).float(),
            ],
            1,
        )
        return torch.reshape(self.central_vf(input_), [-1])

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])
