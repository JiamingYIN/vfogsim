import os
import sys
import gym
from typing import Callable, List, Union

sys.path.append(os.path.join('/usr/local/Cellar/sumo@1.8.0/1.8.0/tools'))
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
from gym import spaces

class VFN(object):
    def __init__(self, config):
        self.id = config['vfn_id']
        self.veh_id = config['vfn_id']  # SUMO vehicle ID
        self.CPU_capacity = config['CPU_capacity']
        self.GPU_capacity = config['GPU_capacity']
        self.action_space = spaces.Discrete(5)
        self.config = config
        self.actions = ["straight", "left", "right", "stay", "uturn"]
        self.time_to_act = True
        self.max_usr_num = config['max_usr_num']
        self.cumulated_reward = 0.0
        self.last_act_step = 0

    def step(self, action_road, simulation):
        if isinstance(action_road, str):
            simulation.vehicle.changeTarget(self.veh_id, action_road)
        else:
            simulation.vehicle.setRoute(self.veh_id, action_road)

    def reset(self):
        self.veh_id = self.config['vfn_id']
        self.CPU_capacity = self.config['CPU_capacity']
        self.GPU_capacity = self.config['GPU_capacity']
        self.road_id = None
        self.cumulated_reward = 0
        self.last_act_step = 0
        self.time_to_act = True


    def update_road(self, road_id):
        self.road_id = road_id


    def update(self, tasks):
        # TODO
        pass

    def compute_observation(self):
        # TODO: return CPU, GPU, road segment
        # return [self.road_id, self.CPU_capacity, self.GPU_capacity]
        return [0.0, 0.0]

    def compute_reward(self):
        # TODO: computed according to finished tasks, travel_cost, penalty for latency
        return 0.

