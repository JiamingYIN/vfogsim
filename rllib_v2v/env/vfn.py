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
        self.veh_name = config['vfn_name']  # OMNET vehicle name
        self.CPU_capacity = config['CPU_capacity']
        self.GPU_capacity = config['GPU_capacity']
        self.action_space = spaces.Discrete(5)
        self.config = config
        self.actions = ["straight", "left", "right", "stay", "uturn"]
        self.time_to_act = True
        self.max_usr_num = config['max_usr_num']
        self.cumulated_reward = 0.0
        self.served_tasks = []
        self.last_act_step = 0

    def step(self, action_road, action_path):
        '''
        if isinstance(action_road, str):
            simulation.vehicle.changeTarget(self.veh_id, action_road)
        else:
            simulation.vehicle.setRoute(self.veh_id, action_road)
        '''
        with open(os.path.join(action_path, self.veh_name + '.txt'), 'w') as file:
            file.write(action_road)


    def reset(self):
        self.veh_id = self.config['vfn_id']
        self.veh_name = self.config['vfn_name']
        self.CPU_capacity = self.config['CPU_capacity']
        self.GPU_capacity = self.config['GPU_capacity']
        self.road_id = self.config['road_id']
        self.cumulated_reward = 0
        self.last_act_step = 0
        self.time_to_act = True
        self.served_tasks = []


    def update_road(self, road_id):
        self.road_id = road_id

    def update(self, tasks):
        # TODO
        pass

    def compute_observation(self):
        # TODO: return CPU, GPU, road segment
        # return [self.road_id, self.CPU_capacity, self.GPU_capacity]
        return [0.0, 0.0]

    def compute_reward(self, current_step):
        # TODO: computed according to finished tasks, travel_cost, penalty for latency

        revenue_reward = 0.0
        price_table = {'1': {'price': 2, 'delay1': 5, 'delay2': 10},
                       '2': {'price': 5, 'delay1': 5, 'delay2': 10},
                       '3': {'price': 10, 'delay1': 5, 'delay2': 10}}
        print("Compute reward:", self.served_tasks)
        for task in self.served_tasks:
            p = price_table[task[0]]['price']
            d1 = price_table[task[0]]['delay1']
            d2 = price_table[task[0]]['delay2']
            delay = float(task[2])
            if delay < d1:
                revenue_reward += p
            elif delay < d2:
                revenue_reward += p * (d2 - delay) / (d2 - d1)

        self.served_tasks = []
        self.last_act_step = current_step
        return revenue_reward

