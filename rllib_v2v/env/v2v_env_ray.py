import collections
import logging
import os
import sys

import pandas as pd
import traci
import sumolib
import numpy as np
from pprint import pformat
from utils.load_data import *


# sys.path.append(os.path.join('/usr/local/Cellar/sumo@1.8.0/1.8.0/tools'))
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


import gym
from gym.spaces import Discrete, Box, Dict
from ray.rllib.env import MultiAgentEnv
from env import VFN


class V2V_Env_ray(MultiAgentEnv):
    """
    A RLLIB environment for testing MARL environments with SUMO simulations.
    """
    metadata = {
        'render_modes': ['human', 'rgb_array'],
    }
    CONNECTION_LABEL = 0  # For traci multi-client support
    def __init__(self, config):
        """Initialize the environment."""
        super(V2V_Env_ray, self).__init__()
        print(config)
        self._config = config
        self._sumo_cfg = config['sumo_cfg']
        self._net = config['net_file']
        self._route = config['route_file']
        self.render_mode = config['render_mode']
        self.virtual_display = config['virtual_display']
        self.begin_time = config['begin_time']
        self.sim_max_time = config['num_seconds']
        # self.delta_time = config['delta_time']  # seconds on sumo at each step
        self.max_depart_delay = config['max_depart_delay']  # Max wait time to insert a vehicle
        self.waiting_time_memory = config['waiting_time_memory']  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = config['time_to_teleport']
        self.sumo_warnings = config['sumo_warnings']
        self.additional_sumo_cmd = config['additional_sumo_cmd']
        self.episode_env_steps = config['episode_env_steps']

        self.use_gui = config['use_gui']
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.label = str(V2V_Env_ray.CONNECTION_LABEL)
        V2V_Env_ray.CONNECTION_LABEL += 1
        self.sumo = None
        # SUMO Connector

        if LIBSUMO:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net], label='init_connection'+self.label)
            conn = traci.getConnection('init_connection'+self.label)

        # Load Data
        self.data = load_data(config['path'])
        self.E = len(self.data['edge_ids'])

        # Agent initialization
        print("Agent initialization")
        self.agents_init_list = dict()
        self._agent_ids = []
        self.agents = dict()
        for agent, agent_config in self._config["agent_init"].items():
            self.agents[agent_config['vfn_id']] = VFN(agent_config)
            self._agent_ids.append(agent_config['vfn_id'])
        self.MAX_NEW_ID = 0
        self._agent_id_dict = dict(zip(self._agent_ids, self._agent_ids))  # agent_id: vehicle_id
        conn.close()

        # Spaces
        # self.observation_space = gym.spaces.Dict({agent: self.get_obs_space(agent)
        #                                           for agent in self._agent_ids})
        self.observation_space = Dict({agent: Dict({
                                                    "env_obs": self.get_env_obs_space(agent),
                                                    "mask": self.get_mask_space(agent)})
                                      for agent in self._agent_ids})
        self.action_space = gym.spaces.Dict({agent: self.get_action_space(agent)
                                             for agent in self._agent_ids})

        # Environment initialization
        self.resetted = True
        self.episodes = 0
        self.steps = 0
        self.run = 0

    def _start_simulation(self):
        sumo_cmd = [self._sumo_binary,
                    '-c', self._sumo_cfg]
        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        # if self.sumo_seed == 'random':
        #     sumo_cmd.append('--random')
        # else:
        #     sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append('--no-warnings')
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
            if self.render_mode is not None and self.render_mode == 'rgb_array':
                sumo_cmd.extend(['--window-size', f'{self.virtual_display[0]},{self.virtual_display[1]}'])
                from pyvirtualdisplay.smartdisplay import SmartDisplay
                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        print(sumo_cmd)

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        # if self.disp is not None:
        #     self.disp.stop()
        #     self.disp = None
        self.sumo = None

    def __del__(self):
        self.close()

    def get_agents(self):
        """Returns a list of the agents."""
        return self.agents.keys()


    ###########################################################################
    # OBSERVATIONS

    def get_observation(self, edge_states, agent_id):
        """
        Returns the observation of a given agent.
        See http://sumo.sourceforge.net/pydoc/traci._simulation.html
        """
        # road_id = self.get_edge_id(agent_id)
        road_id = self.agents[agent_id].road_id
        print("Get observation (agent road id):", agent_id, ':', road_id)

        action_roads = self.data['action_map'].loc[road_id].values
        env_obs = []
        mask = []
        for road in action_roads:
            if road < 0:
                env_obs.extend([0.0] * 5)
                mask.append(0.0)
            else:
                act_road = self.data['road_id_map'][road]
                env_obs.extend(edge_states[act_road])
                mask.append(1.0)
        mask[3] = 0.0  # Mask "Stay"
        obs = dict()
        obs['env_obs'] = env_obs
        obs['mask'] = mask  # 0 for invalid

        # print("Agent id:", agent_id)
        # print("Current (Last) Road:", road_id)
        # print("Action mask:", mask)
        # print("Action roads:", action_roads)
        # return obs
        return obs

    def get_edge_id(self, id):
        # Get edge id in topology graph (road id -> edge id)
        road_id = self.sumo.vehicle.getRoadID(id)

        if road_id == None:  # Vehicles haven't been added
            road_id = self.sumo.vehicle.getRoute(id)[0]

        # Check if the car is at a node
        if road_id[0] == ':':
            if id in self._agent_id_dict.values():  # VFN
                road_id = self.sumo.vehicle.getRoute(id)[-1]
            else:  # Other vehicles (ignore)
                return None

        # Check if the road is valid (not dead-end)
        if road_id not in self.data['road_map'].keys():
            if id in self._agent_id_dict.values():
                print('Invalid Road:', id, road_id)
            return None

        road_id = self.data['road_map'][road_id]
        return road_id

    def compute_observations(self):
        """For each agent in the list, return the observation."""
        # Road information
        road_speeds = []
        for d in self.data['detectors']:
            speed = self.sumo.inductionloop.getLastStepMeanSpeed('e1det_' + d + '_0')
            if speed < 0:
                speed = 8.33
            road_speeds.append(speed)

        # Vehicle information
        ids = self.sumo.vehicle.getIDList()
        users = []
        vfns = []

        for id in ids:
            road_id = self.get_edge_id(id)
            if road_id is None:
                continue

            if id in self._agent_id_dict.values():
                # TODO: Remaining CPU & GPU
                vfns.append([id, road_id, 1.0, 1.0])
            elif id in self.data['user_ids']:
                # todo: Computation of CPU & GPU
                users.append([id, road_id, 1.0, 1.0])
        users = pd.DataFrame(users, columns=['id', 'road_id', 'CPU', 'GPU'])
        vfns = pd.DataFrame(vfns, columns=['id', 'road_id', 'CPU', 'GPU'])

        # [Traffic speed, CPU(Task), GPU(Task), CPU(VFN), GPU(VFN)]
        edge_states = {}
        for i, id in enumerate(self.data['edge_ids']):
            edge_vfn = vfns[vfns['road_id'] == id]
            edge_user = users[users['road_id'] == id]
            # TODO: 1. Neighbor, 2. Prediction
            edge_states[id] = [road_speeds[i],
                               edge_user['CPU'].sum(),
                               edge_user['GPU'].sum(),
                               edge_vfn['CPU'].sum(),
                               edge_vfn['GPU'].sum()]

        obs = {agent.id: self.get_observation(edge_states, agent.id) for agent in self.agents.values() if agent.time_to_act}
        return obs

    def add_new_vehicle(self, aid):
        new_id = 'new' + str(self.MAX_NEW_ID)
        route_id = 'route_' + self.agents[aid].road_id
        self.sumo.vehicle.add(vehID=new_id,
                              typeID='vfn',
                              routeID=route_id,
                              )
        self.MAX_NEW_ID += 1
        self._agent_id_dict[aid] = new_id
        self.agents[aid].veh_id = new_id

        print('=' * 89)
        print("New vehicle added!")
        print("Agent:", aid, "New vehicle id:", new_id, route_id)
        road_id = self.agents[aid].road_id

        return road_id

    def update_agent_status(self):
        # Update agent status (time_to_act):
        ids = self.sumo.vehicle.getIDList()
        for id in self._agent_ids:
            if self._agent_id_dict[id] not in ids:
                road_id = self.add_new_vehicle(id)
            else:
                road_id = self.get_edge_id(self._agent_id_dict[id])
            self.agents[id].road_id = road_id
            current_des = self.sumo.vehicle.getRoute(self._agent_id_dict[id])[-1]
            self.agents[id].time_to_act = current_des == road_id or road_id == self.data['road_map'][current_des]
    ###########################################################################
    # REWARDS

    def compute_rewards(self):
        """For each agent in the list, return the rewards."""
        # print("compute_rewards")

        ids = self.sumo.vehicle.getIDList()
        users = []
        vfns = []

        inv_id_dict = dict(zip(self._agent_id_dict.values(), self._agent_id_dict.keys()))

        for id in ids:
            x, y = self.sumo.vehicle.getPosition(id)

            if id in self._agent_id_dict.values():
                vfns.append([inv_id_dict[id], x, y])
            # elif id in self.data['user_ids']:
            else:
                users.append([id, x, y])

        users = pd.DataFrame(users, columns=['id', 'x', 'y'])
        vfns = pd.DataFrame(vfns, columns=['id', 'x', 'y'])

        for i, row in vfns.iterrows():
            users['dist'] = (users['x'] - row['x']) ** 2 + (users['y'] - row['y']) ** 2
            candidates = users[users['dist'] < 300 ** 2]
            num_users = min(self.agents[row['id']].max_usr_num, candidates.shape[0])
            self.agents[row['id']].cumulated_reward += num_users

        rew = {agent.id: agent.cumulated_reward / max(1, self.steps - agent.last_act_step) \
               for agent in self.agents.values() if agent.time_to_act}

        for id in self._agent_ids:
            if self.agents[id].time_to_act:
                self.agents[id].last_act_step = self.steps

        return rew

    ###########################################################################
    # REST & LEARNING STEP

    def reset(self):
        """Resets the env and returns observations from ready agents."""
        print("env:reset")
        self.resetted = True
        self.episodes += 1
        self.steps = 0

        # Reset the SUMO simulation
        if self.run != 0:
            self.close()
        self.run += 1
        self.MAX_NEW_ID = 0
        self._agent_id_dict = dict(zip(self._agent_ids, self._agent_ids))
        self._start_simulation()

        # Reset the agents
        for aid in self._agent_ids:
            self.agents[aid].reset()
            self.agents[aid].road_id = np.random.choice(self.data['edge_ids'], 1)[0]
            self.add_new_vehicle(aid)

        self._sumo_step()

        ids = self.sumo.vehicle.getIDList()
        print(ids)

        self.update_agent_status()

        # Observations
        initial_obs = self.compute_observations()

        return initial_obs

    def _sumo_step(self):
        self.sumo.simulationStep()

    def step(self, action_dict):
        """
        Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs: New observations for each ready agent.
            rewards: Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones: Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos: Optional info values for each agent id.
        """
        self.resetted = False
        self.steps += 1

        if action_dict is None or action_dict == {}:
            pass
        else:
            print("Step:", self.steps)
            print("Selected actions", action_dict)
        shuffled_agents = sorted(
            action_dict.keys()
        )

        # Take action
        for agent_id in shuffled_agents:
            road_id = self.agents[agent_id].road_id  # Last road id
            if self.agents[agent_id].time_to_act:
                road_action_map = self.data['action_map'].loc[road_id].values
                action_road_ind = road_action_map[action_dict[agent_id]]
                print("Take action (Agent ", agent_id, "):", road_id, self.data['road_id_map'][action_road_ind])
                if action_dict[agent_id] == 4:  # U-turn
                    next_road_ids = self.data['edges'].loc[action_road_ind]['edge_list'].split()
                    ind = min(1, len(next_road_ids) - 1)
                    self.agents[agent_id].step(next_road_ids[ind], self.sumo)
                else:
                    self.agents[agent_id].step(self.data['road_id_map'][action_road_ind], self.sumo)

        self._sumo_step()
        self.update_agent_status()
        obs = self.compute_observations()
        rewards = self.compute_rewards()
        infos = {agent.id: {} for agent in self.agents.values() if agent.time_to_act}
        dones = {agent.id: False for agent in self.agents.values() if agent.time_to_act}
        dones['__all__'] = True if self.steps > self.episode_env_steps else False

        # print(obs, rewards, dones, infos)
        return obs, rewards, dones, infos

    ###########################################################################
    # ACTIONS & OBSERATIONS SPACE

    def get_action_space_size(self, agent):
        """Returns the size of the action space."""
        return len(self.agents[agent].actions)

    def get_action_space(self, agent):
        """Returns the action space."""
        return gym.spaces.Discrete(self.get_action_space_size(agent))

    def get_set_of_actions(self, agent):
        """Returns the set of possible actions for an agent."""
        return set(range(self.get_action_space_size(agent)))

    def get_obs_space_size(self, agent):
        """Returns the size of the observation space."""
        return 25

    def get_env_obs_space(self, agent):
        """Returns the observation space."""
        # TODO: Range
        return gym.spaces.Box(0.0, 20.0, shape=(25,), dtype=np.float32)

    def get_obs_space(self, agent):
        return gym.spaces.Dict(
            {"obs": gym.spaces.Box(0.0, 20.0, shape=(25,), dtype=np.float32),
             "mask": gym.spaces.Box(0, 1, shape=(5,), dtype=np.float32)}
        )
    def get_mask_space(self, agent):
        return gym.spaces.Box(0, 1, shape=(5,), dtype=np.float32)