import os, shutil
import sys
import sumolib
import numpy as np
from utils.load_data import *
import json


import gym
from gym.spaces import Discrete, Box, Dict
from ray.rllib.env import MultiAgentEnv
from env import VFN


class V2V_Env_ray_OMNET_centralized(MultiAgentEnv):
    """
    A RLLIB environment for testing MARL environments with SUMO simulations.
    """
    def __init__(self, config):
        """Initialize the environment."""
        super(V2V_Env_ray_OMNET_centralized, self).__init__()
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

        # Omnet Connection
        self.omnet_time_path = config['omnet']['sumo_time_path']
        self.rllib_time_path = config['omnet']['rllib_time_path']
        self.user_road = config['omnet']['user_road']
        self.user_route = config['omnet']['user_route']
        self.vfn_road = config['omnet']['vfn_road']
        self.vfn_route = config['omnet']['vfn_route']
        self.action_path = config['omnet']['action_path']
        self.user_task_served = config['omnet']['user_task_served']
        self.user_task_unserved = config['omnet']['user_task_unserved']


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
        self._agent_id_dict_omnet = {x.id: x.veh_name for x in self.agents.values()}
        print("ID dict:", self._agent_id_dict_omnet)

        # Spaces
        '''
        self.observation_space = Dict({agent: Dict({
                                                    "env_obs": self.get_env_obs_space(agent),
                                                    "mask": self.get_mask_space(agent)})
                                      for agent in self._agent_ids})
        self.action_space = gym.spaces.Dict({agent: self.get_action_space(agent)
                                             for agent in self._agent_ids})
        '''
        self.observation_space = Dict({"env_obs": self.get_env_obs_space(len(self._agent_ids)),
                                       "mask": self.get_mask_space(len(self._agent_ids))})
        self.action_space = self.get_action_space(len(self._agent_ids))

        # Environment initialization
        self.resetted = True
        self.episodes = 0
        self.steps = 0
        self.run = 0
        self.latency = 0
        self.num_tasks = 0
        self.revenue = 0

    def _start_simulation(self):
        with open(self.rllib_time_path, 'w') as file:
            file.write('0')
        with open(self.omnet_time_path, 'w') as file:
            file.write('-1')

        # clear history
        for dir in [self.user_road, self.user_route, self.vfn_road, self.vfn_route,
                    self.user_task_unserved, self.user_task_served, self.action_path]:
            # shutil.rmtree(dir)
            # os.mkdir(dir)
            for f in os.listdir(dir):
                if os.path.isfile(os.path.join(dir, f)):
                    os.remove(os.path.join(dir, f))

        # omnet_process = subprocess.Popen(['/home/vfogsim/Documents/rllib_v2v/run_omnet.sh'])

    def close(self):
        pass

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
        """
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
        obs['env_obs'] = np.array(env_obs)
        obs['mask'] = np.array(mask)  # 0 for invalid

        # print("Agent id:", agent_id)
        # print("Current (Last) Road:", road_id)
        # print("Action mask:", mask)
        # print("Action roads:", action_roads)
        # return obs
        return obs


    def compute_observations(self):
        """For each agent in the list, return the observation."""

        users = []
        for user in os.listdir(self.user_road):
            with open(os.path.join(self.user_road, user), 'r') as file:
                line = file.readline().replace('\n', '').split(' ')
                if len(line) > 1:
                    road_id = line[1]
                    ts = float(line[2])

                    if ts > self.steps - 1:
                        if line[1][0] != ':' and road_id in self.data['road_map'].keys():
                            road_id = self.data['road_map'][road_id]

                            # Read Resource and Delay
                            if os.path.exists(os.path.join(self.user_task_unserved, user)):
                                with open(os.path.join(self.user_task_unserved, user), 'r') as file:
                                    user_task = file.readline().replace('\n', '').split(' ')
                                    # user_task: [task, resource, delay, timestamp, speed]
                                    if len(user_task) > 1 and float(user_task[3]) > self.steps - 1:
                                        users.append([line[0], road_id, float(user_task[1]), float(user_task[2]),
                                                      float(user_task[4])])

        vfns = []
        for vfn_id in self._agent_ids:
            with open(os.path.join(self.vfn_road, self.agents[vfn_id].veh_name + '.txt'), 'r') as file:
                line = file.readline().replace('\n', '').split(' ')
                road_id = line[1]
                if line[1][0] == ':':
                    with open(os.path.join(self.vfn_route, self.agents[vfn_id].veh_name + '.txt'), 'r') as file:
                        vfn_route = file.readline().replace('\n', '').split(' ')
                        road_id = vfn_route[-1]
                if road_id in self.data['road_map'].keys():
                    road_id = self.data['road_map'][road_id]
                    vfns.append([vfn_id, road_id, 1.0, 1.0])
                else:
                    print("Invalid road id for VFN (Dead-end)! ", road_id)

        # Vehicle information
        users = pd.DataFrame(users, columns=['id', 'road_id', 'Resource', 'Delay', 'Speed'])
        vfns = pd.DataFrame(vfns, columns=['id', 'road_id', 'Resource', 'Delay'])

        # [Traffic speed, Resource(Task), Delay(Task), Number(Task), Resource(VFN)]
        edge_states = {}
        for i, id in enumerate(self.data['edge_ids']):
            edge_vfn = vfns[vfns['road_id'] == id]
            edge_user = users[users['road_id'] == id]
            # TODO: 1. Neighbor, 2. Prediction
            edge_speed = edge_user['Speed'].mean() if edge_user.shape[0] > 0 else 8.33
            edge_states[id] = [edge_speed,
                               edge_user['Resource'].sum() / 100.0,
                               edge_user['Delay'].sum() / 10.0,
                               edge_user.shape[0],
                               edge_vfn['Resource'].sum() / 100.0]

        obs = {agent.id: self.get_observation(edge_states, agent.id) for agent in self.agents.values() if agent.time_to_act}

        obs = dict()
        obs['env_obs'] = np.concatenate([self.get_observation(edge_states, agent.id)['env_obs'] for agent in self.agents.values()], axis=0)
        obs['mask'] = np.concatenate([self.get_observation(edge_states, agent.id)['mask'] for agent in self.agents.values()])
        print(obs['env_obs'].shape)
        obs = dict({0: obs})

        return obs


    def update_agent_status(self):
        # Update agent status (time_to_act):
        print("Update VFN State:")
        for id in self._agent_ids:
            vfn_road_path = os.path.join(self.vfn_road, self.agents[id].veh_name + '.txt')
            vfn_route_path = os.path.join(self.vfn_route, self.agents[id].veh_name + '.txt')
            with open(vfn_road_path, 'r') as road_file:
                vfn_road = road_file.readline().replace('\n', '').strip().split(' ')
            with open(vfn_route_path, 'r') as route_file:
                vfn_route = route_file.readline().replace('\n', '').strip().split(' ')

            if len(vfn_road) < 1 or len(vfn_route) < 1:
                print("The file is written by omnet")
                continue

            print(vfn_road)
            print(vfn_route)

            road_id = vfn_road[1]
            if vfn_road[1][0] == ':':
                road_id = vfn_route[-1]
            if road_id in self.data['road_map'].keys():
                road_id = self.data['road_map'][road_id]
                self.agents[id].road_id = road_id
                current_des = vfn_route[-1]
                print(road_id, current_des)
                # self.agents[id].time_to_act = current_des == road_id or road_id == self.data['road_map'][current_des]
            # print('Agent ', id, ': ', self.agents[id].road_id, self.agents[id].time_to_act)

    ###########################################################################
    # REWARDS

    def compute_rewards(self):
        """For each agent in the list, return the rewards."""
        served_task_files = os.listdir(self.user_task_served)
        inv_id_dict = dict(zip(self._agent_id_dict_omnet.values(), self._agent_id_dict_omnet.keys()))
        for file in served_task_files:
            with open(os.path.join(self.user_task_served, file), 'r') as f:
                served_task = f.readline().replace('\n', '').strip().split(' ')
                # [task_type, resource, delay, timestamp, vfn]
                if len(served_task) == 5 and float(served_task[3]) > self.steps - 1:
                    vfn_id = inv_id_dict[served_task[4]]
                    self.agents[vfn_id].served_tasks.append([served_task[0], served_task[1], served_task[2]])
                    self.latency += float(served_task[2])
                    self.num_tasks += 1
                    # print("agent", self.agents[vfn_id].id, self.agents[vfn_id].served_tasks)

        rew = {agent.id: agent.compute_reward(self.steps) for agent in self.agents.values() if agent.time_to_act}
        total_rew = sum([agent.compute_reward(self.steps) for agent in self.agents.values() if agent.time_to_act])


        for agent in self.agents.values():
            if agent.time_to_act:
                self.revenue += rew[agent.id]

        return {0: total_rew}

    ###########################################################################
    # REST & LEARNING STEP

    def reset(self):
        """Resets the env and returns observations from ready agents."""
        print("env:reset")
        self.resetted = True
        self.episodes += 1
        self.steps = 0
        self.latency = 0
        self.num_tasks = 0
        self.revenue = 0

        # Reset the SUMO simulation
        self._start_simulation()

        # Reset the agents

        for aid in self._agent_ids:
            self.agents[aid].reset()

        self._sumo_step()

        # Observations
        initial_obs = self.compute_observations()

        return initial_obs

    def _sumo_step(self):
        #self.sumo.simulationStep()

        with open(self.rllib_time_path, 'w') as file:
            file.write(str(self.steps))

        while True:
            with open(self.omnet_time_path, 'r') as file:
                omnet_ts = file.readline()
                if omnet_ts == '' or omnet_ts.strip(' ') == '':
                    # print('omnet ts:', "'", omnet_ts, "'")
                    continue
                else:
                    try:
                        omnet_ts = float(omnet_ts)
                        if self.steps <= omnet_ts:
                            print('omnet ts:', omnet_ts)
                            break
                    except:
                        print('omnet ts:', "'", omnet_ts, "'")
                        continue
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
        print("Step:", self.steps)
        if action_dict is None or action_dict == {}:
            pass
        else:
            print("Selected actions", action_dict)

        actions = action_dict[0]
        actions = dict(zip(self._agent_ids, actions))
        # print(actions)
        # Take action
        for agent_id in self._agent_ids:
            road_id = self.agents[agent_id].road_id  # Last road id
            if self.agents[agent_id].time_to_act:
                road_action_map = self.data['action_map'].loc[road_id].values
                action_road_ind = road_action_map[actions[agent_id]]
                print("Take action (Agent ", agent_id, "):", road_id, self.data['road_id_map'][action_road_ind])
                if actions[agent_id] == 4:  # U-turn
                    next_road_ids = self.data['edges'].loc[action_road_ind]['edge_list'].split()
                    ind = min(1, len(next_road_ids) - 1)
                    self.agents[agent_id].step(next_road_ids[ind], self.action_path)
                else:
                    self.agents[agent_id].step(self.data['road_id_map'][action_road_ind], self.action_path)

        self._sumo_step()
        self.update_agent_status()
        obs = self.compute_observations()
        rewards = self.compute_rewards()
        # infos = {agent.id: {} for agent in self.agents.values() if agent.time_to_act}
        # dones = {agent.id: False for agent in self.agents.values() if agent.time_to_act}

        infos = {0: {}}
        dones = {0: False}

        # dones['__all__'] = True if self.steps >= self.episode_env_steps else False
        dones['__all__'] = False
        if self.steps >= self.episode_env_steps:
            dones['__all__'] = True
            metrics = {'latency': self.latency / max(1, self.num_tasks),
                       'num_tasks': self.num_tasks,
                       'revenue': self.revenue}
            with open('metrics.txt', 'w') as file:
                file.write(json.dumps(metrics))

        return obs, rewards, dones, infos

    ###########################################################################
    # ACTIONS & OBSERATIONS SPACE

    def get_action_space_size(self, agent):
        """Returns the size of the action space."""
        return len(self.agents[agent].actions)

    def get_action_space(self, num_agent):
        """Returns the action space."""
        #return gym.spaces.Discrete(5 * num_agent)
        return gym.spaces.MultiDiscrete([5] * num_agent)

    def get_set_of_actions(self, agent):
        """Returns the set of possible actions for an agent."""
        return set(range(self.get_action_space_size(agent)))

    def get_obs_space_size(self, num_agent):
        """Returns the size of the observation space."""
        return 25 * num_agent

    def get_env_obs_space(self, num_agent):
        """Returns the observation space."""
        # TODO: Range
        return gym.spaces.Box(0.0, 100.0, shape=(25 * num_agent,), dtype=np.float32)

    def get_obs_space(self, num_agent):
        return gym.spaces.Dict(
            {"obs": gym.spaces.Box(0.0, 100.0, shape=(25 * num_agent,), dtype=np.float32),
             "mask": gym.spaces.Box(0, 1, shape=(5 * num_agent,), dtype=np.float32)}
        )
    def get_mask_space(self, num_agent):
        return gym.spaces.Box(0, 1, shape=(5 * num_agent,), dtype=np.float32)
