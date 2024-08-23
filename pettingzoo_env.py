#______unfinished_________




import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from tensordict import TensorDict
import torch
import gym
from gym import spaces
import pygame
import numpy as np
import random
from env.simulator import Simulator
import env.constants as constants
from torchrl.envs.utils import check_env_specs
from gym.utils.env_checker import check_env
from gym.envs.registration import register
from tensordict import TensorDict
import torch

from pettingzoo import ParallelEnv




class LISPredatorPreyEnvZoo(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self):
        
        # 初始化模拟器
        self.simulator = Simulator(screen_width=3840, screen_height=2160)
        self.group_map = {}
        self.current_step = 0
        self.max_steps = 10_000


        
        # 初始化观察和动作空间
        self.max_range = max(600, 1000)  
        self.zero_list = [[0 for _ in range(3)] for _ in range(25)]
        self.num_entities = constants.NUM_PREDATORS + constants.NUM_PREY
        self.new_shape = (self.num_entities, 25, 3)
        self.obs_low = np.full(self.new_shape, -self.max_range, dtype=np.float32) #inf maybe not the best choice , let us decide later
        self.obs_high = np.full(self.new_shape, self.max_range, dtype=np.float32)
        self.action_shape = (self.num_entities,2)
        self.action_speed_range = max(constants.PREY_MAX_SPEED,constants.PREY_MAX_SPEED)
        self.action_low = np.full(self.action_shape, -self.action_speed_range, dtype=np.float32)
        self.action_high = np.full(self.action_shape, self.action_speed_range, dtype=np.float32)
        # obs_low = np.array([0, 0, 0] * 25*(constants.NUM_PREDATORS+constants.NUM_PREY))
        # obs_high = np.array([max_range, max_range, max_range] * 25*(constants.NUM_PREDATORS+constants.NUM_PREY))
        # self.observation_space_shape = (constants.NUM_PREDATORS+constants.NUM_PREY) * 3 * 25
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high,dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.interation = 0
        # 调用 reset 方法初始化环境
        self.reset() 
        self.agent_keys = []

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - prisoner x and y coordinates
        - guard x and y coordinates
        - escape x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.group_map.clear()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        allalgorithms= self.reset_algorithm()
        all_pred_algorithms, all_prey_algorithms = allalgorithms[:constants.NUM_PREDATORS],allalgorithms[constants.NUM_PREDATORS:]
        self.simulator.initialize(all_pred_algorithms, all_prey_algorithms)
        self.agent_keys = list(self.simulator.agent_status.keys())
        self.map_agents_to_groups(self.simulator.predators, self.simulator.preys)
        # 初始化环境信息（捕食者、猎物、食物和障碍物）
        for predator in self.simulator.predators:
            self._set_agent_env(predator)
        for prey in self.simulator.preys:
            self._set_agent_env(prey)
        # 获取所有智能体的初始观测数据
        all_observations = []
        for group_name in self.group_map.keys():
            for agent in getattr(self.simulator, group_name):
                all_observations.append(agent.get_observe_info())
        obs_tensors = {
            agent_key: torch.tensor(all_observation, dtype=torch.float32)
            for agent_key, all_observation in zip(self.agent_keys, all_observations)
        }
        sample_key = next(iter(obs_tensors))
        batch_size = obs_tensors[sample_key].shape[0]
        obs_tensordict = TensorDict(obs_tensors, batch_size=[batch_size])
        obs = np.array(all_observations, dtype=np.float32) #np.shape(all_observations) = (2250,3)
        # print(np.shape(all_observations))
        # obs = TensorDict({
        #     'observation': torch.tensor(obs)
        # }, batch_size=[])
        info = {}
        # print("Initial Observation:", obs)
        # assert self.observation_space.contains(obs), "Initial observation is out of bounds!"

        # 返回一个数组，符合 observation_space 的定义
        return obs_tensordict, info
    def map_agents_to_groups(self,simPredators,simPreys):
        self.group_map['predators'] = [predator.name for predator in simPredators]
        self.group_map['preys'] = [prey.name for prey in simPreys]

    def reset_algorithm(self):
        prey_algorithms = ["PPO","PPO","PPO","DDPG","DDPG","DDPG"]
        pred_algorithms = ["PPO","PPO","PPO","DDPG","DDPG","DDPG"]
        all_pred_algorithms = self.assign_algorithms_to_agents(constants.NUM_PREDATORS,pred_algorithms)
        all_prey_algorithms = self.assign_algorithms_to_agents(constants.NUM_PREY,prey_algorithms)
        return all_pred_algorithms+all_prey_algorithms
        # assigned_Predalgorithms = assign_algorithms_to_agents(self.simulator.predators, all_pred_algorithms)
        # assigned_Preyalgorithms = assign_algorithms_to_agents(self.simulator.preys, all_prey_algorithms)
    def assign_algorithms_to_agents(self,len_agents, algorithm_names):
        """
        分配算法给每个智能体。

        参数:
        - agents: 智能体列表。
        - algorithm_names: 预定义的算法名称列表。

        返回:
        - 包含算法名称的列表，长度与agents列表相同。如果算法名称不足，则用'random'补充。
        """
        assigned_algorithms = []
        for i in range(len_agents):
            if i < len(algorithm_names):
                assigned_algorithms.append(algorithm_names[i])
            else:
                assigned_algorithms.append('random')
        return assigned_algorithms
    def _set_agent_env(self, agent):
        agent.env_predators = self.simulator.predators
        agent.env_prey = self.simulator.preys
        agent.env_food = self.simulator.foods
        agent.env_obstacles = self.simulator.obstacles



    def step(self, actions):
        new_states, rewards, dones, infos = [], [], [], []
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        # 获取所有键并转换为列表
        # keys = list(self.simulator.agent_status.keys())

        # 获取从 constants.NUM_PREDATORS 开始往后 constants.NUM_PREYS 个键
        selected_keys = self.agent_keys[constants.NUM_PREDATORS:constants.NUM_PREDATORS + constants.NUM_PREY]

        # 构建新的字典

        # all_actions =[]
        # all_pred_actions, all_prey_actions = actions[:constants.NUM_PREDATORS],actions[constants.NUM_PREDATORS:]
        # 独立处理每个组的动作
        for group_name in self.group_map.keys():
            if group_name == "predators":
                group_actions = actions[:constants.NUM_PREDATORS]
                agent_status = {k: self.simulator.agent_status[k] for k in list(self.simulator.agent_status.keys())[:constants.NUM_PREDATORS]}
            if group_name == "preys":
                group_actions = actions[constants.NUM_PREDATORS:]
                agent_status = {k: self.simulator.agent_status[k] for k in selected_keys}
            
            # 获取每个组的数据，并将其展开添加到主列表中
            group_states, group_rewards, group_dones, group_infos = self._step_group(group_name, group_actions,agent_status)
            new_states.extend(group_states)
            rewards.extend(group_rewards)
            dones.extend(group_dones)
            infos.extend(group_infos)


        new_state_tensors = {
            agent_key: torch.tensor(new_state, dtype=torch.float32)
            for agent_key, new_state in zip(self.agent_keys, new_states)
        }
        sample_key = next(iter(new_state_tensors))
        batch_size = new_state_tensors[sample_key].shape[0]
        new_state_tensordict = TensorDict(new_state_tensors, batch_size=[batch_size])


        reward_tensors = {
            agent_key: torch.tensor(reward, dtype=torch.float32)
            for agent_key, reward in zip(self.agent_keys, rewards)
        }
        sample_key = next(iter(reward_tensors))
        # batch_size = reward_tensors[sample_key].shape[0]
        reward_tensordict = TensorDict(reward_tensors, batch_size=[])


        done_tensors = {
            agent_key: torch.tensor(done, dtype=torch.float32)
            for agent_key, done in zip(self.agent_keys, dones)
        }
        sample_key = next(iter(done_tensors))
        # batch_size = done_tensors[sample_key].shape[0]
        done_tensordict = TensorDict(done_tensors, batch_size=[])


        # info_tensors = {
        #     agent_key: torch.tensor(info, dtype=torch.float32)
        #     for agent_key, info in zip(self.agent_keys, infos)
        # }
        # sample_key = next(iter(info_tensors))
        # # batch_size = info_tensors[sample_key].shape[0]
        # info_tensordict = TensorDict(info_tensors, batch_size=[])

        # infos = {
        #     f"info_{i}": info_item for i, info_item in enumerate(infos)
        # }
        # terminated = all(dones)   
        # 将 observations 列表转换为 numpy 数组
        # new_observations = np.array(new_state, dtype=np.float32)
        # 调用模拟器的其他方法
        self.simulator.add_food()  # 传递时间间隔
        self.simulator.prey_hunt()
        self.simulator.check_collisions()
        self.simulator.decrease_health()  # 更新健康值
        self.simulator.remove_dead()  # 清理死亡个体

        return new_state_tensordict, reward_tensordict, done_tensordict, truncated, infos


    # def _step_group(self, group_name, group_actions):
    #     # 执行每个组的动作，并获取新的状态、奖励、是否完成和信息
    #     new_observations = []
    #     rewards = []
    #     dones = []
    #     infos = []
    #     # print(np.shape(group_actions))
    #     group = getattr(self.simulator, group_name)
    #     for agent, action in zip(group, group_actions):
    #         agent.move_strategy(action)
    #         agent.move(constants.CONTROL_PANEL_WIDTH, self.simulator.screen_width, self.simulator.screen_height, self.simulator.obstacles)
            
    #         new_observations.append(agent.get_observe_info())
    #         rewards.append(self._compute_reward(agent, group_name))
    #         dones.append(not agent.is_alive)  # 这里假设死亡标志环境结束
    #         infos.append({})  # 可以添加更多的调试信息

    #     return new_observations, rewards, dones, infos
    # def _step_group(self, group_name, group_actions):
    #     # 执行每个组的动作，并获取新的状态、奖励、是否完成和信息
    #     new_observations = []
    #     rewards = []
    #     dones = []
    #     infos = []
        
    #     group = getattr(self.simulator, group_name)
    #     for agent, action in zip(group, group_actions):
    #         agent.move_strategy(action)
    #         agent.move(constants.CONTROL_PANEL_WIDTH, self.simulator.screen_width, self.simulator.screen_height, self.simulator.obstacles)
            
    #         new_observations.append(agent.get_observe_info())
    #         rewards.append(self._compute_reward(agent, group_name))
    #         dones.append(not agent.is_alive)  # 这里假设死亡标志环境结束
    #         infos.append({})  # 可以添加更多的调试信息

    #     return new_observations, rewards, dones, infos
    def _step_group(self, group_name, group_actions, agent_status):
        # 执行每个组的动作，并获取新的状态、奖励、是否完成和信息
        temp_observations = {}
        temp_rewards = {}
        temp_dones = {}
        temp_infos = {}

        group = getattr(self.simulator, group_name)
        
        for agent, action in zip(group, group_actions):
            agent.move_strategy(action)
            agent.move(constants.CONTROL_PANEL_WIDTH, self.simulator.screen_width, self.simulator.screen_height, self.simulator.obstacles)
            
            temp_observations[agent.name] = agent.get_observe_info()
            temp_rewards[agent.name] = self._compute_reward(agent, group_name)
            temp_dones[agent.name] = not agent.is_alive  # 这里假设死亡标志环境结束
            temp_infos[agent.name] = {}  # 可以添加更多的调试信息

        # 根据 agent_status 比对结果，生成最终的列表
        new_observations = []
        rewards = []
        dones = []
        infos = []
        
        for agent_name in agent_status.keys():
            if agent_name in temp_observations:
                new_observations.append(temp_observations[agent_name])
                rewards.append(temp_rewards[agent_name])
                dones.append(temp_dones[agent_name])
                infos.append(temp_infos[agent_name])
            else:
                new_observations.append(np.array(self.zero_list))  # 如果 agent 不存在，则返回0
                rewards.append(0)  # 如果 agent 不存在，则返回0
                dones.append(True)  # 如果 agent 不存在，则认为它完成了
                infos.append({})  # 如果 agent 不存在，则返回空信息字典

        return new_observations, rewards, dones, infos



    def _compute_reward(self, agent, group_name):
        # 根据组别计算奖励
        if group_name == 'predators':
            # 捕食者奖励
            return agent.health if agent.health > 0 else -1.0
        elif group_name == 'preys':
            # 猎物奖励
            return agent.health if agent.health > 0 else -1.0
        return 0

    def render(self):
        """Renders the environment."""
        grid = np.full((7, 7), " ")
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "E"
        print(f"{grid} \n")


def run_random_simulation(env):
    
    # env = PredatorPreyEnv()
    observations,infos = env.reset()
    obs = observations
    # print("Returned Observation:", obs)
    # print("Observation Space Low:", env.observation_space.low)
    # print("Observation Space High:", env.observation_space.high)
    # print("Observation dtype:", obs.dtype)
    # print("Expected dtype:", env.observation_space.dtype)
    # assert env.observation_space.contains(obs), "Observation is out of bounds!"

    # print(np.shape(observations),end="---")
    # print(np.shape(env.observation_space))
    # check_env_specs(env)

    # check_env(env)
    # rollout = env.rollout(10)
    # print(f"rollout of {10} steps:", rollout)
    # print("Shape of the rollout TensorDict:", rollout.batch_size)
    

    # observations = env.reset()
    #print(np.shape(observation))
    done = False
    iteration = 0
    while not done:
        # actions = {
        #     'predators': generate_random_actions(len(env.simulator.predators), env.action_space),
        #     'preys': generate_random_actions(len(env.simulator.preys), env.action_space),
        # }
        actions = env.action_space.sample()  # 从动作空间中采样一个随机动作 # you can change this with your algorithm
        new_state, rewards, done, truncated,infos = env.step(actions)

        # 判断是否所有智能体都已经完成
        # done = all(all(done_group) for done_group in dones.values())
        iteration +=1

        # 渲染环境（可选）
        env.render()
        if iteration % 100 == 1:   
            pass
            # print(f"iteration: {iteration}, num_predators: {len(env.simulator.predators)}, num_preys: {len(env.simulator.preys)}")

            # print(iteration,end="\t")
            # print(len(env.simulator.predators),end="\t")
            # print(len(env.simulator.preys))
        # 打印当前状态、奖励、是否结束
            # print(f"New State: {new_state}")
            # print(f"Rewards: {new_state}")
            # print(len(new_state))
            # print(f"Dones: {np.shape(done)}")
            # print(f"Dones length:{len(done)}")
            # print(f"Infos: {infos}")

from pettingzoo.test import parallel_api_test
if __name__ == "__main__":

    # register(
    #     id='LISPredatorPreyZooEnv-v0',
    #     entry_point='pettingzoo_env:LISPredatorPreyZooEnv',
    # )

    # env = gym.make('LISPredatorPreyZooEnv-v0')
    env = LISPredatorPreyEnvZoo()
    parallel_api_test(env, num_cycles=1_000_000)
    obs,infos = env.reset()  # 重置环境并获取初始观测
    # print("Initial observation:", obs)
    # print(env.simulator.agent_status)

    # for _ in range(10):  # 运行10个时间步
    #     action = env.action_space.sample()  # 随机采样一个动作
    #     ew_observations, rewards, terminated, truncated, infos = env.step(action)  # 采取一步行动
    #     # print(f"Observation: {obs}, Reward: {rewards}, Done: {terminated}, Info: {infos}")
    #     print(f"Observation: {type(obs)}, Reward: {np.shape(rewards)}, Done: {np.shape(terminated)}, Info: {np.shape(infos)}")

    #     # print(np.shape(obs))
    #     if terminated:
    #         obs = env.reset()  # 如果环境结束了，则重置环境
    
    # check_env(env.unwrapped)
    # check_env(env)
    # run_random_simulation(env)