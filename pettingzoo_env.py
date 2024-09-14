import gymnasium as gym
import gymnasium 
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from pettingzoo.test import parallel_api_test
# from pettingzoo.test import api_test
from pettingzoo_api_test import api_test
import pygame
from typing import Any, Collection, Dict, List, Optional, Sequence

import numpy as np
import random
from env.simulator import Simulator
import env.constants as constants
# from stable_baselines3.common.env_checker import check_env
from gymnasium.utils.env_checker import check_env
import torch
# import matplotlib.pyplot as plt
from pettingzoo.utils import aec_to_parallel, parallel_to_aec
# from parallel_test import parallel_api_test

class LISPredatorPreyEnv(ParallelEnv):
    metadata = {"name": "LISPredatorPreyEnv"}
    def __init__(self, prey_algorithms=[], pred_algorithms=[], predator_algorithms_predict={}, prey_algorithms_predict={}):
        super(LISPredatorPreyEnv, self).__init__()

        self.simulator = Simulator(screen_width=3840, screen_height=2160)
        self.group_map = {}
        self.current_step = 0
        self.max_steps = 10_000
        self.traing_algorithm = ''


        # 初始化算法
        self.prey_algorithms = prey_algorithms
        self.pred_algorithms = pred_algorithms
        self._initialize_algorithm_encoding()

        # self.simulator.predator_algorithms_predict = predator_algorithms_predict
        # self.simulator.prey_algorithms_predict = prey_algorithms_predict
        self.simulator.prey_algorithm_encoding = self.prey_algorithm_encoding
        self.simulator.pred_algorithm_encoding = self.pred_algorithm_encoding
        self.simulator.predator_algorithms_predict = self.convert_dicts_to_numeric_keys(self.pred_algorithm_encoding,predator_algorithms_predict)
        self.simulator.prey_algorithms_predict = self.convert_dicts_to_numeric_keys(self.prey_algorithm_encoding,prey_algorithms_predict)
        self.agents= []
        self.possible_agents =[]
        self._initialize_agent_environment()
        
        # 初始化观察和动作空间
        self._initialize_spaces()
    def _initialize_agent_environment(self):
        """将reset中不需要每次重置的部分移到这里初始化。"""
        all_pred_algorithms, all_prey_algorithms = self.reset_algorithm()
        self.simulator.initialize(all_pred_algorithms, all_prey_algorithms)
        self.agents = [agent.name for agent in self.simulator.preys + self.simulator.predators]
        self.possible_agents = [agent.name for agent in self.simulator.preys + self.simulator.predators]
        self.map_agents_to_groups(self.simulator.predators, self.simulator.preys)

        for predator in self.simulator.predators:
            self._set_agent_env(predator)
        for prey in self.simulator.preys:
            self._set_agent_env(prey)

    def _initialize_spaces(self):
        """Initialize observation and action spaces."""
        self.max_range = max(constants.PREY_HEARING_RANGE, constants.PREDATOR_HEARING_RANGE)
        self.num_entities = constants.NUM_PREDATORS + constants.NUM_PREY
        self.observation_shape = (self.num_entities, 25, 4)
        self.obs_low = np.full(self.observation_shape, -self.max_range, dtype=np.float32)
        self.obs_high = np.full(self.observation_shape, self.max_range, dtype=np.float32)
        # self.action_shape = (self.num_entities, 3)
        # self.action_speed_range = max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
        # self.action_low = np.full(self.action_shape, -self.action_speed_range, dtype=np.float32)
        # self.action_high = np.full(self.action_shape, self.action_speed_range, dtype=np.float32)
        # self.action_low[:, 0] = 0.0
        # self.action_high[:, 0] = 1.0
        MAX_CONSTANT = 10
        # self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        ob_env1_space = gymnasium.spaces.Tuple((
            spaces.Discrete(4),                    # 第一列，离散值 1 到 4
            spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),  # 第二、三列，位置坐标，连续值
            spaces.Discrete(MAX_CONSTANT)          # 第四列，离散值 0 到 MAX_CONSTANT
        ))
        ob_env1_space = gymnasium.spaces.Tuple([ob_env1_space] * 20)
        ob_env2_space = spaces.Box(
            low=np.array([[0, -np.pi]] * 5),    # 形状 (5, 2)
            high=np.array([[1, np.pi]] * 5),
            dtype=np.float32
        )
        ob_env3_space = spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)
        # self.observation_space = gymnasium.spaces.Dict({
        #     "sight": ob_env1_space,   # 5 行 4 列的 ob_env1 (通过 Tuple 实现离散和连续的组合)
        #     "hearing": ob_env2_space,   # 5 行 2 列的 ob_env2
        #     "delta_energy": ob_env3_space    # 单个连续值的 ob_env3
        # })
        self.totalobservation_space = spaces.Dict({
            f"{agent.name}": spaces.Dict({
                "sight": ob_env1_space,   # 每个智能体的视觉信息
                "hearing": ob_env2_space,  # 每个智能体的听觉信息
                "delta_energy": ob_env3_space  # 每个智能体的能量信息
            }) for agent in self.simulator.predators+self.simulator.preys
        })
        # self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.totalaction_space = spaces.Dict({
            f"{agent.name}": spaces.Dict({
                'makeAChild': spaces.Discrete(2),  # 是否生成后代 (0 或 1)
                'moveVector': spaces.Box(-10.0, 10.0, (2,), dtype=np.float32)  # 移动向量 (x, y)
            }) for agent in self.simulator.predators+self.simulator.preys
        })
        # Action space modification:
        # The first value is now discrete {0, 1}
        # The second and third values are continuous, as before
        # self.action_speed_range = max(constants.PREY_MAX_SPEED, constants.PREDATOR_MAX_SPEED)
        # self.action_low = np.full((2,), -self.action_speed_range, dtype=np.float32)
        # self.action_high = np.full((2,), self.action_speed_range, dtype=np.float32)
        # self.print_obs()
        # Define action space as Tuple of (Discrete, Box)
        # self.action_space = spaces.Tuple((
        #     spaces.Discrete(2),  # First value: discrete (0 or 1)
        #     spaces.Box(low=-self.action_speed_range, high=self.action_speed_range, shape=(2,), dtype=np.float32)  # 第二个和第三个变量是连续的，表示速度的 x 和 y 分量
        # ))
        # self.action_space = spaces.Tuple([self.action_space] * self.num_entities)

    # def print_obs(self):
    #     print(self.observation_space.sample())
    def observation_space(self, agent):
        """返回特定智能体的观测空间。"""

        return self.totalobservation_space[agent]

    def action_space(self, agent):
        """返回特定智能体的动作空间。"""
        return self.totalaction_space[agent]

    def _initialize_algorithm_encoding(self):
        """Initialize algorithm encoding."""
        self.agents = []
        
        self.prey_algorithm_encoding = {algo: idx + 2 for idx, algo in enumerate(set(self.prey_algorithms))}
        self.prey_algorithm_encoding["random"] = 1
        self.pred_algorithm_encoding = {algo: idx + 2 for idx, algo in enumerate(set(self.pred_algorithms))}
        self.pred_algorithm_encoding["random"] = 1

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        # super().reset(seed=seed, **kwargs)
        self.agents.clear()
        self.group_map.clear()
        self._set_random_seed(seed)
        self._initialize_agent_environment()


        # all_pred_algorithms, all_prey_algorithms = self.reset_algorithm()
        # self.simulator.initialize(all_pred_algorithms, all_prey_algorithms)

        # self.initialnames.extend(agent.name for agent in self.simulator.preys + self.simulator.predators)
        # self.map_agents_to_groups(self.simulator.predators, self.simulator.preys)

        # for predator in self.simulator.predators:
        #     self._set_agent_env(predator)
        # for prey in self.simulator.preys:
        #     self._set_agent_env(prey)

        obs = {agent.name: agent.get_observe_info() for group_name in self.group_map.keys() for agent in getattr(self.simulator, group_name)}
        # zero_data = {
        #     'sight': [
        #         (0, np.zeros(2, dtype=np.float32), 0) for _ in range(20)
        #     ],
        #     'hearing': np.zeros((5, 2), dtype=np.float32),
        #     'delta_energy': np.array(0., dtype=np.float32)
        # }
        infos =  {agent.name: {} for group_name in self.group_map.keys() for agent in getattr(self.simulator, group_name)}
        # obstestinfo = np.array([1,2,3],dtype=np.float32)
        # obstest = {agent.name: obstestinfo for group_name in self.group_map.keys() for agent in getattr(self.simulator, group_name)}

        return obs,infos

    def _set_random_seed(self, seed):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    def convert_dicts_to_numeric_keys(self,algorithm_encoding, algorithm_predict):
        """
        将 algorithm_encoding 中的值作为键映射到 algorithm_predict 中，生成数值键的字典。
        
        :param algorithm_encoding: 包含算法名称到数字映射的字典
        :param algorithm_predict: 包含算法名称到函数的字典
        :return: 使用数值键的新字典，数值来自 algorithm_encoding 的值
        """
        new_algorithm_predict = {}

        # 遍历 algorithm_encoding，重新构建使用数值键的字典
        for algorithm_name, numeric_key in algorithm_encoding.items():
            if algorithm_name in algorithm_predict:
                # 将数值键和对应的算法函数映射到新字典
                new_algorithm_predict[numeric_key] = algorithm_predict[algorithm_name]
            else:
                print(f"Warning: {algorithm_name} not found in algorithm_predict")

        return new_algorithm_predict

    def map_agents_to_groups(self, simPredators, simPreys):
        """Map agents to their respective groups."""
        self.group_map['predators'] = [predator.name for predator in simPredators]
        self.group_map['preys'] = [prey.name for prey in simPreys]

    def reset_algorithm(self):
        """Assign algorithms to agents and encode them."""
        all_pred_algorithms = self.assign_algorithms_to_agents(constants.NUM_PREDATORS, self.pred_algorithms)
        encoded_all_pred_algorithms = [self.pred_algorithm_encoding[algo] for algo in all_pred_algorithms]

        all_prey_algorithms = self.assign_algorithms_to_agents(constants.NUM_PREY, self.prey_algorithms)
        encoded_all_prey_algorithms = [self.prey_algorithm_encoding[algo] for algo in all_prey_algorithms]

        return encoded_all_pred_algorithms, encoded_all_prey_algorithms

    def _set_agent_env(self, agent):
        """Set environment properties for the agent."""
        agent.env_predators = self.simulator.predators
        agent.env_prey = self.simulator.preys
        agent.env_food = self.simulator.foods
        agent.env_obstacles = self.simulator.obstacles

    def step(self, actions):
        """Execute a step in the environment."""
        # initialdicts = dict(zip(self.initialnames, actions))
        self.simulator.add_food()
        self.simulator.move_models(actions=actions)
        self.simulator.prey_hunt()
        self.simulator.check_collisions()
        self.simulator.decrease_health()
        # self.simulator.remove_dead()

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        new_state, rewards, dones, infos = self._process_agents()
        self.agents = [agent for agent, done in dones.items() if not done]
        terminated = dones
        return new_state, rewards, terminated, terminated, infos
        
    def _process_agents(self):
        """Process each agent's state, reward, and done status."""
        new_state, rewards, dones, infos = {}, {}, {}, {}
        zero_data = {
            'sight': [
                (0, np.zeros(2, dtype=np.float32), 0) for _ in range(20)
            ],
            'hearing': np.zeros((5, 2), dtype=np.float32),
            'delta_energy': np.array(0., dtype=np.float32)
        }
        for name in self.possible_agents:
            matching_agent = next((agent for agent in self.simulator.preys + self.simulator.predators if agent.name == name), None)
            if matching_agent and matching_agent.is_alive:
                new_state[name]= matching_agent.get_observe_info()
                rewards[name]=self._compute_reward(matching_agent)
                dones[name]=False
                infos[name]={}
            else:
                new_state[name]=zero_data
                rewards[name]=0
                dones[name]=True # for test
                infos[name]={}
        return new_state, rewards, dones, infos

    def _compute_reward(self, agent):
        # 根据组别计算奖励
        if agent.type == 'predator':
            # 捕食者奖励
            return agent.health if agent.health > 0 else -1.0
        elif agent.type == 'prey':
            # 猎物奖励
            return agent.health if agent.health > 0 else -1.0
        return 0

    def render(self, mode='human'):
        """Render the environment."""
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((self.simulator.screen_width, self.simulator.screen_height)) if mode == 'human' else pygame.Surface((self.simulator.screen_width, self.simulator.screen_height))

        self.screen.fill((0, 0, 0))
        self.simulator.draw_models(self.screen)
        pygame.display.flip() if mode == 'human' else self._get_rgb_array()

    def _get_rgb_array(self):
        """Get the RGB array from the Pygame surface."""
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        """Close the environment."""
        pass

    def assign_algorithms_to_agents(self, len_agents, algorithm_names):
        """Assign algorithms to agents."""
        return [algorithm_names[i] if i < len(algorithm_names) else 'random' for i in range(len_agents)]


def update_and_plot(iteration, env, data_storage):
    """Update data and plot results."""
    predator_count = len(env.simulator.predators)
    prey_count = len(env.simulator.preys)
    food_count = len(env.simulator.foods)

    predator_total_health = sum(predator.health for predator in env.simulator.predators)
    prey_total_health = sum(prey.health for prey in env.simulator.preys)
    food_total_health = len(env.simulator.foods) * constants.FOOD_HEALTH_GAIN
    total_energy = predator_total_health + prey_total_health + food_total_health

    data_storage['iterations'].append(iteration)
    data_storage['predator_counts'].append(predator_count)
    data_storage['prey_counts'].append(prey_count)
    data_storage['total_counts'].append(predator_count + prey_count)
    data_storage['predator_healths'].append(predator_total_health)
    data_storage['prey_healths'].append(prey_total_health)
    data_storage['total_healths'].append(total_energy)

    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(data_storage['iterations'], data_storage['predator_counts'], label='Predator Count')
    plt.plot(data_storage['iterations'], data_storage['prey_counts'], label='Prey Count')
    plt.plot(data_storage['iterations'], data_storage['total_counts'], label='total Count')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.title('Number of Predators, Prey, and total Over Time')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(data_storage['iterations'], data_storage['predator_healths'], label='Predator Total Health')
    plt.plot(data_storage['iterations'], data_storage['prey_healths'], label='Prey Total Health')
    plt.plot(data_storage['iterations'], data_storage['total_healths'], label=' Total Health')
    plt.xlabel('Iteration')
    plt.ylabel('Total Health')
    plt.title('')
    plt.legend()

    plt.pause(0.01)  # Pause to update the plot in real-time


def run_random_simulation(env):
    """Run a random simulation with the environment."""
    obs, info = env.reset()

    data_storage = {
        'iterations': [],
        'predator_counts': [],
        'prey_counts': [],
        'total_counts': [],
        'predator_healths': [],
        'prey_healths': [],
        'total_healths': []
    }

    plt.figure(figsize=(10, 8))
    plt.ion()
    done = False
    iteration = 0
    while not done:
        env.render()

        if iteration % 100 == 1:
            update_and_plot(iteration, env, data_storage)
            print(len(env.simulator.predators),end="\t")
            print(len(env.simulator.preys))

        actions = env.action_space.sample()
        new_state, rewards, done, truncated, infos = env.step(actions)
        iteration += 1

        if iteration % 100 == 1:
            pass

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    prey_algorithms = ["PPO", "PPO", "PPO", "PPO", "DDPG", "DDPG", "DDPG"]
    pred_algorithms = ["PPO", "PPO", "PPO", "DDPG", "DDPG", "DDPG"]

    def ppo_predator_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        # 生成第一个变量a，离散值0或1
        a = np.random.choice([0, 1])
        
        # 生成x和y，连续变量，代表速度
        angle = np.random.uniform(0, 2 * np.pi)  # 随机角度
        length = np.random.uniform(0, max_speed)  # 随机长度，速度范围内
        x = length * np.cos(angle)  # x方向速度
        y = length * np.sin(angle)  # y方向速度

        # 将动作组合为(a, array([x, y]))，确保格式符合动作空间定义
        action = (a, np.array([x, y], dtype=np.float32))
        # if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
        #     return action
        # else:
        #     raise ValueError(f"ppo_predator_algorithm Generated action {action} is out of the action space bounds.")
        return action
    def dqn_predator_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        # 生成第一个变量a，离散值0或1
        a = np.random.choice([0, 1])
        
        # 生成x和y，连续变量，代表速度
        angle = np.random.uniform(0, 2 * np.pi)  # 随机角度
        length = np.random.uniform(0, max_speed)  # 随机长度，速度范围内
        x = length * np.cos(angle)  # x方向速度
        y = length * np.sin(angle)  # y方向速度

        # 将动作组合为(a, array([x, y]))，确保格式符合动作空间定义
        action = (a, np.array([x, y], dtype=np.float32))
        # if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
        #     return action
        # else:
        #     raise ValueError(f"dqn_predator_algorithm Generated action {action} is out of the action space bounds.")
        return action
    def random_predator_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        # 生成第一个变量a，离散值0或1
        a = np.random.choice([0, 1])
        
        # 生成x和y，连续变量，代表速度
        angle = np.random.uniform(0, 2 * np.pi)  # 随机角度
        length = np.random.uniform(0, max_speed)  # 随机长度，速度范围内
        x = length * np.cos(angle)  # x方向速度
        y = length * np.sin(angle)  # y方向速度

        # 将动作组合为(a, array([x, y]))，确保格式符合动作空间定义
        action = (a, np.array([x, y], dtype=np.float32))
        # if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
        #     return action
        # else:
        #     raise ValueError(f"random_predator_algorithm Generated action {action} is out of the action space bounds.")
        return action
    def ppo_prey_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        # 生成第一个变量a，离散值0或1
        a = np.random.choice([0, 1])
        
        # 生成x和y，连续变量，代表速度
        angle = np.random.uniform(0, 2 * np.pi)  # 随机角度
        length = np.random.uniform(0, max_speed)  # 随机长度，速度范围内
        x = length * np.cos(angle)  # x方向速度
        y = length * np.sin(angle)  # y方向速度

        # 将动作组合为(a, array([x, y]))，确保格式符合动作空间定义
        action = (a, np.array([x, y], dtype=np.float32))
        # if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
        #     return action
        # else:
        #     raise ValueError(f"ppo_prey_algorithm Generated action {action} is out of the action space bounds.")
        return action
    def dqn_prey_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        # 生成第一个变量a，离散值0或1
        a = np.random.choice([0, 1])
        
        # 生成x和y，连续变量，代表速度
        angle = np.random.uniform(0, 2 * np.pi)  # 随机角度
        length = np.random.uniform(0, max_speed)  # 随机长度，速度范围内
        x = length * np.cos(angle)  # x方向速度
        y = length * np.sin(angle)  # y方向速度

        # 将动作组合为(a, array([x, y]))，确保格式符合动作空间定义
        action = (a, np.array([x, y], dtype=np.float32))
        # if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
        #     return action
        # else:
        #     raise ValueError(f"dqn_prey_algorithm Generated action {action} is out of the action space bounds.")
        return action
    def random_prey_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        # 生成第一个变量a，离散值0或1
        a = np.random.choice([0, 1])
        
        # 生成x和y，连续变量，代表速度
        angle = np.random.uniform(0, 2 * np.pi)  # 随机角度
        length = np.random.uniform(0, max_speed)  # 随机长度，速度范围内
        x = length * np.cos(angle)  # x方向速度
        y = length * np.sin(angle)  # y方向速度

        # 将动作组合为(a, array([x, y]))，确保格式符合动作空间定义
        action = (a, np.array([x, y], dtype=np.float32))
        # if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
        #     return action
        # else:
        #     raise ValueError(f"random_prey_algorithm Generated action {action} is out of the action space bounds.")
        return action

    predator_algorithms_predict = {
        "PPO": lambda obs: ppo_predator_algorithm(obs, constants.PREY_MAX_SPEED), #change the function to yours ,must have random.
        "DDPG": lambda obs: dqn_predator_algorithm(obs, constants.PREY_MAX_SPEED),
        "random": lambda obs: random_predator_algorithm(obs, constants.PREY_MAX_SPEED)
    }

    prey_algorithms_predict = {
        "PPO": lambda obs: ppo_prey_algorithm(obs, constants.PREY_MAX_SPEED),
        "DDPG": lambda obs: dqn_prey_algorithm(obs, constants.PREY_MAX_SPEED),
        "random": lambda obs: random_prey_algorithm(obs, constants.PREY_MAX_SPEED)
    }

    env = LISPredatorPreyEnv(
        prey_algorithms=prey_algorithms,
        pred_algorithms=pred_algorithms,
        predator_algorithms_predict=predator_algorithms_predict,
        prey_algorithms_predict=prey_algorithms_predict
    )
    # env = parallel_to_aec(env)
    # # parallel_api_test(env)
    # api_test(env, num_cycles=10, verbose_progress=True)
    # parallel_api_test(env)
    run_random_simulation(env)
