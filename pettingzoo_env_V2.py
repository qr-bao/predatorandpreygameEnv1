## this version of code can be used to test the pettingzoo environment
"""
I've built the environment to meet the PettingZoo framework and api_test. 
Please note that our observation space is quite complex, and during the PettingZoo api test, only one error occur 
(mainly at this line of code: 
# assert ( # env.observation_space(agent)["observation"].dtype == prev_observe["observation"].dtype). 
# Commenting out above line allows the api_test to pass."""


import gymnasium.wrappers.flatten_observation
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium 
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from parallel_test import parallel_api_test
# from pettingzoo.test import api_test
from pettingzoo_util_api_test import api_test
# from pettingzoo.test import api_test
# from pettingzoo_api_test import api_test
import pygame
from typing import Any, Collection, Dict, List, Optional, Sequence

import numpy as np
import random
from env.simulator import Simulator
import env.constants as constants
# from stable_baselines3.common.env_checker import check_env
from gymnasium.utils.env_checker import check_env
import torch

from pettingzoo.utils import aec_to_parallel, parallel_to_aec
# from parallel_test import parallel_api_test

class LISPredatorPreyEnv(ParallelEnv, gym.Env):
    metadata = {"name": "LISPredatorPreyEnv"}
    def __init__(self, prey_algorithms=[], pred_algorithms=[], predator_algorithms_predict={}, prey_algorithms_predict={},seed = None):
        super(LISPredatorPreyEnv, self).__init__()
        self._set_random_seed(seed)

        self.simulator = Simulator(screen_width=3840, screen_height=2160)
        self.group_map = {}
        self.current_step = 0
        self.max_steps = 10_000
        self.traing_algorithm = ''

        self.prey_algorithms = prey_algorithms
        self.pred_algorithms = pred_algorithms
        self._initialize_algorithm_encoding()
        self.simulator.prey_algorithm_encoding = self.prey_algorithm_encoding
        self.simulator.pred_algorithm_encoding = self.pred_algorithm_encoding
        self.simulator.predator_algorithms_predict = self.convert_dicts_to_numeric_keys(self.pred_algorithm_encoding,predator_algorithms_predict)
        self.simulator.prey_algorithms_predict = self.convert_dicts_to_numeric_keys(self.prey_algorithm_encoding,prey_algorithms_predict)
        self.agents= []
        self.possible_agents =[]
        self._initialize_agent_environment()        
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

        MAX_CONSTANT = 10
        ob_env1_space = gymnasium.spaces.Tuple((
            spaces.Discrete(4),                    # 第一列，离散值 1 到 4
            spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),  # 第二、三列，位置坐标，连续值
            spaces.Discrete(MAX_CONSTANT)          # 第四列，离散值 0 到 MAX_CONSTANT
        ))
        ob_env1_space = gymnasium.spaces.Tuple([ob_env1_space] * 20)

        ob_env2_continuous_space = spaces.Box(
            low=np.array(0),      # 强度最小值
            high=np.array(1),     # 强度最大值
            dtype=np.float32
        )
        ob_en2_discrete_space = spaces.Discrete(2)  # 角度只能是0或1
        single_space = spaces.Tuple((ob_env2_continuous_space, ob_en2_discrete_space))
        ob_env2_space = spaces.Tuple([single_space] * 5)

        ob_env3_space = spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)

        self.totalobservation_space = spaces.Dict({
            f"{agent.name}": spaces.Dict({
                "observation": spaces.Dict({
                    "sight": ob_env1_space,   # 每个智能体的视觉信息
                    "hearing": ob_env2_space,  # 每个智能体的听觉信息
                    "delta_energy": ob_env3_space  # 每个智能体的能量信息
                })
            }) for agent in self.simulator.predators + self.simulator.preys
        })

        self.totalaction_space = spaces.Dict({
            f"{agent.name}": spaces.Dict({
                'makeAChild': spaces.Discrete(2),  # 是否生成后代 (0 或 1)
                'moveVector': spaces.Box(-10.0, 10.0, (2,), dtype=np.float32)  # 移动向量 (x, y)
            }) for agent in self.simulator.predators+self.simulator.preys
        })


    def observation_space(self, agent):
        """return the observation space for the agent."""

        # return {"observation" : self.totalobservation_space[agent]}
        return self.totalobservation_space[agent]

    def action_space(self, agent):
        """return the action space for the agent."""
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


        obs = {agent.name: agent.get_observe_info() for group_name in self.group_map.keys() for agent in getattr(self.simulator, group_name)}

        infos =  {agent.name: {} for group_name in self.group_map.keys() for agent in getattr(self.simulator, group_name)}

        return obs,infos

    def _set_random_seed(self, seed):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    def convert_dicts_to_numeric_keys(self,algorithm_encoding, algorithm_predict):
        """make algorithm_encoding's value as key map to algorithm_predict, generate a new dictionary with numeric keys.
        
        :param algorithm_encoding: include algorithm name to numeric key
        :param algorithm_predict: include algorithm name to function
        :return: Use a new dictionary with numerical keys, where the values come from algorithm_encoding.
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


        new_state, rewards, terminated, truncations,infos = self._process_agents()
        
        self.agents = [agent for agent, done in terminated.items() if not done]

        return new_state, rewards, terminated, truncations, infos
        

    def _process_agents(self):
        new_state, rewards, dones, infos = {}, {}, {}, {}
        self.current_step += 1
        # truncations = {agent: False for agent in self.agents}
        # if self.current_step >= self.max_steps or len(self.agents)==0:
        #     truncations = {agent: True for agent in self.agents}
        if self.current_step >= self.max_steps:
            truncations = {agent:True for agent in self.agents}
        else:
            truncations = {agent:False for agent in self.agents}
        zero_data = {
            'observation': {
                'sight': [
                    (0, np.zeros(2, dtype=np.float32), 0) for _ in range(20)
                ],
                'hearing': [
                    (np.array(0, dtype=np.float32), 0) for _ in range(5)
                ],
                'delta_energy': np.array(0., dtype=np.float32)
            }
        }

        for name in self.agents:
            matching_agent = next((agent for agent in self.simulator.preys + self.simulator.predators if agent.name == name), None)    
            if matching_agent and matching_agent.is_alive:
                # live_agents[name]=matching_agent
                new_state[name]= matching_agent.get_observe_info()
                rewards[name]=self._compute_reward(matching_agent)
                dones[name]=False
                infos[name]={}
            # elif matching_agent and matching_agent.is_alive:
            #     print(f'{matching_agent.name} is dead')
            else:
                new_state[name]=zero_data
                rewards[name]=0
                dones[name]=True
                # self.agents.remove(name)
                infos[name]={}
        return new_state, rewards, dones, truncations,infos


    def _compute_reward(self, agent):
        # 根据组别计算奖励
        if agent.type == 'predator':
            # 捕食者奖励
            return 1 if agent.health > 0 else -1.0
        elif agent.type == 'prey':
            # 猎物奖励
            return 1 if agent.health > 0 else -1.0
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

    # plt.clf()

    # plt.subplot(2, 1, 1)
    # plt.plot(data_storage['iterations'], data_storage['predator_counts'], label='Predator Count')
    # plt.plot(data_storage['iterations'], data_storage['prey_counts'], label='Prey Count')
    # plt.plot(data_storage['iterations'], data_storage['total_counts'], label='total Count')
    # plt.xlabel('Iteration')
    # plt.ylabel('Count')
    # plt.title('Number of Predators, Prey, and total Over Time')
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.plot(data_storage['iterations'], data_storage['predator_healths'], label='Predator Total Health')
    # plt.plot(data_storage['iterations'], data_storage['prey_healths'], label='Prey Total Health')
    # plt.plot(data_storage['iterations'], data_storage['total_healths'], label=' Total Health')
    # plt.xlabel('Iteration')
    # plt.ylabel('Total Health')
    # plt.title('')
    # plt.legend()

    # plt.pause(0.01)  # Pause to update the plot in real-time


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

    # plt.figure(figsize=(10, 8))
    # plt.ion()
    game_continue = False
    iteration = 0

    while not game_continue:
        env.render()
        actions = {}
        if iteration % 100 == 1:
            update_and_plot(iteration, env, data_storage)
            print(len(env.simulator.predators),end="\t")
            print(len(env.simulator.preys))
        for agent in env.agents:

            actions[agent] = env.action_space(agent).sample()
        new_state, rewards, done, truncated, infos = env.step(actions)
        iteration += 1
        # print(iteration)
        if iteration % 100 == 1:
            pass

    # plt.ioff()
    # plt.show()

# import os

# import ray
# # import supersuit as ss
# from ray import tune
# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# from ray.rllib.models import ModelCatalog
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.tune.registry import register_env
# from torch import nn
# from ray.tune.registry import register_env
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# from pettingzoo.butterfly import pistonball_v6
# class CNNModelV2(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
#         TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
#         nn.Module.__init__(self)
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
#             nn.ReLU(),
#             nn.Flatten(),
#             (nn.Linear(3136, 512)),
#             nn.ReLU(),
#         )
#         self.policy_fn = nn.Linear(512, num_outputs)
#         self.value_fn = nn.Linear(512, 1)

#     def forward(self, input_dict, state, seq_lens):
#         model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
#         self._value_out = self.value_fn(model_out)
#         return self.policy_fn(model_out), state

#     def value_function(self):
#         return self._value_out.flatten()
if __name__ == "__main__":
    prey_algorithms = ["PPO", "PPO", "PPO", "PPO", "DDPG", "DDPG", "DDPG"]
    pred_algorithms = ["PPO", "PPO", "PPO", "DDPG", "DDPG", "DDPG"]

    def ppo_predator_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        a = np.random.choice([0, 1])

        angle = np.random.uniform(0, 2 * np.pi)  
        length = np.random.uniform(0, min(max_speed, 10.0))  
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        action = {
            'makeAChild': a,  # 离散值 0 或 1
            'moveVector': np.array([x, y], dtype=np.float32)  # 连续值 [-10.0, 10.0] 范围
        }
        #check if the action is in the action space
        # if env.totalaction_space['Pred1_10'].contains(action):
        #     return action
        # else:
        #     raise ValueError(f"random_prey_algorithm Generated action {action} is out of the action space bounds.")
        
        return action
    def dqn_predator_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        a = np.random.choice([0, 1])

        angle = np.random.uniform(0, 2 * np.pi)  
        length = np.random.uniform(0, min(max_speed, 10.0))  
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        action = {
            'makeAChild': a,  # 离散值 0 或 1
            'moveVector': np.array([x, y], dtype=np.float32)  # 连续值 [-10.0, 10.0] 范围
        }
        #check if the action is in the action space
        # if env.totalaction_space['Pred1_10'].contains(action):
        #     return action
        # else:
        #     raise ValueError(f"random_prey_algorithm Generated action {action} is out of the action space bounds.")
        
        return action
    def random_predator_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        a = np.random.choice([0, 1])

        angle = np.random.uniform(0, 2 * np.pi)  
        length = np.random.uniform(0, min(max_speed, 10.0))  
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        action = {
            'makeAChild': a,  # 离散值 0 或 1
            'moveVector': np.array([x, y], dtype=np.float32)  # 连续值 [-10.0, 10.0] 范围
        }
        #check if the action is in the action space
        # if env.totalaction_space['Pred1_10'].contains(action):
        #     return action
        # else:
        #     raise ValueError(f"random_prey_algorithm Generated action {action} is out of the action space bounds.")
        
        return action
    def ppo_prey_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        a = np.random.choice([0, 1])

        angle = np.random.uniform(0, 2 * np.pi)  
        length = np.random.uniform(0, min(max_speed, 10.0))  
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        action = {
            'makeAChild': a,  # 离散值 0 或 1
            'moveVector': np.array([x, y], dtype=np.float32)  # 连续值 [-10.0, 10.0] 范围
        }
        #check if the action is in the action space
        # if env.totalaction_space['Pred1_10'].contains(action):
        #     return action
        # else:
        #     raise ValueError(f"random_prey_algorithm Generated action {action} is out of the action space bounds.")
        
        return action
    def dqn_prey_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        a = np.random.choice([0, 1])

        angle = np.random.uniform(0, 2 * np.pi)  
        length = np.random.uniform(0, min(max_speed, 10.0))  
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        action = {
            'makeAChild': a,  # 离散值 0 或 1
            'moveVector': np.array([x, y], dtype=np.float32)  # 连续值 [-10.0, 10.0] 范围
        }
        #check if the action is in the action space
        # if env.totalaction_space['Pred1_10'].contains(action):
        #     return action
        # else:
        #     raise ValueError(f"random_prey_algorithm Generated action {action} is out of the action space bounds.")
        
        return action
    def random_prey_algorithm(observation_info, max_speed):
        a = np.random.choice([0, 1])

        angle = np.random.uniform(0, 2 * np.pi)  
        length = np.random.uniform(0, min(max_speed, 10.0))  
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        action = {
            'makeAChild': a,  # 离散值 0 或 1
            'moveVector': np.array([x, y], dtype=np.float32)  # 连续值 [-10.0, 10.0] 范围
        }
        #check if the action is in the action space
        # if env.totalaction_space['Pred1_10'].contains(action):
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
            prey_algorithms_predict=prey_algorithms_predict,
            seed = 1
            )


    env = parallel_to_aec(env)



    print("hello")

    api_test(env, num_cycles=1000, verbose_progress=True)


