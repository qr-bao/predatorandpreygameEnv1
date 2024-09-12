
# import gym
import gymnasium as gym
# from gym import spaces
from gymnasium import spaces
import pygame
import numpy as np
import random
from env.simulator import Simulator
import env.constants as constants
from torchrl.envs.utils import check_env_specs
# from gym.utils.env_checker import check_env
# from gym.envs.registration import register
from gymnasium.envs.registration import register
from tensordict import TensorDict
# import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch
import pandas as pd
import matplotlib.pyplot as plt


class LISPredatorPreyEnv(gym.Env):
    def __init__(self,prey_algorithms =[],pred_algorithms=[],predator_algorithms_predict ={},prey_algorithms_predict={} ):
        super(LISPredatorPreyEnv, self).__init__()
        
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
        self.action_shape = (self.num_entities,3)
        self.action_speed_range = max(constants.PREY_MAX_SPEED,constants.PREY_MAX_SPEED)
        self.action_low = np.full(self.action_shape, -self.action_speed_range, dtype=np.float32)
        self.action_high = np.full(self.action_shape, self.action_speed_range, dtype=np.float32)
        self.action_low[:, 0] = 0.0   # 将第一列的低值设为0
        self.action_high[:, 0] = 1.0  # 将第一列的高值设为1
        # obs_low = np.array([0, 0, 0] * 25*(constants.NUM_PREDATORS+constants.NUM_PREY))
        # obs_high = np.array([max_range, max_range, max_range] * 25*(constants.NUM_PREDATORS+constants.NUM_PREY))
        # self.observation_space_shape = (constants.NUM_PREDATORS+constants.NUM_PREY) * 3 * 25
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high,dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)

        self.interation = 0
        self.initialnames = []
        self.prey_algorithms = prey_algorithms
        self.pred_algorithms = pred_algorithms
        self.simulator.predator_algorithms_predict = predator_algorithms_predict
        self.simulator.prey_algorithms_predict = prey_algorithms_predict

        # self.initialdicts = {}
        # 调用 reset 方法初始化环境
        # self.reset() 


        
    def reset(self, seed=None, **kwargs):
        # 重置模拟器
        super().reset(seed=seed, **kwargs)
        self.initialnames = []
        self.group_map.clear()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        allalgorithms= self.reset_algorithm()
        all_pred_algorithms, all_prey_algorithms = allalgorithms[:constants.NUM_PREDATORS],allalgorithms[constants.NUM_PREDATORS:]
        self.simulator.initialize(all_pred_algorithms, all_prey_algorithms)
        for agent in self.simulator.preys +self.simulator.predators: 
            self.initialnames.append(agent.name)
        self.map_agents_to_groups(self.simulator.predators, self.simulator.preys)
        # 初始化环境信息（捕食者、猎物、食物和障碍物）
        for predator in self.simulator.predators:
            self._set_agent_env(predator)
        for prey in self.simulator.preys:
            self._set_agent_env(prey)
        # 获取所有智能体的初始观测数据
        all_observations = []
        # change here if you want change agent
        for group_name in self.group_map.keys():
            for agent in getattr(self.simulator, group_name):
                all_observations.append(agent.get_observe_info())
        
        obs = np.array(all_observations, dtype=np.float32) #np.shape(all_observations) = (2250,3)
        # print(np.shape(all_observations))
        # obs = TensorDict({
        #     'observation': torch.tensor(obs)
        # }, batch_size=[])
        info = {}
        # print("Initial Observation:", obs)
        # assert self.observation_space.contains(obs), "Initial observation is out of bounds!"

        # 返回一个数组，符合 observation_space 的定义
        return obs,info

    def map_agents_to_groups(self,simPredators,simPreys):
        self.group_map['predators'] = [predator.name for predator in simPredators]
        self.group_map['preys'] = [prey.name for prey in simPreys]

    def reset_algorithm(self):

        all_pred_algorithms = self.assign_algorithms_to_agents(constants.NUM_PREDATORS,self.pred_algorithms)
        all_prey_algorithms = self.assign_algorithms_to_agents(constants.NUM_PREY,self.prey_algorithms)

        return all_pred_algorithms + all_prey_algorithms

    def _set_agent_env(self, agent):
        agent.env_predators = self.simulator.predators
        agent.env_prey = self.simulator.preys
        agent.env_food = self.simulator.foods
        agent.env_obstacles = self.simulator.obstacles



    def step(self, actions):
        new_state, rewards, dones, infos = [], [], [], []
        initialdicts = dict(zip(self.initialnames, actions))
        self.simulator.add_food()  # 传递时间间隔
        self.simulator.move_models(actions =initialdicts)

        self.simulator.prey_hunt()
        self.simulator.check_collisions(initialdicts)
        self.simulator.decrease_health()  # 更新健康值
        self.simulator.remove_dead()  # 清理死亡个体

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        for name in self.initialnames:
            # 查找 B 列表中是否有 agent 的名字和 A 列表中的名字匹配
            matching_agent = next((agent for agent in self.simulator.preys+self.simulator.predators if agent.name == name), None)
            
            if matching_agent and matching_agent.is_alive == True:
                # 如果在 B 列表中找到匹配的 agent，获取其状态和奖励信息
                new_state.append(matching_agent.get_observe_info())
                rewards.append(self._compute_reward(matching_agent))  # 或者根据需要计算奖励
                dones.append(False)
                infos.append({})

            else:
                # 如果未找到匹配的 agent，说明该 agent 已死亡
                new_state.append(np.zeros((25, 3)))
                rewards.append(0)
                dones.append(True)
                infos.append({})

        terminated = all(dones)   

        return np.array(new_state,dtype=np.float32), sum(rewards), terminated, truncated,{}
    
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
        # Initialize Pygame screen if not already initialized
        if not hasattr(self, 'screen'):
            pygame.init()
            if mode == 'human':
                self.screen = pygame.display.set_mode((self.simulator.screen_width, self.simulator.screen_height))
            elif mode == 'rgb_array':
                self.screen = pygame.Surface((self.simulator.screen_width, self.simulator.screen_height))

        # Fill the background with black color
        self.screen.fill((0, 0, 0))
        # print(self.simulator.predators)
        # Draw models onto the screen
        self.simulator.draw_models(self.screen)

        # Update the display if mode is 'human'
        if mode == 'human':
            pygame.display.flip()
        elif mode == 'rgb_array':
            return self._get_rgb_array()

    def _get_rgb_array(self):
        # Convert Pygame surface to an RGB array (numpy)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)),
            axes=(1, 0, 2)
        )

    def close(self):
        # 关闭环境
        pass




    def generate_random_actions(self,num_agents, action_space):
        actions = []
        for _ in range(num_agents):
            action = action_space.sample()  # 从动作空间中采样一个随机动作
            # print(action)
            actions.append(action)
        return actions

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
    def apply_algorithms_to_agents(self,agents, algorithms):
        """
        将算法分配给每个智能体。

        参数:
        - agents: 智能体列表。
        - algorithms: 已分配的算法名称列表。
        """
        for agent, algorithm in zip(agents, algorithms):
            agent.algorithm = algorithm  # 将算法分配给智能体

if __name__ == "__main__":

    # register(
    #     id='LISPredatorPreyEnv-v0',
    #     entry_point='gym_env:LISPredatorPreyEnv',
    # )

    env = LISPredatorPreyEnv()
    prey_algorithms = ["PPO","PPO","PPO","PPO","DDPG","DDPG","DDPG"]
    pred_algorithms = ["PPO","PPO","PPO","DDPG","DDPG","DDPG"]
    # Define the algorithm functions
    def ppo_predator_algorithm(observation_info):
        angle = np.random.uniform(0, 2 * np.pi)

        # Generate random length less than A
        length = np.random.uniform(0, constants.PREY_MAX_SPEED)

        # Calculate x and y based on angle and length
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        velocity = np.array([a,x, y], dtype=np.float32)
        return velocity

    def dqn_predator_algorithm(observation_info):
        angle = np.random.uniform(0, 2 * np.pi)

        # Generate random length less than A
        length = np.random.uniform(0, constants.PREY_MAX_SPEED)

        # Calculate x and y based on angle and length
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        velocity = np.array([a,x, y], dtype=np.float32)
        return velocity

    def random_predator_algorithm(observation_info):
        angle = np.random.uniform(0, 2 * np.pi)

        # Generate random length less than A
        length = np.random.uniform(0, constants.PREY_MAX_SPEED)

        # Calculate x and y based on angle and length
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        velocity = np.array([a,x, y], dtype=np.float32)
        return velocity

    def ppo_prey_algorithm(observation_info):
        angle = np.random.uniform(0, 2 * np.pi)

        # Generate random length less than A
        length = np.random.uniform(0, constants.PREY_MAX_SPEED)

        # Calculate x and y based on angle and length
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        velocity = np.array([a,x, y], dtype=np.float32)
        return velocity

    def dqn_prey_algorithm(observation_info):
        angle = np.random.uniform(0, 2 * np.pi)

        # Generate random length less than A
        length = np.random.uniform(0, constants.PREY_MAX_SPEED)

        # Calculate x and y based on angle and length
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        velocity = np.array([a,x, y], dtype=np.float32)
        return velocity
    
    def random_prey_algorithm(observation_info):
        angle = np.random.uniform(0, 2 * np.pi)

        # Generate random length less than A
        length = np.random.uniform(0, constants.PREY_MAX_SPEED)

        # Calculate x and y based on angle and length
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        velocity = np.array([a,x, y], dtype=np.float32)
        return velocity

    # Create dictionaries to store algorithms
    predator_algorithms_predict = {
        "PPO": ppo_predator_algorithm,
        "DDPG": dqn_predator_algorithm,
        "random": random_predator_algorithm
    }

    prey_algorithms_predict = {
        "PPO": ppo_prey_algorithm,
        "DDPG": dqn_prey_algorithm,
        "random":random_prey_algorithm
    }
    env = LISPredatorPreyEnv(prey_algorithms=prey_algorithms,pred_algorithms=pred_algorithms,predator_algorithms_predict =predator_algorithms_predict,prey_algorithms_predict =prey_algorithms_predict)
    check_env(env.unwrapped)
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)
    
    
    model.learn(total_timesteps=2)  # 可以根据需求调整时间步数
    model.save("ppo_predator_prey")
    # 加载训练好的模型
    model = PPO.load("ppo_predator_prey")

    # 重置环境
    obs, info = env.reset()

    # 测试模型
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done,truncated, info = env.step(action)
        print(len(env.simulator.agent_status))
        env.render()  # 可视化环境（如果需要）

    check_env(env.unwrapped)
