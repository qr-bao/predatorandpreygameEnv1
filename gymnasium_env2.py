
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
    def __init__(
            self,
            prey_algorithms =[],
            pred_algorithms=[],
            predator_algorithms_predict ={},
            prey_algorithms_predict={} 
            ):
        super(LISPredatorPreyEnv, self).__init__()
        
        # 初始化模拟器
        self.simulator = Simulator(screen_width=3840, screen_height=2160)
        self.group_map = {}
        self.current_step = 0
        self.max_steps = 10_000


        
        # 初始化观察和动作空间
        self.max_range = max(600, 1000)  
        # self.zero_list = [[0 for _ in range(4)] for _ in range(25)]
        self.num_entities = constants.NUM_PREDATORS + constants.NUM_PREY
        self.new_shape = (self.num_entities, 25, 4)
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
        self.prey_algorithm_encoding  = {algo: idx + 2 for idx, algo in enumerate(set(prey_algorithms))}
        self.prey_algorithm_encoding["random"] = 1
        self.pred_algorithms = pred_algorithms
        self.pred_algorithm_encoding = {algo: idx + 2 for idx, algo in enumerate(set(pred_algorithms))}
        self.pred_algorithm_encoding["random"] = 1

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
        # allalgorithms= self.reset_algorithm()
        all_pred_algorithms, all_prey_algorithms =  self.reset_algorithm()
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
        encoded_all_pred_algorithms = [self.pred_algorithm_encoding[algo] for algo in all_pred_algorithms]

        all_prey_algorithms = self.assign_algorithms_to_agents(constants.NUM_PREY,self.prey_algorithms)
        encoded_all_prey_algorithms = [self.prey_algorithm_encoding[algo] for algo in all_prey_algorithms]


        return encoded_all_pred_algorithms ,encoded_all_prey_algorithms

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
                new_state.append(np.zeros((25, 4)))
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

import matplotlib.pyplot as plt

def update_and_plot(iteration, env, data_storage):
    # Count the number of predators, prey, and food
    predator_count = len(env.simulator.predators)
    prey_count = len(env.simulator.preys)
    food_count = len(env.simulator.foods)

    # Calculate total healths
    predator_total_health = sum(predator.health for predator in env.simulator.predators)
    prey_total_health = sum(prey.health for prey in env.simulator.preys)
    food_total_health = (len(env.simulator.foods) * constants.FOOD_HEALTH_GAIN)
    total_energy = predator_total_health+prey_total_health+food_total_health

    # Store the current values
    data_storage['iterations'].append(iteration)
    data_storage['predator_counts'].append(predator_count)
    data_storage['prey_counts'].append(prey_count)
    data_storage['total_counts'].append(predator_count+prey_count)
    data_storage['predator_healths'].append(predator_total_health)
    data_storage['prey_healths'].append(prey_total_health)
    data_storage['total_healths'].append(total_energy)

    # Update plots
    plt.clf()

    # Plot counts
    plt.subplot(2, 1, 1)
    plt.plot(data_storage['iterations'], data_storage['predator_counts'], label='Predator Count')
    plt.plot(data_storage['iterations'], data_storage['prey_counts'], label='Prey Count')
    plt.plot(data_storage['iterations'], data_storage['total_counts'], label='total Count')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.title('Number of Predators, Prey, and total Over Time')
    plt.legend()

    # Plot healths
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
    # Initialize lists to store the values for each iteration

    # Initialize a dictionary to store the data
    data_storage = {
        'iterations': [],
        'predator_counts': [],
        'prey_counts': [],
        'total_counts': [],
        'predator_healths': [],
        'prey_healths': [],
        'total_healths': []
    }

    # Initialize the plot
    plt.figure(figsize=(10, 8))
    plt.ion()  # Enable interactive mode
    done = False
    iteration = 0

    while not done:
        

        # actions = {
        #     'predators': generate_random_actions(len(env.simulator.predators), env.action_space),
        #     'preys': generate_random_actions(len(env.simulator.preys), env.action_space),
        # }
        if iteration % 100 == 1:  
            update_and_plot(iteration, env, data_storage)

        actions = env.action_space.sample()  # 从动作空间中采样一个随机动作 # you can change this with your algorithm
        new_state, rewards, done,truncated, infos = env.step(actions)

        # 判断是否所有智能体都已经完成
        # done = all(all(done_group) for done_group in dones.values())
        iteration +=1

        # 渲染环境（可选）
        env.render()
        if iteration % 100 == 1:   
            pass
            # print(f"iteration: {iteration}, num_predators: {len(env.simulator.predators)}, num_preys: {len(env.simulator.preys)}")

            # print(iteration,end="\t")
            print(len(env.simulator.predators),end="\t")
            print(len(env.simulator.preys))
        # 打印当前状态、奖励、是否结束
            # print(f"New State: {new_state}")
            # print(f"Rewards: {new_state}")
            # print(len(new_state))
            # print(f"Dones: {np.shape(done)}")
            # print(f"Dones length:{len(done)}")
            # print(f"Infos: {infos}")
    plt.ioff()  # Disable interactive mode
    plt.show()  # Keep the final plot open after the loop ends

if __name__ == "__main__":

    # register(
    #     id='LISPredatorPreyEnv-v0',
    #     entry_point='gym_env:LISPredatorPreyEnv',
    # )

    env = LISPredatorPreyEnv()
    # obs,info = env.reset()  # 重置环境并获取初始观测
    # print("Initial observation:", obs)
    # print(env.simulator.agent_status)

    # for _ in range(10):  # 运行10个时间步
    #     action = env.action_space.sample()  # 随机采样一个动作
    #     new_observations, rewards, terminated,truncated, infos = env.step(action)  # 采取一步行动
    #     # print(f"Observation: {obs}, Reward: {rewards}, Done: {terminated}, Info: {infos}")
    #     print(f"Observation: {type(obs)}, Reward: {np.shape(rewards)}, Done: {np.shape(terminated)}, Info: {np.shape(infos)}")

    #     # print(np.shape(obs))
    #     if terminated:
    #         obs,info = env.reset()  # 如果环境结束了，则重置环境
    check_env(env.unwrapped)
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=10)
    
    
    model.learn(total_timesteps=10)  # 可以根据需求调整时间步数
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
        # env.render()  # 可视化环境（如果需要）

    check_env(env.unwrapped)
    # check_env(env)

    # run_random_simulation(env)
