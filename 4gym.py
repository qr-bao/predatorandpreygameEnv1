
import gym
from gym import spaces
import pygame
import numpy as np
from simulator import Simulator
import constants
from torchrl.envs.utils import check_env_specs
from gym.utils.env_checker import check_env

class PredatorPreyEnv(gym.Env):
    def __init__(self):
        super(PredatorPreyEnv, self).__init__()
        
        # 初始化模拟器
        self.simulator = Simulator(screen_width=3840, screen_height=2160)

        self.group_map = {}
        self.current_step = 0
        self.max_steps = 10_000


        
        # 初始化观察和动作空间
        max_range = max(600, 800)
        obs_low = np.array([0, 0, 0] * 25 * (constants.NUM_PREDATORS+constants.NUM_PREY))
        obs_high = np.array([max_range, max_range, max_range] * 25 *(constants.NUM_PREDATORS+constants.NUM_PREY))
        self.observation_space_shape = 3* 25 *(constants.NUM_PREDATORS+constants.NUM_PREY)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(self.observation_space_shape,),dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.interation = 0
        # 调用 reset 方法初始化环境
        self.reset()

    # def reset(self):
    #     # 重置模拟器
    #     all_pred_algorithms,all_prey_algorithms = self.reset_algorithm()
    #     self.simulator.initialize(all_pred_algorithms,all_prey_algorithms)
    #     self.map_agents_to_groups(self.simulator.predators,self.simulator.preys)
    #     # 初始化环境信息（捕食者、猎物、食物和障碍物）
    #     for predator in self.simulator.predators:
    #         self._set_agent_env(predator)
    #     for prey in self.simulator.preys:
    #         self._set_agent_env(prey)
    #     # 获取所有智能体的初始观测数据
    #     observations = {}
    #     for group_name in self.group_map.keys():
    #         observations[group_name] = []
    #         for agent in getattr(self.simulator, group_name):
    #             observations[group_name].append(agent.get_observe_info())
    #     info  = {}

    #     # 返回包含所有智能体观测数据的字典
    #     return observations,info
        
    def reset(self):
        # 重置模拟器
        all_pred_algorithms, all_prey_algorithms = self.reset_algorithm()
        self.simulator.initialize(all_pred_algorithms, all_prey_algorithms)
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
                all_observations.extend(agent.get_observe_info())
        
        obs = np.array(all_observations, dtype=np.float32)
        info = {}

        # 返回一个数组，符合 observation_space 的定义
        return obs, info

    def map_agents_to_groups(self,simPredators,simPreys):
        self.group_map['predators'] = [predator.name for predator in simPredators]
        self.group_map['preys'] = [prey.name for prey in simPreys]

    def reset_algorithm(self):
        prey_algorithms = ["PPO","PPO","PPO","DDPG","DDPG","DDPG"]
        pred_algorithms = ["PPO","PPO","PPO","DDPG","DDPG","DDPG"]
        all_pred_algorithms = assign_algorithms_to_agents(constants.NUM_PREDATORS,pred_algorithms)
        all_prey_algorithms = assign_algorithms_to_agents(constants.NUM_PREY,prey_algorithms)
        return all_pred_algorithms,all_prey_algorithms
        # assigned_Predalgorithms = assign_algorithms_to_agents(self.simulator.predators, all_pred_algorithms)
        # assigned_Preyalgorithms = assign_algorithms_to_agents(self.simulator.preys, all_prey_algorithms)

    def _set_agent_env(self, agent):
        agent.env_predators = self.simulator.predators
        agent.env_prey = self.simulator.preys
        agent.env_food = self.simulator.foods
        agent.env_obstacles = self.simulator.obstacles

    def step(self, actions):
        new_state, rewards, dones, infos = {}, {}, {}, {}
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        # self.simulator.check_events()

        
        # self.simulator.move_models()

        # 独立处理每个组的动作
        for group_name in self.group_map.keys():
            group_actions = actions[group_name]
            new_state[group_name], rewards[group_name], dones[group_name], infos[group_name] = self._step_group(group_name, group_actions)
        self.simulator.add_food()  # 传递时间间隔
        self.simulator.prey_hunt()
        self.simulator.check_collisions()

        # self.simulator.predator_hunt()
        self.simulator.decrease_health()  # 更新健康值
        self.simulator.remove_dead()  # 清理死亡个体
        # self.simulator.draw_models(screen)

        # sim.check_events()

        
        # sim.move_models()
        # sim.add_food()  # 传递时间间隔
        # sim.prey_hunt()
        # sim.check_collisions()
        # # sim.predator_hunt()
        # # new_prey_born, new_predator_born = sim.applyGeneticAlgorithm()
        # sim.decrease_health()  # 更新健康值
        # sim.remove_dead()  # 清理死亡个体
        # iteration_count += 1  # 增加迭代计数器
        # sim.draw_models(screen)

        # new_prey_born, new_predator_born = self.simulator.applyGeneticAlgorithm()

        return new_state["predators"][0], rewards["predators"][0], dones["predators"][0], truncated, infos

    def _step_group(self, group_name, group_actions):
        # 执行每个组的动作，并获取新的状态、奖励、是否完成和信息
        new_observations = []
        rewards = []
        dones = []
        infos = []
        # print(np.shape(group_actions))
        group = getattr(self.simulator, group_name)
        for agent, action in zip(group, group_actions):
            agent.move_strategy(action)
            agent.move(constants.CONTROL_PANEL_WIDTH, self.simulator.screen_width, self.simulator.screen_height, self.simulator.obstacles)
            
            new_observations.append(agent.get_observe_info())
            rewards.append(self._compute_reward(agent, group_name))
            dones.append(not agent.is_alive)  # 这里假设死亡标志环境结束
            infos.append({})  # 可以添加更多的调试信息

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
def generate_random_actions(num_agents, action_space):
    actions = []
    for _ in range(num_agents):
        action = action_space.sample()  # 从动作空间中采样一个随机动作
        # print(action)
        actions.append(action)
    return actions

def assign_algorithms_to_agents(len_agents, algorithm_names):
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
def apply_algorithms_to_agents(agents, algorithms):
    """
    将算法分配给每个智能体。

    参数:
    - agents: 智能体列表。
    - algorithms: 已分配的算法名称列表。
    """
    for agent, algorithm in zip(agents, algorithms):
        agent.algorithm = algorithm  # 将算法分配给智能体

def run_random_simulation():
    
    env = PredatorPreyEnv()
    observations = env.reset()
    # check_env_specs(env)
    check_env(env)

    # rollout = env.rollout(10)
    # print(f"rollout of {10} steps:", rollout)
    # print("Shape of the rollout TensorDict:", rollout.batch_size)
    

    observations = env.reset()
    #print(np.shape(observation))
    done = False
    iteration = 0
    while not done:
        actions = {
            'predators': generate_random_actions(len(env.simulator.predators), env.action_space),
            'preys': generate_random_actions(len(env.simulator.preys), env.action_space),
        }

        new_state, rewards, dones, infos = env.step(actions)

        # 判断是否所有智能体都已经完成
        done = all(all(done_group) for done_group in dones.values())
        iteration +=1

        # 渲染环境（可选）
        env.render()
        if iteration == 10:   
              
            print(iteration)
        # 打印当前状态、奖励、是否结束
            # print(f"New State: {new_state}")
            # print(f"Rewards: {rewards}")
            print(f"Dones: {np.shape(dones)}")
            print(f"Dones length:{len(dones)}")
            # print(f"Infos: {infos}")

if __name__ == "__main__":
    run_random_simulation()