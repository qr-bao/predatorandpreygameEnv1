import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from env.simulator import Simulator
import env.constants as constants
from stable_baselines3.common.env_checker import check_env
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
import torch
import pandas as pd
import matplotlib.pyplot as plt

class LISPredatorPreyEnv(gym.Env):
    def __init__(self, prey_algorithms=[], pred_algorithms=[], predator_algorithms_predict={}, prey_algorithms_predict={}):
        super(LISPredatorPreyEnv, self).__init__()

        self.simulator = Simulator(screen_width=3840, screen_height=2160)
        self.group_map = {}
        self.current_step = 0
        self.max_steps = 10_000
        self.traing_algorithm = ''

        # 初始化观察和动作空间
        self._initialize_spaces()

        # 初始化算法
        self.prey_algorithms = prey_algorithms
        self.pred_algorithms = pred_algorithms
        self._initialize_algorithm_encoding()

        self.simulator.predator_algorithms_predict = predator_algorithms_predict
        self.simulator.prey_algorithms_predict = prey_algorithms_predict

    def _initialize_spaces(self):
        """Initialize observation and action spaces."""
        self.max_range = max(600, 1000)
        self.num_entities = constants.NUM_PREDATORS + constants.NUM_PREY
        self.new_shape = (self.num_entities, 25, 4)
        self.obs_low = np.full(self.new_shape, -self.max_range, dtype=np.float32)
        self.obs_high = np.full(self.new_shape, self.max_range, dtype=np.float32)
        self.action_shape = (self.num_entities, 3)
        self.action_speed_range = max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
        self.action_low = np.full(self.action_shape, -self.action_speed_range, dtype=np.float32)
        self.action_high = np.full(self.action_shape, self.action_speed_range, dtype=np.float32)
        self.action_low[:, 0] = 0.0
        self.action_high[:, 0] = 1.0
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)

    def _initialize_algorithm_encoding(self):
        """Initialize algorithm encoding."""
        self.initialnames = []
        self.prey_algorithm_encoding = {algo: idx + 2 for idx, algo in enumerate(set(self.prey_algorithms))}
        self.prey_algorithm_encoding["random"] = 1
        self.pred_algorithm_encoding = {algo: idx + 2 for idx, algo in enumerate(set(self.pred_algorithms))}
        self.pred_algorithm_encoding["random"] = 1

    def reset(self, seed=None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)
        self.initialnames.clear()
        self.group_map.clear()
        self._set_random_seed(seed)

        all_pred_algorithms, all_prey_algorithms = self.reset_algorithm()
        self.simulator.initialize(all_pred_algorithms, all_prey_algorithms)

        self.initialnames.extend(agent.name for agent in self.simulator.preys + self.simulator.predators)
        self.map_agents_to_groups(self.simulator.predators, self.simulator.preys)

        for predator in self.simulator.predators:
            self._set_agent_env(predator)
        for prey in self.simulator.preys:
            self._set_agent_env(prey)

        obs = np.array([agent.get_observe_info() for group_name in self.group_map.keys() for agent in getattr(self.simulator, group_name)], dtype=np.float32)

        info = {}
        return obs, info

    def _set_random_seed(self, seed):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

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
        initialdicts = dict(zip(self.initialnames, actions))
        self.simulator.add_food()
        self.simulator.move_models(actions=initialdicts)
        self.simulator.prey_hunt()
        self.simulator.check_collisions()
        self.simulator.decrease_health()
        self.simulator.remove_dead()

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        new_state, rewards, dones, infos = self._process_agents()

        terminated = all(dones)
        return np.array(new_state, dtype=np.float32), sum(rewards), terminated, truncated, {}

    def _process_agents(self):
        """Process each agent's state, reward, and done status."""
        new_state, rewards, dones, infos = [], [], [], []
        for name in self.initialnames:
            matching_agent = next((agent for agent in self.simulator.preys + self.simulator.predators if agent.name == name), None)
            if matching_agent and matching_agent.is_alive:
                new_state.append(matching_agent.get_observe_info())
                rewards.append(self._compute_reward(matching_agent))
                dones.append(False)
                infos.append({})
            else:
                new_state.append(np.zeros((25, 4)))
                rewards.append(0)
                dones.append(True)
                infos.append({})
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
        # env.render()

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
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0, max_speed)
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        # 检查生成的动作是否符合定义的动作空间
        action = np.array([a, x, y], dtype=np.float32)
        if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
            return action
        else:
            raise ValueError(f"Generated action {action} is out of the action space bounds.")
    def dqn_predator_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0, max_speed)
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        # 检查生成的动作是否符合定义的动作空间
        action = np.array([a, x, y], dtype=np.float32)
        if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
            return action
        else:
            raise ValueError(f"Generated action {action} is out of the action space bounds.")
    def random_predator_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0, max_speed)
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        # 检查生成的动作是否符合定义的动作空间
        action = np.array([a, x, y], dtype=np.float32)
        if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
            return action
        else:
            raise ValueError(f"Generated action {action} is out of the action space bounds.")
    def ppo_prey_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0, max_speed)
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        # 检查生成的动作是否符合定义的动作空间
        action = np.array([a, x, y], dtype=np.float32)
        if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
            return action
        else:
            raise ValueError(f"Generated action {action} is out of the action space bounds.")
        
    def dqn_prey_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0, max_speed)
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        # 检查生成的动作是否符合定义的动作空间
        action = np.array([a, x, y], dtype=np.float32)
        if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
            return action
        else:
            raise ValueError(f"Generated action {action} is out of the action space bounds.")
    def random_prey_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0, max_speed)
        a = np.random.uniform(0, 1)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        # 检查生成的动作是否符合定义的动作空间
        action = np.array([a, x, y], dtype=np.float32)
        if env.action_space.contains(action):# env action space : [a,x,y] a:from 0 to 1,x,y: ± max(constants.PREY_MAX_SPEED, constants.PREY_MAX_SPEED)
            return action
        else:
            raise ValueError(f"Generated action {action} is out of the action space bounds.")

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

    check_env(env.unwrapped)
    check_env(env)

    # model = TD3("MlpPolicy", env, verbose=1)
    
    
    # model.learn(total_timesteps=10000)  # 可以根据需求调整时间步数
    # model.save("algorithms/TD3_predator_prey")
    # 加载训练好的模型
    model = TD3.load("algorithms/TD3_predator_prey")

    # 重置环境
    obs, info = env.reset()
    
    # 测试模型
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done,truncated, info = env.step(action)
        print(len(env.simulator.agent_status))

