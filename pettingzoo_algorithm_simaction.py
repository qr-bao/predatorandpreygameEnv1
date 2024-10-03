import supersuit as ss
import env.constants as constants
from ray.rllib.env import PettingZooEnv
import matplotlib.pyplot as plt

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import ray
from pettingzoo_env import LISPredatorPreyEnv
import gymnasium as gym
import numpy as np
from collections import OrderedDict
from gymnasium.spaces import Space

# env = parallel_to_aec(env)
# # env = PettingZooEnv(env)
# env = flatten_v0(env)
# api_test(env)



def flatten_data(data):
    flat_list = []
    for item in data:
        if isinstance(item, tuple):
            flat_list.extend(flatten_data(item))
        elif isinstance(item, np.ndarray):
            flat_list.extend(item.flatten().tolist())
        else:
            flat_list.append(item)
    return flat_list

def flatten_observation(observation):
    flat_obs = []
    for key, value in observation.items():
        if isinstance(value, (list, tuple)):
            flat_obs.extend(value)
        elif isinstance(value, dict):
            flat_obs.extend(flatten_observation(value))
        else:
            flat_obs.append(value)
    return np.array(flatten_data(flat_obs),dtype=np.float32)
# from gymnasium import spaces
# def flatten(env):
#     original_step = env.step
#     original_reset = env.reset
#     # original_observation_space = env.observation_space
#     original_observation_spaces = env.observation_spaces
#     def step(self,action):
#         obs, reward, done,trunctions, info = original_step(action)
#         flattened_obs = {agent_name: flatten_observation(agent_obs) for agent_name, agent_obs in obs.items()}
#         return flattened_obs, reward, done,trunctions, info

#     def reset(self, seed=None, options=None):
#         obs, info = original_reset(seed=seed, options=options)
#         flattened_obs = {agent_name: flatten_observation(agent_obs) for agent_name, agent_obs in obs.items()}

#         return flattened_obs,info
    
#     def observation_space(agent):
#         space = gym.spaces.Box(
#             low=-np.inf,     # 最小值为负无穷
#             high=np.inf,     # 最大值为正无穷
#             shape=(91,),      # 数组的形状
#             dtype=np.float32  # 数据类型
#         )
#         return space

#     env.step = step.__get__(env)
#     env.reset = reset.__get__(env)
#     # env.observation_space = observation_space.__get__(env)
#     env.observation_spaces = {agent_name: observation_space(agent_name) for agent_name, agent_obs in env.observation_spaces.items()}

#     # 创建新的 observation_space
#     def get_flattened_observation_space(agent):
#         sample_obs,info = env.reset()
#         flat_obs = flatten_observation(sample_obs)
#         return gym.spaces.Box(low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32)

#     def flatten_space(space: spaces.Space) -> spaces.Space:
#         """Flatten a given Gym space."""
#         if isinstance(space, spaces.Dict):
#             # Flatten each key-value pair in the dictionary
#             flattened_space = {}
#             for key, value in space.spaces.items():
#                 flattened_space[key] = flatten_space(value)
#             return spaces.Dict(flattened_space)

#         elif isinstance(space, spaces.Tuple):
#             # Flatten each space in the tuple
#             flattened_space = [flatten_space(s) for s in space.spaces]
#             return spaces.Tuple(flattened_space)

#         elif isinstance(space, spaces.Box):
#             # Flatten Box space
#             return spaces.Box(
#                 low=space.low.flatten(),
#                 high=space.high.flatten(),
#                 dtype=space.dtype
#             )

#         elif isinstance(space, spaces.Discrete):
#             # Discrete space remains the same
#             return space

#         else:
#             raise NotImplementedError(f"Unsupported space type: {type(space)}")

#     # env.observation_space = lambda agent: get_flattened_observation_space(agent)

#     return env
from gymnasium import spaces
import numpy as np

def flatten_observation(observation):
    """
    将输入的观测数据展平为一维数组。
    """
    # 处理不同类型的观测数据
    if isinstance(observation, dict):
        # 对字典中的每个值进行展平，并将它们拼接在一起
        return np.concatenate([flatten_observation(v) for v in observation.values()])
    elif isinstance(observation, tuple) or isinstance(observation, list):
        # 对元组或列表中的每个元素进行展平，并将它们拼接在一起
        return np.concatenate([flatten_observation(v) for v in observation])
    elif isinstance(observation, np.ndarray):
        # 如果是 NumPy 数组，直接展平
        return observation.flatten()
    else:
        # 如果是标量，将其转换为一维数组
        return np.array([observation],dtype=np.float32)

def flatten(env_class):
    class FlattenedEnv(env_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Flatten the observation spaces for each agent
            self.observation_spaces = {
                agent_name: self.flatten_space(agent_space)
                for agent_name, agent_space in self.observation_spaces.items()
            }

        def step(self, action):
            obs, reward, done, truncation, info = super().step(action)
            # Flatten the observations for all agents
            flattened_obs = {agent_name: flatten_observation(agent_obs) for agent_name, agent_obs in obs.items()}
            return flattened_obs, reward, done, truncation, info

        def reset(self, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            # Flatten the observations for all agents
            flattened_obs = {agent_name: flatten_observation(agent_obs) for agent_name, agent_obs in obs.items()}
            return flattened_obs, info

        def flatten_space(self, space):
            """Flatten the given Gym space."""
            if isinstance(space, spaces.Dict):
                # Flatten each key-value pair in the dictionary
                flattened_spaces = [self.flatten_space(v) for v in space.spaces.values()]
                total_dim = sum([flat_space.shape[0] for flat_space in flattened_spaces])
                return spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

            elif isinstance(space, spaces.Tuple):
                # Flatten each space in the tuple
                flattened_spaces = [self.flatten_space(s) for s in space.spaces]
                total_dim = sum([flat_space.shape[0] for flat_space in flattened_spaces])
                return spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

            elif isinstance(space, spaces.Box):
                # Flatten Box space
                return spaces.Box(
                    low=np.concatenate([space.low.flatten()]),
                    high=np.concatenate([space.high.flatten()]),
                    dtype=np.float32
                )

            elif isinstance(space, spaces.Discrete):
                # Discrete space represented as a single integer value
                return spaces.Box(low=0, high=space.n - 1, shape=(1,), dtype=np.float32)

            elif isinstance(space, spaces.MultiBinary):
                # MultiBinary space represented as binary values (0 or 1)
                return spaces.Box(low=0, high=1, shape=(space.n,), dtype=np.float32)

            elif isinstance(space, spaces.MultiDiscrete):
                # MultiDiscrete space represented as concatenated one-hot vectors
                total_dim = sum(space.nvec)
                return spaces.Box(low=0, high=1, shape=(total_dim,), dtype=np.float32)

            else:
                raise NotImplementedError(f"Unsupported space type: {type(space)}")


    return FlattenedEnv



prey_algorithms = ["PPO", "PPO", "PPO", "PPO", "DDPG", "DDPG", "DDPG"]
pred_algorithms = ["PPO", "PPO", "PPO", "DDPG", "DDPG", "DDPG"]

def ppo_predator_algorithm(observation_info, max_speed):## writing the function like this input observation_info and out put action action must fit
    a = np.random.choice([0, 1])

    angle = np.random.uniform(0, 2 * np.pi)  
    length = np.random.uniform(0, min(max_speed, 10.0))  
    x = length * np.cos(angle)
    y = length * np.sin(angle)

    action = np.array([x, y], dtype=np.float32)
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

    action = np.array([x, y], dtype=np.float32)
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

    action = np.array([x, y], dtype=np.float32)
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

    action = np.array([x, y], dtype=np.float32)
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

    action = np.array([x, y], dtype=np.float32)
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

    action = np.array([x, y], dtype=np.float32)
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


FlattenedEnvClass = flatten(LISPredatorPreyEnv)
# env_instance = FlattenedEnvClass()

env = FlattenedEnvClass(
    prey_algorithms=prey_algorithms,
    pred_algorithms=pred_algorithms,
    predator_algorithms_predict=predator_algorithms_predict,
    prey_algorithms_predict=prey_algorithms_predict,
)
gym_env = ss.pettingzoo_env_to_vec_env_v1(env)
gym_env = ss.concat_vec_envs_v1(gym_env, num_vec_envs=4, num_cpus=1, base_class='gymnasium')

# Apply supersuit wrappers
# env = ss.black_death_v3(env)
# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, num_vec_envs=1, base_class='gymnasium')
# env = ss.flatten_v0(env)

# Convert to rllib environment
# rllib_env = PettingZooEnv(env)

# Convert to rllib environment
# rllib_env = PettingZooEnv(env)
# import numpy as np

# obs = env.reset()
# for _ in range(100):
#     action = np.array([env.action_space.sample()])  # 采样一个随机动作
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()
# env.close()
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)
import os

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
# from pettingzoo.butterfly import pistonball_v6
from pettingzoo.butterfly import knights_archers_zombies_v10


from torch.distributions import Categorical, Normal
env.reset(seed=1)
class MLPModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)

        self.model = nn.Sequential(
            nn.Linear(obs_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Separate output layers for discrete and continuous actions
        self.discrete_policy_fn = nn.Linear(128, 2)  # For 'makeAChild'
        self.continuous_policy_fn = nn.Linear(128, 4)  # For 'moveVector' (mean and log_std for x and y)
        self.value_fn = nn.Linear(128, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"])
        self._value_out = self.value_fn(model_out)

        # Output for discrete action ('makeAChild')
        discrete_action_logits = self.discrete_policy_fn(model_out)

        # Output for continuous action ('moveVector')
        continuous_params = self.continuous_policy_fn(model_out)
        mean_x = continuous_params[:, 0]
        log_std_x = continuous_params[:, 1]
        mean_y = continuous_params[:, 2]
        log_std_y = continuous_params[:, 3]

        # Create continuous action distribution
        std_x = torch.exp(log_std_x)
        std_y = torch.exp(log_std_y)
        continuous_action_mean = torch.stack([mean_x, mean_y], dim=-1)
        continuous_action_std = torch.stack([std_x, std_y], dim=-1)
        # print({
        #     "makeAChild": discrete_action_logits,
        #     "moveVector": (continuous_action_mean, continuous_action_std)  # Return mean and std
        # })

        return {(continuous_action_mean, continuous_action_std)  # Return mean and std
        }, state

    def value_function(self):
        return self._value_out.flatten()

# from ray.tune.registry import register_env
# if __name__ == "__main__":
#     ray.init()

#     env_name = "LISPredatorPreyFlattenedEnv"

#     def env_creator(env_config):
#         return FlattenedEnvClass()
#     register_env("LISPredatorPreyFlattenedEnv", env_creator)
#     register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
#     ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)

#     config = (
#         PPOConfig()
#         .environment(env=env_name, clip_actions=True)
#         .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
#         .training(
#             train_batch_size=512,
#             lr=2e-5,
#             gamma=0.99,
#             lambda_=0.9,
#             use_gae=True,
#             clip_param=0.4,
#             grad_clip=None,
#             entropy_coeff=0.1,
#             vf_loss_coeff=0.25,
#             sgd_minibatch_size=64,
#             num_sgd_iter=10,
#         )
#         .debugging(log_level="ERROR")
#         .framework(framework="torch")
#         .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
#     )

#     tune.run(
#         "PPO",
#         name="PPO",
#         stop={"timesteps_total": 50000 if not os.environ.get("CI") else 500},

#         checkpoint_freq=1,
#         storage_path="~/ray_results/" + env_name,
#         config=config.to_dict(),
#     )
import os
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv

# Make sure to use the same environment creation function
def env_creator(env_config):
    return FlattenedEnvClass()

import os
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env import ParallelPettingZooEnv

# Make sure to use the same environment creation function
def env_creator(env_config):
    return FlattenedEnvClass()

if __name__ == "__main__":
    ray.init()

    # Register the environment
    env_name = "LISPredatorPreyFlattenedEnv"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    # Path to the saved checkpoint (replace with your actual checkpoint path)
    checkpoint_path = os.path.expanduser("/home/qrbao/ray_results/LISPredatorPreyFlattenedEnv/PPO/PPO_LISPredatorPreyFlattenedEnv_cce1a_00000_0_2024-10-03_10-42-50/checkpoint_000097")

    # Create a PPO configuration
    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
    )

    # Initialize the PPO algorithm with the configuration
    ppo_agent = PPO(config=config)
    # Restore from checkpoint
    ppo_agent.restore(checkpoint_path)

    # Run evaluation with the restored model
    env = env_creator({})
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
    done = False
    total_reward = 0
    plt.figure(figsize=(10, 8))
    plt.ion()
    game_continue = False
    iteration = 0

    while not game_continue:
        env.render()
        actions = {}
        if iteration % 100 == 1:
            update_and_plot(iteration, env, data_storage)
            print(len(env.simulator.predators),end="\t")
            for agent in env.simulator.predators + env.simulator.preys:
                print(f"{agent.name} health is {agent.health}-||||||",end="\t")
            print(len(env.simulator.preys))
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = ppo_agent.compute_single_action(agent_obs)
        new_state, rewards, done, truncated, infos = env.step(actions)
        iteration += 1
        # print(iteration)
        if iteration % 100 == 1:
            pass

    plt.ioff()
    plt.show()
    
