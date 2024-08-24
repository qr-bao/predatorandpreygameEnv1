import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from gym import spaces
from collections import deque
import random

class SimplePredatorPreyEnv(gym.Env):
    def __init__(self):
        super(SimplePredatorPreyEnv, self).__init__()
        
        # 假设有 5 个智能体，每个智能体的观察值是一个 3 维向量
        self.num_entities = 5
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.num_entities,25, 3), dtype=np.float32)
        
        # 每个智能体的动作空间也是一个 2 维向量
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_entities, 2), dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = 100

    def reset(self):
        # 重置环境并返回初始观察值
        self.current_step = 0
        initial_observation = np.random.uniform(low=-10, high=10, size=(self.num_entities, 25,3)).astype(np.float32)
        return initial_observation, {}

    def step(self, action):
        self.current_step += 1
        
        # 模拟一个简单的环境响应
        next_state = np.random.uniform(low=-10, high=10, size=(self.num_entities,25, 3)).astype(np.float32)
        reward = np.random.uniform(low=-1, high=1, size=(self.num_entities,))
        done = self.current_step >= self.max_steps
        truncated = self.current_step >= self.max_steps
        
        # 返回观察值、奖励、是否完成、截断标志和额外信息
        return next_state, reward, done, truncated
    def render(self, mode='human'):
        pass

    def close(self):
        pass


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = torch.tensor(max_action, dtype=torch.float32).to(device)

    def forward(self, state):
        # 将输入展平为一维向量
        state = state.view(state.size(0), -1)  # 等同于 flatten(start_dim=1)
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action.view(1, -1) * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer()

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state).reshape(1, -1).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device).reshape(-1, 1)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device).reshape(-1, 1)

        # Critic loss
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions)
        target_Q = rewards + (1 - dones) * self.gamma * target_Q
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward, next_state, done)


# 注册新环境
gym.envs.registration.register(
    id='SimplePredatorPreyEnv-v0',
    entry_point='__main__:SimplePredatorPreyEnv',
)

# 训练循环
if __name__ == "__main__":
    device = torch.device("cpu")
    iteration = 0
    # 创建简易环境
    env = gym.make('SimplePredatorPreyEnv-v0')
    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)
    max_action = env.action_space.high

    # 初始化 DDPG 算法
    ddpg = DDPG(state_dim, action_dim, max_action)

    num_episodes = 10
    batch_size = 4

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            iteration += 1
            action = ddpg.select_action(state)
            # step_info = env.step(action)

            next_state, reward, done, truncated = env.step(action)
            ddpg.store_transition(state, action, reward, next_state, done)
            ddpg.train(batch_size)

            state = next_state
            episode_reward += np.sum(reward)

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()

