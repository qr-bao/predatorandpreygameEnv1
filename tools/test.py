import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(2)  # 0表示向左，1表示向右
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)

        # 初始化小车位置
        self.position = 0

    def reset(self):
        # 重置环境，将小车放置在起始位置
        self.position = 0
        return np.array([self.position])

    def step(self, action):
        # 执行动作，更新小车位置并返回奖励和观察结果
        if action == 0:
            self.position -= 1
        else:
            self.position += 1

        # 计算奖励
        reward = 1 if action == 1 else -1

        # 规定位置范围在 [0, 100] 之间
        self.position = np.clip(self.position, 0, 100)

        # 返回观察结果、奖励、是否终止和其他信息
        return np.array([self.position]), reward, False, {
   }

# 创建环境实例
env = CustomEnv()

# 测试环境
for episode in range(5):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()  # 随机选择动作
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    