import gym
from gym import spaces
import numpy as np
import constants
from simulator import Simulator
class MyCustomEnv(gym.Env):
    def __init__(self,MAX=constants.max_observation_count+constants.max_hearing_count,):
        super(MyCustomEnv, self).__init__()
        self.sim = Simulator(constants.SCREEN_WIDTH1, constants.SCREEN_HEIGHT1)
        self.iteration_count = 0 
        self.prey_counts = []
        self.predator_counts = []
        self.prey_born_count = 0
        self.predator_born_count = 0
        # 定义动作空间和观察空间
        self.action_space = spaces.Box(low=-constants.PREY_MAX_SPEED, high=constants.PREY_MAX_SPEED, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=MAX, shape=(10, 3), dtype=np.int32)

        # 初始化环境的状态
        self.state = np.zeros((MAX, 3), dtype=np.int32)

    def reset(self):
        self.sim.initialize()

        # 重置环境到初始状态并返回初始观测值
        self.state = np.zeros(3)  # 例如：初始状态为全零
        return self.state

    def step(self, action):
        # 执行动作并返回新的观测值、奖励、是否终止和其他信息
        self.state = self.state + action  # 更新状态的逻辑
        reward = 0  # 根据您的逻辑计算奖励
        done = False  # 判断是否结束
        return self.state, reward, done, {}

    def render(self, mode='human'):
        # 渲染环境 (可选)
        print(f"State: {self.state}")

    def close(self):
        # 关闭环境 (可选)
        pass