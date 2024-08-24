import gym

# 注册新环境
gym.envs.registration.register(
    id='SimplePredatorPreyEnv-v0',
    entry_point='__main__:SimplePredatorPreyEnv',
)

# 训练循环
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建简易环境
    env = gym.make('SimplePredatorPreyEnv-v0')
    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)
    max_action = env.action_space.high[0]

    # 初始化 DDPG 算法
    ddpg = DDPG(state_dim, action_dim, max_action)

    num_episodes = 10
    batch_size = 64

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = ddpg.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            ddpg.store_transition(state, action, reward, next_state, done)
            ddpg.train(batch_size)

            state = next_state
            episode_reward += np.sum(reward)

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()
