
# predatorandpreygameEnv1

## Running the Game

We can run the game using the `main.py` file, which using a mathematically designed movement strategy. This file does not set up an environment.

We can also run the game using the `gym_env.py` file, which sets up an environment and uses a random strategy.

---

## Tasks need to do future
- using one RL algorithm to play game
- Refactoring and organizing the code.
- Structuring the architecture and using appropriate variables.
- The obstacle collision detection part currently has issues, and the edge collision detection function needs to be rewritten.(finished)
- A new reproduction model has been adopted, so the game parameters need to be adjusted.
- There are still many small bugs that need to be fixed.
- 

---

## Current Multi-Agent Reinforcement Learning Organization

![Image Description](tools/README.png)
# 项目细节
## 环境概述

## 环境的输入和输出
### exec mode
    Preliminary preparation when env initial 
        length of predator,length of prey, a list of agent algorithm.{like:num_pred = 2,num_prey = 3. [ppo,ddpg,ddpg,ppo,random] }
        a dict of algorithm function ,like {'ppo':function_ppo,'ddpg':function_ddpg,'random':function_random}
    env output
        energy graph 
### training mode
    same Preliminary preparation when env initial 
        length of predator,length of prey, a list of agent algorithm.{like:num_pred = 2,num_prey = 3. [ppo,ddpg,ddpg,ppo,random] }
        a dict of algorithm function ,like {'ppo':function_ppo,'ddpg':function_ddpg,'random':function_random}
        for agent be used to training the algorithm(now is all agent if we want to part of agent to training and other agent using different algorithm we can esay change)
            the agent who can be using to train now only the initial agent now.initial agent get action from RL algorithm
            observation space  [agent_type, relative_x, relative_y,agent.algorithm]
            obsercation spce  = (observed_predator[:5]) + (observed_prey[:5]) + (observed_food[:5]) + (observed_obstacle[:5])+(hearning[:5])
            action space = (self.num_entities, 3)[position_x,position_y,reproduction_intention](reproduction_intention value range:from 0 to 1)
            done when all agent die or some step 
            truncated when reached a certain number of steps
            info 
            reward now is simple ,if we want to change . override the reward of function of Env()
            

            
        算法库
        输入要训练的agent是哪些（如果这些agent都是初始的还好说，如果游戏过程中的agent也要参与的话，那么怎么等长？如果这样子的话，怎么确定游戏done的结果）
        输入actions
    环境的输出
        obs = reset（）：输出要训练的算法的观察信息
        step ：输出要训练的算法的信息


    模型的输入输出
        输入obsercation 
        输出actions

    能量
        目前的能量是这个样子
        能量的输入：食物的产生 能量的增长非常简单，每产生一个食物，能量增长一个定值
        能量的消耗；agent移动带来的和随着时间的消耗 （按照这样的逻辑，agent数量越多消耗越大）捕食能量转移过程中的消耗
        目前有几个不确定的东西，一个是我们构建的这个能量系统的总体目标是什么，为什么我们要这样设计能量。捕食过程中是否需要存在能量的消耗（predator eat prey ，predator += 1/3 predator。predator=0）
        