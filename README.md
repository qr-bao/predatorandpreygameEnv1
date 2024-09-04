
# predatorandpreygameEnv1

## Running the Game

We can run the game using the `main.py` file, which using a mathematically designed movement strategy. This file does not set up an environment.

We can also run the game using the `gym_env_refactor.py` and `PPO_gym_env_refactor.py` file, which sets up an environment and uses a random strategy.

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
# project detail
## env abstract

## env input and output
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