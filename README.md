
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

- Initial Parameters:
    - Number of predators and prey, and a list of algorithms for each agent. For example: num_pred = 2, num_prey = 3, and [ppo, ddpg, ddpg, ppo, random].
    - A dictionary of algorithm functions, such as {'ppo': function_ppo, 'ddpg': function_ddpg, 'random': function_random}.

- Agent Algorithm Assignment:
    - Agents can be assigned to different algorithms for execution. Currently, all initial agents are trained using RL algorithms, and their springs actions are determined by their algorithms type and a dictionary of algorithm functions set up during the environment initialization. However, we can easily configure the system so that some initial agents are trained while other initial agent operate using different algorithms.

- Observation Space:
    - The observation space for each agent includes the following: [agent_type, relative_x, relative_y, agent.algorithm].
    - The full observation space is composed of up to 5 entries from each of the following: observed predators, observed preys, observed foods, observed obstacles, and hearing.

- Action Space:
    - The action space is structured as (self.num_agents, 3), representing [position_x, position_y, reproduction_intention].
    - The reproduction_intention value ranges from 0 to 1.

- Termination Criteria:
    - The episode ends (done) when all agents have died or when a specific step limit is reached (truncated).

- Reward System:
    - The current reward system is simple, but it can be customized by overriding the reward function in the `Env()` class.