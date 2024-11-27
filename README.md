
# predatorandpreygameEnv1

## Running the Game

Currently, it is recommended to run the pettingzoo_env_V2.py file

---

## Tasks need to do future
- using one RL algorithm to play game(finished)
- The obstacle collision detection part currently has issues, and the edge collision detection function needs to be rewritten.(finished)
- A new reproduction model has been adopted, so the game parameters need to be adjusted.(finished)
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
