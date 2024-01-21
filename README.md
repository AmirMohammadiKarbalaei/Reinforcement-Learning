# Reinforcement-Learning OpenAI Gym


## CONCEPT
The primary goal of this project is to leverage Q-Learning, a reinforcement learning algorithm, to train an agent capable of playing games within various OpenAI Gym environments. OpenAI Gym provides a standardized platform for developing and comparing reinforcement learning algorithms.

## Training Pipeline:
### Random Action: 
In the early stages of training, the agent takes random actions to explore the environment and populate its Q-table.
### Calculate Reward and Expected Reward:  
The agent interacts with the environment by taking actions, receiving rewards, and updating its Q-table. It calculates the reward for each action and estimates the expected reward for all possible actions in the current state.
repeat until session time is over
### Repeat Until Session Time is Over:
The exploration-exploitation loop continues until the predefined session time is over. The agent refines its policy over multiple episodes, gradually shifting from random actions to exploiting the learned Q-values.

## Q-Learning Algorithm:
### Initialization: 
The agent initializes its Q-table, which represents the state-action space, with arbitrary values. This Q-table is gradually updated through the learning process.
### Exploration-Exploitation Strategy: 
Q-Learning employs an exploration-exploitation strategy to balance the agent's exploration of the game environment and exploitation of learned knowledge. The agent randomly selects actions during the initial phase to explore the environment.
### Q-Table Update: 
After each action, the Q-table is updated based on the observed reward and the difference between the expected and actual Q-values. The learning rate influences the magnitude of these updates.
