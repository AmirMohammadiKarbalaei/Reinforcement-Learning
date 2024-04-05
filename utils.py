from collections import deque
import numpy as np
import random

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import relu, linear


class DQN:

    """ Iplementing a  deep Q learning algorithm """

    def __init__(self,action_space,observation_space,episode,env):
        """Hyper Parameters"""
        self.action_space = action_space
        self.observation_space = observation_space
        self.epsilon = 1.0
        self.epsilon_max = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        self.gamma = 0.99      
        self.memory = deque(maxlen=1000000)
        self.batch_size = 64
        self.learning_rate = 0.001
        self.episode = episode
        self.env = env
        self.NN_model = self.build_NNmodel()
        self.env = env

    def build_NNmodel(self):

        NN_model = Sequential()
        NN_model.add(Dense(124, input_dim=self.observation_space, activation=relu))
        NN_model.add(Dense(124, activation=relu))
        NN_model.add(Dense(self.action_space, activation=linear))
        NN_model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return NN_model

    def save_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def Take_action(self, state):
        """Allowing the agent to take random actions  as well as 
        actions from NN to allow exploration as well as exploitation"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.NN_model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        """Updating Q values"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([data[0] for data in minibatch])
        actions = np.array([data[1] for data in minibatch])
        rewards = np.array([data[2] for data in minibatch])
        next_states = np.array([data[3] for data in minibatch])
        dones = np.array([data[4] for data in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        Q = self.NN_model.predict_on_batch(states) #  Predicted Q-values
        Q_sa = self.NN_model.predict_on_batch(next_states)
        targets = rewards + self.gamma*(np.amax(Q_sa, axis=1))*(1-dones)
        index = np.array([i for i in range(self.batch_size)]) 
        Q[[index], [actions]] = targets 
    

        self.NN_model.fit(states, Q, epochs=1, verbose=0) 
        
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min)* np.exp(-self.epsilon_decay*self.episode)
                


def train_dqn(episode,max_steps,env):

    loss = []
    DQN_agent = DQN(env.action_space.n, np.prod(env.observation_space.shape) ,episode,env)
    for e in range(episode):
        state = env.reset()
        ###reshaping state to fit the input of layer of NN
        state = np.reshape(state[0], [1, DQN_agent.observation_space]) 
        
        
        
        
        score = 0
        for i in range(max_steps):
            action = DQN_agent.Take_action(state)
            env.render()
            
            next_state, reward, done, truncated,info= env.step(action)
            score += reward
            ###reshaping next_state to fit the input of layer of NN
            next_state = np.reshape(next_state, [1, DQN_agent.observation_space])

            DQN_agent.save_to_memory(state, action, reward, next_state, done)
            state = next_state
            DQN_agent.replay()
            if done or truncated:
                print(F"episode: {e}/{episode}, score: {score}")
                break
        loss.append(score)

        # Average score of last 50 episode
        Average_score = np.mean(loss[-50:])
        if Average_score > 200:
            print('\n Task Completed! \n')
            break
        print(f"Average over last 50 episode: {Average_score} \n")
    return loss

