import random
from collections import deque
import numpy as np
import copy

class ReplayBuffer:
    def __init__(self, memory_size=50000):
        # The memory essentially stores transitions recorder from the agent taking actions in the environment.
        self.memory = deque()
        self.memory_size = memory_size
        self.burn_in_memory = [([0.9999812841415405, 0.29494113801132704, 0.9031153336211255, 1.0039291635683614e-30], 24.15746906968441),([0.8039205530027025, 0.6477571056480582, 0.4443356275953143, 1.277050190098549e-22], 36.64912525807603), ([0.84602713595538, 0.4351510319445798, 0.6792249226224799, 3.7388543492557677e-22],36.03595212592558 ), ([0.47509393095970154, 0.47235093907018477, 0.32160722655553686, 0.2272885529391785], 27.831850851456863)]


    # transition = [current_state, action, reward, nextstate, is_terminal]
    def burn_in(self):
        for config, reward in self.burn_in_memory:
            init = np.zeros(5)
            state0 = np.zeros(5)
            state0[0] = config[0]
            state0[4] = 1
            trans0 = [init, config[0], reward, state0, False]

            state1 = copy.copy(state0)
            state1[1] = config[1]
            state1[4] = 2
            trans1 = [state0, config[1], reward, state1, False]

            state2 = copy.copy(state1)
            state2[2] = config[2]
            state2[4] = 3
            trans2 = [state1, config[2], reward, state2, False]

            state3 = copy.copy(state2)
            state3[3] = config[3]
            state3[4] = 4
            trans3 = [state2, config[3], reward, state3, True]
 
            self.append(trans0)
            self.append(trans1)
            self.append(trans2)
            self.append(trans3)


    def sort_batch(self, batch_size=32):
        
        current_states = []
        next_states = []
        rewards = []
        actions = []
        for current_state, action, reward, next_state, is_terminal in self.memory:
            actions.append(action)
            rewards.append(reward)
            current_states.append(current_state)
            if is_terminal:
                next_states.append(current_state)
            else:
                next_states.append(next_state)
        sort_rewards = np.argsort(rewards)[::-1]
        if len(self.memory) < batch_size:
            sort_index = sort_rewards[:len(self.memory)]
        else:
            sort_size = int (batch_size / 2)
            random_size = batch_size - sort_size
            sort_index = sort_rewards[:sort_size]
            random_index = random.sample(list(sort_rewards[sort_size:]), random_size)
            sort_index = np.concatenate((sort_index,random_index))
        return np.array(self.memory)[sort_index], np.array(current_states)[sort_index], np.array(next_states)[sort_index], np.array(rewards)[sort_index], np.array(actions)[sort_index]


    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions -
        #   i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        current_states = []
        next_states = []
        rewards = []
        actions = []

        if len(self.memory) < batch_size:
            samples = random.sample(self.memory, len(self.memory))
        else:
            samples = random.sample(self.memory, batch_size)
        
        for current_state, action, reward, next_state, is_terminal in samples:
            actions.append(action)
            rewards.append(reward)
            current_states.append(current_state)
            if is_terminal:
                next_states.append(current_state)
            else:
                next_states.append(next_state)
        return samples, current_states, next_states, rewards, actions


    def append(self, transition):
        # Appends transition to the memory. 
        self.memory.append(transition)
        if len(self.memory) > self.memory_size:
            self.memory.popleft()
