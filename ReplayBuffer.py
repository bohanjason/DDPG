import random
from collections import deque


class ReplayBuffer:
    def __init__(self, memory_size=50000):
        # The memory essentially stores transitions recorder from the agent taking actions in the environment.
        self.memory = deque()
        self.memory_size = memory_size

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
