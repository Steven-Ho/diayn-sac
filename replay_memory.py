import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, context, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (context, state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        context, state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return context, state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class Buffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.pointer = 0

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, context = map(np.stack, zip(*batch))
        return context, state       

    def push(self, state, context):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pointer] = (state, context)
        self.pointer = (self.pointer + 1) % self.capacity

    def retrieve(self):
        batch = np.asarray(self.buffer)
        state, context = map(np.stack, zip(*batch))
        return context, state