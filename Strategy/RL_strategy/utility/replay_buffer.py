import collections
import random
from torch import FloatTensor

class ReplayBuffer():

    def __init__(self, max_size, num_steps=4, obs_model="MLP"):
        self.buffer = collections.deque(maxlen = max_size)
        self.num_steps = num_steps
        self.obs_model = obs_model

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)

        if self.obs_model == "GCN" or self.obs_model == "Transformer" or self.obs_model == "T-GCN" or self.obs_model == "GRNN":
            obs_batch = obs_batch
            next_obs_batch = next_obs_batch
        elif self.obs_model == "MLP":
            obs_batch = FloatTensor(obs_batch)
            next_obs_batch = FloatTensor(next_obs_batch)

        action_batch = FloatTensor(action_batch)
        reward_batch = FloatTensor(reward_batch)
        done_batch = FloatTensor(done_batch)
            
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    
    def __len__(self):
        return len(self.buffer)