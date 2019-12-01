import random
import numpy as np
import torch
from collections import deque


class SimpleBuffer:

  def __init__(self, max_size, device):
    self.device = device
    self.max_size = max_size
    self.buffer = deque(maxlen=max_size)

  def push(self, state, action, reward, next_state, done):
    experience = (state, action, np.array(reward), next_state, done)
    self.buffer.append(experience)

  def sample(self, batch_size):
    batch = np.asarray(random.sample(self.buffer, batch_size))
    states = np.vstack(batch[:, 0])
    actions = np.vstack(batch[:, 1])
    rewards = np.vstack(batch[:, 2])
    next_states = np.vstack(batch[:, 3])
    dones = np.vstack(batch[:, 4])
    states = torch.from_numpy(states).float().to(device=self.device)
    actions = torch.from_numpy(actions).long().to(device=self.device)
    rewards = torch.from_numpy(rewards).float().to(device=self.device)
    next_states = torch.from_numpy(next_states).float().to(device=self.device)
    dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device=self.device)
    return states, actions, rewards, next_states, dones

  def sample_sequence(self, batch_size):
    raise NotImplementedError("Noch nicht fertig!")
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch = []

    min_start = len(self.buffer) - batch_size
    start = np.random.randint(0, min_start)

    for sample in range(start, start + batch_size):
      state, action, reward, next_state, done = self.buffer[start]
      state_batch.append(state)
      action_batch.append(action)
      reward_batch.append(reward)
      next_state_batch.append(next_state)
      done_batch.append(done)

    state_batch = torch.FloatTensor(state_batch).to(device=self.device)
    action_batch = torch.FloatTensor(action_batch).to(device=self.device)
    reward_batch = torch.FloatTensor(reward_batch).to(device=self.device)
    next_state_batch = torch.FloatTensor(next_state_batch).to(device=self.device)
    done_batch = torch.FloatTensor(done_batch).to(device=self.device)

    return (
    state_batch, action_batch, reward_batch, next_state_batch, done_batch)

  def __len__(self):
    return len(self.buffer)
