import random
import numpy as np
import torch
from collections import deque


class SimpleBuffer:

  def __init__(self, max_size, device):
    self.device = device
    self.max_size = max_size
    self.buffer = deque(maxlen=max_size)

  def push(self,
           state_pix,
           state_sem,
           action,
           reward,
           next_state_pix,
           next_state_sem,
           done):
    experience = (state_pix, state_sem,
                  action,
                  np.array(reward),
                  next_state_pix, next_state_sem,
                  done)
    self.buffer.append(experience)

  def sample(self, batch_size):
    batch = np.asarray(random.sample(self.buffer, batch_size))
    state_pix = np.stack(batch[:, 0])
    state_sem = np.vstack(batch[:, 1])
    actions = np.vstack(batch[:, 2])
    rewards = np.vstack(batch[:, 3])
    next_state_pix = np.stack(batch[:, 4])
    next_state_sem = np.vstack(batch[:, 5])
    dones = np.vstack(batch[:, 6])
    state_pix = torch.from_numpy(state_pix).float().to(device=self.device)
    state_sem = torch.from_numpy(state_sem).float().to(device=self.device)
    actions = torch.from_numpy(actions).float().to(device=self.device)
    rewards = torch.from_numpy(rewards).float().to(device=self.device)
    next_state_pix = torch.from_numpy(next_state_pix).float().to(device=self.device)
    next_state_sem = torch.from_numpy(next_state_sem).float().to(device=self.device)
    dones = torch.from_numpy(dones.astype(np.uint8)).float().to(
      device=self.device)
    states = (state_pix, state_sem)
    next_states = (next_state_pix, next_state_sem)
    return states, actions, next_states, rewards, dones

  def __len__(self):
    return len(self.buffer)
