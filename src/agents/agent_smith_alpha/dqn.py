import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agents.agent_smith_alpha.qnet import QNet
from src.agents.agent_smith_alpha.buffer_simple import SimpleBuffer
from src.agents.agent_smith_alpha.plotting import plot_training_progress

class DQNAgent():
  """Normal and Clipped Double Deep Q-Learning Agent."""

  def __init__(self,
               state_dim,
               action_dim,
               buffer_size,
               batch_size,
               gamma,
               tau,
               lr,
               training_interval,
               epsilon,
               epsilon_decay,
               epsilon_min,
               device):
    """
    Initialize the Deep Q Learning agent.

    :param state_dim: Int or tuple(Ints)
      State dimension:  Int for dense network
                        Tuple for conv network
    :param action_size: Int
      Number of actions.
    :param buffer_size: Int
      Size of the replay buffer.
    :param batch_size: Int
      Number of samples to use for computing the loss at a time.
    :param gamma: Float
      Discount factor between 0 and 1.
    :param tau: Float
      Value between 0 and 1 used for updating the target network.
      Only used in the case of ordinary Double Deep Q Learning
    :param lr: Float
      The learning rate used for both Q networks.
    :param training_interval: Int
      Defining the interval on how often to update the network.
    :param epsilon: Float
      Start value for epsilon greedy. Between 0 and 1.
    :param epsilon_decay: Float
      Rate at which epsilon will decay during training. Between 0 and 1.
    :param epsilon_min: Float
      Min value epsilon can reach. Between 0 and 1.
    :param layer_param: Dict
      Desciption of the Q-Net architecture
    :param device: String
      Set 'cpu' or 'cuda' for either using the cpu or gpu the neural
      network calculations repectively.
    """
    self.device=device

    self.action_dim = action_dim
    self.batch_size = batch_size
    self.gamma = gamma
    self.training_interval = training_interval
    self.tau = tau
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.memory = SimpleBuffer(max_size=buffer_size, device=device)
    self.global_training_step = 0
    self.qnet1 = QNet(state_dim, action_dim).to(device)
    self.qnet2 = QNet(state_dim, action_dim).to(device)
    self.optimizer1 = optim.Adam(self.qnet1.parameters(), lr=lr)
    self.optimizer2 = optim.Adam(self.qnet2.parameters(), lr=lr)
    self.training_summary = []

  def __call__(self, obs):
    return self.plan(obs)

  def plan(self, obs):
    """
    Epsilon-greedy action selection.

    :param obs: nd.array
      The observation of the state.

    :return action: nd.array
      The action which will be executed next.
    """
    if random.random() > self.epsilon:
      obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
      self.qnet1.eval()
      with torch.no_grad():
        action_values = self.qnet1(obs)
      self.qnet1.train()
      action = np.argmax(action_values.cpu().data.numpy())
    else:
      action = random.choice(np.arange(self.action_dim))
    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon, self.epsilon_min)
    return action

  def step(self, state, action, reward, next_state, done):
    self.memory.push(state, action, reward, next_state, done)

    if self.global_training_step % self.training_interval == 0:
      if len(self.memory) > self.batch_size:
        batch = self.memory.sample(self.batch_size)
        self.optimize_regular(batch)
    self.global_training_step += 1


  def optimize_regular(self, batch):
    """Optimize the Q networks corresponding to Double Q-Learning."""
    loss = self._compute_regular_loss(batch)
    self.optimizer1.zero_grad()
    loss.backward()
    self.optimizer1.step()
    self._update_target_network(self.qnet1, self.qnet2)
    self.log_progress(loss.detach().numpy())

  def _compute_regular_loss(self, batch):
    """
    Compute the loss given a batch of (s,a,s',r,t).

    Regular loss for Double Deep Q Learning where the next_a is computed
    using the target network.

    :param batch: Tuple(torch.FloatTensor,
                        torch.LongTensor,
                        torch.FloatTensor,
                        torch.FloatTensor,
                        torch.FloatTensor)
      Batch of (state, action , next_state, reward, terminal)-tuples.

    :return loss1:
      The MSE loss of Q-Net1.
    """
    states, actions, rewards, next_states, dones = batch
    q_targets_next = self.qnet2(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
    q_current = self.qnet1(states).gather(1, actions)
    loss = F.mse_loss(q_current, q_targets)
    return loss

  def optimize_clipped(self, batch):
    """Optimize the Q networks corresponding to Clipped Double Q-Learning."""
    #batch = self.replay_buffer.sample(batch_size)
    loss1, loss2 = self._compute_clipped_loss(batch)
    self.optimizer1.zero_grad()
    loss1.backward()
    self.optimizer1.step()
    self.optimizer2.zero_grad()
    loss2.backward()
    self.optimizer2.step()

  def _compute_clipped_loss(self, batch):
    """
    Compute the loss given a batch of (s,a,s',r,t).

    Calculating the loss for Clipped Double Q-Learning from:
    "Addressing Function Approximation Error in Actor-Critic Methods", Fujimoto et al. (2018)

    :param batch: Tuple(torch.FloatTensor,
                        torch.LongTensor,
                        torch.FloatTensor,
                        torch.FloatTensor,
                        torch.FloatTensor)
      Batch of (state, action , next_state, reward, terminal)-tuples.

    :return loss1, loss2:
      The MSE loss of Q-Net1 and Q-Net2.
    """
    states, actions, rewards, next_states, dones = batch
    # Target Q
    q1_targets_next = self.qnet1.forward(next_states).detach()
    q2_targets_next = self.qnet2.forward(next_states).detach()
    q_targets_next = torch.min(torch.max(q1_targets_next, 1)[0],
                       torch.max(q2_targets_next, 1)[0])
    #q_targets_next = q_targets_next.view(q_targets_next.size(0), 1)
    target_q = rewards + (self.gamma * q_targets_next * (1 - dones))
    # Current Q
    q1_current = self.qnet1(states).gather(1, actions)
    q2_current = self.qnet2(states).gather(1, actions)
    # Loss
    loss1 = F.mse_loss(q1_current, target_q.detach())
    loss2 = F.mse_loss(q2_current, target_q.detach())
    return loss1, loss2

  def _update_target_network(self, local_model, target_model):
    """Update target network: θ_target = τ*θ_local + (1 - τ)*θ_target."""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

  def log_progress(self, loss_npy):
    self.training_summary.append(loss_npy)
    if self.global_training_step % 100 == 0:
      np.save("dqn.npy", self.training_summary)
      plot_training_progress(self.training_summary, save_dir="dqn.png")
