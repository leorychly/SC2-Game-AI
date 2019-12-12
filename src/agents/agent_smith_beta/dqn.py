import numpy as np
import random
from pathlib2 import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agents.agent_smith_beta.qnet_conv import ConvQNet
from src.agents.agent_smith_beta.buffer_simple import SimpleBuffer
from src.agents.agent_smith_alpha.buffer_prioritized import PrioritizedBuffer
from src.agents.agent_smith_alpha.plotting import plot_training_progress

# TODO: log_progress() add avrg reward score as third row in the plot


class DQNAgent:
  """Normal and Clipped Double Deep Q-Learning Agent."""

  def __init__(self,
               state_dim,
               action_dim,
               buffer_size=int(1e6),
               batch_size=128,
               gamma=0.99,
               tau=1e-3,
               lr=8e-5,
               training_interval=1,
               epsilon=0.9999,
               epsilon_decay=0.9999,
               epsilon_min=0.05,
               device="cpu"):
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
    self.device = device
    self.save_path = Path("./results")
    self.train_process_data_fname = Path("dqn_training.npy")
    self.train_process_plot_fname = Path("dqn_training.png")
    self.save_path.mkdir(parents=True, exist_ok=True)

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
    self.qnet1 = ConvQNet(state_dim, action_dim).to(device)
    self.qnet2 = ConvQNet(state_dim, action_dim).to(device)
    self.optimizer1 = optim.Adam(self.qnet1.parameters(), lr=lr)
    self.optimizer2 = optim.Adam(self.qnet2.parameters(), lr=lr)
    self.training_summary = []
    self.epsilon_history = []
    self.total_reward = 0
    self.reward_history = []

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
    state_pix, state_sem = state
    next_state_pix, next_state_sem = next_state
    self.memory.push(state_pix, state_sem,
                     action,
                     reward,
                     next_state_pix, next_state_sem,
                     done)
    self.total_reward += reward

    if self.global_training_step % self.training_interval == 0:
      if len(self.memory) > self.batch_size:
        batch = self.memory.sample(self.batch_size)
        self.optimize_regular(batch)
        self.reward_history.append(self.total_reward)
        self.total_reward = 0
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
    # TODO: 
    states, actions, rewards, next_states, dones = batch
    q_targets_next = self.qnet2(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
    q_current = self.qnet1(states).gather(1, actions)
    loss = F.mse_loss(q_current, q_targets)
    return loss

  def _update_target_network(self, local_model, target_model):
    """Update target network: θ_target = τ*θ_local + (1 - τ)*θ_target."""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

  def log_progress(self, loss_npy):
    self.epsilon_history.append(self.epsilon)
    self.training_summary.append(loss_npy)
    if self.global_training_step % 100 == 0 and self.global_training_step > 0:
      np.save(str((self.save_path / self.train_process_data_fname).absolute()),
              self.training_summary)
      plot_training_progress(self.reward_history,
                             self.training_summary,
                             self.epsilon_history,
                             save_dir=str((self.save_path / self.train_process_plot_fname).absolute()))

  def save(self, file_path):
    file_path.mkdir(parents=True, exist_ok=True)
    torch.save(self.qnet1.state_dict(),
               (file_path / "qnet1").absolute().as_posix())
    torch.save(self.optimizer1.state_dict(),
               (file_path / "qnet1_optimizer").absolute().as_posix())
    torch.save(self.qnet2.state_dict(),
               (file_path / "qnet2").absolute().as_posix())
    torch.save(self.optimizer2.state_dict(),
               (file_path / "qnet2_optimizer").absolute().as_posix())

  def load(self, file_path):
    file_path.mkdir(parents=True, exist_ok=True)
    self.qnet1.load_state_dict(torch.load(
      (file_path / "qnet1").absolute().as_posix()))
    self.optimizer1.load_state_dict(torch.load(
      (file_path / "qnet1_optimizer").absolute().as_posix()))
    self.qnet2.load_state_dict(torch.load(
      (file_path / "qnet2").absolute().as_posix()))
    self.optimizer2.load_state_dict(torch.load(
      (file_path / "qnet2_optimizer").absolute().as_posix()))
