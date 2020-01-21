import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib2 import Path
from absl import logging
import torch
import torch.nn.functional as F#
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.agents.agent_smith_gamma import dqn_model, prioritized_buffer

#https://github.com/Kaixhin/Rainbow/


class RainbowAgent:

  def __init__(
    self,
    state_dim,
    action_dim,
    device,
    #lr=1e-3,
    batch_size=128,
    buffer_size=int(1e6),
    discount=0.99,
    training_interval=4,
    n_steps_before_training=int(20e3),    # 20e3 Number of steps before starting training
    target_update=int(8e3),  # Number of steps after which to update target network
    adam_eps=1.5e-4,  # Adam epsilon
    atoms=51,  # Discretised size of value distribution
    history_length=3,  # Number of consecutive states processed
    noisy_std=0.1,  # Initial standard deviation of noisy linear layers
    multi_step=3,  # Number of steps for multi-step return
    priority_weight=0.4,  # Initial prioritised experience replay importance sampling weight
    priority_exponent=0.5,  # Prioritised experience replay exponent
    V_min=-10.,  # Minimum of value distribution support
    V_max = 10.,  # Maximum of value distribution support
    epsilon=0.999,
    epsilon_decay=0.99,
    epsilon_min=0.01,
    save_interval=10000,
    **unused_kwargs):

    # TODO: set calc_priority_weight_increase() befor training starts

    self.save_path = Path("./results/rainbow/")
    self.train_process_data_fname = "rainbow_training.npy"
    self.train_process_plot_fname = "rainbow_training.png"

    self.action_dim = action_dim
    self.batch_size = batch_size
    self.discount = discount
    self.device = device
    self.training_interval = training_interval
    self.target_update = target_update
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.priority_weight = priority_weight
    self.priority_weight_increase = None
    self.n = multi_step
    self.Vmin = V_min
    self.Vmax = V_max
    self.atoms = atoms
    self.save_interval = save_interval

    self.support = torch.linspace(V_min, V_max, atoms).to(device=self.device)
    self.delta_z = (V_max - V_min) / (atoms - 1)

    # Tensorboard
    tensorboard_dir = self.save_path / "tensorboard_log"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    self.writer = SummaryWriter(
      log_dir=tensorboard_dir.absolute().as_posix())

    # Main Q-Net
    self.online_net = dqn_model.DQN(atoms,
                                    action_space=self.action_dim,
                                    history_length=history_length,
                                    noisy_std=noisy_std
                                    ).to(device=self.device)
    self.online_net.train()

    # Target Q-Net
    self.target_net = dqn_model.DQN(atoms,
                                    action_space=self.action_dim,
                                    history_length=history_length,
                                    noisy_std=noisy_std
                                    ).to(device=self.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    # Optimizer
    self.optimiser = optim.Adam(self.online_net.parameters(),
                                #lr=lr,
                                eps=adam_eps)

    # Replay Buffer
    self.mem = prioritized_buffer.ReplayMemory(device=device,
                                               state_shape=state_dim,
                                               history_length=history_length,
                                               discount=discount,
                                               multi_step=multi_step,
                                               priority_weight=priority_weight,
                                               priority_exponent=priority_exponent,
                                               capacity=buffer_size)

    # Load models
    try:
      self.load(self.save_path)
      logging.info(f"The model was loaded from "
                   f"'{self.save_path.absolute().as_posix()}'")
    except Exception as e:
      logging.info(f"No model loaded from "
                   f"'{self.save_path.absolute().as_posix()}'")

    self.global_step = 0
    self.n_steps_before_training = n_steps_before_training
    self.total_reward = []
    self.reward_history = []

  def __call__(self, state):
    return self.plan(state)

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()

  def plan(self, state):
    with torch.no_grad():
      #state = torch.from_numpy(np.flip(state, axis=0).copy())
      #state = state.permute(2, 0, 1).float().to(self.device)
      action = (self.online_net(state.unsqueeze(0))
                * self.support).sum(2).argmax(1).item()
    return action

  def act_e_greedy(self, state):
    if np.random.random() < self.epsilon:
      action = np.random.randint(0, self.action_dim)
    else:
      action = self.plan(state)
    return action

  def step(self, state, action, next_state, reward, done):
    assert self.priority_weight_increase is not None
    # TODO: add pix and sem states
    self.mem.append(state=state,
                    action=action,
                    reward=reward,
                    terminal=done)
    self.total_reward.append(reward)

    if (self.global_step >= self.n_steps_before_training
        and len(self.mem) > self.batch_size):
      self.mem.priority_weight = min(self.mem.priority_weight
                                     + self.priority_weight_increase, 1)
      if self.global_step % self.training_interval == 0:
        self._optimize()
      if self.global_step % self.target_update == 0:
        self.update_target_net()

    if self.global_step % 1000 == 0:
      avrg_reward = sum(self.total_reward) / len(self.total_reward)
      self.reward_history.append(avrg_reward)
      self.total_reward = []
      self._plot_results()

    if self.global_step % self.save_interval == 0:
      self.save(file_path=self.save_path)

    self.global_step += 1

  def reset_noise(self):
    self.online_net.reset_noise()

  def calc_priority_weight_increase(self, n_total_train_steps):
    self.priority_weight_increase = (1 - self.priority_weight) / (n_total_train_steps - self.n_steps_before_training)

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def evaluate_q(self, state):
    """Evaluates Q-value based on single state (no batch)"""
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0))
              * self.support).sum(2).max(1)[0].item()

  def _optimize(self):
    """
    Train with n-step distributional double-Q learning
    [Source: https://github.com/Kaixhin/Rainbow/blob/master/agent.py]
    """
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = self.mem.sample(
      self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(
        1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(
        self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(
        0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
        self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1),
                            (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1),
                            (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.to_tensorboard(var=torch.mean(loss), name="Avrg Batch Q-Net Loss")
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    self.optimiser.step()

    self.mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def save(self, file_path):
    file_path.mkdir(parents=True, exist_ok=True)
    torch.save(self.online_net.state_dict(),
               (file_path / "rainbow_online_net").absolute().as_posix())
    torch.save(self.target_net.state_dict(),
               (file_path / "rainbow_target_net").absolute().as_posix())
    torch.save(self.optimiser.state_dict(),
               (file_path / "rainbow_optimiser").absolute().as_posix())

  def load(self, file_path):
    file_path.mkdir(parents=True, exist_ok=True)
    self.online_net.load_state_dict(torch.load(
      (file_path / "rainbow_online_net").absolute().as_posix()))
    self.target_net.load_state_dict(torch.load(
      (file_path / "rainbow_target_net").absolute().as_posix()))
    self.optimiser.load_state_dict(torch.load(
      (file_path / "rainbow_optimiser").absolute().as_posix()))

  def _plot_results(self):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(self.reward_history)), self.reward_history)
    ax.set(xlabel="Iterations", ylabel="Average Reward",
           title="Reward during Training")
    ax.grid()
    fig.savefig(
      (self.save_path / self.train_process_plot_fname).absolute().as_posix())
    plt.close()
    np.save(self.train_process_data_fname, self.reward_history)

  def to_tensorboard(self, var, name):
    self.writer.add_scalar(name, var, self.global_step)
