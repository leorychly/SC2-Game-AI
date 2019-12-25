import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib2 import Path
from absl import logging
from gym import wrappers
import torch
import torch.nn.functional as F

from src.agents.agent_smith_beta import td3_actor, td3_critic, buffer_simple


class TD3Agent:

  def __init__(
    self,
    state_dim,
    action_dim,
    action_limits,
    device,
    actor_lr=3e-4,
    critic_lr=3e-4,
    batch_size=64,
    buffer_size=10000,
    discount=0.99,
    tau=0.005,
    policy_noise=0.2,
    expl_noise=0.1,
    noise_clip=0.5,
    training_interval=2,
    policy_update_freq=2,
    **unused_kwargs):
    """
    Initialize the TD3 Agent.

    :param state_dim: Tuple
      Containing the state sizes of the tensor and vector state parts.
      E.g.: ( (28, 28, 3),  7 )
    :param action_dim: Tuple
      Containing the both, the number of possible actions as well as the
      contrinuous actions.
      E.g.: For 13 1-hot encoded action and 2 continuous
      actions: (13, 2)
    :param action_limits: nd.array
      Min and Max values of the actions in the shape of
      [[min1, min2, ...], [max1, max2, ...]]
    :param device:
    :param actor_lr: Float
    :param critic_lr: Float
    :param buffer_size: Int
    :param discount: Float
    :param tau: Float
    :param policy_noise: Float
    :param expl_noise: Float
    :param noise_clip: Float or Int
    :param policy_update_freq: Int
    :param unused_kwargs: Dict
    """
    self.save_path = Path("./results")
    self.train_process_data_fname = "td3_training.npy"
    self.train_process_plot_fname = "td3_training.png"

    self.min_action_limit = action_limits[:, ]  # TODO: Test if vector of size len(actions)
    self.max_action_limit = action_limits[:, 1]  # TODO: Test if vector of size len(actions)
    policy_noise = policy_noise * self.max_action_limit
    noise_clip = noise_clip * self.max_action_limit

    self.actor = td3_actor.Actor(img_state_channel_dim=state_dim[0],
                                 vect_state_len=state_dim[1],
                                 action_space_dim=sum(action_dim))
    self.actor = self.actor.to(device)
    self.actor_target = copy.deepcopy(self.actor)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                            lr=actor_lr)

    self.critic = td3_critic.Critic(img_state_channel_dim=state_dim[0],
                                    vect_state_len=state_dim[1],
                                    action_dim=1 + action_dim[1])
    self.critic = self.critic.to(device)
    self.critic_target = copy.deepcopy(self.critic)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                             lr=critic_lr)

    self.replay_buffer = buffer_simple.SimpleBuffer(max_size=buffer_size,
                                                    device=device)

    self.action_dim = action_dim
    self.batch_size = batch_size
    self.discount = discount
    self.tau = tau
    self.policy_noise = policy_noise
    self.noise_clip = noise_clip
    self.expl_noise = expl_noise
    self.device = device

    self.training_interval = training_interval
    self.policy_update_freq = policy_update_freq

    self.global_step = 0
    self.total_reward = []
    self.reward_history = []

  def plan(self, state, exploration_noise_on=True):
    noise = np.zeros(sum(self.action_dim))
    if exploration_noise_on:
      noise = np.random.normal(loc=0,
                               scale=max(abs(self.min_action_limit),
                                         abs(self.max_action_limit))
                                     * self.expl_noise,
                               size=sum(self.action_dim))
    action = self.actor(state)
    action += noise
    action = action.clip(self.min_action, self.max_action)
    return action

  def setp(self, state, action, next_state, reward, done):
    self.replay_buffer.push(state_pix=state[0],
                            state_sem=state[1],
                            action=action,
                            reward=reward,
                            next_state_pix=next_state[0],
                            next_state_sem=next_state[1],
                            done=done)
    self.total_reward.append(reward)

    if (self.global_step % self.training_interval == 0
        and len(self.replay_buffer) > self.batch_size):
      batch = self.replay_buffer.sample(self.batch_size)
      self._optimize(batch)
      avrg_reward = sum(self.total_reward) / len(self.total_reward)
      self.reward_history.append(avrg_reward)
      self.total_reward = []

    if self.global_step % 100 == 0:
      self._plot_results()

    self.global_step += 1

  def _optimize(self, batch):
    state, action, next_state, reward, done = batch
    # TODO: check if "done" means done or notDone

    with torch.no_grad():
      noise = (torch.randn_like(action) * self.policy_noise).clamp(
        -self.noise_clip, self.noise_clip)
      next_action = (self.actor_target(next_state) + noise).clamp(
        -self.max_action, self.max_action)

      # Compute target Q value
      target_q1, target_q2 = self.critic_target((next_state, next_action))
      target_q = torch.min(target_q1, target_q2)
      target_q = reward + (1 - done) * self.discount * target_q

    # Get current Q estimates
    current_q1, current_q2 = self.critic((state, action))

    # Compute critic loss
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Delayed policy updates
    if self.total_iter % self.policy_update_freq == 0:

      # Compute actor losse
      actor_loss = -self.critic.Q1((state, self.actor(state))).mean()

      # Optimize the actor
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      # Update the frozen target models
      for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

      for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

  def save(self, file_path):
    file_path.mkdir(parents=True, exist_ok=True)
    torch.save(self.critic.state_dict(),
               (file_path / "td3_critic").absolute().as_posix())
    torch.save(self.critic_optimizer.state_dict(),
               (file_path / "td3_critic_optimizer").absolute().as_posix())
    torch.save(self.actor.state_dict(),
               (file_path / "td3_actor").absolute().as_posix())
    torch.save(self.actor_optimizer.state_dict(),
               (file_path / "td3_actor_optimizer").absolute().as_posix())

  def load(self, file_path):
    file_path.mkdir(parents=True, exist_ok=True)
    self.critic.load_state_dict(torch.load(
      (file_path / "td3_critic").absolute().as_posix()))
    self.critic_optimizer.load_state_dict(torch.load(
      (file_path / "td3_critic_optimizer").absolute().as_posix()))
    self.actor.load_state_dict(torch.load(
      (file_path / "td3_actor").absolute().as_posix()))
    self.actor_optimizer.load_state_dict(torch.load(
      (file_path / "td3_actor_optimizer").absolute().as_posix()))

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
