import gym
import numpy as np
from absl import logging
import torch
torch.backends.cudnn.enabled = True

from src.agents.agent_smith_gamma.rainbow_dqn import RainbowAgent


class Action:
  def __init__(self, n_action):
    self.action_cmd = np.zeros(n_action)
    self.modification_lst = [
      [0, 0, 0],
      [0.1, 0, 0],
      [0.2, 0, 0],
      [0.4, 0, 0],
      [-0.2, 0, 0],
      [-0.4, 0, 0],
      [0, 0.1, 0],
      [0, 0.2, 0],
      [0, 0.4, 0],
      [0, -0.1, 0],
      [0, -0.2, 0],
      [0, -0.4, 0],
      [0, 0, 0.1],
      [0, 0, 0.2],
      [0, 0, 0.4],
      [0, 0, -0.1],
      [0, 0, -0.2],
      [0, 0, -0.4]
    ]

  def __len__(self):
    return len(self.modification_lst)

  def __call__(self, idx):
    return self.action_cmd + self.modification_lst[idx]


def training(env,
             agent,
             max_steps=int(50e6),
             replay_frequency=4,  # Frequency of sampling from memory
             reward_clip=1,
             learn_start=int(20e3)):
  action_op = Action(n_action=env.action_space.shape[0])

  agent.train()
  done = True
  reward_total = 0
  reward_window = 0

  agent.calc_priority_weight_increase(max_steps)
  prev_reward = 0
  action_cmd = np.zeros(3)

  for step in range(1, max_steps + 1):
    if done:
      state = env.reset()
    if step % replay_frequency == 0:
      agent.reset_noise()  # Draw a new set of noisy weights

    state = torch.from_numpy(np.flip(state, axis=0).copy())
    state = state.permute(2, 0, 1).float()

    action_idx = agent(state.to("cuda"))
    action_cmd += action_op(action_idx)
    action_cmd = action_cmd.clip(min=[-1, 0, 0], max=[1, 1, 1])
    next_state, reward, done, _ = env.step(action_cmd)

    if reward_clip > 0:
      reward = max(min(reward, reward_clip), -reward_clip)

    agent.step(state=state,  # torch.from_numpy(np.flip(state, axis=0).copy())
               action=action_idx,
               next_state=None,
               reward=reward,
               done=done)

    state = next_state
    reward_window += reward
    reward_total += reward

    logging_interval = 1000
    delta_reward = reward_total - prev_reward
    prev_reward = reward_total
    agent.to_tensorboard(var=delta_reward,
                         name=f"Reward gain in last {logging_interval}"
                              f"episode")
    agent.to_tensorboard(var=reward_total, name="Total Reward")
    if step % logging_interval == 0:
      reward_window = 0
      logging.info(f">Step {step}\tDelta Reward{delta_reward:.3f}")


if __name__ == '__main__':
  env = gym.make("CarRacing-v0")
  a = Action(n_action=env.action_space.shape[0])
  agent = RainbowAgent(state_dim=env.observation_space.shape,  # (96,96,3)
                       action_dim=len(a),
                       device="cuda")
  training(env, agent)

