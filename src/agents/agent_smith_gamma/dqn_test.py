import gym
import numpy as np
from absl import logging
from collections import deque
import torch
torch.backends.cudnn.enabled = True
logging.set_verbosity('info')
logging.set_stderrthreshold('info')

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
             ep_steps=1000,
             replay_frequency=4,  # Frequency of sampling from memory
             reward_clip=1,
             eval_freq_ep=10,
             stack_len=3,
             learn_start=int(20e3)):
  action_op = Action(n_action=env.action_space.shape[0])

  agent.train()
  done = True
  reward_total = 0
  reward_ep = -100
  agent.calc_priority_weight_increase(max_steps)

  state_stack = deque(maxlen=4)

  n_ep = 0

  for step in range(1, max_steps + 1):
    if done or step % ep_steps == 0:
      logging.info(f"Ep {n_ep}\t Step {step}\t Ep Reward {reward_ep:.3f}")
      agent.to_tensorboard(var=reward_ep,
                           name=f"Episode Reward")
      action_cmd = np.zeros(3)
      reward_ep = 0
      n_ep += 1
      state = env.reset()

    if step % replay_frequency == 0:
      agent.reset_noise()  # Draw a new set of noisy weights

    state = torch.from_numpy(np.flip(state, axis=0).copy())
    state = state.permute(2, 0, 1).float()
    state_stack.append(state)

    action_idx = agent(state.to("cuda"))
    action_cmd += action_op(action_idx)
    action_cmd = action_cmd.clip(min=[-1, 0, 0], max=[1, 1, 1])

    #if n_ep % eval_freq_ep == 0:
    #  env.render()
    next_state, reward, done, _ = env.step(action_cmd)

    if reward_clip > 0:
      reward = max(min(reward, reward_clip), -reward_clip)

    if len(state_stack) >= stack_len:
      agent.step(state=np.vstack(state_stack),  # torch.from_numpy(np.flip(state, axis=0).copy())
                 action=action_idx,
                 next_state=None,
                 reward=reward,
                 done=done)

    state = next_state
    reward_ep += reward
    reward_total += reward


if __name__ == '__main__':
  env = gym.make("CarRacing-v0")
  stack_len = 3
  a = Action(n_action=env.action_space.shape[0])
  state_dim = np.asarray(env.observation_space.shape)
  state_dim[-1] *= stack_len
  agent = RainbowAgent(state_dim=tuple(state_dim),  # (96,96,3)
                       action_dim=len(a),
                       device="cuda")
  training(env, agent, stack_len=stack_len)

