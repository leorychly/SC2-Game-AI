import gym
import torch
import cv2
import numpy as np
from collections import deque
from PIL import Image
import time
import copy

from src.agents.agent_smith_beta.td3 import TD3Agent


class TestTD3:

  def __init__(self):
    self.render = False
    self.goal_score = 100
    self.batch_size = 128
    self.n_images_in_stack = 3
    self.rewards = []
    self.short_term_memory = deque(maxlen=self.n_images_in_stack)

    env = gym.make('CarRacing-v0')
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    state_dim = ((64, 64, self.n_images_in_stack * 3), (10,))  # ((64,64,5), (398))
    action_dim = (0, 3)  # (6,2)
    self.policy = TD3Agent(state_dim=state_dim,
                           action_dim=action_dim,
                           device=self.device)
    self.test_is_done = False
    while not self.test_is_done:
      self.run_test(env)
    env.close()

  def _latest_stacked_state(self):
    state = np.vstack(self.short_term_memory)
    return state

  @staticmethod
  def _resize_state(state):
     # state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = resized = cv2.resize(
      state, (64, 64), interpolation=cv2.INTER_AREA)
    #state = cv2.resize(state, dsize=(64, 64))
    #state = np.expand_dims(state, axis=0)
    state = np.rollaxis(state, 2, 0)
    return state

  def run_test(self, env, n_ep=1000, n_steps=500, print_interval=10):
    for ep in range(n_ep):
      scores_deque = deque(maxlen=100)
      scores = []
      state_img = env.reset()
      state_img = self._resize_state(state_img)

      for i in range(self.n_images_in_stack + 1):
        self.short_term_memory.append(state_img)
      episode_reward = 0
      frame_idx = 0

      for step in range(n_steps):

        state_stack = self._latest_stacked_state()
        state = (state_stack, np.zeros(10))

        action = self.policy(state)
        action_exe = copy.deepcopy(action)
        action_exe[0] *= 2
        action_exe[0] -= 1
        #action_exe = np.clip(action_exe, a_min=[-1, 0, 0], a_max=[1, 1, 1])
        if min(action_exe[1:]) < 0:
          print(f"\n\n\n\n\n\nSTOOOOOOOP\n\n\n\n\n\n"
                f"action_exe: {action_exe}\n\n\n")

        if self.render:
          env.render()
        next_state_img, reward, done, _ = env.step(action_exe.flatten())

        next_state_img = self._resize_state(next_state_img)
        self.short_term_memory.append(next_state_img)
        next_state_stack = self._latest_stacked_state()
        next_state = (next_state_stack, np.zeros(10))

        self.policy.step(state=state,
                         action=action,
                         next_state=next_state,
                         reward=reward,
                         done=done)

        self.policy.add_to_tensorboard(action_exe[0], "Action 1")
        self.policy.add_to_tensorboard(action_exe[1], "Action 2")
        self.policy.add_to_tensorboard(action_exe[2], "Action 3")

        #state = next_state
        episode_reward += reward
        frame_idx += 1

        if ep >= n_ep - 1:
          self.test_is_done = True

        if done:
          print(f"\tEp {ep}, Total Reward {episode_reward}")
          break

      scores_deque.append(episode_reward)
      scores.append(episode_reward)

      if ep % print_interval == 0:
        print('\nEpisode {}\tAverage Score: {:.2f}'.format(ep, np.mean(scores_deque)))

      if np.mean(scores_deque) >= self.goal_score:
        print(f'\nEnvironment solved in {ep - 100:d} episodes!'
              f'\tAverage Score: {np.mean(scores_deque):.2f}')
        break

      self.rewards.append(episode_reward)


if __name__ == '__main__':
  test = TestTD3()
