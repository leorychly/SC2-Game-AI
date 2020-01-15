import gym
import torch
import cv2
import numpy as np
from collections import deque
from PIL import Image
import time

from src.agents.agent_smith_beta.td3 import TD3Agent


class TestTD3:

  def __init__(self):
    self.render = True
    self.goal_score = 100
    self.batch_size = 256
    self.n_images_in_stack = 5
    self.rewards = []
    self.short_term_memory = deque(maxlen=self.n_images_in_stack)

    env = gym.make('CarRacing-v0')
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    state_dim = ((64, 64, 5), (10,))  # ((64,64,5), (398))
    action_dim = (3, 0)  # (6,2)
    self.policy = TD3Agent(state_dim=state_dim,
                           action_dim=action_dim,
                           device=self.device)
    self.run_test(env)
    env.close()

  def _latest_stacked_state(self):
    state = np.vstack(self.short_term_memory)
    return state

  @staticmethod
  def _resize_state(state):
      state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
      state = cv2.resize(state, dsize=(64, 64))
      #img = Image.fromarray(state)
      #img.save(f"./img/im{time.time()}.png")
      state = np.expand_dims(state, axis=0)
      return state

  def run_test(self, env, n_ep=1000, n_steps=1000, print_interval=10):
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
        action[0] *= 2
        action[0] -= 1
        action = np.clip(action, a_min=[-1, 0, 0], a_max=[1, 1, 1])

        if self.render:
          env.render()
        next_state_img, reward, done, _ = env.step(action.flatten())

        next_state_img = self._resize_state(next_state_img)
        self.short_term_memory.append(next_state_img)
        next_state_stack = self._latest_stacked_state()
        next_state = (next_state_stack, np.zeros(10))

        self.policy.step(state=state,
                         action=action,
                         next_state=next_state,
                         reward=reward,
                         done=done)

        self.policy.add_to_tensorboard(action.flatten()[0], "Action 1")
        self.policy.add_to_tensorboard(action.flatten()[1], "Action 2")
        self.policy.add_to_tensorboard(action.flatten()[2], "Action 3")

        #state = next_state
        episode_reward += reward
        frame_idx += 1

        if done:
          break

      scores_deque.append(episode_reward)
      scores.append(episode_reward)

      if ep % print_interval == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(ep, np.mean(scores_deque)))

      if np.mean(scores_deque) >= self.goal_score:
        print(f'\nEnvironment solved in {ep - 100:d} episodes!'
              f'\tAverage Score: {np.mean(scores_deque):.2f}')
        break

      self.rewards.append(episode_reward)


if __name__ == '__main__':
    test = TestTD3()