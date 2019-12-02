import torch
import numpy as np
from physt import h1
from absl import logging
import matplotlib.pyplot as plt
from pysc2.lib import actions, features, units

from src.commons import WorldState
from src.agents.base_agent import Agent
from src.pysc2_interface.interface import Interface
from src.pysc2_actions.actions import Actions
from src.observer.crafted_observer import CraftedObserver
from src.agents.agent_smith_alpha import reward_fn
from src.agents.agent_smith_alpha.dqn import DQNAgent
from src.agents.agent_smith_alpha.plotting import plot_progress


class AgentSmithAlpha(Agent):
  def __init__(self,
               buffer_size=int(1e6),
               batch_size=128,
               gamma=0.99,
               tau=1e-3,
               lr=1e-4,
               training_interval=2,
               epsilon=0.999,
               epsilon_decay=0.995,
               epsilon_min=0.01):
    super(AgentSmithAlpha, self).__init__()
    self.interface = Interface()
    self.actions = Actions()
    self.observer = CraftedObserver()
    self.reward_fn = reward_fn.KillScoreRewardFn()

    self.prev_state = None
    self.prev_action = None
    self.base_top_left = None
    self.logging = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.policy = DQNAgent(state_dim=len(self.observer),
                           action_dim=len(self.actions),
                           buffer_size=buffer_size,
                           batch_size=batch_size,
                           gamma=gamma,
                           tau=tau,
                           lr=lr,
                           training_interval=training_interval,
                           epsilon=epsilon,
                           epsilon_decay=epsilon_decay,
                           epsilon_min=epsilon_min,
                           device=device)

    self.game_step = 0  # Is updated prior to each step execution
    self.new_game()

  def reset(self):
    super(AgentSmithAlpha, self).reset()
    self.new_game()

  def new_game(self):
    self.base_top_left = None
    self.prev_state = None
    self.prev_action = None
    self.game_step = 0
    self.logging.append({"game_result": None,
                         "game_length": self.game_step,
                         "actions_taken": h1([np.arange(len(self.actions))],
                                             len(self.actions))})

  def step(self, obs):
    self.game_step += 1
    if obs.first():
      self._first_step(obs)
    if obs.last():
      pysc2_action = self._last_step(obs)
    else:
      pysc2_action = self._step(obs)
    return pysc2_action

  def _step(self, obs):
    state = self.observer.get_state(obs)
    action = self.choose_action(state)
    reward = self.reward_function(obs)
    done = obs.last()

    if self.prev_action is not None:
      #if not self.is_same_state(self.prev_state, state):
      self.policy.step(state=self.prev_state,
                       action=self.prev_action,
                       reward=reward,
                       next_state=state,
                       done=done)

    self.log_results(done, reward, action)
    self.prev_state = state
    self.prev_action = action

    world_state = WorldState(obs=obs, base_top_left=self.base_top_left)
    pysc2_action = self.actions(action)
    return pysc2_action(world_state)

  def _first_step(self, obs):
    super(AgentSmithAlpha, self).step(obs)
    self.actions.set_base_pos(self.base_top_left)

  def _last_step(self, obs):
    done = True
    reward = obs.reward
    state = self.observer.get_state(obs)
    self.policy.step(state=self.prev_state,
                         action=self.prev_action,
                         reward=reward,
                         next_state=state,
                         done=done)
    pysc2_action = Actions.do_nothing()
    world_state = WorldState(obs=obs, base_top_left=self.base_top_left)
    return pysc2_action(world_state)

  def is_same_state(self, s1, s2):
    #return True if np.allclose(s1, s2, rtol=0.1) else False
    return np.array_equal(s1, s2)

  def choose_action(self, state):
    state = np.asarray(state)
    action = self.policy(state)
    return action

  def reward_function(self, obs):
    reward = self.reward_fn(obs)
    return reward

  def log_results(self, done, reward, action):
    self.logging[-1]["actions_taken"] << action
    if done:
      self.logging[-1]["game_result"] = reward
      self.logging[-1]["game_length"] = self.game_step
      plot_progress(logging_data=self.logging, save_dir="./progress.png")
      np.save('./my_file.npy', self.logging)
      self.logging[-1]["actions_taken"].plot()
      plt.savefig("./hist.png")
