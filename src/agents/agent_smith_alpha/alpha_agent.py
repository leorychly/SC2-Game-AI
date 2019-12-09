import torch
import numpy as np
from pathlib2 import Path
from absl import logging
from pysc2.lib import actions, features, units

from src.commons import WorldState
from src.agents.base_agent import Agent
from src.pysc2_interface.interface import Interface
from src.pysc2_actions.actions import Actions
from src.observer.crafted_observer import CraftedObserver
from src.agents.agent_smith_alpha import reward_fn
from src.agents.agent_smith_alpha.dqn import DQNAgent
from src.agents.agent_smith_alpha import plotting

# TODO: add avrg reward per game
# TODO: change all file paths to pathlib2.Paths
# TODO: add agent playing against it self
# TODO: win/draw/loss plotting is not at 100% after first games

class AgentSmithAlpha(Agent):
  def __init__(self):
    super(AgentSmithAlpha, self).__init__()
    self.interface = Interface()
    self.actions = Actions()
    self.observer = CraftedObserver()
    self.reward_fn = reward_fn.SparseRewardFn()

    self.prev_state = None
    self.prev_action = None
    self.base_top_left = None
    self.progress_data = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.policy = DQNAgent(state_dim=len(self.observer),
                           action_dim=len(self.actions),
                           device=device)

    self.action_hist_fname = "./results/action_hist.png"
    self.data_progress_fname = "./results/training_progress.npy"
    self.plot_progress_fname = "./results/training_progress.png"
    self.model_path = Path("./results/dqn_model/")
    try:
      self.policy.load(self.model_path)
      logging.info(f"The model was loaded from "
                   f"'{self.model_path.absolute().as_posix()}'")
    except Exception as e:
      logging.info(f"No model loaded from "
                   f"'{self.model_path.absolute().as_posix()}'")

    self.game_step = 0  # Is updated prior to each step execution
    #self.new_game()

  def reset(self):
    super(AgentSmithAlpha, self).reset()
    self.new_game()

  def new_game(self):
    self.base_top_left = None
    self.prev_state = None
    self.prev_action = None
    self.game_step = 0
    self.progress_data.append({"game_result": None,
                               "game_length": self.game_step,
                               "actions_taken": np.zeros(len(self.actions))})

  def step(self, obs):
    self.game_step += 1
    state = self.observer.get_state(obs)
    reward = self.reward_function(obs)
    done = obs.last()
    if obs.first():
      self._first_step(obs)
    if obs.last():
      pysc2_action = self._last_step(obs, state, reward, done)
    else:
      pysc2_action = self._step(obs, state, reward, done)
    return pysc2_action

  def _step(self, obs, state, reward, done):
    action = self.choose_action(state)
    if self.prev_action is not None:
      if not self.is_same_state(self.prev_state, state):
        self.policy.step(state=self.prev_state,
                       action=self.prev_action,
                       reward=reward,
                       next_state=state,
                       done=done)
    self.log_results(obs, action)
    self.prev_state = state
    self.prev_action = action

    world_state = WorldState(obs=obs, base_top_left=self.base_top_left)
    pysc2_action = self.actions(action)
    return pysc2_action(world_state)

  def _first_step(self, obs):
    super(AgentSmithAlpha, self).step(obs)
    self.actions.set_base_pos(self.base_top_left)

  def _last_step(self, obs, state, reward, done):
    if obs.reward == 1:
      reward += 10
    self.policy.step(state=self.prev_state,
                     action=self.prev_action,
                     reward=reward,
                     next_state=state,
                     done=done)
    action = 0
    self.log_results(obs, action)
    pysc2_action = self.actions.do_nothing()
    world_state = WorldState(obs=obs, base_top_left=self.base_top_left)
    return pysc2_action(world_state)

  def is_same_state(self, s1, s2):
    #return True if np.allclose(s1, s2, rtol=0.1) else False
    return np.array_equal(s1, s2)

  def choose_action(self, state):
    state = np.asarray(state)
    action = self.policy(state)
    return action

  def choose_action_custom(self, state):
    if np.random.random() < 0.2:
      a_idx = np.random.randint(len(self.actions) - 2)
    elif np.random.random() < 0.2:
      a_idx = 5  # attack
    else:
      a_idx = 4  # build marine
    return a_idx

  def reward_function(self, obs):
    reward = self.reward_fn(obs)
    return reward

  def log_results(self, obs, action):
    self.progress_data[-1]["actions_taken"][action] += 1
    if obs.last():
      game_result = obs.reward
      self.policy.save(self.model_path)
      logging.info(f"Model saved to '{self.model_path}'")
      self.progress_data[-1]["game_result"] = game_result
      self.progress_data[-1]["game_length"] = self.game_step
      plotting.plot_progress(data=self.progress_data,
                             save_dir=self.plot_progress_fname)
      np.save(self.data_progress_fname, self.progress_data)

      plotting.plot_action_histogram(self.progress_data,
                                     self.action_hist_fname)
