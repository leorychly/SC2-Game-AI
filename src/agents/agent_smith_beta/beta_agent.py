import torch
import numpy as np
from pathlib2 import Path
from absl import logging

from src.commons import ActionData
from src.agents.base_agent import Agent
from src.pysc2_interface.interface import Interface
from src.pysc2_actions.hybrid_actions import ActionsHybrid
from src.observer.hybrid_observer import HybridObserver

from src.agents import reward_fn
from src.agents.agent_smith_beta.td3 import TD3Agent
from src.utils import plotting


# TODO:
# add avrg reward per game
# change all file paths to pathlib2.Paths
# add agent playing against it self
# win/draw/loss plotting is not at 100% after first games

# TODO: Adjust this script to state split


class AgentSmithBeta(Agent):

  def __init__(self, game_screen_resolution):
    super(AgentSmithBeta, self).__init__()
    n_continuous_action = 2
    self.game_screen_resolution = game_screen_resolution  # TODO: (x,y)?

    self.interface = Interface()
    self.pysc2_actions = ActionsHybrid()
    self.observer = HybridObserver()
    self.reward_fn = reward_fn.ScoreRewardFn()

    self.prev_state = None
    self.prev_action = None
    self.base_top_left = None
    self.progress_data = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    action_dim = (len(self.pysc2_actions), n_continuous_action)
    action_limits = [[0, 1] for _ in range(len(self.pysc2_actions))]
    action_limits.append([0, game_screen_resolution[0]])
    action_limits.append([0, game_screen_resolution[1]])
    action_limits = np.asarray(action_limits)

    self.policy = TD3Agent(state_dim=self.observer.shape,
                           action_dim=action_dim,
                           device=device)

    self.action_hist_fname = "./results/action_hist.png"
    self.data_progress_fname = "./results/training_progress.npy"
    self.plot_progress_fname = "./results/training_progress.png"
    self.model_path = Path("./results/td3_model/")
    try:
      self.policy.load(self.model_path)
      logging.info(f"The model was loaded from "
                   f"'{self.model_path.absolute().as_posix()}'")
    except Exception as e:
      logging.info(f"No model loaded from "
                   f"'{self.model_path.absolute().as_posix()}'")

    self.game_step = 0  # Is updated prior to each step execution

  def reset(self):
    super(AgentSmithBeta, self).reset()
    self.new_game()

  def new_game(self):
    self.base_top_left = None
    self.prev_state = None
    self.prev_action = None
    self.game_step = 0
    self.progress_data.append({"game_result": None,
                               "game_length": self.game_step,
                               "actions_taken": np.zeros(len(self.pysc2_actions))})

  def step(self, obs):
    self.game_step += 1
    state = self.observer.get_state(obs)
    assert isinstance(state, tuple)
    #state_pix, state_sem = state
    reward = self.reward_function(obs)
    done = obs.last()
    if obs.first():
      self._first_step(obs)
    if obs.last():
      pysc2_action = self._last_step(
        obs=obs, state=state, reward=reward, done=done)
    else:
      pysc2_action = self._step(
        obs=obs, state=state, reward=reward, done=done)
    return pysc2_action

  def _step(self, obs, state, reward, done):
    action = self.choose_action(state)
    x = action[-2]
    y = action[-1]
    if self.prev_action is not None:
      self.policy.step(state=self.prev_state,
                       action=self.prev_action,
                       reward=reward,
                       next_state=state,
                       done=done)
    self.log_results(obs, action)
    self.prev_state = state
    self.prev_action = action

    action_args = ActionData(obs=obs,
                             #base_top_left=self.base_top_left,
                             x=x,
                             y=y)
    pysc2_action = self.pysc2_actions(np.argmax(action[:-2]))
    return pysc2_action(action_args)

  def _first_step(self, obs):
    super(AgentSmithBeta, self).step(obs)
    self.pysc2_actions.set_base_pos(self.base_top_left)

  def _last_step(self, obs, state, reward, done):
    if obs.reward == 1:
      reward += 10
    self.policy.step(state=self.prev_state,
                     action=self.prev_action,
                     reward=reward,
                     next_state=state,
                     done=done)
    action = np.expand_dims(np.zeros(len(self.pysc2_actions) + 2), axis=0)
    action[0, 0] = 1  # Do Nothing action at index 0
    self.log_results(obs, action)
    pysc2_action = self.pysc2_actions.do_nothing()
    action_args = ActionData(obs=obs,
                             base_top_left=self.base_top_left)
    return pysc2_action(action_args)

  def is_same_state(self, s1, s2):
    #return True if np.allclose(s1, s2, rtol=0.1) else False
    return np.array_equal(s1, s2)

  def choose_action(self, state):
    """Return 1-hot action index and X,Y cursor position."""
    policy_output = self.policy(state)
    action_idx = np.argmax(policy_output[:-2])
    x = policy_output[-2]
    y = policy_output[-1]
    assert 0 <= x <= 1
    assert 0 <= y <= 1
    x *= self.game_screen_resolution[0]
    y *= self.game_screen_resolution[1]
    categorical_action = self._as_one_hot(action_idx, len(policy_output) - 2)
    action = np.concatenate((categorical_action, np.array([x, y])))
    return action

  def _as_one_hot(self, idx, length):
    one_hot_vect = np.zeros(length)
    one_hot_vect[idx] += 1
    return one_hot_vect

  #def choose_action_custom(self, state):
  #  if np.random.random() < 0.2:
  #    a_idx = np.random.randint(len(self.pysc2_actions) - 2)
  #  elif np.random.random() < 0.2:
  #    a_idx = 5  # attack
  #  else:
  #    a_idx = 4  # build marine
  #  return a_idx

  def reward_function(self, obs):
    reward = self.reward_fn(obs)
    return reward

  def log_results(self, obs, action):
    action_idx = np.argmax(action[:, :-2])
    self.progress_data[-1]["actions_taken"][action_idx] += 1
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
