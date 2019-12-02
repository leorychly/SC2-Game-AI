from abc import ABC, abstractmethod

"""
Observation object description:
https://github.com/deepmind/pysc2/blob/master/pysc2/lib/features.py
"""


class BaseRewardFn(ABC):

  def __call__(self, obs):
    return self._compute_reward(obs)

  @abstractmethod
  def _compute_reward(self, obs):
    raise NotImplementedError


class SparseRewardFn(BaseRewardFn):

  def _compute_reward(self, obs):
    reward = obs.reward
    return reward


class KillScoreRewardFn(BaseRewardFn):

  def _compute_reward(self, obs):
    reward = 0
    reward += obs.observation["score_cumulative"][5]  # killed_value_units
    reward += obs.observation["score_cumulative"][6]  # killed_value_structures
    reward += obs.observation["score_by_category"][1].army  # killed_minerals
    reward += obs.observation["score_by_category"][1].economy  # killed_minerals
    return reward
