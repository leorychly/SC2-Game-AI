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


class ScoreRewardFn(BaseRewardFn):
  def _compute_reward(self, obs):
    reward = obs.observation["score_cumulative"][0]  # total score
    return reward / 10000


class KillScoreRewardFn(BaseRewardFn):
  def _compute_reward(self, obs):
    reward = 0
    reward += obs.observation["score_cumulative"][5]  # killed_value_units
    reward += obs.observation["score_cumulative"][6]  # killed_value_structures
    reward += obs.observation["score_by_category"][1].army  # killed_minerals
    reward += obs.observation["score_by_category"][1].economy  # killed_minerals
    return reward / 10000


class ComprehensiveScoreRewardFn(BaseRewardFn):
  def _compute_reward(self, obs):
    reward = 0
    reward += obs.observation["score_cumulative"][3]  # total_value_units
    reward += obs.observation["score_cumulative"][5]  # killed_value_units
    reward += obs.observation["score_cumulative"][6]  # killed_value_structures
    reward += obs.observation["score_cumulative"][7]  # collected_minerals
    reward += obs.observation["score_by_category"][1]  # killed_minerals
    reward -= obs.observation["score_by_category"][3]  # lost_minerals
    reward += obs.observation["score_categories"][1]  # army
    reward += obs.observation["score_categories"][2]  # economy
    return reward / 10000
