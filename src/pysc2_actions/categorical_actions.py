import random
import numpy as np
from src.pysc2_interface.interface import Interface
from src.pysc2_actions import action_fn_categorical
from src.pysc2_actions import action_fn_categorical_weak


class ActionsCategorical(object):

  """Only categorical actions without specific X,Y coordiante selection"""

  def __init__(self):
    self.fn = Interface()
    self.base_top_left = "Not set"
    self._actions = [
      action_fn_categorical.do_nothing,
      action_fn_categorical.harvest_minerals,
      action_fn_categorical.build_supply_depot,
      action_fn_categorical.build_barracks,
      action_fn_categorical.train_marine,
      action_fn_categorical.attack]
    self._weak_actions = [
      action_fn_categorical_weak.do_nothing,
      action_fn_categorical_weak.harvest_minerals,
      action_fn_categorical_weak.build_supply_depot,
      action_fn_categorical_weak.build_barracks,
      action_fn_categorical_weak.train_marine,
      action_fn_categorical_weak.attack]
#    assert len(self._actions) == self.get_limits().shape[0]

  def __len__(self):
    return len(self._actions)

  def __call__(self, idx, weak=False):
    assert self.base_top_left != "Not set"
    if weak:
      return self._weak_actions[idx]
    else:
      return self._actions[idx]

  def sample(self, weak=False):
    if weak:
      return random.choice(self._weak_actions)
    else:
      return random.choice(self._actions)

  def set_base_pos(self, base_top_left):
    self.base_top_left = base_top_left

  def do_nothing(self):
    return self._actions[0]

