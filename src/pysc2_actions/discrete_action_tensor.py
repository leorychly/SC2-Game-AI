import random
import numpy as np
from src.pysc2_interface.interface import Interface
from src.pysc2_actions import action_fn_categorical
from src.pysc2_actions import action_fn_hybrid


class DiscreteActionTensor:

  def __init__(self, world_map_size, action_map_size):
    self.world_map_size = world_map_size
    self.action_map_size = action_map_size
    self.n_actions = 5
    self.action_layer = [
      action_fn_categorical.do_nothing,
      action_fn_categorical.harvest_minerals,
      action_fn_hybrid.build_supply_depot,
      action_fn_hybrid.build_barracks,
      action_fn_hybrid.train_marine,
      action_fn_hybrid.attack
    ]

  def __len__(self):
    return self.n_actions

  def __call__(self, action_mat):
    """
    Args:
      action_mat (nd.array): Specifying which action and where
        to execute it. 
        E.g.: For a map of size (64, 64) with 5 possible action,
        the action array could have the following shape (5,64,64)
    Returns (pysc2.action): pysc2 action

    """
    action_idxs = np.unravel_index(action_mat.argmax(), action_mat.shape)
    action_level = action_idxs[0]
    action_pos_idx = (action_idxs[1], action_idxs[2])


  def _actionmap_to_worldmap(self, coords):
    """
    Convert action map coords to real map coords

    Args:
      coords (Tuple): Coordinates on the action map.

    Returns:
      coords (Tuple): Coordinates on the real game map.
    """

  def sample(self):
    return random.choice(self._actions)

  def set_base_pos(self, base_top_left):
    self.base_top_left = base_top_left

  def do_nothing(self):
    return self._actions[0]

