import numpy as np
from pysc2.lib import actions, features, units


class Interface(object):

  def __init__(self):
    pass

  def get_my_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.SELF]

  def get_enemy_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.ENEMY]

  def get_my_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.SELF]

  def get_enemy_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.ENEMY]

  def get_distances(self, obs, units, xy):
    units_xy = [(unit.x, unit.y) for unit in units]
    return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

