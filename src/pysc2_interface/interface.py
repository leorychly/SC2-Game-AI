import numpy as np
from pysc2.lib import actions, features, units


class Interface(object):
  """Class for storing all interface function to PySC2."""

  @staticmethod
  def get_units_by_type(obs, unit_type, enemy=False):
    if enemy:
      return [unit for unit in obs.observation.raw_units
              if unit.unit_type == unit_type
              and unit.alliance == features.PlayerRelative.ENEMY]
    else:
      return [unit for unit in obs.observation.raw_units
              if unit.unit_type == unit_type
              and unit.alliance == features.PlayerRelative.SELF]

  @staticmethod
  def get_completed_units_by_type(obs, unit_type, enemy=False):
    if enemy:
      return [unit for unit in obs.observation.raw_units
              if unit.unit_type == unit_type
              and unit.build_progress == 100
              and unit.alliance == features.PlayerRelative.ENEMY]
    else:
      return [unit for unit in obs.observation.raw_units
              if unit.unit_type == unit_type
              and unit.build_progress == 100
              and unit.alliance == features.PlayerRelative.SELF]

  @staticmethod
  def get_distances(obs, units, xy):
    units_xy = [(unit.x, unit.y) for unit in units]
    return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

