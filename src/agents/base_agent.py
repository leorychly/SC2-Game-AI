import random
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

from src.pysc2_interface.interface import Interface


class Agent(base_agent.BaseAgent):

  def step(self, obs):
    super(Agent, self).step(obs)
    if obs.first():
      command_center = Interface.get_units_by_type(
        obs=obs, unit_type=units.Terran.CommandCenter, enemy=False)[0]
      self.base_top_left = (command_center.x < 32)
