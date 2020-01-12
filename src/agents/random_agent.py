import random
from src.agents.base_agent import Agent
from src.commons import WorldState
from src.pysc2_actions.categorical_actions import Actions


class RandomAgent(Agent):

  def __init__(self):
    super(RandomAgent, self).__init__()
    self.actions = Actions()
    self.base_top_left = None

  def step(self, obs):
    if obs.first():
      super(RandomAgent, self).step(obs)
      self.actions.set_base_pos(self.base_top_left)
    world_state = WorldState(obs=obs, base_top_left=self.base_top_left)
    action = self.actions.sample()(world_state)
    return action

