import random
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units


class Agent(base_agent.BaseAgent):
  actions = ("do_nothing",
             "harvest_minerals",
             "build_supply_depot",
             "build_barracks",
             "train_marine",
             "attack")

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

  def step(self, obs):
    super(Agent, self).step(obs)
    if obs.first():
      command_center = self.get_my_units_by_type(
        obs, units.Terran.CommandCenter)[0]
      self.base_top_left = (command_center.x < 32)

  def do_nothing(self, obs):
    return actions.RAW_FUNCTIONS.no_op()

  def harvest_minerals(self, obs):
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    idle_scvs = [scv for scv in scvs if scv.order_length == 0]
    if len(idle_scvs) > 0:
      mineral_patches = [unit for unit in obs.observation.raw_units
                         if unit.unit_type in [
                           units.Neutral.BattleStationMineralField,
                           units.Neutral.BattleStationMineralField750,
                           units.Neutral.LabMineralField,
                           units.Neutral.LabMineralField750,
                           units.Neutral.MineralField,
                           units.Neutral.MineralField750,
                           units.Neutral.PurifierMineralField,
                           units.Neutral.PurifierMineralField750,
                           units.Neutral.PurifierRichMineralField,
                           units.Neutral.PurifierRichMineralField750,
                           units.Neutral.RichMineralField,
                           units.Neutral.RichMineralField750
                         ]]
      scv = random.choice(idle_scvs)
      distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
      mineral_patch = mineral_patches[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
        "now", scv.tag, mineral_patch.tag)
    return actions.RAW_FUNCTIONS.no_op()

  def build_supply_depot(self, obs):
    supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    if (len(supply_depots) == 0 and obs.observation.player.minerals >= 100 and
      len(scvs) > 0):
      supply_depot_xy = (22, 26) if self.base_top_left else (35, 42)
      distances = self.get_distances(obs, scvs, supply_depot_xy)
      scv = scvs[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
        "now", scv.tag, supply_depot_xy)
    return actions.RAW_FUNCTIONS.no_op()

  def build_barracks(self, obs):
    completed_supply_depots = self.get_my_completed_units_by_type(
      obs, units.Terran.SupplyDepot)
    barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    if (len(completed_supply_depots) > 0 and len(barrackses) == 0 and
      obs.observation.player.minerals >= 150 and len(scvs) > 0):
      barracks_xy = (22, 21) if self.base_top_left else (35, 45)
      distances = self.get_distances(obs, scvs, barracks_xy)
      scv = scvs[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Build_Barracks_pt(
        "now", scv.tag, barracks_xy)
    return actions.RAW_FUNCTIONS.no_op()

  def train_marine(self, obs):
    completed_barrackses = self.get_my_completed_units_by_type(
      obs, units.Terran.Barracks)
    free_supply = (obs.observation.player.food_cap -
                   obs.observation.player.food_used)
    if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
      and free_supply > 0):
      barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
      if barracks.order_length < 5:
        return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
    return actions.RAW_FUNCTIONS.no_op()

  def attack(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
      attack_xy = (38, 44) if self.base_top_left else (19, 23)
      distances = self.get_distances(obs, marines, attack_xy)
      marine = marines[np.argmax(distances)]
      x_offset = random.randint(-4, 4)
      y_offset = random.randint(-4, 4)
      return actions.RAW_FUNCTIONS.Attack_pt(
        "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
    return actions.RAW_FUNCTIONS.no_op()