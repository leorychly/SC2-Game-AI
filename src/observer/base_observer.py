from pysc2.lib import actions, features, units

from src.pysc2_interface.interface import Interface
from src.pysc2_actions.actions import Actions


class BaseObserver(object):

  def __init__(self):
    self.interface = Interface()
    self.actions = Actions()

  def get_state(self, obs):
    raise NotImplementedError

  # === Info about the own state ===============================================

  def get_eco_info(self, obs):
    free_supply = (obs.observation.player.food_cap -
                   obs.observation.player.food_used)
    can_afford_supply_depot = obs.observation.player.minerals >= 100
    can_afford_barracks = obs.observation.player.minerals >= 150
    can_afford_marine = obs.observation.player.minerals >= 100
    # TODO add minerals, gas (abs and rates)
    return (free_supply, can_afford_supply_depot, can_afford_barracks,
            can_afford_marine)

  def get_basic_info(self, obs):
    scvs = self.interface.get_units_by_type(
      obs, units.Terran.SCV, enemy=False)
    idle_scvs = [scv for scv in scvs if scv.order_length == 0]
    cmd_centers = self.interface.get_units_by_type(
      obs, units.Terran.CommandCenter, enemy=False)
    completed_cmd_centers = self.interface.get_completed_units_by_type(
      obs, units.Terran.CommandCenter, enemy=False)
    supply_depots = self.interface.get_units_by_type(
      obs, units.Terran.SupplyDepot, enemy=False)
    completed_supply_depots = self.interface.get_completed_units_by_type(
      obs, units.Terran.SupplyDepot, enemy=False)
    return (scvs, idle_scvs, cmd_centers, completed_cmd_centers, supply_depots,
            completed_supply_depots)

  def get_marine_info(self, obs):
    completed_rax = self.interface.get_completed_units_by_type(
      obs, units.Terran.Barracks, enemy=False)
    marines = self.interface.get_units_by_type(
      obs, units.Terran.Marine, enemy=False)
    queued_marines = 0
    for rax in completed_rax:
      queued_marines += (rax.order_length if len(completed_rax) > 0 else 0)
    return marines, queued_marines

  # === Info about the enemy state =============================================

  def get_enemy_info(self, obs):
    enemy_scvs = self.interface.get_units_by_type(
      obs, units.Terran.SCV, enemy=True)
    enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
    enemy_command_centers = self.interface.get_units_by_type(
      obs, units.Terran.CommandCenter, enemy=True)
    enemy_supply_depots = self.interface.get_units_by_type(
      obs, units.Terran.SupplyDepot, enemy=True)
    enemy_completed_supply_depots = self.interface.get_completed_units_by_type(
      obs, units.Terran.SupplyDepot, enemy=True)
    enemy_barrackses = self.interface.get_units_by_type(
      obs, units.Terran.Barracks)
    enemy_completed_barrackses = self.interface.get_completed_units_by_type(
      obs, units.Terran.Barracks, enemy=True)
    enemy_marines = self.interface.get_units_by_type(
      obs, units.Terran.Marine, enemy=True)
    return (enemy_scvs, enemy_idle_scvs, enemy_command_centers,
            enemy_supply_depots, enemy_completed_supply_depots,
            enemy_barrackses, enemy_completed_barrackses, enemy_marines)