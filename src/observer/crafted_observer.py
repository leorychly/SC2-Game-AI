from pysc2.lib import actions, features, units

from src.observer.base_observer import BaseObserver
from src.pysc2_interface.interface import Interface
from src.pysc2_actions.actions import Actions


class CraftedObserver(BaseObserver):

  def __init__(self):
    self.state_dim = 21
    self.interface = Interface()
    self.actions = Actions()

  def __len__(self):
    return self.state_dim

  def player_info(self, obs):
    player_info = obs.observation["player"]  # (11,)

  def get_state(self, obs):
    scvs, idle_scvs, command_centers, completed_command_centers, \
    supply_depots, completed_supply_depots = self.get_basic_info(obs)

    barrackses = self.interface.get_units_by_type(
      obs, units.Terran.Barracks, enemy=False)
    completed_barrackses = self.interface.get_completed_units_by_type(
      obs, units.Terran.Barracks, enemy=False)

    marines, queued_marines = self.get_marine_info(obs)

    free_supply, can_afford_supply_depot, can_afford_barracks, \
    can_afford_marine = self.get_eco_info(obs)

    enemy_scvs, enemy_idle_scvs, enemy_command_centers, \
    enemy_supply_depots, enemy_completed_supply_depots, \
    enemy_barrackses, enemy_completed_barrackses, \
    enemy_marines = self.get_enemy_info(obs)

    state = (len(command_centers),
            len(scvs),
            len(idle_scvs),
            len(supply_depots),
            len(completed_supply_depots),
            len(barrackses),
            len(completed_barrackses),
            len(marines),
            queued_marines,
            free_supply,
            can_afford_supply_depot,
            can_afford_barracks,
            can_afford_marine,
            len(enemy_command_centers),
            len(enemy_scvs),
            len(enemy_idle_scvs),
            len(enemy_supply_depots),
            len(enemy_completed_supply_depots),
            len(enemy_barrackses),
            len(enemy_completed_barrackses),
            len(enemy_marines))
    assert len(state) == self.state_dim
    return state

