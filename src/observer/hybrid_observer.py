import numpy as np

from torch import nn
from torch import optim

from pysc2.lib import actions, features, units

from src.observer.base_observer import BaseObserver
from src.observer.conv_autoencoder import ConvAutoencoder
from src.pysc2_interface.interface import Interface
from src.pysc2_actions.categorical_actions import Actions


class HybridObserver(BaseObserver):
  """An observer containing raw screen data as well as semantic parameter."""

  def __init__(self):
    super(HybridObserver, self).__init__()
    self.dim_pix = (64, 64, 5)
    self.dim_sem = (312,)
    self.interface = Interface()
    self.actions = Actions()

  @property
  def shape(self):
    return self.dim_pix, self.dim_sem

  def get_state(self, obs):
    pixel_state = self._pixel_state(obs)

    semantic_state = np.concatenate((self._unit_state(obs),
                                     self._custom_state(obs),
                                     self._player_state(obs)))
    print(f"Semantic state space size: {semantic_state.shape} (Set in NN)")
    return pixel_state, semantic_state

  def _pixel_state(self, obs):
    """
    feature_minimap 0  [height_map,
                    1   visibility_map,
                    2   creep,
                    3   camera,
                    4   player_id,
                    5   player_relative,
                    6   selected,
                    7   unit_type,
                    8   alerts,
                    9   pathable,
                    10  buildable]
    feature_screen  0  [height_map,
                    1   visibility_mapc,
                    2   reep,
                    3   power,
                    4   player_id,
                    5   player_relative,
                    6   unit_type,
                    7   selected,
                    8   unit_hit_points,
                    9   unit_hit_points_ratio,
                    10  unit_energy,
                    11  unit_energy_ratio,
                    12  unit_shields,
                    13  unit_shields_ratio,
                    14  unit_density,
                    15  unit_density_aa,
                    16  effects,
                    17  hallucinations,
                    18  cloaked,
                    19  blip,
                    20  buffs,
                    21  buff_duration,
                    22  active,
                    23  build_progress,
                    24  pathable,
                    25  buildable,
                    26  placeholder]

    :param obs:
    :return:
    """
    state = np.stack((
      obs.observation["feature_minimap"][0],
      obs.observation["feature_minimap"][4],
      obs.observation["feature_minimap"][6],
      obs.observation["feature_minimap"][7],
      obs.observation["feature_minimap"][10],
    ), axis=0)
    return state

  def _player_state(self, obs):
    """
    obs.observation["player"] = [
    0   player.player_id,
    1   player.minerals,
    2   player.vespene,
    3   player.food_used,
    4   player.food_cap,
    5   player.food_army,
    6   player.food_workers,
    7   player.idle_worker_co
    8   player.army_count,
    9   player.warp_gate_coun
    10  player.larva_count]

    :param obs:
    :return:
    """
    state = np.concatenate(
      obs.observation["player"][:9])
    return state

  def _unit_state(self, obs):
    """
    raw_units -> features.FeatureUnit

    :param obs:
    :return:
    """
    state = np.concatenate((
      obs.observation["raw_units"][0],
      obs.observation["raw_units"][2],
      obs.observation["raw_units"][5],
      obs.observation["raw_units"][6],
      obs.observation["raw_units"][11],
      obs.observation["raw_units"][12],
      obs.observation["raw_units"][13],
      obs.observation["raw_units"][17]
    ))


  def _custom_state(self, obs):
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

