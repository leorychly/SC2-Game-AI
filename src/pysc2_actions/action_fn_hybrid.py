""""
Actions to adapt to use numeric inputs as x,y coords, number of units to train:
      #action_fn_categorical.build_supply_depot,
      #action_fn_categorical.build_barracks,
      #action_fn_categorical.train_marine,
      #action_fn_categorical.attack
"""
import numpy as np
from pysc2.lib import actions, features, units
from src.pysc2_interface.interface import Interface


def build_supply_depot(world_state, max_count=2):
  obs = world_state.obs
  pos = (world_state.kwargs["x"], world_state.kwargs["y"])
  supply_depots = Interface.get_units_by_type(
    obs, units.Terran.SupplyDepot, enemy=False)
  scvs = Interface.get_units_by_type(obs, units.Terran.SCV, enemy=False)
  if (len(supply_depots) < max_count
      and obs.observation.player.minerals >= 100
      and len(scvs) > 0):
    distances = Interface.get_distances(obs, scvs, pos)
    scv = scvs[np.argmin(distances)]
    return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
      "now", scv.tag, pos)
  return actions.RAW_FUNCTIONS.no_op()


def build_barracks(world_state, max_count=2):
  obs = world_state.obs
  pos = (world_state.kwargs["x"], world_state.kwargs["y"])
  completed_supply_depots = Interface.get_completed_units_by_type(
    obs, units.Terran.SupplyDepot, enemy=False)
  barrackses = Interface.get_units_by_type(
    obs, units.Terran.Barracks, enemy=False)
  scvs = Interface.get_units_by_type(obs, units.Terran.SCV, enemy=False)
  if (len(completed_supply_depots) > 0
      and len(barrackses) < max_count
      and obs.observation.player.minerals >= 150
      and len(scvs) > 0):
    distances = Interface.get_distances(obs, scvs, pos)
    scv = scvs[np.argmin(distances)]
    return actions.RAW_FUNCTIONS.Build_Barracks_pt(
      "now", scv.tag, pos)
  return actions.RAW_FUNCTIONS.no_op()


def train_marine(world_state):
  obs = world_state.obs
  completed_barrackses = Interface.get_completed_units_by_type(
    obs, units.Terran.Barracks, enemy=False)
  free_supply = (obs.observation.player.food_cap -
                 obs.observation.player.food_used)
  if (len(completed_barrackses) > 0
      and obs.observation.player.minerals >= 100
      and free_supply > 0):
    barracks = Interface.get_units_by_type(
      obs, units.Terran.Barracks, enemy=False)[0]
    if barracks.order_length < 5:
      return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
  return actions.RAW_FUNCTIONS.no_op()


def attack(world_state):
  obs = world_state.obs
  pos = (world_state.kwargs["x"], world_state.kwargs["y"])
  marines = Interface.get_units_by_type(obs, units.Terran.Marine, enemy=False)
  if len(marines) > 0:
    distances = Interface.get_distances(obs, marines, pos)
    marine = marines[np.argmax(distances)]
    return actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag, pos)
  return actions.RAW_FUNCTIONS.no_op()
