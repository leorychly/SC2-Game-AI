""""
Actions to adapt to use numeric inputs as x,y coords, number of units to train:
      #action_fn_categorical.build_supply_depot,
      #action_fn_categorical.build_barracks,
      #action_fn_categorical.train_marine,
      #action_fn_categorical.attack
"""

def build_supply_depot(world_state):
  obs = world_state.obs
  base_top_left = world_state.kwargs["base_top_left"]
  supply_depots = Interface.get_units_by_type(
    obs, units.Terran.SupplyDepot, enemy=False)
  scvs = Interface.get_units_by_type(obs, units.Terran.SCV, enemy=False)
  if (len(supply_depots) < 2 and obs.observation.player.minerals >= 100 and
    len(scvs) > 0):
    if len(supply_depots) == 0:
      supply_depot_xy = (22, 26) if base_top_left else (35, 42)
    elif len(supply_depots) == 1:
      supply_depot_xy = (18, 28) if base_top_left else (38, 40)
    else:
      supply_depot_xy = (20, 28) if base_top_left else (33, 44)
    distances = Interface.get_distances(obs, scvs, supply_depot_xy)
    scv = scvs[np.argmin(distances)]
    return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
      "now", scv.tag, supply_depot_xy)
  return actions.RAW_FUNCTIONS.no_op()


def build_barracks(world_state):
  obs = world_state.obs
  base_top_left = world_state.kwargs["base_top_left"]
  completed_supply_depots = Interface.get_completed_units_by_type(
    obs, units.Terran.SupplyDepot, enemy=False)
  barrackses = Interface.get_units_by_type(
    obs, units.Terran.Barracks, enemy=False)
  scvs = Interface.get_units_by_type(obs, units.Terran.SCV, enemy=False)
  if (len(completed_supply_depots) > 0 and len(barrackses) < 2 and
    obs.observation.player.minerals >= 150 and len(scvs) > 0):
    if len(barrackses) == 0:
      barracks_xy = (22, 21) if base_top_left else (35, 45)
    else:
      barracks_xy = (26, 21) if base_top_left else (30, 45)
    distances = Interface.get_distances(obs, scvs, barracks_xy)
    scv = scvs[np.argmin(distances)]
    return actions.RAW_FUNCTIONS.Build_Barracks_pt(
      "now", scv.tag, barracks_xy)
  return actions.RAW_FUNCTIONS.no_op()


def train_marine(world_state):
  obs = world_state.obs
  completed_barrackses = Interface.get_completed_units_by_type(
    obs, units.Terran.Barracks, enemy=False)
  free_supply = (obs.observation.player.food_cap -
                 obs.observation.player.food_used)
  if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
    and free_supply > 0):
    barracks = Interface.get_units_by_type(
      obs, units.Terran.Barracks, enemy=False)[0]
    if barracks.order_length < 5:
      return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
  return actions.RAW_FUNCTIONS.no_op()


def attack(world_state):
  obs = world_state.obs
  base_top_left = world_state.kwargs["base_top_left"]
  marines = Interface.get_units_by_type(obs, units.Terran.Marine, enemy=False)
  if len(marines) > 0:
    attack_xy = (38, 44) if base_top_left else (19, 23)
    distances = Interface.get_distances(obs, marines, attack_xy)
    marine = marines[np.argmax(distances)]
    x_offset = random.randint(-4, 4)
    y_offset = random.randint(-4, 4)
    return actions.RAW_FUNCTIONS.Attack_pt(
      "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
  return actions.RAW_FUNCTIONS.no_op()