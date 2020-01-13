import time
from absl import app
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop

from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Bot, Agent
from pysc2.env.sc2_env import Race, Difficulty, Dimensions

from src.agents.agent_smith_alpha.alpha_agent import AgentSmithAlpha
from src.agents.agent_smith_beta.beta_agent import AgentSmithBeta
from src.agents.random_agent import RandomAgent


def main(unused_argv):
  # RandomAgent()
  # Bot(Race.terran, Difficulty.easy)
  # AgentSmithAlpha()
  # AgentSmithBeta()

  screen_dim = (64, 64)
  #player_1 = AgentSmithAlpha()
  player_1 = AgentSmithBeta(screen_dim)
  player_2 = RandomAgent()

  players = [player_1, player_2]
  interface = AgentInterfaceFormat(
    feature_dimensions=Dimensions(screen=screen_dim,
                                  minimap=screen_dim),
    action_space=actions.ActionSpace.RAW,
    use_raw_units=True,
    raw_resolution=64,
  )
  # raw_resolution=64

  try:
    with sc2_env.SC2Env(
      map_name="Simple64",
      players=[sc2_env.Agent(sc2_env.Race.terran),
               sc2_env.Agent(sc2_env.Race.terran)],
      agent_interface_format=interface,
      step_mul=15,
      realtime=False,
      disable_fog=True,
      score_index=-1,
      random_seed=None,
      visualize=False) as env:
      run_loop.run_loop(players, env, max_episodes=10000)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)
