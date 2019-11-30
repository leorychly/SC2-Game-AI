from absl import app
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop

from src.agents.agent_smith_alpha.agent_alpha import AgentSmithAlpha
from src.agents.random_agent import RandomAgent


def main(unused_argv):
  agent1 = AgentSmithAlpha()
  agent2 = RandomAgent()
  try:
    with sc2_env.SC2Env(
      map_name="Simple64",
      players=[sc2_env.Agent(sc2_env.Race.terran),
               sc2_env.Agent(sc2_env.Race.terran)],
      agent_interface_format=features.AgentInterfaceFormat(
        action_space=actions.ActionSpace.RAW,
        use_raw_units=True,
        raw_resolution=64,
      ),
      step_mul=48,
      disable_fog=True,
    ) as env:
      run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)
