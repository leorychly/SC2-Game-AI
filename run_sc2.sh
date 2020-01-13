#!/usr/bin/env bash

#python -m pysc2.bin.agent \
#--map Simple64 \
#--agent src.agents.agent_smith_alpha.alpha_agent.AgentSmithAlpha \
#--agent_race terran \
#--max_agent_steps 0 \
#--norender

for i in {1..100}
do
   python run_sc2.py
done
