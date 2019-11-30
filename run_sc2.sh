#!/usr/bin/env bash

python -m pysc2.bin.agent \
--map Simple64 \
--agent src.run_sc2.SimpleAgent \
--agent_race terran