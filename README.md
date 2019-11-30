# Agent Smith Playing StarCraft 2

## Challenges
* Huge state and action spaces
* Long term and delayed rewards
* Hierarchical strategy structure
* Limited computing resources

## Problem Formulation
For building a StarCraft2 bot the formulation of the decision making problem is especially important if learning algorithms are applied in environments with large state and action spaces as well as possibly very sparse rewards. 
Ideally, the agent should only be rewarded for winning or losing games, since this is the single objective of StarCraft2.
However, since typical games range from 5 to 15 minutes with approximately 100 action per minute, the rewards become extremely sparse and in combination with the large state and action space
the agent is unlikely to converge to any useful optima. 

One crucial factor is the design of a suitable __state representation__ that encodes what an action has actually achieved. 
Otherwise, we introduce additional hidden variables (additional to the opponents actions) that a Q function cannot learn about because it can never be in the state s for Q(s,a). 
For Reinforcement Learning algorithms to be reliable, the state value s has to encode all relevant information about how future states and rewards will progress.

So different state observer creating different state representations will be implemented:
* (1) A simple handcrafted state which encodes all buildings, units, and the disposable resources.
* (2) Raw "sensor" data containing the the feature screens provided by PySC2.
* (3) A combination of a latent state representation directly learned from the raw game input with the handcrafted state vector.

## Agents

#### Agent Structure
<img src="https://github.com/LeRyc/SC2-Game-AI/blob/master/docs/img/general_structure.png" height="350">


#### Agent Decision Making
* AgentSmith: Scripted agent
* AgentSmithAlpha: Q-Tabel
* AgentSmithBeta: Q-Learning
* ...many more to come
