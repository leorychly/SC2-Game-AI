<img align="right" src="https://github.com/LeRyc/SC2-Game-AI/blob/master/docs/img/agent_smith.png" height="200">

# Agent Smith Playing StarCraft 2


## Challenges
* Huge state and action spaces
* Long term and delayed rewards
* Hierarchical strategy structure
* Limited computing resources

## Problem Formulation
For building a StarCraft2 bot the correct formulation of the decision making and learning problem is not straigt forward. Especially environments with large state and action spaces as well as very sparse rewards require a precise game state definition for the agent to observe the correct evolution of the state based on the chosen rewards.  

A suitable __state representation__ should, therefore, encode the long-term consequences of actions. Otherwise, we introduce additional hidden variables (additional to the opponents actions) that a (Action-)Value Function cannot learn because it never is in state s for Q(s,a). 
Therefore, the state value s has to encode all relevant information about how future states and rewards will progress for Reinforcement Learning algorithms to be reliable.

Ideally, the agent should only be rewarded for winning or losing games, since this is the single objective of StarCraft2. However, since typical games range from 5 to 15 minutes with approximately 100 action per minute, the rewards become extremely sparse and in combination with the large state and action space the agent is likely to converge to suboptimal extrema.

So different state observer creating different state representations will be implemented:
* (1) A simple handcrafted state which encodes all buildings, units, and the disposable resources.
* (2) Raw "sensor" data containing the the feature screens provided by PySC2.
* (3) A combination of a latent state representation directly learned from the raw game input with the handcrafted state vector.

---

## Agents

#### Agent Structure
<img src="https://github.com/LeRyc/SC2-Game-AI/blob/master/docs/img/general_structure.png" height="350">


#### Agent Decision Making
* AgentSmith: Random / Scripted Agent
* AgentSmithAlpha: Double Deep Q-Learning Agent with restricted action space
* ... more to come

---

## Install and Run
To create a virtualenv and install all required libraries run ```install.sh```

To start the training of the DDQN Agent vs a Random Agent run ```python run_sc2.py```

To start the training of the DDQN Agent vs a standard StarCraft Bot run ```python run_sc2.sh```

---

# Results
### DQN with semantic state and limited action space
Because of the limited action space a random policy is already very good since the necessary action like building barracks, supply depots and training marines are executed regularly. 
* __State Space__: custom vector (src/observer/crafted_observer.py) of size 21. 
```
[#CommandCenter, #SCVs, #IdleS_CVs, #Depots, #Completed_Depots, #Barracks, #Completed_Barracks, #Marines, #Queued_Marines, Free_Supply, BOOL_can_afford_depot, BOOL_can_afford_barracks, BOOL_can_afford_marine, #Enemy_Command_Centers, #Enemy_SCVs, #Enemy_Idle_SCVs, #Enemy_Depots, #Enemy_Completed_Depots, #Enemy_Barracks, #Enemy_Completed_Barracks, #Enemy_Marines]
```
* __Action Space__: Custom actions of size 6.
	* Do nothing
	* Harvesting minirals
	* Build supply depot (up to two)
	* Build barracks (up to two)
	* Train marine
	* Attack with one marine
