import gym
import time
import numpy as np
from collections import deque
from src.agents.agent_smith_gamma.iqn.implicit_quantile_agent import IQNAgent


STATE_LEN = 4  # sequential images to define state
LEARN_FREQ = 4  # simulator steps for learning interval
N_QUANT = 64  # simulator steps for learning interval
QUANTS = np.linspace(0.0, 1.0, N_QUANT + 1)[1:]  # quantiles
N_ENVS = 16  # number of environments for C51
STEP_NUM = int(1e+8)  # Total simulation step

ENV_NAME = 'BreakoutNoFrameskip-v4'
env = gym.make(ENV_NAME)
n_actions = env.action_space.n
state_space = env.observation_space.shape

n_steps_before_training = int(1e+3)

agent = IQNAgent(state_space=state_space,
                 action_dim=n_actions,
                 device="cuda",
                 n_steps_before_training=n_steps_before_training)


print('Collecting experience...')

# episode step for accumulate reward
epinfobuf = deque(maxlen=100)
# check learning time
start_time = time.time()
result = []

# env reset
s = np.array(env.reset())

for i in range(STEP_NUM):
  if i == n_steps_before_training:
    print("Done with random runs. Start Training")
  if i < n_steps_before_training:
    a = agent.choose_action(s, is_random=True)
  else:
    a = agent.choose_action(s, is_random=False)

  #time.sleep(0.01)
  #env.render("human")

  s_next, r, done, info = env.step(a)
  s_next = np.asarray(s_next)
  clip_r = np.sign(r)  # clip rewards for numerical stability

  maybeepinfo = info.get('episode')
  if maybeepinfo:
    epinfobuf.append(maybeepinfo)

  agent.step(state=s,
             action=a,
             reward=clip_r,
             next_state=s_next,
             done=done)


  agent.to_tensorboard(var=r, name="Reward")
  agent.to_tensorboard(var=clip_r, name="Reward (Clipped)")
  s = s_next

  if i % 1000 == 0:
    mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 4)
    result.append(mean_100_ep_return)
    print(f"Step {i}\t|\tMean (100 Ep) Reward {mean_100_ep_return}"
          f"\t|\tEpsilon {agent.epsilon:.3f}")
    agent.to_tensorboard(var=mean_100_ep_return, name="Mean (100 Ep) Reward")

print("The training is done!")