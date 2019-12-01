import time
import gym
import unittest
import numpy as np
import torch

from src.agents.dqn.dqn_agent import DQNAgent


class DQNAgentTest(unittest.TestCase):

  def test_dqn_agent(self):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    env_id = "CartPole-v1"
    buffer_size = int(1e5)
    batch_size = 128
    gamma = 0.99
    tau = 1e-3
    lr = 1e-4
    training_interval = 2
    n_episodes = 10000
    max_ep_steps = 1000
    epsilon = 0.999
    epsilon_decay = 0.99995
    epsilon_min = 0.01
    logging_interval = 100
    layer_param = [
          {"type": "linear", "n_neurons": [0, 128]},
          {"type": "relu"},
          {"type": "linear", "n_neurons": [128, 128]},
          {"type": "relu"},
          {"type": "linear", "n_neurons": [128, 0]}
        ]

    env = gym.make(env_id)
    agent = DQNAgent(state_dim=env.observation_space.shape[0],
                     action_dim=env.action_space.n,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     gamma=gamma,
                     tau=tau,
                     lr=lr,
                     training_interval=training_interval,
                     epsilon=epsilon,
                     epsilon_decay=epsilon_decay,
                     epsilon_min=epsilon_min,
                     layer_param=layer_param,
                     device=device)

    t0 = time.time()
    overall_results = []
    for episode in range(n_episodes):
      state = env.reset()
      ep_steps = []

      for step in range(max_ep_steps):
        action = agent.plan(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state

        if done or step >= max_ep_steps - 1:
          ep_steps.append(step + 1.)

          if episode % logging_interval == 0:
            avrg_steps = np.mean(np.asarray(ep_steps))
            overall_results.append(avrg_steps)
            print(f"Ep {episode},\tAvrg steps reached {avrg_steps},"
                  f"\tEpsilon: {agent.epsilon:.2f},\ttook {time.time() - t0:.2f} sec")
            t0 = time.time()

          break

    assert 500.0 in overall_results


if __name__ == '__main__':
  unittest.main()
