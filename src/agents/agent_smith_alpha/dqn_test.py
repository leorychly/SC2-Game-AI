import time
import gym
import unittest
import numpy as np

from src.agents.agent_smith_alpha.dqn import DQNAgent


class DQNAgentTest(unittest.TestCase):

  def run_lunarlander(self,
                      buffer_size=int(1e5),
                      batch_size=64,
                      gamma=0.99,
                      tau=1e-3,
                      lr=1e-5,
                      training_interval=1,
                      n_episodes=10000,
                      max_ep_steps=500,
                      epsilon=0.999,
                      epsilon_decay=0.9999,
                      epsilon_min=0.01,
                      logging_interval=50):
    device = "cpu"
    env_id = "LunarLander-v2"

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
                     device=device)

    t0 = time.time()
    overall_results = []
    for episode in range(n_episodes):
      state = env.reset()
      ep_reward = 0

      for step in range(max_ep_steps):
        action = agent.plan(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward

        if done or step >= max_ep_steps - 1:
          if episode % logging_interval == 0 or episode == n_episodes - 1:
            overall_results.append(ep_reward)
            print(f"Ep {episode},"
                  f"\tTotal Reward {ep_reward:.3f}, "
                  f"\tSteps: {step + 1}, "
                  f"\tEpsilon: {agent.epsilon:.2f},"
                  f"\ttook {time.time() - t0:.2f} sec")
            t0 = time.time()
          break
    return overall_results

  def run_cartpole(self,
                   buffer_size=int(1e5),
                   batch_size=64,
                   gamma=0.99,
                   tau=1e-3,
                   lr=1e-5,
                   training_interval=1,
                   n_episodes=1000,
                   max_ep_steps=300,
                   epsilon=0.999,
                   epsilon_decay=0.9999,
                   epsilon_min=0.01,
                   logging_interval=50):
    device = "cpu"
    env_id = "CartPole-v1"


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

          if episode % logging_interval == 0 or episode == n_episodes - 1:
            avrg_steps = np.mean(np.asarray(ep_steps))
            ep_steps = []
            overall_results.append(avrg_steps)
            print(f"Ep {episode},"
                  f"\tAvrg steps reached {avrg_steps} / {max_ep_steps},"
                  f"\tEpsilon: {agent.epsilon:.2f},"
                  f"\ttook {time.time() - t0:.2f} sec")
            t0 = time.time()

          break
    return overall_results

  def test_dqn_agent(self):
    print(f"\nRun Training Test!\n")
    self.run_cartpole(batch_size=64,
                      lr=1e-5,
                      n_episodes=10000,
                      max_ep_steps=500)

  def test_gridsearch(self):
    batch_sizes = [64, 128, 256]
    lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    for batch_size in batch_sizes:
      for lr in lrs:
        for _ in range(2):
          overall_results = self.run_cartpole(batch_size=batch_size, lr=lr)
          print(f"\n=> Batch Size {batch_size}, LR {lr}: {overall_results}\n")

    #assert 500.0 in overall_results


if __name__ == '__main__':
  unittest.main()
