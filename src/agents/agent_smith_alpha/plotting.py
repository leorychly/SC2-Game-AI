import numpy as np
import matplotlib.pyplot as plt


def plot_progress(logging_data, save_dir):
  # TODO: Check why test plot shows 0% Loss rate at beginning
  n = 10 if len(logging_data) >= 10 else len(logging_data)
  game_results = [logging_data[i]["game_result"] for i in range(len(logging_data))]
  win_rates = [game_results[:i].count(1) / (i + 1) for i in range(n)]
  draw_rates = [game_results[:i].count(0) / (i + 1) for i in range(n)]
  loss_rates = [game_results[:i].count(-1) / (i + 1) for i in range(n)]
  win_rates.extend([game_results[i:i + n].count(1) / n for i in range(len(game_results) - n)])
  draw_rates.extend([game_results[i:i + n].count(0) / n for i in range(len(game_results) - n)])
  loss_rates.extend([game_results[i:i + n].count(-1) / n for i in range(len(game_results) - n)])

  fig1 = plt.figure(constrained_layout=True)
  gs1 = fig1.add_gridspec(4, 4)
  f1_ax1 = fig1.add_subplot(gs1[:3, :])
  f1_ax1.set_title("Win Rate")
  f1_ax1.set_xlabel("Games Played")
  f1_ax1.set_ylabel("Win rate over the last 10 Games")
  f1_ax1.plot(np.arange(len(game_results)), win_rates, "g", label="Win %", alpha=0.7)
  f1_ax1.plot(np.arange(len(game_results)), draw_rates, "orange", label="Draw %", alpha=0.7)
  f1_ax1.plot(np.arange(len(game_results)), loss_rates, "r", label="Loss %", alpha=0.7)
  f1_ax1.legend()

  game_results = [logging_data[i]["game_result"] for i in range(len(logging_data))]
  steps = [logging_data[i]["game_length"] for i in range(len(logging_data))]
  total_steps = np.sum(steps)
  y_ticks = np.cumsum(steps)
  c_map = {1: "green", 0: "orange", -1: "red"}
  c = [c_map[outcome] for outcome in game_results]

  f1_ax2 = fig1.add_subplot(gs1[3, :])
  f1_ax2.set_title("Games Outcomes")
  f1_ax2.set_xlabel("Time Steps over all Games")
  f1_ax2.set_ylabel("Game Result")
  # f1_ax2.set_xticks([-1, 0, 1])
  # f1_ax2.set_yticks(y_ticks)
  f1_ax2.set_xlim([0, total_steps + 1])
  f1_ax2.set_ylim([-1.5, 1.5])
  f1_ax2.scatter(y_ticks, game_results, color=c, alpha=0.85, s=10)

  fig1.savefig(save_dir)
  plt.close(fig1)


def plot_training_progress(losses, save_dir):
  fig = plt.figure()
  plt.plot(np.arange(len(losses)), losses)
  plt.title("Q-Network Training Loss")
  plt.xlabel("Updates")
  plt.ylabel("MSE TD Error")
  plt.yscale("log")
  fig.savefig(save_dir)
  plt.close(fig)


if __name__ == '__main__':
  data= [{"game_result": -1, "game_length": 10},
         {"game_result": -1, "game_length": 10},
         {"game_result": 0, "game_length": 20},
         {"game_result": -1, "game_length": 10},
         {"game_result": 0, "game_length": 20},
         {"game_result": 1, "game_length": 30},
         {"game_result": 0, "game_length": 20},
         {"game_result": 1, "game_length": 30},
         {"game_result": 1, "game_length": 30},
         {"game_result": 0, "game_length": 30},
         {"game_result": 1, "game_length": 30},
         {"game_result": 1, "game_length": 30},
         {"game_result": 1, "game_length": 30},
         {"game_result": 0, "game_length": 30},
         {"game_result": 1, "game_length": 30},
         {"game_result": 1, "game_length": 30}]
  plot_progress(logging_data=data, save_dir="./test.png")
