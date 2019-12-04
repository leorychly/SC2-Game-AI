import numpy as np
import matplotlib.pyplot as plt


def plot_progress(data, save_dir):
  n = 10 if len(data) >= 10 else len(data)
  game_results = [data[i]["game_result"] for i in range(len(data))]
  game_results = [0 if x is None else x for x in game_results] # TODO: Check why reward can be None

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

  game_results = [data[i]["game_result"] for i in range(len(data))]
  steps = [data[i]["game_length"] for i in range(len(data))]
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


def plot_training_progress(losses, epsilons, save_dir):
  fig, ax1 = plt.subplots()
  ax1 = fig.add_subplot()
  ax1.set_title("Q-Network Training Loss")
  ax1.set_xlabel("Updates")
  ax1.set_ylabel("MSE TD Error")
  ax1.set_yscale("log")
  ax1.plot(np.arange(len(losses)), losses)
  ax2 = ax1.twinx()
  ax2.set_ylabel("Epsilon")
  ax2.plot(np.arange(len(losses)), epsilons, c="grey", alpha=0.75)
  fig.savefig(save_dir)
  plt.close(fig)


def plot_action_histogram(data, save_dir):
  a_lst = [d["actions_taken"] for d in data]
  a_lst = np.asarray(a_lst)
  sum_a_lst = np.sum(a_lst, axis=1)
  normalized_a_list = a_lst / sum_a_lst[:, None]
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.set_title("Actions Chosen during Games")
  ax.set_xlabel("Games Played")
  ax.set_ylabel("Action Chosen [%}")
  for a_nr in range(a_lst.shape[1]):
    ax.plot(np.arange(a_lst.shape[0]), normalized_a_list[:, a_nr],
            label=f"Action {a_nr}", alpha=0.75)
  ax.legend()
  fig.savefig(save_dir)
  plt.close(fig)


def test_plot_action_histogram():
  data = [{"actions_taken": [10,4,2]},
          {"actions_taken": [2,1,1]}]
  plot_action_histogram(data, "./test_actions.png")


def test_plot_progress():
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


if __name__ == '__main__':
  test_plot_progress()
  test_plot_action_histogram()
