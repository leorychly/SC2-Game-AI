import random
from collections import deque
from absl import logging
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch import optim

from pysc2.lib import actions, features, units

from src.observer.base_observer import BaseObserver
from src.observer.conv_autoencoder import ConvAutoencoder
from src.pysc2_interface.interface import Interface
from src.pysc2_actions.actions import Actions


class AutoencoderObserver(BaseObserver):
  """An observer containing raw screen data as well as semantic parameter.

  The screen data is encoded using a autoencoder and concatinated with the
  semantic state definition.
  """

  def __init__(self, device, buffer_size):
    super(AutoencoderObserver, self).__init__()
    self.input_dim = ((64, 64, 7), 3)
    self.output_dim = (10,)
    self.interface = Interface()
    self.actions = Actions()
    self.autoencoder = ConvAutoencoder(self.input_dim[0]).to(device)
    self.ae_loss_fn = nn.MSELoss()
    self.ae_optimizer = optim.Adam(self.autoencoder.parameters(),
                                   weight_decay=1e-5)
    self.buffer = deque(maxlen=buffer_size)
    self.ae_training_setp = 0
    self.ae_losses = []
    self.ae_training_plot_dir = "./ae_training.png"

  @property
  def shape(self):
    return self.output_dim

  def fit_autoencoder(self, batch_size):
    batch = random.sample(self.buffer, batch_size)
    batch_reconstructed = self.autoencoder(batch)
    loss = self.ae_loss_fn(batch, batch_reconstructed)
    self.ae_losses.append(loss)
    self.ae_optimizer.zero_grad()
    self.autoencoder.eval()
    loss.backward()
    self.ae_optimizer.step()
    self.autoencoder.test()
    logging.info(f"Autoencoder training step {self.ae_training_setp}, "
                 f"loss:{loss.data.to_n():.4f}")
    self.plot_loss()
    self.ae_training_setp += 1
    return loss

  def player_info(self, obs):
    player_info = obs.observation["player"]  # (11,)

  def get_state(self, obs):
    semantic_state = self._get_semantic_state(obs)
    autoencoder_state = self._get_autoencoder_state(obs)
    return np.concatenate(semantic_state, autoencoder_state)

  def _get_autoencoder_state(self, obs):
    sc2_feature_maps = obs.observation[0]  # TODO: add feature maps
    state = self.autoencoder.encode(sc2_feature_maps)
    state = state.detach().to_numpy()
    return state

  def _get_semantic_state(self, obs):
    scvs, idle_scvs, command_centers, completed_command_centers, \
    supply_depots, completed_supply_depots = self.get_basic_info(obs)

    barrackses = self.interface.get_units_by_type(
      obs, units.Terran.Barracks, enemy=False)
    completed_barrackses = self.interface.get_completed_units_by_type(
      obs, units.Terran.Barracks, enemy=False)

    marines, queued_marines = self.get_marine_info(obs)

    free_supply, can_afford_supply_depot, can_afford_barracks, \
    can_afford_marine = self.get_eco_info(obs)

    enemy_scvs, enemy_idle_scvs, enemy_command_centers, \
    enemy_supply_depots, enemy_completed_supply_depots, \
    enemy_barrackses, enemy_completed_barrackses, \
    enemy_marines = self.get_enemy_info(obs)

    state = (len(command_centers),
            len(scvs),
            len(idle_scvs),
            len(supply_depots),
            len(completed_supply_depots),
            len(barrackses),
            len(completed_barrackses),
            len(marines),
            queued_marines,
            free_supply,
            can_afford_supply_depot,
            can_afford_barracks,
            can_afford_marine,
            len(enemy_command_centers),
            len(enemy_scvs),
            len(enemy_idle_scvs),
            len(enemy_supply_depots),
            len(enemy_completed_supply_depots),
            len(enemy_barrackses),
            len(enemy_completed_barrackses),
            len(enemy_marines))
    return state

  def plot_loss(self):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(len(self.ae_losses)), self.ae_losses)
    ax.set_title("Autoencoder Training")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Loss (MSE)")
    plt.savefig(fig, self.ae_training_plot_dir)



