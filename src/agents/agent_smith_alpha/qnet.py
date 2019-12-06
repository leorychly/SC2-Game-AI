import torch.nn as nn


class QNet(nn.Module):

  def __init__(self, state_dim, action_dim):
    """
    Init the Q-Network: Q(s) = r_a.

    The Q-Net returns the expected reward for all actions at the current time
    step.

    :param state_dim:
    The number of channel of input.
    i.e The number of most recent frames stacked together.

    :param action_dim:
    Number of action-values to output, one-to-one correspondence to actions in
    game.

    """
    super(QNet, self).__init__()
    self.module_list = self._create_network(state_dim, action_dim)

  def _create_network(self, state_dim, action_dim):
    module_list = nn.Sequential(
      nn.Linear(in_features=state_dim, out_features=256),
      nn.ReLU(),
      nn.Linear(in_features=256, out_features=256),
      nn.ReLU(),
      nn.Linear(in_features=256, out_features=256),
      nn.ReLU(),
      nn.Linear(in_features=256, out_features=action_dim))
    return module_list

  def forward(self, x):
    for layer in self.module_list:
      x = layer(x)
    x = x.clone()
    return x
