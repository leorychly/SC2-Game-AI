import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
  def __init__(self,
               n_actions,
               state_space,
               device,
               n_quant=64):
    super(ConvNet, self).__init__()
    self.device = device
    self.n_quant = n_quant

    self.feature_extraction = nn.Sequential(
      # Conv2d(channels, channels, kernel_size, stride)
      nn.Conv2d(state_space[-1], 32, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=4, stride=1),
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=2, stride=1),
      nn.ReLU()
    )
    self.feature_size = 3584
    # self.phi = nn.Linear(1, 7 * 7 * 64, bias=False)
    # self.phi_bias = nn.Parameter(torch.zeros(7 * 7 * 64))
    self.phi = nn.Linear(1, self.feature_size, bias=False)
    self.phi_bias = nn.Parameter(torch.zeros(self.feature_size))
    self.fc = nn.Linear(self.feature_size, 512)

    # action value distribution
    self.fc_q = nn.Linear(512, n_actions)

    # Initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0.0)
      elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0.0)

  def forward(self, x):
    # x.size(0) : minibatch size
    mb_size = x.size(0)
    x = self.feature_extraction(x.permute(0, 3, 1, 2).float() / 255.0)  # (m, 7 * 7 * 64)
    # Rand Initlialization
    tau = torch.rand(self.n_quant, 1)  # (N_QUANT, 1)
    # Quants=[1,2,3,...,N_QUANT]
    quants = torch.arange(0, self.n_quant, 1.0)  # (N_QUANT,1)
    tau = tau.to(self.device)
    quants = quants.to(self.device)
    # phi_j(tau) = RELU(sum(cos(π*i*τ)*w_ij + b_j))
    cos_trans = torch.cos(quants * tau * 3.141592).unsqueeze(2)  # (N_QUANT, N_QUANT, 1)
    rand_feat = F.relu(self.phi(cos_trans).mean(dim=1) + self.phi_bias.unsqueeze(0)).unsqueeze(0)
    # (1, N_QUANT, 7 * 7 * 64)
    x = x.view(x.size(0), -1).unsqueeze(1)  # (m, 1, 7 * 7 * 64)
    # Zτ(x,a) ≈ f(ψ(x) @ φ(τ))a
    x = x * rand_feat  # (m, N_QUANT, 7 * 7 * 64)
    # x.shape=(32,1,22528), rand_feat.shape=(1,64,3136)
    x = F.relu(self.fc(x))  # (m, N_QUANT, 512)

    # note that output of IQN is quantile values of value distribution
    action_value = self.fc_q(x).transpose(1, 2)  # (m, N_ACTIONS, N_QUANT)
    return action_value, tau

  def save(self, PATH):
    torch.save(self.state_dict(), PATH)

  def load(self, PATH):
    self.load_state_dict(torch.load(PATH))