import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ConvAutoencoder(nn.Module):
  """Convolutional Autoencoder for creating game state representation."""

  def __init__(self, input_shape):
    """

    :param input_shape: tuple
      (H, W, Channels)
    """
    super(ConvAutoencoder, self).__init__()
    hidden_dim = 1024

    self.encoder = nn.Sequential(
      nn.Conv2d(input_shape[-1], 32, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5),
      nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=4),
      nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=4))
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5),
      nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=4),
      nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=4))

    print(f"Autoencoder input shape {input_shape}, "
          f"Hidden state size: {hidden_dim}")
    summary(self.modules, input_shape, device="cpu")

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

  def encode(self, x):
    h = self.encoder(x)
    return h
