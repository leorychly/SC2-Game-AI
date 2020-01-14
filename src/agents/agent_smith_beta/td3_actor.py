import numpy as np
from absl import logging

import torch
import torch.nn as nn
from torchsummary import summary


class Actor(nn.Module):

  def __init__(self, img_state_dim,
               vect_state_len,
               action_space_dim,
               device):
    """
    Create the actor network of the TD3 Algorithm.

    :param img_state_dim: Tuple
      Number of channels of the image input tensor at last place. (h,w,c)
    :param vect_state_len: Int
      Size of th semantic state input vector.
    :param action_space_dim: Int
      Shape of the action space.
      E.g. for a combination of a 10-Action 1-hot encoding + 2 Regression
      outputs, the action_space_dim would be of size 12.
    """
    super(Actor, self).__init__()
    self.device = device
    conv_output_dim = 32 * 4 * 4

    self.conv_modules = nn.Sequential(
      nn.Conv2d(img_state_dim[-1], 32, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3),
      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2))
    input_shape = img_state_dim[2], img_state_dim[0], img_state_dim[1]
    print(f"\nActor:"
          f"\n\tQ-Network Conv Input Shape {input_shape}, "
          f"\n\tConv Output: {conv_output_dim}")
    summary(self.conv_modules, input_shape, device="cpu")

    self.dense_modules = nn.Sequential(
      nn.Linear(conv_output_dim + vect_state_len[0], 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, action_space_dim),
      nn.Sigmoid())
    print(f"\nActor Dense Input Shape {input_shape}, "
          f"\n\tConv Output: {conv_output_dim}")
    summary(self.dense_modules,
            (conv_output_dim + vect_state_len[0],),
            device="cpu")
    logging.info("Actor initialized")

  def forward(self, x):
    """
    Forward pass of the network.

    Transorm input x from (B, (C, H, W), (N))
    to (B, C, H, W) and (B, N)

    :param x:
      Tuple of (Image, Data) as ndarray or torch.tensor
      - Image input data of shape (B x C x H x W) with
            B: batch size
            C: number of channels
            H: hight of the input data
            W  width of the input data
      - Data input data as a vector of shape (B, N)
            B: batch size
            N: array length
    :return x:
      Network output.
    """
    #x_img = np.stack(x[0])
    #x_data = np.stack(x[1])

    if isinstance(x[0], np.ndarray) and isinstance(x[1], np.ndarray):
      x_img = torch.from_numpy(x[0]).float().to(device=self.device)
      #x_img = x_img.permute(0, 3, 1, 2)
      x_data = torch.from_numpy(x[1]).float().to(device=self.device)
    else:
      x_img = x[0]
      x_data = x[1]

    x_img = self.conv_modules(x_img)
    x = torch.cat((x_img.reshape(x_img.size(0), -1),
                   x_data.reshape(x_img.size(0), -1)), dim=1)
    x = self.dense_modules(x)
    return x
