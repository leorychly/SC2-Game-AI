import numpy as np
from absl import logging

import torch
import torch.nn as nn
from torchsummary import summary


class Actor(nn.Module):

  def __init__(self, img_state_channel_dim, vect_state_len, action_space_dim):
    """
    Create the actor network of the TD3 Algorithm.

    :param img_state_channel_dim: Int
      Number of channels of the image input tensor.
    :param vect_state_len: Int
      Size of th semantic state input vector.
    :param action_space_dim: Int
      Shape of the action space.
      E.g. for a combination of a 10-Action 1-hot encoding + 2 Regression
      outputs, the action_space_dim would be of size 12.
    """
    super(Actor, self).__init__()

    input_shape = (3, 7, 7)
    conv_output_dim = 16 * 2 * 2

    self.conv_modules = nn.Sequential(
      nn.Conv2d(img_state_channel_dim, 32, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5),
      nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5),
      nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5))
    print(f"Q-Network Conv Input Shape {input_shape}, "
          f"Conv Output: {conv_output_dim}")
    summary(self.conv_modules, input_shape, device="cpu")

    self.dense_modules = nn.Sequential(
      nn.Linear(conv_output_dim + vect_state_len, 128),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, action_space_dim))
    print(f"Actor Dense Input Shape {input_shape}, "
          f"Conv Output: {conv_output_dim}")
    summary(self.dense_modules, conv_output_dim + vect_state_len, device="cpu")
    logging.info("Actor initialized")

  def forward(self, x):
    """
    Forward pass of the network.
    :param x:
      Tuple of (Image, Data)
        Image input data of shape (N x C x H x W) with
            N: batch size
            C: number of channels
            H: hight of the input data
            W  width of the input data
        Data input data as a vector of shape (n,)
    :return x:
      Network output.
    """
    x_img, x_data = x
    x_img = x_img.permute(0, 3, 1, 2)
    for conv_layer in self.conv_modules:
      x_img = conv_layer(x_img)
    x = torch.cat((x_img.reshape(x_img.size(0), -1),
                   x_data.reshape(x_img.size(0), -1)), dim=1)
    for dense_layer in self.dense_modules:
      x = dense_layer(x)
    return x
