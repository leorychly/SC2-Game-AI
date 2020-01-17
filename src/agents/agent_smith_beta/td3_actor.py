import numpy as np
from absl import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Actor(nn.Module):

  def __init__(self, img_state_dim,
               vect_state_len,
               action_space,
               device):
    """
    Create the actor network of the TD3 Algorithm.

    :param img_state_dim: Tuple
      Number of channels of the image input tensor at last place. (h,w,c)
    :param vect_state_len: Int
      Size of th semantic state input vector.
    :param action_space: Tupel of Ints
      Shape of the action space.
      E.g. for a combination of 10 categorical actions 1-hot encoded
       together with 2 continuous regression outputs,
       the action_space_dim would be of size (10, 2).
    """
    action_space_dim = sum(action_space)
    self.action_space = action_space
    super(Actor, self).__init__()
    self.device = device
    conv_output_dim = 256 * 2 * 2

    self.conv_modules = nn.Sequential(
      nn.Conv2d(img_state_dim[-1], 32, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.Conv2d(64, 128, kernel_size=4, stride=4, padding=2),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2)
      #nn.ReLU()
    )
    input_shape = img_state_dim[2], img_state_dim[0], img_state_dim[1]
    print(f"\nActor:"
          f"\n\tQ-Network Conv Input Shape {input_shape}, "
          f"\n\tConv Output: {conv_output_dim}")
    summary(self.conv_modules, input_shape, device="cpu")

    self.dense_modules = nn.Sequential(
      nn.Linear(conv_output_dim + vect_state_len[0], 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, action_space_dim),
      nn.Sigmoid()
    )
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
    if isinstance(x[0], np.ndarray) and isinstance(x[1], np.ndarray):
      x_img = torch.from_numpy(x[0]).float().to(device=self.device)
      x_data = torch.from_numpy(x[1]).float().to(device=self.device)
    else:
      x_img = x[0]
      x_data = x[1]
    x_img = self.conv_modules(x_img)
    x = torch.cat((x_img.reshape(x_img.size(0), -1),
                   x_data.reshape(x_img.size(0), -1)), dim=1)
    x = self.dense_modules(x)
    #x = F.softmax(x[:self.action_space[0]], dim=1)
    return x

# TODO: argmax(sigmoid(x)) vs softmax layer ? o.O