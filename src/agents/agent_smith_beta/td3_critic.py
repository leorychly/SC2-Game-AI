import numpy as np
import copy
from absl import logging
import torch
import torch.nn as nn
from torchsummary import summary


class Critic(nn.Module):

  def __init__(self,
               img_state_dim,
               vect_state_len,
               action_dim,
               device):
    """
    Create the critic network of the TD3 Algorithm.

    :param img_state_dim: Tuple
      Number of channels of the image input tensor at last place (h,w,c).
    :param vect_state_len: Int
      Size of th semantic state input vector.
    :param action_dim: Int
      Shape of the specific executed action.
      E.g. for a combination of a 10-Action 1-hot encoding + 2 Regression
      outputs, the action_shape would be of size 12.
    """
    super(Critic, self).__init__()
    self.device = device

    conv_output_dim = 32 * 4 * 4

    self.conv_modules_q1 = nn.Sequential(
      nn.Conv2d(img_state_dim[-1], 64, kernel_size=5, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=0),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2))

    input_shape = img_state_dim[2], img_state_dim[0], img_state_dim[1]
    print(f"\nCritic:"
          f"\tQ-Network Conv Input Shape {input_shape}, "
          f"\tConv Output: {conv_output_dim}")
    summary(self.conv_modules_q1, input_shape, device="cpu")

    self.dense_modules_q1 = nn.Sequential(
      nn.Linear(conv_output_dim + vect_state_len[0] + action_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 1))
    print(f"\n\tActor Dense Input Shape {input_shape}, "
          f"\tConv Output: {conv_output_dim}")
    summary(self.dense_modules_q1,
            (conv_output_dim + vect_state_len[0] + action_dim,),
            device="cpu")
    logging.info("Actor initialized")

    self.conv_modules_q2 = copy.deepcopy(self.conv_modules_q1)
    self.dense_modules_q2 = copy.deepcopy(self.dense_modules_q1)

  def forward(self, x):
    """
    Forward pass of both critic networks.

    :param x: Tuple
     (state, action)-Tuple with state being state=(Image,Data)
        Image input data of shape (N x C x H x W) with
            N: batch size
            C: number of channels
            H: hight of the input data
            W  width of the input data
        Data input data as a vector of shape (n,)
    :param a: torch.tensor
      The agent's action.

    :return x: torch.Tensor
      Both network outputs.
    """
    (x_img_1, x_data_1), action_1 = x
    #x_img_1 = x_img_1.permute(0, 3, 1, 2)
    x_img_2 = copy.deepcopy(x_img_1)
    x_data_2 = copy.deepcopy(x_data_1)
    action_2 = copy.deepcopy(action_1)

    x_img_1 = self.conv_modules_q1(x_img_1)
    x1 = torch.cat((x_img_1.reshape(x_img_1.size(0), -1),
                   x_data_1.reshape(x_img_1.size(0), -1),
                   action_1), dim=1)
    x1 = self.dense_modules_q1(x1)

    x_img_2 = self.conv_modules_q2(x_img_2)
    x2 = torch.cat((x_img_2.reshape(x_img_2.size(0), -1),
                    x_data_2.reshape(x_img_2.size(0), -1),
                    action_2), dim=1)
    x2 = self.dense_modules_q2(x2)
    return x1, x2

  def Q1(self, x):
    """
    Forward pass of the first critic network.

    :param x: Tuple
      ( (state_pix, state_semantic),  action )

    :return: torch.Tensor
      Network output.
    """
    (x_img, x_data), action = x
    #x = torch.cat((x_img.reshape(x_img.size(0), -1),
    #               x_data.reshape(x_img.size(0), -1),
    #               action), dim=1)
    #for layer in self.module_list_q1:
    #  x = layer(x)
    x_img = self.conv_modules_q1(x_img)
    x_out = torch.cat((x_img.reshape(x_img.size(0), -1),
                    x_data.reshape(x_img.size(0), -1),
                    action), dim=1)
    x_out = self.dense_modules_q1(x_out)
    return x_out
