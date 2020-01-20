###########################################################################################
# Implementation of Implicit Quantile Networks (IQN)
# Author for codes: Chu Kun(chukun1997@163.com)
# Paper: https://arxiv.org/abs/1806.06923v1
# Reference: https://github.com/sungyubkim/Deep_RL_with_pytorch
###########################################################################################
import numpy as np
from pathlib2 import Path
from absl import logging
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.agents.agent_smith_gamma.iqn.replay_buffer import ReplayBuffer
from src.agents.agent_smith_gamma.iqn.conv_net import ConvNet

logging.set_verbosity('info')
logging.set_stderrthreshold('info')


def huber(x):
  """Define huber function."""
  cond = (x.abs() < 1.0).float().detach()
  return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)


class IQNAgent(object):
  def __init__(self,
               state_space,
               action_dim,  # INT!
               device,
               lr=1e-4,
               batch_size=32,
               buffer_size=int(1e+5),
               gamma=0.99,
               epsilon=0.999,
               epsilon_decay=0.9999,
               epsilon_min=0.1,
               target_update_interval=1,
               training_interval=4,
               n_steps_before_training=int(1e+3),
               save_interval=1000
               ):
    self.save_path = Path("./results/iqn/")
    self.device = device
    self.action_dim = action_dim
    self.batch_size = batch_size
    self.gamma = gamma
    self.epsilon =epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.target_update_interval = target_update_interval
    self.n_steps_before_training = n_steps_before_training
    self.training_interval = training_interval
    self.save_interval = save_interval

    # Networks
    self.pred_net = ConvNet(n_actions=action_dim, state_space=state_space, device=device)
    self.target_net = ConvNet(n_actions=action_dim, state_space=state_space, device=device)
    self.update_target(self.target_net, self.pred_net, 1.0)
    self.pred_net.to(self.device)
    self.target_net.to(self.device)
    self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=lr)
    # Replay Buffer
    self.replay_buffer = ReplayBuffer(buffer_size)
    #Tensorboard
    tensorboard_dir = self.save_path / "tensorboard_log"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    self.writer = SummaryWriter(
      log_dir=tensorboard_dir.absolute().as_posix())

    # Load
    try:
      self.load_model()
      logging.info(f"The model was loaded from "
                   f"'{self.save_path.absolute().as_posix()}'")
    except Exception as e:
      logging.info(f"No model loaded from "
                   f"'{self.save_path.absolute().as_posix()}'")

    self.memory_counter = 0  # simulator step counter
    self.learn_step_counter = 0  # target network step counter
    self.global_step = 0

  def choose_action(self, x, is_random=False):
    if is_random or np.random.uniform() < self.epsilon:
      #action = np.random.randint(0, self.action_dim, (x.size(0)))
      action = np.random.randint(low=0, high=self.action_dim)
    else:
      x = np.expand_dims(x, axis=0)
      x = torch.from_numpy(x)
      #x = torch.FloatTensor(x)
      x = x.to(self.device)
      action_value, tau = self.pred_net(x)  # (N_ENVS, N_ACTIONS, N_QUANT)
      action_value = action_value.mean(dim=2)
      action = torch.argmax(action_value, dim=1).data.cpu().numpy()[0]
    if not is_random and self.epsilon >= self.epsilon_min:
      self.epsilon *= self.epsilon_decay
    return int(action)

  def step(self, state, action, reward, next_state, done):
    self._store_transition(s=state,
                           a=action,
                           r=reward,
                           s_=next_state,
                           done=done)
    if (self.global_step >= self.n_steps_before_training
        and len(self.replay_buffer) > self.batch_size):  # TODO
      if self.global_step % self.training_interval == 0:
        self._optimize()
        if self.global_step % self.target_update_interval:
          self.update_target(self.target_net, self.pred_net, 1e-2)

    if self.global_step % self.save_interval == 0:
      self.save_model()

    self.global_step += 1

  def _store_transition(self, s, a, r, s_, done):
    self.memory_counter += 1
    self.replay_buffer.add(s, a, r, s_, float(done))  # TODO

  def _optimize(self):
    self.learn_step_counter += 1

    b_s, b_a, b_r, b_s_, b_d = self.replay_buffer.sample(self.batch_size)
    #print(b_d)
    b_w, b_idxes = np.ones_like(b_r), None

    b_s = torch.FloatTensor(b_s).to(self.device)
    b_a = torch.LongTensor(b_a).to(self.device)
    b_r = torch.FloatTensor(b_r).to(self.device)
    b_s_ = torch.FloatTensor(b_s_).to(self.device)
    b_d = torch.FloatTensor(b_d).to(self.device)

    # action value distribution prediction
    q_eval, q_eval_tau = self.pred_net(b_s)  # (m, N_ACTIONS, N_QUANT), (N_QUANT, 1)
    mb_size = q_eval.size(0)
    q_eval = torch.stack([q_eval[i].index_select(0, b_a[i]) for i in range(mb_size)]).squeeze(1)
    # (m, N_QUANT)
    # 在q_eval第二维后面加一个维度
    q_eval = q_eval.unsqueeze(2)  # (m, N_QUANT, 1)
    # note that dim 1 is for present quantile, dim 2 is for next quantile

    # get next state value
    q_next, q_next_tau = self.target_net(b_s_)  # (m, N_ACTIONS, N_QUANT), (N_QUANT, 1)
    best_actions = q_next.mean(dim=2).argmax(dim=1)  # (m)
    q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
    # q_nest: (m, N_QUANT)
    # q_target = R + gamma * (1 - terminate) * q_next
    q_target = b_r.unsqueeze(1) + self.gamma * (1. - b_d.unsqueeze(1)) * q_next
    # q_target: (m, N_QUANT)
    # detach表示该Variable不更新参数
    q_target = q_target.unsqueeze(1).detach()  # (m , 1, N_QUANT)

    # quantile Huber loss
    #print('q_target', q_target.shape)
    #print('q_eval', q_eval.shape)
    #print('q_target_', q_target.detach().shape)
    u = q_target.detach() - q_eval  # (m, N_QUANT, N_QUANT)
    tau = q_eval_tau.unsqueeze(0)  # (1, N_QUANT, 1)
    # note that tau is for present quantile
    # w = |tau - delta(u<0)|
    weight = torch.abs(tau - u.le(0.).float())  # (m, N_QUANT, N_QUANT)
    q_eval = q_eval.permute(0, 2, 1)
    loss = F.smooth_l1_loss(q_eval, q_target.detach(), reduction='none')
    # (m, N_QUANT, N_QUANT)
    loss = torch.mean(weight * loss, dim=1).mean(dim=1)
    self.to_tensorboard(var=torch.mean(loss), name="Loss")

    # calculate importance weighted loss
    b_w = torch.Tensor(b_w)
    b_w = b_w.to(self.device)
    loss = torch.mean(b_w * loss)

    # backprop loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss

  def update_target(self, target, pred, update_rate):
    """Update target network parameters using predcition network."""
    for target_param, pred_param in zip(target.parameters(), pred.parameters()):
      target_param.data.copy_((1.0 - update_rate)
                              * target_param.data + update_rate * pred_param.data)

  def save_model(self):
    """Save prediction network and target network"""
    self.pred_net.save((self.save_path / "iqn_pred_net").absolute().as_posix())
    self.target_net.save((self.save_path / "iqn_target_net").absolute().as_posix())

  def load_model(self):
    """Load prediction network and target network."""
    self.pred_net.load((self.save_path / "iqn_pred_net").absolute().as_posix())
    self.target_net.load((self.save_path / "iqn_target_net").absolute().as_posix())

  def to_tensorboard(self, var, name):
    self.writer.add_scalar(name, var, self.global_step)
