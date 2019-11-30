from abc import ABC, abstractmethod


class BaseObserver(ABC):

  def get_state(self, obs):
    raise NotImplementedError
