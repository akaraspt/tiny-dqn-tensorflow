import numpy as np

class History(object):
  def __init__(
    self,
    data_format,
    batch_size,
    history_length,
    screen_height,
    screen_width
  ):
    self.data_format = data_format

    batch_size, history_length, screen_height, screen_width = \
        batch_size, history_length, screen_height, screen_width

    self.history = np.zeros(
        [history_length, screen_height, screen_width], dtype=np.float32)

  def add(self, screen):
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen

  def reset(self):
    self.history *= 0

  def get(self):
    if self.data_format == 'NHWC':
      return np.transpose(self.history, (1, 2, 0))
    else:
      return self.history
