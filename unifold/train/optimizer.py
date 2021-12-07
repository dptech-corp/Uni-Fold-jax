# Copyright 2021 Beijing DP Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapped optimizer for training Uni-Fold."""

from absl import logging
import jax
import jax.numpy as jnp
from jax.experimental import optimizers as opt
# in newer version of jax you may need to
# from jax.example_libraries import optimizers as opt
from jax.tree_util import tree_map
import os
import pickle

from unifold.train import utils

OPTIM_DICT = {
    'adam':     opt.adam,
    'sgd':      opt.sgd,
    'adagrad':  opt.adagrad,
    'nesterov': opt.nesterov
}

LR_SCHEDULE_DICT = {
    'exp': opt.exponential_decay
}

def linear_warm_up(peak_lr, num_steps):
  return lambda i: (i + 1) / num_steps * peak_lr

class Optimizer:
  """
  wrapped optimizer using functions in jax.experimental.optimizers.
  this class saves no data but methods.
  """
  def __init__(self,
               optim_config):

    self.config = optim_config

    peak_lr = self.config.learning_rate
    warm_up_steps = self.config.warm_up_steps

    self.warm_up_schedule = linear_warm_up(
        peak_lr=peak_lr,
        num_steps=warm_up_steps)
    
    self.decay_schedule = LR_SCHEDULE_DICT[self.config.decay.name](
        peak_lr,
        self.config.decay.decay_steps,
        self.config.decay.decay_rate)
    
    self.lr_schedule = lambda i: \
        (i < warm_up_steps) * self.warm_up_schedule(i) + \
        (i >= warm_up_steps) * self.decay_schedule(i - warm_up_steps)

    self.opt_init, \
    self.opt_update, \
    self.get_params = OPTIM_DICT[self.config.name](self.lr_schedule)

  def init_state(self, params):
    opt_state = self.opt_init(params)
    return opt_state
  
  unpack_optimizer_state = opt.unpack_optimizer_state

  def clip_grads(self, grads):
    return opt.clip_grads(grads, self.config.clip_norm)

  def save(self, opt_state, path: str):
    # create directory
    dir = os.path.dirname(path)
    logging.debug(f"saving optimizer state to {dir}...")
    if not os.path.exists(dir):
      logging.warning(f"directory {dir} unexisted. creating the directory ...")
      os.makedirs(dir)
    # save opt_state
    if path.endswith('.pkl'):
      params = opt.unpack_optimizer_state(opt_state)
      pickle.dump(params, open(path, "wb"))
    elif path.endswith('.npz'):
      params = self.get_params(opt_state)
      jnp.savez(path, params)
    else:
      raise ValueError(f"unknown parameter format specified: `{path}`.")
  
  def load(self, path: str):
    if path.endswith('.pkl'):
      opt_state = utils.load_opt_state_from_pkl(path)
    elif path.endswith('.npz'):
      params = utils.load_params_from_npz(path)
      opt_state = self.optimizer.init_state(params)
    else:
      raise ValueError(f"unknown type of params: {path}")
    return opt_state

