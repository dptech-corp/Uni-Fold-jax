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

"""Container (Trainer) for training Uni-Fold."""

# major imports
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import os
import time


# following packages for mpi communication will be just-in-time imported if use_mpi is True
# from mpi4py import MPI
# import mpi4jax
# from jax.tree_util import tree_flatten, tree_unflatten 

# import type name specifications
from unifold.model.features import FeatureDict
from ml_collections import ConfigDict
from typing import Optional

# import major classes & functions
from unifold.model.modules import AlphaFold
from unifold.train.data_system import cast_to_precision
from unifold.train.optimizer import Optimizer
from unifold.train.mixed_precision import normalize_precision, set_unifold_policy


class Trainer:
  """
  main class to train the UniFold model.
  """
  def __init__(
      self,
      global_config: ConfigDict,
      optim_config: ConfigDict,
      model_config: ConfigDict,
      **kwargs):

    self.gc = global_config

    self.precision = normalize_precision(self.gc.precision)
    set_unifold_policy(self.precision)
    
    def _forward_fn(batch):
      model = AlphaFold(model_config.model)
      return model(
          batch,
          is_training=True,
          compute_loss=True,
          ensemble_representations=False)

    self._init_fn = jax.jit(hk.transform(_forward_fn).init)
    self._apply_fn = jax.jit(hk.transform(_forward_fn).apply)
    self._loss_fn = None        # has to be initialized by external call on `Trainer.initialize()`
    self._update_fn = None      # has to be initialized by external call on `Trainer.initialize()`

    # optimizer variables, have to be initialized by external call on `Trainer.initialize()`
    self.optim_config = optim_config
    self.optimizer = None       # instance of unifold.trainer.optimizer.Optimizer
    self.optim_state = None     # optimizer state, key variable of maintaining model parameters

    # logging variables organized in format [(step, loss, time), (step, loss, time), ...]
    self.train_losses = []
    self.eval_losses = []

    # timing variables
    self._tic = time.time()      # tic & toc style for timing. handled in `Trainer.logging_on_the_fly`

    # mpi variables
    if self.gc.use_mpi:
      self.mpi_comm = kwargs['mpi_comm']
      self.mpi_rank = self.mpi_comm.Get_rank()
      self.world_size = self.mpi_comm.Get_size()

    # path formatters of ckpts and loss curves
    self.auto_ckpt_name = \
        lambda step, format: f"{self.gc.model_name}_{step:05d}.{format}"
    self.auto_curve_name = \
        lambda is_train: f"{self.gc.model_name}_{'train' if is_train else 'eval'}_curve.npy"
    
    # step specifier
    self.is_logging_step = lambda i: i % self.gc.logging_freq == 0
    self.is_save_step = lambda i: (i + 1) % self.gc.save_freq == 0
    self.is_eval_step = lambda i: i % self.gc.eval_freq == 0


  @property
  def params(self):
    if self.optim_state is None:
      return None
    else:
      return self.optimizer.get_params(self.optim_state)


  def initialize(
      self,
      batch: Optional[FeatureDict] = None,
      load_format: Optional[str] = None,
      random_seed: Optional[int] = None):
    
    # create optimizer instance
    self.optimizer = Optimizer(self.optim_config)

    use_autoload = self.gc.start_step > 0
    if use_autoload:
      assert load_format is not None, \
          "must provide `load_format` to auto load models when assigning `start_step` > 0."
      self.autoload(self.gc.start_step, load_format)
    else:
      assert batch is not None, \
          "must provide a batch and a random seed to initialize model from scratch."
      if self.optim_state is not None:
        logging.warning("existed optimizer states are reinitialized.")
      if random_seed is not None:
        logging.warning("external random seed overrides the one in global config.")
      else:
        random_seed = self.gc.random_seed
        rng = jax.random.PRNGKey(random_seed)    # all ranks initialized equally.
      params = hk.data_structures.to_mutable_dict(self._init_fn(batch=batch, rng=rng))
      self.optim_state = self.optimizer.init_state(params)
    
    # define loss_fn
    def _loss_fn(params, batch, rng):
      # TODO: user external RNG
      _, loss = self._apply_fn(params=params, batch=batch, rng=rng)
      seq_length_weight = jnp.sqrt(jnp.sum(batch['all_atom_mask'][0,:,0]))
      return loss * seq_length_weight

    # define reduce_fn for mpi communication.
    if self.gc.use_mpi:
      # just-in-time imports
      from mpi4py import MPI
      import mpi4jax
      from jax.tree_util import tree_flatten, tree_unflatten
      def _mpi_reduce_value(value):
        value, _ = mpi4jax.allreduce(value, op=MPI.SUM, comm=self.mpi_comm)
        value /= self.world_size
        return value
      def _mpi_reduce_tree(tree):
        flat_tree, tree_struct = tree_flatten(tree)
        for i, val in enumerate(flat_tree):
          flat_tree[i] = _mpi_reduce_value(val)
        tree = tree_unflatten(tree_struct, flat_tree)
        return tree
    
    # define update_fn.  
    def _update_fn(step, opt_state, batch, rng):
      loss, grads = jax.value_and_grad(_loss_fn)(
          self.optimizer.get_params(opt_state), batch, rng)
      grads = self.optimizer.clip_grads(grads)
      if self.gc.use_mpi:
        loss = _mpi_reduce_value(loss)
        grads = _mpi_reduce_tree(grads)
      opt_state = self.optimizer.opt_update(step, grads, opt_state)
      return opt_state, loss
    
    # define eval_fn for validation.
    def _eval_fn(params, batch, rng):
      loss = _loss_fn(params, batch, rng)
      if self.gc.use_mpi:
        loss = _mpi_reduce_value(loss)
      return loss
    
    self._loss_fn = _loss_fn     # this is not re-jit as loss_fn is much of a wrapped apply_fn.
    self._eval_fn = _eval_fn
    self._update_fn = jax.jit(_update_fn)   # jit transformation of update_fn.
    
    # start ticking after initialization.
    self._tic = time.time()
  

  def autosave(self, step):
    # save ckpt in both npz and pkl formats.
    save_path_npz = os.path.join(self.gc.save_dir, self.auto_ckpt_name(step + 1, 'npz'))
    self.optimizer.save(self.optim_state, save_path_npz)
    save_path_pkl = os.path.join(self.gc.save_dir, self.auto_ckpt_name(step + 1, 'pkl'))
    self.optimizer.save(self.optim_state, save_path_pkl)
    # save loss curve
    train_curve_path = os.path.join(self.gc.save_dir, self.auto_curve_name(is_train=True))
    eval_curve_path = os.path.join(self.gc.save_dir, self.auto_curve_name(is_train=False))
    np.save(train_curve_path, np.asarray(self.train_losses))
    np.save(eval_curve_path, np.asarray(self.eval_losses))
    logging.info(f"model autosaved at step {step:05d} successfully.")


  def autoload(self, step, format='pkl'):
    # load ckpt
    load_path = os.path.join(self.gc.load_dir, self.auto_ckpt_name(step, format))
    self.optimizer.load(load_path)
    # load loss curve
    train_curve_path = os.path.join(self.gc.save_dir, self.auto_curve_name(is_train=True))
    eval_curve_path = os.path.join(self.gc.save_dir, self.auto_curve_name(is_train=False))
    def load_loss_curve(loss_curve_path: str):
      try:
        return np.load(loss_curve_path).tolist()
      except:
        logging.warning(f"failed to load curve from {loss_curve_path}. reset loss curve.")
        return []
    self.train_losses = load_loss_curve(train_curve_path)
    self.eval_losses = load_loss_curve(eval_curve_path)
    logging.info(f"model autoloaded at step {step:05d} successfully.")


  def update(self, step, batch, rng):
    # wrapped update_fn for external calls.
    opt_state, loss = self._update_fn(step, self.optim_state, batch, rng)
    self.optim_state = opt_state
    return loss


  def _logging(self, step, loss):
    # print and record training stats at the step.
    toc = time.time()
    step_time = (toc - self._tic) / (
        1 if step == 0 else self.gc.logging_freq)
    self.train_losses.append((step, loss, step_time))
    logging.info(f"step: {step:05d}\ttrain_loss: {loss:3.4f}\tstep_time: {step_time:.2f}s")
    self._tic = time.time()


  def train_step(self, step, batch, rng, silent=True):
    batch = cast_to_precision(batch, self.precision)
    loss = self.update(step, batch, rng)
    if not silent:
      if self.is_logging_step(step):
        self._logging(step, loss)
      if self.is_save_step(step):
        self.autosave(step)


  def eval_step(self, step, batch, rng, silent=True):
    # evaluation on the fly
    tmp_tic = time.time()
    loss = self._eval_fn(self.params, batch, rng)
    eval_time = time.time() - tmp_tic
    if not silent:
      self.eval_losses.append((step, loss, eval_time))
      logging.info(f"step: {step:05d}\teval_loss:  {loss:3.4f}\teval_time: {eval_time:.2f}s")



