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

"""Run Uni-Fold with preprocessed protein features (features.pkl)."""

import glob
import os
import pickle

from absl import app, flags, logging

from unifold.inference.inference_pipeline import predict_from_pkl
from unifold.model.config import model_config as get_model_config
from unifold.model.model import RunModel
from unifold.train.mixed_precision import normalize_precision
from unifold.train.utils import load_params
from unifold.relax.relax import AmberRelaxation


flags.DEFINE_list(
    'pickle_paths', None, 'Paths to processed protein features (.pkl), '
    'separated by commas.')
flags.DEFINE_string(
    'pickle_dir', None, 'Path to a directory which contains sub-folders of '
    'processed protein features (.pkl), each named as `<name>/features.pkl`. '
    'Used for automatically predicting all feature files under the directory. '
    'See `./example_data/features` for an example. This argument is ignored '
    'if `pickle_paths` is set.')
flags.DEFINE_list(
    'model_names', None, 'Names of models to use, separated by commas. Each '
    'model name should correspond to a model configuration in '
    '`unifold/model/config.py`')
flags.DEFINE_list(
    'model_paths', None, 'Paths of saved models, separated by commas. Must '
    'be in *.npz format.')
flags.DEFINE_string(
    'output_dir', None, 'Path to a directory that will store the results.')
flags.DEFINE_string(
    'precision', 'fp32', 'Precision used in inference. Uni-Fold supports '
    'inferencing with float32 (\'fp32\'), float16 (\'fp16\') and bfloat16 '
    '(\'bf16\'). Generally, using lower precisions does not siginificantly '
    'influence the accuracies, yet faster inference may be achieved.')
flags.DEFINE_bool(
    'use_amber_relax', True, 'Whether to use the Amber99 Force Field to relax '
    'the predicted structure.')
flags.DEFINE_integer(
    'random_seed', 181129, 'The random seed used for model prediction. This '
    'majorly influences how MSAs are sampled and clustered.')
flags.DEFINE_bool(
    'benchmark', False, 'Whether to re-run the model to derive JAX model '
    'running time without compilation. Only set True if you want to derive '
    'the clean running time on GPU.')
flags.DEFINE_bool(
    'dump_pickle', True, 'Whether to dump the model output in pickle format. '
    'If set True, pickled results will be saved.')

FLAGS = flags.FLAGS

# Configurations for the AmberRelaxer.
# Uni-Fold used the same setups as AlphaFold2.
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  if FLAGS.pickle_dir is None and FLAGS.pickle_paths is None:
    raise app.UsageError("Must provide `pickle_dir` or `pickle_paths`.")
  
  if FLAGS.pickle_dir is not None and FLAGS.pickle_paths is not None:
    logging.warning(f"`pickle_dir` {FLAGS.pickle_dir} is ignored, as "
                    f"`pickle_paths` {FLAGS.pickle_paths} is provided.")
  
  precision = normalize_precision(FLAGS.precision)
  model_runners = {}

  for model_name, model_path in zip(FLAGS.model_names, FLAGS.model_paths):
    model_config = get_model_config(model_name, is_training=False)
    model_params = load_params(model_path)
    model_runner = RunModel(
       config=model_config,
       params=model_params,
       precision=precision)
    model_runners[model_name] = model_runner

  logging.info(f"Input {len(model_runners)} models with "
               f"names: {list(model_runners.keys())}.")

  if FLAGS.use_amber_relax:
    amber_relaxer = AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)
  else:
    amber_relaxer = None

  random_seed = FLAGS.random_seed if FLAGS.random_seed is not None else 0
  logging.info(f"Using random seed {random_seed} for the data pipeline")

  if FLAGS.pickle_paths:    # use protein id 'prot_{idx}' of given list.
    protein_dict = {
        f'prot_{idx:05d}': p for idx, p in enumerate(FLAGS.pickle_paths)}
  else:                     # use basename of sub-directories as protein ids.
    sub_dirs = [
        p for p in glob.glob(os.path.join(FLAGS.pickle_dir, '*'))
            if os.path.isdir(p)]
    protein_dict = {
        os.path.basename(p): os.path.join(p, "features.pkl")
            for p in sub_dirs}

  for id, feature_path in protein_dict.items():
    with open(feature_path, 'rb') as fp:
      features = pickle.load(fp)
    try:
      predict_from_pkl(
          features=features,
          name=id,
          output_dir=FLAGS.output_dir,
          model_runners=model_runners,
          amber_relaxer=amber_relaxer,
          random_seed=random_seed,
          benchmark=FLAGS.benchmark,
          dump_pickle=FLAGS.dump_pickle,
          timings=None)
    except Exception as ex:
      logging.warning(f"failed to predict structure for protein {id} with "
                      f"feature path {feature_path}. Error message: \n{ex}")
      


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'output_dir',
      'model_names',
      'model_paths'
  ])
  app.run(main)
