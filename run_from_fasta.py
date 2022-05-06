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

"""Full Uni-Fold protein structure prediction script."""

import glob
import json
import numpy as np
import os
import pathlib
import pickle
import random
import sys
import time
from typing import Dict

from absl import app
from absl import flags
from absl import logging
from unifold.common import protein
from unifold.data import pipeline
from unifold.data import templates
from unifold.model import data
from unifold.model import config
from unifold.model import model
from unifold.relax import relax


from unifold.inference.inference_pipeline import predict_from_fasta
from unifold.model.config import model_config as get_model_config
from unifold.model.model import RunModel
from unifold.train.mixed_precision import normalize_precision
from unifold.train.utils import load_params
from unifold.relax.relax import AmberRelaxation

#### USER CONFIGURATION ####

# Note: If your databases and toolkits were configured directly using the scripts 
# provided by Uni-Fold, the following code is directly useful. If not, you need 
# to customize your own paths of the downloaded databases and toolkits. 
# See README.md for more details.

# Default paths. Note that these arguments can be set directly by flags from command line.
default_database_dir = '.' 
default_fasta_dir = './example_data/fasta/'
default_output_dir = './out/features/'

# Set to database directory with Uniref90, MGnify, and BFD, Uniclust30 database.
# Path to the Uniref90 database for use by JackHMMER.
uniref90_database_path = os.path.join(
    default_database_dir, 'uniref90', 'uniref90.fasta')

# Path to the MGnify database for use by JackHMMER.
mgnify_database_path = os.path.join(
    default_database_dir, 'mgnify', 'mgy_clusters.fa')

# Path to the BFD database for use by HHblits.
bfd_database_path = os.path.join(
    default_database_dir, 'bfd',
    'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt')

# Path to the Uniclust30 database for use by HHblits.
uniclust30_database_path = os.path.join(
    default_database_dir, 'uniclust30', 'UniRef30_2020_06/UniRef30_2020_06')

# Path to the PDB70 database for use by HHsearch.
pdb70_database_path = os.path.join(default_database_dir, 'pdb70', 'pdb70')

# Path to a directory with template mmCIF structures, each named <pdb_id>.cif')
template_mmcif_dir = os.path.join(default_database_dir, 'pdb_mmcif', 'mmcif_files')

# Path to a file mapping obsolete PDB IDs to their replacements.
obsolete_pdbs_path = os.path.join(default_database_dir, 'pdb_mmcif', 'obsolete.dat')

#### END OF USER CONFIGURATION ####



flags.DEFINE_list(
    'fasta_paths', None, 'Paths to protein fasta files (.fasta), each '
    'containing one sequence. Paths should be separated by commas.')
flags.DEFINE_string(
    'fasta_dir', None, 'Path to a directory which contains target fasta '
    'files, each named as `<name>.fasta`. Used for automatically predicting '
    'all fasta files under a directory. See `./example_data/fasta` for an '
    'example. This argument is ignored if `fasta_paths` is set.')
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

#### Arguments for the data pipeline. ###

flags.DEFINE_string('data_dir', default_database_dir,
                    'Path to directory of supporting data.')
flags.DEFINE_string('jackhmmer_binary_path', 'jackhmmer',
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', 'hhblits',
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', 'hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('kalign_binary_path', 'kalign',
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', uniref90_database_path,
                    'Path to the Uniref90 database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', mgnify_database_path,
                    'Path to the MGnify database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', bfd_database_path, 
                    'Path to the BFD database for use by HHblits.')
flags.DEFINE_string('uniclust30_database_path', uniclust30_database_path,
                    'Path to the Uniclust30 database for use by HHblits.')
flags.DEFINE_string('pdb70_database_path', pdb70_database_path,
                    'Path to the PDB70 database for use by HHsearch.')
flags.DEFINE_string('template_mmcif_dir', template_mmcif_dir,
                    'Path to a directory with template mmCIF structures, '
                    'each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', '2020-4-30',
                    'Maximum template release date to consider. Important '
                    'if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', obsolete_pdbs_path,
                    'Path to file containing a mapping from obsolete PDB IDs '
                    'to the PDB IDs of their replacements.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  if FLAGS.fasta_dir is None and FLAGS.fasta_paths is None:
    raise app.UsageError("Must provide `fasta_dir` or `fasta_paths`.")
  
  if FLAGS.fasta_dir is not None and FLAGS.fasta_paths is not None:
    logging.warning(f"`fasta_dir` {FLAGS.fasta_dir} is ignored, as "
                    f"`fasta_paths` {FLAGS.fasta_paths} is provided.")
  
  # Set template featurizer, data pipeline, and models.
  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=FLAGS.template_mmcif_dir,
      max_template_date=FLAGS.max_template_date,
      max_hits=MAX_TEMPLATE_HITS,
      kalign_binary_path=FLAGS.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      hhsearch_binary_path=FLAGS.hhsearch_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      small_bfd_database_path=None,
      pdb70_database_path=FLAGS.pdb70_database_path,
      template_featurizer=template_featurizer,
      use_small_bfd=False)

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

  if FLAGS.fasta_paths:    # use protein id 'prot_{idx}' of given list.
    protein_dict = {
        f'prot_{idx:05d}': p for idx, p in enumerate(FLAGS.fasta_paths)}
  else:                     # use basename of sub-directories as protein ids.
    fasta_paths = [
        p for p in glob.glob(FLAGS.fasta_dir + '*') if p.endswith('.fasta')]
    protein_dict = {
        pathlib.Path(p).stem: p for p in fasta_paths}

  for id, fasta_path in protein_dict.items():
    try:
      predict_from_fasta(
          fasta_path=fasta_path,
          name=id,
          output_dir=FLAGS.output_dir,
          data_pipeline=data_pipeline,
          model_runners=model_runners,
          amber_relaxer=amber_relaxer,
          random_seed=random_seed,
          benchmark=FLAGS.benchmark,
          dump_pickle=FLAGS.dump_pickle,
          timings=None)
    except Exception as ex:
      logging.warning(f"failed to predict structure for protein {id} with "
                      f"fasta path {fasta_path}. Error message: \n{ex}")


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'output_dir',
      'model_names',
      'model_paths',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'pdb70_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
  ])
  app.run(main)
