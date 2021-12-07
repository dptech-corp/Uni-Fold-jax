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

"""Methods for inferencing with Uni-Fold."""

from absl import logging
import json
import os
import numpy as np
import pickle
import time
from typing import Dict, Optional

from unifold.common import protein
from unifold.data.pipeline import DataPipeline
from unifold.model.features import FeatureDict
from unifold.model.model import RunModel
from unifold.relax.relax import AmberRelaxation


def generate_pkl_features_from_fasta(
    fasta_path: str,
    name: str,
    output_dir: str,
    data_pipeline: DataPipeline,
    timings: Optional[Dict[str, float]] = None):
  
  """Predicts structure using Uni-Fold for the given sequence."""
  if timings is None:
    timings = {}
  
  # Check output dir.
  output_dir = os.path.join(output_dir, name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  pt = time.time()
  logging.info(f"processing file {fasta_path}...")
  features = data_pipeline.process(
      input_fasta_path=fasta_path,
      msa_output_dir=msa_output_dir)
  timings['data_pipeline'] = time.time() - pt

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(features, f, protocol=4)
  logging.info(f"process file {fasta_path} done.")
  
  # Save timings.
  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as fp:
    json.dump(timings, fp, indent=4)

  return features


def predict_from_pkl(
    features: FeatureDict,
    name: str,
    output_dir: str,
    model_runners: Dict[str, RunModel],
    amber_relaxer: Optional[AmberRelaxation],
    random_seed: int,
    benchmark: bool = False,
    dump_pickle: bool = True,
    timings: Optional[Dict[str, float]] = None):
  """Predicts structure using Uni-Fold for the given features."""

  if not timings:
    timings = {}

  output_dir = os.path.join(output_dir, name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  output_pdbs = {}
  plddts = {}

  # Run the models.
  for model_name, model_runner in model_runners.items():
    logging.info(f"Running model {model_name} ...")
    # Process features.
    pt = time.time()
    processed_features = model_runner.process_features(
        features, random_seed=random_seed)
    timings[f'process_features_{model_name}'] = time.time() - pt
    # Run the prediction code.
    pt = time.time()
    prediction_result = model_runner.predict(processed_features)
    t_diff = time.time() - pt
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(f"Total JAX model {model_name} predict time (compilation "
                 f"included): {t_diff:.0f}.")

    # If benchmarking, re-run to test JAX running time without compilation.
    if benchmark:
      pt = time.time()
      model_runner.predict(processed_features)
      timings[f'predict_benchmark_{model_name}'] = time.time() - pt

    # Save the model outputs in pickle format.
    if dump_pickle:
      result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
      with open(result_output_path, 'wb') as fp:
        pickle.dump(prediction_result, fp, protocol=4)
    
    # Save residue-wise pLDDT.
    plddt_out_path = os.path.join(output_dir, f'res_plddt_{model_name}.txt')
    np.savetxt(plddt_out_path, prediction_result['plddt'])

    # Get mean pLDDT confidence metric.
    plddts[model_name] = np.mean(prediction_result['plddt'])

    # Get and save unrelaxed protein.
    unrelaxed_protein = protein.from_prediction(processed_features,
                                                prediction_result)

    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    unrelaxed_pdb_str = protein.to_pdb(unrelaxed_protein)
    with open(unrelaxed_pdb_path, 'w') as fp:
      fp.write(unrelaxed_pdb_str)

    # Relax the prediction.
    if amber_relaxer is not None:
      # Run the relaxation.
      pt = time.time()
      relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
      timings[f'relax_{model_name}'] = time.time() - pt
      # Save the relaxed PDB.
      output_pdbs[model_name] = relaxed_pdb_str
      relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}.pdb')
      with open(relaxed_output_path, 'w') as fp:
        fp.write(relaxed_pdb_str)
    else:
      output_pdbs[model_name] = unrelaxed_pdb_str

  # Rank by pLDDT and write out PDBs in rank order.
  ranked_order = []
  for idx, (model_name, _) in enumerate(
      sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
    ranked_order.append(model_name)
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as fp:
      fp.write(output_pdbs[model_name])

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as fp:
    json.dump({'plddts': plddts, 'order': ranked_order}, fp, indent=4)

  logging.info(f"Final timings for {name}: {timings}")

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as fp:
    json.dump(timings, fp, indent=4)
  
  return output_pdbs, plddts


def predict_from_fasta(
    fasta_path: str,
    name: str,
    output_dir: str,
    data_pipeline: DataPipeline,
    model_runners: Dict[str, RunModel],
    amber_relaxer: Optional[AmberRelaxation],
    random_seed: int,
    benchmark: bool = False,
    dump_pickle: bool = True,
    timings: Optional[Dict[str, float]] = None):      # kwargs are passed to predict_from_pkl.
  """Predicts structure using Uni-Fold for the given fasta file: """
  """generates a features.pkl file and then calls predict_from_pkl."""
  
  timings = {}

  # generate feature dict
  features = generate_pkl_features_from_fasta(
      fasta_path=fasta_path,
      name=name,
      output_dir=output_dir,
      data_pipeline=data_pipeline,
      timings=timings)

  output_pdbs, plddts = predict_from_pkl(
     features=features,
     name=name,
     output_dir=output_dir,
     model_runners=model_runners,
     amber_relaxer=amber_relaxer,
     random_seed=random_seed,
     benchmark=benchmark,
     dump_pickle=dump_pickle,
     timings=timings)
  
  return features, output_pdbs, plddts

  
