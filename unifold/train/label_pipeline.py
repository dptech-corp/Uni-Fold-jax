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

"""Label Preprocessing pipeline for training Uni-Fold."""

import numpy as np
import jax.numpy as jnp
from unifold.common import residue_constants
from unifold.model import quat_affine
from unifold.model import all_atom
from unifold.model.all_atom import atom37_to_frames
from unifold.model.all_atom import atom37_to_torsion_angles
from unifold.model.modules \
    import pseudo_beta_fn as pseudo_beta_fn_jnp
from unifold.model.tf.data_transforms \
    import pseudo_beta_fn as pseudo_beta_fn_tf
from unifold.model.tf.input_pipeline import compose
from unifold.data.pipeline import FeatureDict


prot: FeatureDict


def check_input_completeness(prot):
  """"""
  required_keys = {
    # basic requirements:
    "aatype_index",                 # [NR,]
    "all_atom_positions",     # [NR, 37, 3]
    "all_atom_mask"           # [NR, 37]
  }
  for k in required_keys:
    assert k in prot.keys(), f"required key {k} unsatisfied."
  # assert the num_res are compatible:
  assert prot['aatype_index'].shape[0] == prot['all_atom_positions'].shape[0], \
      f"ziyao: incompatible shapes of num_res: " \
      f"{prot['aatype_index'].shape[0]} from aatype_index, " \
      f"{prot['all_atom_positions'].shape[0]} from all_atom_positions."
  # assert aatype_index is provided instead of one-hot features:
  assert len(prot['aatype_index'].shape) == 1, \
      f"ziyao: wrong aatype_index shape: {prot['aatype_index'].shape}." \
      f"make sure indices of residues instead of one-hot features are provided."
  return prot


def make_atom14_data(prot):
  """
  ziyao: this function generates all atom14-related keys including:
  [
    "atom14_atom_exists",
    "atom14_gt_exists",
    "atom14_gt_positions",
    "residx_atom14_to_atom37",
    "residx_atom37_to_atom14",
    "atom37_atom_exists",
    "atom14_alt_gt_positions",
    "atom14_alt_gt_exists",
    "atom14_atom_is_ambiguous"
  ]
  """
  # ziyao: this function assumes numpy inputs.
  make_atom14_positions(prot)
  return prot


def make_atom14_positions(prot):
  """Copied function from amber_minimize. change key 'aatype' to 'aatype_index'. """
  # ziyao: input prot is a mapping with key 'aatype', 'all_atom_positions' and 'all_atom_mask';
  #        corresponding values should be np.ndarray of shape [NR, ?], [NR, 37, 3] and [NR, 37].
  restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
  restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
  restype_atom14_mask = []

  for rt in residue_constants.restypes:
    atom_names = residue_constants.restype_name_to_atom14_names[
        residue_constants.restype_1to3[rt]]

    restype_atom14_to_atom37.append([
        (residue_constants.atom_order[name] if name else 0)
        for name in atom_names
    ])

    atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
    restype_atom37_to_atom14.append([
        (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
        for name in residue_constants.atom_types
    ])

    restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

  # Add dummy mapping for restype 'UNK'.
  restype_atom14_to_atom37.append([0] * 14)
  restype_atom37_to_atom14.append([0] * 37)
  restype_atom14_mask.append([0.] * 14)

  restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
  restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
  restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

  # Create the mapping for (residx, atom14) --> atom37, i.e. an array
  # with shape (num_res, 14) containing the atom37 indices for this protein.
  residx_atom14_to_atom37 = restype_atom14_to_atom37[prot['aatype_index']]
  residx_atom14_mask = restype_atom14_mask[prot['aatype_index']]

  # Create a mask for known ground truth positions.
  residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
      prot["all_atom_mask"], residx_atom14_to_atom37, axis=1).astype(np.float32)

  # Gather the ground truth positions.
  residx_atom14_gt_positions = residx_atom14_gt_mask[:, :, None] * (
      np.take_along_axis(prot["all_atom_positions"],
                         residx_atom14_to_atom37[..., None],
                         axis=1))

  prot["atom14_atom_exists"] = residx_atom14_mask
  prot["atom14_gt_exists"] = residx_atom14_gt_mask
  prot["atom14_gt_positions"] = residx_atom14_gt_positions

  prot["residx_atom14_to_atom37"] = residx_atom14_to_atom37

  # Create the gather indices for mapping back.
  residx_atom37_to_atom14 = restype_atom37_to_atom14[prot['aatype_index']]
  prot["residx_atom37_to_atom14"] = residx_atom37_to_atom14

  # Create the corresponding mask.
  restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
  for restype, restype_letter in enumerate(residue_constants.restypes):
    restype_name = residue_constants.restype_1to3[restype_letter]
    atom_names = residue_constants.residue_atoms[restype_name]
    for atom_name in atom_names:
      atom_type = residue_constants.atom_order[atom_name]
      restype_atom37_mask[restype, atom_type] = 1

  residx_atom37_mask = restype_atom37_mask[prot['aatype_index']]
  prot["atom37_atom_exists"] = residx_atom37_mask

  # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
  # alternative ground truth coordinates where the naming is swapped
  restype_3 = [
      residue_constants.restype_1to3[res] for res in residue_constants.restypes
  ]
  restype_3 += ["UNK"]

  # Matrices for renaming ambiguous atoms.
  all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
  for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
    correspondences = np.arange(14)
    for source_atom_swap, target_atom_swap in swap.items():
      source_index = residue_constants.restype_name_to_atom14_names[
          resname].index(source_atom_swap)
      target_index = residue_constants.restype_name_to_atom14_names[
          resname].index(target_atom_swap)
      correspondences[source_index] = target_index
      correspondences[target_index] = source_index
      renaming_matrix = np.zeros((14, 14), dtype=np.float32)
      for index, correspondence in enumerate(correspondences):
        renaming_matrix[index, correspondence] = 1.
    all_matrices[resname] = renaming_matrix.astype(np.float32)
  renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])

  # Pick the transformation matrices for the given residue sequence
  # shape (num_res, 14, 14).
  renaming_transform = renaming_matrices[prot['aatype_index']]

  # Apply it to the ground truth positions. shape (num_res, 14, 3).
  alternative_gt_positions = np.einsum("rac,rab->rbc",
                                       residx_atom14_gt_positions,
                                       renaming_transform)
  prot["atom14_alt_gt_positions"] = alternative_gt_positions

  # Create the mask for the alternative ground truth (differs from the
  # ground truth mask, if only one of the atoms in an ambiguous pair has a
  # ground truth position).
  alternative_gt_mask = np.einsum("ra,rab->rb",
                                  residx_atom14_gt_mask,
                                  renaming_transform)

  prot["atom14_alt_gt_exists"] = alternative_gt_mask

  # Create an ambiguous atoms mask.  shape: (21, 14).
  restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
  for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
    for atom_name1, atom_name2 in swap.items():
      restype = residue_constants.restype_order[
          residue_constants.restype_3to1[resname]]
      atom_idx1 = residue_constants.restype_name_to_atom14_names[resname].index(
          atom_name1)
      atom_idx2 = residue_constants.restype_name_to_atom14_names[resname].index(
          atom_name2)
      restype_atom14_is_ambiguous[restype, atom_idx1] = 1
      restype_atom14_is_ambiguous[restype, atom_idx2] = 1

  # From this create an ambiguous_mask for the given sequence.
  prot["atom14_atom_is_ambiguous"] = (
      restype_atom14_is_ambiguous[prot['aatype_index']])

  return prot


def make_backbone_affine(prot):
  # ziyao: these functions assume jnp inputs.
  n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]
  rot, trans = quat_affine.make_transform_from_reference(
          n_xyz=prot['all_atom_positions'][:, n],
          ca_xyz=prot['all_atom_positions'][:, ca],
          c_xyz=prot['all_atom_positions'][:, c])
  backbone_affine_tensor = quat_affine.QuatAffine(
          quaternion=quat_affine.rot_to_quat(rot, unstack_inputs=True),
          translation=trans,
          rotation=rot,
          unstack_inputs=True).to_tensor()
  backbone_affine_mask = (
          prot['all_atom_mask'][..., n] *
          prot['all_atom_mask'][..., ca] *
          prot['all_atom_mask'][..., c])
  prot['backbone_affine_tensor'] = backbone_affine_tensor
  prot['backbone_affine_mask'] = backbone_affine_mask
  return prot


def make_pseudo_beta(prot):
  # ziyao: this function assumes jnp inputs.
  pseudo_beta_fn = pseudo_beta_fn_jnp
  prot['pseudo_beta'], prot['pseudo_beta_mask'] = (
      pseudo_beta_fn(
          prot['aatype_index'],   # ziyao: 'all_atom_aatype' was actually required. don't know the difference. test needed.
          prot['all_atom_positions'],
          prot['all_atom_mask']))
  return prot


def make_rigidgroups_data(prot):
  """
  ziyao: this function generates all rigid-groups-related keys including:
  [
    'rigidgroups_gt_frames',
    'rigidgroups_gt_exists',
    'rigidgroups_group_exists',
    'rigidgroups_group_is_ambiguous',
    'rigidgroups_alt_gt_frames'
  ]
  """
  # ziyao: this function assumes jnp inputs.
  rigidgroups = atom37_to_frames(
      prot['aatype_index'],
      prot['all_atom_positions'],
      prot['all_atom_mask'])
  prot.update(rigidgroups)
  return prot


def make_torsion_angles(prot):
  aatype_index = np.expand_dims(prot['aatype_index'], 0)
  all_atom_positions = np.expand_dims(prot['all_atom_positions'], 0)
  all_atom_mask = np.expand_dims(prot['all_atom_mask'], 0)
  torsion_angles_dict = atom37_to_torsion_angles(
      aatype=aatype_index,
      all_atom_pos=all_atom_positions,
      all_atom_mask=all_atom_mask,
      placeholder_for_undefined=True)
  chi_angles_sin_cos = torsion_angles_dict['torsion_angles_sin_cos'][0, :, 3:, :]  # [B, NR, 7, 2] -> [NR, 4, 2]
  chi_mask = torsion_angles_dict['torsion_angles_mask'][0, :, 3:]                  # [B, NR, 7]    -> [NR, 4]
  prot['chi_angles_sin_cos'] = chi_angles_sin_cos
  prot['chi_mask'] = chi_mask
  return prot


def to_numpy(prot):
  return {k: np.array(v) for k, v in prot.items()}



def to_jax_numpy(prot):
  return {k: jnp.asarray(v) for k, v in prot.items()}


def remove_keys_in_features(prot):
  # ziyao: remove existing keys in features.
  keys_in_features = [
    "aatype",
    "residue_index",
    "seq_length",
    "is_distillation",
    "seq_mask",
    "msa_mask",
    "msa_row_mask",
    "random_crop_to_size_seed",
    "atom14_atom_exists",
    "residx_atom14_to_atom37",
    "residx_atom37_to_atom14",
    "atom37_atom_exists",
    "extra_msa",
    "extra_msa_mask",
    "extra_msa_row_mask",
    "bert_mask",
    "true_msa",
    "extra_has_deletion",
    "extra_deletion_value",
    "msa_feat",
    "target_feat"
  ]
  prot = {
    k: v for k, v in prot.items() if k not in keys_in_features
  }
  return prot


def add_batch_dim(prot):
  return {k: np.expand_dims(v, 0) for k, v in prot.items()}


map_fns = [
    check_input_completeness,
    make_pseudo_beta,
    make_atom14_data,
    make_backbone_affine,
    make_rigidgroups_data,
    make_torsion_angles,
    to_numpy,                   # ziyao: perhaps optional.
    remove_keys_in_features,
    add_batch_dim             # ziyao: canceled. this is now done in process_features().
                              # weijie: cannot be canceled, need an auxiliary dimension for computing loss.
]


def process_labels(prot: FeatureDict) -> FeatureDict:
  from unifold.model.tf.input_pipeline import process_tensors_from_config
  from unifold.model.tf.proteins_dataset import np_to_tensor_dict
  # ziyao: this function simulates & simplifies the imported function above.
  #        for batch compatibility, a possible solution may be:
  #             1. change np.ndarray to tf.tensor (by adapting np_to_tensor_dict)
  #             2. add them to input tensors of process_tensors_from_config
  #             3. use process_tensors_from_config to make data copies.
  prot = compose(map_fns)(prot)
  return prot

