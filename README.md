# Uni-Fold: Training your own deep protein-folding models.

This package provides an implementation of a trainable, Transformer-based deep protein folding model. We modified the open-source code of DeepMind AlphaFold v2.0 and provided code to train the model from scratch. See the [reference](https://doi.org/10.1038/s41586-021-03819-2) and the [repository](https://github.com/deepmind/alphafold) of DeepMind AlphaFold v2.0. To train your own Uni-Fold models, please follow the steps below:

## 1. Install the environment.

Run the following code to install the dependencies of Uni-Fold:

```bash
  conda create -n unifold python=3.8.10 -y
  conda activate unifold
  ./install_dependencies.sh
``` 

Uni-Fold has been tested for Python 3.8.10, CUDA 11.1 and OpenMPI 4.1.1. We recommend using Conda >= 4.10 to install the environment: using Conda with lower versions may lead to some conflicts between packages.

## 2. Prepare data before training.

Before you start to train your own folding models, you shall prepare the features and labels of the training proteins. Features of proteins mainly include the amino acid sequence, MSAs and templates of proteins. These messages should be contained in a pickle file `<name>/features.pkl` for each training protein. Uni-Fold provides scripts to process input FASTA files, relying on several external databases and tools. Labels are CIF files containing the structures of the proteins.

### 2.1 Datasets and external tools.

Uni-Fold adopts the same data processing pipeline as AlphaFold2. We kept the scripts of downloading corresponding databases for searching sequence homologies and templates in the AlphaFold2 repo. Use the command

```bash
  bash scripts/download_all_data.sh /path/to/database/directory
```

to download all required databases of Uni-Fold.

If you successfully installed the Conda environment in Section 1, external tools of search sequence homologies and templates should be installed properly. As an alternative, you can customize the arguments of the feature preparation script (`generate_pkl_features.py`) to refer to your own databases and tools.

### 2.2 Run the preparation code.

An example command of running the feature preparation pipeline would be

```bash
  python generate_pkl_features.py \
    --fasta_dir ./example_data/fasta \
    --output_dir ./out \
    --data_dir /path/to/database/directory \
    --num_workers 1
```

This command automatically processes all FASTA files under `fasta_dir`, and dumps the results to `output_dir`. Note that each FASTA file should contain only one sequence. The default number of CPUs used in hhblits and jackhmmer are 4 and 8. You can modify them in `unifold/data/tools/hhblits.py` and `unifold/data/tools/jackhmmer.py`, respectively.

### 2.3 Organize your training data.

Uni-Fold uses the class [`DataSystem`](./unifold/train/data_system.py) to automatically sample and load the training proteins. To make everything goes right, you shall pay attention to how the training data is organized. Two directories should be established, one with input features (`features.pkl` files, referred to as `features_dir`) and the other with labels (`*.cif` files, referred to as `mmcif_dir`). The feature directory should have its files named as `<pdb_id>_<model_id>_<chain_id>/features.pkl`, e.g. `101m_1_A/features.pkl`, and the label directory should have its files named as `<pdb_id>.cif`, e.g. `101m.cif`. See [`./example_data/features`](./example_data/features) and [`./example_data/mmcif`](./example_data/mmcif) for instances of the two directories. Notably, users shall make sure that all proteins used for training have their corresponding labels. This is checked by `DataSystem.check_completeness()`.

## 3. Train Uni-Fold.

### 3.1 Configuration.
Before you conduct any actual training processes, please make sure that you correctly configured the code. Modify the training configurations in [`unifold/train/train_config.py`](./unifold/train/train_config.py). We annotated the default configurations to reproduce AlphaFold in the script. Specifically, modify the configurations of data paths:
    
  ```json
  "data": {
    "train": {
      "features_dir": "where/training/protein/features/are/stored/",
      "mmcif_dir": "where/training/mmcif/files/are/stored/",
      "sample_weights": "which/specifies/proteins/for/training.json"
    },
    "eval": {
      "features_dir": "where/validation/protein/features/are/stored/",
      "mmcif_dir": "where/validation/mmcif/files/are/stored/",
      "sample_weights": "which/specifies/proteins/for/training.json"
    }
  }
  ```
  
  The specified data should be contained in two folders, namely a `features_dir` and a `mmcif_dir`. Organizations of the two directories are introduced in Section 2.3. Meanwhile, if you want to specify a subset of training data under the directories, or assign customized sample weights for each protein, write a json file and feed its path to `sample_weights`. This is optional, as you can leave it as `None` (and the program will attempt to use all entries under `features_dir` with uniform weights). The json file should be a dictionary containing the basenames of directories of protein features ([pdb_id]\_[model_id]\_[chain_id]) and the sample weight of each protein in the training process (integer or float), such as:

  ```json
  {"1am9_1_C": 82, "1amp_1_A": 291, "1aoj_1_A": 60, "1aoz_1_A": 552}
  ```
  or for uniform sampling, simply using a list of protein entries suffices:

  ```json
  ["1am9_1_C", "1amp_1_A", "1aoj_1_A", "1aoz_1_A"]
  ```

For users who want to customize their own folding models, configurations of model hyperparameters can be edited in [`unifold/model/config.py`](./unifold/model/config.py) .

### 3.2 Run the training code!
To train the model on a single node without MPI, run
```bash
python train.py
```

You can also train the model with multiple GPUs using MPI (or workload managers that supports MPI, such as PBS or Slurm) by running:
```bash
mpirun -n <num_of_gpus> python train.py
```

In either way, make sure you properly configurate the option `use_mpi` and `gpus_per_node` in [`unifold/train/train_config.py`](./unifold/train/train_config.py).

## 4. Infer with trained models.

### 4.1 Infer from features.pkl.

We provide the [`run_from_pkl.py`](./run_from_pkl.py) script to support inferring protein structures from `features.pkl` inputs. A demo command would be

```bash
python run_from_pkl.py \
  --pickle_dir ./example_data/features \
  --model_names unifold \
  --model_paths /path/to/unifold.npz \
  --output_dir ./out
```

or

```bash
python run_from_pkl.py \
  --pickle_paths ./example_data/features/101m_1_A/features.pkl \
  --model_names unifold \
  --model_paths /path/to/unifold.npz \
  --output_dir ./out
```

The command will generate structures (in PDB format) from input features predicted by different input models, the running time of each component, and corresponding residue-wise confidence score (predicted LDDT, or pLDDT).

### 4.2 Infer from FASTA files.

Essentially, inferring the structures from given FASTA files includes two steps, i.e. generating the pickled features and predicting structures from them. We provided a script, [`run_from_fasta.py`](./run_from_fasta.py), as a friendlier user interface. An example usage would be

```bash
python run_from_pkl.py \
  --fasta_paths ./example_data/fasta/101m_1_A.fasta \
  --model_names model_2 \
  --model_paths /path/to/model_2.npz \
  --data_dir /path/to/database/directory
  --output_dir ./out
```

### 4.3 Generate MSAs with MMseqs2.

It may take hours and much memory to generate MSA for sequences, especially for long sequences. In this condition, MMseqs2 may be a more efficient way. It can be used in the following way after it is [installed](https://github.com/soedinglab/mmseqs2/wiki#installation):

```bash

# download and build database
mkdir mmseqs_db && cd mmseqs_db
wget http://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2103.tar.gz
wget http://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz
tar xzvf uniref30_2103.tar.gz
tar xzvf colabfold_envdb_202108.tar.gz
mmseqs tsv2exprofiledb uniref30_2103 uniref30_2103_db
mmseqs tsv2exprofiledb colabfold_envdb_202108 colabfold_envdb_202108_db
mmseqs createindex uniref30_2103_db tmp
mmseqs createindex colabfold_envdb_202108_db tmp
cd ..

# MSA search
./scripts/colabfold_search.sh mmseqs "query.fasta" "mmseqs_db/" "result/" "uniref30_2103_db" "" "colabfold_envdb_202108_db" "1" "0" "1"

```
## 5. Changes from AlphaFold to Uni-Fold.

- We implemented classes and methods for training and inference pipelines by adding scripts under `unifold/train` and `unifold/inference`.
- We added scripts for installing the environment, training and inferencing.
- Files under `unifold/common`, `unifold/data` and `unifold/relax` are minimally altered for re-structuring the repository.
- Files under `unifold/model` are slightly altered to allow mixed-precision training.
- We removed some unused scripts in training AlphaFold model.

## 6. License and disclaimer.

### 6.1 Uni-Fold code license.

Copyright 2021 Beijing DP Technology Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at 
<http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

### 6.2 Use of third-party software.

Use of the third-party software, libraries or code may be governed by separate terms and conditions or license provisions. Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.

### 6.3 Contributing to Uni-Fold.

Uni-Fold is an ongoing project. Our target is to design better protein folding models and to apply them in real scenarios.
We welcome the community to join us in developing the repository together, including but not limited to 1) reports and fixes of bugs,2) new features and 3) better interfaces. Please refer to [`CONTRIBUTING.md`](./CONTRIBUTING.md) for more information.
