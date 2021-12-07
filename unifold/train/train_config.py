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

"""Configurations for training Uni-Fold."""

from ml_collections import ConfigDict


train_config = ConfigDict({
    'global_config':{
        # whether you are using MPI communication for multi-gpu training.
        'use_mpi': False,
        # This specifies a model config defined in `unifold/model/config.py`. 
        # 'model_1' to 'model_5' are the settings used in the AlphaFold paper.
        # Setting this config to 'unifold' to reproduce Uni-Fold, or 'demo' 
        # for fast demonstration. You can also customize your own model config 
        # in `unifold/model/config.py` and specify it here.
        'model_name': 'demo',
        # Verbosity of logging messages.
        'verbose': 'info',
        # The number of processes/gpus per node
        'gpus_per_node': 8,
        # The format for autoloading the checkpoint, choose from 'pkl' and 
        # 'npz'. Note that `pkl` format stores necessary variables of 
        # optimizers, yet `npz` saves only model parameters.
        'ckpt_format': 'pkl',
        # Initial step. if > 0, the model will auto-load ckpts from `load_dir`.
        'start_step': 0,                # 0 by default
        # Max steps for training. Accumulated from 'start_step' instead of 0.
        'end_step': 200,                # 80000 in af2
        # Frequency of logging messages and the training loss curve.
        'logging_freq': 10,
        # Frequency of validation.
        'eval_freq': 50,
        # Frequency of saving ckpts.
        'save_freq': 50,
        # Directory to save ckpts. used for auto-saving ckpts.
        'save_dir': './out/ckpt',
        # Directory to load ckpts. used for auto-loading ckpts.
        # ignored if start_step == 0.
        'load_dir': './out/ckpt',
        # Training precision, generally in ['fp32', 'bf16'].
        # Set for mixed precision training.
        'precision': 'fp32',
        # Max queue size. Specifies the queue size of the pre-processed
        # batches. Generally has little impact on code efficiency.
        'max_queue_size': 16,
        # Random seed for initializing model parameters. Ignored when attempting to auto load ckpts.
        'random_seed': 181129
    },
    'optimizer': {
        # Optimizer class.
        'name': 'adam',                 # in ['adam', 'sgd', ...]
        # Learning rate. if warm up steps > 0, this specifies the peak learning rate. 
        'learning_rate': 1e-3,          # 1e-3 in af2
        # The number of warm-up steps.
        'warm_up_steps': 10,            # 1000 in af2
        # Learning rate decay configs.
        'decay':{
            'name': 'exp',              # only 'exp' supported
            'decay_rate': 0.95,         # 0.95 in af2
            'decay_steps': 10           # 5000? in af2
        },
        # Global clip norm of gradients.
        'clip_norm': 1e-1,
    },
    'data':{
        'train': {
            # Directory to store features (features.pkl files)
            'features_dir': "./example_data/features",
            # Directory to store labels (.mmcif files)
            'mmcif_dir': "./example_data/mmcif",
            # Json file that specifies sampling weights of each sample.
            'sample_weights': "./example_data/sample_weights.json"
        },
        'eval': {
            # Directory to store features (features.pkl files)
            'features_dir': "./example_data/features",
            # Directory to store labels (.mmcif files)
            'mmcif_dir': "./example_data/mmcif",
            # Json file that specifies sampling weights of each sample.
            'sample_weights': "./example_data/sample_weights.json"
        },
    }
})
