#!/bin/bash
#
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

# Shell script for installing the environment.
# Usage: bash install_dependencies.sh

#######################################
# dependencies of feature processing  #
#######################################

# install conda packages
conda install -y -c conda-forge openmm=7.5.1 pdbfixer cudatoolkit=11.1
conda install -y -c bioconda hmmer hhsuite==3.3.0 kalign2

# update openmm
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SITE_PACKAGES_DIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" # path to /path/to/conda/envs/XXX/lib/pythonX.Y/site-packages
patch -d "${SITE_PACKAGES_DIR}" -p0 <"${SCRIPT_DIR}/openmm.patch"

#######################################
# dependencies of training Uni-Fold   #
#######################################

# install openmpi
wget -O - https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz |
  tar -xz --strip-components=1 --one-top-level=openmpi &&
  (
    cd openmpi &&
      ./configure --prefix=/usr/local &&
      make all install
  )

# install conda and pip packages
conda install -y -c nvidia cudnn==8.0.4
pip install --upgrade pip &&
  pip install -r ./requirements.txt &&
  pip install jaxlib==0.1.67+cuda111 -f \
    https://storage.googleapis.com/jax-releases/jax_releases.html
