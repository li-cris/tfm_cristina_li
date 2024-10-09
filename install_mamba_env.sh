#!/usr/bin/bash

#? Set strict mode
set -euxo pipefail

#? Define constants
CONDA_INSTALL_DIR=/home/adhisant/tmp/miniforge3 #? <--- Change this value
CONDA_PATH=${CONDA_INSTALL_DIR}/etc/profile.d/conda.sh
MAMBA_PATH=${CONDA_INSTALL_DIR}/etc/profile.d/mamba.sh
MAMBA_BIN=mamba
ENV_NAME=transmet-develop
PYTHON_VER=3.8

#? Define CPLEX and COMPASS installation settings
CPLEX_BIN_FPATH=/data/gidb/shared/bins/cplex_studio2211.linux_x86_64.bin
CPLEX_INSTALL_PATH=${PWD}/tmp/libs/cplex
COMPASS_GIT_URL="https://github.com/YosefLab/Compass.git"

#? Load conda and mamba environments
set +ux
source ${CONDA_PATH}
unset __conda_setup
source ${MAMBA_PATH}
set -ux

#? Create and activate environment
if ! ${MAMBA_BIN} create -y -n "${ENV_NAME}" "conda-forge::openjdk" "python=${PYTHON_VER}"; then
  echo "Failed to create environment ${ENV_NAME}"
  exit 1
fi
if ! ${MAMBA_BIN} activate "${ENV_NAME}"; then
  echo "Failed to activate environment ${ENV_NAME}"
  exit 1
fi

#? Check if environment is active
if [ -z "${CONDA_DEFAULT_ENV}" ] || [ "${CONDA_DEFAULT_ENV}" = "base" ]; then
  echo "Environment is not active"
  exit 1
fi

#? Remove existing CPLEX installation (if any)
if [ -d "${CPLEX_INSTALL_PATH}" ]; then
  echo "Removing existing CPLEX installation at ${CPLEX_INSTALL_PATH}"
  rm -rf "${CPLEX_INSTALL_PATH}"
fi

#? Install CPLEX
if ! bash "${CPLEX_BIN_FPATH}" -i silent -DINSTALLER_UI=silent -DLICENSE_ACCEPTED=TRUE -DUSER_INSTALL_DIR="${CPLEX_INSTALL_PATH}"; then
  echo "Failed to install CPLEX"
  exit 1
fi

#? Install CPLEX Python package
if ! python "${CPLEX_INSTALL_PATH}/python/setup.py" install; then
  echo "Failed to install CPLEX Python package"
  exit 1
fi

#? Install COMPASS
if ! pip install git+${COMPASS_GIT_URL}; then
  echo "Failed to install COMPASS"
  exit 1
fi

pip install -r requirements.txt