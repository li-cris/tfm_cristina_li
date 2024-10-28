#!/bin/bash

#SBATCH --mail-user=adhisant@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=transmet-norman
#SBATCH --array=0-64
#SBATCH --mem=15G
#SBATCH --cpus-per-task=8

set -euxo pipefail

CONDA_PATH=<CONDA_PATH> #? Path to conda.sh
MAMBA_PATH=<MAMBA_PATH> #? Path to mamba.sh
MAMBA_BIN=<MAMBA_BIN> #? Binary name
ENV_NAME=<ENV_NAME> #? Environment name

WORKING_DIR=<REPO_PATH> #? Path to Repo
DATA_DIR=<DATA_PATH> #? Path to dataset
CACHE_DIR=<TMP_PATH> #? path to temporary data
OUTPUT_DIR=<OUT_PATH> #? Path to output

set +x
source ${CONDA_PATH}
unset __conda_setup
source ${MAMBA_PATH}
${MAMBA_BIN} activate ${ENV_NAME}
set -x

cd $WORKING_DIR

CURR_CACHE_DIR="${CACHE_DIR}/${SLURM_ARRAY_TASK_ID}"
mkdir -p ${CURR_CACHE_DIR}

CURR_OUTPUT_DIR="${OUTPUT_DIR}/${SLURM_ARRAY_TASK_ID}"
mkdir -p ${CURR_OUTPUT_DIR}

compass \
        --num-process 8 \
        --data "${DATA_DIR}/expression.${SLURM_ARRAY_TASK_ID}.tsv" \
        --species "homo_sapiens" \
        --temp-dir "${CURR_CACHE_DIR}" \
        --output-dir "${CURR_OUTPUT_DIR}"
