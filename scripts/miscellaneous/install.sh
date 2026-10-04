#!/bin/bash
# Install the runtime environment. Run from the repository root, either directly
# on a GPU node or via `sbatch scripts/miscellaneous/install.sh`.

#SBATCH --job-name=install
#SBATCH --output=logs/%x/%j.log
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

# Load site configuration (run from the repository root)
source scripts/config.sh

set -euo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"

conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install --no-build-isolation flash_attn==2.6.3

pushd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
popd

pushd TransformerEngine
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvtx/include:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/include
export C_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvtx/include:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/include
export CUDNN_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn
export NVTE_FRAMEWORK=pytorch
export MAX_JOBS=16
pip install --no-build-isolation .
popd

pip install -e lm-evaluation-harness
