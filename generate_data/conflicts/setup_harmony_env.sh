#!/bin/bash
# setup_harmony_env.sh
#
# Creates a NEW conda env (separate from `syc`) with vLLM + openai-harmony,
# for running gpt-oss-20b / harmony-format inference. Doesn't touch `syc`.
#
# Run this on a LOGIN NODE (needs internet: PyPI for packages, and
# openaipublic.blob.core.windows.net to fetch+cache the harmony tokenizer
# vocab file, which compute nodes usually can't reach). Once cached, compute
# nodes can load the vocab from local disk with no network needed.
#
# Usage:
#   bash setup_harmony_env.sh
#   bash setup_harmony_env.sh my_custom_env_name

set -euo pipefail

ENV_NAME="${1:-gpt_oss_harmony}"
PYTHON_VERSION="3.11"

echo "Creating conda env '$ENV_NAME' (python $PYTHON_VERSION)..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

source ~/miniconda/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

echo "Installing vllm and openai-harmony into '$ENV_NAME' (not touching syc)..."
pip install --upgrade pip
pip install vllm
pip install openai-harmony

echo
echo "--- Verifying installs ---"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import openai_harmony; print('openai_harmony import OK')"

echo
echo "--- Pre-warming the harmony tokenizer vocab cache (needs internet; do this now so compute nodes can run offline) ---"
python -c "
from openai_harmony import load_harmony_encoding, HarmonyEncodingName
load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
print('harmony vocab cached OK -- compute nodes can now load this from local cache without network access')
"

echo
echo "Done. '$ENV_NAME' is separate from 'syc' -- your existing pipeline is untouched."
echo "Activate it with: conda activate $ENV_NAME"
echo "Remember to update your sbatch script to 'conda activate $ENV_NAME' instead of 'conda activate syc'."
