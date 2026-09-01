#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -t 01:00:00
#SBATCH -J pos_loc
#SBATCH -o /home/ubansal/orcd/scratch/conflicts/gcm-interp/logs/%x_%j.out
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH -c 4

source ~/.bashrc
# Was /home/ubansal/orcd/scratch/gcm-interp -- a DIFFERENT checkout from the one
# cd'd into below. Anything reading this variable was pointed at the wrong repo.
export RM_INTERP_REPO="/home/ubansal/orcd/scratch/conflicts/gcm-interp"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export TOKENIZERS_PARALLELISM="false"
export ATP_MEM=1
# Fragmentation: the activation cache allocates and frees large blocks per
# batch, and the allocator ends up unable to satisfy a 1.3GB request with
# hundreds of MB nominally free.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Activation-caching batch size. The cache holds batch x seq x hidden for every
# layer at once, so prompt length costs the same as batch size: the rule-form
# corpora are about twice as long as the request-form ones the default of 9 was
# tuned for, which OOMs a 140GB card. Lower it further if a cell still fails.
export CACHE_BS="${CACHE_BS:-3}"
source /home/ubansal/miniconda/etc/profile.d/conda.sh
conda activate /home/ubansal/miniconda/envs/conflict-syc
cd "$RM_INTERP_REPO"
mkdir -p logs

declare -a models=("openai/gpt-oss-20b")
algos=("atp")

for ARM in devuser sysuser sysdev; do
  PYTHONPATH=generate_data/conflict python make_position_datasets.py --arm $ARM \
    --from_corpus data/gpt-oss-20b/$ARM \
    --out_dir data/gpt-oss-20b/pos-$ARM
done

for ARM in devuser sysuser sysdev; do
  PYTHONPATH=generate_data/conflict python run.py --model_id openai/gpt-oss-20b --patch_algo atp \
    --data_dir pos-$ARM --source user-single --base dev-single \
    --device cuda:0 --patch_model --batch_size 1
done
