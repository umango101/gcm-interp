# make conda available (your miniconda root still exists)
source ~/miniconda/etc/profile.d/conda.sh

# clear the stray user-site that kept leaking python3.12 packages
rm -rf ~/.local/lib/python3.12

# build the one canonical env, at the path your sbatch scripts reference
conda create -y -p ~/miniconda/envs/syc python=3.10
conda activate ~/miniconda/envs/syc
export PYTHONNOUSERSITE=1

# (torch 2.7.0 is already in — this just confirms it)
python -m pip install torch==2.7.0
python -c "import torch; print(torch.__version__)"   # expect 2.7.0+cu126

cd /orcd/scratch/orcd/008/ubansal/gcm-interp

# strip every hand-pinned backend lib + pyreft/flash-attn; keep everything else
grep -vE '^(nvidia-|triton|torch($|[<>=])|pyreft|flash-attn)' requirements.txt > /tmp/req.txt
python -m pip install -r /tmp/req.txt

# pyreft WITHOUT flash-attn — its real deps (transformers, pyvene, torch) are already installed
python -m pip install --no-deps pyreft

# sanity
python -c "import torch, transformers, numpy, yaml; print(torch.__version__, transformers.__version__)"