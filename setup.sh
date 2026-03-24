echo 'export RM_INTERP_REPO="/home/ubansal/orcd/scratch/gcm-interp"' >> ~/.bashrc
echo '' >> ~/.bashrc
echo 'export HF_TOKEN="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"' >> ~/.bashrc
echo '' >> ~/.bashrc
git clone git@github.com:roonbug/gcm-interp.git

# Download the Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Run the installer
bash miniconda.sh -b -p $HOME/miniconda

$HOME/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$HOME/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Initialize Conda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"

# Activate base environment
conda activate

# (Optional) Add Conda to future shells
echo "eval \"\$($HOME/miniconda/bin/conda shell.bash hook)\"" >> ~/.bashrc

conda create -y -n syc python=3.10

conda activate syc >> ~/.bashrc

cd gcm-interp
python -m pip install -r requirements.txt
python -m pip install -U pyreft
python -m pip install -U transformers

# pyreft copy
if [ -d "/home/ubuntu/pyreft" ] && [ -d "/home/ubuntu/miniconda/envs/syc/lib/python3.10/site-packages/pyreft" ]; then
    echo "Copying pyreft..."
    cp -R /home/ubuntu/pyreft/* /home/ubuntu/miniconda/envs/syc/lib/python3.10/site-packages/pyreft/
else
    echo "Skipping pyreft (directory missing)"
fi

# pyvene copy
if [ -d "/home/ubuntu/pyvene" ] && [ -d "/home/ubuntu/miniconda/envs/syc/lib/python3.10/site-packages/pyvene" ]; then
    echo "Copying pyvene..."
    cp -R /home/ubuntu/pyvene/* /home/ubuntu/miniconda/envs/syc/lib/python3.10/site-packages/pyvene/
else
    echo "Skipping pyvene (directory missing)"
fi
