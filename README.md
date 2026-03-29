# README

## Setup

**Switch to the correct branch**
- Type `git checkout cleanup` into your terminal


**Modify setup.sh**
- Replace <PATH-TO-RM-INTERP-REPO> with the path to the gcm-interp repo on your machine
- Replace <HF-TOKEN> with your HF-TOKEN


## Localization

**Modify the localization script**
- In the `scripts` folder, identify `localize.sh`
- Replace the file path in `RM_INTERP_REPO` with the path to the gcm-interp repo on your machine (there are two locations in the same file where this is defined)
- Replace the file path in the `conda activate` command with the path to your miniconda syc file
- Replace the file path in the `cd` command with the path to the gcm-interp repo on your machine
- Replace the pairs in the `declare -a pairs` command with the experiment pairs in the format `source-dataset_base-dataset` (`source-dataset` is the behavior you want to induce and `base-dataset` is the baseline behavior of the model)
- Replace `mit_preemptable` in the `#SBATCH -p` command with the cluster you plan to use
- Replace `q-vp-l` in the `SBATCH -J` command with a short identifier for the experiment
- Replace `gpu:h100:1` in the `#SBATCH --gres=` command with the required GPU type and number
- Replace `48G` in the `#SBATCH --mem=` command with the required memory


**Run the localization script**
- In terminal, run the localization script by running `sbatch scripts/localize.sh`
- Confirm the script is running with this command: `squeue --me`
- You can remove a script from the cluster by running `scancel -r [username]`


## Evaluation

**Modify the evaluation script**
- In the `scripts` folder, identify `eval.sh`
- Replace the file path in `RM_INTERP_REPO` with the path to the gcm-interp repo on your machine (there are two locations in the same file where this is defined)
- Replace the file path in the `conda activate` command with the path to your miniconda syc file
- Replace the file path in the `cd` command with the path to the gcm-interp repo on your machine
- Replace the pairs in the `declare -a pairs` command with the experiment pairs in the format `source-dataset_base-dataset` -- *there should be only one pair per eval script*
- Replace the file paths in the `--eval_test`, `steering_add_path`, and `steering_sub_path` with the paths to the `base-test-dataset`, `source-desired-all-dataset`, and `base-desired-all-dataset` for all invocations of the `run.py` script
- Replace `mit_preemptable` in the `#SBATCH -p` command with the cluster you plan to use
- Replace `q-vp-l` in the `SBATCH -J` command with a short identifier for the experiment
- Replace `gpu:h100:1` in the `#SBATCH --gres=` command with the required GPU type and number
- Replace `48G` in the `#SBATCH --mem=` command with the required memory

**Run the evaluation script**
- In terminal, run the evaluation script by running `sbatch scripts/eval.sh`
- Confirm the script is running with this command: `squeue --me`
- You can remove a script from the cluster by running `scancel -r [username]`
- If running a long script, you can retrigger the cluster to run the script at a later time by running the command `sbatch --begin=now+740minutes scripts/eval.sh`
