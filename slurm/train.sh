#!/bin/bash
#SBATCH --job-name=ficbo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gpus=1

# Example SLURM job for one training run. Adapt partition, modules and env to your cluster.
# All arguments are passed to train_ficbo.py as Hydra overrides. Run from the repo root:
#
#   mkdir -p logs
#   base=$(sbatch --parsable slurm/train.sh --config-name train task=general_gp_prior \
#       file_name=gp_icbo_base.pth hydra.run.dir=outputs/runs/gp_icbo_base)
#   sbatch --dependency=afterok:$base slurm/train.sh --config-name train_main task=ficbo_prior \
#       file_name=ficbo_mixture.pth hydra.run.dir=outputs/runs/ficbo_mixture \
#       expert_path=outputs/runs/gp_icbo_base/model/gp_icbo_base.pth

eval "$(conda shell.bash hook)"
conda activate ficbo
export HYDRA_FULL_ERROR=1

python train_ficbo.py "$@"
