#!/bin/bash
#SBATCH -J "base"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=azm0269@auburn.edu
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH -D /aiau010_scratch/azm0269/federated_reasoning
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=1-05:00:00
#SBATCH --nodelist=aiau001
#SBATCH --gres=gpu:1

# module load python3

HF_HOME="/aiau010_scratch/azm0269/hub"
env_dir=/aiau010_scratch/azm0269/
cd $env_dir
source /aiau010_scratch/azm0269/.venv/bin/activate
workdir=/aiau010_scratch/azm0269/federated_reasoning
cd $workdir
source .env
HF_HOME="/aiau010_scratch/azm0269/hub"
HF_TOKEN="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"
HF_HOME="/aiau010_scratch/azm0269/hub"
# CUDA_VISIBLE_DEVICES=6,7
# TOKENIZERS_PARALLELISM=true

# for baseline
nohup srun python3 -m run_experiments > qwen_training_baseline.log 2>&1
wait

deactivate