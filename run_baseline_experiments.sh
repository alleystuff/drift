#!/bin/bash
#SBATCH -J "base"
#SBATCH --mail-type=ALL
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=1-05:00:00
#SBATCH --nodelist=aiau001
#SBATCH --gres=gpu:1

# module load python3

# CUDA_VISIBLE_DEVICES=6,7
# TOKENIZERS_PARALLELISM=true

# for baseline
nohup srun python3 -m run_experiments > qwen_training_baseline.log 2>&1
wait

deactivate
