#!/bin/bash

#SBATCH -A es_chatzi
#SBATCH -G 1
#SBATCH -n 10
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=dmm-
#SBATCH --output=output_1.txt

source /cluster/apps/local/env2lmod.sh
module load gcc/8.2.0 cuda/11.3.1 cudnn/8.2.1.32

python train_dmm_offline.py \
--n_batch 10000 \
--eps 0.2 \
--length 100 \
--annealing_epochs 100000 \
--learning-rate 0.0005 \
--learning-rate-decay 1. \
--beta1 0.9 \
--beta2 0.999 \
--emitter_hidden_dim 100 \
--transition_hidden_dim 100 \
--inference_hidden_dim 100 \
--num_epochs 5000 \
--mini_batch_size 250 \
--minimum_annealing_factor 0.0  \
--use-cuda 1