#!/bin/bash

#SBATCH -A es_chatzi
##SBATCH -G 1
#SBATCH -n 10
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=dmm-discrete-offline
#SBATCH --output=output.txt

source /cluster/apps/local/env2lmod.sh
module load gcc/8.2.0 cuda/11.3.1 cudnn/8.2.1.32

python train_dmm_offline.py \
--n_batch 5000 \
--eps 0.2 \
--length 100 \
--annealing_epochs 10000 \
--learning-rate 0.001 \
--learning-rate-decay 0.99996 \
--beta1 0.95 \
--beta2 0.999 \
--emitter_hidden_dim 100 \
--transition_hidden_dim 100 \
--inference_hidden_dim 100 \
--num_epochs 10000 \
--mini_batch_size 250 \
--minimum_annealing_factor 0.0  \
--use-cuda 1
