#!/bin/bash

#SBATCH -A es_chatzi
#SBATCH -G 1
#SBATCH -n 10
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=dmm-cont-onl
#SBATCH --output=output/output_continuous_online.txt

source /cluster/apps/local/env2lmod.sh
module load gcc/8.2.0 cuda/11.3.1 cudnn/8.2.1.32

python evaluate_dmm_continuous_online.py \
--number_evaluations 5 \
--number_trials_per_evaluations 500 \
--length 100 \
--annealing_epochs 1000 \
--learning_rate 0.0001 \
--learning_rate_decay 1. \
--beta1 0.9 \
--beta2 0.999 \
--emitter_hidden_dim 100 \
--transition_hidden_dim 100 \
--inference_hidden_dim 100 \
--num_epochs 1000 \
--mini_batch_size 50 \
--minimum_annealing_factor 0.0  \
--power 1 \
--use_cuda 1 \
--elbo gaussian \
--train_last_batch 0 \
--save_evaluation_results 0 \
--workers 10