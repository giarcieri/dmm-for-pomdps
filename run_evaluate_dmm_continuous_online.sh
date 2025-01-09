#!/bin/bash

#SBATCH -A es_chatzi
#SBATCH -G 1
#SBATCH -n 10
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=dmm-cont-onl
#SBATCH --output=output/output_continuous_online.txt

module load stack/2024-04 gcc/8.5.0 cudnn/8.9.7.29-12 cuda/11.8.0

python evaluate_dmm_continuous_online.py \
--number_evaluations 25 \
--number_trials_per_evaluations 500 \
--length 100 \
--annealing_epochs 10 \
--learning_rate 0.0001 \
--learning_rate_decay 0.999 \
--beta1 0.9 \
--beta2 0.999 \
--emitter_hidden_dim 100 \
--transition_hidden_dim 100 \
--inference_hidden_dim 100 \
--num_epochs 10 \
--mini_batch_size 50 \
--minimum_annealing_factor 0.0  \
--power 1 \
--use_cuda 1 \
--elbo gaussian \
--train_last_batch 1 \
--save_evaluation_results 0 \
--workers 10