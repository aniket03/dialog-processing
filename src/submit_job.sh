#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=18000
#SBATCH --job-name=train_unsup_roberta
#SBATCH --mail-type=END
#SBATCH --mail-user=ab8700@nyu.edu
#SBATCH --output=slurm_%j.out

CUDA_VISIBLE_DEVICES=0 python -m tasks.response_eval.train_unsupervised --corpus dd --model roberta --model_size large
--tokenizer roberta --init_lr 3e-6 --batch_size 3 --eval_batch_size 30 --seed 10 --validate_after_n_step 5000
--n_epochs 2 --enable_log True --save_model True
