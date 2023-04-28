#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=04-00:00:00

sh /home2/esh/miniconda3/etc/profile.d/conda.sh
conda activate nlp 
module load u18/cuda/11.6
cd /home2/esh//smai-proj/Simple-Black-box-Adversarial-Attacks

CUDA_VISIBLE_DEVICES=0
python black-box2.py