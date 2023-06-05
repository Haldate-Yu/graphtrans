#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

config=$1

#echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
#source ~/.bashrc
#conda activate graph-aug

#

# ogbg-molpcba
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES



# ogbg-molhiv
