#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

config='../configs/NCI1/transformer-gnn/no-virtual/gd=128+gdp=0.1+tdp=0.1+l=3+cosine.yml'

#echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
#source ~/.bashrc
#conda activate graph-aug

#echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
#
#echo "python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES=4

# TU datasets test
python ../main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES --epochs 1
