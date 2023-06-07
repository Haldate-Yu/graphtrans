#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

#echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
#source ~/.bashrc
#conda activate graph-aug

#echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
#
#echo "python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES"

if [ -z "$1" ]; then
  echo "empty cuda input!"
  cuda=0
else
  cuda=$1
fi

dataset=DD
config='../configs/TUs/base.yml'

# TU datasets
for batch_size in 64 128 256; do
  for nhead in 4 8 16; do
    for d_model in 64 128 256; do
      for num_encoders in 4 8 12; do
        python ../main.py --configs $config --num_workers 8 --dataset $dataset --device $cuda --batch_size $batch_size --nhead $nhead --d_model $d_model --gnn_emb_dim $d_model --num_encoder_layers $num_encoders
      done
    done
  done
done
