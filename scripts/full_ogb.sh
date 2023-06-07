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
###############################
# One Key To RULE THEM ALL!!!
###############################

cuda1=2
cuda2=3


tmux new -s GraphTrans_OGBG -d
tmux send-keys "source activate pyg" C-m
tmux send-keys "
bash run_HIV.sh $cuda1 &
bash run_PCBA.sh $cuda2 &
wait" C-m
tmux send-keys "tmux kill-session -t GraphTrans_OGBG" C-m
