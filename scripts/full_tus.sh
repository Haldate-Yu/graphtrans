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

cuda1=4
cuda2=5
cuda3=6
cuda4=7

tmux new -s GraphTrans_TUs -d
tmux send-keys "source activate pyg" C-m
tmux send-keys "
bash run_NCI1.sh $cuda1 &
bash run_DD.sh $cuda2 &
bash run_ENZYMES $cuda3 &
bash run_IMDB $cuda4 &
wait" C-m
tmux send-keys "
bash run_REDDIT.sh $cuda1 &
wait" C-m
tmux send-keys "tmux kill-session -t GraphTrans_TUs" C-m
