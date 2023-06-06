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

############
# Usage
############

# python main.py --configs xxx.yml --runs 5


############
# TUs - 10 RUNS
############


config='../configs/TUs/base.yml'

tmux new -s GraphTrans_TUs -d
tmux send-keys "source activate pyg" C-m
# NCI1
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset NCI1 --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset NCI1 --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset NCI1 --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset NCI1 --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset NCI1 --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset NCI1 --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset NCI1 --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset NCI1 --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset NCI1 --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset NCI1 --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset NCI1 --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset NCI1 --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset NCI1 --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset NCI1 --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset NCI1 --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset NCI1 --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset NCI1 --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset NCI1 --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset NCI1 --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset NCI1 --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset NCI1 --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset NCI1 --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset NCI1 --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset NCI1 --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset NCI1 --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset NCI1 --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset NCI1 --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
wait" C-m
# DD
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset DD --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset DD --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset DD --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset DD --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset DD --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset DD --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset DD --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset DD --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset DD --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset DD --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset DD --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset DD --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset DD --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset DD --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset DD --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset DD --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset DD --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset DD --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset DD --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset DD --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset DD --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset DD --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset DD --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset DD --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset DD --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset DD --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset DD --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
wait" C-m
# ENZYMES
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset ENZYMES --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset ENZYMES --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset ENZYMES --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset ENZYMES --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset ENZYMES --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset ENZYMES --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset ENZYMES --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset ENZYMES --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset ENZYMES --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset ENZYMES --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset ENZYMES --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset ENZYMES --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset ENZYMES --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset ENZYMES --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset ENZYMES --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset ENZYMES --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset ENZYMES --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset ENZYMES --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset ENZYMES --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset ENZYMES --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset ENZYMES --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset ENZYMES --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset ENZYMES --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset ENZYMES --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset ENZYMES --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset ENZYMES --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset ENZYMES --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
wait" C-m
# IMDB-BINARY
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset IMDB-BINARY --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset IMDB-BINARY --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset IMDB-BINARY --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset IMDB-BINARY --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset IMDB-BINARY --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset IMDB-BINARY --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset IMDB-BINARY --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset IMDB-BINARY --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset IMDB-BINARY --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset IMDB-BINARY --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset IMDB-BINARY --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset IMDB-BINARY --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset IMDB-BINARY --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset IMDB-BINARY --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset IMDB-BINARY --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset IMDB-BINARY --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset IMDB-BINARY --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset IMDB-BINARY --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset IMDB-BINARY --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset IMDB-BINARY --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset IMDB-BINARY --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset IMDB-BINARY --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset IMDB-BINARY --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset IMDB-BINARY --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset IMDB-BINARY --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset IMDB-BINARY --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset IMDB-BINARY --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
wait" C-m
# REDDIT
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset REDDIT-BINARY --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset REDDIT-BINARY --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset REDDIT-BINARY --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset REDDIT-BINARY --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset REDDIT-BINARY --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset REDDIT-BINARY --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset REDDIT-BINARY --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset REDDIT-BINARY --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset REDDIT-BINARY --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset REDDIT-BINARY --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset REDDIT-BINARY --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset REDDIT-BINARY --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset REDDIT-BINARY --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset REDDIT-BINARY --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset REDDIT-BINARY --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset REDDIT-BINARY --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset REDDIT-BINARY --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset REDDIT-BINARY --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset REDDIT-BINARY --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset REDDIT-BINARY --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset REDDIT-BINARY --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset REDDIT-BINARY --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset REDDIT-BINARY --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 7 --dataset REDDIT-BINARY --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 4 --dataset REDDIT-BINARY --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 5 --dataset REDDIT-BINARY --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 6 --dataset REDDIT-BINARY --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "tmux kill-session -t GraphTrans_TUs" C-m
