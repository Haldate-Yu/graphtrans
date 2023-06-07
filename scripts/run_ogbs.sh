#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

#echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
#source ~/.bashrc
#conda activate graph-aug

############
# Usage
############

# python main.py --configs xxx.yml --runs 5
# 2, 3卡可用

############
# HIV - 10 RUNS
############

config='../configs/molhiv/gnn-transformer/no-virtual/JK=cat/pooling=cls+gin+norm_input.yml'

tmux new -s GraphTrans_OGBG -d
tmux send-keys "source activate pyg" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
wait" C-m

############
# pcba - 5 RUNS
############

config='configs/molpcba/gnn-transformer/JK=cat/pooling=cls+gin+norm_input.yml'

tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 4 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 4 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 4 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 8 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 8 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 8 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 16 --d_model 64 --gnn_emb_dim 64 --num_encoder_layers 12 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 4 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 8 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 16 --d_model 128 --gnn_emb_dim 128 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 4 &
python ../main.py --configs $config --num_workers 8 --device 3 --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 8 &
wait" C-m
tmux send-keys "
python ../main.py --configs $config --num_workers 8 --device 2 --nhead 16 --d_model 256 --gnn_emb_dim 256 --num_encoder_layers 12 &
wait" C-m
tmux send-keys "tmux kill-session -t GraphTrans_TUs" C-m
