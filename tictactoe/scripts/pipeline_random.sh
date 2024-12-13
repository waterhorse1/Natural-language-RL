#!/bin/bash
set -ex
export PYTHONPATH="$PWD:$PYTHONPATH"

# export CUDA_VISIBLE_DEVICES=4,5,6,7
TOP_K_SAMPLE=10
NUM_POLICY_SAMPLE=10
N_MC_TRAJ=5
NUM_ROLLOUTS=512
OPPONENT_POLICY_NAME="Random"
NUM_TRAIN_EPOCH=2
N_HISTORY=3
EXP_BASE_DIR=./results-tictactoe-random/exp
exp_dir="${EXP_BASE_DIR}/oppo_${OPPONENT_POLICY_NAME}_train_${NUM_TRAIN_EPOCH}_hist_${N_HISTORY}_sample_${NUM_POLICY_SAMPLE}_top_${TOP_K_SAMPLE}_nmc_${N_MC_TRAJ}_rollout_${NUM_ROLLOUTS}"
BASE_EVAL_TRAJ_PATH="${exp_dir}/data/eval/replay_buffer"

EVAL_TRAJ_PATH="${BASE_EVAL_TRAJ_PATH}_0.jsonl"
python3 pipeline/collect_rollout_data.py \
    --policy_name "Random" \
    --opponent_policy_name $OPPONENT_POLICY_NAME \
    --replay_buffer_path $EVAL_TRAJ_PATH \
    --rollout_method scratch \
    --num_rollouts $NUM_ROLLOUTS \
    --model_path "" \
    --epsilon_greedy 0 --temp 0
python nlrl/evaluate.py --data_dir ${exp_dir}/data/eval