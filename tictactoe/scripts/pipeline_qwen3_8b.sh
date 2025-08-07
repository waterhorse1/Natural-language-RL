#!/bin/bash
# Pipeline script for running NLAC with Qwen3-8B model
# This script demonstrates how to use Qwen3 models with the Natural Language RL framework

set -ex
export PYTHONPATH="$PWD:$PYTHONPATH"

# Set Qwen3 model paths
export SMALL_LLM_PATH="Qwen/Qwen3-8B"
export BIG_LLM_PATH="Qwen/Qwen3-8B"  # You can use a larger Qwen model here if available

# Get script arguments with defaults
TOP_K_SAMPLE=${1:-10}
NUM_POLICY_SAMPLE=${2:-10}
N_MC_TRAJ=${3:-5}
NUM_ROLLOUTS=${4:-512}
OPPONENT_POLICY_NAME=${5:-"Random"}
NUM_TRAIN_EPOCH=${6:-2}
NUM_HISTORY=${7:-3}
EXP_DATE=${8:-$(date +%Y%m%d)}
START_ITERATION=${9:-1}
END_ITERATION=${10:-31}

echo "Running Qwen3-8B pipeline with:"
echo "- Model: $SMALL_LLM_PATH"
echo "- Top K: $TOP_K_SAMPLE"
echo "- Policy samples: $NUM_POLICY_SAMPLE"
echo "- MC trajectories: $N_MC_TRAJ"
echo "- Rollouts: $NUM_ROLLOUTS"
echo "- Opponent: $OPPONENT_POLICY_NAME"
echo "- Training epochs: $NUM_TRAIN_EPOCH"
echo "- History: $NUM_HISTORY"
echo "- Iterations: $START_ITERATION to $END_ITERATION"

# Run the main pipeline with Qwen3 models
bash tictactoe/scripts/pipeline_nlac.sh \
    $TOP_K_SAMPLE \
    $NUM_POLICY_SAMPLE \
    $N_MC_TRAJ \
    $NUM_ROLLOUTS \
    $OPPONENT_POLICY_NAME \
    $NUM_TRAIN_EPOCH \
    $NUM_HISTORY \
    "qwen3_${EXP_DATE}" \
    $START_ITERATION \
    $END_ITERATION

echo "Qwen3 pipeline completed successfully!"