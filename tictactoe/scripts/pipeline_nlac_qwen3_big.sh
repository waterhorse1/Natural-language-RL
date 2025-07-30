#!/bin/bash

set -ex
export PYTHONPATH="$PWD:$PYTHONPATH"

# export CUDA_VISIBLE_DEVICES=4,5,6,7
TOP_K_SAMPLE=$1
NUM_POLICY_SAMPLE=${2:-10}
N_MC_TRAJ=${3:-5}
NUM_ROLLOUTS=${4:-512}
OPPONENT_POLICY_NAME=${5:-"Random"}
NUM_TRAIN_EPOCH=${6:-2}
NUM_HISTORY=${7:-3}
EXP_BASE_DIR=./results-tictactoe-nlrl/${8:-"exp"}
exp_dir="${EXP_BASE_DIR}/oppo_${OPPONENT_POLICY_NAME}_train_${NUM_TRAIN_EPOCH}_hist_${NUM_HISTORY}_sample_${NUM_POLICY_SAMPLE}_top_${TOP_K_SAMPLE}_nmc_${N_MC_TRAJ}_rollout_${NUM_ROLLOUTS}"

# Base paths
BASE_POLICY_MODEL_PATH="${exp_dir}/models/policy"
BASE_VALUE_MODEL_PATH="${exp_dir}/models/value"
BASE_REPLAY_BUFFER_PATH="${exp_dir}/data/replay_buffer"
BASE_Q_TARGET_PATH="${exp_dir}/data/q_target"
BASE_IMPROVE_TARGET_PATH="${exp_dir}/data/improve_target"
BASE_EVAL_TRAJ_PATH="${exp_dir}/data/eval/replay_buffer"
BATCH_SIZE=10000

START_ITERATION_NUM=${9:-1}
FINAL_ITERATION_NUM=${10:-31}

echo start: $START_ITERATION_NUM end: $FINAL_ITERATION_NUM

mkdir -p ${exp_dir}/data/
mkdir -p ${exp_dir}/models/

if [ $START_ITERATION_NUM -gt 1 ]; then
    POLICY_MODEL_PATH="${BASE_POLICY_MODEL_PATH}_$(($START_ITERATION_NUM-1))"
    VALUE_MODEL_PATH="${BASE_VALUE_MODEL_PATH}_$(($START_ITERATION_NUM-1))"
    POLICY_CHECKPOINT=$(find $POLICY_MODEL_PATH -type d -name 'checkpoint*' | sort | head -n 1)
    VALUE_CHECKPOINT=$(find $VALUE_MODEL_PATH -type d -name 'checkpoint*' | sort | head -n 1)
fi
echo $POLICY_CHECKPOINT

SMALL_LLM_NAME=${SMALL_LLM_PATH:-"path/to/llama-3.1-8b"}
BIG_LLM_NAME=${BIG_LLM_PATH:-"Qwen/Qwen3-8B"}

QWEN3_TEMP=0.6
QWEN3_TOP_K=20
QWEN3_TOP_P=0.95

if [[ $SMALL_LLM_NAME == *"qwen"* ]] || [[ $SMALL_LLM_NAME == *"Qwen"* ]]; then
    FSDP_TRANSFORMER_LAYER="Qwen2DecoderLayer"
else
    FSDP_TRANSFORMER_LAYER="LlamaDecoderLayer"
fi

for i in $(seq $START_ITERATION_NUM $FINAL_ITERATION_NUM)
do
    echo $i
    if [ $i -eq 1 ]; then
        POLICY_NAME="Random"
        ROLLOUT_POLICY_MODEL_PATH=$SMALL_LLM_NAME
        POLICY_TRAIN_START=None
        VALUE_TRAIN_START=None
        EVAL_TRAJ_PATH="${BASE_EVAL_TRAJ_PATH}_0.jsonl"
        python3 tictactoe/collect_rollout_data.py --policy_name "LLM" --opponent_policy_name $OPPONENT_POLICY_NAME --replay_buffer_path $EVAL_TRAJ_PATH --rollout_method scratch --num_rollouts $NUM_ROLLOUTS --model_path $ROLLOUT_POLICY_MODEL_PATH --epsilon_greedy 0 --temp 0
    else
        OLD_POLICY_MODEL_PATH="${BASE_POLICY_MODEL_PATH}_$(($i-1))"
        OLD_VALUE_MODEL_PATH="${BASE_VALUE_MODEL_PATH}_$(($i-1))"
        echo $OLD_POLICY_MODEL_PATH
        echo $OLD_VALUE_MODEL_PATH
        POLICY_CHECKPOINT=$(find $OLD_POLICY_MODEL_PATH -type d -name 'checkpoint*' | sort | head -n 1)
        VALUE_CHECKPOINT=$(find $OLD_VALUE_MODEL_PATH -type d -name 'checkpoint*' | sort | head -n 1)
        POLICY_NAME="LLM"
        ROLLOUT_POLICY_MODEL_PATH=$POLICY_CHECKPOINT
        POLICY_TRAIN_START=$POLICY_CHECKPOINT
        VALUE_TRAIN_START=$VALUE_CHECKPOINT
    fi
    POLICY_MODEL_PATH="${BASE_POLICY_MODEL_PATH}_${i}"
    VALUE_MODEL_PATH="${BASE_VALUE_MODEL_PATH}_${i}"
    REPLY_BUFFER_PATH="${BASE_REPLAY_BUFFER_PATH}_${i}.jsonl"
    EVAL_TRAJ_PATH="${BASE_EVAL_TRAJ_PATH}_${i}.jsonl"
    Q_TARGET_PATH="${BASE_Q_TARGET_PATH}_${i}.jsonl"
    Q_TRAIN_PATH="${BASE_Q_TARGET_PATH}_for_train_${i}.jsonl"
    IMPROVE_TARGET_PATH="${BASE_IMPROVE_TARGET_PATH}_${i}.jsonl"
    IMPROVE_TRAIN_PATH="${BASE_IMPROVE_TARGET_PATH}_for_train_${i}.jsonl"
    
    python3 tictactoe/collect_rollout_data.py --policy_name $POLICY_NAME --opponent_policy_name $OPPONENT_POLICY_NAME --replay_buffer_path $REPLY_BUFFER_PATH --rollout_method scratch --num_rollouts $NUM_ROLLOUTS --model_path $ROLLOUT_POLICY_MODEL_PATH
    
    python3 tictactoe/prompt_llm.py --method mc_value_q --max_tokens=4096 --model_path $BIG_LLM_NAME --batch_size $BATCH_SIZE --input_path $REPLY_BUFFER_PATH --output_path $Q_TARGET_PATH --n_mc_trajs $N_MC_TRAJ --temp $QWEN3_TEMP --top_k $QWEN3_TOP_K --top_p $QWEN3_TOP_P
    
    python3 tictactoe/data_for_train.py --data_path $Q_TARGET_PATH --output_path $Q_TRAIN_PATH --method mc_value_q

    Q_TRAIN_PATH_MERGE="${BASE_Q_TARGET_PATH}_for_train_merged.jsonl"
    python3 tictactoe/merge_train_data.py --data_path $Q_TRAIN_PATH --output_path $Q_TRAIN_PATH_MERGE --history $NUM_HISTORY
    torchrun --nproc_per_node=8 --master_port=20002 tictactoe/train/train_sft.py \
        --model_name_or_path=$SMALL_LLM_NAME \
        --train_dataset_path=$Q_TRAIN_PATH_MERGE \
        --max_seq_length=1024 \
        --eval_dataset_path=None \
        --report_to="tensorboard" \
        --learning_rate=1e-5 \
        --torch_dtype="bfloat16" \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "constant" \
        --per_device_train_batch_size=8 \
        --max_seq_length 1024 \
        --gradient_accumulation_steps=2 \
        --attn_implementation=sdpa \
        --output_dir=$VALUE_MODEL_PATH \
        --logging_steps=1 \
        --num_train_epoch=$NUM_TRAIN_EPOCH \
        --save_strategy "steps" \
        --save_steps 0.999 \
        --max_steps=-1 \
        --gradient_checkpointing \
        --bf16 \
        --evaluation_strategy "no" \
        --eval_steps 0.1 \
        --per_device_eval_batch_size 8 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap $FSDP_TRANSFORMER_LAYER \
        --checkpoint_path $VALUE_TRAIN_START

    VALUE_CHECKPOINT=$(find $VALUE_MODEL_PATH -type d -name 'checkpoint*' | sort | head -n 1)
    rm -rf ${VALUE_CHECKPOINT}/rng*
    rm -rf ${VALUE_CHECKPOINT}/train*
    
    python3 tictactoe/prompt_llm.py --method improve --max_tokens=4096 --model_path $BIG_LLM_NAME --batch_size $BATCH_SIZE --input_path $REPLY_BUFFER_PATH --output_path $IMPROVE_TARGET_PATH --value_model_path $VALUE_CHECKPOINT --policy_model_path $ROLLOUT_POLICY_MODEL_PATH --num_policy_sample $NUM_POLICY_SAMPLE --max_use_action $TOP_K_SAMPLE --temp $QWEN3_TEMP --top_k $QWEN3_TOP_K --top_p $QWEN3_TOP_P
    
    python3 tictactoe/data_for_train.py --data_path $IMPROVE_TARGET_PATH --output_path $IMPROVE_TRAIN_PATH --method improve
    torchrun --nproc_per_node=8 --master_port=20002 tictactoe/train/train_sft.py \
        --model_name_or_path=$SMALL_LLM_NAME \
        --train_dataset_path=$IMPROVE_TRAIN_PATH \
        --max_seq_length=1024 \
        --eval_dataset_path=None \
        --report_to="tensorboard" \
        --learning_rate=1e-5 \
        --torch_dtype="bfloat16" \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "constant" \
        --per_device_train_batch_size=8 \
        --max_seq_length 1024 \
        --gradient_accumulation_steps=2 \
        --attn_implementation=sdpa \
        --output_dir=$POLICY_MODEL_PATH \
        --logging_steps=1 \
        --num_train_epoch=$NUM_TRAIN_EPOCH \
        --save_strategy "steps" \
        --save_steps 0.999 \
        --max_steps=-1 \
        --gradient_checkpointing \
        --bf16 \
        --evaluation_strategy "no" \
        --eval_steps 0.1 \
        --per_device_eval_batch_size 8 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap $FSDP_TRANSFORMER_LAYER \
        --checkpoint_path $POLICY_TRAIN_START

    POLICY_CHECKPOINT=$(find $POLICY_MODEL_PATH -type d -name 'checkpoint*' | sort | head -n 1)
    rm -rf ${POLICY_CHECKPOINT}/rng*
    rm -rf ${POLICY_CHECKPOINT}/train*

    if [ $i -gt 1 ]; then
        rm -rf ${BASE_POLICY_MODEL_PATH}_$(($i-1))/checkpoint*
        rm -rf ${BASE_VALUE_MODEL_PATH}_$(($i-1))/checkpoint*
    fi
    python3 tictactoe/collect_rollout_data.py --policy_name $POLICY_NAME --opponent_policy_name $OPPONENT_POLICY_NAME --replay_buffer_path $EVAL_TRAJ_PATH --rollout_method scratch --num_rollouts $NUM_ROLLOUTS --model_path $POLICY_CHECKPOINT --epsilon_greedy 0 --temp 0
    python nlrl/evaluate.py --data_dir ${exp_dir}/data/eval
done


ROLLOUT_POLICY_MODEL_PATH=$POLICY_CHECKPOINT
POLICY_NAME="LLM"
calculated_value=$((FINAL_ITERATION_NUM + 1))
REPLY_BUFFER_PATH="${BASE_REPLAY_BUFFER_PATH}_${calculated_value}.jsonl"
python3 tictactoe/collect_rollout_data.py --policy_name $POLICY_NAME --opponent_policy_name $OPPONENT_POLICY_NAME --replay_buffer_path $REPLY_BUFFER_PATH --rollout_method scratch --num_rollouts $NUM_ROLLOUTS --model_path $ROLLOUT_POLICY_MODEL_PATH

rm $POLICY_CHECKPOINT/optimizer.bin
rm $POLICY_CHECKPOINT/pytorch_model_fsdp.bin
rm $VALUE_CHECKPOINT/optimizer.bin
rm $VALUE_CHECKPOINT/pytorch_model_fsdp.bin


python nlrl/evaluate.py --data_dir ${exp_dir}/data/eval