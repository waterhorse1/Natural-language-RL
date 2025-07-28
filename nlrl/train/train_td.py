import torch
import json
import transformers
import dataclasses
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import numpy as np
np.random.seed(42)  # 设置随机种子以确保可重复性
# 从 trl 库导入我们需要的模型
from trl import AutoModelForCausalLMWithValueHead

from nlrl.envs.breakthrough.prompt import GAME_RULE_PROMPT

EVAL_PROMPT = (
    GAME_RULE_PROMPT
    + """
Given the following board state, please evalaute the position and provide which side takes advantage. -1.0 for white, 1.0 for black.

{current_state}

It is {turn}'s turn to move. Please provide the value of this position.
"""
)

TERMINAL_STATE = "Terminal State."


def read_jsonl(file_path):
    all_data = []
    for line in open(file_path, "r"):
        data = json.loads(line)
        all_data.append(data)
    return all_data


def get_train_data(file_path):
    train_data = read_jsonl(file_path)
    train_data = np.random.choice(train_data, size=int(1e4), replace=False)
    transitions = []
    for data in train_data:
        state = data["current_state"]
        for pv in data["pv"]:
            if pv["final_state"]["state"] == "Terminal State.":
                value = pv["reward"][-1][0]  # 1.0 for black, -1.0 for white
            else:
                value = "placeholder"
            transition = {
                "input": (state, pv["turn"][0]),
                "final_state": (pv["final_state"]["state"], pv["final_state"]["turn"]),
                "value": value,
            }
            transitions.append(transition)
    return transitions


def get_eval_data(file_path):
    eval_data = read_jsonl(file_path)
    evaldata_all = []
    for data in eval_data:
        if abs(data["win_rate"]) < 0.2:
            continue
        evaldata_all.append(
            {
                "input": data["state_turn"],
                "value": data["win_rate"],
            }
        )
    return evaldata_all


def format_input(board, turn):
    """
    Format the input for the evaluation prompt.
    Args:
        board (str): The current state of the board.
    Returns:
        str: The formatted input for the evaluation prompt.
    """
    return EVAL_PROMPT.format(current_state=board, turn="white" if turn else "black")


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # 移除 squeeze() 以便 collate_fn 正确处理
        return {key: val.squeeze(0) for key, val in encoding.items()}


@dataclasses.dataclass
class UniversalDataCollator(object):
    """
    A universal data collator that handles both training (with labels)
    and inference (without labels).
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [instance["input_ids"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # 检查批次中是否有 value_labels
        if "value_labels" in instances[0]:
            value_labels = [instance["value_labels"] for instance in instances]
            batch["value_labels"] = torch.stack(value_labels)

        return batch


def get_subsequent_states_value(
    trainer,
    tokenizer,
    train_subsequent_states,
    max_length=512,  # 需要根据你的prompt和board state长度来设定
):
    """
    Performs batch inference to get the value for a list of subsequent states.

    Args:
        value_model: The AutoModelForCausalLMWithValueHead model.
        tokenizer: The tokenizer for the model.
        train_subsequent_states (list[str]): A list of board state strings.
        batch_size (int): The batch size for inference.
        max_length (int): The maximum length for tokenization.

    Returns:
        torch.Tensor: A tensor containing the predicted values for each state.
    """
    prompts = [format_input(state, turn) for state, turn in train_subsequent_states]
    inference_dataset = InferenceDataset(prompts, tokenizer, max_length)
    raw_predictions = trainer.predict(inference_dataset)
    return torch.from_numpy(raw_predictions.predictions)


class RegressionDataset(Dataset):
    def __init__(self, data_dict, tokenizer):
        self.texts = [format_input(*item["input"]) for item in data_dict]
        self.labels = [item["value"] for item in data_dict]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,  # 建议保留截断
            max_length=512,  # 设置一个合理的模型最大长度
        )

        # 返回未填充的 input_ids
        item = {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "value_labels": torch.tensor(label, dtype=torch.float32),
        }
        return item


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 从输入中提取标签
        labels = None
        if "value_labels" in inputs:
            labels = inputs.pop("value_labels")

        # 将输入传递给模型
        # model的输出是元组: (lm_logits, lm_loss, values)
        # 我们关心的是第三个元素: values
        outputs = model(**inputs)
        values = outputs[2]  # Shape: [batch_size, sequence_length]

        # 找到每个序列中最后一个非填充token的位置
        # attention_mask中1代表有效token, 0是填充token
        attention_mask = inputs.get("attention_mask")
        sequence_lengths = attention_mask.sum(dim=1) - 1

        # 提取每个序列最后一个token对应的value
        # `torch.arange(values.size(0))` 创建一个从0到batch_size-1的索引
        last_token_values = values[
            torch.arange(values.size(0), device=values.device), sequence_lengths
        ]

        # 计算均方误差(MSE)损失
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(last_token_values.squeeze(), labels.squeeze())

        return (loss, last_token_values, labels) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        """
        Perform a prediction step for evaluation/prediction.
        This is automatically handled in a distributed manner.
        """
        # 从 inputs 中移除不必要的标签（如果有的话）
        # 在我们的新 InferenceDataset 中没有标签，所以这步是安全的
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, last_token_values, labels = self.compute_loss(
                model, inputs, return_outputs=True
            )
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, last_token_values, labels)


def compute_eval_metrics(eval_pred):
    """
    Computes regression metrics for evaluation.

    Args:
        eval_pred (EvalPrediction): An object containing predictions and label_ids.

    Returns:
        dict: A dictionary of computed metrics.
    """
    # eval_pred.predictions 是模型在 prediction_step 中返回的第二个元素
    # eval_pred.label_ids 是数据集中真实的标签
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # 确保它们是一维的
    predictions = predictions.squeeze()
    labels = labels.squeeze()

    accuracy = (predictions * labels > 0).mean()

    return {
        "Accuracy": accuracy,
    }


def main():
    # MODEL_NAME = "distilgpt2"
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    value_model.pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    value_model = value_model.to(torch.bfloat16)

    train_data = get_train_data(
        "Breakthrough_dataset/train_45k/look_ahead/replay_buffer.jsonl"
    )
    subsequent_states = [
        data["final_state"]
        for data in train_data
        if data["final_state"][0] != TERMINAL_STATE
    ]
    print(f"Eval prompt: {EVAL_PROMPT}")
    print(f"Number of training data: {len(train_data)}")
    print(f"Number of subsequent states: {len(subsequent_states)}")
    for train_iter in range(50):
        training_args = TrainingArguments(
            output_dir=f"./results_trl_iter_{train_iter}",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_dir=f"./logs_trl_iter_{train_iter}",
            logging_steps=5,
            evaluation_strategy="epoch",
            save_strategy="no",
            remove_unused_columns=False,
            save_safetensors=False,
            fsdp="full_shard auto_wrap",
            fsdp_transformer_layer_cls_to_wrap="LlamaDecoderLayer",
            bf16=True,
            bf16_full_eval=True,
        )
        trainer = RegressionTrainer(
            model=value_model,
            args=training_args,
            # We will set the train_dataset later
            train_dataset=None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=UniversalDataCollator(tokenizer=tokenizer),
            compute_metrics=compute_eval_metrics,
        )
        subsequent_state_eval = get_subsequent_states_value(
            trainer,
            tokenizer,
            subsequent_states,
            max_length=512,  # 需要根据你的prompt和board state长度来设定
        )
        index = 0
        for data in train_data:
            if data["final_state"][0] == TERMINAL_STATE:
                assert (
                    data["value"] == 1.0 or data["value"] == -1.0
                ), "Terminal state value must be 1 or -1."
                continue
            data["value"] = subsequent_state_eval[index].item()
            index += 1
        assert index == len(subsequent_states), "Index mismatch after evaluation."

        train_dataset = RegressionDataset(train_data, tokenizer)
        trainer.train_dataset = train_dataset

        eval_data = get_eval_data(
            "Breakthrough_dataset/evaluation_3000/win_rate_collection/gt_result.jsonl"
        )
        trainer.eval_dataset = RegressionDataset(eval_data, tokenizer)

        # 开始训练
        print(
            f"--- Starting Training using TRL's ValueHead Model on Iteration {train_iter} ---"
        )
        trainer.train()
        print(f"--- Training on Iteration {train_iter} Finished ---")


if __name__ == "__main__":
    main()
