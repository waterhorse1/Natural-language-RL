"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-nodes cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""

from random import sample
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import ray.data
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from nlrl.llm_call import llama3_instruct_format, qwen3_instruct_format, parse_qwen3_output
from nlrl.config import LLMSamplingParams
from transformers import AutoTokenizer
from functools import partial

from vllm import LLM, SamplingParams

assert Version(ray.__version__) >= Version(
    "2.22.0"
), "Ray version must be at least 2.22.0"


def offline_ray_vllm_infer(
    model: str,
    tensor_parallel_size: Optional[int],
    messages: List[List[Dict[str, str]]],
    sample_config: LLMSamplingParams,
    prompt_format=None,
):
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=sample_config.temperature,
        top_p=sample_config.top_p,
        max_tokens=sample_config.max_tokens,
        n=sample_config.n,
        top_k=sample_config.top_k,
        prompt_logprobs=sample_config.prompt_logprobs,
    )

    assert sampling_params.n == 1
    
    is_qwen = "qwen" in model.lower()
    if prompt_format is None:
        if is_qwen:
            prompt_format = qwen3_instruct_format
        else:
            prompt_format = llama3_instruct_format

    class LLMPredictor:

        def __init__(self):
            # Create an LLM.
            self.llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                disable_custom_all_reduce=True,
                trust_remote_code=True if "qwen" in model.lower() else False,
            )
            if "qwen" in model.lower():
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                except Exception as e:
                    print(f"Failed to load tokenizer with trust_remote_code, trying without fast: {e}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model)

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            outputs = self.llm.generate(batch["text"], sampling_params)
            prompt: List[str] = []
            generated_text: List[str] = []
            indices: List[int] = []
            for i, output in enumerate(outputs):
                indices.append(batch["i"][i])
                prompt.append(output.prompt)
                text = output.outputs[0].text
                if is_qwen:
                    text = parse_qwen3_output(text, self.tokenizer)
                generated_text.append(text)
            return {
                "i": indices,
                "prompt": prompt,
                "generated_text": generated_text,
            }

    # Set number of instances. Each instance will use tensor_parallel_size GPUs.
    num_instances = torch.cuda.device_count() // tensor_parallel_size

    if "qwen" in model.lower():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load tokenizer with trust_remote_code, trying without fast: {e}")
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
        
    messages_with_idx = [
        {"i": i, "text": prompt_format(tokenizer=tokenizer, messages=d)}
        for i, d in enumerate(messages)
    ]
    ds = ray.data.from_items(messages_with_idx)

    # For tensor_parallel_size > 1, we need to create placement groups for vLLM
    # to use. Every actor has to have its own placement group.
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{"GPU": 1, "CPU": 1}] * tensor_parallel_size,
            strategy="STRICT_PACK",
        )
        return dict(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                pg, placement_group_capture_child_tasks=True
            )
        )

    resources_kwarg: Dict[str, Any] = {}
    if tensor_parallel_size == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    # Apply batch inference for all input data.
    ds = ds.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        # batch_size=(len(messages) + 1) // num_instances,
        batch_size=len(messages) // num_instances + 1,
        **resources_kwarg,
    )
    outputs = ds.take_all()
    assert len(outputs) == len(messages)
    # print([o["i"] for o in outputs], flush=True)
    # for i, (output, message) in enumerate(zip(outputs, messages)):
    #     print(i, output["i"])
    #     generated_text = output["generated_text"]
    #     assert output["i"] == i, (output["i"], i)
    #     message.append({"role": "assistant", "content": generated_text})

    # !!!(ziyu): for UNORDERED map of ray.data
    for output in outputs:
        idx = output["i"]
        generated_text = output["generated_text"]
        message = messages[idx]
        message.append({"role": "assistant", "content": generated_text})

    return messages


# import pdb; pdb.set_trace()

if __name__ == "__main__":
    messages = offline_ray_vllm_infer(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        1,
        [[{"role": "system", "content": "Who are you?"}] * 1024],
        LLMSamplingParams(n=1),
    )

    offline_ray_vllm_infer(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        4,
        [[{"role": "system", "content": "Who are you?"}] * 1024],
        LLMSamplingParams(n=1),
    )
