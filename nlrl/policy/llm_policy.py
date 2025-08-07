import tokenize
from typing import Optional

from regex import F
import vllm
from nlrl.envs.tictactoe.prompt import POLICY_prompt
from nlrl.llm_call import vllm_model
from nlrl.config import LLMSamplingParams
import numpy as np
from nlrl.offline_infer import offline_ray_vllm_infer
from openai import OpenAI
from transformers import AutoTokenizer
import torch

class Agent:
    def __init__(
        self,
        model_path,
        sample_config,
        model_tp_size=None,
        remote=False,
        epsilon_greedy: Optional[float] = None,
    ):
        self.model_path = model_path
        self.remote = remote
        self.tp_size = model_tp_size
        self.sample_config = sample_config
        self.is_gpt4 = "gpt-4" in model_path.lower()
        self.is_ppo = "-ppo" in model_path.lower()
        print(f"model_path: {model_path}, is_gpt4: {self.is_gpt4}, is_ppo: {self.is_ppo}")

        if self.is_gpt4:
            self.client = OpenAI(
                api_key=openai_api_key,
            )  # Uses OPENAI_API_KEY from environment
        elif self.is_ppo:
            sample_config.prompt_logprobs = True
            self.model = vllm_model(
                model_path=self.model_path,
                sample_config=sample_config,
                tensor_parallel_size=model_tp_size,
            )
            if "qwen" in model_path.lower():
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                except Exception as e:
                    print(f"Failed to load tokenizer with trust_remote_code, trying without fast: {e}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif not self.remote:
            self.model = vllm_model(
                model_path=self.model_path,
                sample_config=sample_config,
                tensor_parallel_size=model_tp_size,
            )
            # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.select_action_prompter = POLICY_prompt()
        if epsilon_greedy is not None and epsilon_greedy <= 0:
            epsilon_greedy = None
        self._eps_greedy = epsilon_greedy
        assert self._eps_greedy is None or 1 > self._eps_greedy > 0
        self.is_ppo = False

    def get_prompt(self, state):
        response_type = "LLM"  # This will give us the GPT format
        text_prompt = self.select_action_prompter(state, response_type=response_type)
        return text_prompt["prompt"]

    def extract_answer(self, response, available_actions):
        try:
            if isinstance(response, str):
                result = response.split("""\"best_move\": """)[1][0]
            else:  # OpenAI ChatCompletion response
                response_text = response.choices[0].message.content.strip()
                result = response_text.split("""\"best_move\": """)[1].split("}")[0].strip()

            result = int(result) - 1

            assert (
                result in available_actions
            ), f"Error: {result} not in {available_actions}, response: {response}"
        except Exception as e:
            result = available_actions[
                int(np.random.choice(len(available_actions), 1)[0])
            ]
            print(
                f"Error: random action selected {result} from {available_actions}, response: {response}\
                    \nexception {e}\n"
            )
        return result

    def extract_move_logprobs(self, outputs, valid_moves, verbose=False):
        """
        Extract move logprobs robustly handling additional tokens and -inf values

        Args:
            outputs: vLLM output containing logprobs
            valid_moves: List of valid moves (0-8)
            verbose: Whether to print debug info

        Returns:
            torch.Tensor: Logprobs for all moves (shape: [9])
        """
        move_logprobs = torch.full((9,), float("-inf"))
        found_valid_logprob = False

        for output_idx, output in enumerate(outputs):
            if verbose:
                print(f"\nProcessing output {output_idx}:")

            all_logprobs = output.prompt_logprobs
            if verbose:
                print("All logprobs sequences:")
                for i, logprob_dict in enumerate(all_logprobs):
                    print(f"Position {i}:")
                    for token_id, logprob_info in logprob_dict.items():
                        print(f"  {token_id}: {logprob_info}")

            move_logprob_sequences = []
            for i, logprob_dict in enumerate(all_logprobs):
                move_found = False
                header_follows = False
                current_logprob = float("-inf")
                current_move = None

                if type(logprob_dict) is not dict:
                    if verbose:
                        print(f"Skipping non-dict at position {i}: {logprob_dict}")
                    continue

                for token_id, logprob_info in logprob_dict.items():
                    token = logprob_info.decoded_token
                    if token in [str(i) for i in range(1, 10)]:
                        move_found = True
                        current_move = int(token) - 1  # Convert to 0-8 format
                        current_logprob = logprob_info.logprob
                    if token in [
                        "<|start_header_id|>",
                        "assistant",
                        "<|end_header_id|>",
                        "\n\n",
                    ]:
                        header_follows = True

                if move_found and header_follows and current_move in valid_moves:
                    move_logprob_sequences.append((i, current_logprob, current_move))
                    if current_logprob > float("-inf"):
                        found_valid_logprob = True

            if verbose and move_logprob_sequences:
                print(f"Found move sequences at positions: {move_logprob_sequences}")

            if move_logprob_sequences:
                # Sort by logprob, highest first
                move_logprob_sequences.sort(key=lambda x: x[1], reverse=True)
                for _, logprob, move in move_logprob_sequences:
                    move_logprobs[move] = logprob

        # Handle case where all logprobs are -inf
        if not found_valid_logprob:
            if verbose:
                print(
                    "Warning: All logprobs are -inf, using uniform distribution over valid moves"
                )
            # Use uniform distribution over valid moves
            for move in valid_moves:
                move_logprobs[move] = 0.0

        return move_logprobs

    def get_batch_action(self, state_list):
        random_eps = np.random.random(len(state_list))
        prompt_list = [self.get_prompt(state) for state in state_list]
        assert len(prompt_list) == len(state_list)
        chosen_prompt_input_list = []
        random_act_indices = []
        non_random_states = []  # Store (index, prompt) pairs for non-epsilon-greedy states

        if self._eps_greedy is not None:
            for i, ep in enumerate(random_eps):
                if ep < self._eps_greedy:
                    random_act_indices.append(i)
                else:
                    non_random_states.append((i, prompt_list[i]))  # Keep original index
                    chosen_prompt_input_list.append(prompt_list[i])
            prompt_list = chosen_prompt_input_list
        else:
            non_random_states = list(enumerate(prompt_list))  # All states with their indices

        if self.is_gpt4:
            responses = []
            for prompt in prompt_list:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=prompt,  # POLICY_prompt already formats messages correctly
                        temperature=self.sample_config.temperature,
                        max_tokens=self.sample_config.max_tokens,
                        top_p=self.sample_config.top_p,
                    )
                    responses.append(response)
                except Exception as e:
                    print(f"Error in API call: {e}")
                    responses.append(None)
        elif self.is_ppo:
            all_prompts = []
            state_move_map = {}  # Track which prompts correspond to which state
            response_map = {}  # Map to store responses for each state index

            for orig_idx, prompt in non_random_states:  # Now we use the pairs directly
                valid_moves = [j for j in range(9) if state_list[orig_idx][0][j] == 0]
                state_prompts = []
                for move in valid_moves:
                    move_prompt = prompt + [
                        {'role': 'assistant', 'content': f"{move + 1}"}
                    ]
                    state_prompts.append(move_prompt)

                start_idx = len(all_prompts)
                all_prompts.extend(state_prompts)
                state_move_map[orig_idx] = {
                    'start': start_idx,
                    'count': len(state_prompts),
                    'valid_moves': valid_moves
                }

            # Get vLLM outputs for all prompts
            responses = []
            if all_prompts:

                _, outputs = self.model.generate(all_prompts)

                # print(f"outputs: {outputs}")
                # current_resp_idx = 0
                for original_idx in range(len(state_list)):
                    if original_idx in random_act_indices:
                        # Skip - we'll handle epsilon-greedy in final loop
                        valid_moves = [j for j in range(9) if state_list[original_idx][0][j] == 0]
                        random_move = np.random.choice(valid_moves)
                        response_map[original_idx] = str(random_move)
                    elif original_idx in state_move_map:  # Check if it's a non-epsilon-greedy state
                        start_idx = state_move_map[original_idx]['start']
                        count = state_move_map[original_idx]['count']
                        valid_moves = state_move_map[original_idx]['valid_moves']

                        # Extract state's outputs
                        state_outputs = outputs[start_idx:start_idx + count]

                        try:
                            move_logprobs = self.extract_move_logprobs(
                                state_outputs,
                                valid_moves,
                                verbose=False,
                            )

                            probs = torch.softmax(move_logprobs, dim=0)
                            selected_move = torch.argmax(probs).item()
                            response_map[original_idx] = str(selected_move)

                        except Exception as e:
                            print(f"Error processing state {original_idx}: {e}")
                            fallback_move = np.random.choice(valid_moves)
                            response_map[original_idx] = str(fallback_move)

                responses = [response_map[i] for i in range(len(state_list))]

        elif not self.remote:
            outputs, _ = self.model.generate(prompt_list)
            responses = [output[-1]["content"] for output in outputs]

        else:
            outputs = offline_ray_vllm_infer(
                self.model_path, self.tp_size, prompt_list, self.sample_config,
            )
            responses = [output[-1]["content"] for output in outputs]

        # responses = []
        # for output in outputs:
        #     responses.append((output[-1]["content"]))

        actions = []
        available_actions_list = [
            [i for i in range(9) if state[0][i] == 0] for state in state_list
        ]
        if self.is_ppo:
            for i, available_actions in enumerate(available_actions_list):
                result = int(responses[i])
                assert result in available_actions, (
                    f"Error: {result} not in {available_actions}, "
                    f"response: {responses[i]} for state {i}"
                )
                actions.append(result)
        else:
            llm_i = 0
            for i, available_actions in enumerate(available_actions_list):
                if self._eps_greedy is not None and random_eps[i] < self._eps_greedy:
                    actions.append(np.random.choice(available_actions).item())
                    print("Random pick {} from {}".format(actions[-1], available_actions))
                else:
                    actions.append(self.extract_answer(responses[llm_i], available_actions))
                    llm_i += 1

            assert llm_i == len(responses), (
                f"Mismatch in processed responses: "
                f"processed {llm_i} but have {len(responses)} responses"
            )
        # print(f"actions: {actions}")
        return actions

    def __call__(self, state):
        if isinstance(state, list):
            return self.get_batch_action(state)
        return self.get_action(state)
