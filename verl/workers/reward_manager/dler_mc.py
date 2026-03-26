# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import numpy as np
from scipy.interpolate import interp1d
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dler_mc")
class DLERMCRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        mc_samples: int = 4,
        mc_token_interval: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.mc_samples = mc_samples
        self.mc_token_interval = mc_token_interval

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def _apply_stop_properly_penalty(self, reward_tensor: torch.Tensor, batch: DataProto):
        # apply stop properly penalty
        # get coefficient from config
        stop_penalty_coef = self.overlong_buffer_cfg.penalty_factor
        # stop_properly is a tensor of shape (bs) float type
        stop_properly = batch.non_tensor_batch["stop_reason"] == "completed"
        stop_properly = torch.tensor(stop_properly).type(reward_tensor.dtype).device(reward_tensor.device)

        # if stop_properly is True, the value is 1.0, otherwise, it is self.config.stop_properly_penalty.penalty_coef
        stop_properly_scale = (1.0 - stop_properly) * stop_penalty_coef + stop_properly
        stop_properly_scale = stop_properly_scale.unsqueeze(-1)
        reward_tensor = reward_tensor * stop_properly_scale  # apply the scale to the reward_tensor
        return reward_tensor

    def _compute_mc_returns(self, final_score, mc_samples, data_source, ground_truth, extra_info, resp_len):
        eos_token = self.tokenizer.eos_token
        scores = []

        for i in range(len(mc_samples)):
            response_str = self.tokenizer.decode(mc_samples[i], skip_special_tokens=True)

            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            result = self.compute_score(
                data_source=data_source, solution_str=response_str, ground_truth=ground_truth, extra_info=extra_info
            )

            if isinstance(result, dict):
                score = result["score"]
            else:
                score = result

            scores.append(score)

        scores = np.array(scores)
        assert scores.ndim == 1
        scores = np.reshape(scores, (-1, self.mc_samples))
        scores = np.mean(scores, axis=1)
        scores = np.concat((scores, [final_score]))

        adv = np.copy(scores)
        adv[:-1] = scores[1:] - scores[:-1]
        adv_idx = np.arange(scores.shape[0] - 1) * self.mc_token_interval
        adv_idx = np.concat((adv_idx, [resp_len - 1]))
        adv_interp_fn = interp1d(adv_idx, adv, kind="previous")
        response_idx = np.arange(resp_len)
        per_token_adv = adv_interp_fn(response_idx)

        # TODO: ADV - other approaches:
        #   - linear interpolation between computed adv points
        #   - `scores` average takes into account all future rewards (from that timestep onwards)

        scores = np.repeat(scores, self.mc_token_interval)
        scores = np.pad(scores, (0, resp_len - scores.shape[0]), mode="constant", constant_values=scores[-1])
        assert scores.shape[0] == per_token_adv.shape[0] == resp_len
        return scores, per_token_adv

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)

        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        print("DLER (MC) - computing rewards")
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        adv_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token

            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            result = self.compute_score(
                data_source=data_source, solution_str=response_str, ground_truth=ground_truth, extra_info=extra_info
            )

            score: float

            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():  # Store the information including original reward
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            # reward = score
            print("WHAT ARE MC SAMPLES?")
            print(data_item.non_tensor_batch["mc_samples"])
            reward, adv = self._compute_mc_returns(
                score,
                data_item.non_tensor_batch["mc_samples"],
                data_source,
                ground_truth,
                extra_info,
                valid_response_length,
            )

            reward_tensor[i, :valid_response_length] = torch.tensor(
                reward, dtype=reward_tensor.dtype, device=reward_tensor.device
            )

            adv_tensor[i, :valid_response_length] = torch.tensor(adv, dtype=adv_tensor.dtype, device=adv_tensor.device)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)

                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        score_tensor = reward_tensor

        if self.overlong_buffer_cfg.enable:
            reward_tensor = self._apply_stop_properly_penalty(reward_tensor, data)
            adv_tensor = self._apply_stop_properly_penalty(adv_tensor, data)

        if return_dict:
            return {
                "score_tensor": score_tensor,
                "reward_tensor": reward_tensor,
                "adv_tensor": adv_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
