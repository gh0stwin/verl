import inspect

import numpy as np
import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@register("dler_mc")
class DLERMonteCarloRewardManager(RewardManagerBase):
    """DLER Reward Manager."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        overlong_buffer_cfg = config.reward_model.get("reward_kwargs", {}).get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = config.reward_model.get("reward_kwargs", {}).get("max_resp_len", None)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        extra_reward_kwargs = (
            {"reward_router_address": self.reward_router_address, "reward_model_tokenizer": self.reward_model_tokenizer}
            if self.reward_router_address is not None
            else {}
        )

        scores = []

        print("INSIDE dler_mc")
        print(data_item)
        print(data_item.non_tensor_batch)
        mc_samples = data_item.non_tensor_batch["mc_samples"].item()
        print(mc_samples)

        for i in range(len(mc_samples)):
            response_str = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(mc_samples[i], skip_special_tokens=True)
            )

            if self.is_async_reward_score:
                result = await self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                )
            else:
                result = await self.loop.run_in_executor(
                    None,
                    lambda: self.compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                        **extra_reward_kwargs,
                    ),
                )

            if isinstance(result, dict):
                score = result["score"]
            else:
                score = result

            scores.append(score)

        scores = np.array(scores)
        assert scores.ndim == 1
        scores = np.reshape(scores, (-1, self.config.actor_rollout_ref.rollout.mc_samples))
        scores = np.mean(scores, axis=1)
        scores = np.repeat(scores, self.config.actor_rollout_ref.mc_token_interval)[:valid_response_length]

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {"reward_router_address": self.reward_router_address, "reward_model_tokenizer": self.reward_model_tokenizer}
            if self.reward_router_address is not None
            else {}
        )

        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        if isinstance(result, dict):
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            reward_extra_info["acc"] = result

        reward = scores
        assert reward.shape[0] <= response_length
        reward = np.concat((reward, np.zeros((response_length - valid_response_length,))))
        reward = torch.tensor(reward, dtype=torch.float32, device=response_ids.device)

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            stop_penalty_coef = self.overlong_buffer_cfg.penalty_factor
            stop_properly = data.non_tensor_batch["stop_reason"] == "completed"
            stop_properly_scale = (1.0 - stop_properly) * stop_penalty_coef + stop_properly
            reward = reward * stop_properly_scale

            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong"] = not stop_properly

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
