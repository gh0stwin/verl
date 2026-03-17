import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("monte_carlo_agent")
class SingleTurnMonteCarloAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.mc_token_interval = self.config.actor_rollout_ref.rollout.mc_token_interval
        self.mc_samples = self.config.actor_rollout_ref.rollout.mc_samples

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        print("messages")
        print(len(messages), messages[0].__class__)  # Check that this is a list with one element?
        print("out")
        prompt_ids = await self.apply_chat_template(messages)
        metrics = {}

        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params
            )

        i = 0
        all_gens = []
        response_ids = output.token_ids[: self.response_length]
        max_len = len(response_ids) - self.mc_token_interval // 2

        while i < max_len:
            _prompt_ids = prompt_ids + response_ids[:i]

            for _ in range(self.mc_samples):
                all_gens.append(
                    asyncio.create_task(
                        self.server_manager.generate(
                            request_id=uuid4().hex, prompt_ids=_prompt_ids, sampling_params=sampling_params
                        )
                    )
                )

            i += self.mc_token_interval

        mc_outputs = await asyncio.gather(*all_gens)
        mc_outputs = [out.token_ids[: self.response_length] for out in mc_outputs]
        assert len(mc_outputs) == (i // self.mc_token_interval - 1) * self.mc_samples
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            num_turns=2,
            metrics=metrics,
            extra_fields={"mc_responses": mc_outputs},
        )

        return output
