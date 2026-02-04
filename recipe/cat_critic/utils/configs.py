from dataclasses import dataclass

from verl.workers.config.critic import FSDPCriticConfig


@dataclass
class FSDPCatCriticConfig(FSDPCriticConfig):
    num_labels: int = 255
