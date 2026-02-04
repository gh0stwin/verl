import warnings

from omegaconf import DictConfig

from .core_algos import AdvantageEstimator


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic."""
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator in (AdvantageEstimator.GAE, AdvantageEstimator.GAE_CAT):
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False
