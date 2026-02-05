import torch
from tensordict import TensorDict

from verl.utils.torch_functional import masked_mean
from verl.workers.config.critic import CriticConfig
from verl.workers.utils.losses import _slice_response_from_unpad_output

from .utils.functional import symexp, symlog


def compute_two_hot_encoding_loss(logits, targets, response_mask, min_return, max_return):
    min_bucket = symlog(torch.tensor(min_return))
    max_bucket = symlog(torch.tensor(max_return))
    buckets = torch.linspace(min_bucket, max_bucket, logits.shape[-1], device=logits.device)

    below = torch.sum((buckets <= targets[..., None]).to(torch.int32), dim=-1) - 1  # number of buckets below
    above = len(buckets) - torch.sum((buckets > targets[..., None]).to(torch.int32), dim=-1)  # number of buckets above

    # this is implemented using clip as the gradients are not backpropagated for the out of limits.
    below = torch.clip(below, 0, len(buckets) - 1)
    above = torch.clip(above, 0, len(buckets) - 1)
    equal = below == above
    dist_to_below = torch.where(equal, 1, torch.abs(buckets[below] - targets))
    dist_to_above = torch.where(equal, 1, torch.abs(buckets[above] - targets))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target = torch.nn.functional.one_hot(below, num_classes=len(buckets)) * weight_below[..., None]
    target += torch.nn.functional.one_hot(above, num_classes=len(buckets)) * weight_above[..., None]
    log_pred = logits - torch.logsumexp(logits, -1, keepdim=True)
    target = target.squeeze(-2)
    # target = target.unsqueeze(-1)
    print("two-hot enconding")
    print(logits.shape, target.shape)
    return (target * log_pred).sum(-1) * response_mask


def two_hot_encoding_value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    print("THIS IS NOT BEING USED", flush=True)
    print("CHECK SHAPES:")
    print(model_output["value_logits"].shape)
    print(data["returns"].shape)
    value_logits = _slice_response_from_unpad_output(model_output["value_logits"], data)  # (bsz, response_length)
    # value_logits = model_output["value_logits"]
    print("after slice call")
    print(model_output["value_logits"].shape)
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    loss, values = compute_two_hot_encoding_loss(
        logits=value_logits,
        targets=returns,
        response_mask=response_mask,
        min_return=config.min_return,
        max_return=config.max_return,
    )

    metrics = {
        "critic/vf_loss": loss.detach().item(),
        "critic/vpred_mean": masked_mean(values, response_mask).detach().item(),
    }

    return loss, metrics
