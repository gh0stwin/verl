import torch
from tensordict import TensorDict

from verl.utils.torch_functional import masked_mean
from verl.workers.config.critic import CriticConfig


def compute_two_hot_encoding_loss(logits, targets, response_mask, min_return, max_return):
    bins = torch.linspace(min_return, max_return, steps=logits.shape[-1]).to(logits.device)
    below = torch.sum((bins <= targets[..., None]).to(torch.int32), dim=-1) - 1
    above = len(bins) - torch.sum((bins > targets[..., None]).to(torch.int32), dim=-1)

    # this is implemented using clip as the gradients are not backpropagated for the out of limits.
    below = torch.clip(below, 0, len(bins) - 1)
    above = torch.clip(above, 0, len(bins) - 1)
    equal = below == above
    dist_to_below = torch.where(equal, 1, torch.abs(bins[below] - targets))
    dist_to_above = torch.where(equal, 1, torch.abs(bins[above] - targets))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target = torch.nn.functional.one_hot(below, num_classes=len(bins)) * weight_below[..., None]
    target += torch.nn.functional.one_hot(above, num_classes=len(bins)) * weight_above[..., None]
    log_pred = logits - torch.logsumexp(logits, -1, keepdim=True)
    # target = target.squeeze(-2)
    target = target.unsqueeze(-1)
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
    print("CHECK SHAPES:")
    print(model_output["value_logits"].shape)
    print(data["returns"].shape)
    # value_logits = _slice_response_from_unpad_output(model_output["value_logits"], data)  # (bsz, response_length)
    value_logits = model_output["value_logits"]
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
