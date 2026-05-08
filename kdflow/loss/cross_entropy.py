import torch


@torch.compile()
def compute_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "none",
    **kwargs
) -> torch.Tensor:
    logsumexp = torch.logsumexp(logits, dim=-1)
    log_probs = logits - logsumexp.unsqueeze(-1)
    ce_loss = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    if reduction == "mean":
        return ce_loss.mean()
    elif reduction == "sum":
        return ce_loss.sum()
    return ce_loss