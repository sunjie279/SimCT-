import torch
import torch.nn.functional as F 

from kdflow.loss import register_loss


@register_loss("srkl")
@torch.compile()
def compute_skewed_rkl_div(
    student_logits,
    teacher_logits, 
    temperature=1.0,
    skew_lambda=0.1,
    reduction="none",
    **kwargs
):
    """Skewed Reverse KL Divergence loss (https://arxiv.org/abs/2402.03898)."""
    assert skew_lambda > 0 and skew_lambda < 1, "skew_lambda must be in (0, 1)"
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_lprobs = torch.log_softmax(student_logits, -1, dtype=torch.float32)
    teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    mixed_lprobs = torch.logaddexp(
        student_lprobs + torch.tensor(skew_lambda, dtype=torch.float32).log(), 
        teacher_lprobs + torch.tensor(1 - skew_lambda, dtype=torch.float32).log(),
    )
    kl_div = (torch.exp(student_lprobs) * (student_lprobs - mixed_lprobs)).sum(-1)
    
    if reduction == "mean":
        kl_div = kl_div.mean()
    elif reduction == "sum":
        kl_div = kl_div.sum()
    
    return kl_div