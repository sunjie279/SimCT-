import torch

from kdflow.loss import register_loss


@register_loss("tvd")
@torch.compile()
def compute_tvd(
    student_logits,
    teacher_logits, 
    temperature=1.0,
    reduction="none",
    **kwargs
):
    """Total Variation Distance (TVD) loss."""
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_probs = torch.softmax(student_logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    tvd = 0.5 * torch.abs(student_probs - teacher_probs).sum(-1)
    
    if reduction == "mean":
        tvd = tvd.mean()
    elif reduction == "sum":
        tvd = tvd.sum()
    
    return tvd