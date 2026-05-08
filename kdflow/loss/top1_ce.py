import torch

from kdflow.loss import register_loss


@register_loss("top1_ce")
@torch.compile()
def compute_top1_ce(
    student_logits,
    teacher_logits, 
    temperature=1.0,
    reduction="none",
    **kwargs
):
    """Sometimes top1_ce performs better than fkl."""
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    teacher_top1_probs, indices = torch.softmax(
        teacher_logits, -1, dtype=torch.float32
    ).max(-1)
    student_lprobs = torch.log_softmax(
        student_logits, -1, dtype=torch.float32
    ).gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    ce_loss = -teacher_top1_probs * student_lprobs
    
    if reduction == "mean":
        return ce_loss.mean()
    elif reduction == "sum":
        return ce_loss.sum()

    return ce_loss