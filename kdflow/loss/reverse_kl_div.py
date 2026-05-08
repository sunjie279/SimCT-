import torch

from kdflow.loss import register_loss


@register_loss("rkl")
@torch.compile()
def compute_reverse_kl_div(
    student_logits,
    teacher_logits, 
    temperature=1.0,
    reduction="none",
    **kwargs
):
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = torch.log_softmax(student_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    student_probs = student_log_probs.exp()
    rkl_div = (student_probs * (student_log_probs - teacher_log_probs)).sum(-1)
    
    if reduction == "mean":
        rkl_div = rkl_div.mean()
    elif reduction == "sum":
        rkl_div = rkl_div.sum()
    
    return rkl_div