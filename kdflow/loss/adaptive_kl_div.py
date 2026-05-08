import torch
import torch.nn.functional as F 

from kdflow.loss import register_loss


@register_loss("akl")
@torch.compile()
def compute_adaptive_kl_div(
    student_logits,
    teacher_logits, 
    temperature=1.0,
    adaptive_kl_alpha=0.5,
    reduction="none",
    **kwargs
):
    """Adaptive KL Divergence loss (https://arxiv.org/abs/2404.02657)"""
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_probs = torch.softmax(student_logits, dim=-1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    sorted_teacher_probs, sorted_idx = teacher_probs.sort(-1)
    sorted_probs = student_probs.gather(-1, sorted_idx)
    gap = (sorted_teacher_probs - sorted_probs).abs()
    cum_teacher_probs = torch.cumsum(sorted_teacher_probs, -1)
    tail_mask = cum_teacher_probs.le(adaptive_kl_alpha).float()
    g_head = (gap * (1 - tail_mask)).sum(-1).detach()
    g_tail = (gap * tail_mask).sum(-1).detach()

    student_lprobs = torch.log_softmax(student_logits, -1, dtype=torch.float32)
    teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    fkl = (teacher_probs * (teacher_lprobs - student_lprobs)).sum(-1)
    rkl = (student_probs * (student_lprobs - teacher_lprobs)).sum(-1)

    akl = (g_head / (g_head + g_tail)) * fkl + (g_tail / (g_head + g_tail)) * rkl
    
    if reduction == "mean":
        akl = akl.mean()
    elif reduction == "sum":
        akl = akl.sum()
    
    return akl