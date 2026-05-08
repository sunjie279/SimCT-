import torch
import torch.nn.functional as F 

from kdflow.loss import register_loss


@register_loss("kl")
@torch.compile()
def compute_kl_div(
    student_logits,
    teacher_logits, 
    temperature=1.0,
    reduction="none",
    **kwargs
):
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    log_probs = torch.log_softmax(student_logits, -1, dtype=torch.float32)
    target_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    kl_div = F.kl_div(log_probs, target_probs, reduction=reduction).sum(-1)
    
    return kl_div