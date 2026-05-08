import torch

from kdflow.loss import register_loss


@register_loss("jsd")
@torch.compile()
def compute_js_div(
    student_logits,
    teacher_logits, 
    temperature=1.0,
    jsd_beta=0.5,
    reduction="none",
    **kwargs
):
    """Jensen-Shannon Divergence (JSD) loss in GKD (https://arxiv.org/pdf/2306.13649)."""
    assert jsd_beta > 0 and jsd_beta < 1, "jsd_beta must be in (0, 1)"
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_lprobs = student_logits.log_softmax(-1, dtype=torch.float32)
    teacher_lprobs = teacher_logits.log_softmax(-1, dtype=torch.float32)
    log_M = torch.logaddexp(
        student_lprobs + torch.tensor(jsd_beta, dtype=torch.float32).log(),
        teacher_lprobs + torch.tensor(1 - jsd_beta, dtype=torch.float32).log(),
    )
    div1 = jsd_beta * (student_lprobs.exp() * (student_lprobs - log_M)).sum(-1)
    div2 = (1 - jsd_beta) * (teacher_lprobs.exp() * (teacher_lprobs - log_M)).sum(-1)
    js_div = div1 + div2
    
    if reduction == "mean":
        js_div = js_div.mean()
    elif reduction == "sum":
        js_div = js_div.sum()
    
    return js_div