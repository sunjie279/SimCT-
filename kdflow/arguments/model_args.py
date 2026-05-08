from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """ Arguments for model."""
    
    student_name_or_path: str = field(
        default=None,
        metadata={"help": "Student model name or path."}
    )
    teacher_name_or_path: str = field(
        default=None,
        metadata={"help": "Teacher model name or path."}
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)."}
    )
    use_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Use Liger Kernel in LLM."}
    )
    lora_rank: int = field(
        default=0
    )
    lora_alpha: int = field(
        default=16
    )
    target_modules: str = field(
        default="all-linear",
    )
    lora_dropout: float = field(
        default=0.0,
    )
    disable_fast_tokenizer: bool = field(
        default=False
    )
    enable_thinking: bool = field(
        default=False,
    )
    ring_attn_size: int = field(
        default=1
    )