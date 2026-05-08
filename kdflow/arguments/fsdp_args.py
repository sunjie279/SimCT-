from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FSDPArguments:
    """ Arguments for fsdp backend."""
    
    fsdp_size: int = field(
        default=-1,
        metadata={"help": "FSDP shard size (for HSDP)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Offload Adam optimizer to GPU."}
    )
    
    