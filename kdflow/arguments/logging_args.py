from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoggingArguments:
    """ Arguments for logging (e.g., wandb and tensorboard)."""
    
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log results every n steps."}
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Use wandb for logging."}
    )
    wandb_org: str = field(
        default=None
    )
    wandb_project: str = field(
        default=None
    )
    wandb_group: str = field(
        default=None
    )
    wandb_run_name: str = field(
        default=None
    )
    wandb_mode: str = field(
        default="online",
        metadata={"help": "wandb mode: online, offline, or disabled."}
    )
    wandb_dir: str = field(
        default=None,
        metadata={"help": "Directory to store wandb offline logs."}
    )
    