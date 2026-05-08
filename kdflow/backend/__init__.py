from kdflow.backend.fsdp import FSDP2Strategy


def get_strategy(args):
    if args.train.backend == "fsdp2":
        strategy = FSDP2Strategy(
            seed=getattr(args.train, "seed", 42),
            full_determinism=getattr(args.train, "full_determinism", False),
            max_norm=getattr(args.train, "max_norm", 1.0),
            micro_train_batch_size=getattr(args.train, "micro_train_batch_size", 1),
            train_batch_size=getattr(args.train, "train_batch_size", 128),
            bf16=getattr(args.train, "bf16", True),
            args=args,
        )    
    else:
        raise NotImplementedError
    
    return strategy