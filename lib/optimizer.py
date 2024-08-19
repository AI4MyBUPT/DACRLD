import torch
from torch.optim import AdamW

def separate_weight_decayable_params(params):
    # Exclude affine params in norms (e.g. LayerNorm, GroupNorm, etc.) and bias terms
    no_wd_params = [param for param in params if param.ndim < 2]
    wd_params = [param for param in params if param not in set(no_wd_params)]
    return wd_params, no_wd_params

def get_adamw_optimizer(params, lr = 3e-4, betas = (0.9, 0.99), weight_decay=0.01):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr = lr, weight_decay = weight_decay, betas=betas)

def build_lr_scheduler(cfg, optimizer):
    cfg = cfg.lr_scheduler
    if cfg.type == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.step_size, cfg.gamma)
    elif cfg.type == "cosine":
        print(f"Using CosineAnnealingWarmRestarts lr_scheduler, T_0={cfg.step_size}, T_mult={cfg.gamma}")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=cfg.step_size,
                                                                            T_mult=int(cfg.gamma)
                                                                            )
    else:
        raise NotImplementedError

    return lr_scheduler
