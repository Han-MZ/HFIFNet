# Custom optimizer builders
import torch
from core.misc import OPTIMS

@OPTIMS.register_func('Adamw_optim')
def build_Adam_optim(params, C):
    return torch.optim.AdamW(
        params,
        betas=(0.9, 0.999),
        lr=C['lr'],
        weight_decay=C['weight_decay']
    )
