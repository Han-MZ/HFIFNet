# Custom criterion builders

import torch
import torch.nn as nn

from core.misc import CRITNS

@CRITNS.register_func('WNLL_critn')
def build_weighted_nll_critn(C):
    return nn.NLLLoss(weight=torch.Tensor(C['weights']))


@CRITNS.register_func('BCE_critn')
def build_weighted_bce_critn(C):
    return nn.BCELoss()


@CRITNS.register_func('WBCE_critn')
def build_weighted_wbce_critn(C):
    assert len(C['weights']) == 2
    pos_weight = C['weights'][1]/C['weights'][0]
    return nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))


@CRITNS.register_func('Dice_critn')
def build_dice_critn(C):
    from utils.losses import DiceLoss
    return DiceLoss()


@CRITNS.register_func('BC_critn')
def build_bc_critn(C):
    from utils.losses import BCLoss
    return BCLoss(margin=2*C['threshold'])