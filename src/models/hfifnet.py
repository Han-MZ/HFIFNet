from torch import Tensor
from .module.base_vmifnet import (
    getBackbone,
    DBIM,
    MFFM,
    Head,
    Up2,
    BaseModelVM,
    ConnectX2Y
)
from torch.nn import ModuleList, Module, Sigmoid

class VMFF(Module):
    def __init__(self, bkbn_size:str):
        super().__init__()

        # Load the pretrained backbone according to parameters:
        self._base_model = BaseModelVM(bkbn_size=bkbn_size)
        self._backbone = getBackbone()

        # Initialize Upsampling blocks:
        if bkbn_size == 'base':
            in_c = 128
        elif bkbn_size == 'small':
            in_c = 96
        elif bkbn_size == 'tiny':
            in_c = 96
        self._fpn = MFFM(96, 64, 32)
        self._fusion = ModuleList([
            DBIM(64, 32),
            DBIM(128, 64),
            DBIM(192, 96)
        ])
        self._up1 = Up2(96,64)
        self._connectx2y = ModuleList([
            ConnectX2Y(32, in_c),
            ConnectX2Y(64, in_c * 2),
            ConnectX2Y(96, in_c * 4),
        ])
        # Final classification layer:
        self._classify = Head([64, 32, 16], [32, 16, 1], Sigmoid())

    def forward(self, t1, t2) -> Tensor:
        ts_fpn = self._encode(t1, t2)

        ups = self._up1(ts_fpn)
        return self._classify(ups)

    def _encode(self, t1, t2):
        t1_base = self._base_model(t1)
        t2_base = self._base_model(t2)
        mixs = []
        for num, layer in enumerate(self._backbone):
            t1 = layer(t1)
            t2 = layer(t2)
            t1 = self._connectx2y[num](t1, t1_base[num])
            t2 = self._connectx2y[num](t2, t2_base[num])
            mix = self._fusion[num](t1, t2)
            mixs.append(mix)
        mixs_fpn = self._fpn(mixs)

        return mixs_fpn
