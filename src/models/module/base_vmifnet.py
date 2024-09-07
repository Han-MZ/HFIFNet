import math
import os
import torchvision
from torch import Tensor, reshape, stack, cat, tensor, mean, max
from typing import List
from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    PReLU,
    Sequential,
    ConvTranspose2d,
    Module,
    AdaptiveAvgPool2d,
    Sigmoid,
    Conv1d,
    ModuleList,
    functional as F
)


def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module


build = import_abspy(
    "vmamba",
    "/home/xutao/hanmz/workspace/VMFF-CD/classification/models")

VSSM: Module = build.Backbone_VSSM

def BaseModelVM(bkbn_size):
    if bkbn_size == 'small' :
        pretrained = "/YOUR_PATH/checkpoints/vssmsmall_dp03_ckpt_epoch_238.pth"
        depths = (2, 2, 27)
        dims = 96
    elif bkbn_size == 'base' :
        pretrained = "/YOUR_PATH/checkpoints/vssmbase_dp05_ckpt_epoch_260.pth"
        depths = (2, 2, 27)
        dims = 128
    elif bkbn_size == 'tiny':
        pretrained = "/YOUR_PATH/checkpoints/vssmtiny_dp01_ckpt_epoch_292.pth"
        depths = (2, 2, 9)
        dims = 96
    basemodel = VSSM(out_indices=(0, 1, 2),
        pretrained=pretrained,
        depths=depths,
        dims = dims
    )
    for param in basemodel.parameters():
        param.requires_grad = False

    return basemodel


def getBackbone(bkbn_name = 'efficientnet_v2_l') :
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(
        weights='EfficientNet_V2_L_Weights.DEFAULT'
    ).features
    # Slicing it:
    stage1 = Sequential(
        entire_model[0],
        entire_model[1],
    )
    stage2 = Sequential(
        entire_model[2],
    )
    stage3 = Sequential(
        entire_model[3],
    )
    # used:
    derived_model = ModuleList([
        stage1,
        stage2,
        stage3
    ])

    return derived_model

class DBIM(Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self._conv1 = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            InstanceNorm2d(ch_out),
            PReLU(),
        )
        self._conv2 = Sequential(
            Conv2d(ch_in, ch_out, 3, stride=1, padding=1),
            InstanceNorm2d(ch_out),
            PReLU(),
        )
        self._conv3 = Sequential(
            Conv2d(ch_in, ch_out, kernel_size=1, stride=1,),
            InstanceNorm2d(ch_out),
            PReLU(),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # branch 1:
        stacked = stack((x, y), dim=2)
        stacked = reshape(stacked, (x.shape[0], -1, x.shape[2], x.shape[3]))
        stacked = self._conv1(stacked)

        # branch 2
        cated = cat([x,y], dim=1)
        cated = self._conv2(cated)

        # cat branch 1 branch 2
        dualed = cat([stacked, cated], dim=1)
        dualed = self._conv3(dualed)

        return dualed

class Up2(Module):
    def __init__(
        self,
        nin: int,
        nout: int,
    ):
        super().__init__()

        self._upsample =Sequential(
            ConvTranspose2d(in_channels=nin, out_channels=nin,
                            kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            InstanceNorm2d(nin),
            PReLU(),
        )

        self._convolution = Sequential(
            Conv2d(nin, nin, 3, 1, padding=1),
            InstanceNorm2d(nin),
            PReLU(),
            Conv2d(nin, nout, kernel_size=1, stride=1),
            InstanceNorm2d(nout),
            PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._upsample(x)
        return self._convolution(x)

class Head(Module):
    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)

class MFFM(Module):
    def __init__(self, in_c1, in_c2, in_c3): # 80, 48, 24
        super().__init__()

        ###代表3*3的卷积融合，目的是消除上采样过程带来的重叠效应，以生成最终的特征图。
        self.smooth2 = Conv2d(in_c2, in_c2, 3, 1, 1)
        self.smooth1 = Conv2d(in_c3, in_c3, 3, 1, 1)
        ###自上而下的上采样模块
        self._up1 = Up2(in_c1, in_c2)
        self._up2 = Up2(in_c2, in_c3)

    def forward(self, x):
        ###自下而上
        c1 = x[0] #(1, 24, 256, 256)
        c2 = x[1] #(1, 48, 128, 128)
        c3 = x[2] #(1, 80, 64, 64)
        ###自上而下
        p3 = c3
        p2 = self._up1(p3) + c2 # (1, 48, 128, 128)
        p2 = self.smooth2(p2)
        p1 = self._up2(p2) + c1 # (1, 24, 256, 256)
        p1 = self.smooth1(p1)
        p11 = self._up2(self._up1(c3)) + c1 # (1, 24, 256, 256)
        p11 = self.smooth1(p11)
        p111 = self._up2(p2)

        p_cat = cat([p1, p11, p111], dim=1)

        return p_cat # (1, 72, 256, 256)

class FPN(Module):
    def __init__(self, in_c1, in_c2, in_c3):
        super(FPN, self).__init__()

        ###定义toplayer层，对C5减少通道数，得到P5
        ###代表3*3的卷积融合，目的是消除上采样过程带来的重叠效应，以生成最终的特征图。
        self.smooth3 = Conv2d(in_c3, in_c3, 3, 1, 1)
        ###横向连接，保证通道数目相同
        self.latlayer1 = Conv2d(in_c1, in_c3, 1, 1, 0) #c4
        self.latlayer2 = Conv2d(in_c2, in_c3, 1, 1, 0) #c3
        self.latlayer3 = Conv2d(in_c3, in_c3, 1, 1, 0) #c2

###自上而下的上采样模块
    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        ###自下而上
        c2 = x[0]
        c3 = x[1]
        c4 = x[2]
        ###自上而下
        p4 = self.latlayer1(c4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        ###卷积融合，平滑处理
        p2 = self.smooth3(p2)
        return p2

class ConnectX2Y(Module):
    def __init__(self, x_ch, x_base_ch, b=1, gamma=2):
        super().__init__()
        self.conv3x3 =Sequential(
            Conv2d(1, 1, 7, padding=3, bias=False),
            Sigmoid()
        )

        self._up = Up2(x_base_ch, x_ch)
        kernel_size = int(abs((math.log(x_base_ch, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = AdaptiveAvgPool2d(1)
        self.conv = Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x, x_base):
        mask = self.avg_pool(x_base)
        mask = self.conv(mask.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        mask = self.sigmoid(mask)
        x_base = x_base * mask.expand_as(x_base)

        max_out, _ = max(x_base, dim=1, keepdim=True)
        amask = self.conv3x3(max_out)
        x_base = x_base * amask
        x_base = self._up(x_base)

        return x + x_base