
from collections import OrderedDict
from collections.abc import Iterable
import cv2

from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
import torch.optim.lr_scheduler
import numpy as np

def mod_crop(blob, N):
    if isinstance(blob, np.ndarray):
        # For numpy arrays, channels at the last dim
        h, w = blob.shape[-3:-1]
        nh = h - h % N
        nw = w - w % N
        return blob[..., :nh, :nw, :]
    else: 
        # For 4-D pytorch tensors, channels at the 2nd dim
        with torch.no_grad():
            h, w = blob.shape[-2:]
            nh = h - h % N
            nw = w - w % N
            return blob[..., :nh, :nw]


class FeatureContainer:
    r"""A simple wrapper for OrderedDict."""
    def __init__(self):
        self._dict = OrderedDict()

    def __setitem__(self, key, val):
        if key not in self._dict:
            self._dict[key] = list()
        self._dict[key].append(val)

    def __getitem__(self, key):
        return self._dict[key]

    def __repr__(self):
        return self._dict.__repr__()

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()


class HookHelper:
    def __init__(self, model, fetch_dict, out_dict, hook_type='forward_out'):
        # XXX: A HookHelper object should only be used as a context manager and should not 
        # persist in memory since it may keep references to some very large objects.
        self.model = model
        self.fetch_dict = fetch_dict
        self.out_dict = out_dict
        self._handles = []
        self.hook_type = hook_type

    def __enter__(self):
        def _hook_proto(x, entry):
            # x should be a tensor or a tuple;
            # entry is expected to be a string or a non-nested tuple.
            if isinstance(entry, tuple):
                for key, f in zip(entry, x):
                    self.out_dict[key] = f.data.clone()
            else:
                self.out_dict[entry] = x.data.clone()

        if self.hook_type == 'forward_in':
            # NOTE: Register forward hooks for MODULEs.
            for name, module in self.model.named_modules():
                if name in self.fetch_dict:
                    entry = self.fetch_dict[name]
                    self._handles.append(
                        module.register_forward_hook(
                            lambda m, x, y, entry=entry:
                                # x is a tuple
                                _hook_proto(x[0] if len(x)==1 else x, entry)
                        )
                    )
        elif self.hook_type == 'forward_out':
            # NOTE: Register forward hooks for MODULEs.
            for name, module in self.model.named_modules():
                if name in self.fetch_dict:
                    entry = self.fetch_dict[name]
                    self._handles.append(
                        module.register_forward_hook(
                            lambda m, x, y, entry=entry:
                                # y is a tensor or a tuple
                                _hook_proto(y, entry)
                        )
                    )
        elif self.hook_type == 'backward':
            # NOTE: Register backward hooks for TENSORs.
            for name, param in self.model.named_parameters():
                if name in self.fetch_dict:
                    entry = self.fetch_dict[name]
                    self._handles.append(
                        param.register_hook(
                            lambda grad, entry=entry:
                                _hook_proto(grad, entry)
                        )
                    )
        else:
            raise RuntimeError("Hook type is not implemented.")

    def __exit__(self, exc_type, exc_val, ext_tb):
        for handle in self._handles:
            handle.remove()


def build_schedulers(cfg_list, optimizers):
    if not isinstance(optimizers, Iterable):
        optimizers = [optimizers]
    schedulers = []
    if len(cfg_list) != len(optimizers):
        raise ValueError("The number of schedulers does not match the number of optimizers.")
    for cfg, optim in zip(cfg_list, optimizers):
        name = cfg.pop('name')
        try:
            sched_cls = getattr(torch.optim.lr_scheduler, name)
            sched_obj = sched_cls(optim, **cfg)
        except AttributeError:
            raise RuntimeError("{} is not a supported scheduler type.".format(name))
        schedulers.append(sched_obj)
    return schedulers

def feature_imshow(inp):
    inp = inp.cpu()
    inp = inp.squeeze(0)
    plt.figure(figsize=(12, 7))

    rgb_image = inp.mean(dim=0, keepdim=True)
    to_pil = transforms.ToPILImage()(rgb_image)

    # 显示图像
    plt.imshow(to_pil)

    # for i in range(len(inp)):
    #     plt.subplot(8, 8, i + 1)  # 第一二个参数为图像个数，第三参数为图像位置
    #     plt.imshow(transforms.ToPILImage()(inp[i]))
    #     i += 1
    plt.show()
    plt.pause(0.001)

def heat_map(f):
    heatmap = torch.sum(f, dim=1)  # 所有通道求和
    max_value = torch.max(heatmap)
    min_value = torch.min(heatmap)
    heatmap = (heatmap - min_value) / (max_value - min_value) * 255

    heatmap = heatmap.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)  # 提取热力图
    heatmap = cv2.resize(heatmap, (f.shape[2],f.shape[3]), interpolation=cv2.INTER_LINEAR)  # 还原尺寸

    # 将矩阵转换为image类
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
    heatimg = Image.fromarray(heatmap)
    plt.imshow(heatimg)
    plt.axis('off')  # 去坐标轴
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./maps/'+str(f.shape[1])+str(f.shape[2])+'.png',bbox_inches='tight',pad_inches=0.0)
    plt.show()
