
[English](README.md) | 简体中文

## 依赖库

> torch   
  torchvision  
  torchaudio  
  packaging  
  triton  
  timm==0.4.12  
  pytest  
  chardet  
  yacs  
  termcolor  
  submitit  
  tensorboardX  
  fvcore  
  seabornpytorch==1.6.0  
  pyyaml==5.1.2  
  scikit-image==0.15.0  
  scikit-learn==0.21.3  
  scipy==1.3.1  
  tqdm==4.35.0  
  opencv-python

在 Python 3.10，Ubuntu 16.04 环境下测试通过。

## 快速上手

在 `src/constants.py` 文件中将相应常量修改为数据集存放的位置。

### 模型训练

运行如下指令以从头训练一个模型：

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE
```

在 `configs/` 目录中已经包含了一部分现成的配置文件，可供直接使用。*注意，由于时间和精力的限制，配置文件中提供的超参数可能并没有经过充分调优，您可以通过修改超参数进一步提升模型的效果。*

训练脚本开始执行后，首先会输出配置信息到屏幕，然后会有出现一个提示符，指示您输入一些笔记。这些笔记将被记录到日志文件中。如果在一段时间后，您忘记了本次实验的具体内容，这些笔记可能有助于您回想起来。当然，您也可以选择按下回车键直接跳过。

如果需要从一个检查点（checkpoint）开始继续训练，运行如下指令：

```bash
python train.py train --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT
```

以下是对其它一些常用命令行选项的介绍：

- `anew`: 如果您希望指定的检查点只是用于初始化模型参数，指定此选项。请注意，从一个不兼容的模型中获取部分层的参数对待训练的模型进行初始化也是允许的。
- `save_on`: 如果需要在进行模型评估的同时储存模型的输出结果，指定此选项。项目默认采用基于 epoch 的训练器。在每个 epoch 末尾，训练器也将在验证集上评估模型的性能。
- `log_off`: 指定此选项以禁用日志文件。
- `tb_on`: 指定此选项以启用 TensorBoard 日志。
- `debug_on`: 指定此选项以在程序崩溃处自动设置断点，便于进行事后调试。

在训练过程中或训练完成后，您可以在 `exp/DATASET_NAME/weights/` 目录下查看模型权重文件，在 `exp/DATASET_NAME/logs/` 目录下查看日志文件，在 `exp/DATASET_NAME/out/` 目录下查看输出的变化图。

### 模型评估

使用如下指令在测试集上评估已训练好的模型：

```bash
python train.py eval --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT --save_on --subset test
```

本项目也提供在大幅栅格影像上进行滑窗测试的功能。使用如下指令：

```bash
python sw_test.py --exp_config PATH_TO_CONFIG_FILE \
  --resume PATH_TO_CHECKPOINT --ckp_path PATH_TO_CHECKPOINT \
  --t1_dir PATH_TO_T1_DIR --t2_dir PATH_TO_T2_DIR --gt_dir PATH_TO_GT_DIR
```

对于 `src/sw_test.py` 文件，其它一些常用的可选命令行参数包括：
- `--window_size`: 设置滑窗大小。
- `--stride`: 设置滑窗滑动步长。
- `--glob`: 指定在 `t1_dir`、`t2_dir`、和 `gt_dir` 中匹配文件名的 pattern （通配符）。
- `--threshold`: 指定将模型输出的变化概率二值化时使用的阈值。

不过，请注意当前 `src/sw_test.py` 功能有限，并不支持一些较为复杂的自定义预处理和后处理模块。

### 数据集（弱目标）

**S2Looking-Weak**：[百度网盘](https://pan.baidu.com/s/1HNYiivLxLT8I7ogiJ4V1xA?pwd=xizc)

**LEVIR-CD-Weak**：[百度网盘](https://pan.baidu.com/s/1jWaWooz5hnR-I2NY44BJ3g?pwd=f659)

## 开发指南

请参见 [CDLab](https://github.com/Bobholamovic/CDLab) `docs/` 目录中的内容。

## 开源许可证

该项目采用 [Unlicense 开源许可证](/LICENSE)。

## 贡献

欢迎大家为本项目贡献代码和意见。

## 致谢

这个项目是基于 [CDLab](https://github.com/Bobholamovic/CDLab) 和 [VMamba](https://github.com/MzeroMiko/VMamba)进行二次开发
