# PyTorch Semantic Segmentation Experiment

这个目录给 `exp2` 补了一个基于 PyTorch 的图像语义分割版本，和原有 MindSpore notebook 对应，默认使用 `torchvision` 的 `DeepLabV3-ResNet50` 在 VOC2012 上完成训练、评估和单图推理。

## 目录结构

```text
exp2/pytorch_segmentation
├── config.py          # 默认配置、VOC 类别和调色板
├── dataset.py         # VOC2012 数据集读取
├── download.py        # 数据下载和解压
├── engine.py          # 训练/评估循环
├── evaluate.py        # 验证入口
├── metrics.py         # Pixel Acc / mIoU
├── model.py           # DeepLabV3 模型构建
├── predict.py         # 单图或目录推理
├── train.py           # 训练入口
├── transforms.py      # 训练与评估的数据预处理
└── utils.py           # 随机种子、设备、checkpoint 等工具
```

## 环境

仓库根目录的 `pyproject.toml` 已经包含了当前实验用到的依赖：

- `torch`
- `torchvision`
- `numpy`
- `opencv-python`
- `matplotlib`

如果你使用 `uv`：

```bash
uv sync
```

如果你直接使用 Python 环境，也至少需要安装 `torch torchvision pillow numpy`。

## 数据准备

默认数据目录是 `exp2/VOC2012`，下载缓存目录是 `exp2/downloads`。

当前仓库里已经存在：

- `exp2/VOC2012`
- `exp2/downloads/VOCtrainval_11-May-2012.tar`

如果数据已经在位，可以跳过下载。若需要重新准备数据：

```bash
python exp2/run_pytorch_download.py
```

该脚本会优先复用本地 `exp2/downloads/VOCtrainval_11-May-2012.tar`，如果不存在，再从官方 VOC 地址下载并解压。

如果你更喜欢直接改代码配置，可以打开 [run_pytorch_download.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_download.py) 修改顶部的 `CONFIG`。

## 训练

从仓库根目录执行：

```bash
python exp2/run_pytorch_train.py
```

训练参数都在 [run_pytorch_train.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_train.py) 顶部的 `CONFIG` 里，改完直接运行即可。

常改的项：

- `epochs`
- `batch_size`
- `lr`
- `weights`：设为 `"voc"` 时直接使用 `torchvision` 提供的 VOC 预训练分割权重
- `backbone_weights`：设为 `"imagenet"` 时只加载 ImageNet 预训练主干
- `resume`
- `freeze_bn`
- `device`

训练输出：

- `last.pth`：最新 checkpoint
- `best.pth`：按验证集 `mIoU` 保存的最佳 checkpoint

## 评估

使用自己训练得到的模型：

```bash
python exp2/run_pytorch_evaluate.py
```

评估参数在 [run_pytorch_evaluate.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_evaluate.py) 的 `CONFIG` 里改。

如果只想快速验证 `torchvision` 自带 VOC 权重，把 `CONFIG["weights"]` 改成 `"voc"`，并把 `checkpoint` 设为 `None`。

如果你仍然想用命令行参数方式，也支持：

```bash
python -m exp2.pytorch_segmentation.evaluate \
  --weights voc
```

评估结果会打印：

- `loss`
- `pixel_accuracy`
- `mean_iou`
- 每个类别的 IoU

## 推理与可视化

对单张图像推理：

```bash
python exp2/run_pytorch_predict.py
```

推理参数在 [run_pytorch_predict.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_predict.py) 的 `CONFIG` 里改，`input` 可以是单张图，也可以是一个目录。输出包括：

- `*_mask.png`：调色板语义分割结果
- `*_overlay.png`：原图和分割图叠加可视化

## 和实验手册的对应关系

- 数据集：使用 VOC2012
- 模型：使用 DeepLabV3
- 训练方式：支持从 ImageNet 主干或 VOC 语义分割预训练权重开始微调
- 数据处理：包含随机缩放、随机裁剪、水平翻转、归一化
- 评估指标：提供像素准确率和 mIoU

## 建议

- CPU 也可以运行，但训练会比较慢，优先使用 CUDA 或 MPS。
- 若显存不足，优先降低 `--batch-size`。
- 如果只是完成实验报告，建议先跑少量 epoch 验证流程，再补完整训练。
- 如果你习惯“改代码再运行”，优先修改 `exp2/run_pytorch_*.py` 这几个入口文件顶部的 `CONFIG`。
