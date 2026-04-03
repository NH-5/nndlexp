# PyTorch Semantic Segmentation Experiment

这个目录给 `exp2` 补了一个基于 PyTorch 的图像语义分割版本，和原有 MindSpore notebook 对应，默认使用 `torchvision` 的 `DeepLabV3-ResNet50` 在 VOC2012 上完成训练、评估和单图推理。

这版现在额外支持详细日志记录：

- 训练日志同时输出到终端和日志文件
- 评估日志同时输出到终端和日志文件
- 推理日志同时输出到终端和日志文件
- 训练历史、评估结果、预测清单会额外保存为 `json`

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

## 推荐使用方式

如果你现在的习惯是“改脚本里的配置，然后直接 `python 文件.py`”，建议只使用这 4 个入口文件：

- [run_pytorch_download.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_download.py)
- [run_pytorch_train.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_train.py)
- [run_pytorch_evaluate.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_evaluate.py)
- [run_pytorch_predict.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_predict.py)

每个入口文件顶部都有一个 `CONFIG` 字典。一般只需要改这里，不需要动底层模块。

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

下载完成后，建议确认目录结构至少包含：

```text
exp2/VOC2012
├── JPEGImages
├── SegmentationClass
└── ImageSets/Segmentation
```

## 训练

从仓库根目录执行：

```bash
python exp2/run_pytorch_train.py
```

训练参数都在 [run_pytorch_train.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_train.py) 顶部的 `CONFIG` 里，改完直接运行即可。

训练前最常改的是这些项：

- `epochs`
- `batch_size`
- `lr`
- `log_interval`：多少个 step 打印一次训练日志
- `weights`：设为 `"voc"` 时直接使用 `torchvision` 提供的 VOC 预训练分割权重
- `backbone_weights`：设为 `"imagenet"` 时只加载 ImageNet 预训练主干
- `resume`
- `freeze_bn`
- `device`
- `output_dir`

推荐的第一次运行方式：

1. 先把 `epochs` 设小一些，例如 `1` 或 `2`
2. 保持 `batch_size=2` 或 `4`
3. 先确认能正常保存 `last.pth`、`best.pth` 和日志
4. 再改成正式训练轮数

训练时会记录这些内容：

- 当前运行时间戳
- 全部训练参数
- 数据集大小和 batch 数
- 每个 `log_interval` 的 step loss 和学习率
- 每个 epoch 的训练 loss
- 每次验证的 `loss`、`pixel_accuracy`、`mean_iou`
- 最新 checkpoint 和最佳 checkpoint 的保存位置

训练输出：

- `last.pth`：最新 checkpoint
- `best.pth`：按验证集 `mIoU` 保存的最佳 checkpoint
- `logs/train_时间戳.log`：训练文本日志
- `logs/train_config_时间戳.json`：本次训练配置
- `logs/train_history_时间戳.json`：每个 epoch 的历史指标
- `logs/train_summary_时间戳.json`：训练总结

一个典型的训练输出目录大概是：

```text
exp2/outputs/deeplabv3_resnet50
├── best.pth
├── last.pth
└── logs
    ├── train_20260403_101500.log
    ├── train_config_20260403_101500.json
    ├── train_history_20260403_101500.json
    └── train_summary_20260403_101500.json
```

## 评估

使用自己训练得到的模型：

```bash
python exp2/run_pytorch_evaluate.py
```

评估参数在 [run_pytorch_evaluate.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_evaluate.py) 的 `CONFIG` 里改。

常改项：

- `checkpoint`
- `weights`
- `backbone_weights`
- `split`
- `device`
- `output_dir`

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

评估输出：

- `output_dir/logs/evaluate_时间戳.log`
- `output_dir/evaluation_metrics_时间戳.json`

## 推理与可视化

对单张图像推理：

```bash
python exp2/run_pytorch_predict.py
```

推理参数在 [run_pytorch_predict.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_predict.py) 的 `CONFIG` 里改，`input` 可以是单张图，也可以是一个目录。输出包括：

- `*_mask.png`：调色板语义分割结果
- `*_overlay.png`：原图和分割图叠加可视化
- `logs/predict_时间戳.log`：推理日志
- `prediction_manifest_时间戳.json`：本次推理所有输入和输出文件清单

如果 `input` 指向目录，脚本会遍历其中的 `.jpg`、`.jpeg`、`.png`、`.bmp` 文件。

## 日志怎么看

如果你想回看某次实验，优先看下面这些文件：

1. 训练过程：`exp2/outputs/deeplabv3_resnet50/logs/train_*.log`
2. 训练指标汇总：`exp2/outputs/deeplabv3_resnet50/logs/train_history_*.json`
3. 某次评估结果：`exp2/outputs/evaluation/evaluation_metrics_*.json`
4. 某次推理输出清单：`prediction_manifest_*.json`

建议你每次正式实验都新开一次运行，不要手动覆盖日志文件。当前实现会自动按时间戳生成新日志文件。

## 一次完整实验怎么做

下面是一套比较稳妥的实验顺序：

1. 运行 `python exp2/run_pytorch_download.py`，确认数据目录完整。
2. 修改 [run_pytorch_train.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_train.py#L14) 里的训练参数。
3. 先跑一个小实验，确认日志、checkpoint、loss 都正常。
4. 再跑正式训练。
5. 修改 [run_pytorch_evaluate.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_evaluate.py#L14)，把 `checkpoint` 指向训练得到的 `best.pth`。
6. 运行评估脚本，记录 `mean_iou`、`pixel_accuracy`。
7. 修改 [run_pytorch_predict.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_predict.py#L14) 的 `input`，导出可视化图片。

## 常见问题

### 1. 评估时报找不到 checkpoint

说明 [run_pytorch_evaluate.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_evaluate.py#L14) 里的 `checkpoint` 路径还没改对，或者训练还没有生成 `best.pth`。

### 2. 训练太慢

- 优先把 `device` 设成 `"cuda"`，前提是机器有可用 GPU
- 先把 `epochs` 调小
- 先把 `batch_size` 调小到能稳定运行
- `num_workers` 可以从 `0`、`2`、`4` 之间试

### 3. 显存不够

- 先减小 `batch_size`
- 必要时减小 `crop_size`

### 4. 推理想一次处理一批图像

把 [run_pytorch_predict.py](/Users/wuzheng/projects/nndlExp/exp2/run_pytorch_predict.py#L14) 里的 `input` 改成目录路径即可。

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
- 写实验报告时，优先引用日志文件和 `json` 指标文件里的数值，避免手动抄错。
