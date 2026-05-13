# 基于 Vision Transformer 的 CIFAR-10 图像分类实验

## 项目简介

本实验使用 PyTorch 在 CIFAR-10 数据集上完成 10 类图像分类，对比简单 CNN baseline 与 `torchvision.models.vit_b_16` Vision Transformer。项目支持自动下载数据、训练、验证、测试评估、保存最佳权重、绘制训练曲线、混淆矩阵、预测样例和模型对比图。

## 实验目的

- 熟悉 CIFAR-10 图像分类流程。
- 实现一个简单 CNN baseline。
- 调用 ImageNet 预训练 ViT-B/16 并替换分类头。
- 对比 CNN 与 ViT 的准确率、宏平均 F1、参数量和训练方式。

## 环境安装

在仓库根目录运行：

```bash
uv sync
```

本实验依赖 PyTorch、torchvision、matplotlib、numpy、scikit-learn、tqdm。若启用 TensorBoard 日志但环境未安装 `tensorboard`，训练仍会正常进行，只是不写 TensorBoard 事件文件。

## 数据集说明

数据集使用 `torchvision.datasets.CIFAR10`：

- 训练集：`train=True`
- 测试集：`train=False`
- 自动下载：`download=True`
- 保存路径：仓库根目录 `./data`
- 类别：`airplane`、`automobile`、`bird`、`cat`、`deer`、`dog`、`frog`、`horse`、`ship`、`truck`

训练集会按 `9:1` 划分为训练集和验证集。训练和测试图像统一 resize 到 `224 x 224`，并使用 CIFAR-10 的均值和标准差归一化。

## 模型介绍

`SimpleCNN` 位于 `models/cnn.py`，结构为三组 `Conv2d + BatchNorm2d + ReLU + MaxPool2d`，随后使用 `AdaptiveAvgPool2d` 和两层全连接分类器输出 10 类。

`build_vit_model` 位于 `models/vit.py`。在新版 torchvision 中会加载 `ViT_B_16_Weights.DEFAULT` ImageNet 预训练权重并替换最后分类头为 10 类输出；在较旧但仍提供 `vit_b_16` 的 torchvision 中会自动兼容旧的 `pretrained=` 参数；如果服务器的 torchvision 完全没有 ViT，则退回到一个轻量级自实现 ViT，保证代码可以继续运行。`train_mode="head_only"` 只训练分类头，`train_mode="full"` 微调整个 ViT。

## 运行方法

不使用命令行参数，直接运行：

```bash
uv run python exp4/main.py
```

主要配置位于 `exp4/main.py` 顶部：

- `RUN_CONFIG`：随机种子、设备、数据路径、输出路径、图像尺寸、是否下载数据、是否限制样本量。
- `download_timeout_seconds`：CIFAR-10 联网下载的 socket 超时时间。服务器网络不通时会快速报错，避免无提示卡住。
- `MODEL_CONFIGS`：CNN 和 ViT 的 epoch、batch size、学习率、权重衰减、ViT 训练模式。

完整实验保持 `train_subset`、`val_subset`、`test_subset` 为 `None`。本地快速冒烟测试可把它们改成小整数。

## 实验结果保存位置

每次运行会创建：

```text
exp4/outputs/{时间戳}/
├── figures/
│   ├── cnn_loss_curve.png
│   ├── cnn_acc_curve.png
│   ├── cnn_confusion_matrix.png
│   ├── cnn_prediction_samples.png
│   ├── vit_loss_curve.png
│   ├── vit_acc_curve.png
│   ├── vit_confusion_matrix.png
│   ├── vit_prediction_samples.png
│   └── model_comparison.png
├── checkpoints/
│   ├── cnn_best.pth
│   ├── cnn_last.pth
│   ├── vit_best.pth
│   └── vit_last.pth
└── logs/
    ├── cnn_history.csv
    ├── cnn_metrics.json
    ├── cnn_classification_report.txt
    ├── vit_history.csv
    ├── vit_metrics.json
    ├── vit_classification_report.txt
    ├── comparison_summary.csv
    └── comparison_summary.json
```

`exp4/outputs/latest_experiment.txt` 会记录最近一次实验目录。

## 常见问题

1. `ImportError: cannot import name 'ViT_B_16_Weights'`：服务器 torchvision 版本过旧。当前代码已做兼容，新版环境使用官方预训练 ViT，旧版环境会退回到轻量级 ViT；如果想严格使用官方预训练 ViT，请升级 torchvision。
2. ViT 权重下载失败：确认网络可访问 PyTorch 权重地址，或先手动缓存 torchvision 的 ViT-B/16 权重。
3. 运行停在 `RequestsDependencyWarning` 附近：通常不是 warning 本身的问题，而是 CIFAR-10 正在下载或服务器无法访问官方数据源。手动把解压后的 `cifar-10-batches-py` 放到仓库根目录 `data/` 下，再把 `RUN_CONFIG["download"]` 改为 `False`。
4. CPU 训练过慢：优先使用 CUDA 或 Mac MPS；也可以临时设置 `train_subset`、`val_subset`、`test_subset` 做小样本调试。
5. MPS 显存不足：减小 `MODEL_CONFIGS["vit"]["batch_size"]`，或保持默认 `head_only` 训练模式。
6. TensorBoard 不生成：安装 `tensorboard` 后重新运行；不安装不会影响普通日志、图片和指标输出。
