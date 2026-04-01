# Exp2 图像语义分割实验说明

本目录提供了一个独立于原 MindSpore notebook 的 PyTorch 版本实验实现，目标是基于 Pascal VOC2012 数据集完成 DeepLabV3 语义分割的训练、验证与可视化。

当前目录中的关键文件：

- [2图像语义分割实验_数据准备.py](/Users/wuzheng/projects/nndlExp/exp2/2图像语义分割实验_数据准备.py)：下载 VOC2012 并生成灰度标签
- [2图像语义分割实验_pytorch.py](/Users/wuzheng/projects/nndlExp/exp2/2图像语义分割实验_pytorch.py)：训练、评估、可视化
- [2图像语义分割实验.ipynb](/Users/wuzheng/projects/nndlExp/exp2/2图像语义分割实验.ipynb)：原实验 notebook
- [2图像语义分割实验手册.docx](/Users/wuzheng/projects/nndlExp/exp2/2图像语义分割实验手册.docx)：原实验手册

## 1. 环境依赖

项目依赖已经在根目录 [pyproject.toml](/Users/wuzheng/projects/nndlExp/pyproject.toml) 中声明，核心依赖包括：

- `torch`
- `torchvision`
- `opencv-python`
- `numpy`
- `matplotlib`
- `Pillow`

如果你使用 `uv`：

```bash
uv sync
```

如果你使用系统 Python 或虚拟环境，请确保上述依赖已安装。

## 2. 实验目录结构

推荐的目录结构如下：

```text
exp2/
  2图像语义分割实验_数据准备.py
  2图像语义分割实验_pytorch.py
  logs/
  downloads/
  VOC2012/
```

其中 `VOC2012/` 目录中需要包含：

```text
VOC2012/
  JPEGImages/
  SegmentationClass/
  SegmentationClassGray/
  ImageSets/Segmentation/train.txt
  ImageSets/Segmentation/val.txt
```

## 3. 运行方式

这套脚本不是通过命令行参数驱动，而是通过文件顶部的 `RUN_CONFIG` 配置运行。

运行方式固定为：

```bash
python3 exp2/2图像语义分割实验_数据准备.py
python3 exp2/2图像语义分割实验_pytorch.py
```

你只需要先修改脚本顶部的 `RUN_CONFIG`，再直接执行脚本。

## 4. 数据准备脚本

文件：[2图像语义分割实验_数据准备.py](/Users/wuzheng/projects/nndlExp/exp2/2图像语义分割实验_数据准备.py)

用途：

- 下载 VOC2012 数据集压缩包
- 自动解压到本地 `VOC2012/`
- 将 `SegmentationClass/` 下的标签准备为 `SegmentationClassGray/`

### 4.1 推荐配置

```python
RUN_CONFIG = {
    "data_root": "./VOC2012",
    "download_dir": "./downloads",
    "log_dir": "./exp2/logs",
    "log_name": None,
    "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    "download_voc": True,
    "prepare_gray_masks": True,
    "force_download": False,
    "overwrite_gray_masks": False,
    "download_retries": 3,
}
```

### 4.2 注意事项

- 如果网络下载中断，脚本会自动删除残缺的 tar 文件并重试
- 如果 tar 文件损坏，脚本会在解压时报错并删除损坏文件
- 如果 `VOC2012/` 已存在且 `force_download=False`，则跳过下载

## 5. 训练/评估/可视化脚本

文件：[2图像语义分割实验_pytorch.py](/Users/wuzheng/projects/nndlExp/exp2/2图像语义分割实验_pytorch.py)

支持三种模式：

- `train`
- `eval`
- `visualize`

通过修改 `RUN_CONFIG["command"]` 切换。

---

## 6. 训练模式

### 6.1 从零开始训练

```python
RUN_CONFIG = {
    "command": "train",
    "data_root": "./VOC2012",
    "project_root": "./exp2",
    "crop_size": 513,
    "batch_size": 4,
    "num_classes": 21,
    "ignore_label": 255,
    "workers": 2,
    "seed": 1,
    "device": "auto",
    "use_amp": True,
    "image_mean": [103.53, 116.28, 123.675],
    "image_std": [57.375, 57.120, 58.395],
    "epochs": 3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "min_scale": 0.5,
    "max_scale": 2.0,
    "output": "./exp2/model_scratch.pth",
    "checkpoint": None,
    "init_mode": "scratch",
    "flip": True,
    "scales": [1.0],
    "num_images": 3,
    "save_dir": None,
    "log_dir": "./exp2/logs",
    "log_name": None,
}
```

### 6.2 用 torchvision 官方权重初始化训练

只需要将：

```python
"init_mode": "torchvision"
```

这样会使用 `torchvision` 官方的 `DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1` 作为初始化权重。

### 6.3 使用已有 checkpoint 继续训练

设置：

```python
"checkpoint": "./exp2/model_scratch.pth"
```

脚本会加载该模型参数，并继续训练。

---

## 7. 评估模式

评估模式会在 VOC2012 验证集上计算 `mean IoU`。

示例：

```python
RUN_CONFIG = {
    "command": "eval",
    "data_root": "./VOC2012",
    "batch_size": 1,
    "crop_size": 513,
    "num_classes": 21,
    "ignore_label": 255,
    "workers": 2,
    "device": "auto",
    "use_amp": False,
    "image_mean": [103.53, 116.28, 123.675],
    "image_std": [57.375, 57.120, 58.395],
    "checkpoint": "./exp2/model_scratch.pth",
    "flip": True,
    "scales": [1.0],
    "log_dir": "./exp2/logs",
    "log_name": None,
}
```

评估日志会记录：

- 使用的 checkpoint
- 验证集样本数量
- 处理进度
- 每类 IoU
- 最终 `mean IoU`

---

## 8. 可视化模式

可视化模式会从验证集中选取前 `num_images` 张图像，显示：

- 原图
- 预测结果叠加图
- Ground Truth 叠加图

示例：

```python
RUN_CONFIG = {
    "command": "visualize",
    "data_root": "./VOC2012",
    "crop_size": 513,
    "batch_size": 1,
    "num_classes": 21,
    "ignore_label": 255,
    "workers": 2,
    "device": "auto",
    "use_amp": False,
    "image_mean": [103.53, 116.28, 123.675],
    "image_std": [57.375, 57.120, 58.395],
    "checkpoint": "./exp2/model_scratch.pth",
    "flip": True,
    "scales": [1.0],
    "num_images": 3,
    "save_dir": "./exp2/vis",
    "log_dir": "./exp2/logs",
    "log_name": None,
}
```

如果 `save_dir` 设为 `None`，结果会直接弹窗显示；如果给定目录，结果会保存成图片文件。

---

## 9. 硬件支持建议

脚本支持：

- `cuda`：Nvidia GPU
- `mps`：Apple Silicon
- `cpu`：纯 CPU
- `auto`：自动选择 `cuda -> mps -> cpu`

### 9.1 Nvidia GPU PC 推荐

```python
"device": "cuda",
"batch_size": 4,
"workers": 2,
"use_amp": True,
```

说明：

- 会自动启用 CUDA AMP 混合精度
- DataLoader 会启用 `pin_memory`
- 数据搬运使用 `non_blocking=True`

### 9.2 M 芯片 MacBook 推荐

```python
"device": "mps",
"batch_size": 2,
"workers": 0,
"use_amp": False,
```

说明：

- MPS 下不启用 CUDA AMP
- `workers=0` 一般更稳，避免 macOS 下多进程读图带来的问题

### 9.3 CPU 推荐

```python
"device": "cpu",
"batch_size": 1,
"workers": 0,
"use_amp": False,
```

---

## 10. 日志记录

两个脚本现在都支持完整日志记录。

默认日志目录：

```text
exp2/logs/
```

### 10.1 日志内容

数据准备日志会记录：

- 运行配置
- 日志文件路径
- 下载地址
- 下载重试次数
- 解压过程
- 灰度标签转换数量

训练日志会记录：

- 运行配置
- 设备类型
- 是否启用 AMP
- 训练样本数
- 每轮 step 的 loss 和 lr
- 每个 epoch 的平均 loss
- checkpoint 保存路径

评估日志会记录：

- 使用的 checkpoint
- 验证集样本数
- 推理进度
- 每类 IoU
- mean IoU

可视化日志会记录：

- 使用的 checkpoint
- 每张图中出现的预测类别
- 每张图中出现的真实类别
- 可视化结果保存路径

### 10.2 自定义日志文件名

如果你不想使用自动时间戳命名，可以在配置中指定：

```python
"log_name": "train_mps_run1"
```

那么日志会保存为：

```text
exp2/logs/train_mps_run1.log
```

---

## 11. 推荐实验流程

### 11.1 第一步：准备数据

修改 [2图像语义分割实验_数据准备.py](/Users/wuzheng/projects/nndlExp/exp2/2图像语义分割实验_数据准备.py) 顶部 `RUN_CONFIG`，然后执行：

```bash
python3 exp2/2图像语义分割实验_数据准备.py
```

### 11.2 第二步：训练模型

修改 [2图像语义分割实验_pytorch.py](/Users/wuzheng/projects/nndlExp/exp2/2图像语义分割实验_pytorch.py) 顶部 `RUN_CONFIG["command"] = "train"`，配置好设备和参数，然后执行：

```bash
python3 exp2/2图像语义分割实验_pytorch.py
```

### 11.3 第三步：评估模型

把 `RUN_CONFIG["command"]` 改成 `"eval"`，并指定 `checkpoint`：

```bash
python3 exp2/2图像语义分割实验_pytorch.py
```

### 11.4 第四步：可视化结果

把 `RUN_CONFIG["command"]` 改成 `"visualize"`，并指定 `checkpoint`：

```bash
python3 exp2/2图像语义分割实验_pytorch.py
```

---

## 12. 常见问题

### 12.1 下载中断

如果下载过程中出现 `ContentTooShortError`，通常是网络中断。当前脚本已经支持自动重试和损坏文件删除。

可以适当增大：

```python
"download_retries": 5
```

### 12.2 Mac 上 DataLoader 卡顿

建议将：

```python
"workers": 0
```

### 12.3 训练显存不足

优先调整：

- `batch_size`
- `crop_size`

例如：

```python
"batch_size": 2,
"crop_size": 321,
```

### 12.4 评估时报 checkpoint 不存在

确认 `RUN_CONFIG["checkpoint"]` 指向的是本地 `.pth` 文件，而不是 MindSpore 的 `.ckpt` 文件。

---

## 13. 当前实现说明

本 PyTorch 实现尽量对齐原实验流程，但模型实现上使用的是 `torchvision` 的 `DeepLabV3-ResNet50`，而不是原 notebook 中的 MindSpore 自定义实现。对于实验复现和本地运行，这样更稳，也更便于在 Mac 和 Nvidia GPU 上统一使用。
