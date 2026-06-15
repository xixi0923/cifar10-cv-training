# CIFAR-10 计算机视觉训练项目

> 基于 CIFAR-10 数据集，使用 OpenCV 进行图像预处理，PyTorch 进行模型训练。支持可配置 CNN 架构、可组合数据增强管线和完整的评估工具。

## 项目简介

本项目提供了一套**完整的生产级训练管线**，用于 CIFAR-10 图像分类任务，核心特性包括：

- **健壮性** — 输入校验、错误处理、可复现种子、早停机制
- **可复用** — 模块化架构、配置驱动设计、可组合增强管线
- **可扩展** — 轻松替换模型、增强策略或数据集

## 功能特性

- 可配置 CNN 架构（卷积层数/滤波器数/全连接层均可调）
- 基于 OpenCV 的可组合数据增强管线（6 种增强方式独立可配）
- AdamW 优化器 + Label Smoothing
- 学习率调度（Step / Cosine / None）
- 混合精度训练（AMP，PyTorch 2.0+ 新 API）
- 早停机制与自动检查点保存
- 完整评估：总体/分类准确率、混淆矩阵、训练曲线
- 命令行接口，所有超参数均可通过参数覆盖

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py (入口)                        │
│  CLI 参数解析 → 配置构建 → 日志/种子初始化 → 训练/评估       │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐ ┌──────────────┐
│  数据模块     │ │  模型模块     │
│  loader.py   │ │  network.py  │
│              │ │              │
│ Torchvision  │ │  ConvBlock   │
│  ↓ fallback  │ │    ↓ ×N      │
│ 本地二进制    │ │  AvgPool     │
│  ↓           │ │    ↓         │
│ CIFAR10Dataset│ │  FC Layers   │
│  ↓           │ │    ↓         │
│ DataLoader   │ │  Output(10)  │
└──────┬───────┘ └──────┬───────┘
       │               │
       ▼               ▼
┌──────────────┐ ┌──────────────┐
│  增强管线     │ │  训练管线     │
│ image_process│ │  train.py    │
│              │ │              │
│ RandomCrop   │ │ AdamW优化器   │
│ H-Flip      │ │ CrossEntropy │
│ V-Flip      │ │ LR Scheduler │
│ Rotation     │ │ AMP(混合精度) │
│ Brightness   │ │ Checkpoint   │
│ Contrast     │ │ Early Stop   │
│ GaussianNoise│ │              │
└──────────────┘ └──────┬───────┘
                      │
                      ▼
              ┌──────────────┐
              │  工具模块     │
              │  helpers.py  │
              │              │
              │ AvgMeter     │
              │ Accuracy     │
              │ ConfusionMat │
              │ Logging      │
              │ Plotting     │
              └──────────────┘
```

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.10+ | 主要编程语言 |
| PyTorch 2.0+ | 模型定义、训练循环、GPU 加速 |
| OpenCV | 图像预处理与数据增强 |
| NumPy | 数值计算 |
| Matplotlib | 训练曲线与混淆矩阵可视化 |
| Torchvision | CIFAR-10 数据集加载 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 默认参数训练

```bash
python main.py
```

### 3. 自定义参数训练

```bash
# 更长训练 + 更低学习率
python main.py --epochs 100 --lr 0.0005

# 更宽的模型
python main.py --filters 64,128,256 --dropout 0.3

# 禁用数据增强（基线对比）
python main.py --no-augment

# 仅使用 CPU
python main.py --device cpu --epochs 20

# 启用混合精度训练
python main.py --mixed-precision

# 使用余弦退火学习率
python main.py --lr-scheduler cosine
```

### 4. 评估已保存的检查点

```bash
python main.py --eval output/checkpoints/best.pt
```

## CLI 用法

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--batch-size` | 64 | 批次大小 |
| `--weight-decay` | 1e-4 | L2 正则化系数 |
| `--device` | auto | 设备：auto / cuda / cpu / mps |
| `--seed` | 42 | 随机种子 |
| `--lr-scheduler` | step | 学习率调度器：step / cosine / none |
| `--mixed-precision` | false | 启用混合精度训练（需 CUDA） |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--filters` | 32,64,128 | 卷积层滤波器数（逗号分隔） |
| `--dropout` | 0.5 | Dropout 率 |

### 增强参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no-augment` | false | 禁用所有数据增强 |
| `--rotation` | 15 | 最大旋转角度（度） |
| `--noise-std` | 5.0 | 高斯噪声标准差 |

### 输出参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save-every` | 10 | 每 N 轮保存检查点 |
| `--early-stop` | 10 | 早停耐心值（0=禁用） |
| `--output` | output | 输出目录 |
| `--eval` | None | 评估模式：指定检查点路径 |

## 项目结构

```
cifar10-cv-training/
├── main.py                    # 入口：训练 / 评估
├── config.py                  # 集中式超参数与路径配置
├── requirements.txt           # Python 依赖
├── README.md
├── LICENSE                    # GPL-3.0
├── .gitignore
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── loader.py          # CIFAR-10 加载（torchvision + 本地回退）
    ├── preprocessing/
    │   ├── __init__.py
    │   └── image_process.py   # OpenCV 增强管线
    ├── models/
    │   ├── __init__.py
    │   ├── network.py         # 可配置 CNN 架构
    │   └── train.py           # 训练循环、检查点、评估
    └── utils/
        ├── __init__.py
        └── helpers.py         # 指标、混淆矩阵、日志、绘图
```

## 配置说明

所有超参数集中在 `config.py` 中，使用 Python dataclass 定义，分为以下模块：

| 配置类 | 说明 | 关键字段 |
|--------|------|----------|
| `DataConfig` | 数据加载配置 | `batch_size`, `val_split`, `num_workers`, `image_size` |
| `AugmentationConfig` | 数据增强配置 | `random_horizontal_flip`, `random_rotation_range`, `gaussian_noise_std` |
| `ModelConfig` | 模型架构配置 | `conv_filters`, `fc_hidden_units`, `dropout_rate`, `image_size` |
| `TrainingConfig` | 训练过程配置 | `epochs`, `learning_rate`, `lr_scheduler`, `mixed_precision` |
| `LogConfig` | 日志配置 | `log_level`, `log_to_file`, `log_to_console` |

配置优先级：CLI 参数 > dataclass 默认值

## 优化修复日志

### P0 关键 Bug 修复

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `src/data/loader.py` | `_SubsetWithTransform` 中 float32 [0,1] 张量直接 `.astype(np.uint8)` 导致所有值截断为 0 | 改为 `(clip(0,1) * 255).astype(np.uint8)` |
| 2 | `config.py` | `ModelConfig` 缺少 `image_size` 字段，`network.py` 引用 `cfg.image_size` 时报 AttributeError | 添加 `image_size: int = 32` |
| 3 | `src/preprocessing/image_process.py` | `horizontal_flip`/`vertical_flip` 内部有 `random.random() < 0.5` 检查，与管线外部概率控制叠加导致实际概率 = `config_prob * 0.5` | 移除函数内部随机检查，概率完全由管线控制 |

### P1 重要 Bug 修复

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 4 | `src/models/train.py` | `from torch.cuda.amp import GradScaler, autocast` 在 PyTorch 2.0+ 已废弃 | 改为 `from torch.amp import GradScaler, autocast`，更新用法为 `autocast("cuda")` 和 `GradScaler("cuda", ...)` |
| 5 | `src/utils/helpers.py` | `confusion_matrix` 使用 Python for 循环，大数组性能差 | 改用 `np.add.at(matrix, (y_true, y_pred), 1)` 向量化实现 |
| 6 | `src/models/train.py` | `optimizer.zero_grad()` 未使用 `set_to_none=True` | 改为 `optimizer.zero_grad(set_to_none=True)` 减少内存分配 |
| 7 | `src/utils/helpers.py` | `setup_logging` 重复调用会叠加 handler，导致日志重复输出 | 添加 handler 前先 `root_logger.handlers.clear()` 清除已有 handler |
| 8 | `main.py` | 缺少 `--lr-scheduler` 和 `--mixed-precision` CLI 参数 | 添加对应参数并传入 `TrainingConfig` |

## 许可证

本项目基于 [GNU General Public License v3.0](LICENSE) 开源。所有修改和分发必须在相同许可证下保持开源。
