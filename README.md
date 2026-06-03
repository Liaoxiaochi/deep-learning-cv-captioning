# 深度学习：图像分类与图像描述生成

> 利兹大学计算机科学硕士课程 **COMP5625M 深度学习** 作业。
> 基于 PyTorch 的端到端实现，涵盖卷积神经网络图像分类，以及结合 RNN 与 Transformer（CLIP + GPT-2）的图像描述生成。

本项目分为两部分：

| 部分 | 任务 | 关键技术 |
|------|------|---------|
| **一** | **TinyImageNet30** 图像分类（30 类、64×64） | 5 层 CNN、数据增强、CutMix、余弦退火、集成 + TTA |
| **二** | **图像描述生成** | ResNet-50 特征编码 + RNN 解码；CLIP 视觉编码 + Transformer 映射 + GPT-2 解码 |

---

## 第一部分 — CNN 图像分类

在 30 类 TinyImageNet 子集（13,500 张 64×64 图像）上**从零训练**一个自定义的 5 层卷积网络（`TinyCNN`）。

**核心要点**（见 [`src/train_improved.py`](src/train_improved.py)）：

- **网络结构** — 5 个卷积块（通道数 64→128→256→256→512），配合 BatchNorm + ReLU + MaxPool，
  最后接 1024 单元的全连接分类头与 Dropout。
- **正则化 / 数据增强** — 随机裁剪、水平翻转、旋转、颜色抖动、随机灰度、随机擦除、标签平滑，
  以及 **CutMix**。
- **优化策略** — AdamW（带权重衰减）配合**余弦退火**学习率调度。
- **推理** — **3 种子模型集成**结合**测试时增强（TTA）**（原图、水平翻转、中心裁剪、裁剪 + 翻转），
  对各路 logits 取平均。

整套方案针对小数据集上的过拟合问题，通过强数据增强与模型集成来提升泛化能力。

## 第二部分 — 图像描述生成

在 COCO 数据集上实现两种互补的图像描述方案：

1. **CNN 编码器 + RNN 解码器**（[`src/helperDL.py`](src/helperDL.py)）— 预训练 **ResNet-50**
   提取图像特征向量；对描述文本做清洗、分词并构建词表（按最小词频过滤）；序列通过
   padding / `pack_padded_sequence` 批处理后送入 RNN 语言模型。

2. **CLIP + Transformer + GPT-2**（[`src/model_CLIP.py`](src/model_CLIP.py)）— 冻结的 **CLIP**
   视觉编码器生成图像嵌入，**Transformer 编码器**将其映射到 GPT-2 的嵌入空间，再由部分冻结的
   **GPT-2** 解码器以温度采样的方式自回归生成描述。仅微调映射网络与 GPT-2 的首尾层，
   使模型保持轻量。

---

## 目录结构

```
deep-learning-cv-captioning/
├── notebooks/
│   └── assessment.ipynb         # 主作业 notebook（第一、二部分）
├── src/
│   ├── train_improved.py        # 第一部分：改进版 CNN 训练、集成 + TTA
│   ├── helperDL.py              # 第二部分：数据集/词表工具、ResNet-50 编码器
│   └── model_CLIP.py            # 第二部分：CLIP 编码器 + Transformer 映射 + GPT-2 解码器
├── docs/
│   └── rendered_notebook.html   # 已执行、含全部输出与图表的 notebook
├── .gitignore
└── README.md
```

> **说明** — 数据集（TinyImageNet30、COCO）、训练好的模型权重（`*.pth`）等大文件已刻意排除在
> 版本控制之外（见 `.gitignore`）。`docs/` 下的 HTML 保留了已执行的运行结果与图表。

## 技术栈

`Python` · `PyTorch` · `torchvision` · `Hugging Face Transformers`（CLIP、GPT-2）·
`scikit-learn` · `pandas` · `NumPy` · `Pillow` · `Matplotlib`

## 运行方式

notebook 面向 **Google Colab GPU 运行时**设计：

1. 在 Colab 中打开 `notebooks/assessment.ipynb` 并选择 GPU 运行时。
2. 为第一部分准备 TinyImageNet30 数据集（`train_set/`、`test_set/`），第二部分准备 COCO 描述与图像。
3. 自上而下运行各单元。第一部分中，`src/train_improved.py` 可作为单元运行以训练集成模型。

---

*作者：廖枭驰（Xiaochi Liao），利兹大学。*
