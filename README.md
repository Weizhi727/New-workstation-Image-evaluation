# New-workstation-Image-evaluation
use new computer vision model to evaluate new stations images, to confirm that those images can train models or etc.


# DINOv3 新工站数据客观评估工具

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

## 📌 简介
本工具基于 **DINOv3** 自监督视觉Transformer，为工业视觉新工站提供**数据可行性客观评估**。在新工站上线前，通过量化指标回答三个关键问题：
- **数据复杂度**：OK样本的外观变化有多大？是否需要更多数据？
- **OK/NG分离度**：缺陷在特征空间中是否容易与正常区分？
- **缺陷可见度**：缺陷信号相对于背景波动的强度如何？

这些指标帮助您避免盲目投入标注和训练，为是否适合部署检测模型提供数据驱动的决策依据。

## ✨ 核心特性
- ✅ **无需标注缺陷位置**：仅需OK/NG图像文件夹，自动聚焦异常区域。
- ✅ **聚焦ROI**：支持固定ROI或动态ROI（基于异常分数自动定位），排除背景干扰。
- ✅ **多维度量化**：三个互补指标全面评估数据质量。
- ✅ **轻量快速**：基于DINOv3 ViT-B/16，单张图像特征提取约0.1秒（GPU）。

## 🔧 安装

### 依赖
- Python 3.8+
- PyTorch 1.10+
- torchvision
- opencv-python
- scikit-learn
- scipy
- matplotlib
- Pillow

### 安装步骤
```bash
# 克隆仓库（或创建项目目录）
git clone https://github.com/yourname/dinov3-data-eval.git
cd dinov3-data-eval

# 安装依赖
pip install torch torchvision opencv-python scikit-learn scipy matplotlib pillow

# 下载DINOv3预训练权重
# 例如：ViT-B/16 权重（约330MB）
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain.pth
