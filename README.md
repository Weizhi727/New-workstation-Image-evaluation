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


# 希望claude code做的事情
1. 讀取目前這個repository的內容跟上傳的一些代碼當作參考，幫我生成一個前端頁面，功能是可以手動放入一個資料夾(裡面有多張圖片)或是一張圖片，結果可視化出我的評估結果，如果有甚麼通過dinov3生成的可視化圖像也可以展示在前端頁面上。
2. 前端第二個功能是ai sorting，通過放入一個資料夾(裡面有多張圖片)或是一張圖片當作標準模板，對這些標準模板在前端介面可以手動框定，並輸入名稱。執行完這些步驟之後，可以在另一個地方去放入一個文件夾(裡面有多張圖片)，幫我返回跟手動框相關的圖片，並將這些圖片都可視化在前端頁面，也提供下載成一個zip檔案。
```






# 雲智判上線自動化流程分析

[![Python 3.8+](<https://img.shields.io/badge/python-3.8+-blue.svg>)](<https://www.python.org/>)
[![PyTorch](<https://img.shields.io/badge/PyTorch-1.10+-red.svg>)](<https://pytorch.org/>)
[![License: MIT](<https://img.shields.io/badge/License-MIT-yellow.svg>)](LICENSE)

> 本專案記錄工業視覺 AI 模型（雲智判）從**初期評估**到**上線部署**的完整自動化流程，並以魚骨圖（Ishikawa Diagram）呈現各階段的核心缺失問題與建議補足技術。

---

## 魚骨圖總覽

![魚骨圖](fishbone_diagram.png)

互動版魚骨圖請直接開啟：[fishbone_diagram.html](fishbone_diagram.html)

---

## 流程階段與缺失分析

### 1. 初期評估
**現況：** 需求評估目前仍完全由開發與規劃人員以人工方式進行，缺乏客觀量化依據。

| 現存問題 | 建議補足技術 |
|---|---|
| 缺乏量化評估指標 | 自動化指標評估模組（DINOv3） |
| 缺乏模型可行性評估工具 | 數據可行性評分系統 |
| 完全依賴人工主觀判斷 | — |
| 評估時程冗長 | — |

---

### 2. 數據標註
**現況：** 評估完成後，由開發人員全程人工標註，效率低、一致性難以保證。

| 現存問題 | 建議補足技術 |
|---|---|
| 完全依賴人工手動標註 | 半自動 / 主動學習標註工具 |
| 缺乏通用標註方法/框架 | 通用標註標準與流程 |
| 標註一致性難以保證 | — |
| 標註效率低，開發時程長 | — |

---

### 3. 模型訓練
**現況：** 前期需要人工撰寫訓練程式碼，超參數調整仍依賴工程師個人經驗。

| 現存問題 | 建議補足技術 |
|---|---|
| 需要手動撰寫訓練程式碼 | AutoML / 訓練流水線自動化 |
| 超參數調整依賴人工經驗 | 超參數自動搜索（HPO） |
| 缺乏自動化訓練流水線 | — |
| 重複性工作佔用大量開發資源 | — |

---

### 4. 模型優化
**現況：** 上線後設備差異（曝光度、機台型號）常造成大量過殺，需要人工不斷重新 Label 優化，例如 10 台機台中只標了 2 台，其餘 8 台陸續上拋後仍需補標，缺乏自動化輔助方案。

| 現存問題 | 建議補足技術 |
|---|---|
| 設備差異導致大量過殺 | 自動域適應（Domain Adaptation） |
| 曝光度/機台調整造成圖像偏移 | AI 輔助標註 / 主動學習 |
| 新機台須大量重新人工 Label | 多機台差異自動補償 |
| 缺乏自動化優化與輔助標註方案 | — |

---

### 5. 用戶反饋
**現況：** 模型上線後僅能取得「復判 OK」的反饋資訊；漏檢 Sorting（從 OK 物料中找出潛在異常）目前缺乏對應模型支援，過殺與漏檢的反饋資訊也無法自動閉環回訓練流程。

| 現存問題 | 建議補足技術 |
|---|---|
| 僅能獲得「復判 OK」反饋資訊 | OK/NG 雙向反饋閉環系統 |
| 缺乏漏檢 Sorting 機制 | 漏檢異常 Sorting 模型 |
| 過殺/漏檢反饋無法閉環進模型 | 持續學習（Continual Learning） |
| 缺乏相似異常物料自動搜尋能力 | — |

---

## 魚骨圖生成

### 環境需求

```bash
pip install matplotlib numpy
