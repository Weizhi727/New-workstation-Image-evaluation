import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2

from transformers import pipeline
from transformers.image_utils import load_image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def compute_patch_similarity_heatmap(patch_features, H, W, target_patch_coord):
    """
    计算指定patch与其他所有patch的余弦相似性，并生成热力图

    Args:
        patch_features: patch特征张量, shape (1, num_patches, feature_dim)
        H: patch网格高度
        W: patch网格宽度
        target_patch_coord: 目标patch坐标 (h_idx, w_idx)

    Returns:
        heatmap: 相似性热力图, shape (H, W)
    """

    assert patch_features.shape[1] == H * W, f"特征数量{H * W}与网格大小{H}x{W}不匹配"

    # 提取目标patch的特征
    target_idx = target_patch_coord[0] * W + target_patch_coord[1]
    target_feature = patch_features[0, target_idx]  # shape (feature_dim,)

    # 计算余弦相似性
    similarities = F.cosine_similarity(
        target_feature.unsqueeze(0),  # shape (1, feature_dim)
        patch_features[0],  # shape (num_patches, feature_dim)
        dim=1
    )

    # 重塑为2D热力图
    heatmap = similarities.reshape(H, W).cpu().numpy()

    return heatmap


def get_last_self_attention(model, x):
    """
    提取 DINOv3 最後一層的 Self-Attention Map
    """
    # DINOv3 (ViT) 通常透過 get_last_selfattention 或內部 blocks 訪問
    # 這裡假設是標準的 ViT 結構，我們直接拿最後一個 block 的 attention
    last_block = model.blocks[-1]

    # 執行前向傳播直到最後一層
    with torch.no_grad():
        # 準備 tokens
        x, (H, W) = model.prepare_tokens_with_masks(x)

        # 遍歷所有 blocks 到倒數第二個
        for blk in model.blocks[:-1]:
            x = blk(x)

        # 針對最後一個 block 手動計算 attention
        # 這裡簡化流程：獲取 attention weights
        B, N, C = x.shape
        qkv = last_block.attn.qkv(last_block.norm1(x)).reshape(B, N, 3, last_block.attn.num_heads,
                                                               C // last_block.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * last_block.attn.scale
        attn = attn.softmax(dim=-1)  # shape: (B, num_heads, num_patches+1, num_patches+1)

    return attn, H, W


def visualize_attention(attn, H, W, head_idx=None):
    """
    可視化 CLS Token 對所有 Patch 的注意力
    """
    # 提取 CLS token 對所有 patches 的注意力 (不含自己)
    # attn shape: (1, num_heads, N, N), N = H*W + 1 (if has CLS token)
    nh = attn.shape[1]  # 獲取 head 數量

    # 我們通常取 CLS token (index 0) 對其他所有 patches 的注意
    # 去掉 CLS 對自己的注意，取後面的 H*W 個值
    mask = attn[0, :, 0, 1:].reshape(nh, H, W)

    if head_idx is not None:
        # 顯示特定 head
        plt.imshow(mask[head_idx].cpu().numpy(), cmap='inferno')
        plt.title(f"Head {head_idx} Attention")
    else:
        # 顯示所有 head 的平均值 (最常用於觀察語義結構)
        avg_mask = torch.mean(mask, dim=0)
        plt.imshow(avg_mask.cpu().numpy(), cmap='inferno')
        plt.title("Average Attention Map (Across all heads)")

    plt.colorbar()
    plt.show()


def get_last_self_attention_optimized(model, x, max_size=512):
    """
    優化版本：限制最大尺寸以防止 OOM，並提取 Attention
    """
    # 1. 縮放圖像以節省顯存
    _, _, h, w = x.shape
    ratio = max_size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    # 確保是 patch_size (16) 的倍數
    new_h = (new_h // 16) * 16
    new_w = (new_w // 16) * 16
    x = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)

    last_block = model.blocks[-1]

    with torch.no_grad():
        x, (H, W) = model.prepare_tokens_with_masks(x)
        for blk in model.blocks[:-1]:
            x = blk(x)

        # 這裡不計算完整的 N x N 矩陣，只計算 CLS token 對其他 patch 的 attention
        B, N, C = x.shape
        num_heads = last_block.attn.num_heads
        head_dim = C // num_heads

        qkv = last_block.attn.qkv(last_block.norm1(x)).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        # 關鍵優化：只計算 CLS token (index 0) 與其他所有 token 的相似度
        # 這樣矩陣大小從 (N, N) 降到 (1, N)
        cls_q = q[:, :, 0:1, :]  # (B, heads, 1, head_dim)
        attn = (cls_q @ k.transpose(-2, -1)) * last_block.attn.scale  # (B, heads, 1, N)
        attn = attn.softmax(dim=-1)

    return attn, H, W

def plot_similarity_heatmap(heatmap, target_patch_coord):
    """
    绘制相似性热力图，并在目标patch位置显示红点

    Args:
        heatmap: 相似性热力图, shape (H, W)
        target_patch_coord: 目标patch坐标 (h_idx, w_idx)
        original_img_size: 原始图像尺寸 (可选，用于调整显示比例)
        patch_size: 每个patch的像素大小
    """
    H, W = heatmap.shape

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 显示热力图
    im = ax.imshow(heatmap, cmap='viridis', aspect='equal')

    # 在目标patch位置添加红点
    target_h, target_w = target_patch_coord
    ax.plot(target_w, target_h, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)

    # 添加颜色条
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    # 设置坐标轴标签
    ax.set_xlabel('Width (patch index)')
    ax.set_ylabel('Height (patch index)')
    ax.set_title(f'Cosine Similarity to Patch at ({target_h}, {target_w})')

    # 设置网格线
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # 设置主刻度
    ax.set_xticks(np.arange(0, W, max(1, W // 10)))
    ax.set_yticks(np.arange(0, H, max(1, H // 10)))

    plt.tight_layout()
    plt.show()

    return fig, ax


def apply_clahe_to_tensor(img_tensor):
    """
    將 CLAHE 應用於 Tensor 圖像 (B, C, H, W)
    """
    # 轉回 Numpy 並轉換為 YUV 空間以保持色彩，或直接轉灰階
    img_np = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img_np = (img_np * 255).astype(np.uint8)

    # 轉換到 LAB 顏色空間進行亮度均衡
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # clipLimit 越大對比度越強，gridSize 是局部塊的大小
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # 轉回 Tensor
    enhanced_tensor = torch.from_numpy(enhanced_img.transpose(2, 0, 1)).float() / 255.0
    return enhanced_tensor.unsqueeze(0).cuda()

# 你的图像的位置
url = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\T1_F0 (3).JPG"
image = load_image(url)
print(image)

transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为Tensor (0-1范围)
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 可选：标准化
])
tensor_image = transform(image).unsqueeze(0)
image = tensor_image.cuda()

pretrained_model_name = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\dinov3 weight\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
model = torch.hub.load("./", 'dinov3_vitb16', source='local', weights=pretrained_model_name)
model.cuda()
print(model)

class DINOv3Encoder:
    def __init__(self, repo_dir = './', model_name = 'dinov3_vitb16', model_path= r"D:\PycharmProjects\other\dinov3-main\dinov3-main\dinov3 weight\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.model = torch.hub.load(repo_dir, model_name, source='local', weights= model_path)

all_attributes = dir(model)
print("所有属性和方法:")
for attr in all_attributes:
    print(attr)
print("\n只显示方法（函数）:")
methods = [attr for attr in all_attributes if callable(getattr(model, attr))]
for method in methods:
    print(method)

# with torch.inference_mode():
#     tokens, (H, W) = model.prepare_tokens_with_masks(image)
#     print(f"Patch 分辨率: 高度 {H} patches, 宽度 {W} patches")
#     print(f"总共 {H * W} 个 patches")
#
#     features_dict = model.forward_features(image)
#
#     patch_features = features_dict["x_norm_patchtokens"]
#     print("patch_features.shape: ", patch_features.shape)
#
#     # 你选中的目标patch的坐标，注意要在[0,H-1]和[0,W-1]范围内。
#     target_patch_coord = (H//2, W//2)
#     heatmap = compute_patch_similarity_heatmap(patch_features, H, W, target_patch_coord)
#
#     # plot_similarity_heatmap(heatmap, target_patch_coord)
#
#     # 這裡 image 是你之前讀入的 tensor
#     attn, H, W = get_last_self_attention_optimized(model, image, max_size=1024)
#
#     # 提取 CLS 對 Patch 的注意力 (去掉 CLS 自己，所以從 index 1 開始)
#     # 注意 DINOv3 可能有 register tokens，這部分需要根據實際 token 數量切片
#     # 這裡假設標準狀況：第一個是 CLS，後面是 H*W 個 patches
#     nh = attn.shape[1]
#     mask = attn[0, :, 0, -H * W:].reshape(nh, H, W)
#
#     # 顯示平均後的 Attention Map
#     avg_mask = torch.mean(mask, dim=0).cpu().numpy()
#     plt.imshow(avg_mask, cmap='magma')
#     plt.title("Image Quality Analysis - Attention Map")
#     plt.show()

# with torch.inference_mode():
#     # 1. 直接釋放之前的顯存緩存，確保乾淨
#     torch.cuda.empty_cache()
#
#     # 2. 只使用優化版本，max_size 設為 512 (如果成功再調大到 896 或 1024)
#     # 注意：這裡直接傳入 image，函數內部會自動 Resize
#     attn, H, W = get_last_self_attention_optimized(model, image, max_size=1024)
#
#     # 3. 提取 CLS 對 Patch 的注意力
#     # DINOv3 的 token 結構通常是 [CLS, *Register_Tokens, *Patch_Tokens]
#     # 我們取最後 H*W 個作為 Patch 的注意力
#     nh = attn.shape[1]
#     mask = attn[0, :, 0, -H * W:].reshape(nh, H, W)
#
#     # 4. 顯示結果
#     avg_mask = torch.mean(mask, dim=0).cpu().numpy()
#
#     plt.figure(figsize=(10, 7))
#     plt.imshow(avg_mask, cmap='magma')
#     plt.colorbar(label='Attention Intensity')
#     plt.title(f"DINOv3 Attention Map (Resized to {H * 16}x{W * 16})")
#     plt.axis('off')
#     plt.show()
#
#     print(f"成功生成熱力圖！當前 Patch 網格尺寸: {H}x{W}")


# 先進行增強在進行attention feature
with torch.inference_mode():
    torch.cuda.empty_cache()

    # 1. 進行圖像增強 (CLAHE)
    print("正在進行圖像對比度增強...")
    enhanced_image = apply_clahe_to_tensor(image)

    # 2. 獲取優化後的 Attention (使用增強後的圖)
    # 建議先用 max_size=512 測試，成功後再慢慢調大
    attn, H, W = get_last_self_attention_optimized(model, enhanced_image, max_size=1024)

    # 3. 處理 Attention Map
    nh = attn.shape[1]
    mask = attn[0, :, 0, -H * W:].reshape(nh, H, W)
    avg_mask = torch.mean(mask, dim=0).cpu().numpy()

    # 4. 可視化對比：原始圖 vs 增強圖 vs Attention
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # 顯示增強後的輸入圖 (Resize 後)
    resized_input = F.interpolate(enhanced_image, size=(H * 16, W * 16),
                                  mode='bilinear').squeeze().cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(resized_input)
    axes[0].set_title("Enhanced Input (CLAHE)")
    axes[0].axis('off')

    # 顯示 Attention Map
    im = axes[1].imshow(avg_mask, cmap='magma')
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title("DINOv3 Attention (After Enhancement)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    if np.max(avg_mask) < (1.0 / (H * W) * 2):  # 經驗值判斷
        print("警告：Attention 分布仍極度均勻，模型可能仍無法識別焊點特徵。")
    else:
        print("分析完成。請觀察亮點是否集中於焊點。")