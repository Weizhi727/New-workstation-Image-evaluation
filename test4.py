import torch
import sys
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import OrderedDict
import os

# --- 1. 環境與顯存優化設定 ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
REPO_DIR = './'
sys.path.append(REPO_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# 檔案路徑
pretrained_model_name = r"D:\Nairb_project\Dinov3\dinov3-main\dinov3-main\dinov3 weight\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
pretrained_detector_weight = r"D:\Nairb_project\Dinov3\dinov3-main\dinov3-main\dinov3 weight\dinov3_vit7b16_coco_detr_head-b0235ff7.pth"
img_a_path = "./DR8HP9002BV0001B6V+A_big_yiwu_crop_0.9507666230201721.jpg"
img_b_path = "./DR8HP60067P0001B6V+A_big_yiwu_crop_0.9993754029273987.jpg"
# img_b_path = "./0.jpg"
# img_b_path = "./DR9HGS000FL00013S5+7_bf.jpg"




# --- 2. 權重清洗函數 (解決 Unexpected key 報錯) ---
def get_clean_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict


# --- 3. 載入模型 ---
print("正在初始化 DINOv3 模型...")
# 使用 try-except 確保結構建立，並關閉自動下載
model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=None, pretrained=False).to(device)

# 手動注入清洗後的權重
backbone_sd = get_clean_state_dict(pretrained_model_name)
model.load_state_dict(backbone_sd, strict=False)
model.eval()

# --- 4. 圖像預處理設定 ---
IMG_H, IMG_W = 602, 1204  # 根據您的需求設定
PATCH_SIZE = 16  # ViT-B/16 的 Patch 為 16
GRID_H, GRID_W = IMG_H // PATCH_SIZE, IMG_W // PATCH_SIZE  # 37 x 75

transform = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- 5. 核心功能函數 ---
def get_feature_map(img_path):
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():  # 比 no_grad 更省顯存
        # 提取最後一層特徵
        features = model.get_intermediate_layers(input_tensor, n=1)[0]
    return features.cpu()  # 移至 CPU 處理以節省 GPU 顯存


def get_region_feature(features, x1, y1, x2, y2):
    """根據像素區域提取平均特徵向量"""
    # 像素轉為網格座標
    gx1, gy1 = max(0, x1 // PATCH_SIZE), max(0, y1 // PATCH_SIZE)
    gx2, gy2 = min(GRID_W - 1, x2 // PATCH_SIZE), min(GRID_H - 1, y2 // PATCH_SIZE)

    indices = []
    for y in range(gy1, gy2 + 1):
        for x in range(gx1, gx2 + 1):
            indices.append(y * GRID_W + x)

    # 對選定區域的所有 patch 取平均
    region_feat = features[0, indices, :].mean(dim=0)
    return region_feat


# --- 6. 執行分析 ---
print("提取特徵圖...")
feat_a = get_feature_map(img_a_path)
feat_b = get_feature_map(img_b_path)

# 設定區域 (ROI): 左上(200, 200) 到 右下(330, 380)
# 若只要單點 (200, 300)，則設為 200, 300, 201, 301
roi = (150, 200, 300, 380)
query_vec = get_region_feature(feat_a, *roi)

# 計算相似度
feat_b_norm = F.normalize(feat_b[0], p=2, dim=-1)
query_norm = F.normalize(query_vec, p=2, dim=-1)
similarity = torch.matmul(feat_b_norm, query_norm)  # [GRID_H * GRID_W]
sim_map = similarity.reshape(GRID_H, GRID_W).numpy()

# --- 7. 視覺化結果 ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 圖 A: 顯示原始選取區域
img_a = Image.open(img_a_path).resize((IMG_W, IMG_H))
axes[0].imshow(img_a)
rect = patches.Rectangle((roi[0], roi[1]), roi[2] - roi[0], roi[3] - roi[1],
                         linewidth=2, edgecolor='r', facecolor='none')
axes[0].add_patch(rect)
axes[0].set_title("Source Image: Selected ROI")

# 圖 B: 顯示匹配結果熱圖
img_b = Image.open(img_b_path).resize((IMG_W, IMG_H))
axes[1].imshow(img_b)
heat = axes[1].imshow(sim_map, cmap='jet', alpha=0.5, extent=(0, IMG_W, IMG_H, 0))

# 找出最強匹配點並標記
max_idx = torch.argmax(similarity).item()
my, mx = divmod(max_idx, GRID_W)
axes[1].plot(mx * PATCH_SIZE + 8, my * PATCH_SIZE + 8, 'y*', markersize=15, markeredgecolor='black')
axes[1].set_title("Target Image: Similarity Matching")

plt.colorbar(heat, ax=axes[1])
plt.tight_layout()
plt.show()

print(f"匹配完成！目標位置約在像素: X={mx * PATCH_SIZE + 8}, Y={my * PATCH_SIZE + 8}")