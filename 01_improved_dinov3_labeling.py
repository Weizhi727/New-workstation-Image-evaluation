# """
# DINOv3 輔助標記系統 - 多區域對比強化版本
# ====================================================
# 改進點：
# 1. 多樣本特徵池：分別提取 WP、WA、BG 的特徵池（不簡單平均）
# 2. 動態特徵匹配：使用最相似的正樣本和最不相似的負樣本進行對比
# 3. 空間感知分數：結合空間距離和特徵相似度
# 4. 區域生長算法：從高信度點開始進行區域生長
#
# Author: K-Dense System (Enhanced)
# Date: 2026-01-12
# """
#
# import json
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from glob import glob
# import matplotlib.patches as patches
# from scipy import ndimage
# import warnings
# from collections import defaultdict
#
# warnings.filterwarnings('ignore')
#
# # ========== 配置參數 ==========
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# PATCH_SIZE = 16
#
# # 權重參數
# ALPHA_WA = 0.8  # 焊前區域懲罰權重
# BETA_BG = 0.3   # 背景懲罰權重
# SPATIAL_DECAY = 0.02
# TOP_K_POSITIVE = 5  # 使用前K個最相似的正樣本
# TOP_K_NEGATIVE = 10  # 使用前K個最不相似的負樣本
#
# # ========== 改進特徵提取核心 ==========
#
# def get_optimal_size(w: int, h: int):
#     return (w // PATCH_SIZE) * PATCH_SIZE, (h // PATCH_SIZE) * PATCH_SIZE
#
# def get_feature_map(model, img_path, return_img=False):
#     """改進特徵提取，可選擇返回處理後的圖像"""
#     img = Image.open(img_path).convert('RGB')
#     orig_w, orig_h = img.size
#     img_w, img_h = get_optimal_size(orig_w, orig_h)
#
#     transform = T.Compose([
#         T.Resize((img_h, img_w)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     input_tensor = transform(img).unsqueeze(0).to(DEVICE)
#
#     with torch.inference_mode():
#         features = model.get_intermediate_layers(input_tensor, n=1)[0]
#
#     # 處理 ViT 輸出
#     grid_w, grid_h = img_w // PATCH_SIZE, img_h // PATCH_SIZE
#     expected_patches = grid_w * grid_h
#     if features.shape[1] == expected_patches + 1:
#         features = features[:, 1:, :]
#
#     if return_img:
#         return features, (img_w, img_h), img.resize((img_w, img_h))  # 不移到CPU
#     return features, (img_w, img_h)  # 不移到CPU
#
# def get_region_patch_indices(shape, target_size, orig_size):
#     """獲取區域內所有patch的索引"""
#     img_w, img_h = target_size
#     orig_w, orig_h = orig_size
#     grid_w = img_w // PATCH_SIZE
#     grid_h = img_h // PATCH_SIZE
#
#     points = shape['points']
#     scale_x, scale_y = img_w / orig_w, img_h / orig_h
#
#     # 計算邊界框
#     x_coords = [p[0] for p in points]
#     y_coords = [p[1] for p in points]
#     x_min, x_max = min(x_coords), max(x_coords)
#     y_min, y_max = min(y_coords), max(y_coords)
#
#     # 轉換到特徵圖座標
#     x1, x2 = x_min * scale_x, x_max * scale_x
#     y1, y2 = y_min * scale_y, y_max * scale_y
#
#     # 計算patch範圍
#     gx1, gx2 = int(max(0, x1 // PATCH_SIZE)), int(min(grid_w - 1, x2 // PATCH_SIZE))
#     gy1, gy2 = int(max(0, y1 // PATCH_SIZE)), int(min(grid_h - 1, y2 // PATCH_SIZE))
#
#     # 收集所有patch索引
#     indices = []
#     for y in range(gy1, gy2 + 1):
#         for x in range(gx1, gx2 + 1):
#             indices.append(y * grid_w + x)
#
#     return indices
#
# # ========== 改進：特徵池管理 ==========
#
# class FeaturePool:
#     """管理多類別特徵池"""
#
#     def __init__(self):
#         self.pools = {
#             'wp': [],  # 焊點特徵
#             'area': [],  # 焊前區域特徵
#             'bg': []    # 背景特徵
#         }
#         self.reference_positions = []
#
#     def add_features(self, features, json_data, target_size, orig_size):
#         """從一張圖片提取三類特徵"""
#         img_w, img_h = target_size
#         orig_w, orig_h = orig_size
#         grid_w = img_w // PATCH_SIZE
#
#         # 初始化mask
#         all_indices = set(range(features.shape[1]))
#         wp_indices = set()
#         area_indices = set()
#
#         # 收集各類別標註
#         for shape in json_data['shapes']:
#             indices = get_region_patch_indices(shape, target_size, orig_size)
#             if shape['label'] == 'wp':
#                 wp_indices.update(indices)
#                 # 記錄第一個wp的中心位置
#                 if not self.reference_positions and indices:
#                     idx = indices[0]
#                     ref_y, ref_x = idx // grid_w, idx % grid_w
#                     self.reference_positions.append((ref_y, ref_x))
#             # elif shape['label'] == 'area':
#             elif shape['label'] == 'ref':
#                 area_indices.update(indices)
#
#         # 背景區域 = 全部 - wp - area
#         bg_indices = list(all_indices - wp_indices - area_indices)
#         wp_indices = list(wp_indices)
#         area_indices = list(area_indices)
#
#         # 將特徵存入對應池中（不平均，保留所有特徵）
#         # 注意：這裡我們移到CPU存儲，以節省GPU內存
#         if wp_indices:
#             self.pools['wp'].extend([features[0, idx, :].cpu().clone() for idx in wp_indices])
#         if area_indices:
#             self.pools['area'].extend([features[0, idx, :].cpu().clone() for idx in area_indices])
#         if bg_indices:
#             # 對背景進行採樣，避免特徵過多
#             if len(bg_indices) > 10000:
#                 import random
#                 bg_indices = random.sample(bg_indices, 10000)
#             self.pools['bg'].extend([features[0, idx, :].cpu().clone() for idx in bg_indices])
#
#         return len(wp_indices), len(area_indices), len(bg_indices)
#
#     def get_statistics(self):
#         """獲取特徵池統計信息"""
#         stats = {}
#         for category, features in self.pools.items():
#             stats[category] = len(features)
#         return stats
#
# # ========== 改進：動態對比評分 ==========
#
# def dynamic_contrastive_scoring(test_features, feature_pool, spatial_positions=None):
#     """動態對比評分：為每個測試特徵選擇最相關的正負樣本"""
#     # 確保test_features在正確設備上
#     test_features = test_features.to(DEVICE)
#     num_patches = test_features.shape[0]
#
#     # 歸一化測試特徵
#     test_norm = F.normalize(test_features, p=2, dim=-1)
#
#     # 初始化分數
#     scores = torch.zeros(num_patches, device=DEVICE)
#
#     # 為每個類別準備特徵（只在需要時移動到GPU）
#     wp_features = None
#     area_features = None
#     bg_features = None
#
#     if feature_pool.pools['wp']:
#         wp_features = torch.stack(feature_pool.pools['wp']).to(DEVICE)
#     if feature_pool.pools['area']:
#         area_features = torch.stack(feature_pool.pools['area']).to(DEVICE)
#     if feature_pool.pools['bg']:
#         bg_features = torch.stack(feature_pool.pools['bg']).to(DEVICE)
#
#     # 批量計算相似度（效率更高）
#     if wp_features is not None:
#         # 批量計算所有測試特徵與所有WP特徵的相似度
#         wp_sim_all = torch.matmul(test_norm, wp_features.T)  # [num_patches, num_wp]
#         # 對每個測試特徵，取Top-K個最相似的WP特徵
#         top_k_wp = min(TOP_K_POSITIVE, wp_sim_all.shape[1])
#         wp_scores, _ = torch.topk(wp_sim_all, top_k_wp, dim=1)
#         wp_scores = wp_scores.mean(dim=1)  # [num_patches]
#     else:
#         wp_scores = torch.zeros(num_patches, device=DEVICE)
#
#     # 計算WA相似度（作為懲罰）
#     if area_features is not None:
#         area_sim_all = torch.matmul(test_norm, area_features.T)
#         # 取平均相似度作為懲罰
#         area_scores = area_sim_all.mean(dim=1)  # [num_patches]
#     else:
#         area_scores = torch.zeros(num_patches, device=DEVICE)
#
#     # 計算BG相似度（作為懲罰）
#     if bg_features is not None:
#         bg_sim_all = torch.matmul(test_norm, bg_features.T)
#         # 取平均相似度作為懲罰
#         bg_scores = bg_sim_all.mean(dim=1)  # [num_patches]
#     else:
#         bg_scores = torch.zeros(num_patches, device=DEVICE)
#
#     # 綜合分數：WP相似度 - WA懲罰 - BG懲罰
#     scores = wp_scores - ALPHA_WA * area_scores - BETA_BG * bg_scores
#
#     # 空間約束（如果有參考位置）
#     if spatial_positions:
#         grid_w = int(np.sqrt(num_patches)) if np.sqrt(num_patches).is_integer() else 0
#         if grid_w > 0:
#             # 創建空間權重矩陣
#             spatial_weights = torch.ones(num_patches, device=DEVICE)
#             for ref_y, ref_x in spatial_positions:
#                 # 為每個patch計算到參考點的距離
#                 for i in range(num_patches):
#                     y, x = i // grid_w, i % grid_w
#                     dist = np.sqrt((x - ref_x)**2 + (y - ref_y)**2)
#                     weight = np.exp(-SPATIAL_DECAY * dist)
#                     # 使用最大權重（即最接近的參考點）
#                     spatial_weights[i] = max(spatial_weights[i].item(), weight)
#
#             scores = scores * spatial_weights
#
#     return scores.cpu()
#
# # ========== 改進：區域生長檢測 ==========
#
# def region_growing_detection(score_map, threshold, min_size=3):
#     """基於區域生長的檢測算法"""
#     grid_h, grid_w = score_map.shape
#     binary_map = (score_map > threshold).astype(np.uint8)
#
#     # 連接區域標記
#     labeled_map, num_features = ndimage.label(binary_map)
#
#     detections = []
#     for i in range(1, num_features + 1):
#         region_mask = (labeled_map == i)
#         region_size = np.sum(region_mask)
#
#         if region_size < min_size:
#             continue
#
#         # 計算區域邊界
#         rows, cols = np.where(region_mask)
#         y_min, y_max = np.min(rows), np.max(rows)
#         x_min, x_max = np.min(cols), np.max(cols)
#
#         # 計算區域平均分數
#         region_score = np.mean(score_map[region_mask])
#
#         # 計算區域緊密度（邊界長度/面積）
#         from scipy.ndimage import binary_erosion
#         eroded = binary_erosion(region_mask)
#         boundary = region_mask & ~eroded
#         perimeter = np.sum(boundary)
#         compactness = 4 * np.pi * region_size / (perimeter ** 2) if perimeter > 0 else 0
#
#         detections.append({
#             'bbox': (x_min, y_min, x_max, y_max),
#             'score': region_score,
#             'size': region_size,
#             'compactness': compactness,
#             'mask': region_mask
#         })
#
#     # 按分數排序
#     detections.sort(key=lambda x: x['score'], reverse=True)
#     return detections
#
# # ========== 主程式邏輯 ==========
#
# def build_feature_pool(model, labeled_dir):
#     """建立特徵池"""
#     print("[1] 建立特徵池...")
#     feature_pool = FeaturePool()
#
#     json_files = glob(os.path.join(labeled_dir, "*.json"))
#     total_wp, total_area, total_bg = 0, 0, 0
#
#     for idx, json_path in enumerate(json_files, 1):
#         print(f"  處理 {idx}/{len(json_files)}: {os.path.basename(json_path)}")
#
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#
#         img_path = os.path.splitext(json_path)[0] + ".jpg"
#         if not os.path.exists(img_path):
#             print(f"  警告: 找不到圖片 {img_path}")
#             continue
#
#         # 提取特徵
#         features, target_size = get_feature_map(model, img_path)
#         orig_size = (data['imageWidth'], data['imageHeight'])
#
#         # 添加到特徵池
#         wp_count, area_count, bg_count = feature_pool.add_features(
#             features, data, target_size, orig_size
#         )
#
#         total_wp += wp_count
#         total_area += area_count
#         total_bg += bg_count
#
#     # 輸出統計
#     stats = feature_pool.get_statistics()
#     print(f"\n特徵池統計:")
#     print(f"  WP特徵數: {stats['wp']} (來自 {total_wp} 個patch)")
#     print(f"  WA特徵數: {stats['area']} (來自 {total_area} 個patch)")
#     print(f"  BG特徵數: {stats['bg']} (來自 {total_bg} 個patch, 採樣到 {min(stats['bg'], 10000)})")
#     print(f"  參考位置數: {len(feature_pool.reference_positions)}")
#
#     return feature_pool
#
# def test_with_dynamic_scoring(model, test_dir, feature_pool, output_dir="./results"):
#     """使用動態評分進行測試"""
#     print("\n[2] 開始測試...")
#     test_images = glob(os.path.join(test_dir, "*.jpg"))
#     os.makedirs(output_dir, exist_ok=True)
#
#     all_detections = []
#
#     for img_idx, img_path in enumerate(test_images, 1):
#         print(f"\n  測試 {img_idx}/{len(test_images)}: {os.path.basename(img_path)}")
#
#         # 提取特徵和圖像
#         features, target_size, img_resized = get_feature_map(
#             model, img_path, return_img=True
#         )
#         img_w, img_h = target_size
#         grid_w, grid_h = img_w // PATCH_SIZE, img_h // PATCH_SIZE
#
#         # 動態評分
#         scores = dynamic_contrastive_scoring(
#             features[0],  # 已經是GPU上的張量
#             feature_pool,
#             spatial_positions=feature_pool.reference_positions
#         )
#
#         score_map = scores.numpy().reshape(grid_h, grid_w)
#
#         # 自適應閾值
#         threshold = np.percentile(score_map, 98)  # 前2%
#
#         # 區域生長檢測
#         detections = region_growing_detection(score_map, threshold)
#
#         # 過濾重疊檢測（NMS簡化版）
#         filtered_detections = []
#         used_mask = np.zeros((grid_h, grid_w), dtype=bool)
#
#         for det in detections:
#             x_min, y_min, x_max, y_max = det['bbox']
#             region_mask = det['mask']
#
#             # 檢查是否與已選區域重疊
#             overlap = np.sum(region_mask & used_mask) / np.sum(region_mask)
#             if overlap > 0.3:  # 重疊超過30%則跳過
#                 continue
#
#             filtered_detections.append(det)
#             used_mask = used_mask | region_mask
#
#             # 限制最多檢測數量
#             if len(filtered_detections) >= 5:
#                 break
#
#         # 可視化
#         fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#         axes = axes.flatten()
#
#         # 1. 原始圖像
#         axes[0].imshow(img_resized)
#         axes[0].set_title("原始圖像")
#         axes[0].axis('off')
#
#         # 2. 熱力圖
#         im = axes[1].imshow(score_map, cmap='jet')
#         axes[1].set_title(f"動態對比分數\n閾值: {threshold:.3f}")
#         axes[1].axis('off')
#         plt.colorbar(im, ax=axes[1])
#
#         # 3. 二值化結果
#         binary_map = (score_map > threshold).astype(np.uint8)
#         axes[2].imshow(binary_map, cmap='gray')
#         axes[2].set_title(f"二值化結果\n檢測區域: {len(filtered_detections)}")
#         axes[2].axis('off')
#
#         # 4. 檢測結果疊加
#         axes[3].imshow(img_resized)
#         for i, det in enumerate(filtered_detections):
#             x_min, y_min, x_max, y_max = det['bbox']
#             rect = patches.Rectangle(
#                 (x_min * PATCH_SIZE, y_min * PATCH_SIZE),
#                 (x_max - x_min + 1) * PATCH_SIZE,
#                 (y_max - y_min + 1) * PATCH_SIZE,
#                 linewidth=2, edgecolor='red', facecolor='none'
#             )
#             axes[3].add_patch(rect)
#
#             # 標注分數
#             center_x = (x_min + x_max) * PATCH_SIZE / 2
#             center_y = (y_min + y_max) * PATCH_SIZE / 2
#             axes[3].text(
#                 center_x, center_y,
#                 f'{det["score"]:.2f}',
#                 color='white', fontsize=10,
#                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.7)
#             )
#         axes[3].set_title("檢測結果")
#         axes[3].axis('off')
#
#         # 5. 分數分布
#         axes[4].hist(score_map.flatten(), bins=50, alpha=0.7, edgecolor='black')
#         axes[4].axvline(threshold, color='red', linestyle='--', label=f'閾值={threshold:.3f}')
#         axes[4].axvline(np.mean(score_map), color='green', linestyle='--',
#                        label=f'均值={np.mean(score_map):.3f}')
#         axes[4].set_xlabel('分數')
#         axes[4].set_ylabel('頻率')
#         axes[4].set_title('分數分布')
#         axes[4].legend()
#         axes[4].grid(True, alpha=0.3)
#
#         # 6. 區域統計
#         if filtered_detections:
#             scores_list = [d['score'] for d in filtered_detections]
#             sizes_list = [d['size'] for d in filtered_detections]
#
#             x = range(len(scores_list))
#             axes[5].bar(x, scores_list, alpha=0.7, label='分數')
#             axes[5].set_xlabel('檢測區域')
#             axes[5].set_ylabel('分數', color='blue')
#             axes[5].tick_params(axis='y', labelcolor='blue')
#
#             ax2 = axes[5].twinx()
#             ax2.plot(x, sizes_list, 'ro-', label='大小')
#             ax2.set_ylabel('Patch數量', color='red')
#             ax2.tick_params(axis='y', labelcolor='red')
#
#             axes[5].set_title('檢測區域統計')
#             axes[5].legend(loc='upper left')
#             ax2.legend(loc='upper right')
#         else:
#             axes[5].text(0.5, 0.5, '無檢測結果',
#                         ha='center', va='center', fontsize=12)
#             axes[5].axis('off')
#
#         plt.suptitle(f"測試結果: {os.path.basename(img_path)}", fontsize=14)
#         plt.tight_layout()
#
#         # 保存結果
#         save_path = os.path.join(output_dir, f"result_{os.path.basename(img_path).replace('.jpg', '.png')}")
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#
#         # 記錄檢測結果（轉換numpy類型為Python原生類型）
#         for det in filtered_detections:
#             # 將numpy類型轉換為Python原生類型
#             bbox = det['bbox']
#             all_detections.append({
#                 'image': os.path.basename(img_path),
#                 'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],  # 轉換為Python int
#                 'score': float(det['score']),
#                 'size': int(det['size']),
#                 'compactness': float(det['compactness'])
#             })
#
#         print(f"  檢測到 {len(filtered_detections)} 個區域")
#         for i, det in enumerate(filtered_detections, 1):
#             print(f"    區域{i}: 分數={det['score']:.3f}, 大小={det['size']} patches")
#
#     # 保存檢測結果到JSON
#     results_json = os.path.join(output_dir, "detections.json")
#     with open(results_json, 'w', encoding='utf-8') as f:
#         json.dump(all_detections, f, indent=2, ensure_ascii=False)
#
#     print(f"\n測試完成！結果已保存到 {output_dir}")
#     print(f"檢測結果總計: {len(all_detections)} 個區域")
#
#     # 分析檢測結果
#     analyze_detection_results(all_detections)
#
# def analyze_detection_results(all_detections):
#     """分析檢測結果"""
#     print("\n[3] 檢測結果分析:")
#
#     if not all_detections:
#         print("  沒有檢測到任何區域")
#         return
#
#     # 按圖像分組
#     results_by_image = {}
#     for det in all_detections:
#         img_name = det['image']
#         if img_name not in results_by_image:
#             results_by_image[img_name] = []
#         results_by_image[img_name].append(det)
#
#     print(f"  總共有 {len(results_by_image)} 張圖像有檢測結果")
#
#     # 統計每張圖像的檢測數量
#     print("\n  每張圖像檢測統計:")
#     for img_name, dets in results_by_image.items():
#         avg_score = np.mean([d['score'] for d in dets])
#         avg_size = np.mean([d['size'] for d in dets])
#         print(f"    {img_name}: {len(dets)} 個區域, 平均分數={avg_score:.3f}, 平均大小={avg_size:.1f} patches")
#
#     # 總體統計
#     all_scores = [d['score'] for d in all_detections]
#     all_sizes = [d['size'] for d in all_detections]
#
#     print(f"\n  總體統計:")
#     print(f"    平均分數: {np.mean(all_scores):.3f} ± {np.std(all_scores):.3f}")
#     print(f"    分數範圍: [{np.min(all_scores):.3f}, {np.max(all_scores):.3f}]")
#     print(f"    平均區域大小: {np.mean(all_sizes):.1f} patches")
#     print(f"    最大區域: {np.max(all_sizes)} patches, 最小區域: {np.min(all_sizes)} patches")
#
#     # 建議閾值調整
#     score_percentiles = np.percentile(all_scores, [50, 75, 90, 95])
#     print(f"\n  分數百分位數:")
#     print(f"    50% (中位數): {score_percentiles[0]:.3f}")
#     print(f"    75%: {score_percentiles[1]:.3f}")
#     print(f"    90%: {score_percentiles[2]:.3f}")
#     print(f"    95%: {score_percentiles[3]:.3f}")
#
#     # 根據結果建議調整參數
#     print(f"\n  參數調整建議:")
#     if np.mean(all_scores) > 1.5:
#         print(f"    目前分數偏高，建議增加 ALPHA_WA 或 BETA_BG")
#     elif np.mean(all_scores) < 0.5:
#         print(f"    目前分數偏低，建議減少 ALPHA_WA 或 BETA_BG")
#
#     if np.mean(all_sizes) > 100:
#         print(f"    檢測區域偏大，建議增加閾值或調整區域生長參數")
#     elif np.mean(all_sizes) < 10:
#         print(f"    檢測區域偏小，建議降低閾值")
#
# # ========== 主執行函數 ==========
#
# def main():
#     print("="*60)
#     print("DINOv3 輔助標記系統 - 多區域對比強化版本")
#     print("="*60)
#
#     # 配置路徑
#     labeled_dir = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\DLW\labeled"
#     test_dir = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\DLW\unlabeled"
#     model_path = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\dinov3 weight\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
#     output_dir = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\DLW\enhanced_results"
#
#     # 1. 加載模型
#     print("\n[0] 加載 DINOv3 模型...")
#     model = torch.hub.load("./", 'dinov3_vitb16', source='local', weights=model_path)
#     model.to(DEVICE).eval()
#
#     # 2. 建立特徵池（訓練階段）
#     feature_pool = build_feature_pool(model, labeled_dir)
#
#     # 3. 測試階段
#     test_with_dynamic_scoring(model, test_dir, feature_pool, output_dir)
#
#     print("\n" + "="*60)
#     print("程式執行完成！")
#     print("="*60)
#
# if __name__ == "__main__":
#     main()


import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from glob import glob

# ========== 配置參數 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 16


def get_feature_map(model, img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    new_w, new_h = (w // PATCH_SIZE) * PATCH_SIZE, (h // PATCH_SIZE) * PATCH_SIZE

    transform = T.Compose([
        T.Resize((new_h, new_w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.get_intermediate_layers(input_tensor, n=1)[0]

    # 移除 [CLS] token
    if features.shape[1] > (new_w // PATCH_SIZE) * (new_h // PATCH_SIZE):
        features = features[:, 1:, :]

    return features[0], (new_w, new_h), img.resize((new_w, new_h))


def run_simple_heatmap():
    # 路徑配置
    labeled_dir = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\DLW\labeled"
    test_dir = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\DLW\unlabeled"
    model_path = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\dinov3 weight\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    output_dir = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\DLW\simple_heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加載模型
    print("正在加載模型...")
    model = torch.hub.load("./", 'dinov3_vitb16', source='local', weights=model_path).to(DEVICE).eval()

    # 2. 建立 WP 特徵池
    print("正在從已標註圖片提取 WP 特徵...")
    wp_features = []
    json_files = glob(os.path.join(labeled_dir, "*.json"))

    for j_path in json_files:
        with open(j_path, 'r') as f:
            data = json.load(f)
        img_path = os.path.splitext(j_path)[0] + ".jpg"

        feats, (tw, th), _ = get_feature_map(model, img_path)
        grid_w = tw // PATCH_SIZE

        for shape in data['shapes']:
            if shape['label'] == 'wp':
                # 簡單提取標註點對應的 patch (取中心點)
                pts = np.array(shape['points'])
                sx, sy = tw / data['imageWidth'], th / data['imageHeight']
                for p in pts:
                    gx, gy = int((p[0] * sx) // PATCH_SIZE), int((p[1] * sy) // PATCH_SIZE)
                    idx = gy * grid_w + gx
                    if idx < feats.shape[0]:
                        wp_features.append(feats[idx].cpu())

    # 計算平均特徵向量並歸一化
    if not wp_features:
        print("錯誤：未找到任何 wp 標註！")
        return
    target_vec = torch.stack(wp_features).mean(dim=0).to(DEVICE)
    target_vec = F.normalize(target_vec, p=2, dim=0)

    # 3. 處理未標註圖片
    print(f"開始處理測試圖片，結果將存至: {output_dir}")
    test_images = glob(os.path.join(test_dir, "*.jpg"))

    for img_path in test_images:
        feats, (tw, th), img_resized = get_feature_map(model, img_path)

        # 計算餘弦相似度
        feats_norm = F.normalize(feats, p=2, dim=1)
        similarity = torch.matmul(feats_norm, target_vec)  # [grid_h * grid_w]

        # 重塑為二維地圖
        grid_w, grid_h = tw // PATCH_SIZE, th // PATCH_SIZE
        sim_map = similarity.view(grid_h, grid_w).cpu().numpy()

        # 插值縮放回原圖大小
        sim_map_resized = Image.fromarray(sim_map).resize((tw, th), resample=Image.BILINEAR)
        sim_map_resized = np.array(sim_map_resized)

        # 輸出圖片
        plt.figure(figsize=(10, 8))
        plt.imshow(img_resized)
        # 使用 jet 顏色地圖，alpha 設定透明度疊加在原圖上
        # vmin/vmax 控制相似度顯示範圍，通常相似度在 0.4 以上才具備參考價值
        im = plt.imshow(sim_map_resized, cmap='jet', alpha=0.5, vmin=0.3, vmax=sim_map.max())
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"Heatmap: {os.path.basename(img_path)}")
        plt.axis('off')

        save_name = os.path.join(output_dir, "heatmap_" + os.path.basename(img_path))
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"已生成: {os.path.basename(img_path)}")


if __name__ == "__main__":
    run_simple_heatmap()