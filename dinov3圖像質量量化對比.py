# # # import os
# # # import torch
# # # import torchvision.transforms as transforms
# # # import torch.nn.functional as F
# # # import cv2
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from transformers.image_utils import load_image
# # #
# # #
# # # # ==========================================
# # # # 1. 核心模型運算函數
# # # # ==========================================
# # #
# # # def get_dino_features_and_attention(model, img_path, max_size=512):
# # #     """提取 Patch Tokens 與 Attention，並處理尺寸"""
# # #     image = load_image(img_path)
# # #     transform = transforms.Compose([transforms.ToTensor()])
# # #     img_tensor = transform(image).unsqueeze(0).cuda()
# # #
# # #     # 尺寸計算與修正 (確保 16 的倍數)
# # #     _, _, h, w = img_tensor.shape
# # #     ratio = float(max_size) / float(max(h, w))
# # #     new_h = int(round(h * ratio / 16.0)) * 16
# # #     new_w = int(round(w * ratio / 16.0)) * 16
# # #
# # #     img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
# # #
# # #     with torch.no_grad():
# # #         # 1. 提取 Patch 特徵
# # #         features_dict = model.forward_features(img_tensor)
# # #         patches = features_dict["x_norm_patchtokens"]  # (1, N, C)
# # #
# # #         # 2. 計算 Attention (用於視覺化)
# # #         last_block = model.blocks[-1]
# # #         H_patch, W_patch = new_h // 16, new_w // 16
# # #         x, _ = model.prepare_tokens_with_masks(img_tensor)
# # #         for blk in model.blocks[:-1]:
# # #             x = blk(x)
# # #
# # #         B, N, C = x.shape
# # #         num_heads = last_block.attn.num_heads
# # #         qkv = last_block.attn.qkv(last_block.norm1(x)).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1,
# # #                                                                                                            4)
# # #         q, k = qkv[0], qkv[1]
# # #         cls_q = q[:, :, 0:1, :]
# # #         attn = (cls_q @ k.transpose(-2, -1)) * last_block.attn.scale
# # #         attn = attn.softmax(dim=-1)
# # #
# # #     return patches, attn, H_patch, W_patch, img_tensor
# # #
# # #
# # # def calculate_richness(patches):
# # #     """計算特徵豐富度 (Richness): 代表圖像包含多少可區分的細節"""
# # #     return torch.std(patches, dim=1).mean().item()
# # #
# # #
# # # def analyze_intra_similarity(patches):
# # #     """計算內部相似度 (Similarity): 代表圖像內容是否過於雷同/模糊"""
# # #     p = F.normalize(patches, p=2, dim=-1)
# # #     sim_matrix = torch.matmul(p, p.transpose(-2, -1))
# # #     n = sim_matrix.shape[1]
# # #     triu_indices = torch.triu_indices(n, n, offset=1)
# # #     avg_sim = sim_matrix[0, triu_indices[0], triu_indices[1]].mean().item()
# # #     return avg_sim
# # #
# # #
# # # # ==========================================
# # # # 2. 批次處理與對比分析
# # # # ==========================================
# # #
# # # def run_diagnostic_report(model, folder_a, folder_b, num_samples=10):
# # #     results = {"A": {"richness": [], "similarity": []}, "B": {"richness": [], "similarity": []}}
# # #
# # #     print("開始執行廠商圖像質量診斷...")
# # #     for label, folder in [("A", folder_a), ("B", folder_b)]:
# # #         files = [os.path.join(folder, f) for f in os.listdir(folder)
# # #                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
# # #
# # #         if not files:
# # #             print(f"警告: 資料夾 {label} 中找不到圖像檔案。")
# # #             continue
# # #
# # #         for f in files:
# # #             patches, _, _, _, _ = get_dino_features_and_attention(model, f)
# # #             results[label]["richness"].append(calculate_richness(patches))
# # #             results[label]["similarity"].append(analyze_intra_similarity(patches))
# # #
# # #     # 數據統計
# # #     report = {}
# # #     for L in ["A", "B"]:
# # #         report[L] = {
# # #             "rich_m": np.mean(results[L]["richness"]),
# # #             "sim_m": np.mean(results[L]["similarity"])
# # #         }
# # #
# # #     # --- 打印報告 ---
# # #     print("\n" + "=" * 50)
# # #     print("      DINOv3 圖像質量量化診斷報告 (深度版)")
# # #     print("=" * 50)
# # #     print(f"指標             廠商 A (YOLO有效)    廠商 B (YOLO失效)")
# # #     print(f"--------------------------------------------------")
# # #     print(f"特徵豐富度 (↑)    {report['A']['rich_m']:.6f}           {report['B']['rich_m']:.6f}")
# # #     print(f"內部相似度 (↓)    {report['A']['sim_m']:.6f}           {report['B']['sim_m']:.6f}")
# # #     print("-" * 50)
# # #
# # #     # 診斷逻辑
# # #     rich_diff = (report['A']['rich_m'] - report['B']['rich_m']) / report['B']['rich_m'] * 100
# # #     sim_diff = (report['B']['sim_m'] - report['A']['sim_m']) / report['A']['sim_m'] * 100
# # #
# # #     print(f"【診斷分析】")
# # #     if rich_diff > 5:
# # #         print(f"* 廠商 B 的特徵豐富度低於 A 約 {rich_diff:.1f}%，代表圖像細節流失。")
# # #     if sim_diff > 0:
# # #         print(f"* 廠商 B 的內容相似度比 A 高出 {sim_diff:.1f}%，說明 B 的圖像更加『模糊/雷同』。")
# # #
# # #     if report['B']['sim_m'] > 0.6:  # 經驗閾值
# # #         print(">> 嚴重警報: 廠商 B 相似度過高，焊點與背景特徵已發生高度重疊，YOLO 無法區分邊界。")
# # #
# # #     print("=" * 50)
# # #
# # #     # 執行最後兩張圖的視覺化對比
# # #     files_a = [os.path.join(folder_a, f) for f in os.listdir(folder_a) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
# # #     files_b = [os.path.join(folder_b, f) for f in os.listdir(folder_b) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
# # #     if files_a and files_b:
# # #         visualize_side_by_side(model, files_a[0], files_b[0])
# # #
# # #
# # # def visualize_side_by_side(model, path_a, path_b):
# # #     """生成對比圖"""
# # #     plt.figure(figsize=(16, 10))
# # #     for i, (path, label) in enumerate([(path_a, "Vendor A"), (path_b, "Vendor B")]):
# # #         _, attn, H, W, img_t = get_dino_features_and_attention(model, path)
# # #         mask = attn[0, :, 0, -H * W:].reshape(attn.shape[1], H, W)
# # #         avg_mask = torch.mean(mask, dim=0).cpu().numpy()
# # #
# # #         # 原圖 (Clip 防止警告)
# # #         plt.subplot(2, 2, i + 1)
# # #         img_np = img_t.squeeze().cpu().numpy().transpose(1, 2, 0)
# # #         plt.imshow(np.clip(img_np, 0, 1))
# # #         plt.title(f"{label} - Original View")
# # #         plt.axis('off')
# # #
# # #         # Attention Map
# # #         plt.subplot(2, 2, i + 3)
# # #         plt.imshow(avg_mask, cmap='magma')
# # #         plt.title(f"{label} - Feature Focus (Attention)")
# # #         plt.axis('off')
# # #
# # #     plt.tight_layout()
# # #     plt.show()
# # #
# # #
# # # # ==========================================
# # # # 3. 主程式入口
# # # # ==========================================
# # #
# # # if __name__ == "__main__":
# # #     # --- 路徑設定 ---
# # #     FOLDER_A = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\plateau\hg"
# # #     FOLDER_B = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\plateau\dz"
# # #     MODEL_PATH = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\dinov3 weight\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
# # #
# # #     # --- 模型加載 ---
# # #     print("正在初始化 DINOv3 模型...")
# # #     # 假設代碼在 dinov3-main 目錄下執行
# # #     model = torch.hub.load("./", 'dinov3_vitb16', source='local', weights=MODEL_PATH)
# # #     model.cuda().eval()
# # #
# # #     # --- 執行報告 ---
# # #     run_diagnostic_report(model, FOLDER_A, FOLDER_B, num_samples=10)
# #
# #
# # import os
# # import torch
# # import torchvision.transforms as transforms
# # import torch.nn.functional as F
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from transformers.image_utils import load_image
# #
# #
# # # ==========================================
# # # 1. 核心邏輯：自動尋找特徵活躍區域
# # # ==========================================
# #
# # def get_unsupervised_feature_scores(model, img_path, max_size=1024):
# #     """
# #     無需標註，自動在大圖中尋找特徵最活躍的區域並評分
# #     """
# #     image = load_image(img_path)
# #     transform = transforms.Compose([transforms.ToTensor()])
# #     img_tensor = transform(image).unsqueeze(0).cuda()
# #
# #     # 縮放至較大尺寸以保留局部細節 (1024 是平衡點)
# #     _, _, h, w = img_tensor.shape
# #     ratio = float(max_size) / float(max(h, w))
# #     new_h = int(round(h * ratio / 16.0)) * 16
# #     new_w = int(round(w * ratio / 16.0)) * 16
# #     img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
# #
# #     with torch.no_grad():
# #         # 提取 Patch 特徵 (1, N, C)
# #         features_dict = model.forward_features(img_tensor)
# #         patches = features_dict["x_norm_patchtokens"]
# #
# #         # 重新排列成 2D 特徵圖 (B, C, H_p, W_p)
# #         H_p, W_p = new_h // 16, new_w // 16
# #         feature_grid = patches.reshape(1, H_p, W_p, -1).permute(0, 3, 1, 2)
# #
# #         # --- 計算特徵活躍度矩陣 ---
# #         # 我們計算每個 patch 位置的特徵標準差，代表該處的資訊豐富度
# #         std_map = torch.std(feature_grid, dim=1).squeeze(0)  # (H_p, W_p)
# #
# #         # --- 指標計算 ---
# #         # 1. 全圖平均背景能量
# #         avg_energy = std_map.mean().item()
# #
# #         # 2. 局部峰值能量 (取前 5% 最活躍的 patches)
# #         # 這些 patches 通常對應焊點、邊緣等關鍵結構
# #         top_k = max(1, int(H_p * W_p * 0.05))
# #         top_values, _ = torch.topk(std_map.flatten(), k=top_k)
# #         peak_energy = top_values.mean().item()
# #
# #         # 3. 特徵信噪比 (SNR-like): 峰值 / 背景
# #         # 數值越高，代表目標與背景越容易區分
# #         feature_contrast = peak_energy / (avg_energy + 1e-6)
# #
# #     return {
# #         "peak_energy": peak_energy,
# #         "feature_contrast": feature_contrast,
# #         "std_map": std_map.cpu().numpy(),
# #         "img_display": img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
# #     }
# #
# #
# # # ==========================================
# # # 2. 批次診斷與視覺化
# # # ==========================================
# #
# # def run_unsupervised_comparison(model, folder_a, folder_b, num_samples=5):
# #     results = {"A": {"peaks": [], "contrast": []}, "B": {"peaks": [], "contrast": []}}
# #
# #     for label, folder in [("A", folder_a), ("B", folder_b)]:
# #         files = [os.path.join(folder, f) for f in os.listdir(folder)
# #                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
# #
# #         for f in files:
# #             res = get_unsupervised_feature_scores(model, f)
# #             results[label]["peaks"].append(res["peak_energy"])
# #             results[label]["contrast"].append(res["feature_contrast"])
# #
# #     # 計算平均值
# #     m_peak_a, m_con_a = np.mean(results["A"]["peaks"]), np.mean(results["A"]["contrast"])
# #     m_peak_b, m_con_b = np.mean(results["B"]["peaks"]), np.mean(results["B"]["contrast"])
# #
# #     print("\n" + "=" * 55)
# #     print("      DINOv3 無監督特徵峰值診斷 (免標註版本)")
# #     print("=" * 55)
# #     print(f"指標                廠商 A (YOLO有效)    廠商 B (YOLO失效)")
# #     print("-" * 55)
# #     print(f"目標區域特徵能量 (↑)  {m_peak_a:.6f}           {m_peak_b:.6f}")
# #     print(f"特徵對比度 (SNR ↑)    {m_con_a:.4f}             {m_con_b:.4f}")
# #     print("-" * 55)
# #
# #     diff = (m_con_a - m_con_b) / m_con_b * 100
# #     print(f"【診斷結論】")
# #     print(f"* 廠商 A 的目標/背景辨識力比 B 高出 {diff:.2f}%。")
# #     if m_con_b < 1.3:  # 經驗門檻
# #         print(">> 嚴重警告: 廠商 B 的特徵對比度過低，目標已淹沒在背景中。")
# #     print("=" * 55)
# #
# #     # 顯示最後一組對比
# #     visualize_std_maps(model, folder_a, folder_b)
# #
# #
# # def visualize_std_maps(model, folder_a, folder_b):
# #     file_a = [os.path.join(folder_a, f) for f in os.listdir(folder_a)][0]
# #     file_b = [os.path.join(folder_b, f) for f in os.listdir(folder_b)][0]
# #
# #     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# #     for i, (f, label) in enumerate([(file_a, "Vendor A"), (file_b, "Vendor B")]):
# #         res = get_unsupervised_feature_scores(model, f)
# #
# #         # 顯示縮放後的原圖
# #         axes[0, i].imshow(np.clip(res["img_display"], 0, 1))
# #         axes[0, i].set_title(f"{label} - Original")
# #         axes[0, i].axis('off')
# #
# #         # 顯示特徵活躍度地圖 (Std Map)
# #         # 越亮的地方代表 DINOv3 認為「這裡有東西」
# #         im = axes[1, i].imshow(res["std_map"], cmap='viridis')
# #         axes[1, i].set_title(f"{label} - Feature Activation (Std Map)")
# #         plt.colorbar(im, ax=axes[1, i])
# #         axes[1, i].axis('off')
# #
# #     plt.tight_layout()
# #     plt.show()
# #
# #
# # # ==========================================
# # # 3. 主程式
# # # ==========================================
# #
# # if __name__ == "__main__":
# #     MODEL_PATH = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\dinov3 weight\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
# #     FOLDER_A = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\plateau\hg"
# #     FOLDER_B = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\plateau\dz"
# #
# #     print("正在加載模型...")
# #     model = torch.hub.load("./", 'dinov3_vitb16', source='local', weights=MODEL_PATH).cuda().eval()
# #
# #     run_unsupervised_comparison(model, FOLDER_A, FOLDER_B, num_samples=5)
#
# import os
# import torch
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from transformers.image_utils import load_image
#
#
# # ==========================================
# # 1. 核心邏輯：提取單張圖的能量統計
# # ==========================================
#
# def get_image_energy_stats(model, img_path, max_size=1024):
#     image = load_image(img_path)
#     transform = transforms.Compose([transforms.ToTensor()])
#     img_tensor = transform(image).unsqueeze(0).cuda()
#
#     _, _, h, w = img_tensor.shape
#     ratio = float(max_size) / float(max(h, w))
#     new_h, new_w = int(round(h * ratio / 16.0)) * 16, int(round(w * ratio / 16.0)) * 16
#     img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
#
#     with torch.no_grad():
#         features_dict = model.forward_features(img_tensor)
#         patches = features_dict["x_norm_patchtokens"]
#         H_p, W_p = new_h // 16, new_w // 16
#         feature_grid = patches.reshape(1, H_p, W_p, -1).permute(0, 3, 1, 2)
#
#         # 特徵活躍度地圖
#         std_map = torch.std(feature_grid, dim=1).squeeze(0)
#
#         # 背景能量 (Background Base)
#         bg_energy = std_map.mean().item()
#
#         # 頂尖峰值 (Top 1%)
#         top_k = max(1, int(H_p * W_p * 0.01))
#         top_values, _ = torch.topk(std_map.flatten(), k=top_k)
#         peak_energy = top_values.mean().item()
#
#     return peak_energy, bg_energy
#
#
# # ==========================================
# # 2. 穩定性分析 (多樣本)
# # ==========================================
#
# def run_stability_analysis(model, folder_a, folder_b, num_samples=10):
#     stats = {"A": {"peaks": [], "bgs": []}, "B": {"peaks": [], "bgs": []}}
#
#     print(f"正在分析穩定度 (取樣數: {num_samples})...")
#     for label, folder in [("A", folder_a), ("B", folder_b)]:
#         files = [os.path.join(folder, f) for f in os.listdir(folder)
#                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
#         for f in files:
#             p, b = get_image_energy_stats(model, f)
#             stats[label]["peaks"].append(p)
#             stats[label]["bgs"].append(b)
#
#     # 計算統計量
#     def get_summary(data_list):
#         return {
#             "mean": np.mean(data_list),
#             "std": np.std(data_list),
#             "cv": np.std(data_list) / (np.mean(data_list) + 1e-6)  # 變異係數 (Coefficient of Variation)
#         }
#
#     sum_a = get_summary(stats["A"]["peaks"])
#     sum_b = get_summary(stats["B"]["peaks"])
#
#     print("\n" + "=" * 65)
#     print("      DINOv3 跨樣本成像穩定度分析 (Stability Report)")
#     print("=" * 65)
#     print(f"指標                 廠商 A (有效)          廠商 B (失效)")
#     print("-" * 65)
#     print(f"峰值能量平均 (Mean)   {sum_a['mean']:.6f}           {sum_b['mean']:.6f}")
#     print(f"能量波動標準差 (Std)  {sum_a['std']:.6f}           {sum_b['std']:.6f}")
#     print(f"變異係數 (CV % ↓)    {sum_a['cv'] * 100:.2f}%               {sum_b['cv'] * 100:.2f}%")
#     print("-" * 65)
#
#     print("【深度診斷】")
#     if sum_b['cv'] > sum_a['cv'] * 1.5:
#         print(f"* 警告：廠商 B 的成像『穩定性』遠低於 A。")
#         print("  這代表 B 的光源或曝光一直在變動，導致 AI 無法記住焊點特徵。")
#     elif abs(sum_a['mean'] - sum_b['mean']) < 0.02:
#         print("* 兩者平均能量極度接近。請檢查兩廠的『標註品質一致性』。")
#         print("  物理特徵既然相同，失效原因極可能是 B 廠的數據標註框不準確。")
#     print("=" * 65)
#
#     # 繪製波動折線圖
#     plt.figure(figsize=(10, 5))
#     plt.plot(stats["A"]["peaks"], 'o-', label='Vendor A (Stable)')
#     plt.plot(stats["B"]["peaks"], 's--', label='Vendor B (Fluctuating?)')
#     plt.title("Feature Peak Stability Comparison (Top 1%)")
#     plt.xlabel("Sample Index")
#     plt.ylabel("Peak Feature Energy")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
#
#
# # ==========================================
# # 3. 主程式執行
# # ==========================================
#
# if __name__ == "__main__":
#     MODEL_PATH = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\dinov3 weight\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
#     FOLDER_A = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\plateau\dz"
#     FOLDER_B = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\plateau\hg"
#
#     print("正在啟動 DINOv3 穩定度分析儀...")
#     model = torch.hub.load("./", 'dinov3_vitb16', source='local', weights=MODEL_PATH).cuda().eval()
#     run_stability_analysis(model, FOLDER_A, FOLDER_B, num_samples=10)


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def analyze_edge_gradients(folder_a, folder_b, num_samples=5):
    """
    計算圖像的邊緣梯度強度，量化人眼感知的『銳利度』與『對比度』
    """
    results = {"A": [], "B": []}

    def get_gradient_magnitude(img_path):
        # 1. 讀取灰階圖
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # 2. 進行輕微的高斯模糊，濾除純數位噪點 (避免噪點虛報梯度)
        img_blurred = cv2.GaussianBlur(img, (3, 3), 0)

        # 3. 計算 Sobel 梯度 (x方向與y方向)
        grad_x = cv2.Sobel(img_blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_blurred, cv2.CV_64F, 0, 1, ksize=3)

        # 4. 計算梯度幅值 (Magnitude)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 5. 返回平均梯度強度 (代表圖像邊緣的清晰程度)
        return np.mean(grad_mag), grad_mag

    # 執行批次分析
    print("正在分析圖像邊緣梯度...")
    for label, folder in [("A", folder_a), ("B", folder_b)]:
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
        for f in files:
            mag, _ = get_gradient_magnitude(f)
            if mag is not None:
                results[label].append(mag)

    # 統計結果
    mean_a = np.mean(results["A"])
    mean_b = np.mean(results["B"])

    print("\n" + "=" * 50)
    print("      圖像邊緣銳利度 (Edge Sharpness) 診斷報告")
    print("=" * 50)
    print(f"指標                 廠商 A (易區分)      廠商 B (難區分)")
    print("-" * 50)
    print(f"平均梯度強度 (↑)      {mean_a:.4f}             {mean_b:.4f}")
    print("-" * 50)

    diff = (mean_a - mean_b) / mean_b * 100
    print(f"【視覺對比診斷】")
    print(f"* 廠商 A 的邊緣強度比 B 高出 {diff:.2f}%。")
    if diff > 15:
        print(">> 診斷：廠商 B 的打光導致焊點與背景發生『語義融合』。")
        print("   這解釋了為何肉眼與 YOLO 均無法識別，因為邊界物理特徵太弱。")
    print("=" * 50)

    # 可視化最後一組對比
    visualize_gradients(folder_a, folder_b, get_gradient_magnitude)


def visualize_gradients(folder_a, folder_b, grad_fn):
    file_a = [os.path.join(folder_a, f) for f in os.listdir(folder_a)][0]
    file_b = [os.path.join(folder_b, f) for f in os.listdir(folder_b)][0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (f, label) in enumerate([(file_a, "Vendor A"), (file_b, "Vendor B")]):
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, grad_map = grad_fn(f)

        # 原圖
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{label} Original")
        axes[0, i].axis('off')

        # 梯度圖 (使用 'inferno' 顏色，越亮代表邊緣越銳利)
        # 我們將顯示範圍限制在 [0, 50] 左右以便觀察微小差異
        im = axes[1, i].imshow(grad_map, cmap='inferno', vmin=0, vmax=50)
        axes[1, i].set_title(f"{label} Edge Gradients (Sharpness)")
        plt.colorbar(im, ax=axes[1, i])
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    FOLDER_A = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\plateau\hg"
    FOLDER_B = r"D:\PycharmProjects\other\dinov3-main\dinov3-main\images\plateau\dz"
    analyze_edge_gradients(FOLDER_A, FOLDER_B)