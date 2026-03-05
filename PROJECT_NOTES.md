# 新工站圖像評估系統 — 開發筆記與討論紀錄

> 最後更新：2026-03-05
> 分支：`claude/implement-readme-requirements-mWCoR`

---

## 目錄

1. [專案背景](#1-專案背景)
2. [系統架構總覽](#2-系統架構總覽)
3. [Feature 1 — 評估結果頁面](#3-feature-1--評估結果頁面)
4. [Feature 2 — AI 分類頁面](#4-feature-2--ai-分類頁面)
5. [三項評估指標詳解](#5-三項評估指標詳解)
6. [DINOv3 模型運作原理](#6-dinov3-模型運作原理)
7. [ROI 提取方法討論](#7-roi-提取方法討論)
8. [後端 API 說明](#8-後端-api-說明)
9. [前端 UI 設計細節](#9-前端-ui-設計細節)
10. [檔案結構](#10-檔案結構)
11. [啟動方式](#11-啟動方式)
12. [待辦 / 後續方向](#12-待辦--後續方向)

---

## 1. 專案背景

### 核心問題

工廠導入新工站時，在**花費大量人力標注、訓練模型之前**，需要先評估：

| 問題 | 對應指標 |
|------|----------|
| OK 樣本的外觀變化有多大？是否需要更多數據？ | 數據複雜度 (DC) |
| 缺陷在特徵空間中是否容易與正常樣本區分？ | OK/NG 分離度 (Sep) |
| 圖像本身的信息量夠不夠讓模型學習？ | 特徵豐富度 (FR) |

### 工具定位

- **不需要標注缺陷位置**，只需 OK / NG 圖片資料夾
- 用 DINOv3（自監督 Vision Transformer）提取特徵
- 以量化指標取代人工主觀判斷，給出「可行 / 需補數據」的決策依據

---

## 2. 系統架構總覽

```
瀏覽器 (index.html)
    │
    │  HTTP / FormData / JSON
    ▼
Flask 後端 (app.py)
    │
    ├── DINOv3 / DINOv2 特徵提取
    │       └── fallback: OpenCV HOG 特徵
    ├── scikit-learn (PCA, cosine_similarity)
    └── matplotlib (圖表生成 → base64 回傳)
```

### 技術選型理由

| 層次 | 選擇 | 原因 |
|------|------|------|
| 後端框架 | Flask | 輕量、Python 原生，不需要額外配置 |
| 特徵模型 | DINOv3 ViT-B/16 | 自監督、無需標注、工業圖像泛化好 |
| 相似度計算 | Cosine Similarity | 特徵向量長度不一致時仍穩定 |
| 可視化 | matplotlib Agg | 無 display server 環境可用 |
| 前端 | 純 HTML/CSS/JS | 無框架依賴，方便移植與截圖 |

---

## 3. Feature 1 — 評估結果頁面

### 使用流程

```
1. 上傳 OK 圖片（必填）  ＋  NG 圖片（可選）
        ↓
2. 點擊「開始評估」
        ↓
3. 後端提取 DINOv3 特徵（每張最多取前 20 張）
        ↓
4. 計算三項指標 → 生成 matplotlib 圖表
        ↓
5. 前端顯示：
   - 指標數字卡（數據複雜度 / 特徵豐富度 / OK/NG 分離度）
   - 評估圖表（條形圖 + 餅圖 + 文字摘要）
   - 每張圖的「原圖 ｜ 注意力熱力圖」對比卡
   - 顏色編碼建議（綠=好 / 黃=注意 / 紅=警告）
```

### UI 元件清單

| 元件 | 說明 |
|------|------|
| OK 上傳區 | 拖曳或點擊，支援多選檔案 / 資料夾 (`multiple`) |
| NG 上傳區 | 同上，可選，上傳後才顯示分離度指標 |
| 指標卡 ×3 | 大數字顯示分數（0-100），顏色表示好壞 |
| matplotlib 圖表 | 條形圖（指標）+ 餅圖（OK/NG 比例）+ 文字摘要 |
| 圖像網格 | 每張圖：左原圖、右熱力圖，角標 OK/NG badge |
| 建議清單 | 根據分數區間自動產生文字建議 |

### 建議邏輯（硬編碼規則）

```
數據複雜度 (DC)：
  DC < 25  → ✅ OK 樣本外觀相似度高，數據一致性佳
  DC > 65  → ⚠️ 多樣性高，建議增加數據
  其他      → 💡 多樣性中等，質量正常

特徵豐富度 (FR)：
  FR > 60  → ✅ 圖像信息量充足
  FR ≤ 60  → ⚠️ 可能影響模型學習效果

OK/NG 分離度 (Sep)（僅當有 NG 圖時）：
  Sep > 60 → ✅ 缺陷容易被識別
  Sep > 30 → 💡 有一定挑戰
  Sep ≤ 30 → ⚠️ 特徵重疊嚴重，難以區分
```

---

## 4. Feature 2 — AI 分類頁面

### 使用流程（三步驟）

```
Step 1：設定模板 ROI
  └── 上傳模板圖片（一張或多張）
  └── 在 Canvas 上拖曳滑鼠框選區域
  └── 輸入 ROI 名稱（例如：焊點、元件A、背景）
  └── 可在同一張或不同圖片上定義多個 ROI
  └── 點擊「確認模板設定」→ 後端提取 ROI 特徵向量

Step 2：上傳目標圖片
  └── 上傳要分類的圖片（資料夾或多選）
  └── 點擊「開始 AI 分類」
  └── 後端對每張目標圖提取特徵，與各 ROI 做 cosine similarity
  └── 每張圖分配到相似度最高的類別

Step 3：查看結果
  └── 依類別顯示分組結果（附相似度分數）
  └── 點擊「下載 ZIP」→ 各類別分資料夾打包
```

### Canvas ROI 繪製機制

```javascript
// 滑鼠事件流程
mousedown → 記錄起點 (startX, startY)
mousemove → 即時繪製虛線矩形（視覺 feedback）
mouseup   → 彈出名稱輸入框 → confirmROI() → 存入 rois[]

// ROI 座標轉換（重要！）
// Canvas 顯示時會縮放圖片，ROI 座標需換算回原始圖片座標
scale = min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1)
roi.x = canvas_x / scale   // 送後端的是原始圖片座標
roi.y = canvas_y / scale
```

### ROI 資料結構

```javascript
// 前端儲存
{
  image_index: 0,        // 第幾張模板圖
  x: 150, y: 200,        // 原始圖片座標（像素）
  w: 300, h: 180,        // 寬高（像素）
  name: "焊點區域",       // 分類名稱
  color: "#f72585"       // 顯示顏色（自動輪替 7 色）
}

// 後端接收（JSON array 隨 FormData 傳遞）
rois = json.loads(request.form.get('rois', '[]'))
```

### 分類邏輯

```python
# 後端 match_images()
for target_image in target_files:
    query_feats = extract_features(target_image)

    # 對每個模板類別計算 cosine similarity
    sims = {cat: cosine_similarity(query_feats, tmpl_feats[cat])
            for cat in categories}

    best_cat = max(sims, key=sims.get)
    results[best_cat].append({filename, similarity, image_b64})

# 每個類別內部按相似度降序排列
results[cat].sort(key=lambda x: x['similarity'], reverse=True)
```

### ZIP 下載結構

```
ai_sorting_results.zip
├── 焊點區域/
│   ├── img_001.jpg
│   ├── img_007.jpg
│   └── ...
├── 元件A/
│   ├── img_003.jpg
│   └── ...
└── 背景/
    └── ...
```

---

## 5. 三項評估指標詳解

### 5.1 數據複雜度 (Data Complexity, DC)

**概念**：OK 樣本在特徵空間中的散布程度。

```python
ok_arr    = np.array(ok_features)      # shape: [N, D]
centroid  = ok_arr.mean(axis=0)        # 特徵中心
distances = np.linalg.norm(ok_arr - centroid, axis=1)
DC        = min(100, distances.mean() * 100)
```

**解讀**：
- 低 DC（< 25）→ OK 樣本長得都很像 → 模型容易學
- 高 DC（> 65）→ OK 樣本外觀差異大 → 需要更多數據

---

### 5.2 特徵豐富度 (Feature Richness, FR)

**概念**：用 PCA 的熵值衡量圖像攜帶的信息量。

```python
pca = PCA(n_components=min(20, n_samples-1))
pca.fit(ok_arr)
evr   = pca.explained_variance_ratio_   # 各主成分解釋比例
evr_n = evr / evr.sum()
# 熵越高 → 信息分散在多個維度 → 圖像更豐富
FR = (-np.sum(evr_n * np.log(evr_n + 1e-8)) / np.log(n_components)) * 100
```

**解讀**：
- 低 FR → 圖像過於均一（可能過曝、低對比）
- 高 FR → 圖像細節豐富，有利特徵提取

---

### 5.3 OK/NG 分離度 (Separation, Sep)

**概念**：類 Fisher 判別比，衡量 OK 與 NG 的特徵距離相對於各自內部散布。

```python
ok_centroid = ok_arr.mean(axis=0)
ng_centroid = ng_arr.mean(axis=0)

inter_dist = np.linalg.norm(ok_centroid - ng_centroid)  # 類間距離
ok_intra   = np.linalg.norm(ok_arr - ok_centroid, axis=1).mean()
ng_intra   = np.linalg.norm(ng_arr - ng_centroid, axis=1).mean()

Sep = min(100, inter_dist / (ok_intra + ng_intra + 1e-8) * 30)
```

**解讀**：
- 高 Sep（> 60）→ 缺陷特徵與正常特徵距離遠 → 容易區分
- 低 Sep（< 30）→ 特徵重疊 → 該工站缺陷模式可能很微妙，需更多數據或更好的特徵

---

## 6. DINOv3 模型運作原理

### 架構

```
輸入圖片 (H × W × 3)
    │
    ▼ Patch Embedding（16×16 pixels / patch）
Patch Tokens [N_patches, 768]   +   CLS Token
    │
    ▼ Transformer Layers × 12
Output:
  ├── CLS Token    → 圖像全局特徵（用於分類、相似度）
  └── Patch Tokens → 每個 16×16 區域的局部特徵（用於熱力圖）
```

### 在本系統中的使用方式

| 用途 | 使用的輸出 | 對應函數 |
|------|-----------|----------|
| 圖像相似度比較 | CLS Token（全局） | `extract_features()` |
| 熱力圖生成 | Patch Token 的 L2 norm | `get_attention_map()` |
| ROI 特徵 | 指定 patch 的平均特徵 | `get_region_feature()` in test4.py |

### 模型載入優先順序（app.py）

```
1. 本地 dinov3 weight 資料夾（有 .pth 檔案時）
2. torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
3. OpenCV HOG-like 特徵（無 GPU / 無網路的 fallback）
```

---

## 7. ROI 提取方法討論

> 這是系統中「圖像前處理」步驟的關鍵分歧點。

### 方法一：人工框定（Manual ROI）

**現狀**：test4.py 目前採用此方式（硬編碼座標）

```python
roi = (150, 200, 300, 380)  # x1, y1, x2, y2
```

| 優點 | 缺點 | 適用場景 |
|------|------|----------|
| 零依賴、速度最快 | 圖片有位移就失效 | 固定治具、固定相機 |
| 最穩定、可重現 | 每個工站需重新設定 | PCB、固定焦距場景 |

**Web UI 的實作**：已在 AI 分類頁面的 Canvas 中實現人工框定功能（可視化、可互動）。

---

### 方法二：開集目標檢測器（Open-set Detector）

#### 2a. Grounding DINO（最推薦）

```python
from groundingdino.util.inference import load_model, predict

model = load_model(config_path, weights_path)
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption="焊點",       # 用文字描述目標！
    box_threshold=0.35,
    text_threshold=0.25,
)
# 輸出: bounding boxes → 直接當 ROI 使用
```

- Zero-shot，不需針對工站訓練
- 可結合 **SAM（Segment Anything）** 生成精細 mask

#### 2b. YOLO-World

```python
from ultralytics import YOLOWorld
model = YOLOWorld('yolov8s-world.pt')
model.set_classes(["solder joint", "component pin", "connector"])
results = model.predict(image, conf=0.3)
```

- 比 Grounding DINO 快很多（適合產線即時場景）
- 類別描述不如 Grounding DINO 靈活

#### 2c. OWL-ViT（Google）

```python
from transformers import OwlViTProcessor, OwlViTForObjectDetection
# zero-shot，純 Transformer 架構
```

---

### 方法三：DINOv3 Attention 自動定位

> **不需要任何額外模型**，直接利用現有 DINOv3 的能力。

```python
# 提取最後一層的 CLS → Patch attention
attention = model.get_last_self_attention(img_tensor)
# Shape: [1, n_heads, N+1, N+1]

# CLS token 對各 patch 的 attention → 哪裡重要
cls_attn = attention[0, :, 0, 1:]   # [n_heads, N_patches]
mean_attn = cls_attn.mean(0)        # 取平均 head

# 轉成空間 heatmap → threshold → 找 bounding box
heatmap = mean_attn.reshape(grid_h, grid_w)
```

**核心原理**：
- 正常圖（OK）→ attention 分散在整張圖
- 缺陷圖（NG）→ attention **自動集中在缺陷區域**
- 可以用 attention map 的高值區域作為動態 ROI

**已有基礎**：`test.py` 中的 `get_last_self_attention()` 函數。

---

### 方法四：傳統 CV 模板匹配

```python
# OpenCV matchTemplate
result = cv2.matchTemplate(img, template_roi, cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(result)
# 以 max_loc 為中心擴展 ROI
x, y = max_loc
roi = (x - offset, y - offset, x + w + offset, y + h + offset)
```

- 適合有明顯固定特徵（對準標記、螺絲孔）的工站
- 速度極快，ARM 嵌入式也能跑

---

### ROI 方法選擇決策樹

```
相機 / 治具固定？
    ├─ Yes → 人工框定（最穩定、本系統已實作）
    └─ No
       │
       需要語義理解（「焊點」「元件」）？
       ├─ Yes，且不追求即時 → Grounding DINO + SAM
       ├─ Yes，需要即時   → YOLO-World
       └─ No，只用 DINOv3 → Attention Map 自動定位
                             （test.py 已有基礎，可直接擴展）
```

---

### 潛在整合方案（下一步）

> **半自動 ROI 建議流程**（使用者體驗最好）

```
使用者輸入文字描述 → Grounding DINO 自動推薦框
        ↓
前端顯示推薦框，使用者可拖曳調整
        ↓
確認後送入 DINOv3 提取 ROI 特徵
```

---

## 8. 後端 API 說明

### `GET /api/status`
回傳模型載入狀態。

```json
{
  "using_dinov2": true,
  "device": "cuda",
  "model_loaded": true
}
```

---

### `POST /api/evaluate`

**請求**（multipart/form-data）：

| 欄位 | 類型 | 說明 |
|------|------|------|
| `ok_images` | File[] | OK 圖片（必填，最多取前 20 張）|
| `ng_images` | File[] | NG 圖片（可選，最多取前 20 張）|

**回應**（JSON）：

```json
{
  "metrics": {
    "data_complexity": 42.3,
    "feature_richness": 68.1,
    "ok_ng_separation": 55.7,
    "has_ng": true
  },
  "chart": "<base64 PNG>",
  "visualizations": [
    {
      "filename": "ok_001.jpg",
      "type": "OK",
      "original": "<base64 JPEG>",
      "heatmap": "<base64 PNG>"
    }
  ],
  "recommendations": [
    {"level": "good", "text": "✅ OK 樣本外觀相似度高…"}
  ],
  "using_dinov2": true
}
```

---

### `POST /api/sort/set-templates`

**請求**（multipart/form-data）：

| 欄位 | 說明 |
|------|------|
| `session_id` | 工作階段 ID（前端自動生成）|
| `images` | 模板圖片（多張）|
| `rois` | JSON array，每個 ROI 包含 `{image_index, x, y, w, h, name}` |

若 `rois` 為空陣列，則以整張圖片為模板（使用檔名作為類別名）。

---

### `POST /api/sort/match`

**請求**（multipart/form-data）：

| 欄位 | 說明 |
|------|------|
| `session_id` | 對應已設定的模板 |
| `images` | 要分類的目標圖片 |

**回應**：

```json
{
  "results": {
    "焊點區域": [
      {"filename": "img_01.jpg", "similarity": 0.923, "image": "<b64>"}
    ]
  },
  "categories": ["焊點區域", "元件A"],
  "previews": {"焊點區域": "<b64 預覽圖>"}
}
```

---

### `POST /api/sort/download`

**請求**（JSON）：
```json
{"results": { "類別名": [{"filename": "...", "image": "<b64>"}] }}
```

**回傳**：`application/zip` 二進位資料流

---

## 9. 前端 UI 設計細節

### 設計風格

- **深色主題**（背景 `#0d0d1a`，卡片 `#1e1e40`）
- 主色調：Cyan `#4cc9f0`，紫 `#7209b7`，粉 `#f72585`
- 圓角 12px，4px 陰影，過渡動畫 0.2s

### 主要 UI 狀態管理（JavaScript）

```javascript
// 全局狀態
let okFiles    = [];       // 評估用 OK 圖片
let ngFiles    = [];       // 評估用 NG 圖片
let tmplFiles  = [];       // AI 分類：模板圖片
let targetFiles = [];      // AI 分類：目標圖片
let rois       = [];       // 已定義的 ROI 列表
let sortResults = null;    // 最後一次分類結果（用於下載）
let sessionId  = 'session_' + Date.now();  // 唯一工作階段 ID
```

### Canvas 座標系統（重要）

```
原始圖片尺寸（送後端）：1204 × 602 px
         ↕ scale = min(maxW/W, maxH/H, 1)
Canvas 顯示尺寸（滑鼠事件）：例如 800 × 400 px

ROI 框選時：
  canvas_x / scale → 原始圖片 x 座標
  canvas_y / scale → 原始圖片 y 座標
```

### 顏色循環（ROI 區分）

```javascript
const roiColors = [
  '#f72585', '#4cc9f0', '#7209b7',
  '#06d6a0', '#ffd166', '#ef476f', '#118ab2'
];
// 每新增一個 ROI 自動取下一個顏色
```

---

## 10. 檔案結構

```
New-workstation-Image-evaluation/
│
├── app.py                        ← Flask 後端（主伺服器）
├── requirements_web.txt          ← Web App 依賴清單
├── flowchart.html                ← PPT 用評估流程圖
├── PROJECT_NOTES.md              ← 本文件（開發筆記）
│
├── templates/
│   └── index.html                ← 單頁前端（Tab1 評估 + Tab2 AI分類）
│
├── 01_improved_dinov3_labeling.py  ← 多區域對比標注系統（原始研究代碼）
├── dinov3圖像質量量化對比.py        ← 圖像質量量化分析（原始研究代碼）
├── test.py                          ← 注意力圖 & CLAHE 測試
├── test4.py                         ← ROI 相似度匹配測試
│
└── README.md                        ← 專案說明
```

### 各 Python 原始碼的核心功能

| 檔案 | 關鍵函數 | 功能 |
|------|----------|------|
| `test4.py` | `get_region_feature()` | 從 patch token 提取指定像素區域的平均特徵 |
| `test.py` | `get_last_self_attention()` | 提取 CLS→Patch attention（可用於動態 ROI）|
| `dinov3圖像質量量化對比.py` | `analyze_edge_gradients()` | Sobel 邊緣梯度強度分析 |
| `01_improved_dinov3_labeling.py` | `dynamic_contrastive_scoring()` | 動態對比評分（缺陷熱力圖）|

---

## 11. 啟動方式

```bash
# 安裝依賴
pip install -r requirements_web.txt

# 啟動伺服器
python app.py

# 開啟瀏覽器
http://localhost:5000
```

### 模型載入說明

| 狀況 | 行為 | 前端顯示 |
|------|------|----------|
| 本地有 `.pth` 權重 | 載入本地 DINOv3 | `DINOv2 已就緒 (cuda)` |
| 有網路無本地權重 | 從 torch.hub 下載 DINOv2 | `DINOv2 已就緒 (cpu)` |
| 無網路無權重 | OpenCV HOG 特徵 | `CV 模式 (cpu)` |

> CV 模式下所有功能仍可使用，但特徵質量較差，分類/評估精度會下降。

---

## 12. 待辦 / 後續方向

### 短期優化

- [ ] **Attention Map 動態 ROI**：利用 `test.py` 的 `get_last_self_attention()` 自動生成 ROI 建議框，讓使用者在前端確認後再送入分類
- [ ] **批次處理進度條**：目前評估 20 張以上圖片時沒有進度回饋，可改為 Server-Sent Events（SSE）串流
- [ ] **評估結果 PDF 匯出**：將指標 + 圖表 + 建議打包成 PDF 報告

### 中期功能

- [ ] **Grounding DINO 整合**：使用者輸入文字描述 → 自動生成 ROI 推薦框 → 前端確認調整
- [ ] **多工站比較**：支援同時上傳多個工站的數據，並排比較三項指標
- [ ] **歷史記錄**：將每次評估結果存入 SQLite，支援時間軸對比

### 長期方向

- [ ] **主動學習標注建議**：根據特徵分布，推薦「哪些樣本最值得標注」
- [ ] **邊緣部署版本**：將特徵提取打包為 ONNX，支援工控機本地運行（無網路環境）
- [ ] **即時產線接入**：支援相機 SDK（Basler / Hikvision）直接串流評估

---

---

## 13. 傳統圖像品質指標（補充實作，2026-03-05）

> 與第 5 節的 DINOv3 語義指標互補，純 OpenCV 計算，無需模型，速度極快。

### 四項指標定義

| 指標 | 計算方式 | 原始值範圍 | 正規化分數（0-100）| 越高越好？ |
|------|----------|-----------|-------------------|-----------|
| **亮度評分** | 灰階均值的鐘形曲線評分，中心127.5 | 0-255 | `100 - |mean-127.5|/127.5*100` | ✅ |
| **清晰度** | Laplacian 算子方差（越高=越銳利）| 0-∞ | `log1p(var)/log1p(1000)*100` | ✅ |
| **對比度** | 灰階像素標準差 / RMS 對比 | 0-127 | `std/60*100`（上限100）| ✅ |
| **抗噪訊評分** | 高頻殘差：`std(gray - GaussianBlur(gray, 5×5))` | 0-∞ | `100 - noise/15*100`（反向）| ✅ |

### 警告旗標（per-image flags）

```python
flags = []
if sharpness_score < 30:     flags.append('blur')          # 模糊
if overexposure_pct  > 5:    flags.append('overexposed')   # 過曝（>240灰階超過5%像素）
if underexposure_pct > 5:    flags.append('underexposed')  # 欠曝（<15灰階超過5%像素）
if noise_score       < 40:   flags.append('noisy')         # 高噪訊
```

### 聚合統計（dataset-level）

```python
ok_agg = {
    'brightness_score_mean': ...,  # 所有OK圖平均亮度評分
    'sharpness_score_mean':  ...,
    'contrast_score_mean':   ...,
    'noise_score_mean':      ...,
    'blur_count':      int,  # 有 'blur' flag 的圖片數
    'overexposed_count': int,
    'underexposed_count': int,
    'noisy_count':     int,
    'total':           int,
}
```

### 對應建議邏輯

| 條件 | 建議文字 | 等級 |
|------|---------|------|
| 平均清晰度 < 40 | 整體偏模糊，建議檢查鏡頭焦距 | ⚠️ warn |
| 平均清晰度 > 70 | 清晰度良好 | ✅ good |
| 過曝/欠曝 > 0 張 | 曝光問題，調整光源 | ⚠️ warn |
| 平均對比度 < 35 | 對比度偏低，建議調整打光 | ⚠️ warn |
| 平均抗噪訊 < 50 | 雜訊高，降低 ISO 或增加曝光 | ⚠️ warn |

### UI 呈現位置

```
評估結果區（evalResults div）
  ├── DINOv3 語義指標卡 ×3（DC / FR / Sep）
  ├── DINOv3 評估圖表（matplotlib）
  ├── [NEW] 傳統圖像品質分析卡
  │     ├── 4 個品質分數卡（亮度/清晰度/對比度/抗噪訊）
  │     │   └── 每張卡：分數大字、標準差、進度條、NG 對比（若有）
  │     ├── 曝光摘要 chips（平均亮度 / 過曝N張 / 欠曝N張 / 模糊N張）
  │     └── 品質專用 matplotlib 圖表（scores比較 + 警告旗標分佈）
  ├── 評估建議（含新增的品質類建議）
  └── 特徵熱力圖 × N
        └── [NEW] 每張圖底部：per-image 旗標 badge + 亮/銳/對/噪 小分數
```

### 與 DINOv3 指標的互補關係

```
DINOv3 指標（語義層）                傳統 CV 指標（像素層）
─────────────────────────────────────────────────────────
數據複雜度（OK 多樣性）  ←→   亮度/對比度一致性（外觀穩定性）
特徵豐富度（PCA 熵）     ←→   清晰度（高頻細節是否存在）
OK/NG 分離度（Fisher）  ←→   對比度差異（缺陷可視性的基礎）
                              雜訊水平（影響DINOv3特徵質量）
```

**建議解讀流程**：
1. 先看傳統指標 → 確保圖像本身質量沒問題（曝光、清晰、低噪訊）
2. 再看 DINOv3 指標 → 評估語義層面的可學習性
3. 若傳統指標差 + DINOv3 指標也差 → 優先改善拍攝條件
4. 若傳統指標正常但 DINOv3 指標差 → 考慮增加數據量或選擇更細的 ROI

---

*本筆記由 Claude Code 協助生成，記錄截至 2026-03-05 的所有討論與實作細節。*
