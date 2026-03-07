#!/usr/bin/env python3
"""
異常檢測工站圖像可行性評估流程圖產生器
========================================
執行此腳本即可產生 evaluation_workflow.png 流程圖。

Usage:
    python generate_workflow_diagram.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ─── 載入 CJK 字體（WenQuanYi Zen Hei）────────────────────────────────────────
_CJK_FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
_cjk_prop = fm.FontProperties(fname=_CJK_FONT_PATH)
matplotlib.rcParams['font.family'] = _cjk_prop.get_name()
fm.fontManager.addfont(_CJK_FONT_PATH)

# ─── 色盤 ────────────────────────────────────────────────────────────────────
C = {
    'bg':       '#EEF2F7',
    'phase1':   '#1565C0',   # 深藍 — 輸入
    'roi_dec':  '#4A148C',   # 深紫 — 決策
    'roi_box':  '#7B1FA2',   # 紫   — ROI 處理
    'quality':  '#0277BD',   # 藍   — 品質評估
    'sep':      '#BF360C',   # 深橘 — 可分性
    'vis':      '#1B5E20',   # 深綠 — 可見性
    'task':     '#880E4F',   # 深粉 — 任務評估
    'output':   '#004D40',   # 深青 — 報告輸出
    'algo':     '#37474F',   # 藍灰 — 演算法模組
    'arrow':    '#546E7A',
    'title':    '#1A237E',
}

fig, ax = plt.subplots(figsize=(22, 36))
fig.patch.set_facecolor(C['bg'])
ax.set_facecolor(C['bg'])
ax.set_xlim(0, 22)
ax.set_ylim(0, 36)
ax.axis('off')

CX   = 11     # 中心 x
WM   = 13.5   # 主框寬度


# ─── 繪圖工具函式 ─────────────────────────────────────────────────────────────

def draw_box(bx, by, bw, bh, text, fc,
             fontsize=10, tc='white', bold=True, alpha=1.0, lh=1.5):
    rect = FancyBboxPatch(
        (bx - bw / 2, by - bh / 2), bw, bh,
        boxstyle="round,pad=0.12,rounding_size=0.22",
        facecolor=fc, edgecolor='white', linewidth=2.2,
        alpha=alpha, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(bx, by, text,
            ha='center', va='center',
            fontsize=fontsize, color=tc,
            fontweight='bold' if bold else 'normal',
            multialignment='center', zorder=4,
            linespacing=lh)


def draw_diamond(dx, dy, dw, dh, text, fc, fontsize=9.5):
    pts = np.array([
        [dx,      dy + dh],
        [dx + dw, dy],
        [dx,      dy - dh],
        [dx - dw, dy],
    ])
    poly = plt.Polygon(pts, facecolor=fc, edgecolor='white', linewidth=2.2, zorder=3)
    ax.add_patch(poly)
    ax.text(dx, dy, text,
            ha='center', va='center',
            fontsize=fontsize, color='white', fontweight='bold',
            multialignment='center', zorder=4)


def draw_arrow(x1, y1, x2, y2, label='', lx=0.2, ly=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=C['arrow'], lw=2.2,
                    mutation_scale=20,
                ), zorder=2)
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly,
                label, fontsize=9.5, color=C['arrow'],
                fontweight='bold', zorder=5)


def draw_line(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=C['arrow'], lw=2.2, zorder=2)


def phase_header(bx, by, bw, bh, phase_num, phase_title, fc):
    """主要階段標題框（Phase N + 標題）。"""
    draw_box(bx, by, bw, bh,
             f'PHASE {phase_num}  ｜  {phase_title}',
             fc, fontsize=11.5)


def algo_row(center_xs, y, texts, color, box_w=2.85, box_h=0.82,
             parent_bottom_y=None, fontsize=8.7):
    """在同一水平排列多個演算法子框，並從 parent_bottom_y 連線。"""
    for bx, text in zip(center_xs, texts):
        if parent_bottom_y is not None:
            draw_line(bx, parent_bottom_y, bx, y + box_h / 2)
        draw_box(bx, y, box_w, box_h, text, color, fontsize=fontsize)


# ═══════════════════════════════════════════════════════════════════════════════
# 標題
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(CX, 35.3, '異常檢測工站  ─  圖像可行性評估分析流程',
        ha='center', va='center', fontsize=18, fontweight='bold', color=C['title'])
ax.text(CX, 34.82,
        'Anomaly Detection Workstation  ·  Image Feasibility Assessment Workflow',
        ha='center', va='center', fontsize=10.5, color='#3949AB', style='italic')

# ══════════════════════════════════════════════
# PHASE 1 — 數據接收
# ══════════════════════════════════════════════
Y1 = 34.0
draw_box(CX, Y1, WM, 0.95,
         'PHASE 1  ｜  數據接收\nOK / NG 圖像   +   規格文檔（異常類型說明、ROI 範圍、判斷標準）',
         C['phase1'], fontsize=10.5)
draw_arrow(CX, Y1 - 0.48, CX, Y1 - 1.02)

# ══════════════════════════════════════════════
# PHASE 2 — ROI 判斷與提取
# ══════════════════════════════════════════════
Y2D = 32.6
draw_diamond(CX, Y2D, 2.8, 0.82,
             'ROI 為\n局部區域？', C['roi_dec'], fontsize=9.5)

# 左分支（是）
draw_arrow(CX - 2.8, Y2D, CX - 5.5, Y2D, label='  是', lx=-1.6, ly=0.18)
draw_box(CX - 7.4, Y2D, 3.6, 1.0,
         'GroundingDINO / SAM\n文字引導偵測並裁切 ROI',
         C['roi_box'], fontsize=9.5)

# 右分支（否）
draw_arrow(CX + 2.8, Y2D, CX + 5.5, Y2D, label='  否', lx=0.1, ly=0.18)
draw_box(CX + 7.2, Y2D, 3.2, 1.0,
         '全圖視為 ROI\n直接進行後續分析',
         C['roi_box'], fontsize=9.5)

# 合流
MERGE_Y = Y2D - 1.3
draw_line(CX - 5.6, Y2D - 0.5, CX - 5.6, MERGE_Y)
draw_line(CX + 5.6, Y2D - 0.5, CX + 5.6, MERGE_Y)
draw_line(CX - 5.6, MERGE_Y, CX + 5.6, MERGE_Y)
draw_arrow(CX, MERGE_Y, CX, MERGE_Y - 0.3)

# ══════════════════════════════════════════════
# PHASE 3 — 圖像品質基礎評估
# ══════════════════════════════════════════════
Y3 = 30.8
phase_header(CX, Y3, WM, 0.82, 3, '圖像品質基礎評估', C['quality'])

A3Y  = Y3 - 1.25
A3XS = [CX - 4.95, CX - 1.65, CX + 1.65, CX + 4.95]
algo_row(
    A3XS, A3Y,
    ['Laplacian 方差\n模糊偵測',
     '亮度 / 對比度\n分布分析',
     '解析度 &\nROI 像素尺寸',
     '標注數量 &\n類別平衡性'],
    C['algo'], parent_bottom_y=Y3 - 0.41,
)
draw_arrow(CX, A3Y - 0.41, CX, A3Y - 0.78)

# ══════════════════════════════════════════════
# PHASE 4 — OK / NG 可分性分析
# ══════════════════════════════════════════════
Y4 = 28.6
phase_header(CX, Y4, WM, 0.82, 4, 'OK / NG 可分性分析', C['sep'])

A4Y  = Y4 - 1.25
A4XS = [CX - 4.95, CX - 1.65, CX + 1.65, CX + 4.95]
algo_row(
    A4XS, A4Y,
    ['CLIP / ResNet\n嵌入特徵提取',
     't-SNE / UMAP\n特徵分布可視化',
     'Fisher 判別比\n(FDR)',
     'Mahalanobis\n距離分析'],
    C['algo'], parent_bottom_y=Y4 - 0.41,
)
draw_arrow(CX, A4Y - 0.41, CX, A4Y - 0.78)

# 補充說明
ax.text(CX + 7.2, Y4, '> FDR 高 → 特徵可分\n> 群集重疊 → 分類困難',
        fontsize=8, color='#6D4C41', va='center', ha='left',
        style='italic', multialignment='left')

# ══════════════════════════════════════════════
# PHASE 5 — 異常可見性評估
# ══════════════════════════════════════════════
Y5 = 26.4
phase_header(CX, Y5, WM, 0.82, 5, '異常可見性評估', C['vis'])

A5Y  = Y5 - 1.25
A5XS = [CX - 4.95, CX - 1.65, CX + 1.65, CX + 4.95]
algo_row(
    A5XS, A5Y,
    ['LBP / GLCM / Gabor\n紋理差異量化',
     'SSIM\n結構相似度分析',
     'FFT 頻域\n異常訊號強度',
     'PatchCore\n無監督異常評分'],
    C['algo'], parent_bottom_y=Y5 - 0.41,
)

ax.text(CX + 7.2, Y5, '> SSIM 低 → 結構差異大\n> Anomaly Score 高 → 可偵測',
        fontsize=8, color='#1B5E20', va='center', ha='left',
        style='italic', multialignment='left')

draw_arrow(CX, A5Y - 0.41, CX, A5Y - 0.78)

# ══════════════════════════════════════════════
# PHASE 6 — 任務適配性評估
# ══════════════════════════════════════════════
Y6 = 24.2
phase_header(CX, Y6, WM, 0.82, 6, '任務適配性評估', C['task'])

T6Y  = Y6 - 1.45
T6XS = [CX - 4.3, CX, CX + 4.3]
algo_row(
    T6XS, T6Y,
    ['目標檢測\n異常可定位？尺寸足夠？',
     '語意分割\n邊緣清晰？形狀可辨？',
     '影像分類\n全局特徵具區分性？'],
    C['task'], box_w=3.9, box_h=1.0,
    parent_bottom_y=Y6 - 0.41, fontsize=9,
)
draw_arrow(CX, T6Y - 0.5, CX, T6Y - 0.88)

# ══════════════════════════════════════════════
# PHASE 7 — 評估報告輸出
# ══════════════════════════════════════════════
Y7 = 21.5
draw_box(CX, Y7, WM, 1.05,
         'PHASE 7  ｜  評估報告輸出\n量化指標摘要   ·   可視化圖表   ·   模型選型與標注策略建議',
         C['output'], fontsize=10.5)

# ══════════════════════════════════════════════
# 工站情境說明（右下角附圖說）
# ══════════════════════════════════════════════
BOX_NOTE_X = 17.5
BOX_NOTE_Y = 26.0
draw_box(BOX_NOTE_X, BOX_NOTE_Y, 4.5, 5.8,
         '常見工站情境說明\n\n'
         '焊接工站\n'
         '> 焊點偏位 / 虛焊\n'
         '> 紋理差異難以肉眼判別\n\n'
         '點膠工站\n'
         '> 膠路斷膠 / 膠寬異常\n'
         '> 異物汙染\n\n'
         '外觀檢測工站\n'
         '> 碰傷 / 刮傷 / 壓傷\n'
         '> 極小異常 / 低對比\n'
         '> 先 Detect 元件再偵測缺陷',
         C['algo'], fontsize=8.5, alpha=0.88, lh=1.6)

# ══════════════════════════════════════════════
# 圖例
# ══════════════════════════════════════════════
LX, LY = 0.25, 20.5
ax.text(LX + 0.1, LY + 0.3, '圖  例',
        fontsize=10, fontweight='bold', color='#37474F')

legend_items = [
    (C['phase1'],  '數據輸入 / 輸出節點'),
    (C['roi_box'], 'ROI 偵測 & 裁切處理'),
    (C['quality'], '圖像品質評估'),
    (C['sep'],     'OK/NG 可分性分析'),
    (C['vis'],     '異常可見性評估'),
    (C['task'],    '任務適配性評估'),
    (C['algo'],    '演算法 / 工具模組'),
    (C['roi_dec'], '決策節點（菱形）'),
]
for i, (color, label) in enumerate(legend_items):
    yi = LY - 0.62 * (i + 1)
    draw_box(LX + 0.7, yi, 1.1, 0.46, '', color)
    ax.text(LX + 1.38, yi, label, fontsize=8.8, color='#37474F', va='center')

# ══════════════════════════════════════════════
# 存檔
# ══════════════════════════════════════════════
plt.tight_layout(pad=0.5)
out = 'evaluation_workflow.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=C['bg'])
print(f'[OK] 已儲存：{out}')
