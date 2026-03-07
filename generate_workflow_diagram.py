#!/usr/bin/env python3
"""
異常檢測工站圖像可行性評估流程圖產生器

Usage:
    python generate_workflow_diagram.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── CJK 字體 ──────────────────────────────────────────────────────────────────
_FONT = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
fm.fontManager.addfont(_FONT)
matplotlib.rcParams['font.family'] = fm.FontProperties(fname=_FONT).get_name()

# ── 色盤 ───────────────────────────────────────────────────────────────────────
C = {
    'bg':      '#F0F4F8',
    'p1':      '#1565C0',  # 輸入
    'roi_d':   '#4A148C',  # 決策菱形
    'roi_b':   '#6A1B9A',  # ROI 框
    'p3':      '#01579B',  # 品質
    'p4':      '#B71C1C',  # 可分性
    'p5':      '#1B5E20',  # 可見性
    'p6':      '#7B1FA2',  # 任務
    'p7':      '#004D40',  # 輸出
    'algo':    '#263238',  # 演算法框
    'thr_ok':  '#2E7D32',  # 門檻-通過
    'thr_ng':  '#C62828',  # 門檻-高風險
    'thr_bd':  '#E65100',  # 門檻-邊界
    'arr':     '#455A64',
    'title':   '#0D1B4B',
}

# ── 畫布 ───────────────────────────────────────────────────────────────────────
W, H = 26, 58
fig, ax = plt.subplots(figsize=(W * 0.52, H * 0.52))
fig.patch.set_facecolor(C['bg'])
ax.set_facecolor(C['bg'])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')

CX = 13          # 主流程中心 x
MW = 15.5        # 主框寬度
AW = 3.45        # 演算法子框寬度
AH = 1.35        # 演算法子框高度（含門檻行）


# ── 工具函式 ───────────────────────────────────────────────────────────────────

def box(bx, by, bw, bh, text, fc, fs=10, tc='white', bold=True, lh=1.55, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (bx - bw / 2, by - bh / 2), bw, bh,
        boxstyle='round,pad=0.1,rounding_size=0.2',
        facecolor=fc, edgecolor='white', linewidth=2.0,
        alpha=alpha, zorder=3,
    ))
    ax.text(bx, by, text, ha='center', va='center', fontsize=fs, color=tc,
            fontweight='bold' if bold else 'normal',
            multialignment='center', zorder=4, linespacing=lh)


def diamond(dx, dy, dw, dh, text, fc, fs=10):
    pts = np.array([[dx, dy+dh], [dx+dw, dy], [dx, dy-dh], [dx-dw, dy]])
    ax.add_patch(plt.Polygon(pts, facecolor=fc, edgecolor='white', lw=2.0, zorder=3))
    ax.text(dx, dy, text, ha='center', va='center', fontsize=fs,
            color='white', fontweight='bold', multialignment='center', zorder=4)


def arrow(x1, y1, x2, y2, lbl='', lx=0.15, ly=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C['arr'], lw=2.0,
                                mutation_scale=18), zorder=2)
    if lbl:
        ax.text((x1+x2)/2 + lx, (y1+y2)/2 + ly, lbl,
                fontsize=9, color=C['arr'], fontweight='bold', zorder=5)


def line(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=C['arr'], lw=2.0, zorder=2)


def phase_bar(by, num, title, fc):
    box(CX, by, MW, 0.85, f'PHASE {num}   {title}', fc, fs=11.5)


def algo_box(bx, by, name, threshold_lines):
    """演算法框：上半方法名、下半門檻說明。"""
    box(bx, by, AW, AH, name + '\n' + threshold_lines,
        C['algo'], fs=8.2, lh=1.45)


def algo_row(xs, y, items, parent_y):
    """items: list of (name, threshold_str)"""
    for bx, (name, thr) in zip(xs, items):
        line(bx, parent_y, bx, y + AH / 2)
        algo_box(bx, y, name, thr)


def section_divider(y, label):
    ax.plot([0.5, W - 0.5], [y, y], color='#B0BEC5', lw=1.0, ls='--', zorder=1)
    ax.text(0.7, y + 0.15, label, fontsize=8, color='#78909C', style='italic')


# ══════════════════════════════════════════════════════════════════════════════
# 標題
# ══════════════════════════════════════════════════════════════════════════════
ax.text(CX, 57.1, '異常檢測工站  —  圖像可行性評估分析流程',
        ha='center', va='center', fontsize=17, fontweight='bold', color=C['title'])
ax.text(CX, 56.45,
        'Anomaly Detection Workstation · Image Feasibility Assessment Workflow',
        ha='center', va='center', fontsize=9.5, color='#3949AB', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — 數據接收
# ══════════════════════════════════════════════════════════════════════════════
Y1 = 55.35
box(CX, Y1, MW, 1.0,
    'PHASE 1   數據接收\nOK / NG 圖像  +  規格文檔（異常類型、ROI 範圍、判斷標準）',
    C['p1'], fs=10.5)
arrow(CX, Y1 - 0.5, CX, Y1 - 1.1)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — ROI 判斷
# ══════════════════════════════════════════════════════════════════════════════
Y2 = 53.8
diamond(CX, Y2, 3.0, 0.88, 'ROI 為局部區域？', C['roi_d'], fs=10)

# 左分支
arrow(CX - 3.0, Y2, CX - 6.0, Y2, lbl=' 是', lx=-1.5, ly=0.2)
box(CX - 8.0, Y2, 3.8, 1.05,
    'GroundingDINO / SAM\n文字引導偵測並裁切 ROI', C['roi_b'], fs=9.5)

# 右分支
arrow(CX + 3.0, Y2, CX + 6.0, Y2, lbl=' 否', lx=0.1, ly=0.2)
box(CX + 7.8, Y2, 3.5, 1.05,
    '全圖視為 ROI\n直接進行後續分析', C['roi_b'], fs=9.5)

# 合流
MY = Y2 - 1.45
line(CX - 6.1, Y2 - 0.53, CX - 6.1, MY)
line(CX + 6.05, Y2 - 0.53, CX + 6.05, MY)
line(CX - 6.1, MY, CX + 6.05, MY)
arrow(CX, MY, CX, MY - 0.32)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — 圖像品質
# ══════════════════════════════════════════════════════════════════════════════
Y3 = 51.5
phase_bar(Y3, 3, '圖像品質基礎評估', C['p3'])
A3Y = Y3 - 1.75
A3XS = [CX - 5.2, CX - 1.73, CX + 1.73, CX + 5.2]
algo_row(A3XS, A3Y, [
    ('Laplacian 方差\n模糊偵測',
     '< 100  高風險\n100~500  可用  > 500  清晰'),
    ('亮度 / 對比度\n分布分析',
     'std < 20  低對比高風險\nstd > 50  對比良好'),
    ('解析度 &\nROI 像素尺寸',
     '< 32px  過小難訓練\n> 64px  可用'),
    ('標注數量 &\n類別平衡性',
     '比例 > 10:1  高風險\n< 5:1  可接受'),
], parent_y=Y3 - 0.43)
arrow(CX, A3Y - AH/2, CX, A3Y - AH/2 - 0.5)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — OK/NG 可分性
# ══════════════════════════════════════════════════════════════════════════════
Y4 = 47.8
phase_bar(Y4, 4, 'OK / NG 可分性分析', C['p4'])
A4Y = Y4 - 1.75
A4XS = [CX - 5.2, CX - 1.73, CX + 1.73, CX + 5.2]
algo_row(A4XS, A4Y, [
    ('CLIP / ResNet\n特徵提取',
     '基礎步驟\n供後續指標使用'),
    ('t-SNE / UMAP\n特徵分布可視化',
     '群集重疊  難以區分\n群集分離  可分性高'),
    ('Fisher 判別比\nFDR',
     '< 0.5  難以區分\n0.5~1  邊界  > 1  可分'),
    ('Mahalanobis\n距離分析',
     '< 2  嚴重重疊\n> 5  分布清楚分離'),
], parent_y=Y4 - 0.43)
arrow(CX, A4Y - AH/2, CX, A4Y - AH/2 - 0.5)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — 異常可見性
# ══════════════════════════════════════════════════════════════════════════════
Y5 = 44.1
phase_bar(Y5, 5, '異常可見性評估', C['p5'])
A5Y = Y5 - 1.75
A5XS = [CX - 5.2, CX - 1.73, CX + 1.73, CX + 5.2]
algo_row(A5XS, A5Y, [
    ('LBP / GLCM / Gabor\n紋理差異量化',
     'Chi2 < 50  紋理相似\nChi2 > 150  差異明顯'),
    ('SSIM\n結構相似度',
     '> 0.95  幾乎一樣高風險\n0.85~0.95  邊界\n< 0.85  差異可偵測'),
    ('FFT 頻域\n異常訊號強度',
     '高頻能量差 < 5%  難偵測\n> 15%  頻域有明顯差異'),
    ('PatchCore\n無監督異常評分',
     'AUC < 0.70  難偵測\n0.70~0.85  邊界\nAUC > 0.85  可偵測'),
], parent_y=Y5 - 0.43)
arrow(CX, A5Y - AH/2, CX, A5Y - AH/2 - 0.5)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — 任務適配性
# ══════════════════════════════════════════════════════════════════════════════
Y6 = 40.4
phase_bar(Y6, 6, '任務適配性評估', C['p6'])
T6Y = Y6 - 1.9
T6XS = [CX - 4.6, CX, CX + 4.6]
T6W = 4.3
for bx, (name, thr) in zip(T6XS, [
    ('目標檢測\n異常可定位？',
     '尺寸 > 8px  FDR > 0.5\nSSIM < 0.90'),
    ('語意分割\n邊緣清晰？',
     'SSIM < 0.85\nLBP Chi2 > 100'),
    ('影像分類 / 無監督\n全局特徵可分？',
     'FDR > 1.0  或\nPatchCore AUC > 0.85'),
]):
    line(bx, Y6 - 0.43, bx, T6Y + AH/2)
    box(bx, T6Y, T6W, AH + 0.2, name + '\n' + thr, C['p6'], fs=8.5, lh=1.45)

arrow(CX, T6Y - AH/2 - 0.1, CX, T6Y - AH/2 - 0.6)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — 報告輸出
# ══════════════════════════════════════════════════════════════════════════════
Y7 = 37.4
box(CX, Y7, MW, 1.0,
    'PHASE 7   評估報告輸出\n量化指標摘要  ·  可視化圖表  ·  模型選型與標注策略建議',
    C['p7'], fs=10.5)

# ══════════════════════════════════════════════════════════════════════════════
# 門檻速查表
# ══════════════════════════════════════════════════════════════════════════════
section_divider(36.5, '門檻速查表  Threshold Reference')

TBL_Y = 36.1   # 表格起始 y（往下增長）
ROW_H = 1.05
HDR_H = 0.72
COL_X = [1.0, 5.8, 10.8, 17.2, 23.3]   # 欄位左緣 x
COL_W = [4.5, 4.7, 6.1,  5.8,  2.5]    # 欄位寬
COLS  = ['指標 / 方法', '通過門檻 (Pass)', '邊界門檻 (Marginal)', '高風險 (High Risk)', '所屬 Phase']

# 表頭
for cx_col, cw, col_name in zip(COL_X, COL_W, COLS):
    box(cx_col + cw/2, TBL_Y - HDR_H/2, cw - 0.12, HDR_H,
        col_name, '#37474F', fs=8.5)

rows = [
    # (指標, Pass, Marginal, High Risk, Phase)
    ('Laplacian 方差\n(模糊偵測)',      '> 500',               '100 ~ 500',            '< 100',              'P3'),
    ('對比度\n(像素 std)',              'std > 50',            'std 20~50',            'std < 20',           'P3'),
    ('ROI 尺寸',                       '> 64 × 64 px',        '32 ~ 64 px',           '< 32 × 32 px',       'P3'),
    ('類別比例',                        '< 3 : 1',             '3:1 ~ 10:1',           '> 10 : 1',           'P3'),
    ('Fisher 判別比\n(FDR)',            '> 1.0',               '0.5 ~ 1.0',            '< 0.5',              'P4'),
    ('Mahalanobis 距離',               '> 5',                 '2 ~ 5',                '< 2',                'P4'),
    ('SSIM\n(OK vs NG 平均)',           '< 0.85  (差異大)',    '0.85 ~ 0.95',          '> 0.95  (幾乎一樣)', 'P5'),
    ('LBP Chi-square\n距離',           '> 150',               '50 ~ 150',             '< 50',               'P5'),
    ('FFT 高頻能量差',                  '> 15%',               '5% ~ 15%',             '< 5%',               'P5'),
    ('PatchCore AUC',                  '> 0.85',              '0.70 ~ 0.85',          '< 0.70',             'P5'),
]

ROW_COLORS = [C['thr_ok'], C['thr_bd'], C['thr_ng'], '#455A64']

for ri, row in enumerate(rows):
    ry = TBL_Y - HDR_H - (ri + 0.5) * ROW_H
    row_bg = '#ECEFF1' if ri % 2 == 0 else '#CFD8DC'
    # 底色
    ax.add_patch(FancyBboxPatch(
        (COL_X[0] - 0.05, ry - ROW_H/2 + 0.05),
        sum(COL_W) + 0.3, ROW_H - 0.1,
        boxstyle='round,pad=0.05,rounding_size=0.1',
        facecolor=row_bg, edgecolor='none', zorder=2, alpha=0.6,
    ))
    for ci, (cx_col, cw, cell) in enumerate(zip(COL_X, COL_W, row)):
        cell_color = '#1A237E'
        if ci == 1:   cell_color = C['thr_ok']
        elif ci == 2: cell_color = C['thr_bd']
        elif ci == 3: cell_color = C['thr_ng']
        ax.text(cx_col + cw/2, ry, cell,
                ha='center', va='center', fontsize=7.8,
                color=cell_color, fontweight='bold' if ci > 0 else 'normal',
                multialignment='center', zorder=4, linespacing=1.4)

# 表格外框
tbl_total_h = HDR_H + len(rows) * ROW_H
ax.add_patch(FancyBboxPatch(
    (COL_X[0] - 0.1, TBL_Y - tbl_total_h - 0.05),
    sum(COL_W) + 0.3, tbl_total_h + 0.1,
    boxstyle='round,pad=0.05,rounding_size=0.15',
    facecolor='none', edgecolor='#90A4AE', linewidth=1.5, zorder=5,
))

# ══════════════════════════════════════════════════════════════════════════════
# 圖例
# ══════════════════════════════════════════════════════════════════════════════
legend_bottom = TBL_Y - tbl_total_h - 0.3
section_divider(legend_bottom, '圖例  Legend')

LX, LY = 1.0, legend_bottom - 0.45
ax.text(LX, LY, '圖例', fontsize=9, fontweight='bold', color='#37474F')

legend_items = [
    (C['p1'],    '數據輸入 / 輸出'),
    (C['roi_b'], 'ROI 提取處理'),
    (C['p3'],    '品質評估'),
    (C['p4'],    '可分性分析'),
    (C['p5'],    '可見性評估'),
    (C['p6'],    '任務適配性'),
    (C['algo'],  '演算法模組（含門檻）'),
    (C['roi_d'], '決策節點（菱形）'),
]
for i, (color, label) in enumerate(legend_items):
    col = i // 4
    row = i % 4
    lxi = LX + col * 5.5
    lyi = LY - 0.65 * (row + 1)
    box(lxi + 0.6, lyi, 1.0, 0.44, '', color)
    ax.text(lxi + 1.2, lyi, label, fontsize=8.5, color='#263238', va='center')

# ══════════════════════════════════════════════════════════════════════════════
# 工站情境附注（右側獨立欄）
# ══════════════════════════════════════════════════════════════════════════════
NX, NY = 23.2, 55.0
box(NX, NY, 4.9, 8.5,
    '常見工站情境\n\n'
    '焊接工站\n'
    '- 焊點偏位 / 虛焊\n'
    '- 紋理差異難肉眼判別\n'
    '- 建議: LBP + PatchCore\n\n'
    '點膠工站\n'
    '- 膠路斷膠 / 膠寬異常\n'
    '- 異物汙染\n'
    '- 建議: SSIM + 目標檢測\n\n'
    '外觀檢測工站\n'
    '- 碰傷 / 刮傷 / 壓傷\n'
    '- 極小異常 / 低對比\n'
    '- 先 Detect 元件\n'
    '  再偵測缺陷',
    C['algo'], fs=8.3, lh=1.6)

# ══════════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
plt.savefig('evaluation_workflow.png', dpi=180, bbox_inches='tight',
            facecolor=C['bg'])
print('[OK] evaluation_workflow.png 已儲存')
