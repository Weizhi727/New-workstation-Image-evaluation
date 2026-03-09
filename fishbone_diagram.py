"""
雲智判上線自動化流程 - 魚骨圖生成腳本
Fishbone Diagram Generator for Cloud AI Inspection Automation Flow
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ── 全域字型設定 ──────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK TC',
                                'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ── 顏色方案 ──────────────────────────────────────────────────────────────────
COLORS = {
    '初期評估':  '#E74C3C',
    '數據標註':  '#E67E22',
    '模型訓練':  '#F1C40F',
    '模型優化':  '#2ECC71',
    '用戶反饋':  '#3498DB',
}
SPINE_COLOR   = '#2C3E50'
EFFECT_COLOR  = '#8E44AD'
BG_COLOR      = '#FAFAFA'
LABEL_BG      = '#ECF0F1'

# ── 魚骨資料定義 ──────────────────────────────────────────────────────────────
BONES = [
    {
        'name': '初期評估',
        'position': 0.15,
        'side': 'top',
        'issues': [
            '缺乏量化評估指標',
            '缺乏模型可行性評估工具',
            '完全依賴人工主觀判斷',
            '評估時程冗長',
        ],
        'missing': [
            '→ 自動化指標評估模組 (DINOv3)',
            '→ 數據可行性評分系統',
        ],
    },
    {
        'name': '數據標註',
        'position': 0.30,
        'side': 'bottom',
        'issues': [
            '完全依賴人工手動標註',
            '缺乏通用標註方法/框架',
            '標註一致性難以保證',
            '標註效率低，開發時程長',
        ],
        'missing': [
            '→ 半自動/主動學習標註工具',
            '→ 通用標註標準與流程',
        ],
    },
    {
        'name': '模型訓練',
        'position': 0.48,
        'side': 'top',
        'issues': [
            '需要手動撰寫訓練程式碼',
            '超參數調整依賴人工經驗',
            '缺乏自動化訓練流水線',
            '重複性工作佔用大量開發資源',
        ],
        'missing': [
            '→ AutoML / 訓練流水線自動化',
            '→ 超參數自動搜索 (HPO)',
        ],
    },
    {
        'name': '模型優化',
        'position': 0.63,
        'side': 'bottom',
        'issues': [
            '設備差異導致大量過殺',
            '曝光度/機台調整造成圖像偏移',
            '新機台須大量重新人工 Label',
            '缺乏自動化優化與輔助標註方案',
            '10 台機台只標 2 台，剩餘陸續補標成本高',
        ],
        'missing': [
            '→ 自動域適應 (Domain Adaptation)',
            '→ AI 輔助標註 / 主動學習',
            '→ 多機台差異自動補償',
        ],
    },
    {
        'name': '用戶反饋',
        'position': 0.80,
        'side': 'top',
        'issues': [
            '僅能獲得「復判 OK」反饋資訊',
            '缺乏漏檢 Sorting 機制',
            '過殺/漏檢反饋無法閉環進模型',
            '缺乏相似異常物料自動搜尋能力',
        ],
        'missing': [
            '→ OK/NG 雙向反饋閉環系統',
            '→ 漏檢異常 Sorting 模型',
            '→ 持續學習 (Continual Learning)',
        ],
    },
]

EFFECT_TEXT = '雲智判上線\n自動化流程\n核心缺失'


def draw_arrow(ax, x0, y0, x1, y1, color, lw=2, headwidth=8, headlength=10):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle='->', color=color,
                                 lw=lw, mutation_scale=headlength))


def fishbone(output='fishbone_diagram.png', dpi=180):
    fig, ax = plt.subplots(figsize=(22, 12))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')

    ax.text(0.5, 0.97,
            '雲智判上線自動化流程魚骨圖 — 缺失問題與技術分析',
            ha='center', va='top', fontsize=17, fontweight='bold',
            color=SPINE_COLOR, transform=ax.transAxes)

    spine_y  = 0.0
    spine_x0 = 0.04
    spine_x1 = 0.88

    ax.annotate('', xy=(spine_x1, spine_y), xytext=(spine_x0, spine_y),
                 arrowprops=dict(arrowstyle='->', color=SPINE_COLOR,
                                 lw=4, mutation_scale=20))

    effect_box = mpatches.FancyBboxPatch(
        (0.885, -0.17), 0.108, 0.34,
        boxstyle='round,pad=0.01',
        linewidth=2, edgecolor=EFFECT_COLOR,
        facecolor='#EBD5F7')
    ax.add_patch(effect_box)
    ax.text(0.939, spine_y, EFFECT_TEXT,
            ha='center', va='center', fontsize=11,
            fontweight='bold', color=EFFECT_COLOR, linespacing=1.6)

    DIAG_LEN = 0.30
    ANGLE_DEG = 45

    for bone in BONES:
        xp    = spine_x0 + bone['position'] * (spine_x1 - spine_x0)
        side  = bone['side']
        sign  = 1 if side == 'top' else -1
        color = COLORS[bone['name']]

        angle_rad = np.radians(ANGLE_DEG)
        dx = DIAG_LEN * np.cos(angle_rad)
        dy = sign * DIAG_LEN * np.sin(angle_rad)
        xb = xp - dx
        yb = spine_y + dy

        ax.annotate('', xy=(xp, spine_y), xytext=(xb, yb),
                     arrowprops=dict(arrowstyle='->', color=color,
                                     lw=2.5, mutation_scale=14))

        label_y = yb + sign * 0.07
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor=color,
                          edgecolor='none', alpha=0.9)
        ax.text(xb, label_y, bone['name'],
                ha='center', va='center', fontsize=12,
                fontweight='bold', color='white', bbox=bbox_props)

        n_issues = len(bone['issues'])
        for i, issue in enumerate(bone['issues']):
            t = (i + 1) / (n_issues + 1)
            xs = xb + t * dx
            ys = yb - t * dy
            sub_len = 0.09
            xs_end  = xs + sub_len
            ax.plot([xs, xs_end], [ys, ys], color=color, lw=1.5, alpha=0.8)
            ax.text(xs_end + 0.005, ys, f'✗ {issue}',
                    ha='left', va='center', fontsize=7.5, color='#2C3E50',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='white', edgecolor=color,
                              alpha=0.85, linewidth=0.8))

        miss_base_y = yb + sign * 0.19
        for j, miss in enumerate(bone['missing']):
            my = miss_base_y + sign * j * 0.09
            ax.text(xb, my, miss,
                    ha='center', va='center', fontsize=7.5,
                    color=color, style='italic',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='#FDFEFE', edgecolor=color,
                              alpha=0.9, linewidth=1, linestyle='--'))

    legend_items = (
        [mpatches.Patch(color=c, label=n) for n, c in COLORS.items()]
        + [mpatches.Patch(facecolor='white', edgecolor='grey',
                          linestyle='--', label='建議補足技術')]
    )
    ax.legend(handles=legend_items, loc='lower center',
              ncol=6, fontsize=9, framealpha=0.8,
              bbox_to_anchor=(0.45, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output, dpi=dpi, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f'[✓] 魚骨圖已儲存至: {output}')


if __name__ == '__main__':
    fishbone()
