"""
DINOv3 Industrial Vision Evaluation Web Server
==============================================
Backend for:
  Feature 1 - Image Quality Evaluation (data complexity, OK/NG separation, feature richness)
  Feature 2 - AI Sorting (ROI-based template matching and image classification)
"""

import os
import io
import sys
import json
import base64
import zipfile
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# ─────────────────────────────────────────────
# Model & device
# ─────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 16

_model = None
_using_dinov2 = False

TRANSFORM = T.Compose([
    T.Resize(448),
    T.CenterCrop(448),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_model():
    global _model, _using_dinov2
    if _model is not None:
        return _model

    # 1. Try local repo model (dinov3 local weights)
    local_weight_paths = [
        os.path.join(os.path.dirname(__file__),
                     'dinov3 weight',
                     'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),
        os.path.join(os.path.dirname(__file__),
                     'dinov3_vitb16_pretrain.pth'),
    ]
    for wp in local_weight_paths:
        if os.path.isfile(wp):
            try:
                print(f'[Model] Loading local DINOv3 from {wp}')
                sys.path.insert(0, os.path.dirname(__file__))
                m = torch.hub.load(
                    os.path.dirname(__file__),
                    'dinov3_vitb16',
                    source='local',
                    weights=None,
                    pretrained=False,
                )
                from collections import OrderedDict
                ckpt = torch.load(wp, map_location='cpu')
                sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
                clean = OrderedDict(
                    (k[7:] if k.startswith('module.') else k, v)
                    for k, v in sd.items()
                )
                m.load_state_dict(clean, strict=False)
                m = m.eval().to(DEVICE)
                _model = m
                _using_dinov2 = True
                print(f'[Model] Local DINOv3 loaded on {DEVICE}')
                return _model
            except Exception as e:
                print(f'[Model] Local load failed: {e}')

    # 2. Try torch hub DINOv2
    try:
        print('[Model] Trying torch.hub DINOv2…')
        m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
        m = m.eval().to(DEVICE)
        _model = m
        _using_dinov2 = True
        print(f'[Model] DINOv2 loaded from hub on {DEVICE}')
        return _model
    except Exception as e:
        print(f'[Model] Hub load failed: {e}')

    # 3. CV fallback – no deep model
    print('[Model] No deep model available; using CV-based fallback')
    _using_dinov2 = False
    return None


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────

def _cv_features(pil_img: Image.Image) -> np.ndarray:
    """OpenCV HOG-like feature vector (fallback)."""
    img = np.array(pil_img.convert('RGB').resize((224, 224)))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    blks = []
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            b = mag[i:i + 16, j:j + 16]
            blks.extend([b.mean(), b.std()])
    feat = np.array(blks, dtype=np.float32)
    feat /= np.linalg.norm(feat) + 1e-8
    return feat


def extract_features(pil_img: Image.Image) -> np.ndarray:
    model = get_model()
    if model is not None and _using_dinov2:
        with torch.no_grad():
            t = TRANSFORM(pil_img.convert('RGB')).unsqueeze(0).to(DEVICE)
            out = model(t)
        return out.cpu().numpy().flatten()
    return _cv_features(pil_img)


def get_attention_map(pil_img: Image.Image) -> np.ndarray:
    """Returns a 2-D float array [0, 1] representing per-patch saliency."""
    model = get_model()
    if model is not None and _using_dinov2:
        with torch.no_grad():
            t = TRANSFORM(pil_img.convert('RGB')).unsqueeze(0).to(DEVICE)
            # Try forward_features (DINOv2 hub API)
            try:
                out = model.forward_features(t)
                ptok = out['x_norm_patchtokens']   # [1, N, D]
            except Exception:
                # Fallback: get_intermediate_layers
                try:
                    feats = model.get_intermediate_layers(t, n=1)[0]  # [1, N+1, D]
                    if feats.shape[1] > 1:
                        ptok = feats[:, 1:, :]
                    else:
                        ptok = feats
                except Exception:
                    ptok = None

            if ptok is not None:
                mag = ptok[0].norm(dim=-1).cpu().numpy()
                side = int(np.sqrt(mag.shape[0]))
                if side * side == mag.shape[0]:
                    feat_map = mag.reshape(side, side)
                else:
                    feat_map = mag.reshape(1, -1)
                mn, mx = feat_map.min(), feat_map.max()
                return (feat_map - mn) / (mx - mn + 1e-8)

    # CV gradient fallback
    img = np.array(pil_img.convert('RGB').resize((224, 224)))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    fm = np.sqrt(gx ** 2 + gy ** 2)
    fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
    return fm


# ─────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────

def heatmap_overlay_b64(feat_map: np.ndarray,
                        pil_img: Image.Image,
                        alpha: float = 0.5) -> str:
    """Overlay coloured heatmap on original image → base64 PNG."""
    orig = np.array(pil_img.convert('RGB'))
    h, w = orig.shape[:2]
    fm = cv2.resize(feat_map.astype(np.float32), (w, h))
    fm = (fm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(fm, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (orig * (1 - alpha) + colored * alpha).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_b64(pil_img: Image.Image,
               max_size: tuple = (400, 400),
               quality: int = 85) -> str:
    img = pil_img.copy()
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────

def compute_metrics(ok_feats: list, ng_feats: list | None = None) -> dict:
    metrics: dict = {}
    ok_arr = np.array(ok_feats)

    # 1. Data Complexity – intra-class scatter
    if len(ok_arr) > 1:
        centroid = ok_arr.mean(axis=0)
        dists = np.linalg.norm(ok_arr - centroid, axis=1)
        raw = float(dists.mean())
        metrics['data_complexity'] = min(100.0, raw * 100)
        metrics['complexity_std'] = float(dists.std())
    else:
        metrics['data_complexity'] = 50.0
        metrics['complexity_std'] = 0.0

    # 2. Feature Richness – PCA entropy
    if len(ok_arr) >= 3:
        n = min(ok_arr.shape[0] - 1, ok_arr.shape[1], 20)
        pca = PCA(n_components=n)
        pca.fit(ok_arr)
        evr = pca.explained_variance_ratio_
        evr_n = evr / (evr.sum() + 1e-8)
        richness = float(-np.sum(evr_n * np.log(evr_n + 1e-8)) / np.log(n))
        metrics['feature_richness'] = richness * 100
    else:
        metrics['feature_richness'] = 50.0

    # 3. OK/NG Separation (Fisher-like ratio)
    if ng_feats and len(ng_feats) > 0:
        ng_arr = np.array(ng_feats)
        ok_c = ok_arr.mean(axis=0)
        ng_c = ng_arr.mean(axis=0)
        inter = float(np.linalg.norm(ok_c - ng_c))
        ok_intra = float(np.linalg.norm(ok_arr - ok_c, axis=1).mean()) if len(ok_arr) > 1 else 1.0
        ng_intra = float(np.linalg.norm(ng_arr - ng_c, axis=1).mean()) if len(ng_arr) > 1 else 1.0
        sep = inter / (ok_intra + ng_intra + 1e-8)
        metrics['ok_ng_separation'] = min(100.0, sep * 30)
        metrics['has_ng'] = True
        metrics['cosine_sim'] = float(
            cosine_similarity(ok_c.reshape(1, -1), ng_c.reshape(1, -1))[0][0]
        )
    else:
        metrics['ok_ng_separation'] = None
        metrics['has_ng'] = False
        metrics['cosine_sim'] = None

    return metrics


def metrics_chart_b64(metrics: dict, ok_count: int, ng_count: int) -> str:
    fig = plt.figure(figsize=(14, 5), facecolor='#0d0d1a')

    # ── bar chart ─────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor('#161630')
    names = ['Data Complexity', 'Feature Richness']
    vals  = [metrics.get('data_complexity', 0), metrics.get('feature_richness', 0)]
    clrs  = ['#4cc9f0', '#7209b7']
    if metrics.get('has_ng') and metrics.get('ok_ng_separation') is not None:
        names.append('OK/NG Separation')
        vals.append(metrics['ok_ng_separation'])
        clrs.append('#f72585')
    ypos = np.arange(len(names))
    bars = ax1.barh(ypos, vals, color=clrs, alpha=0.85, height=0.5)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(names, color='white', fontsize=10)
    ax1.set_xlim(0, 110)
    ax1.set_xlabel('Score (0-100)', color='#aaaacc', fontsize=9)
    ax1.set_title('Evaluation Metrics', color='white', fontsize=12, pad=10)
    ax1.tick_params(axis='x', colors='#aaaacc')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333355')
    ax1.axvline(50, color='#555577', linestyle='--', alpha=0.4)
    for bar, v in zip(bars, vals):
        ax1.text(min(v + 2, 100), bar.get_y() + bar.get_height() / 2,
                 f'{v:.1f}', va='center', color='white', fontsize=10, fontweight='bold')

    # ── pie chart ─────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor('#161630')
    ax2.set_title('Image Distribution', color='white', fontsize=12, pad=10)
    if ng_count > 0:
        pie_d = [ok_count, ng_count]
        pie_l = [f'OK\n{ok_count}', f'NG\n{ng_count}']
        pie_c = ['#4cc9f0', '#f72585']
    else:
        pie_d = [ok_count]
        pie_l = [f'OK\n{ok_count}']
        pie_c = ['#4cc9f0']
    wedges, txts, atxts = ax2.pie(
        pie_d, labels=pie_l, colors=pie_c,
        autopct='%1.0f%%', startangle=90, pctdistance=0.75,
        wedgeprops={'edgecolor': '#0d0d1a', 'linewidth': 2},
    )
    for t in txts:
        t.set_color('white'); t.set_fontsize(10)
    for at in atxts:
        at.set_color('white'); at.set_fontsize(9)

    # ── summary text ──────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor('#161630')
    ax3.set_title('Summary', color='white', fontsize=12, pad=10)
    ax3.axis('off')
    rows = [
        ('Data Complexity',  metrics.get('data_complexity', 0),  '↑ high = diverse OK'),
        ('Feature Richness', metrics.get('feature_richness', 0), '↑ high = info rich'),
    ]
    if metrics.get('has_ng'):
        rows.append(('OK/NG Separation', metrics.get('ok_ng_separation', 0),
                     '↑ high = easy detect'))
    for i, (lbl, val, hint) in enumerate(rows):
        y = 0.88 - i * 0.30
        col = '#06d6a0' if val > 60 else '#f72585' if val < 30 else '#ffd166'
        ax3.text(0.04, y,       lbl,           color='#aaaacc', fontsize=9,  transform=ax3.transAxes)
        ax3.text(0.04, y - 0.09, f'{val:.1f}', color=col,       fontsize=15, fontweight='bold', transform=ax3.transAxes)
        ax3.text(0.62, y - 0.04, hint,         color='#666688', fontsize=8,  transform=ax3.transAxes)

    plt.tight_layout(pad=2.0)
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight',
                facecolor='#0d0d1a', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────
# Template / ROI store  (in-memory)
# ─────────────────────────────────────────────
_store: dict = {}   # session_id → {features, previews}


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    model = get_model()
    return jsonify({
        'using_dinov2': _using_dinov2,
        'device': str(DEVICE),
        'model_loaded': model is not None,
    })


# ── Feature 1: Evaluate ───────────────────────
@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        ok_files = [f for f in request.files.getlist('ok_images') if f.filename]
        ng_files = [f for f in request.files.getlist('ng_images') if f.filename]

        if not ok_files:
            return jsonify({'error': 'Please upload at least one OK image'}), 400

        ok_feats, ng_feats, visuals = [], [], []

        for f in ok_files[:20]:
            img = Image.open(f.stream).convert('RGB')
            ok_feats.append(extract_features(img))
            fm = get_attention_map(img)
            visuals.append({
                'filename': secure_filename(f.filename),
                'type': 'OK',
                'original': pil_to_b64(img),
                'heatmap': heatmap_overlay_b64(fm, img),
            })

        for f in ng_files[:20]:
            img = Image.open(f.stream).convert('RGB')
            ng_feats.append(extract_features(img))
            fm = get_attention_map(img)
            visuals.append({
                'filename': secure_filename(f.filename),
                'type': 'NG',
                'original': pil_to_b64(img),
                'heatmap': heatmap_overlay_b64(fm, img),
            })

        metrics  = compute_metrics(ok_feats, ng_feats if ng_feats else None)
        chart_b64 = metrics_chart_b64(metrics, len(ok_feats), len(ng_feats))

        # Recommendations
        recs = []
        dc = metrics.get('data_complexity', 50)
        fr = metrics.get('feature_richness', 50)
        sep = metrics.get('ok_ng_separation')

        if dc < 25:
            recs.append({'level': 'good', 'text': '✅ OK samples are visually consistent – good data uniformity.'})
        elif dc > 65:
            recs.append({'level': 'warn', 'text': '⚠️ OK samples show high variation – consider collecting more data.'})
        else:
            recs.append({'level': 'info', 'text': '💡 OK sample diversity is moderate – quality looks normal.'})

        if fr > 60:
            recs.append({'level': 'good', 'text': '✅ Images are information-rich – good for model training.'})
        else:
            recs.append({'level': 'warn', 'text': '⚠️ Low feature richness – may limit model learning capacity.'})

        if sep is not None:
            if sep > 60:
                recs.append({'level': 'good', 'text': '✅ OK/NG feature separation is high – defects should be detectable.'})
            elif sep > 30:
                recs.append({'level': 'info', 'text': '💡 Moderate OK/NG separation – model training may be challenging.'})
            else:
                recs.append({'level': 'warn', 'text': '⚠️ Low OK/NG separation – defect features overlap with normal images.'})

        return jsonify({
            'metrics': metrics,
            'chart': chart_b64,
            'visualizations': visuals,
            'recommendations': recs,
            'summary': {'ok_count': len(ok_feats), 'ng_count': len(ng_feats)},
            'using_dinov2': _using_dinov2,
        })

    except Exception as exc:
        import traceback
        return jsonify({'error': str(exc), 'traceback': traceback.format_exc()}), 500


# ── Feature 2: AI Sorting – set templates ────
@app.route('/api/sort/set-templates', methods=['POST'])
def set_templates():
    try:
        session_id = request.form.get('session_id', 'default')
        files  = [f for f in request.files.getlist('images') if f.filename]
        rois   = json.loads(request.form.get('rois', '[]'))

        if not files:
            return jsonify({'error': 'No template images uploaded'}), 400

        loaded = []
        for f in files:
            img = Image.open(f.stream).convert('RGB')
            loaded.append((secure_filename(f.filename), img))

        cat_feats: dict = {}
        cat_previews: dict = {}

        if rois:
            for roi in rois:
                idx  = roi.get('image_index', 0)
                name = roi.get('name', f'region_{idx}')
                if idx >= len(loaded):
                    continue
                _, img = loaded[idx]
                img_np = np.array(img)
                ih, iw = img_np.shape[:2]
                x = max(0, int(roi.get('x', 0)));  y = max(0, int(roi.get('y', 0)))
                w = int(roi.get('w', iw));          h = int(roi.get('h', ih))
                x = min(x, iw - 1);               y = min(y, ih - 1)
                w = min(w, iw - x);                h = min(h, ih - y)
                if w > 5 and h > 5:
                    crop = Image.fromarray(img_np[y:y + h, x:x + w])
                    cat_feats.setdefault(name, []).append(extract_features(crop))
                    cat_previews[name] = pil_to_b64(crop, max_size=(200, 200))
        else:
            for fname, img in loaded:
                name = os.path.splitext(fname)[0]
                cat_feats[name] = [extract_features(img)]
                cat_previews[name] = pil_to_b64(img, max_size=(200, 200))

        if not cat_feats:
            return jsonify({'error': 'Could not extract valid template features'}), 400

        avg_feats = {n: np.mean(v, axis=0) for n, v in cat_feats.items()}
        _store[session_id] = {'features': avg_feats, 'previews': cat_previews}

        return jsonify({
            'success': True,
            'categories': list(cat_feats.keys()),
            'previews': cat_previews,
        })

    except Exception as exc:
        import traceback
        return jsonify({'error': str(exc), 'traceback': traceback.format_exc()}), 500


# ── Feature 2: AI Sorting – match ────────────
@app.route('/api/sort/match', methods=['POST'])
def match_images():
    try:
        session_id   = request.form.get('session_id', 'default')
        target_files = [f for f in request.files.getlist('images') if f.filename]

        if session_id not in _store:
            return jsonify({'error': 'Templates not set – please configure templates first'}), 400
        if not target_files:
            return jsonify({'error': 'No target images uploaded'}), 400

        tmpl_feats = _store[session_id]['features']
        cats       = list(tmpl_feats.keys())
        results    = {c: [] for c in cats}

        for f in target_files:
            img     = Image.open(f.stream).convert('RGB')
            fname   = secure_filename(f.filename)
            q_feats = extract_features(img)

            sims    = {
                c: float(cosine_similarity(
                    q_feats.reshape(1, -1),
                    tmpl_feats[c].reshape(1, -1)
                )[0][0])
                for c in cats
            }
            best_cat = max(sims, key=sims.get)
            results[best_cat].append({
                'filename':   fname,
                'similarity': sims[best_cat],
                'all_sims':   sims,
                'image':      pil_to_b64(img),
            })

        for c in results:
            results[c].sort(key=lambda x: x['similarity'], reverse=True)

        return jsonify({
            'success':    True,
            'results':    results,
            'categories': cats,
            'previews':   _store[session_id]['previews'],
        })

    except Exception as exc:
        import traceback
        return jsonify({'error': str(exc), 'traceback': traceback.format_exc()}), 500


# ── Feature 2: AI Sorting – download ZIP ────
@app.route('/api/sort/download', methods=['POST'])
def download_zip():
    try:
        data    = request.get_json()
        results = data.get('results', {})

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for category, items in results.items():
                safe = ''.join(
                    c for c in category
                    if c.isalnum() or c in (' ', '-', '_', '.')
                ).strip() or 'category'
                for item in items:
                    img_data = base64.b64decode(item['image'])
                    zf.writestr(f'{safe}/{item["filename"]}', img_data)
        buf.seek(0)
        return send_file(
            buf,
            mimetype='application/zip',
            as_attachment=True,
            download_name='ai_sorting_results.zip',
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('─' * 60)
    print('DINOv3 Industrial Vision Evaluation Server')
    print(f'Device : {DEVICE}')
    print('Initialising model (may take a moment)…')
    get_model()
    print(f'Model  : {"DINOv2/3 deep features" if _using_dinov2 else "OpenCV CV fallback"}')
    print('Server : http://localhost:5000')
    print('─' * 60)
    app.run(debug=False, host='0.0.0.0', port=5000)
