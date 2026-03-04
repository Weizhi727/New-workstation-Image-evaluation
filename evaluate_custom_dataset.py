"""
GroundingDINO Custom Dataset Evaluation Script
===============================================
Modified from GroundingDINO's demo/test_ap_on_coco.py

Supports evaluation on any COCO-format annotation file with custom categories.
Unlike the original which hardcodes the 80-class COCO id_map, this script
dynamically builds the category-to-token mapping from your dataset.

Requirements:
    pip install groundingdino-py  (or install from source)
    pip install pycocotools

Usage example:
    python evaluate_custom_dataset.py \\
        --config_file path/to/GroundingDINO_SwinT_OGC.py \\
        --checkpoint_path path/to/groundingdino_swint_ogc.pth \\
        --anno_path path/to/annotations.json \\
        --image_dir path/to/images/ \\
        --output results.json

    # Use a custom text prompt instead of auto-generating from categories:
    python evaluate_custom_dataset.py ... --custom_text "cat . dog . car ."

    # Run on CPU:
    python evaluate_custom_dataset.py ... --device cpu
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig
import torchvision

from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    """Load GroundingDINO model from config and checkpoint."""
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CocoDetection(torchvision.datasets.CocoDetection):
    """COCO-format detection dataset.

    Works with any COCO-format JSON file (custom categories are fully supported).
    Converts bounding boxes from xywh to xyxy and filters degenerate boxes.
    """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # target: list of annotation dicts

        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]   # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Filter degenerate boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {
            "image_id": self.ids[idx],
            "boxes": boxes,
            "orig_size": torch.as_tensor([int(h), int(w)]),
        }

        if self._transforms is not None:
            img, target_new = self._transforms(img, target_new)

        return img, target_new


# ---------------------------------------------------------------------------
# Post-processor (generic — works with any category set)
# ---------------------------------------------------------------------------

class PostProcessCustomGrounding(nn.Module):
    """Convert model outputs to COCO API format for custom datasets.

    The key difference from the original PostProcessCocoGrounding is that
    the category → token-span mapping is built dynamically from the dataset
    instead of relying on the hardcoded 80-class COCO id_map.
    """

    def __init__(self, num_select: int = 300, coco_api=None, tokenlizer=None):
        super().__init__()
        self.num_select = num_select

        assert coco_api is not None, "coco_api is required"

        category_dict = coco_api.dataset["categories"]
        cat_list = [item["name"] for item in category_dict]

        # Build text captions and per-category token spans
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]

        # positive_map: (num_cats, 256) — probability mass over token positions
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist
        )

        # Dynamically build index→category_id mapping from the annotation file.
        # This replaces the hardcoded COCO id_map and works for any category IDs.
        id_map = {i: item["id"] for i, item in enumerate(category_dict)}
        max_cat_id = max(id_map.values())

        # new_pos_map: (max_cat_id + 1, 256), indexed by actual category id
        new_pos_map = torch.zeros((max_cat_id + 1, 256))
        for idx, cat_id in id_map.items():
            new_pos_map[cat_id] = positive_map[idx]

        self.positive_map = new_pos_map
        self.max_cat_id = max_cat_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy: bool = False):
        """
        Args:
            outputs:      raw model outputs dict with 'pred_logits' and 'pred_boxes'
            target_sizes: (batch, 2) tensor with original (H, W) for each image
            not_to_xyxy:  if True, skip box conversion (boxes already in xyxy)
        Returns:
            list of dicts with 'scores', 'labels', 'boxes' per image
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        # Map token-space probabilities to label-space: (bs, Q, max_cat_id+1)
        prob_to_token = out_logits.sigmoid()           # (bs, Q, 256)
        pos_maps = self.positive_map.to(prob_to_token.device)
        prob_to_label = prob_to_token @ pos_maps.T     # (bs, Q, max_cat_id+1)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # Top-k selection over all queries × labels
        topk_values, topk_indexes = torch.topk(
            prob_to_label.view(out_logits.shape[0], -1), num_select, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // prob_to_label.shape[2]
        labels = topk_indexes % prob_to_label.shape[2]

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # Rescale from relative [0, 1] to absolute pixel coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_text_prompt(category_dict, custom_text: str = None) -> str:
    """Build the text prompt from category names, or use a user-supplied one."""
    if custom_text:
        return custom_text.strip()
    cat_names = [item["name"] for item in category_dict]
    return " . ".join(cat_names) + " ."


METRIC_NAMES = [
    "AP@[IoU=0.50:0.95]",
    "AP@[IoU=0.50]",
    "AP@[IoU=0.75]",
    "AP@[IoU=0.50:0.95] area=small",
    "AP@[IoU=0.50:0.95] area=medium",
    "AP@[IoU=0.50:0.95] area=large",
    "AR@[IoU=0.50:0.95] maxDets=1",
    "AR@[IoU=0.50:0.95] maxDets=10",
    "AR@[IoU=0.50:0.95] maxDets=100",
    "AR@[IoU=0.50:0.95] area=small",
    "AR@[IoU=0.50:0.95] area=medium",
    "AR@[IoU=0.50:0.95] area=large",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ── Model ──────────────────────────────────────────────────────────────
    cfg = SLConfig.fromfile(args.config_file)

    print(f"[INFO] Loading model from: {args.checkpoint_path}")
    model = load_model(args.config_file, args.checkpoint_path, device=args.device)
    model = model.to(args.device)
    model.eval()

    # ── Dataset ────────────────────────────────────────────────────────────
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"[INFO] Loading annotations from: {args.anno_path}")
    dataset = CocoDetection(args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    category_dict = dataset.coco.dataset["categories"]
    cat_list = [item["name"] for item in category_dict]
    print(f"[INFO] Dataset : {len(dataset)} images | {len(cat_list)} categories")
    print(f"[INFO] Categories: {cat_list}")

    # ── Post-processor ─────────────────────────────────────────────────────
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCustomGrounding(
        num_select=args.num_select,
        coco_api=dataset.coco,
        tokenlizer=tokenlizer,
    )

    # ── Evaluator ──────────────────────────────────────────────────────────
    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True
    )

    # ── Text prompt ────────────────────────────────────────────────────────
    caption = build_text_prompt(category_dict, args.custom_text)
    print(f"[INFO] Text prompt: {caption}\n")

    # ── Inference loop ─────────────────────────────────────────────────────
    all_predictions = []   # for optional JSON export
    start = time.time()

    for i, (images, targets) in enumerate(data_loader):
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        input_captions = [caption] * bs

        with torch.no_grad():
            outputs = model(images, captions=input_captions)

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0
        ).to(images.device)

        results = postprocessor(outputs, orig_target_sizes)

        # Feed to evaluator
        cocogrounding_res = {
            target["image_id"]: output
            for target, output in zip(targets, results)
        }
        evaluator.update(cocogrounding_res)

        # Collect predictions for JSON export (filtered by score_threshold)
        if args.output:
            for target, result in zip(targets, results):
                image_id = (
                    target["image_id"].item()
                    if isinstance(target["image_id"], torch.Tensor)
                    else target["image_id"]
                )
                scores = result["scores"].cpu().numpy()
                labels = result["labels"].cpu().numpy()
                boxes  = result["boxes"].cpu().numpy()
                for score, label, box in zip(scores, labels, boxes):
                    if score >= args.score_threshold:
                        all_predictions.append({
                            "image_id":   int(image_id),
                            "category_id": int(label),
                            # Convert xyxy → xywh for COCO format
                            "bbox":  [float(box[0]), float(box[1]),
                                      float(box[2] - box[0]), float(box[3] - box[1])],
                            "score": float(score),
                        })

        # Progress
        if (i + 1) % 30 == 0 or (i + 1) == len(data_loader):
            elapsed = time.time() - start
            eta = len(data_loader) / (i + 1e-5) * elapsed - elapsed
            print(
                f"  [{i+1:>{len(str(len(data_loader)))}}/{len(data_loader)}] "
                f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s"
            )

    # ── Evaluation summary ─────────────────────────────────────────────────
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.coco_eval["bbox"].stats.tolist()
    print("\n" + "=" * 50)
    print("  Evaluation Results (bbox)")
    print("=" * 50)
    for name, val in zip(METRIC_NAMES, stats):
        print(f"  {name:<40s}: {val:.4f}")
    print("=" * 50)

    # ── Save JSON output ───────────────────────────────────────────────────
    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        output_data = {
            "config": vars(args),
            "caption": caption,
            "categories": category_dict,
            "stats": dict(zip(METRIC_NAMES, stats)),
            "num_predictions": len(all_predictions),
            "predictions": all_predictions,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Results saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "GroundingDINO — Custom Dataset Evaluation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Model arguments ────────────────────────────────────────────────────
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--config_file", "-c", type=str, required=True,
        help="Path to GroundingDINO config file (e.g. GroundingDINO_SwinT_OGC.py)",
    )
    model_group.add_argument(
        "--checkpoint_path", "-p", type=str, required=True,
        help="Path to model checkpoint (.pth file)",
    )
    model_group.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on: 'cuda' or 'cpu'  (default: cuda)",
    )

    # ── Dataset arguments ──────────────────────────────────────────────────
    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument(
        "--anno_path", type=str, required=True,
        help="Path to COCO-format annotation JSON file (instances_*.json)",
    )
    data_group.add_argument(
        "--image_dir", type=str, required=True,
        help="Root directory containing the dataset images",
    )
    data_group.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of DataLoader worker processes  (default: 4)",
    )

    # ── Inference arguments ────────────────────────────────────────────────
    infer_group = parser.add_argument_group("Inference")
    infer_group.add_argument(
        "--num_select", type=int, default=300,
        help="Top-k predictions to keep per image  (default: 300)",
    )
    infer_group.add_argument(
        "--score_threshold", type=float, default=0.0,
        help="Min score for predictions saved to JSON output  (default: 0.0)",
    )
    infer_group.add_argument(
        "--custom_text", type=str, default=None,
        help=(
            "Override auto-generated text prompt. "
            "Use '. ' between categories, e.g. 'cat . dog . car .' "
            "(default: auto-built from annotation categories)"
        ),
    )

    # ── Output arguments ───────────────────────────────────────────────────
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save evaluation results (metrics + predictions) to a JSON file",
    )

    args = parser.parse_args()
    main(args)
