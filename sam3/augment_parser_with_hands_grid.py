#!/usr/bin/env python3
"""
Augment a packed multi-view Graphonomy parser grid by painting a new 'hands' label using SAM3.

Intended to be executed under the SAM3 repo's own virtualenv (sam3_project/.venv),
because SAM3 dependencies are not part of the 4D-DRESS env.

Inputs:
- --render_grid: packed RGB grid image (render-fXXXX.png)
- --parser_grid: packed RGB parser label image (parser-fXXXX.png)

Output:
- --out_parser_grid: packed RGB parser image with hands painted in (default: overwrite)

Packing convention matches AvatarrexDatasetUtils.pack_multi_view_images_static:
- num_cols fixed to 4
- num_rows = ceil(num_views / num_cols) (views may be padded to fill last row)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


LEFT_ARM_COLOR = (51, 170, 221)
RIGHT_ARM_COLOR = (0, 255, 255)
HANDS_COLOR = (255, 0, 255)  # must match 4D-DRESS SURFACE_LABEL_COLOR_EXTENDED hands entry


def find_bbox_from_color(mask_rgb: np.ndarray, color: Tuple[int, int, int], padding: int) -> Tuple[int, int, int, int] | None:
    target = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    matches = np.all(mask_rgb == target, axis=2)
    if not matches.any():
        return None
    ys, xs = np.where(matches)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return x0 - padding, y0 - padding, x1 + padding, y1 + padding


def clamp_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def extract_union_mask_and_max_score(
    processor: Sam3Processor,
    image: Image.Image,
    prompts: Iterable[str],
    score_threshold: float,
) -> tuple[np.ndarray, float]:
    import torch

    state = processor.set_image(image)
    all_masks: list[torch.Tensor] = []
    max_score: float = 0.0

    for prompt in prompts:
        state = processor.set_text_prompt(prompt=prompt, state=state)
        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None or scores is None:
            continue
        if masks.numel() == 0 or scores.numel() == 0:
            continue
        keep = scores > score_threshold
        if keep.any():
            all_masks.append(masks[keep])
            max_score = max(max_score, float(scores[keep].max().item()))

    if not all_masks:
        return np.zeros((image.height, image.width), dtype=np.uint8), 0.0

    masks_cat = torch.cat(all_masks, dim=0)
    if masks_cat.dim() == 4 and masks_cat.shape[1] == 1:
        masks_bool = masks_cat[:, 0] > 0.5
    elif masks_cat.dim() == 3:
        masks_bool = masks_cat > 0.5
    else:
        raise ValueError(f"Unexpected mask shape: {masks_cat.shape}")

    union = torch.any(masks_bool, dim=0)
    mask_np = union.to("cpu").numpy().astype(np.uint8) * 255
    if mask_np.shape != (image.height, image.width):
        mask_np = np.array(
            Image.fromarray(mask_np, mode="L").resize(image.size, Image.NEAREST),
            dtype=np.uint8,
        )
    return mask_np, max_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_grid", type=str, required=True)
    ap.add_argument("--parser_grid", type=str, required=True)
    ap.add_argument("--out_parser_grid", type=str, required=True)
    ap.add_argument("--num_views", type=int, required=True)
    ap.add_argument("--num_cols", type=int, default=4)
    ap.add_argument("--prompt", type=str, default="hand")
    ap.add_argument("--confidence", type=float, default=0.8)
    ap.add_argument("--score_threshold", type=float, default=0.3)
    ap.add_argument("--bbox_padding", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    render_path = Path(args.render_grid)
    parser_path = Path(args.parser_grid)
    out_path = Path(args.out_parser_grid)

    try:
        render_grid = np.array(Image.open(render_path).convert("RGB"), dtype=np.uint8)
        parser_grid = np.array(Image.open(parser_path).convert("RGB"), dtype=np.uint8)
        if render_grid.shape[:2] != parser_grid.shape[:2]:
            raise ValueError(f"render_grid and parser_grid size mismatch: {render_grid.shape} vs {parser_grid.shape}")
    except Exception as e:
        print(f"Error loading render_grid or parser_grid at {render_path} or {parser_path}: {e}")
        return

    num_cols = int(args.num_cols)
    num_views = int(args.num_views)
    # Views may not be divisible by num_cols after camera filtering. The packer pads
    # the last row with blank images to fill a full grid.
    num_rows = int(np.ceil(num_views / float(num_cols)))

    H, W = render_grid.shape[:2]
    if H % num_rows != 0 or W % num_cols != 0:
        raise ValueError(f"Packed grid size {W}x{H} not divisible by cols/rows ({num_cols}, {num_rows})")
    view_h = H // num_rows
    view_w = W // num_cols

    model = build_sam3_image_model(device=args.device)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=float(args.score_threshold))

    out_parser = parser_grid.copy()

    # Process only the real views; ignore padded empty cells in the last row.
    for vi in range(num_views):
        r = vi // num_cols
        c = vi % num_cols
        x0, y0 = c * view_w, r * view_h
        x1, y1 = x0 + view_w, y0 + view_h

        rgb_view = render_grid[y0:y1, x0:x1]
        parser_view = out_parser[y0:y1, x0:x1]

        union_hand = np.zeros((view_h, view_w), dtype=np.uint8)

        for arm_color in (LEFT_ARM_COLOR, RIGHT_ARM_COLOR):
            bbox = find_bbox_from_color(parser_view, arm_color, padding=int(args.bbox_padding))
            if bbox is None:
                continue
            bbox = clamp_bbox(bbox, view_w, view_h)
            if bbox is None:
                continue
            bx0, by0, bx1, by1 = bbox
            crop = rgb_view[by0:by1, bx0:bx1]
            mask_crop, max_score = extract_union_mask_and_max_score(
                processor=processor,
                image=Image.fromarray(crop, mode="RGB"),
                prompts=[args.prompt],
                score_threshold=float(args.score_threshold),
            )
            if max_score >= float(args.confidence):
                union_hand[by0:by1, bx0:bx1] = np.maximum(union_hand[by0:by1, bx0:bx1], mask_crop)

        if union_hand.any():
            parser_view[union_hand > 0] = np.array(HANDS_COLOR, dtype=np.uint8)
            out_parser[y0:y1, x0:x1] = parser_view

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_parser).save(out_path)


if __name__ == "__main__":
    main()










