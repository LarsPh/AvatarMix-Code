import argparse
import csv
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import trimesh
import pickle
from loguru import logger
from pytorch3d.ops import knn_points
import pytorch3d.io
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

from src.utils.smplx_utils.smplx_utils import (
    create_smplx_model_for_neus2_reposing,
    load_smplx_J_regressor_body25_smplx,
)
from src.utils.smplx_utils.smplx_utils import get_smplx_model_path
from src.utils.smplx_utils.smplx_ani.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES

try:
    import reshape_ops
except Exception:
    reshape_ops = None


def _load_calibration(calib_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    with open(calib_path, "r") as f:
        d = json.load(f)
    out = {}
    for cam_id, cam in d.items():
        out[cam_id] = {
            "K": np.array(cam["K"], dtype=np.float32),
            "R": np.array(cam["R"], dtype=np.float32),
            "T": np.array(cam["T"], dtype=np.float32).reshape(3, 1),
            "imgSize": np.array(cam["imgSize"], dtype=np.int32),
        }
    return out


def _load_keypoints_for_subject(subject_dir: Path) -> Dict[str, np.ndarray]:

    out: Dict[str, np.ndarray] = {}
    for cam_dir in subject_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        cam_id = cam_dir.name
        kps_path = cam_dir / "keypoints_2d" / "0000_openpose_body25.json"
        if not kps_path.exists():
            continue
        with open(kps_path, "r") as f:
            payload = json.load(f)
        kps = np.array(payload["keypoints"], dtype=np.float32)
        if kps.shape != (25, 3):
            logger.warning(f"{kps_path} has shape {kps.shape}, expected (25,3); skipping camera {cam_id}")
            continue
        out[cam_id] = kps
    return out


def _load_keypoints_payload_for_camera(subject_dir: Path, cam_id: str) -> Dict:
    kps_path = subject_dir / cam_id / "keypoints_2d" / "0000_openpose_body25.json"
    with open(kps_path, "r") as f:
        return json.load(f)


def _coco17_to_openpose_body25(kps_xy_17: np.ndarray, kps_conf_17: np.ndarray) -> np.ndarray:

    if kps_xy_17.shape != (17, 2):
        raise ValueError(f"Expected kps_xy_17 shape (17,2), got {kps_xy_17.shape}")
    if kps_conf_17.shape != (17,):
        raise ValueError(f"Expected kps_conf_17 shape (17,), got {kps_conf_17.shape}")

    out = np.zeros((25, 3), dtype=np.float32)

    def _set(body25_idx: int, coco_idx: int):
        c = float(kps_conf_17[coco_idx])
        if c <= 0.0:
            return
        out[body25_idx, 0] = float(kps_xy_17[coco_idx, 0])
        out[body25_idx, 1] = float(kps_xy_17[coco_idx, 1])
        out[body25_idx, 2] = c


    _set(0, 0)
    _set(15, 2)
    _set(16, 1)
    _set(17, 4)
    _set(18, 3)

    _set(5, 5)
    _set(6, 7)
    _set(7, 9)
    _set(2, 6)
    _set(3, 8)
    _set(4, 10)

    _set(12, 11)
    _set(13, 13)
    _set(14, 15)
    _set(9, 12)
    _set(10, 14)
    _set(11, 16)


    c_ls, c_rs = float(out[5, 2]), float(out[2, 2])
    if c_ls > 0.0 and c_rs > 0.0:
        out[1, 0:2] = 0.5 * (out[5, 0:2] + out[2, 0:2])
        out[1, 2] = min(c_ls, c_rs)
    elif c_ls > 0.0:
        out[1, 0:2] = out[5, 0:2]
        out[1, 2] = 0.5 * c_ls
    elif c_rs > 0.0:
        out[1, 0:2] = out[2, 0:2]
        out[1, 2] = 0.5 * c_rs


    c_lh, c_rh = float(out[12, 2]), float(out[9, 2])
    if c_lh > 0.0 and c_rh > 0.0:
        out[8, 0:2] = 0.5 * (out[12, 0:2] + out[9, 0:2])
        out[8, 2] = min(c_lh, c_rh)
    elif c_lh > 0.0:
        out[8, 0:2] = out[12, 0:2]
        out[8, 2] = 0.5 * c_lh
    elif c_rh > 0.0:
        out[8, 0:2] = out[9, 0:2]
        out[8, 2] = 0.5 * c_rh


    return out


def _draw_detected_keypoints(
    rgb_bgr: np.ndarray,
    kps25: np.ndarray,
    conf_threshold: float,
    title: str | None = None,
    draw_conf: bool = True,
    show_low_conf: bool = True,
    masked_out: np.ndarray | None = None,
) -> np.ndarray:

    img = rgb_bgr.copy()


    col_head = (0, 255, 255)
    col_upper = (0, 200, 0)
    col_lower = (255, 128, 0)
    col_core = (255, 255, 255)

    def color_for_idx(i: int) -> Tuple[int, int, int]:
        if i in (0, 15, 16, 17, 18):
            return col_head
        if i in (1, 8):
            return col_core
        if i in (2, 3, 4, 5, 6, 7):
            return col_upper
        return col_lower

    for i in range(25):
        x, y, c = float(kps25[i, 0]), float(kps25[i, 1]), float(kps25[i, 2])
        col = color_for_idx(i)
        xi, yi = int(round(x)), int(round(y))
        if c >= float(conf_threshold):
            cv2.circle(img, (xi, yi), 4, col, -1, lineType=cv2.LINE_AA)
        else:
            if not show_low_conf:
                continue

            col2 = (int(col[0] * 0.6), int(col[1] * 0.6), int(col[2] * 0.6))
            cv2.circle(img, (xi, yi), 4, col2, 1, lineType=cv2.LINE_AA)

        if masked_out is not None and bool(masked_out[i]):
            cv2.drawMarker(
                img,
                (xi, yi),
                (0, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        if draw_conf:

            txt = f"{i}:{c:.2f}"
            cv2.putText(img, txt, (xi + 4, yi - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)

    if title:
        cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if draw_conf:
        cv2.putText(
            img,
            f"conf_thr={float(conf_threshold):.2f}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
    return img


def _ultralytics_device_arg(device: str) -> str | int:

    d = str(device).strip().lower()
    if d in {"cpu"}:
        return "cpu"
    if d.startswith("cuda"):

        if ":" in d:
            return int(d.split(":")[-1])
        return 0

    return str(device)


def _auto_detect_and_write_keypoints_yolo_pose(
    *,
    subject_dir: Path,
    cam_ids: List[str],
    device: str,
    vis_root_dir: Path,
    keypoints_mode: str,
    conf_threshold: float,
    yolo_pose_model: str,
    yolo_det_conf: float,
    yolo_det_iou: float,
    yolo_imgsz: int,
    overwrite_existing: bool = False,
) -> Dict[str, np.ndarray]:

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "lambda_2d>0 but keypoints are missing, and Ultralytics is not available. "
            "Install with `pip install ultralytics` (plus a matching torch/cuda stack), "
            f"or set lambda_2d=0. Original import error: {e}"
        )

    model = YOLO(str(yolo_pose_model))
    dev_arg = _ultralytics_device_arg(device)

    out: Dict[str, np.ndarray] = {}
    vis_dir = vis_root_dir / "keypoints2d"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for cam_id in cam_ids:
        cam_dir = subject_dir / cam_id
        img_path = cam_dir / "0000.jpg"
        if not img_path.exists():
            continue

        kps_path = cam_dir / "keypoints_2d" / "0000_openpose_body25.json"
        if (not overwrite_existing) and kps_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]


        mask01 = None
        mpath = cam_dir / "mask" / "pha" / "0000.png"
        if mpath.exists():
            m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
            if m is not None:
                if m.ndim == 3:
                    m = m[..., 0]
                if m.shape[0] != h or m.shape[1] != w:
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask01 = (m.astype(np.float32) > 127.0).astype(np.uint8)


        try:
            pred = model.predict(
                source=img,
                verbose=False,
                device=dev_arg,
                conf=float(yolo_det_conf),
                iou=float(yolo_det_iou),
                imgsz=int(yolo_imgsz),
            )
        except TypeError:

            pred = model.predict(source=img, verbose=False, device=dev_arg, conf=float(yolo_det_conf))

        kps25 = np.zeros((25, 3), dtype=np.float32)
        masked_out = np.zeros((25,), dtype=bool)
        try:
            if pred:
                r0 = pred[0]
                if getattr(r0, "keypoints", None) is not None and len(r0.keypoints) > 0:

                    best_i = 0
                    if getattr(r0.keypoints, "conf", None) is not None:

                        confs = r0.keypoints.conf.detach().float().cpu().numpy()

                        sumc = confs.sum(axis=1)
                        cnt = (confs > 0.1).sum(axis=1).astype(np.float32)
                        score = sumc + 0.25 * cnt
                        if score.size > 0:
                            best_i = int(np.argmax(score))
                    elif getattr(r0, "boxes", None) is not None and getattr(r0.boxes, "conf", None) is not None:
                        scores = r0.boxes.conf.detach().float().cpu().numpy()
                        best_i = int(np.argmax(scores)) if scores.size > 0 else 0

                    xy17 = r0.keypoints.xy[best_i].detach().float().cpu().numpy()
                    if getattr(r0.keypoints, "conf", None) is not None:
                        c17 = r0.keypoints.conf[best_i].detach().float().cpu().numpy()
                    else:
                        c17 = np.ones((17,), dtype=np.float32)
                    kps25 = _coco17_to_openpose_body25(xy17.astype(np.float32), c17.astype(np.float32))
        except Exception:

            kps25 = np.zeros((25, 3), dtype=np.float32)


        if mask01 is not None:
            for i in range(25):
                c = float(kps25[i, 2])
                if c <= 0.0:
                    continue
                x, y = float(kps25[i, 0]), float(kps25[i, 1])
                xi, yi = int(round(x)), int(round(y))
                if xi < 0 or yi < 0 or xi >= int(w) or yi >= int(h):
                    kps25[i, 2] = 0.0
                    masked_out[i] = True
                    continue
                if int(mask01[yi, xi]) == 0:
                    kps25[i, 2] = 0.0
                    masked_out[i] = True


        kps_dir = cam_dir / "keypoints_2d"
        kps_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "openpose_body25",
            "source": "ultralytics_yolo_pose",
            "source_format": "coco17",
            "model": str(yolo_pose_model),
            "image_size": [int(w), int(h)],
            "keypoints_mode": str(keypoints_mode),
            "conf_threshold": float(conf_threshold),

            "use_mask": [1] * 25,
            "recommended_use_mask": [1] * 25,
            "keypoints": kps25.tolist(),
        }
        with open(kps_path, "w") as f:
            json.dump(payload, f, indent=2)


        vis_img = _draw_detected_keypoints(
            rgb_bgr=img,
            kps25=kps25,
            conf_threshold=float(conf_threshold),
            title=f"{cam_id} YOLO-pose -> Body25",
            draw_conf=True,
            show_low_conf=True,
            masked_out=masked_out,
        )
        cv2.imwrite(str(vis_dir / f"{cam_id}_0000.jpg"), vis_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        out[cam_id] = kps25

    return out


def _apply_mask_filter_to_keypoints(
    *,
    subject_dir: Path,
    cam_ids: List[str],
    kps_by_cam: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:

    for cam_id in cam_ids:
        if cam_id not in kps_by_cam:
            continue
        cam_dir = subject_dir / cam_id
        img_path = cam_dir / "0000.jpg"
        mpath = cam_dir / "mask" / "pha" / "0000.png"
        if (not img_path.exists()) or (not mpath.exists()):
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        mask01 = (m.astype(np.float32) > 127.0).astype(np.uint8)

        kps = np.array(kps_by_cam[cam_id], dtype=np.float32, copy=True)
        for i in range(25):
            c = float(kps[i, 2])
            if c <= 0.0:
                continue
            x, y = float(kps[i, 0]), float(kps[i, 1])
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or yi < 0 or xi >= int(w) or yi >= int(h) or int(mask01[yi, xi]) == 0:
                kps[i, 2] = 0.0
        kps_by_cam[cam_id] = kps
    return kps_by_cam


def _save_keypoints2d_overlays(
    *,
    subject_dir: Path,
    cam_ids: List[str],
    kps_by_cam: Dict[str, np.ndarray],
    conf_threshold: float,
    out_dir: Path,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    for cam_id in cam_ids:
        if cam_id not in kps_by_cam:
            continue
        cam_dir = subject_dir / cam_id
        img_path = cam_dir / "0000.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]


        masked_out = np.zeros((25,), dtype=bool)
        mpath = cam_dir / "mask" / "pha" / "0000.png"
        if mpath.exists():
            m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
            if m is not None:
                if m.ndim == 3:
                    m = m[..., 0]
                if m.shape[0] != h or m.shape[1] != w:
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask01 = (m.astype(np.float32) > 127.0).astype(np.uint8)
                kps = np.array(kps_by_cam[cam_id], dtype=np.float32)
                for i in range(25):
                    c = float(kps[i, 2])
                    if c <= 0.0:
                        continue
                    x, y = float(kps[i, 0]), float(kps[i, 1])
                    xi, yi = int(round(x)), int(round(y))
                    if xi < 0 or yi < 0 or xi >= int(w) or yi >= int(h) or int(mask01[yi, xi]) == 0:
                        masked_out[i] = True

        vis_img = _draw_detected_keypoints(
            rgb_bgr=img,
            kps25=np.array(kps_by_cam[cam_id], dtype=np.float32),
            conf_threshold=float(conf_threshold),
            title=f"{cam_id} keypoints",
            draw_conf=True,
            show_low_conf=True,
            masked_out=masked_out,
        )
        cv2.imwrite(str(out_dir / f"{cam_id}_0000.jpg"), vis_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def _stable_body25_indices() -> List[int]:

    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def _face_body25_indices() -> List[int]:

    return [0, 15, 16, 17, 18]


def _load_scan_labels_extended(label_pkl_path: Path) -> np.ndarray:
    d = pickle.load(open(label_pkl_path, "rb"))
    if not isinstance(d, dict) or "scan_labels" not in d:
        raise ValueError(f"Unexpected label pkl format at {label_pkl_path}. Expected dict with key 'scan_labels'.")
    labels = np.asarray(d["scan_labels"])
    if labels.ndim != 1:
        raise ValueError(f"scan_labels must be 1D, got shape {labels.shape} in {label_pkl_path}")


    if labels.min() < 0 or labels.max() > 8:
        logger.warning(f"scan_labels range looks unexpected: min={labels.min()} max={labels.max()} (expected 0..8)")
    else:

        try:
            uniq = np.unique(labels)
            logger.info(f"scan_labels unique values: {uniq.tolist()}")
        except Exception:
            pass
    return labels.astype(np.int64)


def _sample_points(x: torch.Tensor, max_n: int) -> torch.Tensor:
    if max_n <= 0 or x.shape[0] <= max_n:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:max_n]
    return x.index_select(0, idx)


def _closest_point_on_triangles(q: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:


    ab = v1 - v0
    ac = v2 - v0
    ap = q[:, None, :] - v0

    d1 = (ab * ap).sum(dim=-1)
    d2 = (ac * ap).sum(dim=-1)

    mask_v0 = (d1 <= 0) & (d2 <= 0)

    bp = q[:, None, :] - v1
    d3 = (ab * bp).sum(dim=-1)
    d4 = (ac * bp).sum(dim=-1)

    mask_v1 = (d3 >= 0) & (d4 <= d3)

    vc = d1 * d4 - d3 * d2

    mask_e01 = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v = d1 / (d1 - d3 + 1e-8)

    cp = q[:, None, :] - v2
    d5 = (ab * cp).sum(dim=-1)
    d6 = (ac * cp).sum(dim=-1)

    mask_v2 = (d6 >= 0) & (d5 <= d6)

    vb = d5 * d2 - d1 * d6

    mask_e02 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w = d2 / (d2 - d6 + 1e-8)

    va = d3 * d6 - d5 * d4

    mask_e12 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-8)


    mask_face = ~(mask_v0 | mask_v1 | mask_v2 | mask_e01 | mask_e02 | mask_e12)
    denom = (va + vb + vc + 1e-8)
    v_face = vb / denom
    w_face = vc / denom

    p = torch.empty_like(v0)

    p[mask_v0] = v0[mask_v0]

    p[mask_v1] = v1[mask_v1]

    p[mask_v2] = v2[mask_v2]

    p[mask_e01] = v0[mask_e01] + v[mask_e01][..., None] * ab[mask_e01]

    p[mask_e02] = v0[mask_e02] + w[mask_e02][..., None] * ac[mask_e02]

    p[mask_e12] = v1[mask_e12] + w2[mask_e12][..., None] * (v2 - v1)[mask_e12]

    p[mask_face] = v0[mask_face] + v_face[mask_face][..., None] * ab[mask_face] + w_face[mask_face][..., None] * ac[mask_face]

    dist2 = ((q[:, None, :] - p) ** 2).sum(dim=-1)
    return p, dist2


def _body_pose_joint_names() -> List[str]:

    return list(SMPLX_JOINT_NAMES[1:22])


def _make_body_pose_mask(
    device: torch.device,
    freeze_wrist: bool,
    freeze_feet: bool,
) -> torch.Tensor:

    names = _body_pose_joint_names()
    assert len(names) == 21, f"Unexpected body_pose joint count: {len(names)}"
    mask = torch.ones((1, 63), device=device, dtype=torch.float32)

    def freeze_joint(jname: str):
        if jname not in names:
            logger.warning(f"Joint {jname} not in body_pose joint list; cannot freeze it.")
            return
        j = names.index(jname)
        mask[:, j * 3 : (j + 1) * 3] = 0.0

    if freeze_wrist:
        freeze_joint("left_wrist")
        freeze_joint("right_wrist")

    if freeze_feet:

        for jn in ["left_ankle", "right_ankle", "left_foot", "right_foot"]:
            freeze_joint(jn)

    return mask

def _chamfer_bidirectional(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    if a.numel() == 0 or b.numel() == 0:
        return torch.tensor(0.0, device=a.device)
    da = knn_points(a[None], b[None], K=1).dists[0, :, 0]
    db = knn_points(b[None], a[None], K=1).dists[0, :, 0]
    return da.mean() + db.mean()


def _axis_angle_to_R_t(axis_angle: torch.Tensor) -> torch.Tensor:

    aa = axis_angle.detach().cpu().numpy().astype(np.float32).reshape(3)
    R, _ = cv2.Rodrigues(aa)
    return torch.from_numpy(R.astype(np.float32))


def _apply_external_rigid_transform(
    X_canon: torch.Tensor,
    global_orient: torch.Tensor,
    transl: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:

    assert X_canon.shape[-1] == 3
    R = _axis_angle_to_R_t(global_orient.squeeze(0)).to(X_canon.device)
    X = (X_canon @ R.T) * scale.view(1, 1) + transl.view(1, 3)
    return X


def _project_points(
    X_world: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
) -> torch.Tensor:

    Xw = X_world.t().contiguous()
    Xc = (R @ Xw) + T
    x = K @ Xc
    u = x[0] / (x[2].clamp_min(1e-6))
    v = x[1] / (x[2].clamp_min(1e-6))
    return torch.stack([u, v], dim=-1)


def _huber(x: torch.Tensor, delta: float = 5.0) -> torch.Tensor:
    absx = x.abs()
    quad = torch.minimum(absx, torch.tensor(delta, device=x.device, dtype=x.dtype))
    lin = absx - quad
    return 0.5 * quad**2 + delta * lin


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _alpha_blend_bgr(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    base_f = base.astype(np.float32)
    ov_f = overlay.astype(np.float32)
    out = base_f * (1.0 - alpha) + ov_f * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _export_two_meshes_colored_ply(
    out_path: Path,
    verts_a: torch.Tensor,
    faces_a: torch.Tensor,
    color_a_rgb: Tuple[int, int, int],
    verts_b: torch.Tensor,
    faces_b: torch.Tensor,
    color_b_rgb: Tuple[int, int, int],
) -> None:

    va = verts_a.detach().cpu().numpy().astype(np.float32)
    fa = faces_a.detach().cpu().numpy().astype(np.int64)
    vb = verts_b.detach().cpu().numpy().astype(np.float32)
    fb = faces_b.detach().cpu().numpy().astype(np.int64)

    ca = np.tile(np.array(color_a_rgb, dtype=np.uint8)[None, :], (va.shape[0], 1))
    cb = np.tile(np.array(color_b_rgb, dtype=np.uint8)[None, :], (vb.shape[0], 1))

    v = np.concatenate([va, vb], axis=0)
    c = np.concatenate([ca, cb], axis=0)
    f = np.concatenate([fa, fb + va.shape[0]], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {v.shape[0]}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("property uchar red\n")
        fp.write("property uchar green\n")
        fp.write("property uchar blue\n")
        fp.write(f"element face {f.shape[0]}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        for i in range(v.shape[0]):
            fp.write(
                f"{v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f} {int(c[i,0])} {int(c[i,1])} {int(c[i,2])}\n"
            )
        for i in range(f.shape[0]):
            fp.write(f"3 {int(f[i,0])} {int(f[i,1])} {int(f[i,2])}\n")


def _export_single_mesh_colored_ply(
    out_path: Path,
    verts: torch.Tensor,
    faces: torch.Tensor,
    color_rgb: Tuple[int, int, int],
) -> None:
    v = verts.detach().cpu().numpy().astype(np.float32)
    f = faces.detach().cpu().numpy().astype(np.int64)
    c = np.tile(np.array(color_rgb, dtype=np.uint8)[None, :], (v.shape[0], 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {v.shape[0]}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("property uchar red\n")
        fp.write("property uchar green\n")
        fp.write("property uchar blue\n")
        fp.write(f"element face {f.shape[0]}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        for i in range(v.shape[0]):
            fp.write(f"{v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f} {int(c[i,0])} {int(c[i,1])} {int(c[i,2])}\n")
        for i in range(f.shape[0]):
            fp.write(f"3 {int(f[i,0])} {int(f[i,1])} {int(f[i,2])}\n")


def _export_colored_pointcloud_ply(
    out_path: Path,
    points: np.ndarray,
    colors_rgb: np.ndarray,
) -> None:

    assert points.shape[0] == colors_rgb.shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {points.shape[0]}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("property uchar red\n")
        fp.write("property uchar green\n")
        fp.write("property uchar blue\n")
        fp.write("end_header\n")
        for i in range(points.shape[0]):
            x, y, z = points[i]
            r, g, b = colors_rgb[i]
            fp.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _parse_part_weight_spec(spec: str) -> Dict[str, float]:

    out: Dict[str, float] = {}
    s = (spec or "").strip()
    if not s:
        return out
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            k, v = tok.split(":", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            out[k] = float(v)
        else:
            out[tok] = 1.0
    return out


def _export_lines_obj(
    out_path: Path,
    a: np.ndarray,
    b: np.ndarray,
) -> None:

    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 3
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = a.shape[0]
    with open(out_path, "w") as fp:
        fp.write("# collision debug lines: first N vertices are query points, next N are closest points\n")
        for i in range(n):
            fp.write(f"v {a[i,0]:.6f} {a[i,1]:.6f} {a[i,2]:.6f}\n")
        for i in range(n):
            fp.write(f"v {b[i,0]:.6f} {b[i,1]:.6f} {b[i,2]:.6f}\n")

        for i in range(n):
            fp.write(f"l {i+1} {n+i+1}\n")


def _export_collision_debug_scene_ply(
    out_path: Path,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    query_pts: np.ndarray,
    closest_pts: np.ndarray,
    cyl_radius: float = 0.002,
    cyl_sections: int = 8,
    sphere_radius: float = 0.004,
    add_endpoint_spheres: bool = True,
) -> None:

    assert mesh_verts.ndim == 2 and mesh_verts.shape[1] == 3
    assert mesh_faces.ndim == 2 and mesh_faces.shape[1] == 3
    assert query_pts.shape == closest_pts.shape and query_pts.ndim == 2 and query_pts.shape[1] == 3

    out_path.parent.mkdir(parents=True, exist_ok=True)


    target = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces, process=False)
    target.visual.vertex_colors = np.tile(np.array([160, 160, 160, 255], dtype=np.uint8)[None, :], (mesh_verts.shape[0], 1))


    cylinders = []
    for a, b in zip(query_pts, closest_pts):
        seg = np.stack([a, b], axis=0)
        cyl = trimesh.creation.cylinder(radius=float(cyl_radius), segment=seg, sections=int(cyl_sections))
        cyl.visual.vertex_colors = np.tile(np.array([255, 0, 0, 255], dtype=np.uint8)[None, :], (len(cyl.vertices), 1))
        cylinders.append(cyl)

    spheres = []
    if add_endpoint_spheres and sphere_radius > 0:

        for p in query_pts:
            sp = trimesh.creation.icosphere(subdivisions=2, radius=float(sphere_radius))
            sp.apply_translation(p)
            sp.visual.vertex_colors = np.tile(np.array([0, 0, 255, 255], dtype=np.uint8)[None, :], (len(sp.vertices), 1))
            spheres.append(sp)
        for p in closest_pts:
            sp = trimesh.creation.icosphere(subdivisions=2, radius=float(sphere_radius))
            sp.apply_translation(p)
            sp.visual.vertex_colors = np.tile(np.array([0, 255, 0, 255], dtype=np.uint8)[None, :], (len(sp.vertices), 1))
            spheres.append(sp)

    scene_meshes = [target] + cylinders + spheres if cylinders else [target] + spheres
    combined = trimesh.util.concatenate(scene_meshes)
    combined.export(str(out_path))


def _pack_grid(images_bgr: List[np.ndarray], rows: int, cols: int, pad: int = 8, bg: int = 30) -> np.ndarray:
    assert len(images_bgr) <= rows * cols
    if not images_bgr:
        raise ValueError("No images to pack")
    h, w = images_bgr[0].shape[:2]
    for im in images_bgr:
        if im.shape[:2] != (h, w):
            raise ValueError("All tiles must have same H,W")
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), bg, dtype=np.uint8)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(images_bgr):
                break
            y0 = pad + r * (h + pad)
            x0 = pad + c * (w + pad)
            canvas[y0 : y0 + h, x0 : x0 + w] = images_bgr[idx]
            idx += 1
    return canvas


def _draw_keypoints_overlay(
    rgb_bgr: np.ndarray,
    gt_kps: np.ndarray,
    pred_xy: np.ndarray,
    conf_threshold: float,
    show_indices: List[int],
    title: str | None = None,
    draw_errors: bool = False,
) -> np.ndarray:

    img = rgb_bgr.copy()


    col_head = (0, 255, 255)
    col_upper = (0, 200, 0)
    col_lower = (255, 128, 0)
    col_core = (255, 255, 255)

    def color_for_idx(i: int) -> Tuple[int, int, int]:
        if i in (0, 15, 16, 17, 18):
            return col_head
        if i in (1, 8):
            return col_core
        if i in (2, 3, 4, 5, 6, 7):
            return col_upper
        return col_lower

    for i in show_indices:
        xg, yg, cg = float(gt_kps[i, 0]), float(gt_kps[i, 1]), float(gt_kps[i, 2])
        if cg < conf_threshold:
            continue
        c = color_for_idx(i)
        cv2.circle(img, (int(round(xg)), int(round(yg))), 4, c, -1, lineType=cv2.LINE_AA)

        xp, yp = float(pred_xy[i, 0]), float(pred_xy[i, 1])
        cv2.drawMarker(
            img,
            (int(round(xp)), int(round(yp))),
            c,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=10,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        if draw_errors:
            cv2.line(
                img,
                (int(round(xg)), int(round(yg))),
                (int(round(xp)), int(round(yp))),
                c,
                1,
                lineType=cv2.LINE_AA,
            )

    if title:
        cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _build_p3d_cameras_from_calib(
    calib: Dict[str, Dict[str, np.ndarray]],
    cam_ids: List[str],
    device: torch.device,
    image_size_hw: Tuple[int, int],
):

    Ks = []
    Rs = []
    Ts = []
    for cam_id in cam_ids:
        cam = calib[cam_id]


        K_np = np.array(cam["K"], dtype=np.float32).copy()
        img_size_src = cam.get("imgSize", np.array([1024, 1024], dtype=np.int32))

        src_w = float(img_size_src[0])
        src_h = float(img_size_src[1])
        dst_h, dst_w = float(image_size_hw[0]), float(image_size_hw[1])
        sx = dst_w / max(1.0, src_w)
        sy = dst_h / max(1.0, src_h)
        K_np[0, 0] *= sx
        K_np[0, 2] *= sx
        K_np[1, 1] *= sy
        K_np[1, 2] *= sy
        Ks.append(torch.from_numpy(K_np).float())
        Rs.append(torch.from_numpy(cam["R"]).float())
        Ts.append(torch.from_numpy(cam["T"].reshape(3)).float())
    K = torch.stack(Ks, dim=0).to(device)
    R = torch.stack(Rs, dim=0).to(device)
    tvec = torch.stack(Ts, dim=0).to(device)
    image_size = torch.tensor([list(image_size_hw)], dtype=torch.float32, device=device).repeat(len(cam_ids), 1)
    cameras = cameras_from_opencv_projection(R=R, tvec=tvec, camera_matrix=K, image_size=image_size)
    return cameras


@torch.no_grad()
def _render_silhouette_masks(
    verts_world: torch.Tensor,
    faces: torch.Tensor,
    cameras,
    image_size_hw: Tuple[int, int],
) -> torch.Tensor:

    if verts_world.ndim != 2:
        raise ValueError("verts_world must be (V,3)")
    if faces.dtype != torch.int64:
        faces = faces.to(torch.int64)
    mesh = Meshes(verts=[verts_world], faces=[faces]).extend(len(cameras))
    raster_settings = RasterizationSettings(
        image_size=image_size_hw,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh)
    sil = (fragments.pix_to_face[..., 0] >= 0).float()
    return sil


def _render_soft_silhouette_alpha(
    verts_world: torch.Tensor,
    faces: torch.Tensor,
    cameras,
    image_size_hw: Tuple[int, int],
    sigma: float = 1e-4,
    gamma: float = 1e-4,
    faces_per_pixel: int = 50,
) -> torch.Tensor:

    if verts_world.ndim != 2:
        raise ValueError("verts_world must be (V,3)")
    if faces.dtype != torch.int64:
        faces = faces.to(torch.int64)

    blur_radius = float(np.log(1.0 / 1e-4 - 1.0) * float(sigma))
    raster_settings = RasterizationSettings(
        image_size=image_size_hw,
        blur_radius=blur_radius,
        faces_per_pixel=int(faces_per_pixel),
        bin_size=0,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=float(sigma), gamma=float(gamma)))
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    mesh = Meshes(verts=[verts_world], faces=[faces]).extend(len(cameras))
    rgba = renderer(mesh)
    return rgba[..., 3].clamp(0.0, 1.0)


def _load_gt_silhouette_masks(
    subject_dir: Path,
    cam_ids: List[str],
    image_size_hw: Tuple[int, int],
) -> Dict[str, torch.Tensor]:

    H, W = int(image_size_hw[0]), int(image_size_hw[1])
    out: Dict[str, torch.Tensor] = {}
    for cam_id in cam_ids:
        p = subject_dir / cam_id / "mask" / "pha" / "0000.png"
        if not p.exists():
            continue
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[0] != H or m.shape[1] != W:
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mb = (m.astype(np.float32) > 127.0).astype(np.float32)
        out[cam_id] = torch.from_numpy(mb)
    return out


def _save_visualization_grids(
    subject_dir: Path,
    stage_name: str,
    vis_root_name: str,
    vis_sil_loss_mode: str,
    cam_ids_all: List[str],
    calib: Dict[str, Dict[str, np.ndarray]],
    kps_by_cam: Dict[str, np.ndarray],
    smpl_proj_by_cam: Dict[str, np.ndarray],
    per_cam_err: Dict[str, float],
    conf_threshold: float,
    show_indices: List[int],
    dataset_mesh_verts: torch.Tensor | None,
    dataset_mesh_faces: torch.Tensor | None,
    smpl_mesh_verts: torch.Tensor | None,
    smpl_mesh_faces: torch.Tensor | None,
    neus_mesh_verts: torch.Tensor | None,
    neus_mesh_faces: torch.Tensor | None,
    vis_inside_margin: float = 0.0,
    vis_n_best: int = 12,
    vis_n_worst: int = 12,
    vis_cols: int = 4,
    vis_render_size: int = 512,
    vis_draw_errors: bool = True,
    vis_save_tiles: bool = False,
    vis_save_mesh_ply: bool = False,
) -> None:
    out_dir = subject_dir / vis_root_name
    out_dir.mkdir(parents=True, exist_ok=True)


    if per_cam_err:
        cam_sorted = sorted(cam_ids_all, key=lambda c: per_cam_err.get(c, 1e9))
    else:
        cam_sorted = list(cam_ids_all)
    best = cam_sorted[: min(vis_n_best, len(cam_sorted))]
    worst = list(reversed(cam_sorted))[: min(vis_n_worst, len(cam_sorted))]

    def make_tiles(cam_ids: List[str], tag: str) -> List[np.ndarray]:
        tiles: List[np.ndarray] = []
        if not kps_by_cam or not smpl_proj_by_cam:
            return tiles


        lr_pairs = [
            (2, 5), (3, 6), (4, 7),
            (9, 12), (10, 13), (11, 14),
            (15, 16), (17, 18),
            (19, 22), (20, 23), (21, 24),
        ]

        def _align_gt_for_vis(gt25: np.ndarray, pred25: np.ndarray) -> np.ndarray:
            gt = np.array(gt25, dtype=np.float32, copy=True)
            pred = np.array(pred25, dtype=np.float32, copy=False)
            for a, b in lr_pairs:
                ca, cb = float(gt[a, 2]), float(gt[b, 2])
                wa = float(min(1.0, max(0.0, ca))) if ca >= float(conf_threshold) else 0.0
                wb = float(min(1.0, max(0.0, cb))) if cb >= float(conf_threshold) else 0.0
                if wa <= 0.0 and wb <= 0.0:
                    continue

                e0 = wa * float(np.hypot(pred[a, 0] - gt[a, 0], pred[a, 1] - gt[a, 1])) + wb * float(np.hypot(pred[b, 0] - gt[b, 0], pred[b, 1] - gt[b, 1]))
                e1 = wa * float(np.hypot(pred[a, 0] - gt[b, 0], pred[a, 1] - gt[b, 1])) + wb * float(np.hypot(pred[b, 0] - gt[a, 0], pred[b, 1] - gt[a, 1]))
                if e1 < e0:
                    tmp = gt[a].copy()
                    gt[a] = gt[b]
                    gt[b] = tmp
            return gt

        for cam_id in cam_ids:
            img_path = subject_dir / cam_id / "0000.jpg"
            rgb = _read_bgr(img_path)
            gt = np.array(kps_by_cam[cam_id], dtype=np.float32)
            pred = np.array(smpl_proj_by_cam[cam_id], dtype=np.float32)


            target = int(vis_render_size)
            if target > 0 and (rgb.shape[0] != target or rgb.shape[1] != target):
                orig_h, orig_w = rgb.shape[0], rgb.shape[1]
                sx = float(target) / max(1.0, float(orig_w))
                sy = float(target) / max(1.0, float(orig_h))
                rgb = cv2.resize(rgb, (target, target), interpolation=cv2.INTER_AREA)
                gt = gt.copy()
                pred = pred.copy()
                gt[:, 0] *= sx
                gt[:, 1] *= sy
                pred[:, 0] *= sx
                pred[:, 1] *= sy
            err = per_cam_err.get(cam_id, float("nan"))
            title = f"{cam_id} err={err:.1f}"
            gt_vis = _align_gt_for_vis(gt, pred)
            over = _draw_keypoints_overlay(
                rgb_bgr=rgb,
                gt_kps=gt_vis,
                pred_xy=pred,
                conf_threshold=conf_threshold,
                show_indices=show_indices,
                title=title,
                draw_errors=vis_draw_errors,
            )
            tiles.append(over)
            if vis_save_tiles:
                cv2.imwrite(str(out_dir / f"{tag}_{stage_name}_{cam_id}.png"), over)
        return tiles


    if kps_by_cam and smpl_proj_by_cam:
        for tag, cams in [("best", best), ("worst", worst)]:
            tiles = make_tiles(cams, tag)
            if not tiles:
                continue
            rows = int(np.ceil(len(tiles) / float(vis_cols)))
            grid = _pack_grid(tiles, rows=rows, cols=vis_cols)
            cv2.imwrite(str(out_dir / f"kps_{tag}_grid_{stage_name}.jpg"), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


    if dataset_mesh_verts is not None and dataset_mesh_faces is not None and smpl_mesh_verts is not None and smpl_mesh_faces is not None:
        for tag, cams in [("best", best), ("worst", worst)]:

            image_size_hw = (int(vis_render_size), int(vis_render_size))
            cams_p3d = _build_p3d_cameras_from_calib(calib, cams, device=smpl_mesh_verts.device, image_size_hw=image_size_hw)

            sil_ds = _render_silhouette_masks(dataset_mesh_verts, dataset_mesh_faces, cams_p3d, image_size_hw=image_size_hw)
            sil_sm = _render_silhouette_masks(smpl_mesh_verts, smpl_mesh_faces, cams_p3d, image_size_hw=image_size_hw)
            sil_neus = None
            if neus_mesh_verts is not None and neus_mesh_faces is not None:
                sil_neus = _render_silhouette_masks(neus_mesh_verts, neus_mesh_faces, cams_p3d, image_size_hw=image_size_hw)

            tiles_ds_cur: List[np.ndarray] = []
            tiles_cur_neus: List[np.ndarray] = []
            tiles_cur_gtmask: List[np.ndarray] = []

            def _load_gt_mask(cam_id_: str) -> np.ndarray | None:
                p = subject_dir / cam_id_ / "mask" / "pha" / "0000.png"
                if not p.exists():
                    return None
                m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if m is None:
                    return None
                if m.ndim == 3:
                    m = m[..., 0]
                m = cv2.resize(m, (image_size_hw[1], image_size_hw[0]), interpolation=cv2.INTER_NEAREST)
                return (m.astype(np.float32) > 127.0).astype(np.float32)

            def _mask_boundary(mask01: np.ndarray) -> np.ndarray:

                k = np.ones((3, 3), np.uint8)
                m = (mask01 > 0.5).astype(np.uint8) * 255
                dil = cv2.dilate(m, k, iterations=1)
                ero = cv2.erode(m, k, iterations=1)
                b = cv2.absdiff(dil, ero)
                return (b > 0).astype(np.uint8)

            for i, cam_id in enumerate(cams):
                rgb = _read_bgr(subject_dir / cam_id / "0000.jpg")

                rgb_s = cv2.resize(rgb, (image_size_hw[1], image_size_hw[0]), interpolation=cv2.INTER_AREA)
                ds = sil_ds[i].detach().cpu().numpy()
                sm = sil_sm[i].detach().cpu().numpy()
                err = per_cam_err.get(cam_id, float("nan"))

                ov_a = np.zeros_like(rgb_s, dtype=np.uint8)
                ov_a[ds > 0.5, 1] = 255
                ov_a[sm > 0.5, 2] = 255
                out_a = _alpha_blend_bgr(rgb_s, ov_a, alpha=0.40)
                out_a = cv2.resize(out_a, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                cv2.putText(out_a, f"{cam_id} err={err:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(out_a, "dsSMPL=G  curSMPL=R", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
                tiles_ds_cur.append(out_a)


                if sil_neus is not None:
                    ne = (sil_neus[i].detach().cpu().numpy() > 0.5)
                    ov_b = np.zeros_like(rgb_s, dtype=np.uint8)
                    ov_b[ne, 0] = 255
                    ov_b[ne, 1] = 255
                    ov_b[sm > 0.5, 2] = 255
                    out_b = _alpha_blend_bgr(rgb_s, ov_b, alpha=0.40)
                    out_b = cv2.resize(out_b, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(out_b, f"{cam_id} err={err:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(out_b, "curSMPL=R  NeuS2=C", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
                    tiles_cur_neus.append(out_b)


                gt = _load_gt_mask(cam_id)
                if gt is not None:
                    sm01 = (sm > 0.5).astype(np.float32)
                    outside = (sm01 > 0.5) & (gt < 0.5)
                    missing = (sm01 < 0.5) & (gt > 0.5)
                    ov_c = np.zeros_like(rgb_s, dtype=np.uint8)

                    ov_c[outside, 2] = 255
                    if str(vis_sil_loss_mode) in {"iou", "bce"}:
                        ov_c[missing, 0] = 255

                    b_gt = _mask_boundary(gt)
                    b_sm = _mask_boundary(sm01)
                    ov_c[b_gt > 0, 1] = 255
                    ov_c[b_sm > 0, 0] = np.maximum(ov_c[b_sm > 0, 0], 255)
                    ov_c[b_sm > 0, 1] = np.maximum(ov_c[b_sm > 0, 1], 255)

                    out_c = _alpha_blend_bgr(rgb_s, ov_c, alpha=0.45)
                    out_c = cv2.resize(out_c, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(out_c, f"{cam_id} err={err:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    legend = "GT=Gbd  SMPL=Cbd  outside=R"
                    if str(vis_sil_loss_mode) in {"iou", "bce"}:
                        legend += "  missing=B"
                    cv2.putText(out_c, legend, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
                    tiles_cur_gtmask.append(out_c)

                if vis_save_tiles:
                    cv2.imwrite(str(out_dir / f"mesh_ds_cur_{tag}_{stage_name}_{cam_id}.jpg"), out_a, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    if sil_neus is not None:
                        cv2.imwrite(str(out_dir / f"mesh_cur_neus_{tag}_{stage_name}_{cam_id}.jpg"), out_b, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    if gt is not None:
                        cv2.imwrite(str(out_dir / f"mesh_cur_gtmask_{tag}_{stage_name}_{cam_id}.jpg"), out_c, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            rows = int(np.ceil(len(tiles_ds_cur) / float(vis_cols)))
            grid_a = _pack_grid(tiles_ds_cur, rows=rows, cols=vis_cols)
            cv2.imwrite(str(out_dir / f"mesh_ds_cur_{tag}_grid_{stage_name}.jpg"), grid_a, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if tiles_cur_neus:
                rows = int(np.ceil(len(tiles_cur_neus) / float(vis_cols)))
                grid_b = _pack_grid(tiles_cur_neus, rows=rows, cols=vis_cols)
                cv2.imwrite(str(out_dir / f"mesh_cur_neus_{tag}_grid_{stage_name}.jpg"), grid_b, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if tiles_cur_gtmask:
                rows = int(np.ceil(len(tiles_cur_gtmask) / float(vis_cols)))
                grid_c = _pack_grid(tiles_cur_gtmask, rows=rows, cols=vis_cols)
                cv2.imwrite(str(out_dir / f"mesh_cur_gtmask_{tag}_grid_{stage_name}.jpg"), grid_c, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


    if vis_save_mesh_ply:

        if stage_name == "stage0" and neus_mesh_verts is not None and neus_mesh_faces is not None:
            _export_single_mesh_colored_ply(
                out_dir / "mesh_neus2_cyan.ply",
                neus_mesh_verts,
                neus_mesh_faces,
                color_rgb=(0, 255, 255),
            )


            if float(vis_inside_margin) > 0.0:
                try:
                    m = trimesh.Trimesh(
                        vertices=neus_mesh_verts.detach().cpu().numpy(),
                        faces=neus_mesh_faces.detach().cpu().numpy(),
                        process=False,
                    )
                    vn = np.asarray(m.vertex_normals, dtype=np.float32)
                    v = np.asarray(m.vertices, dtype=np.float32)
                    inset = v - float(vis_inside_margin) * vn
                    _export_single_mesh_colored_ply(
                        out_dir / f"mesh_neus2_inset_{int(round(float(vis_inside_margin) * 1000)):04d}mm_gray.ply",
                        torch.from_numpy(inset).to(neus_mesh_verts.device),
                        neus_mesh_faces,
                        color_rgb=(180, 180, 180),
                    )
                except Exception:
                    pass

        if stage_name == "stage0" and smpl_mesh_verts is not None and smpl_mesh_faces is not None:
            _export_single_mesh_colored_ply(
                out_dir / "mesh_smpl_init_magenta.ply",
                smpl_mesh_verts,
                smpl_mesh_faces,
                color_rgb=(255, 0, 255),
            )

        if stage_name != "stage0" and smpl_mesh_verts is not None and smpl_mesh_faces is not None:
            _export_single_mesh_colored_ply(
                out_dir / f"mesh_smpl_{stage_name}_red.ply",
                smpl_mesh_verts,
                smpl_mesh_faces,
                color_rgb=(255, 0, 0),
            )


def run_finetune(
    subject_dir: Path,
    smpl_model_path: Path,
    neus_mesh_path: Path,
    calib_path: Path,
    device: str = "cuda",
    gender: str = "auto",
    dataset_type: str = "auto",
    iters_stage1: int = 400,
    iters_stage2: int = 200,
    lr_stage1: float = 5e-2,
    lr_stage2: float = 2e-2,
    conf_threshold: float = 0.2,
    keypoints_mode: str = "stable",
    lambda_2d: float = 1.0,
    lambda_collision: float = 0.2,
    lambda_reg: float = 0.01,
    lambda_betas_stage1: float | None = None,
    lambda_betas_stage2: float | None = None,
    lambda_2d_stage1: float | None = None,
    lambda_2d_stage2: float | None = None,
    lambda_collision_stage1: float | None = None,
    lambda_collision_stage2: float | None = None,
    collision_loss_mode: str = "outside_dist2",

    lambda_inside_stage1: float = 0.0,
    lambda_inside_stage2: float = 0.0,
    inside_margin: float = 0.005,
    inside_part_weights: str = "leftArm,rightArm,leftForeArm,rightForeArm,leftLeg,rightLeg,leftUpLeg,rightUpLeg,leftFoot,rightFoot,hips,spine1",
    inside_exclude_parts: str = "spine,spine2,leftShoulder,rightShoulder,neck,head",
    inside_default_weight: float = 0.0,
    inside_debug_vis: bool = False,
    inside_debug_every: int = 100,
    inside_debug_topk: int = 8000,

    lambda_sil_stage1: float = 0.0,
    lambda_sil_stage2: float = 0.0,
    sil_render_size: int = 256,
    sil_cam_sample: int = 8,
    sil_loss_mode: str = "one_sided_outside",
    sil_sigma: float = 1e-4,
    sil_gamma: float = 1e-4,
    sil_faces_per_pixel: int = 50,

    label_pkl_path: Path | None = None,
    lambda_skin_head_stage1: float = 0.0,
    lambda_skin_hands_stage1: float = 0.0,
    lambda_skin_head_stage2: float = 0.2,
    lambda_skin_hands_stage2: float = 0.2,
    skin_sample_head: int = 8000,
    skin_sample_hands: int = 4000,

    collision_sample_n: int = 20000,
    collision_face_knn_k: int = 64,
    collision_debug_vis: bool = False,
    collision_debug_vis_n: int = 2000,
    collision_debug_save_mesh: bool = True,
    collision_debug_every: int = 100,
    collision_debug_cyl_radius: float = 0.002,
    collision_debug_cyl_sections: int = 8,

    lambda_pose_delta_stage2: float = 1e-3,

    include_face_joints_in_2d: bool = True,
    freeze_wrist_pose: bool = False,
    freeze_feet_pose: bool = False,
    run_stage2: bool = False,
    save_debug_meshes: bool = True,
    out_smpl_params_path: Path | None = None,
    vis: bool = False,
    vis_n_best: int = 12,
    vis_n_worst: int = 12,
    vis_cols: int = 4,
    vis_render_size: int = 512,
    vis_save_tiles: bool = False,
    vis_save_mesh_ply: bool = False,

    keypoints_detector: str = "yolo_pose",
    yolo_pose_model: str = "yolov8n-pose.pt",
    yolo_det_conf: float = 0.25,
    yolo_det_iou: float = 0.7,
    yolo_imgsz: int = 1024,

    keypoints_regenerate: bool = False,

    log_dir: Path | None = None,
    log_every: int = 10,
    log_tensorboard: bool = True,
    log_wandb: bool = False,
    wandb_project: str = "mvhumannet_smplx_finetune",
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
    wandb_mode: str = "online",

    collision_debug_sphere_radius: float = 0.004,
    collision_debug_mode: str = "topk_outside",
    collision_backend: str = "pytorch_knn_faces",
    cuda_nn_scale_factor: float = 1000.0,
    exp_name: str = "",


    pipeline: List[Dict[str, Any]] | None = None,
) -> None:
    subject_dir = Path(subject_dir)
    smpl_npz_path = subject_dir / "smpl_params.npz"
    if not smpl_npz_path.exists():
        raise FileNotFoundError(f"Missing smpl_params.npz at {smpl_npz_path}")

    device_t = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    logger.info(f"Using device: {device_t}")

    def _resolve_gender(subject_dir_: Path, gender_: str) -> str:
        g = str(gender_).strip().lower()
        if g in {"male", "female", "neutral"}:
            return g
        if g not in {"auto", ""}:
            logger.warning(f"Unknown gender='{gender_}', falling back to auto->neutral")
        gpath = subject_dir_ / "gender.txt"
        if gpath.exists():
            try:
                txt = gpath.read_text(encoding="utf-8").strip().lower()
                if txt in {"m", "male"}:
                    return "male"
                if txt in {"f", "female"}:
                    return "female"
                if txt in {"n", "neutral"}:
                    return "neutral"
                logger.warning(f"Unrecognized contents in {gpath}: '{txt}', using neutral")
            except Exception as e:
                logger.warning(f"Failed reading {gpath}: {e}, using neutral")
        return "neutral"

    resolved_gender = _resolve_gender(subject_dir, gender)
    logger.info(
        f"SMPL-X gender: {resolved_gender} (gender arg='{gender}', gender.txt={'found' if (subject_dir/'gender.txt').exists() else 'missing'})"
    )
    exp_name_clean = str(exp_name).strip().replace("/", "-").replace("\\", "-").replace(" ", "_")
    exp_suffix = f"_{exp_name_clean}" if exp_name_clean else ""
    vis_root_name = f"smplx_finetune_vis_{resolved_gender}{exp_suffix}"


    lambda_betas_1 = float(lambda_reg) if lambda_betas_stage1 is None else float(lambda_betas_stage1)
    lambda_betas_2 = float(lambda_reg) if lambda_betas_stage2 is None else float(lambda_betas_stage2)
    lambda_2d_1 = float(lambda_2d) if lambda_2d_stage1 is None else float(lambda_2d_stage1)
    lambda_2d_2 = float(lambda_2d) if lambda_2d_stage2 is None else float(lambda_2d_stage2)
    lambda_col_1 = float(lambda_collision) if lambda_collision_stage1 is None else float(lambda_collision_stage1)
    lambda_col_2 = float(lambda_collision) if lambda_collision_stage2 is None else float(lambda_collision_stage2)
    collision_loss_mode_ = str(collision_loss_mode)
    if collision_loss_mode_ not in {"outside", "outside2", "outside_dist", "outside_dist2"}:
        raise ValueError("--collision_loss_mode must be one of outside|outside2|outside_dist|outside_dist2")

    sil_loss_mode_ = str(sil_loss_mode)
    if sil_loss_mode_ not in {"iou", "bce", "one_sided_outside"}:
        raise ValueError("--sil_loss_mode must be one of iou|bce|one_sided_outside")


    calib = _load_calibration(calib_path)


    need_keypoints = False
    need_sil_masks = False
    if pipeline is not None:
        for st in list(pipeline):
            losses = dict(st.get("losses", {}) or {})
            if float(losses.get("lambda_2d", 0.0)) > 0.0:
                need_keypoints = True
            if float(losses.get("lambda_sil", 0.0)) > 0.0:
                need_sil_masks = True
    else:
        if float(lambda_2d_1) > 0.0 or (bool(run_stage2) and float(lambda_2d_2) > 0.0):
            need_keypoints = True
        if float(lambda_sil_stage1) > 0.0 or (bool(run_stage2) and float(lambda_sil_stage2) > 0.0):
            need_sil_masks = True


    kps_by_cam: Dict[str, np.ndarray] = {}
    if need_keypoints:
        det = str(keypoints_detector).strip().lower()


        if bool(keypoints_regenerate):
            if det in {"yolo_pose", "yolo", "ultralytics", "ultralytics_yolo"}:
                logger.warning(
                    f"keypoints_regenerate=true: re-running YOLO pose and overwriting existing keypoints_2d for all cameras "
                    f"(model={yolo_pose_model})."
                )
                _auto_detect_and_write_keypoints_yolo_pose(
                    subject_dir=subject_dir,
                    cam_ids=sorted(calib.keys()),
                    device=str(device_t),
                    vis_root_dir=(subject_dir / vis_root_name),
                    keypoints_mode=str(keypoints_mode),
                    conf_threshold=float(conf_threshold),
                    yolo_pose_model=str(yolo_pose_model),
                    yolo_det_conf=float(yolo_det_conf),
                    yolo_det_iou=float(yolo_det_iou),
                    yolo_imgsz=int(yolo_imgsz),
                    overwrite_existing=True,
                )
            elif det in {"", "none", "off", "disable", "disabled"}:
                raise RuntimeError("keypoints_regenerate=true but keypoints_detector is disabled.")
            else:
                raise RuntimeError(
                    f"keypoints_regenerate=true but keypoints_detector='{keypoints_detector}' is not supported."
                )
        kps_by_cam = _load_keypoints_for_subject(subject_dir)
        if not kps_by_cam:
            if det in {"", "none", "off", "disable", "disabled"}:
                raise RuntimeError(
                    f"No keypoints JSONs found under {subject_dir}/*/keypoints_2d/ but lambda_2d>0. "
                    f"Either export keypoints_2d first, or set lambda_2d=0, or enable keypoints_detector."
                )

            if det not in {"yolo_pose", "yolo", "ultralytics", "ultralytics_yolo"}:
                raise RuntimeError(
                    f"No keypoints JSONs found under {subject_dir}/*/keypoints_2d/ but lambda_2d>0, and "
                    f"keypoints_detector='{keypoints_detector}' is not supported. Use 'yolo_pose' or 'none'."
                )

            logger.warning(
                f"No keypoints JSONs found under {subject_dir}/*/keypoints_2d/ but lambda_2d>0. "
                f"Auto-detecting 2D keypoints with Ultralytics YOLO pose (model={yolo_pose_model})."
            )
            _auto_detect_and_write_keypoints_yolo_pose(
                subject_dir=subject_dir,
                cam_ids=sorted(calib.keys()),
                device=str(device_t),
                vis_root_dir=(subject_dir / vis_root_name),
                keypoints_mode=str(keypoints_mode),
                conf_threshold=float(conf_threshold),
                yolo_pose_model=str(yolo_pose_model),
                yolo_det_conf=float(yolo_det_conf),
                yolo_det_iou=float(yolo_det_iou),
                yolo_imgsz=int(yolo_imgsz),
                overwrite_existing=False,
            )

            kps_by_cam = _load_keypoints_for_subject(subject_dir)
            if not kps_by_cam:
                raise RuntimeError(
                    f"Auto keypoint detection did not produce any usable keypoint JSONs under {subject_dir}/*/keypoints_2d/. "
                    f"Check images exist at <cam>/0000.jpg and Ultralytics model '{yolo_pose_model}' runs."
                )


        if det in {"yolo_pose", "yolo", "ultralytics", "ultralytics_yolo"}:
            missing_cam_ids = sorted([c for c in calib.keys() if c not in kps_by_cam])
            if missing_cam_ids:
                logger.warning(
                    f"{len(missing_cam_ids)} cameras have no valid keypoints_2d but lambda_2d>0. "
                    f"Auto-detecting missing views with YOLO pose (model={yolo_pose_model})."
                )
                _auto_detect_and_write_keypoints_yolo_pose(
                    subject_dir=subject_dir,
                    cam_ids=missing_cam_ids,
                    device=str(device_t),
                    vis_root_dir=(subject_dir / vis_root_name),
                    keypoints_mode=str(keypoints_mode),
                    conf_threshold=float(conf_threshold),
                    yolo_pose_model=str(yolo_pose_model),
                    yolo_det_conf=float(yolo_det_conf),
                    yolo_det_iou=float(yolo_det_iou),
                    yolo_imgsz=int(yolo_imgsz),
                    overwrite_existing=False,
                )
                kps_by_cam = _load_keypoints_for_subject(subject_dir)


        cam_ids = sorted(set(calib.keys()).intersection(set(kps_by_cam.keys())))
        if not cam_ids:
            raise RuntimeError("No camera ids overlap between calibration_full.json and keypoints_2d")
        logger.info(f"Using {len(cam_ids)} cameras with both calibration and keypoints")


        kps_by_cam = _apply_mask_filter_to_keypoints(subject_dir=subject_dir, cam_ids=cam_ids, kps_by_cam=kps_by_cam)


        _save_keypoints2d_overlays(
            subject_dir=subject_dir,
            cam_ids=cam_ids,
            kps_by_cam=kps_by_cam,
            conf_threshold=float(conf_threshold),
            out_dir=(subject_dir / vis_root_name / "keypoints2d"),
        )
    else:
        cam_ids = sorted(calib.keys())
        logger.info(
            f"2D keypoint loss disabled (no stages require it). Using {len(cam_ids)} cameras from calibration for vis/silhouette."
        )


    sil_size_hw = (int(sil_render_size), int(sil_render_size))
    gt_masks_cpu: Dict[str, torch.Tensor] = {}
    if need_sil_masks:
        gt_masks_cpu = _load_gt_silhouette_masks(subject_dir, cam_ids, image_size_hw=sil_size_hw)
        if not gt_masks_cpu:
            logger.warning(
                f"Silhouette loss enabled but no GT masks found under <cam>/mask/pha/0000.png in {subject_dir}"
            )


    smpl_npz = np.load(str(smpl_npz_path), allow_pickle=True)
    smpl_params = {k: torch.from_numpy(v.astype(np.float32)).to(device_t) for k, v in smpl_npz.items()}


    ds = str(dataset_type).strip().lower()
    if ds in {"thu2", "thuman2", "thuman"}:
        ds = "thuman2"
    if ds in {"mvhuman", "mvhumannet"}:
        ds = "mvhumannet"
    if ds in {"actors", "actorshq"}:
        ds = "actorshq"
    if ds in {"talk", "talkbody4d"}:
        ds = "talkbody4d"
    if ds in {"avatar", "avatarrex"}:
        ds = "avatarrex"
    if ds in {"auto", ""}:
        hand_dim = None
        if "left_hand_pose" in smpl_params:
            hand_dim = int(smpl_params["left_hand_pose"].shape[-1])
        elif "right_hand_pose" in smpl_params:
            hand_dim = int(smpl_params["right_hand_pose"].shape[-1])
        if hand_dim == 12:
            ds = "thuman2"
        elif hand_dim == 6:
            ds = "mvhumannet"
        elif hand_dim == 45:
            ds = "talkbody4d"
        else:
            ds = "mvhumannet"
    logger.info(f"SMPL-X dataset_type resolved to '{ds}' (requested='{dataset_type}')")


    smpl_model = create_smplx_model_for_neus2_reposing(
        smpl_params=smpl_params,
        batch_size=1,
        device=str(device_t),
        gender=resolved_gender,
        dataset_type=ds,
        model_path=str(smpl_model_path) if str(smpl_model_path) else None,
    )
    smpl_model.eval()


    J_reg = load_smplx_J_regressor_body25_smplx().to(device_t)
    smpl_faces = torch.from_numpy(smpl_model.faces.astype(np.int64)).to(device_t)


    neus_mesh = trimesh.load_mesh(str(neus_mesh_path), process=False)
    V_neus = torch.from_numpy(np.asarray(neus_mesh.vertices, dtype=np.float32)).to(device_t)
    N_neus = torch.from_numpy(np.asarray(neus_mesh.vertex_normals, dtype=np.float32)).to(device_t)
    F_neus = torch.from_numpy(np.asarray(neus_mesh.faces, dtype=np.int64)).to(device_t)

    v0_all = V_neus.index_select(0, F_neus[:, 0])
    v1_all = V_neus.index_select(0, F_neus[:, 1])
    v2_all = V_neus.index_select(0, F_neus[:, 2])
    face_centers = (v0_all + v1_all + v2_all) / 3.0
    face_normals = torch.cross(v1_all - v0_all, v2_all - v0_all, dim=-1)
    face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    if int(collision_face_knn_k) < 32:
        logger.warning(
            f"collision_face_knn_k={int(collision_face_knn_k)} is small; if closest points look wrong (often on edges/verts), "
            f"try 64 or 128 to improve candidate-face recall."
        )


    tb_writer = None
    wandb_run = None
    csv_fp = None
    csv_writer = None

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:

        log_dir = subject_dir / "smplx_finetune_logs" / f"{run_stamp}_{resolved_gender}{exp_suffix}"
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg_out = {
        "subject_dir": str(subject_dir),
        "neus_mesh_path": str(neus_mesh_path),
        "calib_path": str(calib_path),
        "device": str(device_t),
        "iters_stage1": int(iters_stage1),
        "iters_stage2": int(iters_stage2),
        "lr_stage1": float(lr_stage1),
        "lr_stage2": float(lr_stage2),
        "conf_threshold": float(conf_threshold),
        "keypoints_mode": str(keypoints_mode),
        "lambda_2d": float(lambda_2d),
        "lambda_collision": float(lambda_collision),
        "lambda_reg": float(lambda_reg),
        "lambda_betas_stage1": float(lambda_betas_1),
        "lambda_betas_stage2": float(lambda_betas_2),
        "lambda_2d_stage1": float(lambda_2d_1),
        "lambda_2d_stage2": float(lambda_2d_2),
        "lambda_collision_stage1": float(lambda_col_1),
        "lambda_collision_stage2": float(lambda_col_2),
        "collision_loss_mode": str(collision_loss_mode_),
        "lambda_sil_stage1": float(lambda_sil_stage1),
        "lambda_sil_stage2": float(lambda_sil_stage2),
        "sil_render_size": int(sil_render_size),
        "sil_cam_sample": int(sil_cam_sample),
        "sil_loss_mode": str(sil_loss_mode_),
        "sil_sigma": float(sil_sigma),
        "sil_gamma": float(sil_gamma),
        "sil_faces_per_pixel": int(sil_faces_per_pixel),
        "gender": str(resolved_gender),
        "collision_sample_n": int(collision_sample_n),
        "collision_face_knn_k": int(collision_face_knn_k),
        "collision_debug_vis": bool(collision_debug_vis),
        "collision_debug_vis_n": int(collision_debug_vis_n),
        "collision_debug_every": int(collision_debug_every),
        "collision_debug_cyl_radius": float(collision_debug_cyl_radius),
        "collision_debug_cyl_sections": int(collision_debug_cyl_sections),
        "collision_debug_sphere_radius": float(collision_debug_sphere_radius),
        "collision_debug_mode": str(collision_debug_mode),
        "lambda_skin_head_stage1": float(lambda_skin_head_stage1),
        "lambda_skin_hands_stage1": float(lambda_skin_hands_stage1),
        "lambda_skin_head_stage2": float(lambda_skin_head_stage2),
        "lambda_skin_hands_stage2": float(lambda_skin_hands_stage2),
        "skin_sample_head": int(skin_sample_head),
        "skin_sample_hands": int(skin_sample_hands),
        "lambda_pose_delta_stage2": float(lambda_pose_delta_stage2),
        "freeze_wrist_pose": bool(freeze_wrist_pose),
        "freeze_feet_pose": bool(freeze_feet_pose),
        "include_face_joints_in_2d": bool(include_face_joints_in_2d),
        "run_stage2": bool(run_stage2),
        "vis": bool(vis),
        "log_every": int(log_every),
        "log_tensorboard": bool(log_tensorboard),
        "log_wandb": bool(log_wandb),
        "wandb_project": str(wandb_project),
        "wandb_entity": str(wandb_entity) if wandb_entity else None,
        "wandb_name": str(wandb_name) if wandb_name else None,
        "wandb_mode": str(wandb_mode),
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)

    if log_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            tb_writer = SummaryWriter(log_dir=str(log_dir / "tb"))
            logger.info(f"TensorBoard logs: {log_dir / 'tb'}")
        except Exception as e:
            logger.warning(f"TensorBoard SummaryWriter unavailable: {e}")
            tb_writer = None

    if log_wandb:
        try:
            import wandb  # type: ignore

            wandb_kwargs = dict(
                project=str(wandb_project),
                config=cfg_out,
                dir=str(log_dir),
                mode=str(wandb_mode),
            )
            if wandb_entity:
                wandb_kwargs["entity"] = str(wandb_entity)
            if wandb_name:
                wandb_kwargs["name"] = str(wandb_name)
            wandb_run = wandb.init(**wandb_kwargs)
            logger.info("W&B logging enabled.")
        except Exception as e:
            logger.warning(f"Failed to init wandb; continuing without wandb. err={e}")
            wandb_run = None

    csv_fp = open(log_dir / "losses.csv", "w", newline="")
    csv_writer = csv.DictWriter(
        csv_fp,
        fieldnames=[
            "global_step",
            "stage",
            "iter",
            "total",
            "l2d",
            "lcol",
            "lskin",
            "lreg",
            "lpose",
            "scale",
            "betas_l2",
            "transl_l2",
            "outside_frac",
            "outside_mean",
            "dist_mean",
            "frac_on_face_best",
        ],
    )
    csv_writer.writeheader()

    def _log(step: int, stage: str, it: int, scalars: Dict[str, float]) -> None:
        if int(log_every) > 0 and (it % int(log_every) != 0):
            return
        if tb_writer is not None:
            for k, v in scalars.items():
                tb_writer.add_scalar(f"{stage}/{k}", float(v), int(step))
        if wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb.log({f"{stage}/{k}": float(v) for k, v in scalars.items()}, step=int(step))
            except Exception:
                pass
        if csv_writer is not None:
            row = dict(
                global_step=int(step),
                stage=str(stage),
                iter=int(it),
                total=float(scalars.get("total", np.nan)),
                l2d=float(scalars.get("l2d", np.nan)),
                lcol=float(scalars.get("lcol", np.nan)),
                lskin=float(scalars.get("lskin", np.nan)),
                lreg=float(scalars.get("lreg", np.nan)),
                lpose=float(scalars.get("lpose", np.nan)),
                scale=float(scalars.get("scale", np.nan)),
                betas_l2=float(scalars.get("betas_l2", np.nan)),
                transl_l2=float(scalars.get("transl_l2", np.nan)),
                outside_frac=float(scalars.get("outside_frac", np.nan)),
                outside_mean=float(scalars.get("outside_mean", np.nan)),
                dist_mean=float(scalars.get("dist_mean", np.nan)),
                frac_on_face_best=float(scalars.get("frac_on_face_best", np.nan)),
            )
            csv_writer.writerow(row)
            csv_fp.flush()


    scan_labels = None
    if label_pkl_path is None:
        label_pkl_path = subject_dir / "mesh" / "labeled_no_sam" / "label-f0000_extended.pkl"
    if Path(label_pkl_path).exists():
        scan_labels = _load_scan_labels_extended(Path(label_pkl_path))
        if scan_labels.shape[0] != V_neus.shape[0]:
            raise ValueError(
                f"Label count mismatch: scan_labels has {scan_labels.shape[0]} entries but NeuS2 mesh has {V_neus.shape[0]} vertices. "
                f"label_pkl_path={label_pkl_path} neus_mesh_path={neus_mesh_path}"
            )
    else:
        logger.warning(f"Label PKL not found at {label_pkl_path}; skin tight-fit loss will be disabled.")


    dataset_mesh_path = subject_dir / "mesh" / "processed" / "smpl_body_dataset.obj"
    dataset_mesh_verts = None
    dataset_mesh_faces = None
    if dataset_mesh_path.exists():
        try:
            ds_mesh = trimesh.load_mesh(str(dataset_mesh_path), process=False)
            dataset_mesh_verts = torch.from_numpy(np.asarray(ds_mesh.vertices, dtype=np.float32)).to(device_t)
            dataset_mesh_faces = torch.from_numpy(np.asarray(ds_mesh.faces, dtype=np.int64)).to(device_t)
            logger.info(f"Loaded dataset SMPL mesh for projection: {dataset_mesh_path}")
        except Exception as e:
            logger.warning(f"Failed to load dataset SMPL mesh at {dataset_mesh_path}: {e}")


    betas = smpl_params["betas"].detach().clone().requires_grad_(True)
    global_orient = smpl_params["global_orient"].detach().clone().requires_grad_(True)
    transl = smpl_params["transl"].detach().clone().requires_grad_(True)


    scale_eps = torch.tensor(1e-6, device=device_t, dtype=torch.float32)

    init_scale = torch.tensor(1.0, device=device_t, dtype=torch.float32)
    try:
        if "scale" in smpl_params:
            s = smpl_params["scale"].detach()

            if s.numel() >= 1:
                init_scale = s.view(-1)[0].to(device_t).to(torch.float32).clamp_min(1e-6)
    except Exception:
        pass

    scale_raw_init = torch.log(torch.expm1(torch.clamp(init_scale - scale_eps, min=1e-6)))
    scale_raw = scale_raw_init.view(1).detach().clone().requires_grad_(True)

    def _scale_pos() -> torch.Tensor:
        return F.softplus(scale_raw) + scale_eps


    fixed_keys = [k for k in smpl_params.keys() if k not in {"betas", "global_orient", "transl", "scale"}]
    fixed_params = {k: smpl_params[k].detach().clone() for k in fixed_keys}

    body_pose_init = smpl_params.get("body_pose")
    if body_pose_init is not None:
        body_pose_init = body_pose_init.detach().clone()


    seg_path = Path(get_smplx_model_path()).parent / "smplx_vert_segmentation.json"
    seg_data = json.load(open(seg_path))
    smpl_idx_head = torch.tensor(seg_data["head"], device=device_t, dtype=torch.long)
    smpl_idx_lhand = torch.tensor(seg_data["leftHand"], device=device_t, dtype=torch.long)
    smpl_idx_rhand = torch.tensor(seg_data["rightHand"], device=device_t, dtype=torch.long)


    inside_vertex_weights: torch.Tensor | None = None

    def _build_inside_vertex_weights(n_verts: int) -> torch.Tensor:
        w = torch.full((int(n_verts),), float(inside_default_weight), device=device_t, dtype=torch.float32)
        part_w = _parse_part_weight_spec(str(inside_part_weights))
        for part_name, pw in part_w.items():
            if part_name not in seg_data:
                logger.warning(
                    f"[inside] part '{part_name}' not found in smplx_vert_segmentation.json. "
                    f"Example keys: {list(seg_data.keys())[:20]} ..."
                )
                continue
            idx = torch.tensor(seg_data[part_name], device=device_t, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < w.shape[0])]
            if idx.numel() > 0:
                w.index_put_((idx,), torch.full((idx.numel(),), float(pw), device=device_t), accumulate=False)

        excl = [p.strip() for p in str(inside_exclude_parts).split(",") if p.strip()]
        for part_name in excl:
            if part_name not in seg_data:
                continue
            idx = torch.tensor(seg_data[part_name], device=device_t, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < w.shape[0])]
            if idx.numel() > 0:
                w.index_put_((idx,), torch.zeros((idx.numel(),), device=device_t), accumulate=False)
        return w

    stable_idx = set(_stable_body25_indices())
    use_idx = list(range(25)) if keypoints_mode == "all" else sorted(stable_idx)
    if include_face_joints_in_2d:
        use_idx = sorted(set(use_idx).union(set(_face_body25_indices())))
    use_idx_t = torch.tensor(use_idx, device=device_t, dtype=torch.long)

    def forward_body25_joints_world(curr_betas, curr_global_orient, curr_transl, curr_scale, curr_body_pose=None):


        params = dict(fixed_params)
        if curr_body_pose is not None:
            params["body_pose"] = curr_body_pose
        if str(ds) == "thuman2":

            params.update(
                {
                    "betas": curr_betas,
                    "global_orient": curr_global_orient,
                    "transl": curr_transl,
                    "scale": curr_scale,
                }
            )
            out = smpl_model(**params, return_verts=True)
            V_world = out.vertices[0]
            J_world = J_reg @ V_world
            return J_world, V_world


        params.update(
            {
                "betas": curr_betas,
                "global_orient": torch.zeros_like(curr_global_orient),
                "transl": torch.zeros_like(curr_transl),
                "scale": torch.ones_like(curr_scale),
            }
        )
        out = smpl_model(**params, return_verts=True)
        V_canon = out.vertices[0]
        J_canon = J_reg @ V_canon
        J_world = _apply_external_rigid_transform(J_canon, curr_global_orient, curr_transl, curr_scale)
        V_world = _apply_external_rigid_transform(V_canon, curr_global_orient, curr_transl, curr_scale)
        return J_world, V_world

    def skin_tight_fit_loss_raw(V_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if scan_labels is None:
            z = torch.tensor(0.0, device=device_t)
            return z, z


        raw_head = torch.tensor(0.0, device=device_t)
        raw_hands = torch.tensor(0.0, device=device_t)


        neus_head_idx = np.where(scan_labels == 1)[0]
        if neus_head_idx.size > 0:
            neus_head = V_neus[torch.from_numpy(neus_head_idx).to(device_t)]
            smpl_head = V_world.index_select(0, smpl_idx_head)
            neus_head = _sample_points(neus_head, int(skin_sample_head))
            smpl_head = _sample_points(smpl_head, int(skin_sample_head))
            raw_head = _chamfer_bidirectional(smpl_head, neus_head)


        if int((scan_labels == 7).sum()) > 0:
            neus_hands_idx = np.where(scan_labels == 7)[0]
        else:
            neus_hands_idx = np.where(scan_labels == 8)[0]
        if neus_hands_idx.size > 0:
            neus_hands = V_neus[torch.from_numpy(neus_hands_idx).to(device_t)]
            smpl_hands = torch.cat(
                [
                    V_world.index_select(0, smpl_idx_lhand),
                    V_world.index_select(0, smpl_idx_rhand),
                ],
                dim=0,
            )
            neus_hands = _sample_points(neus_hands, int(skin_sample_hands))
            smpl_hands = _sample_points(smpl_hands, int(skin_sample_hands))
            raw_hands = _chamfer_bidirectional(smpl_hands, neus_hands)

        return raw_head, raw_hands

    def loss_2d(J_world: torch.Tensor) -> torch.Tensor:


        lr_pairs = [
            (2, 5),
            (3, 6),
            (4, 7),
            (9, 12),
            (10, 13),
            (11, 14),
            (15, 16),
            (17, 18),
            (19, 22),
            (20, 23),
            (21, 24),
        ]

        use_idx_set = set(use_idx)
        paired = [(a, b) for (a, b) in lr_pairs if (a in use_idx_set and b in use_idx_set)]
        paired_flat = set([i for ab in paired for i in ab])
        unpaired = [i for i in use_idx if i not in paired_flat]
        unpaired_t = torch.tensor(unpaired, device=device_t, dtype=torch.long) if unpaired else None

        total = torch.tensor(0.0, device=device_t)
        denom = torch.tensor(0.0, device=device_t)
        for cam_id in cam_ids:
            cam = calib[cam_id]
            kps = kps_by_cam[cam_id]
            kps_xy = torch.from_numpy(kps[:, :2]).to(device_t)
            kps_c = torch.from_numpy(kps[:, 2]).to(device_t)

            kps_c = torch.where(kps_c >= float(conf_threshold), kps_c, torch.zeros_like(kps_c))

            K = torch.from_numpy(cam["K"]).to(device_t)
            R = torch.from_numpy(cam["R"]).to(device_t)
            T = torch.from_numpy(cam["T"]).to(device_t)

            proj = _project_points(J_world, K, R, T)


            if unpaired_t is not None and int(unpaired_t.numel()) > 0:
                proj_u = proj.index_select(0, unpaired_t)
                tgt_u = kps_xy.index_select(0, unpaired_t)
                conf_u = kps_c.index_select(0, unpaired_t)
                valid = conf_u > 0
                if bool(valid.any()):
                    diff = proj_u[valid] - tgt_u[valid]
                    per = _huber(diff, delta=5.0).sum(dim=-1)
                    w = conf_u[valid].clamp(0.0, 1.0)
                    total = total + (per * w).sum()
                    denom = denom + w.sum()


            for a, b in paired:
                ca = kps_c[a].clamp(0.0, 1.0)
                cb = kps_c[b].clamp(0.0, 1.0)
                if (ca <= 0) and (cb <= 0):
                    continue


                da = proj[a] - kps_xy[a]
                db = proj[b] - kps_xy[b]
                la = _huber(da[None, :], delta=5.0).sum(dim=-1)[0]
                lb = _huber(db[None, :], delta=5.0).sum(dim=-1)[0]
                num0 = ca * la + cb * lb


                da1 = proj[a] - kps_xy[b]
                db1 = proj[b] - kps_xy[a]
                la1 = _huber(da1[None, :], delta=5.0).sum(dim=-1)[0]
                lb1 = _huber(db1[None, :], delta=5.0).sum(dim=-1)[0]
                num1 = ca * la1 + cb * lb1

                total = total + torch.minimum(num0, num1)
                denom = denom + (ca + cb)

        return total / denom.clamp_min(1.0)

    def _collision_closest_points(Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        Kc = int(max(1, collision_face_knn_k))
        knn_fc = knn_points(Q[None], face_centers[None], K=Kc)
        face_idx = knn_fc.idx[0]

        v0 = v0_all.index_select(0, face_idx.reshape(-1)).view(-1, Kc, 3)
        v1 = v1_all.index_select(0, face_idx.reshape(-1)).view(-1, Kc, 3)
        v2 = v2_all.index_select(0, face_idx.reshape(-1)).view(-1, Kc, 3)
        Pk, dist2k = _closest_point_on_triangles(Q, v0, v1, v2)
        best_k = dist2k.argmin(dim=1)
        P = Pk[torch.arange(Q.shape[0], device=device_t), best_k]
        fbest = face_idx[torch.arange(Q.shape[0], device=device_t), best_k]
        Nf = face_normals.index_select(0, fbest)
        dist = torch.sqrt(dist2k[torch.arange(Q.shape[0], device=device_t), best_k].clamp_min(1e-12))
        d = Q - P
        outside = torch.relu((d * Nf).sum(dim=-1))
        return P, Nf, dist, outside, fbest

    def _collision_closest_points_cuda_point_face(Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if reshape_ops is None:
            raise RuntimeError(
                "reshape_ops is not available. Install/build it in the swapvton environment, e.g.\n"
                "  bash scripts/install_env.sh swapvton  (from the AvatarMix root)\n"
            )
        if not Q.is_cuda:
            raise RuntimeError("cuda_point_face backend requires CUDA tensors.")


        tris = V_neus.index_select(0, F_neus.reshape(-1)).view(-1, 3, 3).contiguous()


        sf = float(cuda_nn_scale_factor)
        Qs = (Q * sf).contiguous()
        tris_s = (tris * sf).contiguous()


        p_first = torch.tensor([0], device=Q.device, dtype=torch.long)
        t_first = torch.tensor([0], device=Q.device, dtype=torch.long)


        dist2_s, face_idx, w0, w1, w2 = reshape_ops.nearest_face_pytorch3d(
            Qs,
            p_first,
            tris_s,
            t_first,
            int(Q.shape[0]),
            5e-3,
        )


        dist = torch.sqrt(dist2_s.clamp_min(0.0)) / sf


        w = torch.stack([w0, w1, w2], dim=-1).to(Q.dtype)
        tri_best = tris.index_select(0, face_idx.to(torch.long))
        P = (w[:, 0:1] * tri_best[:, 0] + w[:, 1:2] * tri_best[:, 1] + w[:, 2:3] * tri_best[:, 2])


        Nf = face_normals.index_select(0, face_idx.to(torch.long))
        d = Q - P
        outside = torch.relu((d * Nf).sum(dim=-1))
        return P, Nf, dist, outside, face_idx.to(torch.long)


    def loss_collision_and_inside(
        V_world: torch.Tensor,
        sample_n: int = 5000,
        lambda_inside_now: float = 0.0,
        debug_tag: str | None = None,
        debug_step: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if V_world.shape[0] > sample_n:
            idx = torch.randperm(V_world.shape[0], device=device_t)[:sample_n]
            Q = V_world.index_select(0, idx)
        else:
            Q = V_world
            idx = None

        if collision_backend == "cuda_point_face":
            P, Nf, dist, outside, fbest = _collision_closest_points_cuda_point_face(Q)
        else:
            P, Nf, dist, outside, fbest = _collision_closest_points(Q)


        signed = ((Q - P) * Nf).sum(dim=-1)


        if float(lambda_inside_now) > 0.0:
            nonlocal inside_vertex_weights
            if inside_vertex_weights is None or int(inside_vertex_weights.shape[0]) < int(V_world.shape[0]):
                inside_vertex_weights = _build_inside_vertex_weights(int(V_world.shape[0]))
            wv = inside_vertex_weights if idx is None else inside_vertex_weights.index_select(0, idx)
            inside_violation = torch.relu(-(signed + float(inside_margin)))
            raw_inside = (inside_violation * wv).mean()
        else:
            inside_violation = torch.zeros_like(signed)
            raw_inside = torch.tensor(0.0, device=device_t)

        if collision_debug_vis and debug_tag is not None and debug_step is not None:
            if collision_debug_every > 0 and (debug_step % int(collision_debug_every) == 0):
                nvis = min(int(collision_debug_vis_n), Q.shape[0])
                mode = str(collision_debug_mode)
                if mode == "topk_outside":
                    score = outside * dist
                    if (score > 0).any():
                        vis_idx = torch.topk(score, k=nvis, largest=True).indices
                    else:
                        vis_idx = torch.randperm(Q.shape[0], device=device_t)[:nvis]
                elif mode == "topk_inside":
                    score = inside_violation * dist
                    if (score > 0).any():
                        vis_idx = torch.topk(score, k=nvis, largest=True).indices
                    else:
                        vis_idx = torch.randperm(Q.shape[0], device=device_t)[:nvis]
                elif mode == "topk_both":
                    half = max(1, nvis // 2)
                    score_o = outside * dist
                    score_i = inside_violation * dist
                    idx_o = (
                        torch.topk(score_o, k=min(half, Q.shape[0]), largest=True).indices
                        if (score_o > 0).any()
                        else torch.randperm(Q.shape[0], device=device_t)[: min(half, Q.shape[0])]
                    )
                    idx_i = (
                        torch.topk(score_i, k=min(nvis - half, Q.shape[0]), largest=True).indices
                        if (score_i > 0).any()
                        else torch.randperm(Q.shape[0], device=device_t)[: min(nvis - half, Q.shape[0])]
                    )
                    vis_idx = torch.unique(torch.cat([idx_o, idx_i], dim=0))[:nvis]
                elif mode == "random":
                    vis_idx = torch.randperm(Q.shape[0], device=device_t)[:nvis]
                else:
                    logger.warning(f"Unknown collision_debug_mode={mode}, fallback to topk_outside")
                    score = outside * dist
                    if (score > 0).any():
                        vis_idx = torch.topk(score, k=nvis, largest=True).indices
                    else:
                        vis_idx = torch.randperm(Q.shape[0], device=device_t)[:nvis]

                Qv = Q.index_select(0, vis_idx).detach().cpu().numpy().astype(np.float32)
                Pv = P.index_select(0, vis_idx).detach().cpu().numpy().astype(np.float32)
                dbg_dir = subject_dir / vis_root_name / "collision_debug"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                ply_path = dbg_dir / f"{debug_tag}_step{debug_step:04d}_scene.ply"
                try:
                    _export_collision_debug_scene_ply(
                        out_path=ply_path,
                        mesh_verts=V_neus.detach().cpu().numpy().astype(np.float32),
                        mesh_faces=F_neus.detach().cpu().numpy().astype(np.int64),
                        query_pts=Qv,
                        closest_pts=Pv,
                        cyl_radius=float(collision_debug_cyl_radius),
                        cyl_sections=int(collision_debug_cyl_sections),
                        sphere_radius=float(collision_debug_sphere_radius),
                        add_endpoint_spheres=True,
                    )
                    logger.info(f"Saved collision debug scene: {ply_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to save collision debug vis: {e}")


        if inside_debug_vis and debug_tag is not None and debug_step is not None:
            if inside_debug_every > 0 and (debug_step % int(inside_debug_every) == 0):
                k = min(int(inside_debug_topk), Q.shape[0])
                score = (outside + inside_violation) * dist
                vis_idx2 = (
                    torch.topk(score, k=k, largest=True).indices
                    if (score > 0).any()
                    else torch.randperm(Q.shape[0], device=device_t)[:k]
                )
                pts = Q.index_select(0, vis_idx2).detach().cpu().numpy().astype(np.float32)
                o = outside.index_select(0, vis_idx2).detach().cpu().numpy().astype(np.float32)
                inn = inside_violation.index_select(0, vis_idx2).detach().cpu().numpy().astype(np.float32)
                o_n = o / (o.max() + 1e-8)
                i_n = inn / (inn.max() + 1e-8)
                rgb = np.stack([o_n, np.zeros_like(o_n), i_n], axis=-1) * 255.0
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                dbg_dir = subject_dir / vis_root_name / "collision_debug"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                pc_path = dbg_dir / f"{debug_tag}_step{debug_step:04d}_band_points.ply"
                try:
                    _export_colored_pointcloud_ply(pc_path, pts, rgb)
                except Exception as e:
                    logger.warning(f"Failed to save inside debug pointcloud: {e}")


        with torch.no_grad():
            A = v0_all.index_select(0, fbest)
            B = v1_all.index_select(0, fbest)
            C = v2_all.index_select(0, fbest)
            v0 = B - A
            v1 = C - A
            v2 = P - A
            d00 = (v0 * v0).sum(dim=-1)
            d01 = (v0 * v1).sum(dim=-1)
            d11 = (v1 * v1).sum(dim=-1)
            d20 = (v2 * v0).sum(dim=-1)
            d21 = (v2 * v1).sum(dim=-1)
            denom = (d00 * d11 - d01 * d01).clamp_min(1e-12)
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            eps = 1e-3
            frac_on_face = float(((u > eps) & (v > eps) & (w > eps)).float().mean().detach().cpu())


        if collision_loss_mode_ == "outside":
            raw = outside.mean()
        elif collision_loss_mode_ == "outside2":
            raw = (outside * outside).mean()
        elif collision_loss_mode_ == "outside_dist":
            raw = (outside * dist).mean()
        else:
            raw = (outside * dist * dist).mean()
        loss_val = raw

        loss_collision_and_inside.last_outside_frac = float((outside > 0).float().mean().detach().cpu())
        loss_collision_and_inside.last_outside_mean = float(outside.mean().detach().cpu())
        loss_collision_and_inside.last_inside_frac = float((inside_violation > 0).float().mean().detach().cpu())
        loss_collision_and_inside.last_inside_mean = float(inside_violation.mean().detach().cpu())
        loss_collision_and_inside.last_dist_mean = float(dist.mean().detach().cpu())
        loss_collision_and_inside.last_frac_on_face_best = float(frac_on_face)
        loss_collision_and_inside.last_lcol_raw = float(raw.detach().cpu())
        loss_collision_and_inside.last_linside_raw = float(raw_inside.detach().cpu())
        return loss_val, raw_inside

    def loss_reg(curr_betas: torch.Tensor) -> torch.Tensor:
        return (curr_betas**2).mean()

    def loss_silhouette_raw(V_world: torch.Tensor) -> torch.Tensor:

        if not gt_masks_cpu:
            return torch.tensor(0.0, device=device_t)

        cams_all = [c for c in cam_ids if c in gt_masks_cpu]
        if not cams_all:
            return torch.tensor(0.0, device=device_t)

        n = min(int(sil_cam_sample), len(cams_all))

        perm = torch.randperm(len(cams_all), device=device_t)[:n].tolist()
        cams = [cams_all[i] for i in perm]

        cams_p3d = _build_p3d_cameras_from_calib(calib, cams, device=device_t, image_size_hw=sil_size_hw)
        pred = _render_soft_silhouette_alpha(
            V_world,
            smpl_faces,
            cams_p3d,
            image_size_hw=sil_size_hw,
            sigma=float(sil_sigma),
            gamma=float(sil_gamma),
            faces_per_pixel=int(sil_faces_per_pixel),
        )
        gt = torch.stack([gt_masks_cpu[c] for c in cams], dim=0).to(device_t).to(pred.dtype)

        if sil_loss_mode_ == "iou":
            inter = (pred * gt).sum(dim=(1, 2))
            union = (pred + gt - pred * gt).sum(dim=(1, 2)).clamp_min(1e-6)
            raw = (1.0 - (inter + 1e-6) / (union + 1e-6)).mean()
        elif sil_loss_mode_ == "bce":
            raw = F.binary_cross_entropy(pred, gt)
        else:
            raw = (pred * (1.0 - gt)).mean()

        return raw

    def save_debug(prefix: str, J_world: torch.Tensor, V_world: torch.Tensor):
        if not save_debug_meshes:
            return


        out_dir = subject_dir / vis_root_name / "debug_meshes"
        out_dir.mkdir(parents=True, exist_ok=True)
        obj_path = out_dir / f"{prefix}.obj"
        pytorch3d.io.save_obj(
            str(obj_path),
            V_world,
            torch.from_numpy(smpl_model.faces.astype(np.int32)).to(device_t),
        )

    def compute_per_cam_projection_and_error(J_world: torch.Tensor) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        smpl_proj_by_cam: Dict[str, np.ndarray] = {}
        per_cam_err: Dict[str, float] = {}
        if not kps_by_cam:

            return smpl_proj_by_cam, per_cam_err
        lr_pairs = [
            (2, 5), (3, 6), (4, 7),
            (9, 12), (10, 13), (11, 14),
            (15, 16), (17, 18),
            (19, 22), (20, 23), (21, 24),
        ]
        use_idx_set = set(use_idx)
        paired = [(a, b) for (a, b) in lr_pairs if (a in use_idx_set and b in use_idx_set)]
        paired_flat = set([i for ab in paired for i in ab])
        unpaired = [i for i in use_idx if i not in paired_flat]

        for cam_id in cam_ids:
            cam = calib[cam_id]
            kps = kps_by_cam[cam_id]
            kps_xy = torch.from_numpy(kps[:, :2]).to(device_t)
            kps_c = torch.from_numpy(kps[:, 2]).to(device_t)
            kps_c = torch.where(kps_c >= float(conf_threshold), kps_c, torch.zeros_like(kps_c))

            K = torch.from_numpy(cam["K"]).to(device_t)
            R = torch.from_numpy(cam["R"]).to(device_t)
            T = torch.from_numpy(cam["T"]).to(device_t)
            proj = _project_points(J_world, K, R, T)
            smpl_proj_by_cam[cam_id] = proj.detach().cpu().numpy().astype(np.float32)


            denom_ = 0.0
            total_ = 0.0


            for i in unpaired:
                c = float(kps[i, 2])
                if c < float(conf_threshold):
                    continue
                px, py = float(proj[i, 0]), float(proj[i, 1])
                gx, gy = float(kps[i, 0]), float(kps[i, 1])
                e = float(np.hypot(px - gx, py - gy))
                w = float(min(1.0, max(0.0, c)))
                total_ += w * e
                denom_ += w


            for a, b in paired:
                ca = float(kps[a, 2])
                cb = float(kps[b, 2])
                wa = float(min(1.0, max(0.0, ca))) if ca >= float(conf_threshold) else 0.0
                wb = float(min(1.0, max(0.0, cb))) if cb >= float(conf_threshold) else 0.0
                if wa <= 0.0 and wb <= 0.0:
                    continue
                pa = proj[a].detach().cpu().numpy()
                pb = proj[b].detach().cpu().numpy()
                ga = np.array(kps[a, :2], dtype=np.float32)
                gb = np.array(kps[b, :2], dtype=np.float32)
                e0 = wa * float(np.hypot(pa[0] - ga[0], pa[1] - ga[1])) + wb * float(np.hypot(pb[0] - gb[0], pb[1] - gb[1]))
                e1 = wa * float(np.hypot(pa[0] - gb[0], pa[1] - gb[1])) + wb * float(np.hypot(pb[0] - ga[0], pb[1] - ga[1]))
                total_ += float(min(e0, e1))
                denom_ += (wa + wb)

            per_cam_err[cam_id] = float(total_ / max(1.0, denom_)) if denom_ > 0 else 1e9
        return smpl_proj_by_cam, per_cam_err


    if vis:
        with torch.no_grad():
            Jw0, Vw0 = forward_body25_joints_world(
                smpl_params["betas"].detach(),
                smpl_params["global_orient"].detach(),
                smpl_params["transl"].detach(),
                _scale_pos().detach(),
            )
        smpl_proj0, err0 = compute_per_cam_projection_and_error(Jw0)
        smpl_faces = torch.from_numpy(smpl_model.faces.astype(np.int64)).to(device_t)
        _save_visualization_grids(
            subject_dir=subject_dir,
            stage_name="stage0",
            vis_root_name=vis_root_name,
            vis_sil_loss_mode=sil_loss_mode_,
            vis_inside_margin=float(inside_margin),
            cam_ids_all=cam_ids,
            calib=calib,
            kps_by_cam=kps_by_cam,
            smpl_proj_by_cam=smpl_proj0,
            per_cam_err=err0,
            conf_threshold=conf_threshold,
            show_indices=use_idx,
            dataset_mesh_verts=dataset_mesh_verts,
            dataset_mesh_faces=dataset_mesh_faces,
            smpl_mesh_verts=Vw0,
            smpl_mesh_faces=smpl_faces,
            neus_mesh_verts=V_neus,
            neus_mesh_faces=F_neus,
            vis_n_best=vis_n_best,
            vis_n_worst=vis_n_worst,
            vis_cols=vis_cols,
            vis_render_size=vis_render_size,
            vis_draw_errors=True,
            vis_save_tiles=vis_save_tiles,
            vis_save_mesh_ply=vis_save_mesh_ply,
        )


    if kps_by_cam:
        try:
            some_cam = cam_ids[0]
            payload = _load_keypoints_payload_for_camera(subject_dir, some_cam)
            logger.info(
                f"Keypoints JSON example ({some_cam}): keypoints_mode={payload.get('keypoints_mode')} conf_threshold={payload.get('conf_threshold')} "
                f"use_mask_head={[(i, payload.get('use_mask',[None]*25)[i]) for i in _face_body25_indices()]}"
            )
        except Exception:
            pass

    if scan_labels is not None:
        n_head = int((scan_labels == 1).sum())
        n_hands7 = int((scan_labels == 7).sum())
        n_hands8 = int((scan_labels == 8).sum())
        logger.info(
            f"NeuS2 labeled verts: head(label=1)={n_head} hands(label=7)={n_hands7} label8={n_hands8} total={scan_labels.shape[0]}"
        )

    def _stage_name_safe(s: str) -> str:
        return str(s).strip().replace(" ", "_").replace("/", "-").replace("\\", "-")


    if pipeline is None:
        pipeline = []
        pipeline.append(
            {
                "name": "stage1",
                "num_steps": int(iters_stage1),
                "lr": float(lr_stage1),
                "optimizer": "adam",
                "optimize": {"betas": True, "global_orient": True, "transl": True, "scale": True, "body_pose": False},
                "losses": {
                    "lambda_2d": float(lambda_2d_1),
                    "lambda_collision": float(lambda_col_1),
                    "lambda_inside": float(lambda_inside_stage1),
                    "lambda_sil": float(lambda_sil_stage1),
                    "lambda_betas": float(lambda_betas_1),
                    "lambda_pose_delta": 0.0,
                    "lambda_skin_head": float(lambda_skin_head_stage1),
                    "lambda_skin_hands": float(lambda_skin_hands_stage1),
                },
            }
        )
        if run_stage2:
            pipeline.append(
                {
                    "name": "stage2",
                    "num_steps": int(iters_stage2),
                    "lr": float(lr_stage2),
                    "optimizer": "adam",
                    "optimize": {"betas": True, "global_orient": True, "transl": True, "scale": True, "body_pose": True},
                    "losses": {
                        "lambda_2d": float(lambda_2d_2),
                        "lambda_collision": float(lambda_col_2),
                        "lambda_inside": float(lambda_inside_stage2),
                        "lambda_sil": float(lambda_sil_stage2),
                        "lambda_betas": float(lambda_betas_2),
                        "lambda_pose_delta": float(lambda_pose_delta_stage2),
                        "lambda_skin_head": float(lambda_skin_head_stage2),
                        "lambda_skin_hands": float(lambda_skin_hands_stage2),
                    },
                }
            )


    body_pose_var: torch.Tensor | None = None
    body_pose_init_local = None
    body_pose_delta = None
    pose_mask = None
    if "body_pose" in smpl_params:
        body_pose_init_local = smpl_params["body_pose"].detach().clone()
        pose_mask = _make_body_pose_mask(device_t, freeze_wrist_pose, freeze_feet_pose)
        body_pose_delta = torch.zeros_like(body_pose_init_local).requires_grad_(True)

    global_step = 0
    did_optimize_pose = False
    for stage_run_idx, st in enumerate(pipeline):
        base_stage_name = _stage_name_safe(str(st.get("name", f"stage{len(pipeline)}")))


        stage_tag = f"{stage_run_idx:02d}_{base_stage_name}"
        num_steps = int(st.get("num_steps", 0))
        lr = float(st.get("lr", lr_stage1))
        optimizer_name = str(st.get("optimizer", "adam")).lower()
        optimize = dict(st.get("optimize", {}))
        losses = dict(st.get("losses", {}))

        opt_betas = bool(optimize.get("betas", True))
        opt_global = bool(optimize.get("global_orient", True))
        opt_transl = bool(optimize.get("transl", True))
        opt_scale = bool(optimize.get("scale", True))
        opt_pose = bool(optimize.get("body_pose", False))
        did_optimize_pose = did_optimize_pose or opt_pose

        params: List[torch.Tensor] = []
        betas.requires_grad_(opt_betas)
        global_orient.requires_grad_(opt_global)
        transl.requires_grad_(opt_transl)
        scale_raw.requires_grad_(opt_scale)
        if opt_betas:
            params.append(betas)
        if opt_global:
            params.append(global_orient)
        if opt_transl:
            params.append(transl)
        if opt_scale:
            params.append(scale_raw)
        if body_pose_delta is not None:
            body_pose_delta.requires_grad_(opt_pose)
            if opt_pose:
                params.append(body_pose_delta)

        if num_steps <= 0 or not params:
            logger.warning(f"Skipping stage '{stage_tag}' (num_steps={num_steps}, params={len(params)})")
            continue

        logger.info(f"Stage '{stage_tag}': steps={num_steps} lr={lr} opt={optimizer_name} optimize={optimize} losses={list(losses.keys())}")


        betas_stage0 = betas.detach().clone()
        global_orient_stage0 = global_orient.detach().clone()
        transl_stage0 = transl.detach().clone()
        scale_stage0 = _scale_pos().detach().clone()

        if optimizer_name == "lbfgs":
            opt_obj = torch.optim.LBFGS(params, lr=float(lr), max_iter=20, line_search_fn="strong_wolfe")
        else:
            opt_obj = torch.optim.Adam(params, lr=float(lr))

        last = {}

        def _compute_loss(step_it: int) -> torch.Tensor:
            nonlocal body_pose_var, last


            if optimizer_name == "lbfgs":
                seed = int(12345 + stage_run_idx * 10000 + int(step_it))
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed)
                if device_t.type == "cuda":
                    torch.cuda.manual_seed_all(seed)

            if body_pose_init_local is not None and body_pose_delta is not None and pose_mask is not None:
                body_pose_var = body_pose_init_local + body_pose_delta * pose_mask
                Jw, Vw = forward_body25_joints_world(
                    betas, global_orient, transl, _scale_pos(), curr_body_pose=body_pose_var
                )
            else:
                body_pose_var = None
                Jw, Vw = forward_body25_joints_world(betas, global_orient, transl, _scale_pos())


            w2d_l = float(losses.get("lambda_2d", 0.0))
            if w2d_l > 0.0:
                if not kps_by_cam:
                    raise RuntimeError("lambda_2d>0 but no keypoints were loaded. Add keypoints_2d or set lambda_2d=0.")
                l2d = loss_2d(Jw)
            else:
                l2d = torch.tensor(0.0, device=device_t)
            lcol, linside = loss_collision_and_inside(
                Vw,
                sample_n=int(collision_sample_n),
                lambda_inside_now=float(losses.get("lambda_inside", 0.0)),
                debug_tag=stage_tag,
                debug_step=step_it,
            )
            wsil_l = float(losses.get("lambda_sil", 0.0))
            if wsil_l > 0.0:
                lsil = loss_silhouette_raw(Vw)
            else:
                lsil = torch.tensor(0.0, device=device_t)
            lskin_head, lskin_hands = skin_tight_fit_loss_raw(Vw)
            lreg = loss_reg(betas)
            lpose = torch.tensor(0.0, device=device_t)
            if body_pose_var is not None and body_pose_init is not None and float(losses.get("lambda_pose_delta", 0.0)) > 0.0:
                lpose = (body_pose_var - body_pose_init).pow(2).mean()


            w2d = w2d_l * l2d
            wcol = float(losses.get("lambda_collision", 0.0)) * lcol
            winside = float(losses.get("lambda_inside", 0.0)) * linside
            wsil = wsil_l * lsil
            wreg = float(losses.get("lambda_betas", 0.0)) * lreg
            wpose = float(losses.get("lambda_pose_delta", 0.0)) * lpose
            wskin = float(losses.get("lambda_skin_head", 0.0)) * lskin_head + float(losses.get("lambda_skin_hands", 0.0)) * lskin_hands
            lskin = lskin_head + lskin_hands

            total = w2d + wcol + winside + wsil + wskin + wreg + wpose
            last = dict(
                total=total,
                l2d=l2d,
                lcol=lcol,
                linside=linside,
                lsil=lsil,
                lskin=lskin,
                lreg=lreg,
                lpose=lpose,
                w2d=w2d,
                wcol=wcol,
                winside=winside,
                wsil=wsil,
                wskin=wskin,
                wreg=wreg,
                wpose=wpose,
            )
            return total

        for it in range(num_steps):
            if optimizer_name == "lbfgs":
                def closure():
                    opt_obj.zero_grad(set_to_none=True)
                    loss_val = _compute_loss(it)
                    loss_val.backward()
                    return loss_val
                opt_obj.step(closure)

                loss = last["total"]
            else:
                opt_obj.zero_grad(set_to_none=True)
                loss = _compute_loss(it)
                loss.backward()
                opt_obj.step()

            _log(
                step=int(global_step),
                stage=stage_tag,
                it=int(it),
                scalars={
                    "total": float(loss.item()),
                    "l2d": float(last["l2d"].item()),
                    "lcol": float(last["lcol"].item()),
                    "linside": float(last["linside"].item()),
                    "lsil": float(last["lsil"].item()),
                    "lskin": float(last["lskin"].item()),
                    "lreg": float(last["lreg"].item()),
                    "lpose": float(last["lpose"].item()),
                    "w2d": float(last["w2d"].item()),
                    "wcol": float(last["wcol"].item()),
                    "winside": float(last["winside"].item()),
                    "wsil": float(last["wsil"].item()),
                    "wskin": float(last["wskin"].item()),
                    "wreg": float(last["wreg"].item()),
                    "wpose": float(last["wpose"].item()),
                    "scale": float(_scale_pos().item()),
                    "betas_l2": float(torch.linalg.norm(betas.detach()).item()),
                    "betas_delta_l2": float(torch.linalg.norm((betas.detach() - betas_stage0)).item()),
                    "global_orient_delta_l2": float(torch.linalg.norm((global_orient.detach() - global_orient_stage0)).item()),
                    "transl_l2": float(torch.linalg.norm(transl.detach()).item()),
                    "transl_delta_l2": float(torch.linalg.norm((transl.detach() - transl_stage0)).item()),
                    "scale_delta": float((_scale_pos().detach() - scale_stage0).abs().item()),
                    "outside_frac": float(getattr(loss_collision_and_inside, "last_outside_frac", np.nan)),
                    "outside_mean": float(getattr(loss_collision_and_inside, "last_outside_mean", np.nan)),
                    "inside_frac": float(getattr(loss_collision_and_inside, "last_inside_frac", np.nan)),
                    "inside_mean": float(getattr(loss_collision_and_inside, "last_inside_mean", np.nan)),
                    "dist_mean": float(getattr(loss_collision_and_inside, "last_dist_mean", np.nan)),
                    "frac_on_face_best": float(getattr(loss_collision_and_inside, "last_frac_on_face_best", np.nan)),
                    "lcol_raw": float(getattr(loss_collision_and_inside, "last_lcol_raw", np.nan)),
                    "lcol_scaled": float(last["wcol"].item()),
                    "linside_raw": float(getattr(loss_collision_and_inside, "last_linside_raw", np.nan)),
                    "linside_scaled": float(last["winside"].item()),
                },
            )

            global_step += 1
            if it % 25 == 0 or it == num_steps - 1:
                logger.info(
                    f"[{stage_tag} {it:04d}] total={last['total'].item():.6f} 2d={last['l2d'].item():.6f} "
                    f"col={last['lcol'].item():.6f} inside={last['linside'].item():.6f} "
                    f"skin={last['lskin'].item():.6f} reg={last['lreg'].item():.6f} pose={last['lpose'].item():.6f} scale={_scale_pos().item():.4f}"
                )

        with torch.no_grad():
            if body_pose_var is not None:
                Jwf, Vwf = forward_body25_joints_world(
                    betas, global_orient, transl, _scale_pos(), curr_body_pose=body_pose_var
                )
            else:
                Jwf, Vwf = forward_body25_joints_world(betas, global_orient, transl, _scale_pos())
        save_debug(f"smpl_body_{stage_tag}", Jwf, Vwf)
        if vis:
            smpl_proj, err = compute_per_cam_projection_and_error(Jwf)
            smpl_faces = torch.from_numpy(smpl_model.faces.astype(np.int64)).to(device_t)
            _save_visualization_grids(
                subject_dir=subject_dir,
                stage_name=stage_tag,
                vis_root_name=vis_root_name,
                vis_sil_loss_mode=sil_loss_mode_,
                vis_inside_margin=float(inside_margin),
                cam_ids_all=cam_ids,
                calib=calib,
                kps_by_cam=kps_by_cam,
                smpl_proj_by_cam=smpl_proj,
                per_cam_err=err,
                conf_threshold=conf_threshold,
                show_indices=use_idx,
                dataset_mesh_verts=dataset_mesh_verts,
                dataset_mesh_faces=dataset_mesh_faces,
                smpl_mesh_verts=Vwf,
                smpl_mesh_faces=smpl_faces,
                neus_mesh_verts=V_neus,
                neus_mesh_faces=F_neus,
                vis_n_best=vis_n_best,
                vis_n_worst=vis_n_worst,
                vis_cols=vis_cols,
                vis_render_size=vis_render_size,
                vis_draw_errors=True,
                vis_save_tiles=vis_save_tiles,
                vis_save_mesh_ply=vis_save_mesh_ply,
            )


    out_params = {k: v.detach().cpu().numpy() for k, v in smpl_params.items()}
    out_params["betas"] = betas.detach().cpu().numpy()
    out_params["global_orient"] = global_orient.detach().cpu().numpy()
    out_params["transl"] = transl.detach().cpu().numpy()
    out_params["scale"] = _scale_pos().detach().cpu().numpy().reshape(1, 1)
    if body_pose_var is not None:
        out_params["body_pose"] = body_pose_var.detach().cpu().numpy()

    if out_smpl_params_path is None:
        if pipeline is not None:
            out_smpl_params_path = subject_dir / f"smpl_params_finetuned_{resolved_gender}{exp_suffix}.npz"
        else:
            out_smpl_params_path = subject_dir / (
                (f"smpl_params_finetuned_{resolved_gender}{exp_suffix}.npz" if run_stage2 else f"smpl_params_finetuned_stage1_{resolved_gender}{exp_suffix}.npz")
            )
    np.savez(str(out_smpl_params_path), **out_params)
    logger.info(f"Saved finetuned SMPL params to: {out_smpl_params_path}")


    try:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
    except Exception:
        pass
    try:
        if csv_fp is not None:
            csv_fp.close()
    except Exception:
        pass
    try:
        if wandb_run is not None:
            wandb_run.finish()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Finetune MVHumanNet SMPL-X params using NeuS2 mesh + 2D OpenPose.")
    parser.add_argument("--subject_dir", type=str, required=True, help="Path to AvatarREX subject-frame dir (e.g., .../100001_1185)")
    parser.add_argument(
        "--gender",
        type=str,
        default="auto",
        choices=["auto", "male", "female", "neutral"],
        help="SMPL-X gender. 'auto' reads <subject_dir>/gender.txt if present, else uses neutral.",
    )
    parser.add_argument(
        "--smpl_model_path",
        type=str,
        default=get_smplx_model_path(),
        help="SMPL models root containing smplx/; defaults to AVATARMIX_ASSET_ROOT/smpl.",
    )
    parser.add_argument("--neus_mesh_path", type=str, required=True, help="Path to watertight NeuS2 mesh (e.g., mesh/trimesh_cleaned/0000.obj)")
    parser.add_argument("--calib_path", type=str, default=None, help="Path to calibration_full.json (default: <subject_dir>/calibration_full.json)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--iters_stage1", type=int, default=400)
    parser.add_argument("--iters_stage2", type=int, default=200)
    parser.add_argument("--lr_stage1", type=float, default=5e-2)
    parser.add_argument("--lr_stage2", type=float, default=2e-2)
    parser.add_argument("--conf_threshold", type=float, default=0.2)
    parser.add_argument("--keypoints_mode", type=str, default="stable", choices=["stable", "all"])
    parser.add_argument("--lambda_2d", type=float, default=1.0)
    parser.add_argument("--lambda_collision", type=float, default=0.2)
    parser.add_argument("--lambda_reg", type=float, default=0.01, help="(Deprecated) Beta L2 weight used for both stages unless per-stage weights are set.")
    parser.add_argument("--lambda_betas_stage1", type=float, default=None, help="Beta L2 weight for stage1 (overrides --lambda_reg).")
    parser.add_argument("--lambda_betas_stage2", type=float, default=None, help="Beta L2 weight for stage2 (overrides --lambda_reg).")
    parser.add_argument("--lambda_2d_stage1", type=float, default=None, help="2D keypoint weight for stage1 (overrides --lambda_2d).")
    parser.add_argument("--lambda_2d_stage2", type=float, default=None, help="2D keypoint weight for stage2 (overrides --lambda_2d).")
    parser.add_argument("--lambda_collision_stage1", type=float, default=None, help="Collision weight for stage1 (overrides --lambda_collision).")
    parser.add_argument("--lambda_collision_stage2", type=float, default=None, help="Collision weight for stage2 (overrides --lambda_collision).")
    parser.add_argument(
        "--collision_loss_mode",
        type=str,
        default="outside_dist2",
        choices=["outside", "outside2", "outside_dist", "outside_dist2"],
        help="Collision loss formulation before scaling: outside_mean, outside^2, outside*dist, or outside*dist^2.",
    )
    parser.add_argument("--lambda_sil_stage1", type=float, default=0.0, help="Silhouette loss weight in stage1 (GT mask supervision).")
    parser.add_argument("--lambda_sil_stage2", type=float, default=0.0, help="Silhouette loss weight in stage2 (GT mask supervision).")
    parser.add_argument("--sil_render_size", type=int, default=256, help="Silhouette loss render resolution (square).")
    parser.add_argument("--sil_cam_sample", type=int, default=8, help="Number of cameras sampled per iteration for silhouette loss.")
    parser.add_argument(
        "--sil_loss_mode",
        type=str,
        default="one_sided_outside",
        choices=["iou", "bce", "one_sided_outside"],
        help="Silhouette loss type: IoU, BCE, or one-sided outside penalty (SMPL outside GT mask).",
    )
    parser.add_argument("--sil_sigma", type=float, default=1e-4, help="Soft silhouette sigma (PyTorch3D BlendParams).")
    parser.add_argument("--sil_gamma", type=float, default=1e-4, help="Soft silhouette gamma (PyTorch3D BlendParams).")
    parser.add_argument("--sil_faces_per_pixel", type=int, default=50, help="faces_per_pixel for soft silhouette rasterization.")
    parser.add_argument("--collision_sample_n", type=int, default=20000, help="Number of SMPL vertices sampled for collision loss.")
    parser.add_argument("--collision_face_knn_k", type=int, default=64, help="Candidate triangle faces per query for collision (KNN over face centers).")
    parser.add_argument(
        "--collision_backend",
        type=str,
        default="pytorch_knn_faces",
        choices=["pytorch_knn_faces", "cuda_point_face"],
        help="Closest-point backend for collision: python KNN-over-face-centers vs CUDA PointFaceDistanceForward (reshape_ops).",
    )
    parser.add_argument("--cuda_nn_scale_factor", type=float, default=1000.0, help="Scale factor for cuda_point_face distance stability.")
    parser.add_argument("--collision_debug_vis", action="store_true", help="Save collision closest-point debug visualization (points+lines).")
    parser.add_argument("--collision_debug_vis_n", type=int, default=2000, help="Number of query points to visualize for collision debug.")
    parser.add_argument("--collision_debug_every", type=int, default=100, help="Save debug every N optimization steps (stage1/stage2).")
    parser.add_argument("--collision_debug_save_mesh", action="store_true", help="Also save NeuS2 mesh OBJ once under collision_debug/.")
    parser.add_argument("--collision_debug_cyl_radius", type=float, default=0.002, help="Cylinder radius for collision debug lines (in world units).")
    parser.add_argument("--collision_debug_cyl_sections", type=int, default=8, help="Cylinder radial sections for collision debug.")
    parser.add_argument("--collision_debug_sphere_radius", type=float, default=0.004, help="Sphere radius for collision debug endpoints.")
    parser.add_argument(
        "--collision_debug_mode",
        type=str,
        default="topk_outside",
        choices=["topk_outside", "topk_inside", "topk_both", "random"],
        help="Which collision query points to visualize (outside / too-inside / both / random).",
    )
    parser.add_argument("--lambda_inside_stage1", type=float, default=0.0, help="Inside-margin band loss weight for stage1 (penalize too-inside).")
    parser.add_argument("--lambda_inside_stage2", type=float, default=0.0, help="Inside-margin band loss weight for stage2 (penalize too-inside).")
    parser.add_argument("--inside_margin", type=float, default=0.005, help="Inside margin (world units). Penalize when signed < -margin.")
    parser.add_argument(
        "--inside_part_weights",
        type=str,
        default="leftArm,rightArm,leftForeArm,rightForeArm,leftLeg,rightLeg,leftUpLeg,rightUpLeg,leftFoot,rightFoot,hips,spine1",
        help="Comma list of SMPL-X seg parts (optionally part:weight) to upweight inside loss.",
    )
    parser.add_argument(
        "--inside_exclude_parts",
        type=str,
        default="spine,spine2,leftShoulder,rightShoulder,neck,head",
        help="Comma list of SMPL-X seg parts to force weight=0 for inside loss (e.g., chest-ish).",
    )
    parser.add_argument("--inside_default_weight", type=float, default=0.0, help="Default inside-loss weight for vertices not in inside_part_weights.")
    parser.add_argument("--inside_debug_vis", action="store_true", help="Export colored pointcloud of (outside vs too-inside) active vertices.")
    parser.add_argument("--inside_debug_every", type=int, default=100, help="Export inside debug pointcloud every N steps.")
    parser.add_argument("--inside_debug_topk", type=int, default=8000, help="Number of active points exported in inside debug pointcloud.")
    parser.add_argument("--label_pkl_path", type=str, default=None, help="Path to label-f0000_extended.pkl (default: subject_dir/mesh/labeled_no_sam/label-f0000_extended.pkl)")
    parser.add_argument("--lambda_skin_head_stage1", type=float, default=0.0)
    parser.add_argument("--lambda_skin_hands_stage1", type=float, default=0.0)
    parser.add_argument("--lambda_skin_head_stage2", type=float, default=0.2)
    parser.add_argument("--lambda_skin_hands_stage2", type=float, default=0.2)
    parser.add_argument("--skin_sample_head", type=int, default=8000)
    parser.add_argument("--skin_sample_hands", type=int, default=4000)
    parser.add_argument("--lambda_pose_delta_stage2", type=float, default=1e-3, help="L2 regularizer on (body_pose - init_body_pose) in stage2.")

    parser.add_argument("--freeze_wrist_pose", action="store_true", help="Freeze left/right wrist pose during pose optimization.")
    parser.add_argument("--freeze_feet_pose", action="store_true", help="Freeze ankle+foot pose during pose optimization.")
    parser.add_argument("--freeze_wrist_pose_stage2", dest="freeze_wrist_pose", action="store_true", help="(Deprecated) Use --freeze_wrist_pose")
    parser.add_argument("--freeze_feet_pose_stage2", dest="freeze_feet_pose", action="store_true", help="(Deprecated) Use --freeze_feet_pose")
    parser.add_argument(
        "--no_face_joints_in_2d",
        dest="include_face_joints_in_2d",
        action="store_false",
        help="Disable adding sparse face joints (nose/eyes/ears) to the 2D loss when keypoints_mode=stable.",
    )
    parser.add_argument("--run_stage2", action="store_true", help="Also optimize body_pose in stage2")
    parser.add_argument("--save_debug_meshes", action="store_true", help="Save debug meshes under smplx_finetune_debug/")
    parser.add_argument("--out_smpl_params", type=str, default=None, help="Output npz path (default: subject_dir/smpl_params_finetuned*.npz)")
    parser.add_argument("--vis", action="store_true", help="Save compact visualization grids under smplx_finetune_vis/")
    parser.add_argument("--vis_n_best", type=int, default=12, help="Number of best (lowest-error) cameras to visualize.")
    parser.add_argument("--vis_n_worst", type=int, default=12, help="Number of worst (highest-error) cameras to visualize.")
    parser.add_argument("--vis_cols", type=int, default=4, help="Number of columns in visualization grids.")
    parser.add_argument("--vis_render_size", type=int, default=512, help="Silhouette render resolution (square).")
    parser.add_argument("--vis_save_tiles", action="store_true", help="Also save per-camera tiles for visualized cams.")
    parser.add_argument(
        "--vis_save_mesh_ply",
        action="store_true",
        help=(
            "Save mesh PLYs for visualization: stage0 exports NeuS2 (cyan) and init SMPL (magenta) separately; "
            "stage1/stage2 export optimized SMPL (red)."
        ),
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Logging directory root (default: <subject_dir>/smplx_finetune_logs/<timestamp>/)")
    parser.add_argument("--log_every", type=int, default=10, help="Log losses every N iterations (per stage).")
    parser.add_argument("--no_tensorboard", dest="log_tensorboard", action="store_false", help="Disable TensorBoard logging.")
    parser.add_argument("--wandb", dest="log_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="mvhumannet_smplx_finetune")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--exp_name", type=str, default="", help="Optional experiment name suffix appended to vis/debug dirs and default output npz.")

    args = parser.parse_args()
    subject_dir = Path(args.subject_dir)
    calib_path = Path(args.calib_path) if args.calib_path else (subject_dir / "calibration_full.json")
    out_path = Path(args.out_smpl_params) if args.out_smpl_params else None

    run_finetune(
        subject_dir=subject_dir,
        smpl_model_path=Path(args.smpl_model_path) if args.smpl_model_path else Path(""),
        neus_mesh_path=Path(args.neus_mesh_path),
        calib_path=calib_path,
        device=args.device,
        gender=str(args.gender),
        iters_stage1=args.iters_stage1,
        iters_stage2=args.iters_stage2,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        conf_threshold=args.conf_threshold,
        keypoints_mode=args.keypoints_mode,
        lambda_2d=args.lambda_2d,
        lambda_collision=args.lambda_collision,
        lambda_reg=args.lambda_reg,
        lambda_betas_stage1=args.lambda_betas_stage1,
        lambda_betas_stage2=args.lambda_betas_stage2,
        lambda_2d_stage1=args.lambda_2d_stage1,
        lambda_2d_stage2=args.lambda_2d_stage2,
        lambda_collision_stage1=args.lambda_collision_stage1,
        lambda_collision_stage2=args.lambda_collision_stage2,
        collision_loss_mode=str(args.collision_loss_mode),
        lambda_inside_stage1=float(args.lambda_inside_stage1),
        lambda_inside_stage2=float(args.lambda_inside_stage2),
        inside_margin=float(args.inside_margin),
        inside_part_weights=str(args.inside_part_weights),
        inside_exclude_parts=str(args.inside_exclude_parts),
        inside_default_weight=float(args.inside_default_weight),
        inside_debug_vis=bool(args.inside_debug_vis),
        inside_debug_every=int(args.inside_debug_every),
        inside_debug_topk=int(args.inside_debug_topk),
        lambda_sil_stage1=float(args.lambda_sil_stage1),
        lambda_sil_stage2=float(args.lambda_sil_stage2),
        sil_render_size=int(args.sil_render_size),
        sil_cam_sample=int(args.sil_cam_sample),
        sil_loss_mode=str(args.sil_loss_mode),
        sil_sigma=float(args.sil_sigma),
        sil_gamma=float(args.sil_gamma),
        sil_faces_per_pixel=int(args.sil_faces_per_pixel),
        label_pkl_path=Path(args.label_pkl_path) if args.label_pkl_path else None,
        lambda_skin_head_stage1=args.lambda_skin_head_stage1,
        lambda_skin_hands_stage1=args.lambda_skin_hands_stage1,
        lambda_skin_head_stage2=args.lambda_skin_head_stage2,
        lambda_skin_hands_stage2=args.lambda_skin_hands_stage2,
        skin_sample_head=args.skin_sample_head,
        skin_sample_hands=args.skin_sample_hands,
        collision_sample_n=args.collision_sample_n,
        collision_face_knn_k=args.collision_face_knn_k,
        collision_debug_vis=bool(args.collision_debug_vis),
        collision_debug_vis_n=int(args.collision_debug_vis_n),
        collision_debug_every=int(args.collision_debug_every),
        collision_debug_save_mesh=bool(args.collision_debug_save_mesh),
        collision_debug_cyl_radius=float(args.collision_debug_cyl_radius),
        collision_debug_cyl_sections=int(args.collision_debug_cyl_sections),
        collision_debug_sphere_radius=float(args.collision_debug_sphere_radius),
        collision_debug_mode=str(args.collision_debug_mode),
        collision_backend=str(args.collision_backend),
        cuda_nn_scale_factor=float(args.cuda_nn_scale_factor),
        lambda_pose_delta_stage2=args.lambda_pose_delta_stage2,
        include_face_joints_in_2d=bool(args.include_face_joints_in_2d),
        freeze_wrist_pose=bool(args.freeze_wrist_pose),
        freeze_feet_pose=bool(args.freeze_feet_pose),
        run_stage2=bool(args.run_stage2),
        save_debug_meshes=bool(args.save_debug_meshes),
        out_smpl_params_path=out_path,
        vis=bool(args.vis),
        vis_n_best=int(args.vis_n_best),
        vis_n_worst=int(args.vis_n_worst),
        vis_cols=int(args.vis_cols),
        vis_render_size=int(args.vis_render_size),
        vis_save_tiles=bool(args.vis_save_tiles),
        vis_save_mesh_ply=bool(args.vis_save_mesh_ply),
        log_dir=Path(args.log_dir) if args.log_dir else None,
        log_every=int(args.log_every),
        log_tensorboard=bool(args.log_tensorboard),
        log_wandb=bool(args.log_wandb),
        wandb_project=str(args.wandb_project),
        wandb_entity=str(args.wandb_entity) if args.wandb_entity else None,
        wandb_name=str(args.wandb_name) if args.wandb_name else None,
        wandb_mode=str(args.wandb_mode),
        exp_name=str(args.exp_name),
    )


if __name__ == "__main__":
    main()
