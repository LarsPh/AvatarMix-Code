import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def _read_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

_TALKBODY4D_SUBJECT_GENDER = {

    "CC_01": "female",
    "CC_02": "female",
    "SC_01": "male",
    "SC_02": "male",
    "SQ_01": "female",
    "SQ_02": "female",
    "XG_01": "male",
    "XG_02": "male",
}


def _resolve_talkbody4d_gender(subject_id: str, gender_arg: str) -> str:

    g = str(gender_arg).strip().lower()
    if g in {"male", "female", "neutral"}:
        return g
    if g != "auto":
        return "neutral"
    return _TALKBODY4D_SUBJECT_GENDER.get(str(subject_id).strip(), "neutral")


def _list_cameras_from_raw_videos(raw_videos_dir: Path) -> List[str]:
    cams: List[str] = []
    for p in sorted(raw_videos_dir.glob("*.mp4")):
        cams.append(p.stem)
    return cams


def _normalize_cam_id(cam: str) -> str:
    s = str(cam).strip()
    if s.lower().endswith(".mp4"):
        s = s[:-4]
    s = s.strip()
    if s.isdigit():

        return s.zfill(3)
    return s


def _load_cameras(calib_path: Path) -> Dict[str, Dict[str, np.ndarray]]:

    calib = _read_json(calib_path)
    cameras: Dict[str, Dict[str, np.ndarray]] = {}

    for cam_id, c in calib["cameras"].items():
        image_size = c["image_size"]
        K = np.array(c["K"], dtype=np.float32).reshape(3, 3)
        dist = np.array(c["dist"], dtype=np.float32).reshape(-1)
        cameras[cam_id] = {
            "K": K,
            "dist": dist,
            "w": np.int32(image_size[0]),
            "h": np.int32(image_size[1]),
        }

    for cam_id_key, p in calib["camera_poses"].items():

        if "to" in cam_id_key:
            cam_id = cam_id_key[: cam_id_key.find("to") - 1]
        else:
            cam_id = cam_id_key
        if cam_id not in cameras:
            continue
        R = np.array(p["R"], dtype=np.float32).reshape(3, 3)
        T = np.array(p["T"], dtype=np.float32).reshape(3)
        cameras[cam_id].update({"R": R, "T": T})

    return cameras


def _axis_angle_to_R(axis_angle: np.ndarray) -> np.ndarray:

    aa = axis_angle.astype(np.float32).reshape(3)
    theta = np.linalg.norm(aa) + 1e-8
    axis = aa / theta
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float32)
    I = np.eye(3, dtype=np.float32)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def _compute_bbox_from_binary_mask(
    mask: np.ndarray,
    dilate_ratio: float = 0.05,
) -> Optional[Tuple[int, int, int, int]]:

    if mask.ndim == 3:
        mask = mask[..., 0]
    fg = mask > 0
    if not np.any(fg):
        return None
    ys, xs = np.where(fg)
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1

    w = max(1, right - left)
    h = max(1, bottom - top)
    d = int(round(max(w, h) * float(dilate_ratio)))

    H, W = mask.shape[:2]
    left = max(0, min(left - d, W - 1))
    top = max(0, min(top - d, H - 1))
    right = max(left + 1, min(right + d, W))
    bottom = max(top + 1, min(bottom + d, H))
    return (left, top, right, bottom)


def _undistort_image_and_mask(
    img_bgr: np.ndarray,
    mask_gray: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    undistort_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    if undistort_maps is None:
        h, w = mask_gray.shape[:2]
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
    else:
        map1, map2 = undistort_maps


    img_und = cv2.remap(img_bgr, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask_und = cv2.remap(mask_gray, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_und, mask_und


def _build_undistort_maps(K: np.ndarray, dist: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:

    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (int(w), int(h)), cv2.CV_32FC1)
    return map1, map2


def _compute_downscale_K_and_size(K: np.ndarray, w: int, h: int, factor: float) -> Tuple[np.ndarray, int, int, Dict]:

    s = float(factor)
    if not (0.0 < s <= 1.0):
        raise ValueError(f"downscale_factor must be in (0,1], got {factor}")
    new_w = max(1, int(round(int(w) * s)))
    new_h = max(1, int(round(int(h) * s)))
    K_out = K.copy()
    K_out[0, 0] *= s
    K_out[1, 1] *= s
    K_out[0, 2] = (float(K[0, 2]) + 0.5) * s - 0.5
    K_out[1, 2] = (float(K[1, 2]) + 0.5) * s - 0.5
    tfm = {"factor": float(s), "src_wh": [int(w), int(h)], "dst_wh": [int(new_w), int(new_h)]}
    return K_out, int(new_w), int(new_h), tfm


def _read_frames_sequential(
    cap: cv2.VideoCapture,
    frame_ids_sorted: List[int],
) -> Dict[int, np.ndarray]:

    out: Dict[int, np.ndarray] = {}
    if not frame_ids_sorted:
        return out
    first = int(frame_ids_sorted[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(first))
    current = first
    for fid in frame_ids_sorted:
        fid_i = int(fid)
        if fid_i < current:

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fid_i))
            current = fid_i

        skip_n = int(fid_i - current)
        for _ in range(max(0, skip_n)):
            cap.grab()
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        out[fid_i] = frame_bgr
        current = fid_i + 1
    return out


def _apply_downscale(
    img_bgr: np.ndarray,
    mask_gray: np.ndarray,
    K: np.ndarray,
    factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:

    s = float(factor)
    if not (0.0 < s <= 1.0):
        raise ValueError(f"downscale_factor must be in (0,1], got {factor}")

    h, w = img_bgr.shape[:2]
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    img_r = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask_gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    K_out = K.copy()
    K_out[0, 0] *= s
    K_out[1, 1] *= s

    K_out[0, 2] = (float(K[0, 2]) + 0.5) * s - 0.5
    K_out[1, 2] = (float(K[1, 2]) + 0.5) * s - 0.5

    tfm = {"factor": float(s), "src_wh": [int(w), int(h)], "dst_wh": [int(new_w), int(new_h)]}
    return img_r, mask_r, K_out, tfm


def _apply_crop_resize_pad(
    img_bgr: np.ndarray,
    mask_gray: np.ndarray,
    K: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    target_long_side: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:

    left, top, right, bottom = crop_box
    img_c = img_bgr[top:bottom, left:right]
    mask_c = mask_gray[top:bottom, left:right]
    crop_h, crop_w = img_c.shape[0], img_c.shape[1]


    K_crop = K.copy()
    K_crop[0, 2] -= float(left)
    K_crop[1, 2] -= float(top)

    long_side = max(crop_w, crop_h)
    resize_scale = float(target_long_side) / float(long_side)
    new_w = int(round(crop_w * resize_scale))
    new_h = int(round(crop_h * resize_scale))
    new_w = max(1, min(target_long_side, new_w))
    new_h = max(1, min(target_long_side, new_h))

    scale_x = new_w / float(crop_w)
    scale_y = new_h / float(crop_h)

    img_r = cv2.resize(img_c, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask_c, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


    K_res = K_crop.copy()
    K_res[0, 0] *= scale_x
    K_res[1, 1] *= scale_y
    K_res[0, 2] = (float(K_crop[0, 2]) + 0.5) * scale_x - 0.5
    K_res[1, 2] = (float(K_crop[1, 2]) + 0.5) * scale_y - 0.5


    target = int(target_long_side)
    pad_left = pad_top = 0
    if new_w > new_h:
        pad_top = (target - new_h) // 2
        pad_bottom = target - new_h - pad_top
        pad_left = 0
        pad_right = 0
    elif new_h > new_w:
        pad_left = (target - new_w) // 2
        pad_right = target - new_w - pad_left
        pad_top = 0
        pad_bottom = 0
    else:
        pad_left = pad_right = pad_top = pad_bottom = 0

    img_p = cv2.copyMakeBorder(img_r, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask_p = cv2.copyMakeBorder(mask_r, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    assert img_p.shape[0] == target and img_p.shape[1] == target
    assert mask_p.shape[0] == target and mask_p.shape[1] == target

    K_final = K_res.copy()
    K_final[0, 2] += float(pad_left)
    K_final[1, 2] += float(pad_top)

    tfm = {
        "crop_box": [int(left), int(top), int(right), int(bottom)],
        "crop_wh": [int(crop_w), int(crop_h)],
        "new_wh": [int(new_w), int(new_h)],
        "scale_xy": [float(scale_x), float(scale_y)],
        "pad_left_top": [int(pad_left), int(pad_top)],
        "target_wh": [int(target), int(target)],
    }
    return img_p, mask_p, K_final, tfm


def _read_video_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None
    return frame_bgr


def _open_video_capture(video_path: Path) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    return cap


def _read_video_frame_from_cap(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        return None
    return frame_bgr


def _get_video_frame_count(video_path: Path) -> Optional[int]:
    cap = _open_video_capture(video_path)
    if cap is None:
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception:
        n = None
    try:
        cap.release()
    except Exception:
        pass
    if n is None or n <= 0:
        return None
    return n


def _load_talkbody4d_smplx_params(params_path: Path) -> Dict[str, np.ndarray]:
    params = _read_json(params_path)

    def _as_vec(name: str) -> np.ndarray:
        v = np.array(params[name], dtype=np.float32)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        return v

    betas = _as_vec("betas")
    expression = _as_vec("expression")
    body_pose = _as_vec("body_pose")
    left_hand_pose = _as_vec("left_hand_pose")
    right_hand_pose = _as_vec("right_hand_pose")
    jaw_pose = _as_vec("jaw_pose")
    leye_pose = _as_vec("leye_pose")
    reye_pose = _as_vec("reye_pose")


    global_orient = _as_vec("global_orient") if "global_orient" in params else np.zeros((3,), dtype=np.float32)
    transl = _as_vec("transl") if "transl" in params else np.zeros((3,), dtype=np.float32)
    Rh = _as_vec("Rh") if "Rh" in params else np.zeros((3,), dtype=np.float32)
    Th = _as_vec("Th") if "Th" in params else np.zeros((3,), dtype=np.float32)


    v_shape = None
    if "v_shape" in params:
        v_shape_arr = np.array(params["v_shape"], dtype=np.float32)

        if v_shape_arr.ndim == 3 and v_shape_arr.shape[0] == 1:
            v_shape_arr = v_shape_arr[0]
        v_shape = v_shape_arr
    v_pose = None
    if "v_pose" in params:
        v_pose_arr = np.array(params["v_pose"], dtype=np.float32)
        if v_pose_arr.ndim == 3 and v_pose_arr.shape[0] == 1:
            v_pose_arr = v_pose_arr[0]
        v_pose = v_pose_arr

    out = {
        "betas": betas,
        "expression": expression,
        "body_pose": body_pose,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,

        "global_orient": global_orient,
        "transl": transl,

        "Rh": Rh,
        "Th": Th,
    }
    if v_shape is not None:
        out["v_shape"] = v_shape
    if v_pose is not None:
        out["v_pose"] = v_pose
    return out


def _analyze_crop_stats_for_subject_frame(
    subject_dir: Path,
    cameras: Dict[str, Dict[str, np.ndarray]],
    frame_idx: int,
    dilate_ratio: float,
) -> Dict[str, float]:

    raw_videos_dir = subject_dir / "raw_videos"
    mattings_dir = subject_dir / "mattings"
    cams = _list_cameras_from_raw_videos(raw_videos_dir)

    long_sides: List[int] = []
    areas: List[int] = []
    used = 0
    for cam in cams:
        if cam not in cameras:
            continue
        mask_path = mattings_dir / cam / f"{frame_idx:06d}.png"
        video_path = raw_videos_dir / f"{cam}.mp4"
        if not mask_path.exists() or not video_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        K = cameras[cam]["K"]
        dist = cameras[cam]["dist"]

        h, w = mask.shape[:2]
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
        mask_und = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bbox = _compute_bbox_from_binary_mask(mask_und, dilate_ratio=dilate_ratio)
        if bbox is None:
            continue
        l, t, r, b = bbox
        bw = r - l
        bh = b - t
        long_sides.append(int(max(bw, bh)))
        areas.append(int(bw * bh))
        used += 1

    if not long_sides:
        return {"num_cams_used": 0.0}
    arr = np.array(long_sides, dtype=np.float32)
    return {
        "num_cams_used": float(used),
        "long_side_max": float(arr.max()),
        "long_side_p90": float(np.percentile(arr, 90)),
        "long_side_median": float(np.median(arr)),
        "area_median": float(np.median(np.array(areas, dtype=np.float32))),
    }


def convert_dataset(
    talkbody4d_root: str,
    output_avatarrex_root: str,
    subjects: List[str],
    frames: List[int],
    use_all_frames_if_unspecified: bool,
    export_mode: str,
    frame_padding: int,
    transform_mode: str,
    downscale_factor: float,
    crop_dilate_ratio: float,
    target_long_side: int,
    analyze_crop_stats_only: bool,
    print_video_stats_only: bool,
    overwrite: bool,
    gender: str,
    smplx_only: bool,
    exclude_cams: Optional[List[str]] = None,
):
    tb_root = Path(talkbody4d_root)


    data_root = tb_root / "data" if (tb_root / "data").exists() else tb_root
    out_root = Path(output_avatarrex_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for subj in subjects:
        subj_dir = data_root / subj
        if not subj_dir.exists():
            print(f"Warning: subject dir not found: {subj_dir}")
            continue
        resolved_gender = _resolve_talkbody4d_gender(subj, gender)

        calib_path = subj_dir / "calibration.json"
        if not calib_path.exists():
            print(f"Warning: calibration.json not found: {calib_path}")
            continue
        cameras = _load_cameras(calib_path)

        raw_videos_dir = subj_dir / "raw_videos"
        mattings_dir = subj_dir / "mattings"
        params_dir = subj_dir / "smplx_fitting"
        if not raw_videos_dir.exists() or not mattings_dir.exists() or not params_dir.exists():
            print(f"Warning: missing required dirs under {subj_dir} (raw_videos/mattings/smplx_fitting)")
            continue

        cams = _list_cameras_from_raw_videos(raw_videos_dir)
        if exclude_cams:
            excl = {_normalize_cam_id(x) for x in exclude_cams if str(x).strip()}
            if excl:
                cams = [c for c in cams if _normalize_cam_id(c) not in excl]
        if not cams:
            print(f"Warning: no mp4s found under {raw_videos_dir}")
            continue

        if print_video_stats_only:
            counts = []
            for cam in cams:
                video_path = raw_videos_dir / f"{cam}.mp4"
                if not video_path.exists():
                    continue
                n = _get_video_frame_count(video_path)
                if n is not None:
                    counts.append(n)
            if counts:
                arr = np.array(counts, dtype=np.int32)
                print(
                    f"[video_stats] {subj}: num_cams={len(counts)} frame_count min/med/max="
                    f"{int(arr.min())}/{int(np.median(arr))}/{int(arr.max())}"
                )
            else:
                print(f"[video_stats] {subj}: no readable videos under {raw_videos_dir}")
            continue

        export_mode_s = str(export_mode).strip().lower()
        if export_mode_s not in {"single_frame", "sequence"}:
            raise ValueError(f"Unknown export_mode={export_mode}. Expected 'single_frame' or 'sequence'.")

        if export_mode_s == "sequence" and transform_mode == "crop_resize_pad" and len(frames) > 1:
            raise ValueError(
                "TalkBody4D export_mode=sequence does not support transform_mode=crop_resize_pad for multi-frame, "
                "because calibration_full.json is currently stored as per-subject (camera-constant) calibration. "
                "Use transform_mode=downscale/fullres, or export_mode=single_frame."
            )


        frames_src_sorted = sorted(set(int(x) for x in frames))

        if export_mode_s == "sequence" and use_all_frames_if_unspecified:
            frames_src_sorted = _discover_all_frame_ids_from_smplx_fitting(params_dir)
            if not frames_src_sorted:
                print(f"Warning: no frame ids discovered under {params_dir}. Falling back to [0].")
                frames_src_sorted = [0]
        if export_mode_s == "single_frame":
            frames_to_convert = frames_src_sorted
        else:

            frames_to_convert = frames_src_sorted

        if export_mode_s == "single_frame":
            for frame_idx in frames_to_convert:
                subject_frame = f"{subj}_{frame_idx:06d}"
                subject_out = out_root / subject_frame
                subject_out.mkdir(parents=True, exist_ok=True)

                if analyze_crop_stats_only:
                    stats = _analyze_crop_stats_for_subject_frame(
                        subject_dir=subj_dir,
                        cameras=cameras,
                        frame_idx=frame_idx,
                        dilate_ratio=crop_dilate_ratio,
                    )
                    print(f"[crop_stats] {subject_frame} (dilate={crop_dilate_ratio}): {stats}")
                    continue


                npz_path = subject_out / "smpl_params.npz"
                calib_path_out = subject_out / "calibration_full.json"
                meta_path_out = subject_out / "talkbody4d_metadata.json"


                params_path = params_dir / f"{frame_idx:06d}.json"
                if not params_path.exists():
                    print(f"Warning: SMPLX params not found: {params_path}")
                    continue
                if overwrite or (not npz_path.exists()):
                    smpl_params = _load_talkbody4d_smplx_params(params_path)
                    np.savez(str(npz_path), **{k: smpl_params[k][None].astype(np.float32) for k in smpl_params})


                gender_path = subject_out / "gender.txt"
                if overwrite or (not gender_path.exists()):
                    gender_path.write_text(str(resolved_gender).strip() + "\n")

                if smplx_only:

                    continue


                calib_out: Dict[str, Dict] = {}
                if (not overwrite) and calib_path_out.exists():
                    try:
                        calib_out = _read_json(calib_path_out)
                    except Exception:
                        calib_out = {}
                undistort_maps_by_cam: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for cam in tqdm(cams, desc=f"TalkBody4D convert {subject_frame}", leave=False):
                    if cam not in cameras:
                        continue
                    video_path = raw_videos_dir / f"{cam}.mp4"
                    mask_path = mattings_dir / cam / f"{frame_idx:06d}.png"
                    if not video_path.exists() or not mask_path.exists():
                        continue

                    cam_out = subject_out / cam
                    (cam_out / "mask" / "pha").mkdir(parents=True, exist_ok=True)
                    rgb_path = cam_out / "0000.jpg"
                    mask_out_path = cam_out / "mask" / "pha" / "0000.png"


                    has_outputs = rgb_path.exists() and mask_out_path.exists()
                    has_calib = cam in calib_out
                    if (not overwrite) and has_outputs and (has_calib or transform_mode in {"fullres", "downscale"}):

                        if not has_calib:
                            K = cameras[cam]["K"].copy()
                            R = cameras[cam]["R"].copy()
                            T = cameras[cam]["T"].copy()
                            if transform_mode == "fullres":
                                K_final = K
                                out_w = int(cameras[cam]["w"])
                                out_h = int(cameras[cam]["h"])
                                tfm = {"mode": "fullres", "undistort": True}
                            else:
                                K_final, out_w, out_h, tfm2 = _compute_downscale_K_and_size(
                                    K, int(cameras[cam]["w"]), int(cameras[cam]["h"]), factor=float(downscale_factor)
                                )
                                tfm = {"mode": "downscale", "undistort": True, "downscale": tfm2}
                            calib_out[cam] = {
                                "K": K_final.tolist(),
                                "R": R.tolist(),
                                "T": T.reshape(3, 1).tolist(),
                                "imgSize": [int(out_w), int(out_h)],
                                "distCoeff": [0.0, 0.0, 0.0, 0.0, 0.0],
                                "transform": tfm,
                            }
                        continue

                    frame_bgr = _read_video_frame(video_path, frame_idx)
                    if frame_bgr is None:
                        continue
                    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask_gray is None:
                        continue

                    K = cameras[cam]["K"].copy()
                    dist = cameras[cam]["dist"].copy()
                    R = cameras[cam]["R"].copy()
                    T = cameras[cam]["T"].copy()


                    if cam not in undistort_maps_by_cam:
                        w0 = int(cameras[cam]["w"])
                        h0 = int(cameras[cam]["h"])
                        undistort_maps_by_cam[cam] = _build_undistort_maps(K, dist, w=w0, h=h0)
                    frame_und, mask_und = _undistort_image_and_mask(
                        frame_bgr, mask_gray, K, dist, undistort_maps=undistort_maps_by_cam[cam]
                    )

                    if transform_mode == "fullres":
                        img_final = frame_und
                        mask_final = mask_und
                        K_final = K
                        tfm = {"mode": "fullres", "undistort": True}
                        out_w = int(img_final.shape[1])
                        out_h = int(img_final.shape[0])
                    elif transform_mode == "downscale":
                        img_final, mask_final, K_final, tfm2 = _apply_downscale(
                            img_bgr=frame_und,
                            mask_gray=mask_und,
                            K=K,
                            factor=float(downscale_factor),
                        )
                        tfm = {"mode": "downscale", "undistort": True, "downscale": tfm2}
                        out_w = int(img_final.shape[1])
                        out_h = int(img_final.shape[0])
                    elif transform_mode == "crop_resize_pad":
                        bbox = _compute_bbox_from_binary_mask(mask_und, dilate_ratio=crop_dilate_ratio)
                        if bbox is None:

                            bbox = (0, 0, int(mask_und.shape[1]), int(mask_und.shape[0]))
                        img_final, mask_final, K_final, tfm2 = _apply_crop_resize_pad(
                            img_bgr=frame_und,
                            mask_gray=mask_und,
                            K=K,
                            crop_box=bbox,
                            target_long_side=int(target_long_side),
                        )
                        tfm = {"mode": "crop_resize_pad", "undistort": True, "crop": tfm2}
                        out_w = int(img_final.shape[1])
                        out_h = int(img_final.shape[0])
                    else:
                        raise ValueError(f"Unknown transform_mode={transform_mode}")
                    if overwrite or (not rgb_path.exists()):
                        cv2.imwrite(str(rgb_path), img_final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if overwrite or (not mask_out_path.exists()):
                        cv2.imwrite(str(mask_out_path), mask_final)

                    calib_out[cam] = {
                        "K": K_final.tolist(),
                        "R": R.tolist(),
                        "T": T.reshape(3, 1).tolist(),
                        "imgSize": [out_w, out_h],
                        "distCoeff": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "transform": tfm,
                    }

                if calib_out and (overwrite or (not calib_path_out.exists())):
                    with open(calib_path_out, "w") as f:
                        json.dump(calib_out, f, indent=2)

                    meta = {
                        subject_frame: {
                            "original_subject": subj,
                            "frame_id": int(frame_idx),
                            "num_cameras_used": int(len(calib_out)),
                            "transform_mode": transform_mode,
                            "downscale_factor": float(downscale_factor),
                            "crop_dilate_ratio": float(crop_dilate_ratio),
                            "target_long_side": int(target_long_side),
                            "gender": str(resolved_gender),
                        }
                    }
                    if overwrite or (not meta_path_out.exists()):
                        with open(meta_path_out, "w") as f:
                            json.dump(meta, f, indent=2)
        else:

            subject_out = out_root / subj
            subject_out.mkdir(parents=True, exist_ok=True)
            (subject_out / "gender.txt").write_text(str(resolved_gender).strip() + "\n")


            pad = int(frame_padding)
            src_frame_ids = frames_to_convert


            smpl_series: Dict[str, List[np.ndarray]] = {}
            vshape_saved = None
            vpose_saved = None
            valid_src_frame_ids: List[int] = []
            for src_f in src_frame_ids:
                params_path = params_dir / f"{int(src_f):06d}.json"
                if not params_path.exists():
                    print(f"Warning: SMPLX params not found: {params_path}")
                    continue
                p = _load_talkbody4d_smplx_params(params_path)
                valid_src_frame_ids.append(int(src_f))
                for k, v in p.items():
                    if k in {"v_shape", "v_pose"}:

                        if k == "v_shape" and vshape_saved is None:
                            vshape_saved = v.astype(np.float32)
                        if k == "v_pose" and vpose_saved is None:
                            vpose_saved = v.astype(np.float32)
                        continue
                    smpl_series.setdefault(k, []).append(v.astype(np.float32))

            if not valid_src_frame_ids:
                print(f"Warning: no valid SMPLX frames found for subject {subj} under {params_dir}. Skipping.")
                continue

            src_frame_ids = valid_src_frame_ids
            out_frame_ids = list(range(len(src_frame_ids)))


            smpl_npz_path = subject_out / "smpl_params.npz"
            if (not overwrite) and smpl_npz_path.exists():
                pass
            else:
                npz_out: Dict[str, np.ndarray] = {}
                for k, arrs in smpl_series.items():
                    if not arrs:
                        continue
                    npz_out[k] = np.stack(arrs, axis=0).astype(np.float32)
                if vshape_saved is not None:
                    npz_out["v_shape"] = vshape_saved[None].astype(np.float32)
                if vpose_saved is not None:
                    npz_out["v_pose"] = vpose_saved[None].astype(np.float32)

                npz_out["talkbody4d_src_frame_ids"] = np.array(src_frame_ids, dtype=np.int32)
                np.savez(str(smpl_npz_path), **npz_out)

            if smplx_only:

                continue


            calib_out: Dict[str, Dict] = {}
            caps: Dict[str, cv2.VideoCapture] = {}
            undistort_maps_by_cam: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            try:
                for cam in cams:
                    if cam not in cameras:
                        continue
                    video_path = raw_videos_dir / f"{cam}.mp4"
                    if not video_path.exists():
                        continue
                    cap = _open_video_capture(video_path)
                    if cap is None:
                        continue
                    caps[cam] = cap

                for cam in tqdm(cams, desc=f"TalkBody4D convert sequence {subj}", leave=False):
                    if cam not in cameras:
                        continue
                    if cam not in caps:
                        continue

                    K = cameras[cam]["K"].copy()
                    dist = cameras[cam]["dist"].copy()
                    R = cameras[cam]["R"].copy()
                    T = cameras[cam]["T"].copy()

                    cam_out = subject_out / cam
                    (cam_out / "mask" / "pha").mkdir(parents=True, exist_ok=True)

                    if cam not in undistort_maps_by_cam:
                        w0 = int(cameras[cam]["w"])
                        h0 = int(cameras[cam]["h"])
                        undistort_maps_by_cam[cam] = _build_undistort_maps(K, dist, w=w0, h=h0)

                    K_final_cached = None
                    out_w_cached = None
                    out_h_cached = None
                    tfm_cached = None


                    needed_src: List[int] = []
                    out_f_for_src: Dict[int, int] = {}
                    for out_f, src_f in zip(out_frame_ids, src_frame_ids):
                        rgb_path = cam_out / f"{int(out_f):0{pad}d}.jpg"
                        mask_out_path = cam_out / "mask" / "pha" / f"{int(out_f):0{pad}d}.png"
                        if (not overwrite) and rgb_path.exists() and mask_out_path.exists():
                            continue
                        needed_src.append(int(src_f))
                        out_f_for_src[int(src_f)] = int(out_f)


                    frames_bgr_by_src = _read_frames_sequential(caps[cam], sorted(set(needed_src)))
                    for out_f, src_f in zip(out_frame_ids, src_frame_ids):
                        rgb_path = cam_out / f"{int(out_f):0{pad}d}.jpg"
                        mask_out_path = cam_out / "mask" / "pha" / f"{int(out_f):0{pad}d}.png"
                        if (not overwrite) and rgb_path.exists() and mask_out_path.exists():
                            continue
                        mask_path = mattings_dir / cam / f"{int(src_f):06d}.png"
                        if not mask_path.exists():
                            continue
                        frame_bgr = frames_bgr_by_src.get(int(src_f), None)
                        if frame_bgr is None:
                            continue
                        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask_gray is None:
                            continue

                        frame_und, mask_und = _undistort_image_and_mask(
                            frame_bgr, mask_gray, K, dist, undistort_maps=undistort_maps_by_cam[cam]
                        )

                        if transform_mode == "fullres":
                            img_final = frame_und
                            mask_final = mask_und
                            K_final = K
                            tfm = {"mode": "fullres", "undistort": True}
                        elif transform_mode == "downscale":
                            img_final, mask_final, K_final, tfm2 = _apply_downscale(
                                img_bgr=frame_und,
                                mask_gray=mask_und,
                                K=K,
                                factor=float(downscale_factor),
                            )
                            tfm = {"mode": "downscale", "undistort": True, "downscale": tfm2}
                        elif transform_mode == "crop_resize_pad":

                            bbox = _compute_bbox_from_binary_mask(mask_und, dilate_ratio=crop_dilate_ratio)
                            if bbox is None:
                                bbox = (0, 0, int(mask_und.shape[1]), int(mask_und.shape[0]))
                            img_final, mask_final, K_final, tfm2 = _apply_crop_resize_pad(
                                img_bgr=frame_und,
                                mask_gray=mask_und,
                                K=K,
                                crop_box=bbox,
                                target_long_side=int(target_long_side),
                            )
                            tfm = {"mode": "crop_resize_pad", "undistort": True, "crop": tfm2}
                        else:
                            raise ValueError(f"Unknown transform_mode={transform_mode}")

                        cv2.imwrite(str(rgb_path), img_final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        cv2.imwrite(str(mask_out_path), mask_final)

                        if K_final_cached is None:
                            K_final_cached = K_final
                            out_w_cached = int(img_final.shape[1])
                            out_h_cached = int(img_final.shape[0])
                            tfm_cached = tfm

                    if K_final_cached is None and transform_mode in {"fullres", "downscale"}:

                        if transform_mode == "fullres":
                            K_final_cached = K
                            out_w_cached = int(cameras[cam]["w"])
                            out_h_cached = int(cameras[cam]["h"])
                            tfm_cached = {"mode": "fullres", "undistort": True}
                        else:
                            K_final_cached, out_w_cached, out_h_cached, tfm2 = _compute_downscale_K_and_size(
                                K, int(cameras[cam]["w"]), int(cameras[cam]["h"]), factor=float(downscale_factor)
                            )
                            tfm_cached = {"mode": "downscale", "undistort": True, "downscale": tfm2}

                    if K_final_cached is not None:
                        calib_out[cam] = {
                            "K": K_final_cached.tolist(),
                            "R": R.tolist(),
                            "T": T.reshape(3, 1).tolist(),
                            "imgSize": [int(out_w_cached), int(out_h_cached)],
                            "distCoeff": [0.0, 0.0, 0.0, 0.0, 0.0],
                            "transform": tfm_cached if tfm_cached is not None else {"mode": transform_mode, "undistort": True},
                        }
            finally:
                for _, cap in caps.items():
                    try:
                        cap.release()
                    except Exception:
                        pass

            calib_path_out = subject_out / "calibration_full.json"
            if calib_out and (overwrite or (not calib_path_out.exists())):
                with open(calib_path_out, "w") as f:
                    json.dump(calib_out, f, indent=2)
                meta = {
                    subj: {
                        "original_subject": subj,
                        "export_mode": "sequence",
                        "frame_padding": int(frame_padding),
                        "num_frames_out": int(len(out_frame_ids)),
                        "src_frame_ids": [int(x) for x in src_frame_ids],
                        "num_cameras_used": int(len(calib_out)),
                        "transform_mode": transform_mode,
                        "downscale_factor": float(downscale_factor),
                        "crop_dilate_ratio": float(crop_dilate_ratio),
                        "target_long_side": int(target_long_side),
                        "gender": str(resolved_gender),
                    }
                }
                meta_path_out = subject_out / "talkbody4d_metadata.json"
                if overwrite or (not meta_path_out.exists()):
                    with open(meta_path_out, "w") as f:
                        json.dump(meta, f, indent=2)


def _parse_frames(args: argparse.Namespace) -> List[int]:
    if args.frames is not None and len(args.frames) > 0:
        return [int(x) for x in args.frames]
    if args.frame_start is None:
        return [0]
    end = args.frame_end if args.frame_end is not None else args.frame_start
    step = args.frame_step if args.frame_step is not None else 1
    return list(range(int(args.frame_start), int(end) + 1, int(step)))


def _discover_all_frame_ids_from_smplx_fitting(params_dir: Path) -> List[int]:

    out: List[int] = []
    for p in sorted(params_dir.glob("*.json")):
        stem = p.stem
        if stem.isdigit():
            out.append(int(stem))
    return sorted(set(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TalkBody4D (video+matting+SMPLX) to AvatarREX format.")
    parser.add_argument("--talkbody4d_dir", type=str, required=True, help="Path to TalkBody4D root (contains data/).")
    parser.add_argument("--avatarrex_dir", type=str, required=True, help="Output directory for AvatarREX-formatted data.")
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=str,
        required=True,
        help=(
            "Subjects to convert.\n"
            "- Base subjects: e.g. SC_01\n"
            "- For export_mode=single_frame, you may also pass subject-frame tokens like SC_01_000123 "
            "(4-6 digit suffix is accepted) and omit --frames/--frame_start/--frame_end.\n"
        ),
    )

    parser.add_argument("--frames", nargs="+", type=int, default=None, help="Explicit frame indices to extract (default: [0]).")
    parser.add_argument("--frame_start", type=int, default=None, help="Start frame (inclusive) if --frames not given.")
    parser.add_argument("--frame_end", type=int, default=None, help="End frame (inclusive) if --frames not given.")
    parser.add_argument("--frame_step", type=int, default=None, help="Frame step if --frames not given.")

    parser.add_argument(
        "--transform_mode",
        type=str,
        default="fullres",
        choices=["fullres", "downscale", "crop_resize_pad"],
        help=(
            "fullres: undistort only; "
            "downscale: undistort then uniformly resize by --downscale_factor; "
            "crop_resize_pad: mask bbox crop + resize long side + pad to square."
        ),
    )
    parser.add_argument(
        "--downscale_factor",
        type=float,
        default=0.5,
        help="Uniform downscale factor for transform_mode=downscale (e.g., 0.5 -> 2000x1500).",
    )
    parser.add_argument("--crop_dilate_ratio", type=float, default=0.05, help="Dilation ratio for mask bbox crop.")
    parser.add_argument("--target_long_side", type=int, default=2048, help="Target long side for crop_resize_pad mode.")
    parser.add_argument(
        "--analyze_crop_stats_only",
        action="store_true",
        help="Only compute and print mask-bbox crop stats (after undistortion) to help pick target resolution.",
    )
    parser.add_argument(
        "--export_mode",
        type=str,
        default="single_frame",
        choices=["single_frame", "sequence"],
        help=(
            "single_frame: export each selected frame as its own subject '<SUBJ>_<frame:06d>' with per-camera '0000.jpg'; "
            "sequence: export one subject '<SUBJ>/' containing multiple frames per camera with per-frame filenames."
        ),
    )
    parser.add_argument(
        "--frame_padding",
        type=int,
        default=4,
        help="Filename padding for per-frame images/masks in export_mode=sequence (e.g., 4 -> 0000.jpg).",
    )
    parser.add_argument(
        "--print_video_stats_only",
        action="store_true",
        help="Print video frame-count stats per subject (min/median/max over cameras) and exit without writing outputs.",
    )
    parser.add_argument(
        "--exclude_cams",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Camera ids to skip (e.g., '056' or '56'). "
            "Matches video stems under raw_videos/*.mp4 after normalizing numeric ids to 3 digits."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported images/masks/npz/json. Default behavior is to skip existing frame files.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="auto",
        choices=["auto", "neutral", "male", "female"],
        help="Gender label to write to gender.txt. 'auto' uses hardcoded TalkBody4D subject->gender mapping.",
    )
    parser.add_argument(
        "--smplx_only",
        action="store_true",
        help="Only export SMPL-X params (smpl_params.npz and gender.txt); skip images/masks/calibration/metadata.",
    )

    args = parser.parse_args()
    use_all_frames_if_unspecified = (
        str(args.export_mode).strip().lower() == "sequence"
        and args.frames is None
        and args.frame_start is None
        and args.frame_end is None
        and args.frame_step is None
    )

    if bool(args.smplx_only) and bool(args.analyze_crop_stats_only):
        raise ValueError("--smplx_only is incompatible with --analyze_crop_stats_only (requires masks/images).")

    export_mode_s = str(args.export_mode).strip().lower()
    has_any_frame_args = not (
        args.frames is None and args.frame_start is None and args.frame_end is None and args.frame_step is None
    )


    by_base: Dict[str, List[int]] = {}
    tokens = [str(s).strip() for s in args.subjects if str(s).strip()]
    any_suffix = False
    for tok in tokens:
        parts = tok.split("_")
        if len(parts) >= 2 and parts[-1].isdigit() and 4 <= len(parts[-1]) <= 6:
            any_suffix = True
            base = "_".join(parts[:-1])
            by_base.setdefault(base, []).append(int(parts[-1]))
        else:
            by_base.setdefault(tok, [])

    if export_mode_s == "single_frame" and any_suffix and not has_any_frame_args:
        for base, frame_list in by_base.items():
            frames = sorted(set(frame_list)) if frame_list else [0]
            convert_dataset(
                talkbody4d_root=args.talkbody4d_dir,
                output_avatarrex_root=args.avatarrex_dir,
                subjects=[base],
                frames=frames,
                use_all_frames_if_unspecified=False,
                export_mode=str(args.export_mode),
                frame_padding=int(args.frame_padding),
                transform_mode=args.transform_mode,
                downscale_factor=float(args.downscale_factor),
                crop_dilate_ratio=float(args.crop_dilate_ratio),
                target_long_side=int(args.target_long_side),
                analyze_crop_stats_only=bool(args.analyze_crop_stats_only),
                print_video_stats_only=bool(args.print_video_stats_only),
                overwrite=bool(args.overwrite),
                gender=args.gender,
                smplx_only=bool(args.smplx_only),
                exclude_cams=args.exclude_cams,
            )
    else:
        frames = _parse_frames(args)
        convert_dataset(
            talkbody4d_root=args.talkbody4d_dir,
            output_avatarrex_root=args.avatarrex_dir,
            subjects=args.subjects,
            frames=frames,
            use_all_frames_if_unspecified=bool(use_all_frames_if_unspecified),
            export_mode=str(args.export_mode),
            frame_padding=int(args.frame_padding),
            transform_mode=args.transform_mode,
            downscale_factor=float(args.downscale_factor),
            crop_dilate_ratio=float(args.crop_dilate_ratio),
            target_long_side=int(args.target_long_side),
            analyze_crop_stats_only=bool(args.analyze_crop_stats_only),
            print_video_stats_only=bool(args.print_video_stats_only),
            overwrite=bool(args.overwrite),
            gender=args.gender,
            smplx_only=bool(args.smplx_only),
            exclude_cams=args.exclude_cams,
        )
