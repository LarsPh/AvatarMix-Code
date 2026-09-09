import argparse
import json
import os
from pathlib import Path
import pickle
import re
import shutil
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


def _parse_split_files(split_files: List[str]) -> List[Tuple[str, str]]:

    subject_frame_pairs: List[Tuple[str, str]] = []
    for split_file in split_files:
        split_path = Path(split_file)
        if not split_path.exists():
            print(f"Warning: split file not found at {split_path}, skipping.")
            continue
        with open(split_path, "r") as f:

            content = f.read().strip()
        if not content:
            continue
        tokens = []

        for part in content.replace("\n", " ").split(","):
            part = part.strip()
            if not part:
                continue
            tokens.extend(part.split())
        for tok in tokens:
            if "_" not in tok:
                print(f"  Warning: token '{tok}' in {split_path} does not contain '_', skipping.")
                continue
            subj_str, frame_str = tok.split("_", 1)
            if not (subj_str.isdigit() and frame_str.isdigit()):
                print(f"  Warning: token '{tok}' in {split_path} is not numeric, skipping.")
                continue

            subj_str = f"{int(subj_str):06d}"
            frame_str = f"{int(frame_str):04d}"
            subject_frame_pairs.append((subj_str, frame_str))


    seen = set()
    unique_pairs: List[Tuple[str, str]] = []
    for pair in subject_frame_pairs:
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)
    return unique_pairs


def _parse_subject_tokens(subject_tokens: List[str]) -> List[Tuple[str, str]]:

    subject_frame_pairs: List[Tuple[str, str]] = []
    for tok in subject_tokens:
        tok = str(tok).strip()
        if not tok:
            continue
        if "_" not in tok:
            raise ValueError(f"MVHumanNet subject token must be like '100001_1185', got: {tok}")
        subj_str, frame_str = tok.split("_", 1)
        if not (subj_str.isdigit() and frame_str.isdigit()):
            raise ValueError(f"MVHumanNet subject token must be numeric like '100001_1185', got: {tok}")
        subj_str = f"{int(subj_str):06d}"
        frame_str = f"{int(frame_str):04d}"
        subject_frame_pairs.append((subj_str, frame_str))


    seen = set()
    unique_pairs: List[Tuple[str, str]] = []
    for pair in subject_frame_pairs:
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)
    return unique_pairs


def _load_mvhumannet_smplx_params(smplx_file_path: Path) -> Dict:

    if not smplx_file_path.exists():
        raise FileNotFoundError(f"SMPL-X JSON file not found: {smplx_file_path}")

    with open(smplx_file_path, "r") as f:
        smplx_list = json.load(f)

    if not isinstance(smplx_list, list) or len(smplx_list) == 0:
        raise ValueError(f"Unexpected SMPL-X JSON structure at {smplx_file_path}")

    smplx_data = smplx_list[0]

    converted_params: Dict[str, np.ndarray] = {}


    if "poses" in smplx_data:
        poses_arr = np.array(smplx_data["poses"], dtype=np.float32)
        if poses_arr.ndim != 2 or poses_arr.shape[0] != 1:
            raise ValueError(f"Unexpected 'poses' shape {poses_arr.shape} in {smplx_file_path}")


        assert np.all(poses_arr[:, :3] == 0), f"'poses' first 3 values not all zero in {smplx_file_path}"

        body_poses = poses_arr[:, 3:]
        if body_poses.shape[1] < 63 + 12 + 9:
            raise ValueError(
                f"Unexpected 'poses' length {body_poses.shape[1]} (expected at least 84) in {smplx_file_path}"
            )

        body_end = 63

        left_hand_pose = body_poses[:, body_end : body_end + 6]
        right_hand_pose = body_poses[:, body_end + 6 : body_end + 12]

        hand_end = body_end + 12
        jaw_pose = body_poses[:, hand_end : hand_end + 3]
        left_eye_pose = body_poses[:, hand_end + 3 : hand_end + 6]
        right_eye_pose = body_poses[:, hand_end + 6 : hand_end + 9]

        converted_params["body_pose"] = body_poses[:, :63].squeeze(0)
        converted_params["left_hand_pose"] = left_hand_pose.squeeze(0)
        converted_params["right_hand_pose"] = right_hand_pose.squeeze(0)
        converted_params["jaw_pose"] = jaw_pose.squeeze(0)
        converted_params["leye_pose"] = left_eye_pose.squeeze(0)
        converted_params["reye_pose"] = right_eye_pose.squeeze(0)

    if "shapes" in smplx_data:
        converted_params["betas"] = np.array(smplx_data["shapes"], dtype=np.float32).squeeze(0)
    if "Rh" in smplx_data:
        converted_params["global_orient"] = np.array(smplx_data["Rh"], dtype=np.float32).squeeze(0)
    if "Th" in smplx_data:
        converted_params["transl"] = np.array(smplx_data["Th"], dtype=np.float32).squeeze(0)
    if "expression" in smplx_data:
        converted_params["expression"] = np.array(smplx_data["expression"], dtype=np.float32).squeeze(0)

    return converted_params


def _build_extrinsics_for_subject(subject_dir: Path) -> Tuple[Dict[str, np.ndarray], float]:

    intrinsics_path = subject_dir / "camera_intrinsics.json"
    extrinsics_path = subject_dir / "camera_extrinsics.json"
    scale_path = subject_dir / "camera_scale.pkl"

    if not intrinsics_path.exists() or not extrinsics_path.exists() or not scale_path.exists():
        raise FileNotFoundError(
            f"Missing camera files in {subject_dir}: "
            f"intrinsics={intrinsics_path.exists()}, extrinsics={extrinsics_path.exists()}, "
            f"scale={scale_path.exists()}"
        )

    with open(intrinsics_path, "r") as f:
        intri_data = json.load(f)
    with open(extrinsics_path, "r") as f:
        extri_data = json.load(f)
    with open(scale_path, "rb") as f:
        camera_scale = pickle.load(f)

    K_orig = np.array(intri_data["intrinsics"], dtype=np.float32)

    extrinsics_by_cam: Dict[str, Dict[str, np.ndarray]] = {}
    for key, entry in extri_data.items():

        cam_id = key.split(".")[0].split("_")[-1]
        R = np.array(entry["rotation"], dtype=np.float32)
        T = np.array(entry["translation"], dtype=np.float32).reshape(3, 1) / 1000.0 * float(camera_scale)
        extrinsics_by_cam[cam_id] = {"R": R, "T": T}

    return extrinsics_by_cam, K_orig


def _compute_intrinsics_for_frame_and_camera(
    K_orig: np.ndarray,
    full_width: int,
    full_height: int,
    image_lr_size: Tuple[int, int],
    bbox_full: List[float],
    bbox_lr_int: Optional[Tuple[int, int, int, int]] = None,
    target_size: int = 1024,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int], Dict[str, float | int | Tuple[int, int] | Tuple[int, int, int, int]]]:

    assert K_orig.shape == (3, 3)


    lr_w, lr_h = image_lr_size
    sx = lr_w / float(full_width)
    sy = lr_h / float(full_height)

    if not (0.4 < sx < 0.6 and 0.4 < sy < 0.6):
        print(
            f"  Warning: unexpected LR scaling factors sx={sx:.3f}, sy={sy:.3f} "
            f"(full {full_width}x{full_height}, lr {lr_w}x{lr_h})"
        )

    K_lr = K_orig.copy()
    K_lr[0, 0] *= sx
    K_lr[1, 1] *= sy


    K_lr[0, 2] = (float(K_orig[0, 2]) + 0.5) * sx - 0.5
    K_lr[1, 2] = (float(K_orig[1, 2]) + 0.5) * sy - 0.5


    if bbox_lr_int is None:

        x1_full, y1_full, x2_full, y2_full, _ = bbox_full
        x1_lr = x1_full * sx
        y1_lr = y1_full * sy
        x2_lr = x2_full * sx
        y2_lr = y2_full * sy


        left = int(np.floor(x1_lr))
        top = int(np.floor(y1_lr))
        right = int(np.ceil(x2_lr))
        bottom = int(np.ceil(y2_lr))
    else:
        left, top, right, bottom = bbox_lr_int


    left = max(0, min(left, lr_w - 1))
    top = max(0, min(top, lr_h - 1))
    right = max(left + 1, min(right, lr_w))
    bottom = max(top + 1, min(bottom, lr_h))

    crop_box_lr_int = (left, top, right, bottom)
    crop_w = right - left
    crop_h = bottom - top


    K_crop = K_lr.copy()
    K_crop[0, 2] -= left
    K_crop[1, 2] -= top


    long_side = max(crop_w, crop_h)
    resize_scale = target_size / float(long_side)


    new_w = int(round(crop_w * resize_scale))
    new_h = int(round(crop_h * resize_scale))

    new_w = max(1, min(int(target_size), new_w))
    new_h = max(1, min(int(target_size), new_h))

    scale_x = new_w / float(crop_w)
    scale_y = new_h / float(crop_h)

    K_res = K_crop.copy()
    K_res[0, 0] *= scale_x
    K_res[1, 1] *= scale_y


    K_res[0, 2] = (float(K_crop[0, 2]) + 0.5) * scale_x - 0.5
    K_res[1, 2] = (float(K_crop[1, 2]) + 0.5) * scale_y - 0.5


    pad_left = pad_top = 0
    if new_w > new_h:

        pad_total = target_size - new_h
        pad_top = pad_total // 2

    elif new_h > new_w:

        pad_total = target_size - new_w
        pad_left = pad_total // 2


    K_final = K_res.copy()
    K_final[0, 2] += pad_left
    K_final[1, 2] += pad_top

    transform_params: Dict[str, float | int | Tuple[int, int] | Tuple[int, int, int, int]] = {
        "sx": float(sx),
        "sy": float(sy),
        "crop_box_lr_int": crop_box_lr_int,
        "crop_w": int(crop_w),
        "crop_h": int(crop_h),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "target_size": int(target_size),
        "target_w": int(target_size),
        "target_h": int(target_size),
        "lr_w": int(lr_w),
        "lr_h": int(lr_h),
        "full_w": int(full_width),
        "full_h": int(full_height),
    }

    return K_final, (target_size, target_size), crop_box_lr_int, transform_params


def _compute_intrinsics_full_to_lr(K_orig: np.ndarray, sx: float, sy: float) -> np.ndarray:

    K_lr = K_orig.copy()
    K_lr[0, 0] *= sx
    K_lr[1, 1] *= sy

    K_lr[0, 2] = (float(K_orig[0, 2]) + 0.5) * sx - 0.5
    K_lr[1, 2] = (float(K_orig[1, 2]) + 0.5) * sy - 0.5
    return K_lr


def _compute_bbox_from_binary_mask_lr(
    mask_lr: Image.Image,
    dilate_ratio: float = 0.05,
) -> Optional[Tuple[int, int, int, int]]:

    arr = np.array(mask_lr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    fg = arr > 0
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

    left -= d
    top -= d
    right += d
    bottom += d


    W, H = mask_lr.size
    left = max(0, min(left, W - 1))
    top = max(0, min(top, H - 1))
    right = max(left + 1, min(right, W))
    bottom = max(top + 1, min(bottom, H))
    return (left, top, right, bottom)


def _openpose_body25_stable_indices() -> List[int]:


    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def _transform_openpose_keypoints_fullres_to_final(
    keypoints_full: List[List[float]],
    transform_params: Dict[str, float | int | Tuple[int, int] | Tuple[int, int, int, int]],
    conf_threshold: float,
) -> List[List[float]]:

    sx = float(transform_params["sx"])
    sy = float(transform_params["sy"])
    left, top, _, _ = transform_params["crop_box_lr_int"]  # type: ignore[misc]
    scale_x = float(transform_params["scale_x"])
    scale_y = float(transform_params["scale_y"])
    pad_left = int(transform_params["pad_left"])
    pad_top = int(transform_params["pad_top"])


    target_w = int(transform_params.get("target_w", transform_params.get("target_size", 0)))
    target_h = int(transform_params.get("target_h", transform_params.get("target_size", 0)))

    out: List[List[float]] = []
    for i, kp in enumerate(keypoints_full):
        if kp is None or len(kp) < 3:
            out.append([0.0, 0.0, 0.0])
            continue
        x_full, y_full, c = float(kp[0]), float(kp[1]), float(kp[2])


        if c < conf_threshold:
            c = 0.0


        if x_full == 0.0 and y_full == 0.0:
            c = 0.0


        x_lr = (x_full + 0.5) * sx - 0.5
        y_lr = (y_full + 0.5) * sy - 0.5


        x_crop = x_lr - float(left)
        y_crop = y_lr - float(top)


        x_res = (x_crop + 0.5) * scale_x - 0.5
        y_res = (y_crop + 0.5) * scale_y - 0.5


        x_out = x_res + float(pad_left)
        y_out = y_res + float(pad_top)


        if not (0.0 <= x_out < float(target_w) and 0.0 <= y_out < float(target_h)):
            c = 0.0

        out.append([float(x_out), float(y_out), float(c)])
    return out


def convert_dataset(
    mvhumannet_root: str,
    output_avatarrex_root: str,
    split_files: Optional[List[str]] = None,
    subjects: Optional[List[str]] = None,
    copy_mesh: bool = True,
    use_gender: bool = True,
    skip_existing: bool = True,
    only_update_smpl_params: bool = False,
    export_2d_keypoints: bool = True,
    only_update_2d_keypoints: bool = False,
    keypoints_mode: str = "stable",
    conf_threshold: float = 0.2,
    bbox_source: str = "mask",
    mask_bbox_dilate_ratio: float = 0.05,
    transform_mode: str = "crop_resize_pad_1024",
    include_cams: Optional[List[str]] = None,
    exclude_cams: Optional[List[str]] = None,
    exclude_cams_regex: Optional[str] = None,
):

    mvhumannet_dir = Path(mvhumannet_root)
    avatarrex_root_dir = Path(output_avatarrex_root)
    avatarrex_root_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {avatarrex_root_dir}")


    text_desc_24_path = mvhumannet_dir / "text_description_24.json"
    text_desc_48_path = mvhumannet_dir / "text_description_48.json"
    text_desc_24: Dict[str, str] = {}
    text_desc_48: Dict[str, str] = {}
    if use_gender:
        if text_desc_24_path.exists():
            try:
                with open(text_desc_24_path, "r") as f:
                    text_desc_24 = json.load(f)
                print(f"Loaded text_description_24.json with {len(text_desc_24)} entries.")
            except Exception as e:
                print(f"Warning: Failed to load {text_desc_24_path}: {e}")
        else:
            print(f"Warning: {text_desc_24_path} not found; 100xxx subjects will have unknown gender.")

        if text_desc_48_path.exists():
            try:
                with open(text_desc_48_path, "r") as f:
                    text_desc_48 = json.load(f)
                print(f"Loaded text_description_48.json with {len(text_desc_48)} entries.")
            except Exception as e:
                print(f"Warning: Failed to load {text_desc_48_path}: {e}")
        else:
            print(f"Warning: {text_desc_48_path} not found; 200xxx subjects will have unknown gender.")

    subject_frame_pairs: List[Tuple[str, str]] = []
    if subjects:
        subject_frame_pairs = _parse_subject_tokens(subjects)
    elif split_files:
        subject_frame_pairs = _parse_split_files(split_files)
    if not subject_frame_pairs:
        print("No subject-frame pairs provided (empty --subjects and empty --splits); nothing to do.")
        return

    print(f"Found {len(subject_frame_pairs)} unique subject-frame pairs to process.")

    processed_subjects: List[str] = []

    for subj_id_str, frame_str in subject_frame_pairs:
        subject_dir = mvhumannet_dir / subj_id_str
        if not subject_dir.exists():
            print(f"Warning: MVHumanNet subject directory {subject_dir} not found, skipping.")
            continue

        subject_frame_name = f"{subj_id_str}_{frame_str}"
        print(f"\nProcessing MVHumanNet subject-frame: {subject_frame_name}")


        subject_output_dir = avatarrex_root_dir / subject_frame_name
        subject_output_dir.mkdir(parents=True, exist_ok=True)


        gender = "unknown"
        if use_gender:

            if subj_id_str.startswith("1"):
                caption = text_desc_24.get(subj_id_str, "")
            elif subj_id_str.startswith("2"):
                caption = text_desc_48.get(subj_id_str, "")
            else:
                caption = ""
            if not caption:
                print(f"Caption not found for subject {subj_id_str}")

            if caption:
                text = caption.lower()
                if re.search(r"\b(woman|lady)\b", text):
                    gender = "female"
                elif re.search(r"\bman\b", text):
                    gender = "male"
                else:
                    gender = "unknown"
                    print(f"Gender not found in caption: {caption}")
            else:
                gender = "unknown"


        try:
            extrinsics_by_cam, K_orig = _build_extrinsics_for_subject(subject_dir)
        except Exception as e:
            print(f"  Error loading camera data for subject {subj_id_str}: {e}")
            continue


        if only_update_2d_keypoints:
            pass
        else:

            frame_int = int(frame_str)
            if frame_int % 5 != 0:
                print(
                    f"  Warning: frame id {frame_str} for subject {subj_id_str} is not multiple of 5; "
                    f"cannot map to SMPL-X index, skipping."
                )
                continue
            smplx_idx = frame_int // 5 - 1
            if smplx_idx < 0:
                print(
                    f"  Warning: computed SMPL-X index {smplx_idx} for {subject_frame_name} is negative; skipping."
                )
                continue
            smplx_json_path = subject_dir / "smplx" / "smpl" / f"{smplx_idx:06d}.json"

            try:
                smplx_converted = _load_mvhumannet_smplx_params(smplx_json_path)
            except Exception as e:
                print(f"  Error loading SMPL-X params from {smplx_json_path}: {e}")
                continue


            final_smpl_params = {}
            for key, val in smplx_converted.items():
                arr = np.array(val, dtype=np.float32)
                final_smpl_params[key] = np.expand_dims(arr, axis=0)


            smpl_npz_path = subject_output_dir / "smpl_params.npz"
            try:

                if not only_update_smpl_params and skip_existing and smpl_npz_path.exists():
                    print(f"  smpl_params.npz already exists at {smpl_npz_path}, skipping write.")
                else:
                    np.savez(str(smpl_npz_path), **final_smpl_params)
                    print(f"  Saved smpl_params.npz to {smpl_npz_path}")
            except Exception as e:
                print(f"  Error saving smpl_params.npz for {subject_frame_name}: {e}")
                continue


            if copy_mesh:
                src_mesh_path = subject_dir / "smplx" / "smplx_mesh" / f"{smplx_idx:06d}.obj"
                dst_mesh_dir = subject_output_dir / "mesh" / "processed"
                dst_mesh_path = dst_mesh_dir / "smpl_body_dataset.obj"
                if src_mesh_path.exists():
                    try:
                        dst_mesh_dir.mkdir(parents=True, exist_ok=True)
                        if skip_existing and dst_mesh_path.exists():
                            print(f"  SMPL-X mesh already exists at {dst_mesh_path}, skipping copy.")
                        else:
                            shutil.copy2(src_mesh_path, dst_mesh_path)
                            print(f"  Copied SMPL-X mesh to {dst_mesh_path}")
                    except Exception as e:
                        print(f"  Warning: Failed to copy SMPL-X mesh from {src_mesh_path} to {dst_mesh_path}: {e}")
                else:
                    print(f"  Warning: SMPL-X mesh not found at {src_mesh_path}; skipping mesh copy.")


            if only_update_smpl_params:
                processed_subjects.append(subject_frame_name)
                continue


        images_lr_dir = subject_dir / "images_lr"
        masks_lr_dir = subject_dir / "fmask_lr"
        annots_dir = subject_dir / "annots"

        if not images_lr_dir.exists() or not masks_lr_dir.exists() or not annots_dir.exists():
            print(
                f"  Warning: Missing images_lr / fmask_lr / annots for subject {subj_id_str}, skipping images."
            )
            continue

        camera_ids = sorted([d.name for d in images_lr_dir.iterdir() if d.is_dir()])
        if not camera_ids:
            print(f"  Warning: No camera subdirectories under {images_lr_dir}, skipping images.")
            continue


        camera_ids_filtered = camera_ids
        if include_cams:
            include_set = set([str(c).strip() for c in include_cams if str(c).strip()])
            camera_ids_filtered = [c for c in camera_ids_filtered if c in include_set]
        if exclude_cams:
            exclude_set = set([str(c).strip() for c in exclude_cams if str(c).strip()])
            camera_ids_filtered = [c for c in camera_ids_filtered if c not in exclude_set]
        if exclude_cams_regex:
            try:
                pat = re.compile(exclude_cams_regex)
                camera_ids_filtered = [c for c in camera_ids_filtered if not pat.search(c)]
            except re.error as e:
                print(f"  Warning: invalid --exclude_cams_regex '{exclude_cams_regex}': {e}; ignoring regex.")

        if camera_ids_filtered != camera_ids:
            print(f"  Cameras: {len(camera_ids)} total, {len(camera_ids_filtered)} after filtering.")
        camera_ids = camera_ids_filtered

        calibration_data: Dict[str, Dict] = {}
        n_cameras_used = 0

        if keypoints_mode not in {"stable", "all"}:
            raise ValueError(f"Unknown keypoints_mode='{keypoints_mode}', expected 'stable' or 'all'")

        stable_indices = set(_openpose_body25_stable_indices())
        recommended_use_mask = [(i in stable_indices) for i in range(25)] if keypoints_mode == "stable" else [True] * 25

        for cam_id in tqdm(camera_ids, desc=f"  Processing cameras for {subject_frame_name}", leave=False):

            if cam_id not in extrinsics_by_cam:

                continue

            img_path = images_lr_dir / cam_id / f"{frame_str}_img.jpg"
            mask_path = masks_lr_dir / cam_id / f"{frame_str}_img_fmask.png"
            annot_path = annots_dir / cam_id / f"{frame_str}_img.json"

            if not (img_path.exists() and mask_path.exists() and annot_path.exists()):

                continue

            try:

                img_lr = Image.open(img_path).convert("RGB")
                mask_lr = Image.open(mask_path)
                lr_w, lr_h = img_lr.size


                with open(annot_path, "r") as f:
                    annot_data = json.load(f)
                full_w = int(annot_data.get("width", 0))
                full_h = int(annot_data.get("height", 0))
                annots_list = annot_data.get("annots", [])
                if not annots_list:
                    print(f"    Warning: No annots in {annot_path}, skipping camera {cam_id}")
                    continue

                bbox_full = annots_list[0]["bbox"]

                if transform_mode == "no_crop_resize_pad":


                    sx = lr_w / float(full_w)
                    sy = lr_h / float(full_h)
                    K_final = _compute_intrinsics_full_to_lr(K_orig=K_orig, sx=sx, sy=sy)
                    final_size = (lr_w, lr_h)
                    crop_box_lr_int = (0, 0, lr_w, lr_h)
                    transform_params = {
                        "sx": float(sx),
                        "sy": float(sy),
                        "crop_box_lr_int": crop_box_lr_int,
                        "crop_w": int(lr_w),
                        "crop_h": int(lr_h),
                        "new_w": int(lr_w),
                        "new_h": int(lr_h),
                        "scale_x": 1.0,
                        "scale_y": 1.0,
                        "pad_left": 0,
                        "pad_top": 0,

                        "target_size": int(max(lr_w, lr_h)),
                        "target_w": int(lr_w),
                        "target_h": int(lr_h),
                        "lr_w": int(lr_w),
                        "lr_h": int(lr_h),
                        "full_w": int(full_w),
                        "full_h": int(full_h),
                    }
                    img_final = img_lr
                    mask_final = mask_lr
                    target_w, target_h = final_size
                else:


                    bbox_lr_int_from_mask: Optional[Tuple[int, int, int, int]] = None
                    if bbox_source == "mask":
                        bbox_lr_int_from_mask = _compute_bbox_from_binary_mask_lr(
                            mask_lr=mask_lr,
                            dilate_ratio=float(mask_bbox_dilate_ratio),
                        )

                        if bbox_lr_int_from_mask is None:
                            bbox_lr_int_from_mask = None


                    K_final, final_size, crop_box_lr_int, transform_params = _compute_intrinsics_for_frame_and_camera(
                        K_orig,
                        full_width=full_w,
                        full_height=full_h,
                        image_lr_size=(lr_w, lr_h),
                        bbox_full=bbox_full,
                        bbox_lr_int=bbox_lr_int_from_mask,
                        target_size=1024,
                    )


                    left, top, right, bottom = crop_box_lr_int
                    img_cropped = img_lr.crop((left, top, right, bottom))
                    mask_cropped = mask_lr.crop((left, top, right, bottom))


                    new_w = int(transform_params["new_w"])
                    new_h = int(transform_params["new_h"])
                    img_resized = img_cropped.resize((new_w, new_h), resample=Image.BILINEAR)

                    mask_resized = mask_cropped.resize((new_w, new_h), resample=Image.NEAREST)


                    target_w, target_h = final_size
                    pad_left = pad_top = 0
                    if new_w > new_h:
                        pad_total = target_h - new_h
                        pad_top = pad_total // 2
                        pad_bottom = target_h - new_h - pad_top
                        padding = (0, pad_top, 0, pad_bottom)
                    elif new_h > new_w:
                        pad_total = target_w - new_w
                        pad_left = pad_total // 2
                        pad_right = target_w - new_w - pad_left
                        padding = (pad_left, 0, pad_right, 0)
                    else:
                        padding = (0, 0, 0, 0)

                    img_final = ImageOps.expand(img_resized, border=padding, fill=(0, 0, 0))

                    mask_final = ImageOps.expand(mask_resized, border=padding, fill=0)

                    assert img_final.size == (target_w, target_h)
                    assert mask_final.size == (target_w, target_h)


                camera_output_dir = subject_output_dir / cam_id
                camera_output_dir.mkdir(parents=True, exist_ok=True)
                mask_output_dir = camera_output_dir / "mask" / "pha"
                mask_output_dir.mkdir(parents=True, exist_ok=True)


                if export_2d_keypoints:
                    try:
                        annots_list = annot_data.get("annots", [])
                        if annots_list and "keypoints" in annots_list[0]:
                            kps_full = annots_list[0]["keypoints"]

                            if isinstance(kps_full, list) and len(kps_full) >= 25:
                                kps_full_25 = kps_full[:25]
                            else:
                                kps_full_25 = kps_full

                            kps_final = _transform_openpose_keypoints_fullres_to_final(
                                keypoints_full=kps_full_25,
                                transform_params=transform_params,
                                conf_threshold=float(conf_threshold),
                            )

                            kps_dir = camera_output_dir / "keypoints_2d"
                            kps_dir.mkdir(parents=True, exist_ok=True)
                            kps_path = kps_dir / "0000_openpose_body25.json"


                            should_write_kps = True
                            if not only_update_2d_keypoints and skip_existing and kps_path.exists():
                                should_write_kps = False

                            if should_write_kps:
                                payload = {
                                    "format": "openpose_body25",
                                    "image_size": [target_w, target_h],
                                    "keypoints_mode": keypoints_mode,
                                    "conf_threshold": float(conf_threshold),


                                    "recommended_use_mask": recommended_use_mask,
                                    "keypoints": kps_final,
                                    "src_full_hw": [int(full_h), int(full_w)],
                                    "src_lr_hw": [int(lr_h), int(lr_w)],
                                    "transform": {
                                        "sx": float(transform_params["sx"]),
                                        "sy": float(transform_params["sy"]),
                                        "crop_box_lr_int": list(crop_box_lr_int),
                                        "new_hw": [int(transform_params["new_h"]), int(transform_params["new_w"])],
                                        "scale_xy": [float(transform_params["scale_x"]), float(transform_params["scale_y"])],
                                        "pad_left_top": [int(transform_params["pad_left"]), int(transform_params["pad_top"])],
                                        "target_wh": [int(target_w), int(target_h)],
                                    },
                                }
                                with open(kps_path, "w") as f:
                                    json.dump(payload, f, indent=2)
                    except Exception as e:
                        print(f"    Warning: Failed to write 2D keypoints for camera {cam_id}: {e}")


                img_out_path = camera_output_dir / "0000.jpg"
                mask_out_path = mask_output_dir / "0000.png"

                if not only_update_2d_keypoints:

                    if skip_existing and img_out_path.exists():
                        print(f"    RGB already exists at {img_out_path}, skipping RGB write.")
                    else:
                        img_final.save(img_out_path, "JPEG", quality=95)

                    if skip_existing and mask_out_path.exists():
                        print(f"    Mask already exists at {mask_out_path}, skipping mask write.")
                    else:
                        mask_final.save(mask_out_path, "PNG")


                R = extrinsics_by_cam[cam_id]["R"]
                T = extrinsics_by_cam[cam_id]["T"]
                calibration_data[cam_id] = {
                    "K": K_final.tolist(),
                    "R": R.tolist(),
                    "T": T.tolist(),
                    "imgSize": [target_w, target_h],
                }
                n_cameras_used += 1

            except Exception as e:
                print(f"    Error processing camera {cam_id} for {subject_frame_name}: {e}")
                continue

        if not calibration_data:
            print(f"  No valid cameras processed for {subject_frame_name}, skipping subject.")

            continue

        if not only_update_2d_keypoints:

            calib_output_path = subject_output_dir / "calibration_full.json"
            with open(calib_output_path, "w") as f:
                json.dump(calibration_data, f, indent=4)
            print(f"  Saved calibration_full.json with {n_cameras_used} cameras to {calib_output_path}")

        if not only_update_2d_keypoints:

            mvhumannet_metadata = {
                subject_frame_name: {
                    "original_subject": subj_id_str,
                    "frame_id": frame_str,
                    "num_cameras_used": n_cameras_used,
                    "imgSize": [1024, 1024],
                    "gender": gender,
                }
            }
            metadata_output_path = subject_output_dir / "mvhumannet_metadata.json"
            with open(metadata_output_path, "w") as f:
                json.dump(mvhumannet_metadata, f, indent=4)
            print(f"  Saved mvhumannet_metadata.json to {metadata_output_path}")


            if use_gender:
                try:
                    gender_path = subject_output_dir / "gender.txt"
                    with open(gender_path, "w") as f:
                        f.write(gender)
                    print(f"  Saved gender.txt ({gender}) to {gender_path}")
                except Exception as e:
                    print(f"  Warning: Failed to write gender.txt for {subject_frame_name}: {e}")

        processed_subjects.append(subject_frame_name)

    print(f"\nConversion complete for MVHumanNet subject-frames: {processed_subjects}")
    print(f"Output data at: {avatarrex_root_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MVHumanNet dataset to AvatarREX format for selected subject-frame pairs."
    )
    parser.add_argument(
        "--mvhumannet_dir",
        type=str,
        required=True,
        help="Path to the MVHumanNet data directory (containing subject subdirectories like 100001).",
    )
    parser.add_argument(
        "--avatarrex_dir",
        type=str,
        required=True,
        help="Path to the output directory for the AvatarREX-formatted dataset.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--splits",
        nargs="+",
        type=str,
        help=(
            "One or more txt files containing ssssss_ffff tokens "
            "(e.g., /path/to/vton360_mvhumannet_train_subjects.txt)."
        ),
    )
    group.add_argument(
        "--subjects",
        nargs="+",
        type=str,
        help="Explicit list of MVHumanNet subject-frame tokens like '100001_1185'.",
    )
    parser.add_argument(
        "--no_copy_mesh",
        dest="copy_mesh",
        action="store_false",
        help="Disable copying SMPL-X mesh into each AvatarREX subject directory.",
    )
    parser.add_argument(
        "--no_gender",
        dest="use_gender",
        action="store_false",
        help="Disable gender inference and writing gender.txt from text_description_*.json.",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Disable skipping of existing outputs; always overwrite files.",
    )
    parser.add_argument(
        "--only_update_smpl_params",
        action="store_true",
        help=(
            "Only (re)generate smpl_params.npz for each selected subject-frame, "
            "skipping camera/image/mask processing. This overwrites smpl_params.npz."
        ),
    )
    parser.add_argument(
        "--only_update_2d_keypoints",
        action="store_true",
        help=(
            "Only (re)generate per-camera 2D OpenPose keypoints JSONs for each selected subject-frame, "
            "skipping RGB/mask/calibration generation. This overwrites keypoint files."
        ),
    )
    parser.add_argument(
        "--no_2d_keypoints",
        dest="export_2d_keypoints",
        action="store_false",
        help="Disable exporting per-camera 2D OpenPose keypoints JSONs.",
    )
    parser.add_argument(
        "--keypoints_mode",
        type=str,
        default="stable",
        choices=["stable", "all"],
        help="Which Body25 joints to use (stable excludes face/toes/heels by default).",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for keypoints; points below are treated as conf=0.",
    )
    parser.add_argument(
        "--bbox_source",
        type=str,
        default="mask",
        choices=["mask", "annot"],
        help=(
            "Where to get the crop bbox. 'mask' computes bbox from fmask_lr (more accurate); "
            "'annot' uses the dataset annotation bbox."
        ),
    )
    parser.add_argument(
        "--mask_bbox_dilate_ratio",
        type=float,
        default=0.05,
        help="Dilation ratio for mask-derived bbox (relative to max(w,h) of the mask bbox).",
    )
    parser.add_argument(
        "--transform_mode",
        type=str,
        default="crop_resize_pad_1024",
        choices=["crop_resize_pad_1024", "no_crop_resize_pad"],
        help=(
            "Image-space transform mode. "
            "'crop_resize_pad_1024' crops to bbox, resizes long side to 1024, then pads to 1024x1024. "
            "'no_crop_resize_pad' exports raw LR images/masks (e.g., 2048x1500) without any transforms."
        ),
    )
    parser.add_argument(
        "--include_cams",
        nargs="+",
        type=str,
        default=None,
        help="Optional allow-list of camera IDs to include (e.g., CC32871A033 CC32871A034).",
    )
    parser.add_argument(
        "--exclude_cams",
        nargs="+",
        type=str,
        default=None,
        help="Optional list of camera IDs to exclude (e.g., CC32871A046).",
    )
    parser.add_argument(
        "--exclude_cams_regex",
        type=str,
        default=None,
        help="Optional regex; any camera ID matching it will be excluded (e.g., 'CC32871A046').",
    )

    args = parser.parse_args()

    if not Path(args.mvhumannet_dir).exists():
        print(f"Error: MVHumanNet directory not found at {args.mvhumannet_dir}")
    else:
        convert_dataset(
            args.mvhumannet_dir,
            args.avatarrex_dir,
            split_files=args.splits,
            subjects=args.subjects,
            copy_mesh=args.copy_mesh,
            use_gender=args.use_gender,
            skip_existing=args.skip_existing,
            only_update_smpl_params=args.only_update_smpl_params,
            export_2d_keypoints=args.export_2d_keypoints,
            only_update_2d_keypoints=args.only_update_2d_keypoints,
            keypoints_mode=args.keypoints_mode,
            conf_threshold=args.conf_threshold,
            bbox_source=args.bbox_source,
            mask_bbox_dilate_ratio=float(args.mask_bbox_dilate_ratio),
            transform_mode=args.transform_mode,
            include_cams=args.include_cams,
            exclude_cams=args.exclude_cams,
            exclude_cams_regex=args.exclude_cams_regex,
        )
