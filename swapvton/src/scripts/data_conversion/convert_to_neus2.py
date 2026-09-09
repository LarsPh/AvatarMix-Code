import os
import os.path as osp
import json
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from loguru import logger
import re

def _normalize_cam_id(cam: str) -> str:
    s = str(cam).strip()
    if s.isdigit():
        return s.zfill(3)
    return s

def find_mask_file(subject_dir, cam, frame, src_pad_digit=8):

    base_path = osp.join(subject_dir, cam, "mask", "pha", f"{frame:0{src_pad_digit}d}")
    for ext in ['.jpg', '.png']:
        mask_path = base_path + ext
        if osp.exists(mask_path):
            return mask_path
    raise FileNotFoundError(f"No mask file found for {base_path} with .jpg or .png extension")

def copy_imgs_masks(out_img_dir, subject_dir, frame, target_frame=None, src_pad_digit=8, exclude_cams=None):
    target_frame = target_frame if target_frame is not None else frame
    is_cam_dir_fn = lambda x: os.path.isdir(os.path.join(subject_dir, x)) and \
        (x.isdigit() or "_p" in x or x.startswith("CC"))
    cams = [d for d in os.listdir(subject_dir) if is_cam_dir_fn(d)]
    if exclude_cams:
        excl = {_normalize_cam_id(x) for x in exclude_cams if str(x).strip()}
        if excl:
            cams = [c for c in cams if _normalize_cam_id(c) not in excl]
    img_path_abs_list = []
    cams_used = []
    os.makedirs(osp.join(out_img_dir, f"{target_frame:04d}"), exist_ok=True)
    for cam in cams:

        img_src_path = osp.join(subject_dir, cam, f"{frame:0{src_pad_digit}d}.jpg")
        img_tgt_path = osp.join(out_img_dir, f"{target_frame:04d}", f"{cam}_{target_frame:04d}.png")
        if osp.exists(img_tgt_path):
            logger.info(f"skip {img_tgt_path} because it already exists")
            img_path_abs_list.append(img_tgt_path)
            cams_used.append(cam)
            continue
        if not osp.exists(img_src_path):
            logger.warning(f"missing image for cam {cam}: {img_src_path} (skipping cam)")
            continue

        try:
            mask_src_path = find_mask_file(subject_dir, cam, frame, src_pad_digit)
        except FileNotFoundError as e:
            logger.warning(f"{e} (skipping cam {cam})")
            continue

        img = cv2.imread(img_src_path)
        if img is None:
            logger.warning(f"unreadable image for cam {cam}: {img_src_path} (skipping cam)")
            continue
        mask = cv2.imread(mask_src_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"unreadable mask for cam {cam}: {mask_src_path} (skipping cam)")
            continue
        mask = mask / 255.0

        rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = img
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        cv2.imwrite(img_tgt_path, rgba)
        img_path_abs_list.append(img_tgt_path)
        cams_used.append(cam)

    return img_path_abs_list, cams_used

def convert_calibs(img_path_rel_list, calib_data, cams, target_frame, scale, offset, output_path):

    all_dict_list = []
    for mode in ["all","train", "test"]:
        out_dict = {
            "frames": []
        }
        for cam, img_path_rel in zip(cams, img_path_rel_list):
            R = calib_data[cam]["R"]
            T = calib_data[cam]["T"]
            K_ = calib_data[cam]["K"]
            dist_coeff = calib_data[cam].get("distCoeff", [0.0, 0.0, 0.0, 0.0, 0.0])
            img_size = calib_data[cam]["imgSize"]

            assert dist_coeff == [0.0, 0.0, 0.0, 0.0, 0.0]
            w, h = img_size
            R = np.array(R).reshape(3, 3)
            T = np.array(T).reshape(3, 1)
            Rt = np.concatenate([R, T], axis=1)
            Rt = np.concatenate([Rt, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)


            pose = np.linalg.inv(Rt)
            K = np.eye(4)
            K[:3, :3] = np.array(K_).reshape(3, 3)

            frame_dict = {
                "file_path": img_path_rel if mode == "all" else osp.join("..", img_path_rel),
                "transform_matrix": pose.tolist(),
                "intrinsic_matrix": K.tolist(),
            }
            out_dict["frames"].append(frame_dict)


            frame_dict.update({
                "h": h,
                "w": w,
                "fl_x": K[0, 0],
                "fl_y": K[1, 1],
                "cx": K[0, 2],
                "cy": K[1, 2],
            })
        out_dict.update(
            {


                "aabb_scale": 1,
                "scale": scale,
                "offset": offset,
                "from_na": False,
            }
        )
        all_dict_list.append(out_dict)

    out_dict_test = all_dict_list[-1]
    test_step = len(out_dict_test["frames"]) // 4
    out_dict_test["frames"] = out_dict_test["frames"][::test_step]
    output_path_test = osp.join(output_path, "test")
    os.makedirs(output_path_test, exist_ok=True)
    with open(osp.join(output_path_test, f"transforms_{target_frame:04d}.json"), "w") as f:
        json.dump(out_dict_test, f, indent=4)

    out_dict_train = all_dict_list[1]
    output_path_train = osp.join(output_path, "train")
    os.makedirs(output_path_train, exist_ok=True)
    with open(osp.join(output_path_train, f"transforms_{target_frame:04d}.json"), "w") as f:
        json.dump(out_dict_train, f, indent=4)

    out_dict = all_dict_list[0]
    with open(osp.join(output_path, f"transforms_{target_frame:04d}.json"), "w") as f:
        json.dump(out_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True)

    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["single_frame", "multi_frame"], required=True, help="Processing mode: single_frame for THuman2, multi_frame for ActorsHQ/AvatarRex")
    parser.add_argument("--subjects", nargs="+", type=str, help="List of subjects to process")

    parser.add_argument("--subject_start", type=int, default=0, help="[Legacy] Start subject ID for single_frame mode")
    parser.add_argument("--subject_end", type=int, default=-1, help="[Legacy] End subject ID for single_frame mode")

    parser.add_argument("--frame_padding", type=int, default=8, help="Frame number padding for multi_frame mode")
    parser.add_argument("--max_frames", type=int, default=2001, help="Maximum number of frames to process in multi_frame mode")
    parser.add_argument("--human_scale", type=float, default=1.0, help="Scale factor for the dataset")
    parser.add_argument("--human_offset", type=float, nargs=3, default=[0.5, 0.5, 0.5], help="Offset for the dataset")
    parser.add_argument(
        "--exclude_cams",
        nargs="+",
        type=str,
        default=None,
        help="Camera ids to skip (e.g., '013' or '13'). Only affects exported camera folders under src_root.",
    )
    args = parser.parse_args()
    src_root = args.src_root
    output_root = args.output_root

    if args.mode == "single_frame":

        if not args.subjects:
            logger.info(f"Processing single_frame dataset from: {src_root}")

            try:
                def is_valid_subject_dir(d):
                    return d.isdigit() or re.match(r'^\d+_\d+$', d)
                subject_dirs_all = sorted([d for d in os.listdir(src_root) if osp.isdir(osp.join(src_root, d)) and is_valid_subject_dir(d)])
            except FileNotFoundError:
                logger.error(f"Error: Source directory {src_root} not found.")
                subject_dirs_all = []

            if not subject_dirs_all:
                logger.error(f"No subject subdirectories found in {src_root}.")


            subject_list_to_process = []
            if args.subject_end == -1:
                subject_list_to_process = subject_dirs_all
            else:
                start_idx = args.subject_start
                temp_subject_ids_int = []
                for s_dir in subject_dirs_all:
                    try:
                        temp_subject_ids_int.append(int(s_dir))
                    except ValueError:
                        logger.warning(f"Non-numeric subject directory found and skipped: {s_dir}")

                temp_subject_ids_int.sort()

                for subj_int_id in temp_subject_ids_int:
                    if subj_int_id >= args.subject_start and subj_int_id <= args.subject_end:
                         subject_list_to_process.append(f"{subj_int_id:04d}")

            if not subject_list_to_process and subject_dirs_all and args.subject_end != -1 :
                 logger.warning(f"No subjects in the range {args.subject_start}-{args.subject_end} found in {subject_dirs_all}")
        else:

            subject_list_to_process = args.subjects
            logger.info(f"Processing explicitly provided subjects: {subject_list_to_process}")

        logger.info(f"Found single_frame subjects to process: {subject_list_to_process}")

        for subject_id_str in tqdm(subject_list_to_process, desc="Processing single_frame Subjects"):
            current_subject_data_dir = osp.join(src_root, subject_id_str)
            if not osp.isdir(current_subject_data_dir):
                logger.warning(f"Skipping {subject_id_str}, not a directory: {current_subject_data_dir}")
                continue

            logger.info(f"\nProcessing subject: {subject_id_str}")

            calib_path = osp.join(current_subject_data_dir, "calibration_full.json")
            if not osp.exists(calib_path):
                logger.warning(f"  Calibration file not found for subject {subject_id_str} at {calib_path}, skipping.")
                continue
            with open(calib_path, "r") as f:
                calib_data = json.load(f)

            output_path_subject = osp.join(output_root, subject_id_str)

            os.makedirs(output_path_subject, exist_ok=True)
            img_dir_neus2 = osp.join(output_path_subject, "images")
            os.makedirs(img_dir_neus2, exist_ok=True)


            frame_id_for_copy = 0
            src_pad_digit_for_copy = 4

            logger.info(f"  Copying images and masks from: {current_subject_data_dir}")
            img_path_abs_list, cams = copy_imgs_masks(
                out_img_dir=img_dir_neus2,
                subject_dir=current_subject_data_dir,
                frame=frame_id_for_copy,
                target_frame=0,
                src_pad_digit=src_pad_digit_for_copy,
                exclude_cams=args.exclude_cams,
            )

            if not img_path_abs_list:
                logger.warning(f"  No images/masks copied for subject {subject_id_str}. Skipping calibration conversion.")
                continue

            img_path_rel_list = [osp.relpath(p, output_path_subject) for p in img_path_abs_list]

            logger.info(f"  Converting calibrations for target_frame 0")
            convert_calibs(img_path_rel_list, calib_data, cams, target_frame=0, scale=args.human_scale, offset=args.human_offset, output_path=output_path_subject)


    elif args.mode == "multi_frame":
        logger.info(f"Processing multi_frame dataset from: {src_root}")
        if not args.subjects:
            raise ValueError("--subjects is required for multi_frame mode")

        subject_list = args.subjects
        frames = list(range(args.max_frames))

        for subject in subject_list:
            subject_dir = osp.join(src_root, subject)
            if not osp.exists(subject_dir):
                logger.warning(f"Warning: Subject directory {subject_dir} not found, skipping.")
                continue

            calib_path = osp.join(subject_dir, "calibration_full.json")
            if not osp.exists(calib_path):
                logger.warning(f"Warning: Calibration file {calib_path} not found, skipping subject {subject}.")
                continue

            with open(calib_path, "r") as f:
                calib_data = json.load(f)

            for frame in tqdm(frames, desc=f"Processing frames for {subject}"):
                output_path = osp.join(output_root, subject)

                os.makedirs(output_path, exist_ok=True)
                img_dir = osp.join(output_path, "images")
                os.makedirs(img_dir, exist_ok=True)

                try:
                    img_path_abs_list, cams = copy_imgs_masks(
                        img_dir, subject_dir, frame, src_pad_digit=args.frame_padding, exclude_cams=args.exclude_cams
                    )
                    if not img_path_abs_list:
                        continue
                    img_path_rel_list = [osp.relpath(p, output_path) for p in img_path_abs_list]
                    convert_calibs(img_path_rel_list, calib_data, cams, frame, scale=args.human_scale, offset=args.human_offset, output_path=output_path)
                except Exception as e:
                    logger.error(f"Error processing frame {frame} for subject {subject}: {e}")
                    continue
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Use 'single_frame' or 'multi_frame'.")
