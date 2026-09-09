import argparse
import json
import numpy as np
from PIL import Image
import csv
from pathlib import Path
import shutil
import os
from typing import List, Tuple, Dict
import re
from tqdm import tqdm


from dataclasses import dataclass

try:
    from scipy.spatial.transform import Rotation
except ImportError:

    pass

@dataclass
class CameraData:

    name: str
    width: int
    height: int
    rotation_axisangle: np.array
    translation: np.array
    focal_length: np.array
    principal_point: np.array
    k1: float = 0
    k2: float = 0
    k3: float = 0

    @property
    def fx_pixel(self):
        return self.width * self.focal_length[0]

    @property
    def fy_pixel(self):
        return self.height * self.focal_length[1]

    @property
    def cx_pixel(self):
        return self.width * self.principal_point[0]

    @property
    def cy_pixel(self):
        return self.height * self.principal_point[1]

    def intrinsic_matrix(self):
        return np.array([
            [self.fx_pixel, 0, self.cx_pixel],
            [0, self.fy_pixel, self.cy_pixel],
            [0, 0, 1],
        ])

    def rotation_matrix_cam2world(self) -> np.array:

        return Rotation.from_rotvec(self.rotation_axisangle).as_matrix()

    def extrinsic_matrix_cam2world(self) -> np.array:

        tfm_cam2world = np.eye(4)
        tfm_cam2world[:3, :3] = self.rotation_matrix_cam2world()
        tfm_cam2world[:3, 3] = self.translation
        return tfm_cam2world


def read_calibration_csv(input_csv_path: Path) -> List[CameraData]:

    cameras = []
    with open(input_csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            camera = CameraData(
                name=row["name"],
                width=int(row["w"]),
                height=int(row["h"]),
                rotation_axisangle=np.array([float(row["rx"]), float(row["ry"]), float(row["rz"])]),
                translation=np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])]),
                focal_length=np.array([float(row["fx"]), float(row["fy"])]),
                principal_point=np.array([float(row["px"]), float(row["py"])]),
            )
            cameras.append(camera)
    return cameras


def _get_available_cameras(actorshq_resolution_dir: Path) -> List[str]:

    rgbs_dir = actorshq_resolution_dir / "rgbs"
    if not rgbs_dir.exists():
        return []

    camera_dirs = [d.name for d in rgbs_dir.iterdir() if d.is_dir()]
    return sorted(camera_dirs)


def _get_frame_count_from_scene(sequence_dir: Path) -> int:

    scene_json_path = sequence_dir / "scene.json"
    if not scene_json_path.exists():
        raise FileNotFoundError(f"scene.json not found at {scene_json_path}")

    with open(scene_json_path, "r") as f:
        scene_data = json.load(f)

    return scene_data["num_frames"]


def _extract_frame_number_from_filename(filename: str) -> int:


    parts = filename.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid filename format: {filename}")


    frame_part = parts[-1].split(".")[0]

    frame_num_str = ''.join(filter(str.isdigit, frame_part))
    if not frame_num_str:
        raise ValueError(f"No frame number found in filename: {filename}")

    return int(frame_num_str)


def _load_and_convert_smplx_params(smplx_file_path: Path) -> Dict:

    if not smplx_file_path.exists():
        raise FileNotFoundError(f"SMPL-X file not found: {smplx_file_path}")

    with open(smplx_file_path, 'r') as f:
        smplx_data = json.load(f)[0]


    converted_params = {}


    if 'poses' in smplx_data:


        assert np.all(np.array(smplx_data['poses'])[:, :3] == 0)
        body_poses = np.array(smplx_data['poses'])[:, 3:]

        body_end = 63
        right_hand_pose = body_poses[:, body_end:body_end+6]
        left_hand_pose = body_poses[:, body_end+6:body_end+6*2]

        hand_end = body_end + 6*2
        jaw_pose = body_poses[:, hand_end:hand_end+3]
        left_eye_pose = body_poses[:, hand_end+3:hand_end+3*2]
        right_eye_pose = body_poses[:, hand_end+3*2:hand_end+3*3]

        converted_params['body_pose'] = body_poses[:,:63].squeeze(0)
        converted_params['right_hand_pose'] = right_hand_pose.squeeze(0)
        converted_params['left_hand_pose'] = left_hand_pose.squeeze(0)
        converted_params['jaw_pose'] = jaw_pose.squeeze(0)
        converted_params['leye_pose'] = left_eye_pose.squeeze(0)
        converted_params['reye_pose'] = right_eye_pose.squeeze(0)

    if 'shapes' in smplx_data:
        converted_params['betas'] = np.array(smplx_data['shapes']).squeeze(0)
    if 'Rh' in smplx_data:
        converted_params['global_orient'] = np.array(smplx_data['Rh']).squeeze(0)
    if 'Th' in smplx_data:
        converted_params['transl'] = np.array(smplx_data['Th']).squeeze(0)


    if 'expression' in smplx_data:
        converted_params['expression'] = np.array(smplx_data['expression']).squeeze(0)


    return converted_params


def _actorshq_gender_from_actor_id(actor_id: str) -> str:

    s = str(actor_id)
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse actor number from actor_id='{actor_id}'")
    actor_num = int(digits)
    if actor_num in (2, 5, 7, 8):
        return "male"
    if actor_num in (1, 3, 4, 6):
        return "female"
    raise ValueError(f"Unexpected ActorHQ actor number {actor_num} parsed from actor_id='{actor_id}'")


def _shorten_sequence_name(sequence_name: str) -> str:

    s = str(sequence_name)
    m = re.match(r"^Sequence(\d+)$", s, flags=re.IGNORECASE)
    if m:
        return f"Seq{m.group(1)}"
    if s.lower().startswith("seq"):
        return s
    return s.replace("Sequence", "Seq")


def _pick_single_frame_number(*, available_frame_numbers: List[int], requested_frame: int) -> int:

    if requested_frame in available_frame_numbers:
        return requested_frame
    if 0 <= requested_frame < len(available_frame_numbers):
        return available_frame_numbers[requested_frame]
    raise ValueError(
        f"Requested frame={requested_frame} is neither an existing frame number nor a valid index "
        f"(available frames: {len(available_frame_numbers)})"
    )


def convert_dataset(
    actorshq_data_root: str,
    output_avatarrex_root: str,
    actor_ids: List[str],
    resolution: str = "4x",
    max_cameras: int = None,
    camera_step: int = 1,
    sequences: List[str] = None,
    export_mode: str = "single_frame",
    frame: int = 0,
    overwrite: bool = False,
):

    actorshq_dir = Path(actorshq_data_root)
    avatarrex_root_dir = Path(output_avatarrex_root)

    avatarrex_root_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {avatarrex_root_dir}")

    if not actor_ids:
        print("No actor IDs provided.")
        return

    processed_subjects = []

    for actor_id in actor_ids:
        actor_dir = actorshq_dir / actor_id
        if not actor_dir.exists():
            print(f"Warning: Actor directory {actor_dir} not found, skipping.")
            continue


        sequence_dirs = [d for d in actor_dir.iterdir() if d.is_dir() and d.name.startswith("Sequence")]


        if sequences is not None:
            sequence_dirs = [d for d in sequence_dirs if d.name in sequences]

        sequence_dirs.sort()

        for sequence_dir in sequence_dirs:
            seq_short = _shorten_sequence_name(sequence_dir.name)
            subject_name_base = f"{actor_id}_{seq_short}"
            subject_name = subject_name_base
            print(f"\nProcessing {subject_name_base}")


            resolution_dir = sequence_dir / resolution
            if not resolution_dir.exists():
                print(f"Warning: Resolution directory {resolution_dir} not found, skipping.")
                continue


            try:
                num_frames = _get_frame_count_from_scene(sequence_dir)
                print(f"  Found {num_frames} frames")
            except Exception as e:
                print(f"  Error reading scene.json: {e}, skipping.")
                continue


            calibration_csv_path = resolution_dir / "calibration.csv"
            if not calibration_csv_path.exists():
                print(f"  Warning: Calibration file {calibration_csv_path} not found, skipping.")
                continue

            try:
                cameras = read_calibration_csv(calibration_csv_path)
                print(f"  Found {len(cameras)} cameras in calibration")
            except Exception as e:
                print(f"  Error reading calibration: {e}, skipping.")
                continue


            available_cameras = _get_available_cameras(resolution_dir)
            camera_dict = {cam.name: cam for cam in cameras}


            valid_cameras = [camera_dict[cam_name] for cam_name in available_cameras
                           if cam_name in camera_dict]


            if camera_step > 1:
                valid_cameras = valid_cameras[::camera_step]

            if max_cameras is not None:
                valid_cameras = valid_cameras[:max_cameras]

            print(f"  Using {len(valid_cameras)} cameras: {[cam.name for cam in valid_cameras]}")

            if not valid_cameras:
                print(f"  No valid cameras found, skipping {subject_name}")
                continue


            first_camera = valid_cameras[0]
            rgb_camera_dir = resolution_dir / "rgbs" / first_camera.name
            all_rgb_files = list(rgb_camera_dir.glob(f"{first_camera.name}_rgb*.jpg"))
            all_rgb_files.sort(key=lambda x: _extract_frame_number_from_filename(x.name))


            smplx_dir = sequence_dir / "smplx"
            if export_mode not in ("single_frame", "sequence"):
                raise ValueError(f"Unsupported export_mode='{export_mode}'. Expected 'single_frame' or 'sequence'.")


            candidate_frames = []
            for rgb_file in all_rgb_files:
                try:
                    fn = _extract_frame_number_from_filename(rgb_file.name)
                except Exception:
                    continue
                if (smplx_dir / f"{fn:06d}.json").exists():
                    candidate_frames.append(fn)
            candidate_frames = sorted(list(dict.fromkeys(candidate_frames)))

            if not candidate_frames:
                print(f"  No SMPL-X jsons found under {smplx_dir}, skipping.")
                continue

            processed_smplx_frames = set()
            picked_frame_number = None

            if export_mode == "single_frame":

                picked_frame_number = _pick_single_frame_number(
                    available_frame_numbers=candidate_frames, requested_frame=int(frame)
                )
                subject_name = f"{subject_name_base}_{picked_frame_number:06d}"
                print(f"  Single-frame export: picked frame {picked_frame_number} -> subject '{subject_name}'")


                subject_output_dir = avatarrex_root_dir / subject_name
                subject_output_dir.mkdir(parents=True, exist_ok=True)


                try:
                    gender = _actorshq_gender_from_actor_id(actor_id)
                    (subject_output_dir / "gender.txt").write_text(gender)
                except Exception as e:
                    print(f"  Warning: could not write gender.txt for {subject_name}: {e}")

                smplx_params = _load_and_convert_smplx_params(smplx_dir / f"{picked_frame_number:06d}.json")
                final_smpl_params = {k: np.expand_dims(v.astype(np.float32), axis=0) for k, v in smplx_params.items()}
                smpl_npz_path = subject_output_dir / "smpl_params.npz"
                np.savez(smpl_npz_path, **final_smpl_params)
                print(f"  Saved smpl_params.npz (single-frame)")

                processed_smplx_frames.add(picked_frame_number)

            else:

                subject_name = subject_name_base
                print(f"  Sequence export: subject '{subject_name}'")

                subject_output_dir = avatarrex_root_dir / subject_name
                subject_output_dir.mkdir(parents=True, exist_ok=True)


                try:
                    gender = _actorshq_gender_from_actor_id(actor_id)
                    (subject_output_dir / "gender.txt").write_text(gender)
                except Exception as e:
                    print(f"  Warning: could not write gender.txt for {subject_name}: {e}")

                smplx_params_by_frame = {}
                print(f"  Processing SMPL-X parameters...")
                for frame_number in tqdm(candidate_frames, desc=f"    Processing SMPL-X for {subject_name}", leave=False):
                    try:
                        smplx_params = _load_and_convert_smplx_params(smplx_dir / f"{frame_number:06d}.json")
                        smplx_params_by_frame[frame_number] = smplx_params
                        processed_smplx_frames.add(frame_number)
                    except Exception as e:
                        print(f"      Warning: Error processing SMPL-X for frame {frame_number}: {e}")
                        continue

                print(f"  Processed SMPL-X parameters for {len(processed_smplx_frames)} frames")
                if smplx_params_by_frame:
                    sorted_frames = sorted(smplx_params_by_frame.keys())
                    first_frame_params = smplx_params_by_frame[sorted_frames[0]]
                    param_names = list(first_frame_params.keys())
                    final_smpl_params = {}
                    num_frames_effective = len(sorted_frames)

                    for param_name in param_names:
                        first_param = first_frame_params[param_name]
                        param_shape = first_param.shape
                        param_array = np.zeros((num_frames_effective, *param_shape), dtype=np.float32)
                        for i, frame_num in enumerate(sorted_frames):
                            if param_name in smplx_params_by_frame[frame_num]:
                                param_array[i] = smplx_params_by_frame[frame_num][param_name]
                        final_smpl_params[param_name] = param_array


                    assert np.all(final_smpl_params['betas'] == final_smpl_params['betas'][:1])
                    final_smpl_params['betas'] = final_smpl_params['betas'][:1]

                    smpl_npz_path = subject_output_dir / "smpl_params.npz"
                    np.savez(smpl_npz_path, **final_smpl_params)
                    print(f"  Saved smpl_params.npz with {num_frames_effective} frames")
                else:
                    print(f"  No SMPL-X parameters to save for {subject_name}")


            calibration_data = {}
            for camera in valid_cameras:

                camera_name_numeric = camera.name.replace("Cam", "") if camera.name.startswith("Cam") else camera.name


                extrinsic_cam2world = camera.extrinsic_matrix_cam2world()

                extrinsic_world2cam = np.linalg.inv(extrinsic_cam2world)
                R_world2cam = extrinsic_world2cam[:3, :3]
                T_world2cam = extrinsic_world2cam[:3, 3]
                intrinsic_matrix = camera.intrinsic_matrix()

                calibration_data[camera_name_numeric] = {
                    "K": intrinsic_matrix.tolist(),
                    "R": R_world2cam.tolist(),
                    "T": T_world2cam.tolist(),
                    "imgSize": [camera.width, camera.height],
                }
            calibration_output_path = subject_output_dir / "calibration_full.json"
            with open(calibration_output_path, "w") as f:
                json.dump(calibration_data, f, indent=4)
            print(f"  Saved calibration.json")


            for camera in tqdm(valid_cameras, desc=f"  Processing cameras for {subject_name}", leave=False):

                camera_name_numeric = camera.name.replace("Cam", "") if camera.name.startswith("Cam") else camera.name
                camera_output_dir = subject_output_dir / camera_name_numeric
                camera_output_dir.mkdir(parents=True, exist_ok=True)


                mask_output_dir = camera_output_dir / "mask" / "pha"
                mask_output_dir.mkdir(parents=True, exist_ok=True)


                rgb_camera_dir = resolution_dir / "rgbs" / camera.name
                mask_camera_dir = resolution_dir / "masks" / camera.name

                if not rgb_camera_dir.exists() or not mask_camera_dir.exists():
                    print(f"    Warning: Missing source data for camera {camera.name}, skipping camera.")
                    continue


                rgb_files = list(rgb_camera_dir.glob(f"{camera.name}_rgb*.jpg"))
                rgb_files.sort(key=lambda x: _extract_frame_number_from_filename(x.name))


                processed_frames = 0
                skipped_existing = 0
                skipped_no_smplx = 0

                for rgb_file in tqdm(rgb_files, desc=f"    Linking frames for {camera.name}", leave=False):
                    try:
                        frame_number = _extract_frame_number_from_filename(rgb_file.name)


                        if export_mode == "single_frame" and picked_frame_number is not None and frame_number != picked_frame_number:
                            continue


                        if frame_number not in processed_smplx_frames:
                            skipped_no_smplx += 1
                            continue


                        mask_filename = rgb_file.name.replace("_rgb", "_mask").replace(".jpg", ".png")
                        mask_file = mask_camera_dir / mask_filename

                        if not mask_file.exists():
                            print(f"      Warning: Mask file {mask_file} not found, skipping frame {frame_number}")
                            continue


                        out_frame_id = 0 if export_mode == "single_frame" else frame_number
                        output_rgb_path = camera_output_dir / f"{out_frame_id:04d}.jpg"
                        output_mask_path = mask_output_dir / f"{out_frame_id:04d}.png"

                        rgb_exists = output_rgb_path.exists()
                        mask_exists = output_mask_path.exists()


                        if (not overwrite) and rgb_exists and mask_exists:
                            skipped_existing += 1
                            continue

                        if overwrite:

                            if rgb_exists:
                                try:
                                    output_rgb_path.unlink()
                                except Exception:
                                    pass
                                rgb_exists = False
                            if mask_exists:
                                try:
                                    output_mask_path.unlink()
                                except Exception:
                                    pass
                                mask_exists = False


                        if not rgb_exists:
                            if export_mode == "single_frame":
                                shutil.copy2(rgb_file, output_rgb_path)
                            else:
                                try:

                                    src_rgb = rgb_file.resolve()
                                    dst_rgb = output_rgb_path.resolve()
                                    os.symlink(src_rgb, dst_rgb)
                                except OSError:

                                    try:
                                        os.link(rgb_file, output_rgb_path)
                                    except OSError:

                                        print(f"      Warning: Could not create link for {rgb_file.name}, copying instead")
                                        shutil.copy2(rgb_file, output_rgb_path)


                        if not mask_exists:
                            if export_mode == "single_frame":
                                shutil.copy2(mask_file, output_mask_path)
                            else:
                                try:

                                    src_mask = mask_file.resolve()
                                    dst_mask = output_mask_path.resolve()
                                    os.symlink(src_mask, dst_mask)
                                except OSError:

                                    try:
                                        os.link(mask_file, output_mask_path)
                                    except OSError:

                                        print(f"      Warning: Could not create link for {mask_filename}, copying instead")
                                        shutil.copy2(mask_file, output_mask_path)

                        processed_frames += 1
                        if export_mode == "single_frame":
                            break

                    except Exception as e:
                        print(f"      Error processing frame {rgb_file.name}: {e}")
                        continue

                if skipped_existing > 0 or skipped_no_smplx > 0:
                    parts = [f"Processed: {processed_frames}"]
                    if skipped_existing > 0:
                        parts.append(f"Skipped existing: {skipped_existing}")
                    if skipped_no_smplx > 0:
                        parts.append(f"Skipped no SMPL-X: {skipped_no_smplx}")
                    print(f"      {', '.join(parts)}")
                else:
                    print(f"      Processed: {processed_frames}")


            actorshq_metadata = {
                subject_name: {
                    "original_actor": actor_id,
                    "sequence": sequence_dir.name,
                    "sequence_short": seq_short,
                    "resolution": resolution,
                    "export_mode": export_mode,
                    "selected_frame": int(picked_frame_number) if export_mode == "single_frame" else None,
                    "scene_num_frames": num_frames,
                    "export_num_frames": 1 if export_mode == "single_frame" else int(len(processed_smplx_frames)),
                    "num_cameras_used": len(valid_cameras),
                    "camera_step": camera_step,
                    "max_cameras": max_cameras,
                }
            }
            metadata_output_path = subject_output_dir / "actorshq_metadata.json"
            with open(metadata_output_path, "w") as f:
                json.dump(actorshq_metadata, f, indent=4)
            print(f"  Saved actorshq_metadata.json")


            processed_subjects.append(subject_name)

    print(f"\nConversion complete for subjects: {processed_subjects}")
    print(f"Output data at: {avatarrex_root_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ActorsHQ dataset to AvatarREX format."
    )
    parser.add_argument(
        "--actorshq_dir",
        type=str,
        required=True,
        help="Path to the ActorsHQ data directory (e.g., ./actorshq/data).",
    )
    parser.add_argument(
        "--avatarrex_dir",
        type=str,
        required=True,
        help="Path to the output directory for the AvatarREX-formatted dataset.",
    )
    parser.add_argument(
        "--actors",
        nargs="+",
        type=str,
        required=True,
        help="List of ActorsHQ actor IDs to process (e.g., Actor01 Actor05).",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        type=str,
        default=None,
        help=(
            "Optional per-subject tokens to convert. If provided, overrides --actors/--sequences.\n"
            "Supported forms:\n"
            "  Actor02_Sequence1\n"
            "  Actor02_Sequence1_0202   (4-6 digit frame selector; normalized to 6 digits)\n"
            "  Actor02_Seq1_000202\n"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="4x",
        help="Resolution to use (default: 4x).",
    )
    parser.add_argument(
        "--max_cameras",
        type=int,
        default=None,
        help="Maximum number of cameras to use (default: all available).",
    )
    parser.add_argument(
        "--camera_step",
        type=int,
        default=1,
        help="Step size for camera selection, e.g., 2 for every other camera (default: 1).",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        type=str,
        default=None,
        help="List of sequence names to process (e.g., Sequence1 Sequence2). If not specified, all sequences will be processed.",
    )
    parser.add_argument(
        "--export_mode",
        type=str,
        default="single_frame",
        choices=["single_frame", "sequence"],
        help="Export mode. 'single_frame' creates Actor05_Seq1_000000-style subjects (default). 'sequence' keeps legacy multi-frame layout.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame selector for export_mode=single_frame. If this matches an existing frame number, it is used; otherwise treated as an index into available frames (0-based).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (RGB/mask). Default: skip only frames where both rgb and mask already exist; write any missing frame.",
    )

    args = parser.parse_args()

    if not Path(args.actorshq_dir).exists():
        print(f"Error: ActorsHQ directory not found at {args.actorshq_dir}")
    else:

        if args.subjects:
            subj_tokens = [str(s).strip() for s in args.subjects if str(s).strip()]
            for tok in subj_tokens:
                parts = tok.split("_")
                if len(parts) < 2:
                    raise ValueError(f"Invalid ActorHQ subject token '{tok}'. Expected ActorXX_SequenceY[_frame].")
                actor_id = parts[0]
                seq_in = parts[1]

                frame_sel = None
                if len(parts) >= 3 and parts[-1].isdigit() and 4 <= len(parts[-1]) <= 6:
                    frame_sel = int(parts[-1])


                if str(seq_in).lower().startswith("seq") and str(seq_in)[3:].isdigit():
                    sequence_disk = f"Sequence{int(str(seq_in)[3:])}"
                else:
                    sequence_disk = seq_in

                convert_dataset(
                    actorshq_data_root=args.actorshq_dir,
                    output_avatarrex_root=args.avatarrex_dir,
                    actor_ids=[actor_id],
                    resolution=args.resolution,
                    max_cameras=args.max_cameras,
                    camera_step=args.camera_step,
                    sequences=[sequence_disk],
                    export_mode=args.export_mode,
                    frame=int(frame_sel if frame_sel is not None else args.frame),
                    overwrite=bool(args.overwrite),
                )
        else:
            convert_dataset(
                args.actorshq_dir,
                args.avatarrex_dir,
                args.actors,
                args.resolution,
                args.max_cameras,
                args.camera_step,
                args.sequences,
                args.export_mode,
                args.frame,
                args.overwrite,
            )
