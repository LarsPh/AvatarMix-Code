import argparse
import json
import numpy as np
from PIL import Image
import pickle
from pathlib import Path
import shutil


def _generate_thuman2_camera_names():

    camera_names = []

    for yaw in range(0, 360, 10):
        camera_names.append(f"{yaw:03d}_p+00")

    for yaw in range(5, 360, 20):
        if yaw <= 345:
            camera_names.append(f"{yaw:03d}_p+20")

    for yaw in range(15, 360, 20):
        if yaw <= 355:
            camera_names.append(f"{yaw:03d}_p-20")
    return camera_names


def _parse_thuman2_calib_file(calib_file_path):

    with open(calib_file_path, "r") as f:
        lines = f.readlines()

    extr_lines = [list(map(float, line.strip().split())) for line in lines[0:4]]
    intr_lines = [list(map(float, line.strip().split())) for line in lines[4:8]]

    E = np.array(extr_lines)
    R_w2c = E[:3, :3]
    T_w2c = E[:3, 3].reshape(3, 1)

    K_mat_from_file = np.array(intr_lines)[:3, :3]
    K = K_mat_from_file

    near_far_line = lines[8].strip().split()
    near_plane = float(near_far_line[0])
    far_plane = float(near_far_line[1])

    return K, R_w2c, T_w2c, near_plane, far_plane


def convert_dataset(thuman2_data_root, output_avatarrex_root, subject_ids_str_list):

    thuman2_dir = Path(thuman2_data_root)
    overall_avatarrex_root_dir = Path(output_avatarrex_root)

    overall_avatarrex_root_dir.mkdir(parents=True, exist_ok=True)
    print(f"Root output directory: {overall_avatarrex_root_dir}")

    if not subject_ids_str_list:
        print(
            "No subject IDs provided. Please provide a list of subject IDs to process."
        )
        return

    subject_ids_int_all = sorted([int(s_id) for s_id in subject_ids_str_list])
    if not subject_ids_int_all:
        print("Parsed subject ID list is empty.")
        return


    avatarrex_smpl_keys = [
        "betas",
        "body_pose",
        "global_orient",
        "transl",
        "jaw_pose",
        "expression",
        "left_hand_pose",
        "right_hand_pose",
        "leye_pose",
        "reye_pose",
        "scale",
    ]
    thuman2_smpl_key_rename_map = {
        "translation": "transl",
    }
    default_smpl_param_shapes = {}


    if subject_ids_int_all:
        first_subject_for_shape_check_id_str = f"{subject_ids_int_all[0]:04d}"
        first_smplx_pkl_path_for_shape = (
            thuman2_dir / "smplx" / f"{first_subject_for_shape_check_id_str}.pkl"
        )
        if first_smplx_pkl_path_for_shape.exists():
            try:
                with open(first_smplx_pkl_path_for_shape, "rb") as f:
                    raw_params_shape_check = pickle.load(f, encoding="latin1")
                for key in avatarrex_smpl_keys:

                    original_key_for_check = key
                    for th_key, ar_key in thuman2_smpl_key_rename_map.items():
                        if ar_key == key:
                            original_key_for_check = th_key
                            break

                    if original_key_for_check in raw_params_shape_check:
                        param_val = np.array(raw_params_shape_check[original_key_for_check]).astype(np.float32)
                        if param_val.ndim > 1 and param_val.shape[0] == 1:
                            data_shape = param_val[0].shape
                        elif param_val.ndim == 0:
                            data_shape = (1,)
                        else:
                            data_shape = param_val.shape
                        default_smpl_param_shapes[key] = data_shape
                print(f"Determined global default SMPL param shapes: {default_smpl_param_shapes}")
            except Exception as e:
                print(
                    f"Warning: Could not load SMPL params from {first_smplx_pkl_path_for_shape} to determine global shapes: {e}"
                )
        else:
            print(
                f"Warning: SMPL PKL for first subject {first_subject_for_shape_check_id_str} not found for global shape determination."
            )
            print(
                "Will attempt to determine shapes from other subjects, but this might lead to inconsistencies if subjects are missing keys."
            )

    processed_subject_ids_log = []

    for subj_int_id in subject_ids_int_all:
        subj_id_str_padded = f"{subj_int_id:04d}"
        print(f"\nProcessing THuman2 subject: {subj_id_str_padded}")


        avatarrex_subj_dir = overall_avatarrex_root_dir / subj_id_str_padded
        avatarrex_subj_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Outputting to subject directory: {avatarrex_subj_dir}")


        calibration_full_data_subject = {}
        thuman2_supplementary_metadata_subject = {subj_id_str_padded: {}}


        smpl_params_collector_subject = {
            key: [None] for key in avatarrex_smpl_keys
        }


        avatarrex_smplx_mesh_dir_subject = avatarrex_subj_dir / "mesh" / "processed"
        avatarrex_smplx_mesh_dir_subject.mkdir(parents=True, exist_ok=True)

        thuman2_subj_render_dir = thuman2_dir / "render" / subj_id_str_padded
        thuman2_subj_smplx_dir = thuman2_dir / "smplx"


        smplx_pkl_path = thuman2_subj_smplx_dir / f"{subj_id_str_padded}.pkl"
        subject_has_smpl_data = False
        if smplx_pkl_path.exists():
            try:
                with open(smplx_pkl_path, "rb") as f:
                    raw_params = pickle.load(f, encoding="latin1")

                for old_key, new_key in thuman2_smpl_key_rename_map.items():
                    if old_key in raw_params:
                        raw_params[new_key] = raw_params.pop(old_key)

                for key in avatarrex_smpl_keys:
                    if key in raw_params:
                        param_val = np.array(raw_params[key]).astype(np.float32)
                        single_subj_param_data = param_val
                        if single_subj_param_data.ndim > 1 and single_subj_param_data.shape[0] == 1:
                            single_subj_param_data = single_subj_param_data[0]
                        elif single_subj_param_data.ndim == 0:
                            single_subj_param_data = single_subj_param_data.reshape(1)

                        smpl_params_collector_subject[key][0] = single_subj_param_data
                        subject_has_smpl_data = True

                        if key not in default_smpl_param_shapes:
                            default_smpl_param_shapes[key] = single_subj_param_data.shape
                            print(
                                f"  Updated global default SMPL param shape for {key}: {default_smpl_param_shapes[key]}"
                            )
            except Exception as e:
                print(f"  Error loading SMPL PKL {smplx_pkl_path} for subject {subj_id_str_padded}: {e}")

        if not subject_has_smpl_data:
            print(f"  Warning: SMPL data not loaded for subject {subj_id_str_padded}.")


        thuman2_smplx_obj_path = thuman2_subj_smplx_dir / f"{subj_id_str_padded}.obj"

        avatarrex_smplx_obj_path_subject = avatarrex_smplx_mesh_dir_subject / "smpl_body.obj"
        if thuman2_smplx_obj_path.exists():
            shutil.copy2(thuman2_smplx_obj_path, avatarrex_smplx_obj_path_subject)
        else:
            print(
                f"  Warning: SMPL OBJ not found for subject {subj_id_str_padded} at {thuman2_smplx_obj_path}"
            )

        generated_camera_names = _generate_thuman2_camera_names()
        first_image_dims_for_subject_cameras = None

        for cam_name_thuman2 in generated_camera_names:
            thuman2_img_path = (
                thuman2_subj_render_dir / "render" / f"{cam_name_thuman2}.png"
            )
            thuman2_calib_path = (
                thuman2_subj_render_dir / "calib" / f"{cam_name_thuman2}.txt"
            )

            if thuman2_img_path.exists() and thuman2_calib_path.exists():
                try:
                    img_rgba_pil = Image.open(thuman2_img_path).convert("RGBA")
                    if first_image_dims_for_subject_cameras is None:
                        first_image_dims_for_subject_cameras = img_rgba_pil.size


                    avatarrex_cam_dir_subject = avatarrex_subj_dir / cam_name_thuman2
                    avatarrex_cam_dir_subject.mkdir(parents=True, exist_ok=True)

                    avatarrex_mask_dir_subject = avatarrex_cam_dir_subject / "mask" / "pha"
                    avatarrex_mask_dir_subject.mkdir(parents=True, exist_ok=True)

                    img_rgb_pil = img_rgba_pil.convert("RGB")
                    alpha_mask_pil = img_rgba_pil.split()[-1]


                    output_image_basename = "0000"
                    avatarrex_img_filepath = (
                        avatarrex_cam_dir_subject / f"{output_image_basename}.jpg"
                    )
                    img_rgb_pil.save(avatarrex_img_filepath, "JPEG", quality=95)

                    avatarrex_mask_filepath = (
                        avatarrex_mask_dir_subject / f"{output_image_basename}.jpg"
                    )
                    alpha_mask_pil.save(avatarrex_mask_filepath, "JPEG", quality=95)

                    K_mat, R_mat, T_vec, near, far = _parse_thuman2_calib_file(
                        thuman2_calib_path
                    )

                    if cam_name_thuman2 not in calibration_full_data_subject:
                        calibration_full_data_subject[cam_name_thuman2] = {
                            "K": K_mat.tolist(),
                            "R": R_mat.tolist(),
                            "T": T_vec.tolist(),
                            "imgSize": list(first_image_dims_for_subject_cameras),
                        }

                    pitch_str = cam_name_thuman2.split("_p")[1]
                    yaw_str = cam_name_thuman2.split("_p")[0]
                    thuman2_supplementary_metadata_subject[subj_id_str_padded][
                        cam_name_thuman2
                    ] = {
                        "near": near,
                        "far": far,
                        "yaw_deg": int(yaw_str),
                        "pitch_deg": int(pitch_str),
                    }
                except Exception as e:
                    print(
                        f"  Error processing subject {subj_id_str_padded}, cam {cam_name_thuman2}: {e}"
                    )


        final_smpl_params_for_npz_subject = {}
        if not default_smpl_param_shapes and subject_has_smpl_data:
             print(
                f"  CRITICAL WARNING for subject {subj_id_str_padded}: Default SMPL parameter shapes could not be determined. "
                "NPZ file may be incorrect or incomplete if some keys were missed."
            )

        for key in avatarrex_smpl_keys:
            if smpl_params_collector_subject[key][0] is not None:
                final_smpl_params_for_npz_subject[key] = np.expand_dims(
                    smpl_params_collector_subject[key][0], axis=0
                ).astype(np.float32)
            elif key in default_smpl_param_shapes:
                param_shape_single = default_smpl_param_shapes[key]
                final_smpl_params_for_npz_subject[key] = np.zeros(
                    (1, *param_shape_single), dtype=np.float32
                )
            else:
                print(
                    f"  Warning: Shape for SMPL param '{key}' unknown for subject {subj_id_str_padded} and no data present. "
                    "Cannot create array for NPZ. Skipping this key for this subject."
                )

        if final_smpl_params_for_npz_subject:
            smpl_npz_path_subject = avatarrex_subj_dir / "smpl_params.npz"
            try:
                np.savez(str(smpl_npz_path_subject), **final_smpl_params_for_npz_subject)
                print(f"  Saved smpl_params.npz to {smpl_npz_path_subject}")
            except Exception as e:
                print(f"  Error saving smpl_params.npz for subject {subj_id_str_padded}: {e}")
        else:
            print(f"  No SMPL data was processed to save in smpl_params.npz for subject {subj_id_str_padded}.")


        calib_output_path_subject = avatarrex_subj_dir / "calibration_full.json"
        with open(calib_output_path_subject, "w") as f:
            json.dump(calibration_full_data_subject, f, indent=4)
        print(f"  Saved calibration_full.json to {calib_output_path_subject}")


        metadata_output_path_subject = avatarrex_subj_dir / "thuman2_metadata.json"
        with open(metadata_output_path_subject, "w") as f:
            json.dump(thuman2_supplementary_metadata_subject, f, indent=4)
        print(f"  Saved thuman2_metadata.json to {metadata_output_path_subject}")

        processed_subject_ids_log.append(subj_id_str_padded)

    print(f"\nConversion complete for subjects: {processed_subject_ids_log}")
    print(f"Output AvatarREX-formatted data at: {overall_avatarrex_root_dir}")
    print(
        "Each subject is in its own subdirectory, containing its own "
        "smpl_params.npz, calibration_full.json, thuman2_metadata.json, mesh, and images."
    )
    print(
        "When using with AvatarRexDataset, each subdirectory can be treated as a separate dataset "
        "or you may need to adapt your dataloader to scan these subdirectories."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert THuman2 dataset to AvatarREX format."
    )
    parser.add_argument(
        "--thuman2_dir",
        type=str,
        required=True,
        help="Path to the root of the THuman2 dataset (e.g., data/thuman2_repose).",
    )
    parser.add_argument(
        "--avatarrex_dir",
        type=str,
        required=True,
        help="Path to the output directory for the AvatarREX-formatted dataset.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=str,
        required=True,
        help="List of THuman2 subject IDs to process (e.g., 0 1 402).",
    )

    args = parser.parse_args()

    if not Path(args.thuman2_dir).exists():
        print(f"Error: THuman2 directory not found at {args.thuman2_dir}")
    else:
        convert_dataset(args.thuman2_dir, args.avatarrex_dir, args.subjects)
