import json
import subprocess
import sys
import os
import re
import time
from datetime import datetime
from pathlib import Path
from loguru import logger


def validate_cloth_fit_inputs(dataset_type, dataset_root, avatar_subject, garment_subject, remove_neck_from_skin_mask=False):


    required_files = {}


    avatar_base = dataset_root / "gs_on_mesh_repose" / f"{dataset_type}_{avatar_subject}_to_{garment_subject}" / "anim_0000_to_targets_meshes"

    avatar_smpl = avatar_base / "smpl_reposed_frame_0000_cleaned.obj"
    if not avatar_smpl.exists():
        raise FileNotFoundError(f"Avatar SMPL mesh not found: {avatar_smpl}")
    required_files['avatar_mesh_path'] = avatar_smpl

    avatar_lbs = avatar_base / "smpl_skeleton_lbs_weights_cleaned.txt"
    if not avatar_lbs.exists():
        raise FileNotFoundError(f"Avatar LBS weights not found: {avatar_lbs}")
    required_files['avatar_lbs_weights_path'] = avatar_lbs


    avatar_skeleton = None
    avatar_skeleton_path = avatar_base / "smpl_reposed_frame_0000_skeleton.obj"
    if not avatar_skeleton_path.exists():
        raise FileNotFoundError(f"Avatar skeleton not found: {avatar_skeleton_path}")
    required_files['avatar_skeleton_path'] = avatar_skeleton_path


    garment_base = dataset_root / garment_subject.zfill(4) / "mesh" / "processed"


    garment_simp_nonhead = garment_base / "nerf_nonhead_simp_cleaned.obj"
    garment_simp_full = garment_base / "nerf_simp_cleaned.obj"

    use_full_mesh = False
    if garment_simp_full.exists():
        use_full_mesh = True
        logger.info("Detected full-mesh workflow (nerf_simp_cleaned.obj)")
        required_files['garment_mesh_simplified_path'] = garment_simp_full
    elif garment_simp_nonhead.exists():
        logger.info("Detected head-removed workflow (nerf_nonhead_simp_cleaned.obj)")
        required_files['garment_mesh_simplified_path'] = garment_simp_nonhead
    else:
        raise FileNotFoundError(
            f"Simplified garment mesh not found. Checked:\n"
            f"  - {garment_simp_nonhead}\n"
            f"  - {garment_simp_full}\n"
            f"Enable process_mesh.create_simplified_mesh=true in config"
        )

    garment_skeleton = garment_base / "smpl_skeleton.obj"
    if not garment_skeleton.exists():
        raise FileNotFoundError(
            f"Garment skeleton not found: {garment_skeleton}\n"
            f"Enable process_mesh.save_skeleton_mesh=true in config"
        )
    required_files['garment_skeleton_path'] = garment_skeleton


    logger.success(f"Validated all required files for {garment_subject}→{avatar_subject} reshaping")
    return required_files


def generate_cloth_fit_setup(
    output_dir,
    avatar_mesh_path,
    avatar_lbs_weights_path,
    garment_mesh_simplified_path,
    garment_skeleton_path,
    avatar_skeleton_path=None,
    similarity_weight_masks=[],
    avatar_hand_removel_mask_path=None,
    **kwargs
):


    setup = {
        "incremental_steps": kwargs.get('incremental_steps', 2),


        "avatar_mesh_path": str(avatar_mesh_path),
        "avatar_skin_weights_path": str(avatar_lbs_weights_path),
        "garment_mesh_path": str(garment_mesh_simplified_path),


        "target_skeleton_path": "",
        "source_skeleton_path": "",

        "no_fit_spec_path": "dummy.txt",


        "similarity_penalty_weight": kwargs.get('similarity_penalty_weight', 1),

        "curvature_penalty_weight": kwargs.get('curvature_penalty_weight', 0.0),
        "twist_penalty_weight": kwargs.get('twist_penalty_weight', 0.0),
        "curve_center_target_weight": kwargs.get('curve_center_target_weight', 1),
        "fit_weight": kwargs.get('fit_weight', 5),

        "symmetry_weight": kwargs.get('symmetry_weight', 0),
        "curve_size_weight": kwargs.get('curve_size_weight', 0),
        "voxel_size": kwargs.get('voxel_size', 0.01),
        "is_skirt": kwargs.get('is_skirt', False),
        "skeleton_prepass": {
            "enable": True,
            "voxel_size": 0.005,
            "samples_per_bone": 40,
            "inside_margin": 0.1,
            "inside_weight": 50.0,
            "length_weight": 5.0,
            "anchor_weight": 10.0,
            "root_vertex": 0,
            "flood_fill_sign": True,
            "close_holes_voxels": 2,
            "cap_open_boundaries": True,
        },

        "contact": {
            "enabled": kwargs.get('contact_enabled', True),
            "dhat": kwargs.get('contact_dhat', 0.002)
        },


        "solver": {
            "max_threads": kwargs.get('max_threads', 16),
            "similarity_hessian_mode": "optimized_serial",
            "similarity_unique_adjacency": True,
            "linear": {
                "solver": [
                    "Eigen::PardisoLDLT",
                    "Eigen::AccelerateLDLT",
                    "Eigen::SimplicialLDLT"
                ]
            },
            "augmented_lagrangian": {
                "initial_weight": 1,
                "max_weight": 1000000.0,
                "eta": 1,
                "nonlinear": {
                    "grad_norm": 1,
                    "max_iterations": 50
                }
            },
            "nonlinear": {
                "Newton": {
                    "use_psd_projection": True,
                    "use_psd_projection_in_regularized": True,
                    "reg_weight_max": 1e+16,
                    "reg_weight_min": 1,
                    "reg_weight_inc": 10000.0
                },
                "grad_norm": 0.01,
                "line_search": {
                    "max_step_size_limiter": 0.5,
                    "use_grad_norm_tol": 1e-4,
                    "method": "Backtracking",
                    "min_step_size": 1e-08
                },
                "max_iterations": 5000
            },
            "contact": {
                "CCD": {
                    "broad_phase": "BVH",
                    "max_iterations": 200,
                    "tolerance": 1e-3
                },
                "barrier_stiffness": 1e8
            }
        },
    }
    if kwargs.get("enable_fit_weight_mask", kwargs.get("enable_skin_weight_mask", False)):
        logger.info("Enabled fit_weight_masks for cloth-fit:")
        setup["fit_weight_masks"] = kwargs.get('fit_weight_masks', [])
    if kwargs.get('enable_no_fit_mask', False):
        logger.info(f"Enabled no fit mask for cloth-fit:")
        no_fit_spec_path = kwargs['fit_weight_masks'][0]['indices_path']
        setup["no_fit_spec_path"] = no_fit_spec_path
    if kwargs.get('enable_similarity_mask', False):
        logger.info(f"Enabled similarity weight mask for cloth-fit:")
        setup["similarity_weight_masks"] = similarity_weight_masks
    if kwargs.get('enable_avatar_hand_removel_mask', False) and avatar_hand_removel_mask_path:
        logger.info(f"Enabled avatar hand removel weight mask for cloth-fit:")
        setup["avatar_remove_indices_path"] = avatar_hand_removel_mask_path


    source_avatar_mesh_path = kwargs.get("source_avatar_mesh_path", None)
    if source_avatar_mesh_path:
        setup["source_avatar_mesh_path"] = str(source_avatar_mesh_path)

    step3_anneal = kwargs.get("step3_anneal", None)
    if isinstance(step3_anneal, dict):
        setup["step3_anneal"] = step3_anneal

    a2 = kwargs.get("a2", None)
    if isinstance(a2, dict):
        setup["a2"] = a2


    fit_weight_vis = kwargs.get("fit_weight_vis", None)
    if isinstance(fit_weight_vis, dict):
        setup["fit_weight_vis"] = fit_weight_vis


    hem_boundary = kwargs.get("hem_boundary", None)
    if isinstance(hem_boundary, dict):
        setup["hem_boundary"] = hem_boundary


    ring_constraints = kwargs.get("ring_constraints", None)
    if isinstance(ring_constraints, dict):
        setup["ring_constraints"] = ring_constraints


    setup["target_skeleton_path"] = str(avatar_skeleton_path)

    setup["source_skeleton_path"] = str(garment_skeleton_path)


    normalization = kwargs.get("normalization", None)
    if normalization is not None:
        setup["normalization"] = normalization
        logger.info(f"Enabled cloth-fit normalization override: {normalization}")


    setup["output"] = {
        "tensorboard": {
            "enabled": True,
            "log_every": 10,
            "energy_every": 50,
            "detailed_energies": True,
            "max_queue_size": 100000,
            "resume": False,
        },
        "profiling": {
            "chrome_trace": { "enabled": True },
            "timing_summary": { "enabled": True }
        },
        "log": {
            "file_enabled": True,
            "level": "debug"
        },
        "progress": {
            "enabled": True,
            "log_every": 10
        },
        "skip_frame": 200
    }

    setup_path = output_dir / "setup.json"
    with open(setup_path, 'w') as f:
        json.dump(setup, f, indent=4)

    logger.info(f"Generated cloth-fit setup.json with {len(setup)} parameters")
    return setup_path


def execute_polyfem_simulation(setup_json_path, cloth_fit_root, build_type="Release", max_threads=16, max_time=None):


    if sys.platform == "win32":
        polyfem_bin = cloth_fit_root / "build" / build_type / "PolyFEM_bin.exe"
    else:
        polyfem_bin = cloth_fit_root / "build" / "PolyFEM_bin"

    if not polyfem_bin.exists():
        error_msg = f"PolyFEM binary not found: {polyfem_bin}"
        logger.error(error_msg)
        return (False, "", "", error_msg)


    setup_dir = Path(setup_json_path).parent
    original_dir = Path.cwd()

    try:
        os.chdir(setup_dir)

        cmd = [
            str(polyfem_bin.resolve()),
            "-j", "setup.json",
            "--max_threads", str(max_threads)
        ]

        logger.info(f"Executing PolyFEM: {' '.join(cmd)}")
        logger.info(f"Working directory: {setup_dir}")
        if max_time:
            logger.info(f"Timeout: {max_time} minutes")

        start_time = time.time()


        process = subprocess.Popen(
            cmd,


            text=True,
            encoding='utf-8',
            errors='replace',
            shell=False
        )

        timed_out = False


        if max_time:
            while process.poll() is None:
                elapsed_time = time.time() - start_time
                elapsed_minutes = elapsed_time / 60

                if elapsed_minutes > max_time:

                    logger.warning(f"PolyFEM simulation exceeded max_time ({max_time} minutes)")
                    logger.warning(f"Actual runtime: {elapsed_minutes:.2f} minutes - forcing termination")


                    process.terminate()
                    time.sleep(2)


                    if process.poll() is None:
                        logger.warning("Process did not terminate gracefully, forcing kill")
                        process.kill()

                    timed_out = True
                    break


                time.sleep(60)


        stdout, stderr = process.communicate()
        elapsed_time = time.time() - start_time


        if timed_out:

            marker_path = setup_dir / "TIMEOUT_FORCE_STOPPED.txt"
            marker_content = f"""PolyFEM simulation force stopped due to timeout
            Max time allowed: {max_time} minutes
            Actual runtime: {elapsed_time / 60:.2f} minutes
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Status: INCOMPLETE - Simulation terminated before completion

            This file indicates that the PolyFEM simulation was forcefully terminated
            because it exceeded the configured maximum runtime limit.
            """
            marker_path.write_text(marker_content)
            logger.error(f"Timeout marker file created: {marker_path}")

            error_msg = f"Timeout after {max_time} minutes (actual: {elapsed_time / 60:.2f} minutes)"
            return (False, stdout, stderr, error_msg)


        if process.returncode != 0:
            logger.error(f"PolyFEM execution failed with code {process.returncode}")
            logger.error(f"stdout: {stdout}")
            logger.error(f"stderr: {stderr}")
            error_msg = f"PolyFEM failed with return code {process.returncode}"
            return (False, stdout, stderr, error_msg)


        logger.success(f"PolyFEM simulation completed successfully in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.debug(f"PolyFEM output:\n{stdout}")
        return (True, stdout, stderr, "")

    except Exception as e:
        logger.error(f"Unexpected error during PolyFEM execution: {e}")
        return (False, "", str(e), f"Unexpected error: {e}")

    finally:
        os.chdir(original_dir)


def find_deformed_mesh_output(output_dir):


    pattern = re.compile(r'step_garment_(\d+)\.obj')
    max_num = -1
    max_file = None

    for file in Path(output_dir).glob("step_garment_*.obj"):
        match = pattern.match(file.name)
        if match:
            num = int(match.group(1))
            if num == 0:
                continue
            if num > max_num:
                max_num = num
                max_file = file

    if max_file is None:
        raise FileNotFoundError(f"No step_garment_*.obj files found in {output_dir}")

    logger.info(f"Found deformed mesh: {max_file.name} (iteration {max_num})")
    return max_file
