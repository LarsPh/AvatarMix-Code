import torch
import numpy as np
import pickle
from pathlib import Path
from loguru import logger


_MANO_HAND_INDICES_CACHE = None


def load_mano_hand_vertex_indices(mano_data_path=None):

    global _MANO_HAND_INDICES_CACHE


    if _MANO_HAND_INDICES_CACHE is not None:
        return _MANO_HAND_INDICES_CACHE


    if mano_data_path is None:
        mano_data_path = "src/utils/smplx_utils/smplx_models/MANO_SMPLX_vertex_ids.pkl"

    mano_data_path = Path(mano_data_path)

    try:
        logger.debug(f"Loading MANO hand vertex indices from: {mano_data_path}")

        with open(mano_data_path, 'rb') as f:
            mano_data = pickle.load(f)


        left_hand_indices = mano_data['left_hand']
        right_hand_indices = mano_data['right_hand']


        if not isinstance(left_hand_indices, np.ndarray) or not isinstance(right_hand_indices, np.ndarray):
            raise ValueError("MANO data contains non-array hand vertex indices")

        if len(left_hand_indices.shape) != 1 or len(right_hand_indices.shape) != 1:
            raise ValueError("MANO hand vertex index arrays should be 1D")

        if left_hand_indices.dtype != np.int64 or right_hand_indices.dtype != np.int64:
            logger.warning(f"MANO hand indices have dtype {left_hand_indices.dtype}, expected int64")


        _MANO_HAND_INDICES_CACHE = (left_hand_indices, right_hand_indices)

        logger.info(f"Successfully loaded MANO hand vertex indices: {len(left_hand_indices)} left + {len(right_hand_indices)} right hand vertices")
        return _MANO_HAND_INDICES_CACHE

    except (FileNotFoundError, KeyError, ValueError, pickle.PickleError) as e:
        logger.warning(f"Failed to load MANO hand vertex indices from {mano_data_path}: {e}")
        logger.warning("Hand vertex override will be disabled - falling back to standard max-weight assignment")


        _MANO_HAND_INDICES_CACHE = (None, None)
        return _MANO_HAND_INDICES_CACHE


def calculate_wrist_joint_indices(skeleton_joint_indices):


    LEFT_WRIST_ORIGINAL = 20
    RIGHT_WRIST_ORIGINAL = 21

    try:

        left_wrist_filtered_idx = skeleton_joint_indices.index(LEFT_WRIST_ORIGINAL)
        right_wrist_filtered_idx = skeleton_joint_indices.index(RIGHT_WRIST_ORIGINAL)

        logger.debug(f"Wrist joint mapping: left_wrist({LEFT_WRIST_ORIGINAL}) → filtered_idx {left_wrist_filtered_idx}, "
                    f"right_wrist({RIGHT_WRIST_ORIGINAL}) → filtered_idx {right_wrist_filtered_idx}")

        return left_wrist_filtered_idx, right_wrist_filtered_idx

    except ValueError as e:

        available_joints = ", ".join(map(str, skeleton_joint_indices))
        raise ValueError(f"Wrist joints not found in skeleton configuration. "
                        f"Expected joints {LEFT_WRIST_ORIGINAL} and {RIGHT_WRIST_ORIGINAL}, "
                        f"but skeleton only contains: [{available_joints}]") from e


def apply_hand_vertex_override(filtered_weights, left_hand_indices, right_hand_indices,
                              left_wrist_idx, right_wrist_idx):


    n_vertices, n_joints = filtered_weights.shape

    if left_wrist_idx >= n_joints or right_wrist_idx >= n_joints:
        raise ValueError(f"Invalid wrist indices: left={left_wrist_idx}, right={right_wrist_idx} "
                        f"for skeleton with {n_joints} joints")

    if len(left_hand_indices) > 0 and np.max(left_hand_indices) >= n_vertices:
        raise IndexError(f"Left hand vertex indices out of bounds: max={np.max(left_hand_indices)} "
                        f"for mesh with {n_vertices} vertices")

    if len(right_hand_indices) > 0 and np.max(right_hand_indices) >= n_vertices:
        raise IndexError(f"Right hand vertex indices out of bounds: max={np.max(right_hand_indices)} "
                        f"for mesh with {n_vertices} vertices")


    override_weights = filtered_weights.copy()


    if len(left_hand_indices) > 0:

        override_weights[left_hand_indices, :] = 0.0

        override_weights[left_hand_indices, left_wrist_idx] = 1.0

        logger.debug(f"Applied left hand override: {len(left_hand_indices)} vertices → wrist joint {left_wrist_idx}")


    if len(right_hand_indices) > 0:

        override_weights[right_hand_indices, :] = 0.0

        override_weights[right_hand_indices, right_wrist_idx] = 1.0

        logger.debug(f"Applied right hand override: {len(right_hand_indices)} vertices → wrist joint {right_wrist_idx}")

    total_overridden = len(left_hand_indices) + len(right_hand_indices)
    logger.info(f"Hand vertex override complete: {total_overridden} vertices manually assigned to wrist joints")

    return override_weights


def validate_hand_assignments(hard_assignment, left_hand_indices, right_hand_indices,
                             left_wrist_idx, right_wrist_idx):

    validation_passed = True


    if len(left_hand_indices) > 0:
        left_hand_weights = hard_assignment[left_hand_indices, :]


        left_wrist_correct = np.allclose(left_hand_weights[:, left_wrist_idx], 1.0, atol=1e-8)
        if not left_wrist_correct:
            logger.error(f"Left hand validation failed: not all vertices assigned to left wrist (joint {left_wrist_idx})")
            validation_passed = False


        left_other_weights = np.concatenate([
            left_hand_weights[:, :left_wrist_idx],
            left_hand_weights[:, left_wrist_idx+1:]
        ], axis=1)
        left_others_zero = np.allclose(left_other_weights, 0.0, atol=1e-8)
        if not left_others_zero:
            logger.error(f"Left hand validation failed: some vertices have non-zero weights at non-wrist joints")
            validation_passed = False

        if left_wrist_correct and left_others_zero:
            logger.debug(f"Left hand validation passed: {len(left_hand_indices)} vertices correctly assigned to joint {left_wrist_idx}")


    if len(right_hand_indices) > 0:
        right_hand_weights = hard_assignment[right_hand_indices, :]


        right_wrist_correct = np.allclose(right_hand_weights[:, right_wrist_idx], 1.0, atol=1e-8)
        if not right_wrist_correct:
            logger.error(f"Right hand validation failed: not all vertices assigned to right wrist (joint {right_wrist_idx})")
            validation_passed = False


        right_other_weights = np.concatenate([
            right_hand_weights[:, :right_wrist_idx],
            right_hand_weights[:, right_wrist_idx+1:]
        ], axis=1)
        right_others_zero = np.allclose(right_other_weights, 0.0, atol=1e-8)
        if not right_others_zero:
            logger.error(f"Right hand validation failed: some vertices have non-zero weights at non-wrist joints")
            validation_passed = False

        if right_wrist_correct and right_others_zero:
            logger.debug(f"Right hand validation passed: {len(right_hand_indices)} vertices correctly assigned to joint {right_wrist_idx}")

    if validation_passed:
        total_hand_vertices = len(left_hand_indices) + len(right_hand_indices)
        logger.info(f"Hand vertex validation passed: {total_hand_vertices} hand vertices correctly assigned to wrist joints")

    return validation_passed


def get_skeleton_joint_indices(include_fingers=False, exclude_foot_joints=True, simplified=True):


    main_body_indices = list(range(22))


    if exclude_foot_joints:

        main_body_indices = [i for i in main_body_indices if i not in [10, 11]]


    if simplified:

        simplification_removals = [3, 6, 13, 14]
        main_body_indices = [i for i in main_body_indices if i not in simplification_removals]

    if include_fingers:

        finger_indices = list(range(25, 55))
        return main_body_indices + finger_indices
    else:
        return main_body_indices


def find_valid_ancestor(joint_idx, parents, idx_mapping):

    current = joint_idx
    while current != -1:
        current = parents[current].item()
        if current in idx_mapping:
            return idx_mapping[current]
    return -1


def filter_joints_and_parents(joint_positions, parents, keep_indices):


    idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}


    filtered_joints = joint_positions[keep_indices]


    filtered_parents = []
    for old_idx in keep_indices:
        old_parent = parents[old_idx].item()
        if old_parent == -1:
            filtered_parents.append(-1)
        elif old_parent in idx_mapping:
            filtered_parents.append(idx_mapping[old_parent])
        else:
            valid_ancestor = find_valid_ancestor(old_parent, parents, idx_mapping)
            filtered_parents.append(valid_ancestor)

    return filtered_joints, filtered_parents


def validate_skeleton_connectivity(filtered_parents):


    root_count = sum(1 for p in filtered_parents if p == -1)
    if root_count != 1:
        raise ValueError(f"Invalid skeleton: {root_count} root joints (expected 1)")


    visited = set()
    for i in range(len(filtered_parents)):
        current = i
        path = set()
        while current != -1 and current not in visited:
            if current in path:
                raise ValueError(f"Cycle detected in skeleton at joint {current}")
            path.add(current)
            if current < len(filtered_parents):
                current = filtered_parents[current]
            else:
                break
        visited.update(path)


def save_skeleton_obj(file_path, joint_positions, skeleton_lines, include_fingers=False, exclude_foot_joints=True, simplified=True):

    joint_count = len(joint_positions)
    line_count = len(skeleton_lines)

    with open(file_path, 'w') as f:

        for joint in joint_positions:
            if torch.is_tensor(joint):
                joint = joint.detach().cpu().numpy()
            f.write(f"v {joint[0]:.6f} {joint[1]:.6f} {joint[2]:.6f}\n")


        for line in skeleton_lines:
            f.write(f"l {line[0]+1} {line[1]+1}\n")


def extract_skeleton_mesh(
    smpl_model,
    smpl_params_batch,
    output_dir,
    include_fingers=False,
    exclude_foot_joints=True,
    simplified=True,
    device="cpu",
    dataset_type: str = "thuman2",
):

    joint_count_desc = "16-joint simplified" if simplified else "20-joint full"
    logger.info(f"Extracting SMPL skeleton ({joint_count_desc}, fingers: {include_fingers}, exclude_foot: {exclude_foot_joints})...")

    try:

        from .process_meshes import build_smplx_mesh


        joint_positions = build_smplx_mesh(
            smpl_model=smpl_model,
            smpl_params_batch=smpl_params_batch,
            device=device,
            return_vertices=False,
            return_full_joints=True,
            dataset_type=dataset_type,
        )

        logger.info(f"Extracted {joint_positions.shape[0]} joints from SMPL model")


        parents = smpl_model.parents


        keep_indices = get_skeleton_joint_indices(include_fingers, exclude_foot_joints, simplified)
        filtered_joints, filtered_parents = filter_joints_and_parents(
            joint_positions, parents, keep_indices)

        logger.info(f"Filtered to {len(filtered_joints)} skeleton joints")


        validate_skeleton_connectivity(filtered_parents)


        skeleton_lines = []
        for i, parent_idx in enumerate(filtered_parents):
            if parent_idx != -1:
                skeleton_lines.append([i, parent_idx])

        logger.info(f"Created {len(skeleton_lines)} skeleton connections")


        skeleton_path = Path(output_dir) / "smpl_skeleton.obj"
        save_skeleton_obj(skeleton_path, filtered_joints, skeleton_lines, include_fingers, exclude_foot_joints, simplified)

        logger.success(f"Skeleton mesh saved to: {skeleton_path}")
        return skeleton_path

    except Exception as e:
        logger.error(f"Skeleton extraction failed: {e}")
        raise RuntimeError(f"Failed to extract skeleton mesh: {e}") from e


def log_skeleton_statistics(joint_positions, skeleton_lines, include_fingers=False):

    logger.info("=== SMPL Skeleton Mesh Statistics ===")
    logger.info(f"Total joints: {len(joint_positions)}")
    logger.info(f"Total connections: {len(skeleton_lines)}")
    logger.info(f"Fingers included: {include_fingers}")

    if torch.is_tensor(joint_positions):
        joint_positions = joint_positions.detach().cpu().numpy()


    min_coords = np.min(joint_positions, axis=0)
    max_coords = np.max(joint_positions, axis=0)
    logger.info(f"Bounding box: min({min_coords[0]:.3f}, {min_coords[1]:.3f}, {min_coords[2]:.3f}), "
                f"max({max_coords[0]:.3f}, {max_coords[1]:.3f}, {max_coords[2]:.3f})")

    logger.success("Skeleton mesh statistics logged successfully")


def convert_to_max_weight_assignment(filtered_weights, skeleton_joint_indices=None,
                                   enable_hand_override=True, mano_data_path=None):

    logger.info(f"Converting to max-weight assignment for {filtered_weights.shape[0]} vertices and {filtered_weights.shape[1]} joints...")
    if enable_hand_override:
        logger.info("Hand vertex override enabled - will assign hand vertices to wrist joints")
    else:
        logger.info("Hand vertex override disabled - using standard max-weight assignment only")


    hand_override_applied = False
    left_hand_indices = right_hand_indices = None
    left_wrist_idx = right_wrist_idx = None

    if enable_hand_override and skeleton_joint_indices is not None:
        try:

            left_hand_indices, right_hand_indices = load_mano_hand_vertex_indices(mano_data_path)

            if left_hand_indices is not None and right_hand_indices is not None:

                left_wrist_idx, right_wrist_idx = calculate_wrist_joint_indices(skeleton_joint_indices)


                filtered_weights = apply_hand_vertex_override(
                    filtered_weights, left_hand_indices, right_hand_indices,
                    left_wrist_idx, right_wrist_idx
                )

                hand_override_applied = True
                logger.info(f"Hand vertex override applied: {len(left_hand_indices) + len(right_hand_indices)} vertices pre-assigned to wrist joints")
            else:
                logger.info("Hand vertex override skipped - MANO data not available")

        except (ValueError, IndexError) as e:
            logger.error(f"Hand vertex override failed: {e}")
            logger.warning("Falling back to standard max-weight assignment")
    elif enable_hand_override and skeleton_joint_indices is None:
        logger.warning("Hand vertex override requested but skeleton_joint_indices not provided - skipping override")


    row_sums = np.sum(filtered_weights, axis=1)
    zero_weight_vertices = row_sums <= 1e-8
    num_zero_weight = np.sum(zero_weight_vertices)

    if num_zero_weight > 0:
        logger.warning(f"Found {num_zero_weight} vertices with all zero weights after joint simplification!")
        logger.warning(f"These vertices were likely influenced only by filtered-out joints.")
        logger.warning(f"Max-weight assignment will assign them to joint 0 (may cause artifacts)")


    max_joint_indices = np.argmax(filtered_weights, axis=1)


    hard_assignment = np.zeros_like(filtered_weights)


    vertex_indices = np.arange(filtered_weights.shape[0])
    hard_assignment[vertex_indices, max_joint_indices] = 1.0


    num_valid = filtered_weights.shape[0] - num_zero_weight
    logger.info(f"Max-weight assignment completed: {num_valid} vertices with valid weights, {num_zero_weight} with zero weights")


    unique_joints, joint_counts = np.unique(max_joint_indices, return_counts=True)
    logger.debug(f"Joint assignment distribution: {len(unique_joints)} joints used, "
                f"avg {np.mean(joint_counts):.1f} vertices per joint")


    final_row_sums = np.sum(hard_assignment, axis=1)
    expected_sums = np.ones_like(final_row_sums)
    expected_sums[zero_weight_vertices] = 0.0
    final_row_sums[zero_weight_vertices] = 0.0

    if not np.allclose(final_row_sums, expected_sums, atol=1e-8):
        logger.error(f"Standard assignment validation failed: expected sums don't match actual sums")
        raise RuntimeError("Max-weight assignment failed standard validation")

    logger.info(f"Standard validation passed: all vertices have exactly one dominant joint assignment")


    if hand_override_applied and left_hand_indices is not None and right_hand_indices is not None:
        hand_validation_passed = validate_hand_assignments(
            hard_assignment, left_hand_indices, right_hand_indices,
            left_wrist_idx, right_wrist_idx
        )

        if not hand_validation_passed:
            logger.error("Hand vertex assignment validation failed")
            raise RuntimeError("Max-weight assignment failed hand vertex validation")

    logger.success("Max-weight assignment completed with validation passed")
    return hard_assignment


def save_lbs_weights_txt(weights_array, output_path):

    logger.info(f"Saving LBS weights to: {output_path}")
    logger.info(f"Array shape: {weights_array.shape} ({weights_array.shape[0]} joints, {weights_array.shape[1]} vertices)")

    try:
        with open(output_path, 'w') as f:
            for joint_idx, joint_weights in enumerate(weights_array):

                formatted_weights = [f'{weight:.5e}' for weight in joint_weights]
                f.write(' '.join(formatted_weights) + '\n')


                if joint_idx == 0 or (joint_idx + 1) % 5 == 0:
                    logger.debug(f"Written {joint_idx + 1}/{weights_array.shape[0]} joint weight rows")

        logger.success(f"LBS weights saved successfully: {weights_array.shape[0]} joints × {weights_array.shape[1]} vertices")

    except Exception as e:
        logger.error(f"Failed to save LBS weights to {output_path}: {e}")
        raise IOError(f"Cannot write LBS weights file: {e}") from e


def export_skeleton_lbs_weights(lbs_weights_npy_path, output_dir,
                               include_fingers=False, exclude_foot_joints=True, simplified=True):

    logger.info("=== SMPL Skeleton LBS Weights Export ===")


    joint_count_desc = "16-joint simplified" if simplified else "20-joint full"
    logger.info(f"Configuration: {joint_count_desc}, fingers: {include_fingers}, exclude_foot: {exclude_foot_joints}")


    lbs_weights_path = Path(lbs_weights_npy_path)
    if not lbs_weights_path.exists():
        raise FileNotFoundError(f"SMPL LBS weights file not found: {lbs_weights_path}")

    try:
        lbs_weights = np.load(str(lbs_weights_path))
        logger.info(f"Loaded LBS weights from: {lbs_weights_path}")
        logger.info(f"Original shape: {lbs_weights.shape} (vertices × joints)")

    except Exception as e:
        raise RuntimeError(f"Failed to load LBS weights from {lbs_weights_path}: {e}") from e


    if len(lbs_weights.shape) != 2:
        raise ValueError(f"Expected 2D LBS weights array, got shape: {lbs_weights.shape}")

    n_vertices, n_joints_full = lbs_weights.shape
    if n_joints_full < 55:
        raise ValueError(f"Expected at least 165 joints in LBS weights, got {n_joints_full}")


    keep_indices = get_skeleton_joint_indices(include_fingers, exclude_foot_joints, simplified)
    logger.info(f"Filtering {n_joints_full} joints down to {len(keep_indices)} skeleton joints")
    logger.debug(f"Keeping joint indices: {keep_indices}")


    max_joint_idx = max(keep_indices)
    if max_joint_idx >= n_joints_full:
        raise ValueError(f"Joint index {max_joint_idx} exceeds available joints ({n_joints_full})")


    filtered_weights = lbs_weights[:, keep_indices]
    logger.info(f"Filtered weights shape: {filtered_weights.shape}")


    hard_assigned_weights = convert_to_max_weight_assignment(
        filtered_weights,
        skeleton_joint_indices=keep_indices,
        enable_hand_override=True
    )


    export_weights = hard_assigned_weights.T
    logger.info(f"Export weights shape: {export_weights.shape} (joints × vertices)")


    output_path = Path(output_dir) / "smpl_skeleton_lbs_weights.txt"
    save_lbs_weights_txt(export_weights, output_path)


    logger.info("=== Export Statistics ===")
    logger.info(f"Skeleton joints: {export_weights.shape[0]}")
    logger.info(f"Vertices: {export_weights.shape[1]}")
    logger.info(f"Total weight values: {export_weights.size}")


    sample_vertex_sums = np.sum(hard_assigned_weights[:5, :], axis=1)
    logger.debug(f"Sample vertex weight sums: {sample_vertex_sums} (should all be 1.0 for valid vertices, 0.0 for zero-weight vertices)")

    logger.success(f"LBS weights export completed: {output_path}")
    return output_path, export_weights
