import torch

# SMPLX joint indices for the 63 pose parameters for the deformation network.
# Order: Head, Neck, LCollar, RCollar, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist,
#        Spine1, Spine2, Spine3, LHip, RHip, LKnee, RKnee, LAnkle, RAnkle, LFoot, RFoot.
DEFORMATION_NET_SMPLX_BODY_JOINT_INDICES = [
    15, 12, 13, 14, 16, 17, 18, 19, 20, 21,
    3, 6, 9,
    1, 2,
    4, 5,
    7, 8,
    10, 11,
]


def get_smplx_pose_63(smplx_params_raw_for_deformnet: dict) -> torch.Tensor:
    """Extract the 63D pose feature vector from a raw SMPL(X) parameter dict.

    Expects key: `body_pose` as (NumBodyJoints, 3) axis-angle tensor/array or a flattened multiple of 3.
    Returns: (63,) float tensor.
    """
    if smplx_params_raw_for_deformnet is None:
        raise ValueError("smplx_params_raw_for_deformnet is None")

    body_pose_full = smplx_params_raw_for_deformnet.get("body_pose")
    if body_pose_full is None:
        raise ValueError("'body_pose' not found in smplx_params_raw_for_deformnet from dataset.")

    if not isinstance(body_pose_full, torch.Tensor):
        body_pose_full = torch.tensor(body_pose_full, dtype=torch.float32)

    if body_pose_full.ndim == 1:
        # Already flattened (e.g. 63 or 66 params). Try to interpret.
        if body_pose_full.shape[0] == (len(DEFORMATION_NET_SMPLX_BODY_JOINT_INDICES) * 3):
            # Ambiguous: assume it is already the 63D feature in the desired joint order.
            return body_pose_full.contiguous()
        if body_pose_full.shape[0] % 3 != 0:
            raise ValueError(f"Cannot interpret flattened body_pose of shape {tuple(body_pose_full.shape)}")
        body_pose_full = body_pose_full.reshape(-1, 3)

    max_req_idx = max(DEFORMATION_NET_SMPLX_BODY_JOINT_INDICES)
    if body_pose_full.shape[0] <= max_req_idx:
        raise ValueError(f"body_pose has {body_pose_full.shape[0]} joints, but require index up to {max_req_idx}")

    selected_joint_rotations = body_pose_full[DEFORMATION_NET_SMPLX_BODY_JOINT_INDICES, :]
    return selected_joint_rotations.reshape(-1).contiguous()

