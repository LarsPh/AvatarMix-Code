import torch
from loguru import logger


def get_rigid_head_suffix(args) -> str:

    if not getattr(args, 'enable_rigid_head_reposing', False):
        return ""

    return "_rigidhead"


def apply_rigid_head_transformation(
    fwd_skinning_mats_nerf: torch.Tensor,
    head_vertex_indices: torch.Tensor,
    enable_filtering: bool = False,
    outlier_threshold: float = 2.0
) -> torch.Tensor:

    try:

        from ..head_alignment import compute_average_transform, compute_robust_average_transform


        modified_mats = fwd_skinning_mats_nerf.clone()


        head_transforms = modified_mats[head_vertex_indices]

        logger.info(f"Applying rigid head reposing to {len(head_vertex_indices)} head vertices")


        if enable_filtering:
            logger.info(f"Using robust averaging with outlier threshold: {outlier_threshold}")
            try:
                avg_head_transform = compute_robust_average_transform(
                    head_transforms, outlier_threshold=outlier_threshold
                )
            except Exception as e:
                logger.warning(f"Robust averaging failed ({e}), falling back to simple averaging")
                avg_head_transform = compute_average_transform(head_transforms)
        else:
            avg_head_transform = compute_average_transform(head_transforms)


        modified_mats[head_vertex_indices] = avg_head_transform

        logger.info(f"Successfully applied rigid head transformation to {len(head_vertex_indices)} vertices")

        return modified_mats

    except Exception as e:
        logger.error(f"Rigid head reposing failed: {e}")
        logger.warning("Falling back to original LBS transformations")

        return fwd_skinning_mats_nerf
