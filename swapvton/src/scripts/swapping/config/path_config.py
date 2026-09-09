import os
from loguru import logger


def resolve_ply_path_with_rigid_head_support(base_path_template, enable_rigid_head, **format_kwargs):

    base_path = base_path_template.format(**format_kwargs)

    if enable_rigid_head:

        rigid_path = base_path.replace('.ply', '_rigidhead.ply')
        if os.path.exists(rigid_path):
            logger.info(f"Using rigid head PLY file: {os.path.basename(rigid_path)}")
            return rigid_path
        else:
            raise FileNotFoundError(f"Rigid head PLY file not found: {rigid_path}")
    else:

        if os.path.exists(base_path):
            logger.info(f"Using regular PLY file: {os.path.basename(base_path)}")
            return base_path
        else:
            raise FileNotFoundError(f"Regular PLY file not found: {base_path}")
