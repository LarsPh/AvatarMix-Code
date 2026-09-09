from typing import Optional, Tuple, List, Union
from pathlib import Path
from loguru import logger
import glob
from pipeline.sampling.subject_discovery import first_swap_reshape_enabled


def _subject_id_padded_for_paths(config: dict, subject: str) -> str:

    data_type = config.get("data_type", "")
    s = str(subject).strip()
    if data_type == "thuman2":
        return f"{int(s):04d}"
    if data_type in {"mvhumannet", "talkbody4d", "actorshq"}:
        return s

    if s.isdigit():
        return f"{int(s):04d}"
    return s


def _pick_any_camera_dir(subject_root: Path) -> Optional[str]:

    if not subject_root.exists():
        return None
    cams = sorted([p.name for p in subject_root.iterdir() if p.is_dir()])

    for c in cams:
        if c.isdigit() or c.startswith("CC") or "_p" in c:
            return c
    return cams[0] if cams else None


class OutputValidator:


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:

        raise NotImplementedError("Validator not implemented")


class RenderOutputValidator(OutputValidator):


    REQUIRED_SUBDIRS = ['calib', 'depth_F', 'render']

    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:

        output_dir = config['paths']['thuman2_render_output']

        for subject in subjects:
            subject_dir = Path(output_dir) / f"{int(str(subject).strip()):04d}"


            if not subject_dir.exists():
                return False, f"Subject directory missing: {subject}"


            missing_subdirs = [
                subdir for subdir in self.REQUIRED_SUBDIRS
                if not (subject_dir / subdir).exists()
            ]

            if missing_subdirs:
                return False, f"Missing subdirectories for subject {subject}: {missing_subdirs}"


            file_counts = {}
            for subdir in self.REQUIRED_SUBDIRS:
                subdir_path = subject_dir / subdir
                try:
                    file_count = len([f for f in subdir_path.iterdir() if f.is_file()])
                    file_counts[subdir] = file_count
                except Exception as e:
                    return False, f"Failed to read directory {subdir} for subject {subject}: {e}"


            unique_counts = set(file_counts.values())
            if len(unique_counts) != 1:
                return False, f"Mismatched file counts for subject {subject}: {file_counts}"


            if list(unique_counts)[0] == 0:
                return False, f"Empty output directories for subject {subject}"


        total_files = list(file_counts.values())[0] if file_counts else 0
        return True, f"Valid output for all {len(subjects)} subjects ({total_files} files each)"

class ConvertToAvatarrexOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:
            subdir = _subject_id_padded_for_paths(config, subject)
            subject_root = Path(config['paths']['avatarrex_output']) / subdir
            cam = _pick_any_camera_dir(subject_root)
            if cam is None:
                return False, f"No camera directory found for subject {subject} at {subject_root}"
            img_path = subject_root / cam / "0000.jpg"
            if not img_path.exists():
                return False, f"Image file missing for subject {subject}, expected: {img_path}"
        return True, f"Valid output for all {len(subjects)} subjects"

class ConvertToNeus2OutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:
            subdir = _subject_id_padded_for_paths(config, subject)
            avatarrex_root = Path(config['paths']['avatarrex_output']) / subdir
            cam = _pick_any_camera_dir(avatarrex_root)
            if cam is None:
                return False, f"No camera directory found for subject {subject} at {avatarrex_root}"

            img_path = Path(config['paths']['neus2_output']) / subdir / "images" / "0000" / f"{cam}_0000.png"
            if not img_path.exists():
                return False, f"Image file missing for subject {subject}, expected: {img_path}"
        return True, f"Valid output for all {len(subjects)} subjects"

class Neus2TrainOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:

        for subject in subjects:
            subject_id_str = _subject_id_padded_for_paths(config, subject)
            subdir = Path(config['pipeline_stages']['4_neus2_train']['output_dir_template'].format(subject_id_padded=subject_id_str))
            subject_dir = Path(config['paths']['neus2_project']) / 'output' / subdir / 'transforms_0000' / 'evaluation' / 'mesh' / 'scene_transforms_0000_frame_0000.obj'
            if not subject_dir.exists():
                return False, f"Mesh file missing for subject {subject}"
        return True, f"Valid output for all {len(subjects)} subjects"

class CopyMeshOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:
            subdir = _subject_id_padded_for_paths(config, subject)
            subject_dir = Path(config['paths']['avatarrex_output']) / subdir / 'mesh' / 'neus2_raw' / '0000.obj'
            if not subject_dir.exists():
                return False, f"Mesh file missing for subject {subject}"
        return True, f"Valid output for all {len(subjects)} subjects"

class CleanMeshOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:
            subdir = _subject_id_padded_for_paths(config, subject)
            subject_dir = Path(config['paths']['avatarrex_output']) / subdir / 'mesh' / 'trimesh_cleaned' / '0000.obj'
            if not subject_dir.exists():
                return False, f"Mesh file missing for subject {subject}"
        return True, f"Valid output for all {len(subjects)} subjects"

class ParseMeshOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:

        sam_label_gain_scale = config['pipeline_stages']['7_parse_mesh'].get('sam_label_gain_scale', 1.0)
        disable_sam_votes = config['pipeline_stages']['7_parse_mesh'].get('disable_sam_votes', False)
        suffix = f"_sam_scale{sam_label_gain_scale}" if sam_label_gain_scale != 1.0 else ""
        suffix += "_no_sam" if disable_sam_votes else ""
        filename = f'label-f0000{suffix}.pkl'
        for subject in subjects:
            subdir = _subject_id_padded_for_paths(config, subject)
            subject_dir = Path(config['paths']['avatarrex_output']) / subdir / 'Semantic' / 'process' / 'labels_auto_extended' / filename
            if not subject_dir.exists():
                return False, f"Mesh file missing for subject {subject}, expected: {subject_dir}"
        return True, f"Valid output for all {len(subjects)} subjects"

class ProcessMeshOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:

            subdir = _subject_id_padded_for_paths(config, subject)
            subject_dir = Path(config['paths']['avatarrex_output']) / subdir / 'mesh' / 'labeled' / f'label-f0000_extended.pkl'
            if not subject_dir.exists():
                return False, f"Point cloud file missing for subject {subject}"
        return True, f"Valid output for all {len(subjects)} subjects"

class SplattingAvatarOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        cfg = config.get('pipeline_stages', {}).get('10_splatting_avatar', {})
        avatarrex_root = Path(config['paths']['avatarrex_output'])
        for i, subject in enumerate(subjects):
            subdir = _subject_id_padded_for_paths(config, subject)
            model_path = cfg.get('model_path_template', 'neusclean_sub{subject_id_padded}').format(subject_id_padded=subdir)


            pc_glob = avatarrex_root / 'output-splatting' / model_path / 'point_cloud' / 'iteration_*' / 'point_cloud.ply'
            matches = glob.glob(str(pc_glob))
            if len(matches) == 0:
                return False, f"Point cloud file missing for subject {subject}, expected like: {pc_glob}"
        return True, f"Valid output for all {len(subjects)} subjects"

class LBSTransferOutputValidator():


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:

            subdir = _subject_id_padded_for_paths(config, subject)
            subject_dir = Path(config['paths']['avatarrex_output']) / subdir / 'mesh' / 'processed' / 'smoothed_inpainted_weights.npy'
            if not subject_dir.exists():
                return False, f"Point cloud file missing for subject {subject}"
        return True, f"Valid output for all {len(subjects)} subjects"

class HeadDonatorReposingOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:
            subject_other = subjects[0] if subject == subjects[1] else subjects[1]

            pose_subdir = f"thuman2_{int(subject):04d}_to_{int(subject_other):04d}"
            file_name = 'reposed_gs_targetframe0000_rigidhead.ply' if config.get('enable_rigid_head_reposing', False) else 'reposed_gs_targetframe0000.ply'
            subject_dir = Path(config['paths']['avatarrex_output']) / 'gs_on_mesh_repose' / pose_subdir / 'reposed_gaussians_ply' / file_name
            matches = glob.glob(str(subject_dir))
            if len(matches) == 0:
                return False, f"Reposed gaussians ply file missing for subject {subject}\n Expected: {subject_dir}"
        return True, f"Valid output for all {len(subjects)} subjects"

class BodyDonatorReshapingOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:
            subject_other = subjects[0] if subject == subjects[1] else subjects[1]
            if not first_swap_reshape_enabled(config):

                continue
            pose_subdir = f"thuman2_{int(subject):04d}_to_{int(subject_other):04d}_reshaped*"
            subject_dir = Path(config['paths']['avatarrex_output']) / 'gs_on_mesh_repose' / pose_subdir / 'reshaped_gaussians_ply_0000' / 'reshaped_gs_target_shape_0000*.ply'
            matches = glob.glob(str(subject_dir))
            if len(matches) == 0:
                return False, f"Reshaped gaussians ply file missing for subject {subject}"
        return True, f"Valid output for all {len(subjects)} subjects"

class DirectSwappingOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:
            subject_other = subjects[0] if subject == subjects[1] else subjects[1]
            swap_subdir = f"A{int(subject):04d}B{int(subject_other):04d}"
            file_name = f"swapped_{int(subject):04d}head_on_{int(subject_other):04d}body_from_point_cloud_in_A_world_with_color_transfer_*_direct.ply"
            if first_swap_reshape_enabled(config):
                swap_subdir += "_reshaped*"
                file_name = f"swapped_{int(subject):04d}head_on_{int(subject_other):04d}body_from_reshaped_gs_target_shape_*_in_A_world_with_color_transfer_*_direct.ply"
            subject_dir = Path(config['paths']['avatarrex_output']) / 'swapped' / swap_subdir / file_name
            matches = glob.glob(str(subject_dir))
            if len(matches) == 0:
                return False, f"Swapped ply file missing for subject {subject}"
        return True, f"Valid output for all {len(subjects)} subjects"

class RenderSwappedGTHeadAlignedOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:
            subject_other = subjects[0] if subject == subjects[1] else subjects[1]

            subject_dir = Path(config['paths']['avatarrex_output']) / 'head_swapped_renders' / f"swapped_{int(subject):04d}head_on_{int(subject_other):04d}body_from_*_with_color_transfer_*_direct" / "355_p-20" / 'head_aligned' / '0000.png'
            matches = glob.glob(str(subject_dir))
            if len(matches) == 0:
                return False, f"Head aligned image file missing for subject {subject}, expected: {subject_dir}"
        return True, f"Valid output for all {len(subjects)} subjects"


class ExtractHeadMasksOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:

            subject_dir = Path(config['paths']['avatarrex_output']) / f"{int(subject):04d}" / '355_p-20' / 'mask' / 'head' / '0000.png'
            if not subject_dir.exists():
                return False, f"Head mask file missing for subject {subject}, expected: {subject_dir}"

        return True, f"Valid output for all {len(subjects)} subjects"

class Neus2TrainSwappedOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:
            other_subject = subjects[0] if subject == subjects[1] else subjects[1]
            swapped_subject_name = f"swapped_{int(subject):04d}head_on_{int(other_subject):04d}body_from_*_with_color_transfer_*_direct"

            subject_dir = Path(config['paths']['neus2_project']) / "output" / f"{swapped_subject_name}_60k_ek0.02" / swapped_subject_name / "transforms_0000" / "evaluation" / "mesh" / "scene_transforms_0000_frame_0000.obj"
            matches = glob.glob(str(subject_dir))
            if len(matches) == 0:
                return False, f"Mesh file missing for subject {subject}, expected: {subject_dir}"
        return True, f"Valid output for all {len(subjects)} subjects"

class SplattingAvatarSwappedOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:
            other_subject = subjects[0] if subject == subjects[1] else subjects[1]
            swapped_subject_name = f"swapped_{int(subject):04d}head_on_{int(other_subject):04d}body_from_*_with_color_transfer_*_direct"
            iteration = config['pipeline_stages']['10_splatting_avatar']['iteration_default']

            subject_dir = Path(config['paths']['avatarrex_output']) / 'head_swapped_renders' / 'output-splatting' / 'swapped' / swapped_subject_name / 'point_cloud' / f"iteration_{iteration}" / 'point_cloud.ply'
            matches = glob.glob(str(subject_dir))
            if len(matches) == 0:
                return False, f"Point cloud file missing for subject {subject}, expected: {subject_dir}"
        return True, f"Valid output for all {len(subjects)} subjects"

class ParseMeshSwappedOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        sam_label_gain_scale = config['pipeline_stages']['19_parse_mesh_swapped'].get('sam_label_gain_scale', 1.0)
        disable_sam_votes = config['pipeline_stages']['19_parse_mesh_swapped'].get('disable_sam_votes', False)
        suffix = f"_sam_scale{sam_label_gain_scale}" if sam_label_gain_scale != 1.0 else ""
        suffix += "_no_sam" if disable_sam_votes else ""
        filename = f'label-f0000{suffix}.pkl'
        for subject in subjects:
            other_subject = subjects[0] if subject == subjects[1] else subjects[1]
            swapped_subject_name = f"swapped_{int(subject):04d}head_on_{int(other_subject):04d}body_from_*_with_color_transfer_*_direct"
            subject_dir = Path(config['paths']['avatarrex_output']) / 'head_swapped_renders' / swapped_subject_name / 'Semantic' / 'process' / 'labels_auto_extended' / filename
            matches = glob.glob(str(subject_dir))
            if len(matches) == 0:
                return False, f"Mesh file missing for subject {subject}, expected: {subject_dir}"
        return True, f"Valid output for all {len(subjects)} subjects"

class ClothFitReshapingOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:

        for subject in subjects:
            other_subject = subjects[0] if subject == subjects[1] else subjects[1]
            subject_dir = Path(config['paths']['avatarrex_output']) / 'cloth_fit_output' / f"{int(subject):04d}_avatar_{int(other_subject):04d}_garment" / 'step_garment_*.obj'
            matches = glob.glob(str(subject_dir))
            if len(matches) == 0:
                return False, f"Cloth fit reshaping file missing for subject {subject}, expected: {subject_dir}"
        return True, f"Valid output for all {len(subjects)} subjects"

class RenderSwappedBackGaussiansOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:

        for subject in subjects:
            other_subject = subjects[0] if subject == subjects[1] else subjects[1]
            subject_dir = Path(config['paths']['avatarrex_output']) / 'head_swapped_back_renders' / f"restored_{int(subject):04d}_from_swapped_{int(subject):04d}head_on_{int(other_subject):04d}body_*_swap_back" / "355_p-20" / "0000.jpg"
            matches = glob.glob(str(subject_dir))
            if len(matches) == 0:
                return False, f"Swapped back gaussians image file missing for subject {subject}, expected: {subject_dir}"
            return True, f"Valid output for all {len(subjects)} subjects"

class CreateCombinedPortraitsOutputValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        for subject in subjects:
            other_subject = subjects[0] if subject == subjects[1] else subjects[1]
            subject_dir = Path(config['paths']['avatarrex_output']) / 'head_swapped_back_renders' / f"restored_{int(subject):04d}_from_swapped_{int(subject):04d}head_on_{int(other_subject):04d}body_*_swap_back" / "355_p-20" / "combined" / "combined.jpg"
            matches = glob.glob(str(subject_dir))
            if len(matches) == 0:
                return False, f"Combined portrait image file missing for subject {subject}, expected: {subject_dir}"
            return True, f"Valid output for all {len(subjects)} subjects"

class NotImplementedValidator(OutputValidator):


    def validate(self, config: dict, subjects: List[str]) -> Tuple[bool, str]:
        raise NotImplementedError("Output validation not implemented for this stage")


OUTPUT_VALIDATORS = {
    'render': RenderOutputValidator(),
    'convert_to_avatarrex': ConvertToAvatarrexOutputValidator(),
    'convert_to_neus2': ConvertToNeus2OutputValidator(),
    'neus2_train': Neus2TrainOutputValidator(),
    'copy_mesh': CopyMeshOutputValidator(),
    'clean_mesh': CleanMeshOutputValidator(),
    'parse_mesh': ParseMeshOutputValidator(),
    'process_mesh': ProcessMeshOutputValidator(),
    'lbs_transfer': LBSTransferOutputValidator(),
    'splatting_avatar': SplattingAvatarOutputValidator(),
    'head_donator_reposing': HeadDonatorReposingOutputValidator(),
    'body_donator_reshaping': BodyDonatorReshapingOutputValidator(),
    'direct_swapping': DirectSwappingOutputValidator(),
    'render_swapped_gaussians': NotImplementedValidator(),
    'render_swapped_gt_head_aligned': RenderSwappedGTHeadAlignedOutputValidator(),
    'refine_rendered_images': NotImplementedValidator(),
    'extract_head_masks': ExtractHeadMasksOutputValidator(),
    'convert_swapped_to_neus2': NotImplementedValidator(),
    'neus2_train_swapped': Neus2TrainSwappedOutputValidator(),
    'copy_mesh_swapped': NotImplementedValidator(),
    'clean_mesh_swapped': NotImplementedValidator(),
    'parse_mesh_swapped': ParseMeshSwappedOutputValidator(),
    'process_mesh_swapped': NotImplementedValidator(),
    'lbs_transfer_swapped': NotImplementedValidator(),
    'splatting_avatar_swapped': SplattingAvatarSwappedOutputValidator(),
    'swapped_head_donator_reposing': NotImplementedValidator(),
    'swapped_body_donator_reshaping': NotImplementedValidator(),
    'swap_back': NotImplementedValidator(),
    'render_swapped_back_gaussians': RenderSwappedBackGaussiansOutputValidator(),
    'refine_swapped_back_images': NotImplementedValidator(),
    'cloth_fit_reshaping': ClothFitReshapingOutputValidator(),
    'create_combined_portraits': CreateCombinedPortraitsOutputValidator(),
}


def get_validator(stage_name: str) -> Optional[OutputValidator]:

    return OUTPUT_VALIDATORS.get(stage_name)
