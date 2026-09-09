import re
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
from loguru import logger


class CameraSelector:


    def __init__(
        self,
        forward_view_gt_a: str,
        forward_view_swapped_a: str,
        forward_view_gt_b: str,
        forward_view_swapped_b: str,
        camera_range_a: Tuple[int, int],
        camera_range_b: Tuple[int, int],
        filtered_pairs_a: Optional[List[str]] = None,
        filtered_pairs_b: Optional[List[str]] = None,
    ):


        self.forward_view_gt_a = forward_view_gt_a
        self.forward_view_swapped_a = forward_view_swapped_a
        self.forward_view_gt_b = forward_view_gt_b
        self.forward_view_swapped_b = forward_view_swapped_b
        self.camera_range_a = camera_range_a
        self.camera_range_b = camera_range_b
        self.filtered_pairs_a = set(filtered_pairs_a or [])
        self.filtered_pairs_b = set(filtered_pairs_b or [])


        self.forward_yaw_gt_a = self._extract_yaw_angle(self.forward_view_gt_a)
        self.forward_yaw_swapped_a = self._extract_yaw_angle(self.forward_view_swapped_a)
        self.forward_yaw_gt_b = self._extract_yaw_angle(self.forward_view_gt_b)
        self.forward_yaw_swapped_b = self._extract_yaw_angle(self.forward_view_swapped_b)

        logger.info(f"CameraSelector initialized with dual subject configuration:")
        logger.info(f"  Subject A - GT forward: {self.forward_view_gt_a} (yaw: {self.forward_yaw_gt_a})")
        logger.info(f"  Subject A - Swapped forward: {self.forward_view_swapped_a} (yaw: {self.forward_yaw_swapped_a})")
        logger.info(f"  Subject A - Range: {self.camera_range_a}, Filtered pairs: {self.filtered_pairs_a}")
        logger.info(f"  Subject B - GT forward: {self.forward_view_gt_b} (yaw: {self.forward_yaw_gt_b})")
        logger.info(f"  Subject B - Swapped forward: {self.forward_view_swapped_b} (yaw: {self.forward_yaw_swapped_b})")
        logger.info(f"  Subject B - Range: {self.camera_range_b}, Filtered pairs: {self.filtered_pairs_b}")

    @staticmethod
    def _extract_yaw_angle(camera_name: str) -> int:


        yaw_match = re.match(r'^(\d+)_p', camera_name)
        if not yaw_match:
            raise ValueError(f"Invalid camera name format: {camera_name}. Expected format: 'XXX_p±YYY'")

        yaw_str = yaw_match.group(1)
        return int(yaw_str)

    def _generate_yaw_range(self, forward_yaw: int, range_config: Tuple[int, int]) -> List[int]:

        before_count, after_count = range_config
        yaw_angles = []


        for i in range(before_count, 0, -1):
            angle = forward_yaw - (i * 5)
            if angle >= 0:
                yaw_angles.append(angle)


        yaw_angles.append(forward_yaw)


        for i in range(1, after_count + 1):
            angle = forward_yaw + (i * 5)
            if angle <= 355:
                yaw_angles.append(angle)

        return yaw_angles

    def _get_subject_camera_config(self, subject_type: str) -> Tuple[str, str, Tuple[int, int], Set[str]]:

        if subject_type.lower() == 'a':
            return (
                self.forward_view_gt_a,
                self.forward_view_swapped_a,
                self.camera_range_a,
                self.filtered_pairs_a
            )
        elif subject_type.lower() == 'b':
            return (
                self.forward_view_gt_b,
                self.forward_view_swapped_b,
                self.camera_range_b,
                self.filtered_pairs_b
            )
        else:
            raise ValueError(f"Invalid subject type: {subject_type}. Must be 'a' or 'b'.")

    def _find_swapped_directory(self, swapped_data_root: Path, source_subject_id: str, target_subject_id: str) -> str:

        pattern = f"swapped_{source_subject_id}head_on_{target_subject_id}body_"

        for item in swapped_data_root.iterdir():
            if item.is_dir() and item.name.startswith(pattern):

                return item.name

        raise ValueError(
            f"No swapped directory found for pattern '{pattern}*' in {swapped_data_root}. "
            f"Available directories: {[d.name for d in swapped_data_root.iterdir() if d.is_dir()][:5]}..."
        )

    def _get_available_cameras(self, data_root: Path, subject_id: str) -> Set[str]:

        subject_path = data_root / subject_id
        if not subject_path.exists():
            logger.debug(f"Subject path does not exist: {subject_path}")
            return set()

        available_cameras = set()
        for camera_dir in subject_path.iterdir():
            if camera_dir.is_dir() and re.match(r'^\d+_p[+-]\d+$', camera_dir.name):
                available_cameras.add(camera_dir.name)


        return available_cameras

    def _filter_available_cameras(
        self,
        yaw_angles: List[int],
        available_cameras: Set[str]
    ) -> List[str]:

        filtered_cameras = []

        for yaw in yaw_angles:

            yaw_str = f"{yaw:03d}"


            matching_cameras = [
                cam for cam in available_cameras
                if cam.startswith(f"{yaw_str}_p")
            ]


            filtered_cameras.extend(sorted(matching_cameras))

        return filtered_cameras

    def generate_camera_pairs(
        self,
        gt_data_root: Path,
        swapped_data_root: Path,
        source_subject_id: str,
        target_subject_id: str,
        subject_type: str
    ) -> List[Tuple[str, str]]:


        forward_view_gt, forward_view_swapped, camera_range, filtered_pairs = self._get_subject_camera_config(subject_type)


        swapped_directory = swapped_data_root


        gt_available = self._get_available_cameras(gt_data_root, source_subject_id)
        swapped_available = self._get_available_cameras(swapped_data_root, swapped_directory)

        logger.debug(f"Subject {source_subject_id}→{target_subject_id} (type {subject_type}): GT={len(gt_available)}, Swapped={len(swapped_available)} cameras")


        gt_forward_yaw = self._extract_yaw_angle(forward_view_gt)
        swapped_forward_yaw = self._extract_yaw_angle(forward_view_swapped)


        gt_range_count = camera_range[0]
        swapped_range_count = camera_range[1]


        gt_yaw_range = self._generate_yaw_range(gt_forward_yaw, (gt_range_count, gt_range_count))
        swapped_yaw_range = self._generate_yaw_range(swapped_forward_yaw, (swapped_range_count, swapped_range_count))

        logger.debug(f"GT yaw range ({forward_view_gt}, ±{gt_range_count}): {gt_yaw_range[:3]}...{gt_yaw_range[-3:]} ({len(gt_yaw_range)} angles)")
        logger.debug(f"Swapped yaw range ({forward_view_swapped}, ±{swapped_range_count}): {swapped_yaw_range[:3]}...{swapped_yaw_range[-3:]} ({len(swapped_yaw_range)} angles)")


        gt_cameras = self._filter_available_cameras(gt_yaw_range, gt_available)
        swapped_cameras = self._filter_available_cameras(swapped_yaw_range, swapped_available)

        logger.debug(f"Filtered cameras: GT={len(gt_cameras)}, Swapped={len(swapped_cameras)}")


        camera_pairs = list(zip(gt_cameras, swapped_cameras))


        filtered_camera_pairs = self._apply_pair_filtering(camera_pairs, filtered_pairs)

        logger.info(f"Generated {len(filtered_camera_pairs)} camera pairs for {source_subject_id}→{target_subject_id} (type {subject_type})")

        return filtered_camera_pairs


    def _apply_pair_filtering(self, camera_pairs: List[Tuple[str, str]], filtered_pairs_set: Set[str]) -> List[Tuple[str, str]]:

        if not filtered_pairs_set:
            return camera_pairs

        filtered_pairs = []

        for gt_camera, swapped_camera in camera_pairs:

            gt_yaw = self._extract_yaw_angle(gt_camera)
            swapped_yaw = self._extract_yaw_angle(swapped_camera)


            gt_yaw_key = f"{gt_yaw:03d}"
            swapped_yaw_key = f"{swapped_yaw:03d}"


            should_filter = False
            for filtered_pair in filtered_pairs_set:
                filtered_gt_yaw, filtered_swapped_yaw = filtered_pair.split(",")
                if gt_yaw_key == filtered_gt_yaw or swapped_yaw_key == filtered_swapped_yaw:
                    should_filter = True
                    break

            if not should_filter:
                filtered_pairs.append((gt_camera, swapped_camera))
            else:
                logger.debug(f"Filtered out camera pair: {gt_camera} <-> {swapped_camera}")

        return filtered_pairs

    def validate_camera_pairs(
        self,
        camera_pairs: List[Tuple[str, str]],
        gt_data_root: Path,
        swapped_data_root: Path,
        gt_subject_id: str,
        swapped_subject_id: str
    ) -> List[Tuple[str, str]]:

        validated_pairs = []

        for gt_camera, swapped_camera in camera_pairs:

            gt_subject_path = gt_data_root / gt_subject_id
            gt_image_path = gt_subject_path / gt_camera / "0000.jpg"
            gt_mask_path = gt_subject_path / gt_camera / "mask" / "head" / "0000.png"


            swapped_subject_path = swapped_data_root / swapped_subject_id
            swapped_image_path = swapped_subject_path / swapped_camera / "0000.jpg"
            swapped_mask_path = swapped_subject_path / swapped_camera / "mask" / "head" / "0000.png"


            required_files = [gt_image_path, gt_mask_path, swapped_image_path, swapped_mask_path]

            if all(file_path.exists() for file_path in required_files):
                validated_pairs.append((gt_camera, swapped_camera))
            else:
                logger.debug(f"Skipping camera pair {gt_camera} <-> {swapped_camera}: missing data")
                for file_path in required_files:
                    if not file_path.exists():
                        logger.debug(f"  Missing: {file_path}")

        logger.info(f"Validated {len(validated_pairs)}/{len(camera_pairs)} camera pairs")

        return validated_pairs

    def get_general_camera_pairs(self, subject_type_a: str = 'a', subject_type_b: str = 'b') -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:


        forward_view_gt_a, forward_view_swapped_a, camera_range_a, filtered_pairs_a = self._get_subject_camera_config(subject_type_a)
        forward_view_gt_b, forward_view_swapped_b, camera_range_b, filtered_pairs_b = self._get_subject_camera_config(subject_type_b)


        gt_forward_yaw_a = self._extract_yaw_angle(forward_view_gt_a)
        swapped_forward_yaw_a = self._extract_yaw_angle(forward_view_swapped_a)
        gt_forward_yaw_b = self._extract_yaw_angle(forward_view_gt_b)
        swapped_forward_yaw_b = self._extract_yaw_angle(forward_view_swapped_b)


        gt_range_count_a = camera_range_a[0]
        swapped_range_count_a = camera_range_a[1]
        gt_range_count_b = camera_range_b[0]
        swapped_range_count_b = camera_range_b[1]


        gt_yaw_range_a = self._generate_yaw_range(gt_forward_yaw_a, (gt_range_count_a, gt_range_count_a))
        swapped_yaw_range_a = self._generate_yaw_range(swapped_forward_yaw_a, (swapped_range_count_a, swapped_range_count_a))
        gt_yaw_range_b = self._generate_yaw_range(gt_forward_yaw_b, (gt_range_count_b, gt_range_count_b))
        swapped_yaw_range_b = self._generate_yaw_range(swapped_forward_yaw_b, (swapped_range_count_b, swapped_range_count_b))


        gt_cameras_a = [f"{yaw:03d}_p+000" for yaw in gt_yaw_range_a]
        swapped_cameras_a = [f"{yaw:03d}_p+000" for yaw in swapped_yaw_range_a]
        gt_cameras_b = [f"{yaw:03d}_p+000" for yaw in gt_yaw_range_b]
        swapped_cameras_b = [f"{yaw:03d}_p+000" for yaw in swapped_yaw_range_b]


        pairs_a = list(zip(gt_cameras_a, swapped_cameras_a))
        pairs_b = list(zip(gt_cameras_b, swapped_cameras_b))


        filtered_pairs_a = self._apply_pair_filtering(pairs_a, filtered_pairs_a)
        filtered_pairs_b = self._apply_pair_filtering(pairs_b, filtered_pairs_b)

        logger.info(f"Generated camera pairs: {len(filtered_pairs_a)} for subject A, {len(filtered_pairs_b)} for subject B")

        return (filtered_pairs_a, filtered_pairs_b)
