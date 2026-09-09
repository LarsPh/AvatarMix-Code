import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from model import libcore # Assuming this contains Camera and DataVec
from loguru import logger

from scene.dataset_readers import convert_to_scene_cameras # Assuming this converts DataVec to SceneCamera
import pytorch3d.io
import pytorch3d.structures
import pickle # Added for PKL loading
# Import SMPL utils if not already present
from model.smplx_utils import smplx_utils
from omegaconf import OmegaConf

# Define the 21 joints for the 63-parameter pose vector (based on user's list, excluding Pelvis 0)
# These are indices into a standard SMPL(X) joint order (e.g., as in smplx.JOINT_NAMES)
# User provided: 'Head':15,'Neck':12,'L_Collar':13,'R_Collar':14,'L_Shoulder':16,'R_Shoulder':17,
# 'L_Elbow':18,'R_Elbow':19,'L_Wrist':20,'R_Wrist':21,'Spine1':3,'Spine2':6,'Spine3':9,
# 'L_Hip':1,'R_Hip':2,'L_Knee':4,'R_Knee':5,'L_Ankle':7,'R_Ankle':8,'L_Foot':10,'R_Foot':11
# These are 21 joints if we exclude Pelvis(0).
DEFORMATION_NET_SMPLX_JOINT_INDICES = [
    15, 12, 13, 14, 16, 17, 18, 19, 20, 21,  # Head, Neck, LCollar, RCollar, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist (10 joints)
    3, 6, 9,                                # Spine1, Spine2, Spine3 (3 joints)
    1, 2,                                   # LHip, RHip (2 joints)
    4, 5,                                   # LKnee, RKnee (2 joints)
    7, 8,                                   # LAnkle, RAnkle (2 joints)
    10, 11                                  # LFoot, RFoot (2 joints)
] # Total 21 joints


class AvatarRexDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.dataset_config = config 
        self.dat_dir = Path(self.dataset_config.dat_dir)
        self.split = split
        self.cameras_extent = self.dataset_config.get('cameras_extent', 1.0) 
        self.img_wh = None

        self.all_camera_params = {} 
        self.data_samples = [] 
        self.frame_id_map = {} 

        self.avatarrex_conf = self.dataset_config.get('avatarrex_config', {})
        self.use_smpl_params = self.avatarrex_conf.get('use_smpl_params', True)
        self.smpl_model_type = self.avatarrex_conf.get('model_type', 'smplx')
        self.smpl_gender = self.avatarrex_conf.get('gender', 'neutral')
        self.use_deformation = self.dataset_config.get('use_deformation', False) # Get from dataset config
        self.free_gaussians = self.dataset_config.get('free_gaussians', False)
        if self.free_gaussians:
            # In free-gaussians mode, we must not depend on SMPL or OBJ meshes
            self.use_smpl_params = False
        # Optional custom relative paths under each camera root for RGB and mask
        # Supports placeholders: {frame} for zero-padded frame_id_str, {cam} for camera id.
        self.custom_rgb_relpath = self.dataset_config.get('custom_rgb_relpath', None)
        self.custom_mask_relpath = self.dataset_config.get('custom_mask_relpath', None)

        # For deformation mode: reference frame assets
        self.ref_frame_smpl_params = None
        self.lbs_weights_for_ref_mesh = None
        self.ref_frame_garment_mask = None
        self.reference_frame_idx_for_smpl = self.avatarrex_conf.get('reference_frame_idx_for_smpl', 36) # Default to 36

        self.garment_label_file_override = self.avatarrex_conf.get('garment_label_file', None)
        self.garment_labels_map = None # To be loaded if path is valid
        self.mask_ext = 'jpg'
        self.fullbody_cam_ids = None
        # Dataset flavor (affects SMPL-X options like hand PCA).
        try:
            sampling_cfg0 = self.avatarrex_conf.get('sampling', {})
        except Exception:
            sampling_cfg0 = {}
        self.dataset_flavor = sampling_cfg0.get('dataset_flavor', 'generic') if hasattr(sampling_cfg0, "get") else 'generic'

        self.load_cameras_and_frames()

        if self.use_smpl_params or self.use_deformation:
            self.load_smpl_params()
            if self.use_deformation:
                self.load_reference_frame_assets()
                self.load_garment_labels_for_ref_mesh()
        else:
            self.data_samples = self._temp_potential_samples
        
        if not self.data_samples:
            print(f"[AvatarRexDataset][{self.split}] No samples found. Check config and data paths.")
        else:
            print(f'[AvatarRexDataset][{self.split}] num_samples = {len(self.data_samples)}')

    def load_reference_frame_assets(self):
        if not self.use_deformation:
            return

        print(f"[AvatarRexDataset] Loading assets for reference frame (deformation mode)... Ref SMPL idx: {self.reference_frame_idx_for_smpl}")
        device = self.dataset_config.get('data_device', 'cpu')

        # 1. Load LBS weights
        lbs_weights_path_str = self.avatarrex_conf.get('lbs_weights_path')
        if not lbs_weights_path_str:
            raise ValueError("[AvatarRexDataset] `lbs_weights_path` not specified in avatarrex_config for deformation mode.")
        lbs_weights_path = Path(lbs_weights_path_str)
        if not lbs_weights_path.is_file(): # Check if it's a file
             lbs_weights_path = self.dat_dir / lbs_weights_path_str # Try relative to dat_dir
        if not lbs_weights_path.exists():
            raise FileNotFoundError(f"[AvatarRexDataset] LBS weights file not found: {lbs_weights_path_str} (abs) or {lbs_weights_path} (rel)")
        try:
            self.lbs_weights_for_ref_mesh = torch.from_numpy(np.load(lbs_weights_path).astype(np.float32)).to(device)
            print(f"  Loaded LBS weights from {lbs_weights_path}, shape: {self.lbs_weights_for_ref_mesh.shape}")
        except Exception as e:
            raise RuntimeError(f"Error loading LBS weights from {lbs_weights_path}: {e}")

        # 2. Load Reference Frame SMPL parameters.
        #
        # IMPORTANT (ActorsHQ): reference Gaussians/mesh may be built with a separately fine-tuned SMPLX
        # for the reference frame (single-frame dataset). In that case, we must load the reference SMPL
        # params from that dataset, otherwise the LBS "ref" joint transforms won't match the embedding mesh.
        #
        # Priority:
        #  (a) avatarrex_config.reference_smpl_params_path (explicit NPZ path)
        #  (b) avatarrex_config.reference_dat_dir + "/smpl_params.npz"
        #  (c) heuristic: look for "smpl_params.npz" near lbs_weights_path (walk parents)
        #  (d) fallback: current sequence dataset's smpl_params_all_frames_npz at reference_frame_idx_for_smpl
        ref_npz_path = None
        ref_npz_cfg = self.avatarrex_conf.get('reference_smpl_params_path', None)
        if ref_npz_cfg:
            p = Path(ref_npz_cfg)
            if not p.is_file():
                p = self.dat_dir / ref_npz_cfg
            if p.exists():
                ref_npz_path = p
            else:
                print(f"[AvatarRexDataset] WARNING: reference_smpl_params_path set but not found: {ref_npz_cfg}")
        if ref_npz_path is None:
            ref_dat_dir = self.avatarrex_conf.get('reference_dat_dir', None)
            if ref_dat_dir:
                p = Path(ref_dat_dir) / "smpl_params.npz"
                if not p.is_file():
                    p = self.dat_dir / ref_dat_dir / "smpl_params.npz"
                if p.exists():
                    ref_npz_path = p
        if ref_npz_path is None:
            # Heuristic: try to find a smpl_params.npz alongside the LBS weights root.
            try:
                cand = None
                for parent in [lbs_weights_path] + list(lbs_weights_path.parents)[:6]:
                    # Prefer fine-tuned params if present.
                    parent_dir = parent if parent.is_dir() else parent.parent
                    finetuned = sorted(parent_dir.glob("smpl_params_finetuned*.npz"))
                    if finetuned:
                        cand = finetuned[0]
                        break
                    p = parent_dir / "smpl_params.npz"
                    if p.exists():
                        cand = p
                        break
                ref_npz_path = cand
            except Exception:
                ref_npz_path = None

        def _load_ref_from_npz(npz_obj, ref_idx: int):
            # Choose a safe index for single-frame NPZs.
            n = None
            if 'body_pose' in npz_obj:
                n = int(npz_obj['body_pose'].shape[0])
            idx = int(ref_idx)
            if n is not None:
                idx = max(0, min(idx, n - 1))
            out = {}
            for key in ['betas', 'body_pose', 'global_orient', 'transl', 'jaw_pose', 'expression',
                        'left_hand_pose', 'right_hand_pose', 'scale', 'Rh', 'Th', 'v_shape', 'v_pose']:
                if key not in npz_obj:
                    continue
                arr = npz_obj[key]
                if key == 'betas':
                    # betas may be (1,10), (N,10) or (10,)
                    if arr.ndim == 1:
                        out[key] = torch.from_numpy(arr.astype(np.float32)).to(device)
                    else:
                        out[key] = torch.from_numpy(arr[0].astype(np.float32) if arr.shape[0] == 1 else arr[idx].astype(np.float32)).to(device)
                elif key in ('v_shape', 'v_pose') and arr.ndim == 3 and arr.shape[0] == 1:
                    # Shared per-vertex offsets; keep (V,3) for the ref frame (batched later as needed).
                    out[key] = torch.from_numpy(arr[0].astype(np.float32)).to(device)
                else:
                    out[key] = torch.from_numpy(arr[idx].astype(np.float32) if arr.ndim > 1 else arr.astype(np.float32)).to(device)
            return out, idx

        ref_idx = self.reference_frame_idx_for_smpl
        if ref_npz_path is not None:
            try:
                ref_npz = np.load(ref_npz_path, allow_pickle=True)
                self.ref_frame_smpl_params, used_idx = _load_ref_from_npz(ref_npz, ref_idx)
                print(f"  Loaded reference SMPL parameters from {ref_npz_path} (idx={used_idx}).")
                return
            except Exception as e:
                print(f"[AvatarRexDataset] WARNING: Failed to load reference SMPL params from {ref_npz_path}: {e}. Falling back to sequence NPZ.")

        # Fallback: current sequence dataset's SMPL params.
        if not hasattr(self, 'smpl_params_all_frames_npz') or self.smpl_params_all_frames_npz is None:
            raise RuntimeError("SMPL NPZ data (smpl_params_all_frames_npz) not loaded before load_reference_frame_assets.")
        if "body_pose" not in self.smpl_params_all_frames_npz:
            raise RuntimeError("SMPL NPZ missing 'body_pose'; cannot index reference frame.")
        if not (0 <= ref_idx < self.smpl_params_all_frames_npz["body_pose"].shape[0]):
            raise ValueError(f"Reference frame index {ref_idx} is out of bounds for loaded SMPL NPZ data.")
        self.ref_frame_smpl_params = {}
        for key in ['betas', 'body_pose', 'global_orient', 'transl', 'jaw_pose', 'expression',
                    'left_hand_pose', 'right_hand_pose', 'scale', 'Rh', 'Th', 'v_shape', 'v_pose']:
            if key not in self.smpl_params_all_frames_npz:
                continue
            if key == 'betas':
                arr = self.smpl_params_all_frames_npz['betas']
                if arr.ndim == 1:
                    self.ref_frame_smpl_params[key] = torch.from_numpy(arr.astype(np.float32)).to(device)
                elif arr.shape[0] == 1:
                    self.ref_frame_smpl_params[key] = torch.from_numpy(arr[0].astype(np.float32)).to(device)
                else:
                    self.ref_frame_smpl_params[key] = torch.from_numpy(arr[ref_idx].astype(np.float32)).to(device)
            elif key in ('v_shape', 'v_pose'):
                arr = self.smpl_params_all_frames_npz[key]
                if arr.ndim == 3 and arr.shape[0] == 1:
                    self.ref_frame_smpl_params[key] = torch.from_numpy(arr[0].astype(np.float32)).to(device)
                else:
                    self.ref_frame_smpl_params[key] = torch.from_numpy(arr[ref_idx].astype(np.float32)).to(device)
            else:
                self.ref_frame_smpl_params[key] = torch.from_numpy(self.smpl_params_all_frames_npz[key][ref_idx].astype(np.float32)).to(device)
        print(f"  Loaded SMPL parameters for reference frame {ref_idx} from sequence NPZ.")

    def load_garment_labels_for_ref_mesh(self):
        if not self.use_deformation:
            return

        garment_path_str = self.avatarrex_conf.get('garment_label_file') # Path from config
        if not garment_path_str:
            print("[AvatarRexDataset] Garment label file not specified in avatarrex_config.garment_label_file. Skipping.")
            return

        garment_path = Path(garment_path_str)
        if not garment_path.is_file(): garment_path = self.dat_dir / garment_path_str # try relative
        if not garment_path.exists():
            print(f"[AvatarRexDataset] WARNING: Garment label file not found: {garment_path_str} or {garment_path}")
            return

        try:
            with open(garment_path, 'rb') as f: loaded_pkl = pickle.load(f)
            if 'scan_labels' not in loaded_pkl:
                print(f"[AvatarRexDataset] WARNING: 'scan_labels' key not found in {garment_path}")
                return

            vertex_labels_indices = loaded_pkl['scan_labels']
            garment_label_indices = [2, 3, 4, 5] if vertex_labels_indices.max() >= 5 else [2, 3, 4]
            garment_class_indices = torch.tensor(garment_label_indices, dtype=torch.long)
            vertex_labels_tensor = torch.from_numpy(vertex_labels_indices).long()
            self.ref_frame_garment_mask = torch.isin(vertex_labels_tensor, garment_class_indices).to(self.dataset_config.get('data_device', 'cpu'))
            
            print(f"[AvatarRexDataset] Loaded garment labels from {garment_path}. "
                  f"Total vertices: {self.ref_frame_garment_mask.shape[0]}, "
                  f"Garment vertices: {self.ref_frame_garment_mask.sum()}")
            print(f"[AvatarRexDataset] Note: Garment mask validation will be done against actual mesh topology at runtime.")
        except Exception as e:
            print(f"[AvatarRexDataset] ERROR loading or processing garment label file {garment_path}: {e}")
            self.ref_frame_garment_mask = None

    def load_cameras_and_frames(self):
        calib_file = self.dat_dir / 'calibration_full.json'
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")

        with open(calib_file, 'r') as f:
            all_calibrations = json.load(f)

        frames_config_for_split = self.avatarrex_conf.get(f'{self.split}_frames', None)
        cameras_to_load = self.avatarrex_conf.get(f'{self.split}_cameras', None)
        frames_to_load_str = []

        frame_format_digits = self.avatarrex_conf.get('frame_format_digits', 8)
        # OmegaConf wraps YAML lists/dicts as ListConfig/DictConfig; use OmegaConf helpers.
        if (frames_config_for_split is not None) and (OmegaConf.is_list(frames_config_for_split) or isinstance(frames_config_for_split, (list, tuple))):
            # Direct list of frame IDs (strings or ints)
            for f in frames_config_for_split:
                if isinstance(f, int):
                    frames_to_load_str.append(f"{f:0{frame_format_digits}d}")
                else:
                    s = str(f)
                    # Accept numeric strings as indices and pad; keep already-padded ids unchanged.
                    if s.isdigit() and len(s) < int(frame_format_digits):
                        frames_to_load_str.append(f"{int(s):0{frame_format_digits}d}")
                    else:
                        frames_to_load_str.append(s)
            print(f"Using provided list of frames for split '{self.split}': {len(frames_to_load_str)} frames.")
        elif (frames_config_for_split is not None) and (OmegaConf.is_dict(frames_config_for_split) or hasattr(frames_config_for_split, "get")):
            # Dictionary with start, end, step
            start_idx = frames_config_for_split.get('start_idx')
            end_idx = frames_config_for_split.get('end_idx')
            step_idx = frames_config_for_split.get('step_idx', 1)

            if start_idx is not None and end_idx is not None:
                for i in range(start_idx, end_idx, step_idx):
                    frames_to_load_str.append(f"{i:0{frame_format_digits}d}")
                print(f"Generated frames for split '{self.split}' from range: {len(frames_to_load_str)} frames.")
            else:
                print(f"Warning: Frame range config for split '{self.split}' incomplete.")
                frames_config_for_split = None 
        else:
            # No explicit frame selection provided
            frames_config_for_split = None

        if not frames_to_load_str and frames_config_for_split is None:
            # Fallback to discovering frames from mesh directory if no explicit frames are given
            # This part might be less relevant if use_smpl_params=True, as frames are dictated by smpl_params.npz
            mesh_dir_obj = self.dat_dir / 'mesh' / 'trimesh_cleaned' # For OBJ fallback
            if not self.use_smpl_params and mesh_dir_obj.exists(): # Only discover if not using SMPL and OBJ dir exists
                discovered_frames = sorted([p.stem for p in mesh_dir_obj.glob('*.obj')])
                if discovered_frames:
                    frames_to_load_str = discovered_frames
                    print(f"Discovered frames from OBJ mesh directory for split '{self.split}': {len(frames_to_load_str)}")
            # If using SMPL, frames will be implicitly defined by what's in smpl_params.npz and selected by indices.
            # If frames_to_load_str is still empty, it means we must rely on smpl_params.npz existing.

        if not frames_to_load_str and self.use_smpl_params:
            # If using SMPL and no frames specified, we assume all frames in smpl_params.npz are candidates
            # The actual frames used will be those for which image/mask data also exists.
            # We'll populate self.frame_id_list more definitively after loading smpl_params.
            print(f"No explicit frames provided for split '{self.split}' with SMPL. Will derive from available SMPL data and images.")
        elif not frames_to_load_str:
            print(f"Warning: No frames specified or discovered for split '{self.split}'. Dataset may be empty if not using SMPL or if SMPL data is also limited.")


        # Create a preliminary list of unique frame IDs specified, will be refined by SMPL if used
        # This self.frame_id_list will be used to index into SMPL parameters
        self.frame_id_list = sorted(list(set(frames_to_load_str))) if frames_to_load_str else []
        
        # Important: If use_smpl_params, self.frame_id_list will be REFINED/OVERWRITTEN in load_smpl_params
        # to match the actual frames available in the smpl_params.npz file that also match any user start/end/step.

        # After load_smpl_params (if use_smpl_params=True), self.frame_id_list will contain the
        # frame ID strings that are *actually available* from SMPL data for this split.
        # Then, self.frame_id_map can be built.
        # If not using SMPL, self.frame_id_map is built from the discovered/specified frames.
        
        # Defer frame_id_map creation until after smpl_params are loaded if use_smpl_params is true,
        # because load_smpl_params might adjust self.frame_id_list
        if not self.use_smpl_params and self.frame_id_list:
            self.frame_id_map = {frame_id_str: i for i, frame_id_str in enumerate(self.frame_id_list)}
            print(f"Non-SMPL mode: Total unique frames for split '{self.split}': {len(self.frame_id_list)}. Mapped to indices 0-{len(self.frame_id_list)-1}.")


        # Stable camera ordering (for index-based selection like [126,127,128]).
        def _cam_sort_key(cid: str):
            try:
                return (0, int(cid))
            except Exception:
                return (1, cid)
        available_camera_ids = sorted(list(all_calibrations.keys()), key=_cam_sort_key)
        self.all_camera_ids_sorted = available_camera_ids

        def _resolve_cameras(cameras_spec, available_ids):
            if cameras_spec is None:
                return list(available_ids)
            resolved = []
            for c in cameras_spec:
                # direct match
                if isinstance(c, str) and c in available_ids:
                    resolved.append(c)
                    continue
                # 1-based index selection
                idx = None
                if isinstance(c, int):
                    idx = c
                elif isinstance(c, str) and c.isdigit() and c not in available_ids:
                    idx = int(c)
                if idx is not None:
                    if 1 <= idx <= len(available_ids):
                        resolved.append(available_ids[idx - 1])
                    else:
                        logger.warning(f"Camera index {idx} out of range [1,{len(available_ids)}]; skipping.")
                else:
                    logger.warning(f"Unknown camera spec '{c}' (not a camera id and not an index); skipping.")
            return resolved

        cameras_to_use_for_split = _resolve_cameras(cameras_to_load, available_camera_ids)

        # Optional: FULLBODY camera filtering for training (ActorsHQ-specific heuristic).
        sampling_cfg = self.avatarrex_conf.get('sampling', {})
        dataset_flavor = sampling_cfg.get('dataset_flavor', 'generic')
        enable_fullbody_cam_filter = sampling_cfg.get('enable_fullbody_cam_filter', bool(self.use_deformation))
        fullbody_cam_allowlist = sampling_cfg.get('fullbody_cam_allowlist', None)
        target_wh = sampling_cfg.get('fullbody_target_wh', [747, 1022])  # [W, H]
        tol = int(sampling_cfg.get('fullbody_tol', 2))

        # Apply allowlist override first (works for any dataset flavor).
        if fullbody_cam_allowlist:
            cameras_to_use_for_split = [cid for cid in fullbody_cam_allowlist if cid in cameras_to_use_for_split]
        elif enable_fullbody_cam_filter and dataset_flavor == 'actorshq':
            try:
                target_w, target_h = int(target_wh[0]), int(target_wh[1])
            except Exception:
                target_w, target_h = 747, 1022
            filtered = []
            for cam_id in cameras_to_use_for_split:
                cam_data = all_calibrations[cam_id]
                img_size_w_h = cam_data.get('imgSize', None)
                if img_size_w_h is None:
                    continue
                w, h = int(img_size_w_h[0]), int(img_size_w_h[1])
                if h > w and abs(w - target_w) <= tol and abs(h - target_h) <= tol:
                    filtered.append(cam_id)
            if filtered:
                cameras_to_use_for_split = filtered
            else:
                print(f"[AvatarRexDataset] WARNING: ActorsHQ fullbody cam filter produced 0 cams; falling back to all cams for split '{self.split}'.")

        # Expose the final camera set used for this split.
        self.fullbody_cam_ids = sorted(list(cameras_to_use_for_split))

        first_cam_loaded = False
        
        # Temp list of samples to be built. If using SMPL, this list is filtered against available SMPL frames.
        _potential_samples = []
        _frames_with_any_camdata = set()

        # Performance note: checking mask/image existence for every (cam,frame) can be very slow on network FS.
        # Optional fast path: assume all requested frames exist for each camera and defer missing-file errors to __getitem__.
        fast_sample_discovery = bool(sampling_cfg.get('fast_sample_discovery', False))
        if fast_sample_discovery:
            for cam_id in cameras_to_use_for_split:
                cam_data = all_calibrations[cam_id]
                K = np.array(cam_data['K']).reshape(3, 3)
                R_w2c = np.array(cam_data['R']).reshape(3, 3)
                T_w2c = np.array(cam_data['T']).reshape(3, 1)
                img_size_w_h = cam_data['imgSize']

                self.all_camera_params[cam_id] = {
                    'K': K, 'R_w2c': R_w2c, 'T_w2c': T_w2c,
                    'width': int(img_size_w_h[0]), 'height': int(img_size_w_h[1])
                }
                if not first_cam_loaded:
                    self.img_wh = (int(img_size_w_h[0]), int(img_size_w_h[1]))
                    first_cam_loaded = True

                for frame_id_str_candidate in self.frame_id_list:
                    _potential_samples.append((frame_id_str_candidate, cam_id))
                    _frames_with_any_camdata.add(frame_id_str_candidate)

            self._temp_potential_samples = _potential_samples
            self._temp_frames_with_any_camdata = _frames_with_any_camdata
            self.cam = libcore.Camera()  # For repose_avatar.py
            return

        for cam_id in cameras_to_use_for_split:
            cam_data = all_calibrations[cam_id]
            K = np.array(cam_data['K']).reshape(3, 3)
            R_w2c = np.array(cam_data['R']).reshape(3, 3)
            T_w2c = np.array(cam_data['T']).reshape(3, 1)
            img_size_w_h = cam_data['imgSize'] 

            self.all_camera_params[cam_id] = {
                'K': K, 'R_w2c': R_w2c, 'T_w2c': T_w2c, 
                'width': int(img_size_w_h[0]), 'height': int(img_size_w_h[1])
            }
            if not first_cam_loaded:
                self.img_wh = (int(img_size_w_h[0]), int(img_size_w_h[1]))
                first_cam_loaded = True

            # Iterate through specified/discovered frames to find valid image/mask pairs
            # If self.frame_id_list is empty here (e.g. use_smpl_params=True and no explicit frames given),
            # then we must iterate through *all* potential frames on disk for this camera.
            # This is inefficient. It's better to have self.frame_id_list populated first,
            # ideally by load_smpl_params if use_smpl_params=True.

            # Let's assume self.frame_id_list is populated (either by user spec or by load_smpl_params later)
            # For now, if it's empty, we can't form samples.
            # The logic is: load_smpl_params will define the true self.frame_id_list for the split.
            # Then this loop runs, or a subsequent filtering step.
            # For simplicity, this loop will run over whatever is in self.frame_id_list. If it's empty and using SMPL,
            # it means no explicit frame range was given, so we need another way to get candidate frames.

            # This loop relies on self.frame_id_list being set *before* this method is fully done
            # if self.use_smpl_params is True.
            # Let's assume self.frame_id_list holds the frame IDs derived from the config's frame range/list for this split.
            # After load_smpl_params, it will be intersected with available SMPL frames.
            
            # The current self.frame_id_list contains frames specified by start/end/step or list.
            for frame_id_str_candidate in self.frame_id_list:
                # Resolve custom or default paths
                if self.custom_rgb_relpath:
                    img_rel = self.custom_rgb_relpath.format(frame=frame_id_str_candidate, cam=cam_id)
                    img_path = self.dat_dir / cam_id / img_rel
                else:
                    img_path = self.dat_dir / cam_id / f"{frame_id_str_candidate}.jpg"
                if self.custom_mask_relpath:
                    mask_rel = self.custom_mask_relpath.format(frame=frame_id_str_candidate, cam=cam_id)
                    mask_path = self.dat_dir / cam_id / mask_rel
                else:
                    mask_path = self.dat_dir / cam_id / 'mask' / 'pha' / f"{frame_id_str_candidate}.{self.mask_ext}"
                # check path first and check separately
                mask_check, img_check = mask_path.exists(), img_path.exists()
                if (not mask_check) and (not self.custom_mask_relpath):
                    self.mask_ext = 'png'
                    mask_path = self.dat_dir / cam_id / 'mask' / 'pha' / f"{frame_id_str_candidate}.{self.mask_ext}"
                    mask_check = mask_path.exists()
                
                if not img_check:
                    logger.warning(f"Image path not found: {img_path}")
                if not mask_check:
                    logger.warning(f"Mask path not found: {mask_path}")

                mesh_check = True
                if (not self.use_smpl_params) and (not self.free_gaussians): # Only require OBJ when not using SMPL and not in free mode
                    mesh_path_obj = self.dat_dir / 'mesh' / 'trimesh_cleaned' / f"{frame_id_str_candidate}.obj"
                    mesh_check = mesh_path_obj.exists()

                if img_check and mask_check and mesh_check:
                    _potential_samples.append((frame_id_str_candidate, cam_id))
                    _frames_with_any_camdata.add(frame_id_str_candidate)
        
        # If using SMPL, self.frame_id_list will be updated by load_smpl_params.
        # We store these potential samples and will filter them *after* load_smpl_params.
        self._temp_potential_samples = _potential_samples
        self._temp_frames_with_any_camdata = _frames_with_any_camdata

        self.cam = libcore.Camera() # For repose_avatar.py

    def get_smpl_config(self):
        # Base config from YAML (AvatarRex / ActorsHQ defaults)
        use_pca = bool(self.avatarrex_conf.get('smpl_use_pca', False))
        num_pca_comps = int(self.avatarrex_conf.get('smpl_num_pca_comps', 6))
        flat_hand_mean = bool(self.avatarrex_conf.get('flat_hand_mean', False))

        # TalkBody4D: full 45D hand poses (no PCA), per your reference.
        # Also auto-detect if NPZ provides 45D hand poses to avoid einsum size mismatch.
        try:
            lh = self.smpl_params.get("left_hand_pose", None) if isinstance(self.smpl_params, dict) else None
            rh = self.smpl_params.get("right_hand_pose", None) if isinstance(self.smpl_params, dict) else None
            lh_dim = int(lh.shape[-1]) if isinstance(lh, torch.Tensor) and lh.ndim >= 2 else None
            rh_dim = int(rh.shape[-1]) if isinstance(rh, torch.Tensor) and rh.ndim >= 2 else None
        except Exception:
            lh_dim, rh_dim = None, None
        if str(getattr(self, "dataset_flavor", "generic")).lower() == "talkbody4d" or lh_dim == 45 or rh_dim == 45:
            use_pca = False
            num_pca_comps = 45
            flat_hand_mean = True

        smpl_config = {
            'model_type': self.smpl_model_type, 'gender': self.smpl_gender, 'ext': 'npz',
            'use_pca': use_pca,
            'num_pca_comps': num_pca_comps,
            'flat_hand_mean': flat_hand_mean,
            'batch_size': len(self.frame_id_list), 
            'num_betas': self.smpl_params['betas'].shape[-1] if 'betas' in self.smpl_params else 10,
            'model_path': self.avatarrex_conf.get('smpl_model_path', None),
            'create_transl': False, 'create_global_orient': False, 'create_body_pose': False, 'create_betas': False,
        }
        # TalkBody4D provides per-vertex offsets (v_shape/v_pose) that require the smplx_ani implementation.
        # We auto-enable it if these fields exist in the NPZ.
        try:
            if ('v_shape' in getattr(self, 'smpl_params_all_frames_npz', {})) or ('v_pose' in getattr(self, 'smpl_params_all_frames_npz', {})):
                smpl_config['implementation'] = 'smplx_ani'
        except Exception:
            pass
        for key_suffix in ['jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'left_hand_pose', 'right_hand_pose']:
            if key_suffix in self.smpl_params: smpl_config[f'create_{key_suffix}'] = False
        return smpl_config

    def load_smpl_params(self):
        smpl_file = self.dat_dir / 'smpl_params.npz'
        if not smpl_file.exists():
             raise FileNotFoundError(f"SMPL parameter file not found: {smpl_file}")
             
        print(f"[AvatarRexDataset] Loading SMPL params from {smpl_file} for split '{self.split}'")
        self.smpl_params_all_frames_npz = dict(np.load(smpl_file, allow_pickle=True))
        all_npz_frame_count = self.smpl_params_all_frames_npz["body_pose"].shape[0]
        
        # Frame selection logic (from previous implementation, uses self.frame_id_list from config)
        # ... (ensure this correctly produces valid_npz_indices and updates self.frame_id_list for the split)
        valid_npz_indices = []
        valid_frame_id_strs_for_split = []
        if not self.frame_id_list: # If no frames specified in config (e.g. train_frames: null)
            # This case should be handled by avatarrex_config.train_frames having a default range or list
            # If truly no frame spec, then this split might be empty or needs to discover frames.
            # For SMPL mode, it usually implies using all frames from NPZ that have corresponding images.
            # For now, require frame_id_list to be populated from config for SMPL mode.
            raise ValueError(f"Frame list for split '{self.split}' is empty. Please specify frames in config (e.g., train_frames). ")

        for frame_id_str in self.frame_id_list: # self.frame_id_list comes from config range/list
            try:
                npz_idx = int(frame_id_str) 
                if 0 <= npz_idx < all_npz_frame_count:
                    valid_npz_indices.append(npz_idx)
                    valid_frame_id_strs_for_split.append(frame_id_str)
            except ValueError:
                pass
        
        if not valid_npz_indices:
             print(f"Warning: No valid SMPL NPZ indices for split '{self.split}'. Max NPZ idx: {all_npz_frame_count-1}.")
             self.use_smpl_params = False; return

        self.frame_id_list = sorted(list(set(valid_frame_id_strs_for_split))) # Update for the split
        self.frame_id_map = {fid_str: i for i, fid_str in enumerate(self.frame_id_list)} # Map for split-local indexing
        print(f"  Split '{self.split}' will use {len(self.frame_id_list)} SMPL frames from NPZ indices: {valid_npz_indices[:5]}...{valid_npz_indices[-5:] if len(valid_npz_indices)>10 else valid_npz_indices}")

        self.smpl_params = {}
        # Note: TalkBody4D NPZs may additionally include:
        # - Rh/Th: post-forward rigid transform (applied after SMPL forward)
        # - v_shape/v_pose: per-vertex offsets (requires smplx_ani implementation)
        for key in ['betas', 'body_pose', 'global_orient', 'transl', 'jaw_pose', 'expression',
                    'left_hand_pose', 'right_hand_pose', 'scale', 'Rh', 'Th', 'v_shape', 'v_pose']:
            if key in self.smpl_params_all_frames_npz:
                full_data = torch.from_numpy(self.smpl_params_all_frames_npz[key].astype(np.float32))
                if key == 'betas' and full_data.shape[0] == 1: # Shared beta
                    self.smpl_params[key] = full_data.repeat(len(valid_npz_indices), 1) if full_data.ndim == 2 else full_data.unsqueeze(0).repeat(len(valid_npz_indices), 1)
                elif key == 'betas' and full_data.ndim == 1 : # (10,) case
                     self.smpl_params[key] = full_data.unsqueeze(0).repeat(len(valid_npz_indices), 1)
                elif key in ('v_shape', 'v_pose') and full_data.ndim == 3 and full_data.shape[0] == 1:
                    # Shared per-vertex offsets; keep shape (1,V,3) to allow broadcasting over batch.
                    self.smpl_params[key] = full_data
                else: # Per-frame params
                    self.smpl_params[key] = full_data[valid_npz_indices]
        
        # ... (rest of smpl_model creation for the split as before, using self.smpl_params for the split)
        smpl_config = self.get_smpl_config()
        self.smpl_model = smplx_utils.create_smplx_model(**smpl_config).to(self.dataset_config.get('data_device', 'cpu'))
        with torch.no_grad():
            model_input_params = {k: v for k, v in self.smpl_params.items()} 

            # Fix shape/expression coeff dim mismatches by truncating inputs to model capacity.
            # This prevents blend_shapes einsum failures when NPZ provides higher-dim expression/betas
            # than the shipped SMPL-X model supports (common for reduced models with 10 expr coeffs).
            try:
                # betas vs shapedirs
                shapedirs = getattr(self.smpl_model, "shapedirs", None)
                shapedirs_l = int(shapedirs.shape[-1]) if isinstance(shapedirs, torch.Tensor) else None
                if shapedirs_l is not None and isinstance(model_input_params.get("betas", None), torch.Tensor):
                    b = model_input_params["betas"]
                    if int(b.shape[-1]) != int(shapedirs_l):
                        model_input_params["betas"] = b[..., : int(shapedirs_l)]

                # expression vs expr_dirs/num_expression_coeffs
                expr_dirs = getattr(self.smpl_model, "expr_dirs", None)
                expr_l = int(expr_dirs.shape[-1]) if isinstance(expr_dirs, torch.Tensor) else None
                if expr_l is None:
                    try:
                        expr_l = int(getattr(self.smpl_model, "num_expression_coeffs"))
                    except Exception:
                        expr_l = None
                if expr_l is not None and isinstance(model_input_params.get("expression", None), torch.Tensor):
                    e = model_input_params["expression"]
                    if int(e.shape[-1]) != int(expr_l):
                        model_input_params["expression"] = e[..., : int(expr_l)]
            except Exception:
                pass

            smpl_output = self.smpl_model(**model_input_params, return_verts=True, return_full_pose=True)
            self.smpl_verts = smpl_output.vertices.detach() 
            self.smpl_full_pose_from_model = smpl_output.full_pose.detach() if hasattr(smpl_output, 'full_pose') else None

        self.mesh_py3d = pytorch3d.structures.Meshes(
            verts=[self.smpl_verts[0]], 
            faces=[torch.tensor(self.smpl_model.faces.astype(np.int64))]
        )
        print(f"  Generated SMPL vertices for split '{self.split}', shape: {self.smpl_verts.shape}")

        final_samples = []
        valid_smpl_frame_ids_set = set(self.frame_id_list)
        for frame_id_str, cam_id in self._temp_potential_samples:
            if frame_id_str in valid_smpl_frame_ids_set:
                final_samples.append((frame_id_str, cam_id))
        self.data_samples = sorted(list(set(final_samples)))
        print(f"  Final valid image samples for split '{self.split}' after SMPL check: {len(self.data_samples)}")

    def get_mesh_data(self, frame_id_str, device='cpu'):
        mesh_path = self.dat_dir / 'mesh' / 'trimesh_cleaned' / f"{frame_id_str}.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        verts, faces_idx, _ = pytorch3d.io.load_obj(str(mesh_path), device=device)
        faces = faces_idx.verts_idx
        
        frame_mesh_py3d = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
        
        return {
            'mesh_verts': verts, 
            'mesh_norms': frame_mesh_py3d.verts_normals_packed(), 
            'mesh_faces': faces, 
        }


    def get_smpl_mesh(self, frame_map_idx, device='cpu'):
        if not hasattr(self, 'smpl_verts') or self.smpl_verts is None: raise RuntimeError("SMPL verts not loaded.")
        if frame_map_idx >= self.smpl_verts.shape[0]: raise IndexError(f"Frame map idx {frame_map_idx} out of bounds.")
        if not hasattr(self, 'mesh_py3d'): raise RuntimeError("mesh_py3d not initialized.")
             
        frame_verts = self.smpl_verts[frame_map_idx:frame_map_idx+1].to(device)
        frame_mesh = self.mesh_py3d.update_padded(frame_verts) 
        mesh_output = {
            'mesh_verts': frame_mesh.verts_packed(), 'mesh_norms': frame_mesh.verts_normals_packed(), 'mesh_faces': frame_mesh.faces_packed(),
        }
        raw_smpl_params_for_frame = {}
        for key in ['body_pose', 'global_orient', 'betas', 'transl', 'jaw_pose', 'expression',
                    'left_hand_pose', 'right_hand_pose', 'scale', 'Rh', 'Th', 'v_shape', 'v_pose']:
            if key in self.smpl_params:
                is_shared_beta = key == 'betas' and self.smpl_params['betas'].shape[0] == 1
                is_shared_vshape = key in ('v_shape', 'v_pose') and self.smpl_params[key].ndim == 3 and self.smpl_params[key].shape[0] == 1
                param_val = self.smpl_params[key][0] if (is_shared_beta or is_shared_vshape) else self.smpl_params[key][frame_map_idx]
                raw_smpl_params_for_frame[key] = param_val.clone()
        mesh_output['smplx_params_raw_for_deformnet'] = raw_smpl_params_for_frame
        return mesh_output

    def __getitem__(self, idx):
        if idx is None or idx >= len(self.data_samples): idx = torch.randint(0, len(self.data_samples), (1,)).item()
        frame_id_str, camera_id = self.data_samples[idx]

        # Resolve per-item image/mask paths (custom or default)
        if self.custom_rgb_relpath:
            img_rel = self.custom_rgb_relpath.format(frame=frame_id_str, cam=camera_id)
            image_path = self.dat_dir / camera_id / img_rel
        else:
            image_path = self.dat_dir / camera_id / f"{frame_id_str}.jpg"
        img_pil = Image.open(image_path).convert("RGB")
        img_bgr_np = np.array(img_pil)[:, :, ::-1].copy() 
        if self.custom_mask_relpath:
            mask_rel = self.custom_mask_relpath.format(frame=frame_id_str, cam=camera_id)
            mask_path = self.dat_dir / camera_id / mask_rel
        else:
            # When fast_sample_discovery is enabled, mask_ext may not have been auto-detected.
            # Try jpg->png (and cache the successful extension) to avoid repeated FS probes.
            base = self.dat_dir / camera_id / 'mask' / 'pha'
            candidates = []
            # prefer currently configured extension first
            if self.mask_ext:
                candidates.append(f"{frame_id_str}.{self.mask_ext}")
            for ext in ['jpg', 'png']:
                if ext != self.mask_ext:
                    candidates.append(f"{frame_id_str}.{ext}")
            found = None
            for fname in candidates:
                p = base / fname
                if p.exists():
                    found = p
                    # cache the extension for future items if not from custom_relpath
                    self.mask_ext = fname.split('.')[-1]
                    break
            mask_path = found if found is not None else (base / f"{frame_id_str}.{self.mask_ext}")
        mask_pil = Image.open(mask_path).convert("L") 
        mask_np = np.array(mask_pil)
        if img_bgr_np.shape[:2] != mask_np.shape: mask_np = cv2.resize(mask_np, (img_bgr_np.shape[1], img_bgr_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        image_bgra_np = np.concatenate([img_bgr_np, mask_np[..., None]], axis=-1)

        cam_p = self.all_camera_params[camera_id]
        cam_obj = libcore.Camera(); cam_obj.h,cam_obj.w=cam_p['height'],cam_p['width']; cam_obj.fx,cam_obj.fy,cam_obj.cx,cam_obj.cy=cam_p['K'][0,0],cam_p['K'][1,1],cam_p['K'][0,2],cam_p['K'][1,2]; cam_obj.R_w2c_direct,cam_obj.T_w2c_direct=cam_p['R_w2c'],cam_p['T_w2c'].flatten(); cam_obj.R,cam_obj.c=np.identity(3),np.zeros(3)
        dv = libcore.DataVec(); dv.cams,dv.frames,dv.images_path=[cam_obj],[image_bgra_np],[str(image_path)]
        scene_cams = convert_to_scene_cameras(dv, self.dataset_config)
        
        batch = {'idx':idx, 'frm_idx_str':frame_id_str, 'frm_idx':int(frame_id_str), 'cam_id_str':camera_id, 'scene_cameras':scene_cams, 'cameras_extent':self.cameras_extent, 'image_path':str(image_path), 'gt_image_bgra':torch.from_numpy(image_bgra_np).float()/255.0}
        if self.use_deformation and self.ref_frame_garment_mask is not None: batch['garment_mask_cano'] = self.ref_frame_garment_mask.clone()

        mesh_dev = self.dataset_config.get('data_device', 'cpu')
        if self.free_gaussians:
            batch['mesh_info'] = None
        else:
            try:
                frame_map_idx = self.frame_id_map.get(frame_id_str)
                if frame_map_idx is None: raise ValueError(f"Frame '{frame_id_str}' not in map for split.")
                batch['mesh_info'] = self.get_smpl_mesh(frame_map_idx, device=mesh_dev) if self.use_smpl_params else self.get_mesh_data(frame_id_str, device=mesh_dev)
            except Exception as e:
                print(f"Error mesh frame {frame_id_str} (map_idx {frame_map_idx if frame_map_idx is not None else 'N/A'}): {e}"); batch['mesh_info']=None
        return batch

    def __len__(self):
        return len(self.data_samples)

if __name__ == '__main__':
    # Example Usage (requires a dummy config and data structure)
    print("AvatarRexDataset Example Usage:")
    dummy_garment_label_path = "dummy_garment_labels.pkl" # Create a dummy for testing
    dummy_smpl_model_path = "dummy_smpl_models" # Create a dummy dir for testing
    
    # Create dummy garment label file
    if not os.path.exists(dummy_garment_label_path):
        try:
            # Assuming 100 vertices for the dummy mesh
            dummy_labels = {'scan_labels': np.random.randint(0, 6, size=(100,))} 
            with open(dummy_garment_label_path, 'wb') as f:
                pickle.dump(dummy_labels, f)
            print(f"Created dummy garment label file: {dummy_garment_label_path}")
        except Exception as e:
            print(f"Could not create dummy garment file: {e}")

    os.makedirs(dummy_smpl_model_path, exist_ok=True) # For smplx_utils

    # --- Test Case 1: Loading OBJ (less relevant now but kept for structure) ---
    # ... (OBJ test case might be simplified or removed if SMPL is mandatory)

    # --- Test Case 2: Loading SMPL ---
    print("\n--- Testing SMPL Loading ---")
    # Create a dummy smpl_params.npz
    dummy_smpl_npz_path = "dummy_smpl_params.npz"
    if not os.path.exists(dummy_smpl_npz_path):
        try:
            num_test_frames = 5
            smpl_test_data = {
                "betas": np.random.rand(1, 10).astype(np.float32), # Shared betas
                "body_pose": np.random.rand(num_test_frames, 63).astype(np.float32), # 21 joints * 3
                "global_orient": np.random.rand(num_test_frames, 3).astype(np.float32),
                "transl": np.random.rand(num_test_frames, 3).astype(np.float32),
                # Add other minimal keys if smplx_utils requires them (e.g., gender for model loading path)
            }
            np.savez(dummy_smpl_npz_path, **smpl_test_data)
            print(f"Created dummy SMPL NPZ: {dummy_smpl_npz_path}")
        except Exception as e:
            print(f"Could not create dummy SMPL NPZ: {e}")


    cfg_smpl = {
        'dat_dir': ".", # Use current dir for dummy files
        'cameras_extent': 1.0,
        'data_device': 'cpu',
        'avatarrex_config': {
            'use_smpl_params': True, 
            'model_type': 'smpl', # Test with smpl, as smplx might need more files
            'gender': 'neutral',        
            'train_frames': {"start_idx":0, "end_idx": 4, "step_idx":1, "frame_format_digits":1}, # Frames "0" to "4"
            'train_cameras': ["cam1"], # Dummy camera
            'garment_label_file': dummy_garment_label_path, # Use dummy garment file
            'smpl_model_path': dummy_smpl_model_path, # Dummy model path
        }
    }
    
    # Create dummy camera and image/mask files for the test
    dummy_data_dir = Path(cfg_smpl['dat_dir'])
    (dummy_data_dir / "cam1" / "mask" / "pha").mkdir(parents=True, exist_ok=True)
    
    # Dummy calibration
    dummy_calib = {"cam1": {"K": [[100,0,50],[0,100,50],[0,0,1]], "R": [[1,0,0],[0,1,0],[0,0,1]], "T":[[0],[0],[0]], "imgSize": [100,100]}}
    with open(dummy_data_dir / "calibration_full.json", 'w') as f: json.dump(dummy_calib, f)

    for i in range(5): # For frames "0" to "4"
        Image.new('RGB', (100,100)).save(dummy_data_dir / "cam1" / f"{i}.jpg")
        Image.new('L', (100,100)).save(dummy_data_dir / "cam1" / "mask" / "pha" / f"{i}.jpg")
    print("Created dummy image/mask files for cam1, frames 0-4.")


    print(f"Attempting to load SMPL dataset from: {cfg_smpl['dat_dir']}")
    try:
        train_dataset_smpl = AvatarRexDataset(config=cfg_smpl, split='train')
        
        if len(train_dataset_smpl) > 0:
            print(f"SMPL Mode: Successfully created dataset with {len(train_dataset_smpl)} samples.")
            sample_item = train_dataset_smpl[0]
            print("Sample item keys:", sample_item.keys())
            if sample_item.get('mesh_info'):
                print("Mesh Verts shape:", sample_item['mesh_info']['mesh_verts'].shape)
                if 'smplx_params_raw_for_deformnet' in sample_item['mesh_info']:
                    print("SMPLX raw params for deformnet found. Keys:", sample_item['mesh_info']['smplx_params_raw_for_deformnet'].keys())
            if sample_item.get('garment_mask_cano') is not None:
                print("Canonical garment mask loaded, shape:", sample_item['garment_mask_cano'].shape, "Num garment verts:", sample_item['garment_mask_cano'].sum())

        else:
            print("SMPL Mode: Dataset created, but no samples were loaded. Check dummy file setup and config.")

    except Exception as e:
        import traceback
        print(f"SMPL Mode: An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Clean up dummy files
        if os.path.exists(dummy_garment_label_path): os.remove(dummy_garment_label_path)
        if os.path.exists(dummy_smpl_npz_path): os.remove(dummy_smpl_npz_path)
        # Could add more cleanup for dummy images/dirs if needed
        print("Cleaned up dummy files.")