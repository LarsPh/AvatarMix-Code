import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
import pytorch3d.io
import pytorch3d.structures

# Assuming these are available in the project structure
from model import libcore # Contains Camera and DataVec
from scene.dataset_readers import convert_to_scene_cameras # Converts DataVec to SceneCamera
from model.smplx_utils import smplx_utils

class THuman2Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train'):
        self.config = config 
        self.dataset_config = config # Assuming config passed IS the dataset sub-config
        self.dat_dir = Path(self.dataset_config.dat_dir)
        self.split = split # Note: THuman2 is single frame, split might apply to camera views or subjects
        self.cameras_extent = self.dataset_config.get('cameras_extent', 1.0)
        self.data_device = self.dataset_config.get('data_device', 'cpu')
        self.img_wh = None # Will be set from first loaded image or config

        self.thuman2_conf = self.dataset_config.get('thuman2_config', {})
        self.subject_ids_to_load = self.thuman2_conf.get('subject_ids', [])
        if isinstance(self.subject_ids_to_load, str): # Handle single subject_id string
            self.subject_ids_to_load = [self.subject_ids_to_load]
            
        self.use_smplx_params = self.thuman2_conf.get('use_smplx_params', False)
        self.smplx_gender = self.thuman2_conf.get('smplx_gender', 'male')
        self.smplx_model_path = self.thuman2_conf.get('smplx_model_path', 'data/smplx_models')
        self.smplx_num_pca_comps = self.thuman2_conf.get('smplx_num_pca_comps', 12)
        # SMPL-X model creation flags (all False as per user prompt)
        self.smplx_create_flags = {
            'create_global_orient': False, 'create_body_pose': False, 'create_betas': False,
            'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_expression': False,
            'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False,
            'create_transl': False
        }

        self.use_neus_mesh = self.thuman2_conf.get('use_neus_mesh', False)
        self.neus_mesh_dir_name = self.thuman2_conf.get('neus_mesh_dir_name', 'neus_mesh') # Relative to dat_dir

        self.all_camera_params = {} # Key: (subject_id_str, camera_name_str)
        self.data_samples = []      # List of (subject_id_str, camera_name_str) tuples
        
        self.smplx_params_per_subject = {} # Key: subject_id_str, Val: dict of smplx params
        self.smplx_models_per_subject = {} # Key: subject_id_str, Val: smplx model instance

        self.load_data_samples()

        if self.use_smplx_params:
            self.load_smplx_data_for_subjects()

        if not self.data_samples:
            print(f"[THuman2Dataset][{self.split}] No samples found for subjects {self.subject_ids_to_load}. Check config and data paths at {self.dat_dir}.")
        else:
            print(f'[THuman2Dataset][{self.split}] Found {len(self.data_samples)} samples for subjects {self.subject_ids_to_load}.')

    def _generate_camera_names(self):
        camera_names = []
        # Pitch +00: yaw 0 to 350, step 10 (36 views)
        for yaw in range(0, 360, 10):
            camera_names.append(f"{yaw:03d}_p+00")
        # Pitch +20: yaw 5 to 345, step 20 (18 views)
        for yaw in range(5, 360, 20): # Corrected range based on 5 to 345
             if yaw <= 345:
                camera_names.append(f"{yaw:03d}_p+20")
        # Pitch -20: yaw 15 to 355, step 20 (18 views)
        for yaw in range(15, 360, 20): # Corrected range based on 15 to 355
            if yaw <= 355:
                camera_names.append(f"{yaw:03d}_p-20")
        return camera_names

    def _parse_calib_file(self, calib_path, img_w, img_h):
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        
        with open(calib_path, 'r') as f:
            lines = f.readlines()

        extr_lines = [list(map(float, line.strip().split())) for line in lines[0:4]]
        intr_lines = [list(map(float, line.strip().split())) for line in lines[4:8]]

        E = np.array(extr_lines)
        R_w2c = E[:3, :3]
        T_w2c = E[:3, 3].reshape(3, 1)

        K_mat_from_file = np.array(intr_lines)[:3,:3]
        
        # K usually has fx, 0, cx; 0, fy, cy; 0, 0, 1
        # If K_mat_from_file cx, cy are absolute pixel values, use them.
        # If they are normalized or something else, adjust accordingly.
        # From example: 1.44e3 0 512; 0 1.44e3 512; 0 0 1. Assuming 512 is cx/cy for 1024 width/height.
        fx = K_mat_from_file[0,0]
        fy = K_mat_from_file[1,1]
        # If K_mat_from_file[0,2] and K_mat_from_file[1,2] are already cx, cy:
        cx = K_mat_from_file[0,2]
        cy = K_mat_from_file[1,2]
        # If img_w, img_h are known and cx, cy from file are e.g. principal point for that size
        # then K is fine. If cx, cy are e.g. 0.5 for normalized, then multiply by img_w/img_h.
        # Given the file format, K_mat_from_file is likely the direct K.
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        return K, R_w2c, T_w2c


    def load_data_samples(self):
        camera_names_all_potential = self._generate_camera_names()
        
        config_img_wh = self.thuman2_conf.get('img_wh', None)

        for subject_id_orig in self.subject_ids_to_load:
            subject_id_str = f"{int(subject_id_orig):04d}" # Pad with 4 zeros
            subject_render_dir = self.dat_dir / "render" / subject_id_str
            
            if not subject_render_dir.exists():
                print(f"Warning: Subject render directory not found: {subject_render_dir}")
                continue

            for cam_name in camera_names_all_potential:
                img_path = subject_render_dir / "rgba" / f"{cam_name}.png"
                calib_path = subject_render_dir / "calib" / f"{cam_name}.txt"
                
                neus_mesh_check = True # Assume exists if not used, or handle path check
                if self.use_neus_mesh:
                    # Assumed path: <dat_dir>/<neus_mesh_dir_name>/<subject_id_str>/mesh.obj
                    neus_path = self.dat_dir / self.neus_mesh_dir_name / subject_id_str / "mesh.obj"
                    neus_mesh_check = neus_path.exists()
                    if not neus_mesh_check:
                         print(f"Warning: NeuS mesh not found for subject {subject_id_str} at {neus_path}")


                if img_path.exists() and calib_path.exists() and neus_mesh_check:
                    try:
                        if self.img_wh is None: # Set from first image or config
                            if config_img_wh:
                                self.img_wh = (int(config_img_wh[0]), int(config_img_wh[1]))
                            else:
                                temp_img = Image.open(img_path)
                                self.img_wh = temp_img.size # (width, height)
                        
                        K, R_w2c, T_w2c = self._parse_calib_file(calib_path, self.img_wh[0], self.img_wh[1])
                        self.all_camera_params[(subject_id_str, cam_name)] = {
                            'K': K, 'R_w2c': R_w2c, 'T_w2c': T_w2c,
                            'width': self.img_wh[0], 'height': self.img_wh[1]
                        }
                        self.data_samples.append((subject_id_str, cam_name))
                    except Exception as e:
                        print(f"Error processing sample (subj: {subject_id_str}, cam: {cam_name}): {e}")
                # else:
                #     print(f"Skipping sample: Subj {subject_id_str}, Cam {cam_name}. Missing Img={img_path.exists()}, Calib={calib_path.exists()}, NeusMeshOK={neus_mesh_check}")

        self.data_samples = sorted(list(set(self.data_samples)))


    def load_smplx_data_for_subjects(self):
        unique_subject_ids = sorted(list(set([s_id for s_id, _ in self.data_samples])))
        
        for subject_id_str in unique_subject_ids:
            smplx_pkl_path = self.dat_dir / "smplx" / f"{subject_id_str}.pkl"
            if not smplx_pkl_path.exists():
                print(f"Warning: SMPLX PKL file not found for subject {subject_id_str} at {smplx_pkl_path}")
                continue
            
            try:
                with open(smplx_pkl_path, 'rb') as f:
                    raw_params = pickle.load(f, encoding='latin1') # Some pkls might need encoding

                # Convert relevant parameters to tensors
                # Based on user prompt and common SMPL-X parameters
                self.smplx_params_per_subject[subject_id_str] = {
                    "betas": torch.tensor(raw_params["betas"].astype(np.float32)),
                    "body_pose": torch.tensor(raw_params["body_pose"].astype(np.float32)),
                    "global_orient": torch.tensor(raw_params["global_orient"].astype(np.float32)),
                    "transl": torch.tensor(raw_params["transl"].astype(np.float32)),
                    "left_hand_pose": torch.tensor(raw_params["left_hand_pose"].astype(np.float32)),
                    "right_hand_pose": torch.tensor(raw_params["right_hand_pose"].astype(np.float32)),
                    "jaw_pose": torch.tensor(raw_params["jaw_pose"].astype(np.float32)),
                    "leye_pose": torch.tensor(raw_params["leye_pose"].astype(np.float32)),
                    "reye_pose": torch.tensor(raw_params["reye_pose"].astype(np.float32)),
                    "expression": torch.tensor(raw_params["expression"].astype(np.float32)),
                }
                
                # Ensure batch dimension for model input (batch_size=1)
                for k, v in self.smplx_params_per_subject[subject_id_str].items():
                    if v.ndim == 1: # e.g. betas [10] -> [1, 10]
                        self.smplx_params_per_subject[subject_id_str][k] = v.unsqueeze(0)
                    elif v.ndim == 2 and k not in ["betas", "body_pose", "left_hand_pose", "right_hand_pose"]: # e.g. global_orient [3,3] -> [1,3,3]
                         self.smplx_params_per_subject[subject_id_str][k] = v.unsqueeze(0)
                    # body_pose is [1, 63], hand_poses [1, 45] or [1, N_pca*3], betas [1,10]
                    # expression [1,10], jaw_pose [1,3], leye_pose [1,3], reye_pose [1,3], transl [1,3]
                    # global_orient [1,3] (axis-angle) or [1,3,3] (rotmat)
                    # The create_smplx_model expects axis-angle for poses by default.
                    # The pkl might store them as axis-angle [1,3] or [1,N*3]. If it's [3,3] for global_orient, convert.
                    # Assuming pkl stores them in the format expected by smplx model (axis-angle)

                # Create SMPL-X model for this subject (can be optimized if gender/model_path is same)
                # For simplicity, creating one per subject now.
                model_init_params = dict(
                    gender=self.smplx_gender, # Could be subject-specific if data provides it
                    model_type='smplx',
                    model_path=self.smplx_model_path,
                    num_pca_comps=self.smplx_num_pca_comps,
                    batch_size=1, # Single subject pose
                    **self.smplx_create_flags 
                )
                # Add num_betas based on loaded data
                if "betas" in self.smplx_params_per_subject[subject_id_str]:
                    model_init_params['num_betas'] = self.smplx_params_per_subject[subject_id_str]['betas'].shape[-1]

                self.smplx_models_per_subject[subject_id_str] = smplx_utils.create_smplx_model(**model_init_params)
                print(f"Loaded SMPL-X params and created model for subject {subject_id_str}")

            except Exception as e:
                print(f"Error loading SMPL-X params for subject {subject_id_str}: {e}")
                if subject_id_str in self.smplx_params_per_subject:
                    del self.smplx_params_per_subject[subject_id_str]


    def get_smplx_mesh_data(self, subject_id_str, device='cpu'):
        if subject_id_str not in self.smplx_params_per_subject or \
           subject_id_str not in self.smplx_models_per_subject:
            raise RuntimeError(f"SMPL-X data or model not loaded for subject {subject_id_str}.")

        smpl_model = self.smplx_models_per_subject[subject_id_str].to(device)
        
        # Prepare parameters for model forward pass
        # Ensure all params are on the correct device and have batch_size=1
        model_input_params = {}
        subject_params = self.smplx_params_per_subject[subject_id_str]

        for k, v_tensor in subject_params.items():
            model_input_params[k] = v_tensor.to(device)
            if model_input_params[k].shape[0] != 1 : # Ensure batch dim is 1
                 if model_input_params[k].ndim == v_tensor.ndim -1 and v_tensor.shape[0] == 1: # Was squeezed, re-add
                      model_input_params[k] = model_input_params[k].unsqueeze(0)
                 elif v_tensor.ndim > 0: # Try to make it [1, ...]
                      model_input_params[k] = v_tensor.reshape(1, -1) if k != 'global_orient' else v_tensor.reshape(1,3) # common cases
                 # A more robust reshaping might be needed depending on exact pkl content and model expectations
        
        # Handle specific parameter name mappings if model expects different keys
        # e.g. if pkl has 'pose' but model wants 'body_pose'

        with torch.no_grad():
            smpl_output = smpl_model(**model_input_params, return_verts=True)
            verts = smpl_output.vertices.squeeze(0).detach() # [N, 3]
            faces = torch.tensor(smpl_model.faces.astype(np.int64), dtype=torch.int64, device=device)

        # Create PyTorch3D mesh for normals calculation
        frame_mesh_py3d = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
        
        return {
            'mesh_verts': verts, 
            'mesh_norms': frame_mesh_py3d.verts_normals_packed(), 
            'mesh_faces': faces,
        }

    def get_neus_mesh_data(self, subject_id_str, device='cpu'):
        neus_mesh_path = self.dat_dir / self.neus_mesh_dir_name / subject_id_str / "mesh.obj"
        if not neus_mesh_path.exists():
            raise FileNotFoundError(f"NeuS mesh file not found: {neus_mesh_path}")

        verts, faces_idx, _ = pytorch3d.io.load_obj(str(neus_mesh_path), device=device)
        faces = faces_idx.verts_idx
        
        frame_mesh_py3d = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
        
        return {
            'mesh_verts': verts, 
            'mesh_norms': frame_mesh_py3d.verts_normals_packed(), 
            'mesh_faces': faces, 
        }

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        if idx is None or idx >= len(self.data_samples):
            # Default to a random sample if idx is out of bounds, or handle error
            idx = torch.randint(0, len(self.data_samples), (1,)).item()

        subject_id_str, camera_name_str = self.data_samples[idx]

        # --- Load Image (RGBA) ---
        img_path = self.dat_dir / "render" / subject_id_str / "rgba" / f"{camera_name_str}.png"
        # PIL loads as RGBA if alpha channel exists
        img_pil_rgba = Image.open(img_path).convert("RGBA") 
        img_rgba_np = np.array(img_pil_rgba) # H, W, 4 (R,G,B,A)

        # Convert RGBA to BGRA for consistency if downstream expects BGR base
        img_bgra_np = img_rgba_np[..., [2,1,0,3]].copy() # B,G,R,A

        if self.img_wh is None: # Should have been set in load_data_samples
            self.img_wh = (img_bgra_np.shape[1], img_bgra_np.shape[0])


        # --- Camera parameters ---
        cam_p = self.all_camera_params[(subject_id_str, camera_name_str)]
        K = cam_p['K']
        R_w2c = cam_p['R_w2c']
        T_w2c = cam_p['T_w2c']
        
        cam_obj = libcore.Camera()
        cam_obj.h, cam_obj.w = cam_p['height'], cam_p['width']
        cam_obj.fx, cam_obj.fy = K[0,0], K[1,1]
        cam_obj.cx, cam_obj.cy = K[0,2], K[1,2]
        cam_obj.R_w2c_direct = R_w2c
        cam_obj.T_w2c_direct = T_w2c.flatten()
        cam_obj.R = np.identity(3) # World-to-camera is R_w2c_direct
        cam_obj.c = np.zeros(3)    # Camera center in world is -(R_w2c_direct.T @ T_w2c_direct)

        # --- Create DataVec and SceneCamera ---
        # Assuming DataVec expects BGRA numpy arrays
        color_frames_dv = libcore.DataVec()
        color_frames_dv.cams = [cam_obj]
        color_frames_dv.frames = [img_bgra_np] # BGRA uint8
        color_frames_dv.images_path = [str(img_path)]
        
        # convert_to_scene_cameras might take the dataset_config part
        scene_cameras = convert_to_scene_cameras(color_frames_dv, self.dataset_config) 
        
        batch = {
            'idx': idx,
            'subj_id_str': subject_id_str,
            'cam_id_str': camera_name_str,
            'color_frames': color_frames_dv, 
            'scene_cameras': scene_cameras,
            'cameras_extent': self.cameras_extent,
            'image_path': str(img_path),
            'gt_image_bgra': torch.from_numpy(img_bgra_np).float() / 255.0 # HWC, BGRA, [0,1]
        }

        # --- Mesh data ---
        mesh_load_device = self.data_device
        try:
            if self.use_smplx_params:
                 mesh_data = self.get_smplx_mesh_data(subject_id_str, device=mesh_load_device)
                 batch['mesh_info'] = mesh_data
            elif self.use_neus_mesh:
                 mesh_data = self.get_neus_mesh_data(subject_id_str, device=mesh_load_device)
                 batch['mesh_info'] = mesh_data
            else:
                batch['mesh_info'] = None
        except (FileNotFoundError, RuntimeError, KeyError) as e: # Added KeyError for missing smplx params
            print(f"Error loading mesh for subject {subject_id_str}, cam {camera_name_str}: {e}")
            batch['mesh_info'] = None
        
        return batch
