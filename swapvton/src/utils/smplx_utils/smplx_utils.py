# SMPL-X helper.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os, copy
from pathlib import Path
import torch
import cv2
import numpy as np
from collections import namedtuple
from pytorch3d.transforms import axis_angle_to_matrix

from . import smplx
from . import smplx_ani
from loguru import logger


AVATARREX_SMPLX_CONFIG = {
    'use_pca': False,
    'num_pca_comps': 45,
    'flat_hand_mean': True,
}

THUMAN2_SMPLX_CONFIG = {
    'use_pca': True,
    'num_pca_comps': 12,
    'flat_hand_mean': False,
}

ACTORSHQ_SMPLX_CONFIG = {
    'use_pca': True,
    'num_pca_comps': 6,
    'flat_hand_mean': True,
}

def create_smplx_model_for_neus2_reposing(
    smpl_params=None,
    batch_size=1,
    device='cpu',
    gender='neutral',
    dataset_type='thuman2',
    model_path: str | None = None,
):

    if dataset_type == 'thuman2':
        dataset_dependent_config = THUMAN2_SMPLX_CONFIG
    elif dataset_type == 'avatarrex':
        dataset_dependent_config = AVATARREX_SMPLX_CONFIG
    elif dataset_type == 'actorshq' or dataset_type == 'mvhumannet':
        dataset_dependent_config = ACTORSHQ_SMPLX_CONFIG
    elif dataset_type == 'talkbody4d':

        dataset_dependent_config = AVATARREX_SMPLX_CONFIG
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    smpl_config = {
        'model_type': 'smplx',
        'gender': gender,
        'ext': 'npz',
        'use_pca': dataset_dependent_config.get('use_pca', False),
        'num_pca_comps': dataset_dependent_config.get('num_pca_comps', 6),
        'flat_hand_mean': dataset_dependent_config.get('flat_hand_mean', False),
        'batch_size': batch_size,
        'num_betas': 10,
    }


    if smpl_params is not None and 'betas' in smpl_params:
        smpl_config['num_betas'] = smpl_params['betas'].shape[-1]
        logger.info(f"Auto-detected num_betas: {smpl_config['num_betas']}")


    if smpl_params is not None and 'expression' in smpl_params:
        smpl_config['num_expression_coeffs'] = smpl_params['expression'].shape[-1]
        logger.info(f"Auto-detected num_expression_coeffs: {smpl_config['num_expression_coeffs']}")


    if smpl_params is not None and ('left_hand_pose' in smpl_params or 'right_hand_pose' in smpl_params):
        lh = smpl_params.get('left_hand_pose', None)
        rh = smpl_params.get('right_hand_pose', None)
        lh_dim = int(lh.shape[-1]) if lh is not None and hasattr(lh, 'shape') else None
        rh_dim = int(rh.shape[-1]) if rh is not None and hasattr(rh, 'shape') else None
        dims = [d for d in [lh_dim, rh_dim] if d is not None]
        if dims:

            hand_dim = int(max(dims))
            if hand_dim == 45:
                if bool(smpl_config.get('use_pca', False)):
                    logger.warning("Detected 45D hand poses in smpl_params; disabling PCA hands (use_pca=False).")
                smpl_config['use_pca'] = False
            else:
                if bool(smpl_config.get('use_pca', False)):
                    if int(smpl_config.get('num_pca_comps', hand_dim)) != hand_dim:
                        logger.warning(
                            f"Overriding num_pca_comps={smpl_config.get('num_pca_comps')} -> {hand_dim} "
                            f"to match hand pose dim in smpl_params."
                        )
                    smpl_config['num_pca_comps'] = hand_dim


    for key_suffix in ['jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'left_hand_pose', 'right_hand_pose']:
        if smpl_params is not None and key_suffix in smpl_params:
            smpl_config[f'create_{key_suffix}'] = False


    if model_path is not None:
        smpl_config["model_path"] = model_path
    return create_smplx_model(**smpl_config, implementation='smplx_ani').to(device)


def compute_reference_transform_data(ref_smpl_params, smpl_model_for_reposing, return_vertices=False):

    with torch.no_grad():

        ref_global_orient = ref_smpl_params['global_orient']
        ref_transl = ref_smpl_params['transl']
        ref_scale = ref_smpl_params.get('scale', torch.ones(1, device=ref_global_orient.device))


        ref_params_canonical = {k: v for k, v in ref_smpl_params.items()
                                if k not in ['global_orient', 'transl', 'scale']}
        ref_params_canonical = {k: v.squeeze(0) if v.ndim == 1 else v for k, v in ref_params_canonical.items()}

        ref_smpl_output = smpl_model_for_reposing.forward(**ref_params_canonical, return_verts=return_vertices)
        j0 = ref_smpl_output.joints[0, 0]
        inv_ref_pose_jnt_mats = torch.linalg.inv(ref_smpl_output.A)

        global_orient_matrix = axis_angle_to_matrix(ref_global_orient)

        result = {
            'inv_ref_pose_jnt_mats': inv_ref_pose_jnt_mats,
            'global_orient_matrix': global_orient_matrix,
            'ref_transl': ref_transl,
            'ref_scale': ref_scale,
            'j0': j0,
            'ref_params_canonical': ref_params_canonical
        }


        if return_vertices and hasattr(ref_smpl_output, 'vertices'):
            result['vertices'] = ref_smpl_output.vertices.squeeze(0)

        return result

def get_smplx_model_path(model_type='smplx', fn=None):
    root = Path(os.environ.get("AVATARMIX_ASSET_ROOT", str(Path(__file__).resolve().parents[4] / "external_assets"))) / "smpl"
    if fn is None:
        return str(root)
    else:
        return str(root / model_type / fn)

def create_smplx_model(model_path=None, gender='neutral', model_type='smplx', ext='npz',
                       skip_betas=False, skip_v_template=False, skip_poses=True, implementation='smplx',
                       **smplx_params):
    if model_path is None:
        model_path = get_smplx_model_path()
    elif not os.path.exists(model_path):
        model_path = get_smplx_model_path(model_type=model_type, fn=model_path)

    if skip_betas:
        if 'betas' in smplx_params:
            smplx_params.pop('betas')

    if skip_v_template:
        if 'v_template' in smplx_params:
            smplx_params.pop('v_template')

    if skip_poses:
        keys = [key for key in smplx_params]
        for key in keys:
            if isinstance(smplx_params[key], torch.Tensor):
                if key != 'betas' and key != 'v_template':
                    smplx_params.pop(key)

    smplx_impl = smplx_ani if implementation == 'smplx_ani' else smplx
    smplx_model = smplx_impl.create(model_path, gender=gender, model_type=model_type, ext=ext,
                               **smplx_params)
    return smplx_model

def create_smplx_lite_model(model_path=None, gender='male', model_type='smplx-lite', ext='pkl',
                       **smplx_params):
    if model_path is None:
        model_path = get_smplx_model_path()
    elif not os.path.exists(model_path):
        model_path = get_smplx_model_path(model_type=model_type, fn=model_path)

    model_path = os.path.join(model_path, model_type, 'SMPLX-LITE_{}.{ext}'.format(gender.upper(), ext=ext))

    smplx_model = smplx.create(model_path, gender=gender, model_type=model_type, ext=ext,
                               **smplx_params)
    return smplx_model

def load_regressor(regressor_path):
    if regressor_path.endswith('.npy'):
        X_regressor = torch.tensor(np.load(regressor_path)).float()
    elif regressor_path.endswith('.txt'):
        data = np.loadtxt(regressor_path)
        with open(regressor_path, 'r') as f:
            shape = f.readline().split()[1:]
        reg = np.zeros((int(shape[0]), int(shape[1])))
        for i, j, v in data:
            reg[int(i), int(j)] = v
        X_regressor = torch.tensor(reg).float()
    else:
        raise ValueError(f"Unsupported regressor format: {regressor_path}; expected .npy or .txt")
    return X_regressor

def load_smplx_J_regressor_body25_smplx(fn='J_regressor_body25_smplx.txt'):
    if not os.path.isabs(fn):
        fn = get_smplx_model_path(model_type='', fn=fn)
    return load_regressor(fn)

def load_smplx_J_regressor_body25_smplx_lite(fn='J_regressor_body25_smplx_lite.txt'):
    if not os.path.isabs(fn):
        fn = get_smplx_model_path(model_type='', fn=fn)
    return load_regressor(fn)

def write_J_regressor(fn, J_regressor):
    with open(fn, 'w') as fp:
        fp.write(f'# {J_regressor.shape[0]} {J_regressor.shape[1]}\n')
        for i in range(J_regressor.shape[0]):
            for j in range(J_regressor.shape[1]):
                if J_regressor[i, j] != 0:
                    fp.write(f'{i} {j} {J_regressor[i, j]}\n')

def load_smplx_part_labels(gender='male', model_type='smplx', model_path=None):
    assert gender in ['male', 'female']

    if model_path is None:
        model_path = get_smplx_model_path()


    import pickle as pkl
    verts_ids = pkl.load(open(os.path.join(model_path, model_type, f'non_watertight_{gender}_vertex_labels.pkl'), 'rb'),
                         encoding='latin1')
    return verts_ids
'''
def convert_smplx_to_meshcpu(smplx_model, V=None):
    if V is None:
        V = smplx_model.v_template
    if isinstance(V, torch.Tensor):
        V = V.detach().cpu().numpy()
 
    mesh = libcore.MeshCpu()
    mesh.V = V
    mesh.F = smplx_model.faces.astype(int)
    mesh.update_per_vertex_normals()
    if hasattr(smplx_model, 'tc'):
        mesh.TC = smplx_model.tc
        mesh.FTC = smplx_model.tc_faces
    return mesh

def save_smplx_to_obj(fn, smplx_model, V=None):
    mesh = convert_smplx_to_meshcpu(smplx_model, V=V)
    mesh.save_to_obj(fn)

def write_smplx_objs(smplx_dir, frm_list, smplx_model, out, max_workers=8):
    def _write_smplx_obj(idx):
        frm_idx = frm_list[idx]
        mesh = convert_smplx_to_meshcpu(smplx_model, V=out['vertices'][idx])
        mesh.save_to_obj(os.path.join(smplx_dir, f'smplx_{frm_idx:06d}.obj'))

    num_frames = len(frm_list)
    idxs = [i for i in range(num_frames)]

    import concurrent.futures
    from tqdm import tqdm
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executrer:
        for res in tqdm(executrer.map(_write_smplx_obj, idxs), total=num_frames):
            pass
'''
def convert_smplx_params_cv2gl(smplx_params):

    smplx_model = create_smplx_model(**smplx_params)


    out = smplx_model(**smplx_params)
    verts_gl = out['vertices'].detach().clone()
    verts_gl[0, :, 1:3] = -verts_gl[0, :, 1:3]


    rvec = smplx_params['poses'][0, :3].numpy()
    smplx_R = cv2.Rodrigues(rvec)[0]
    smplx_R_gl = copy.deepcopy(smplx_R)
    smplx_R_gl[1:3, :] = -smplx_R_gl[1:3, :]
    rvec_gl = cv2.Rodrigues(smplx_R_gl)[0]


    smplx_params_gl = copy.deepcopy(smplx_params)
    smplx_params_gl['poses'][0, :3] = torch.from_numpy(rvec_gl).view(-1)
    smplx_params_gl['transl'] = torch.zeros_like(smplx_params_gl['transl'])
    out = smplx_model(**smplx_params_gl)

    smplx_t_gl = (verts_gl - out['vertices']).mean(dim=-2)
    smplx_params_gl['transl'] = smplx_t_gl

    return smplx_params_gl


def load_and_detach(fn, map_location='cpu'):
    smplx_params = torch.load(fn, map_location=map_location)
    for key in smplx_params:
        if isinstance(smplx_params[key], torch.Tensor):
            smplx_params[key] = smplx_params[key].detach()

            if len(smplx_params[key].shape) == 1:
                smplx_params[key] = smplx_params[key].unsqueeze(0)

    if 'use_pca' not in smplx_params:
        smplx_params['use_pca'] = True
    if 'flat_hand_mean' not in smplx_params:
        smplx_params['flat_hand_mean'] = True
    if 'num_betas' not in smplx_params and 'betas' in smplx_params:
        smplx_params['num_betas'] = smplx_params['betas'].shape[-1]
    if 'num_expression_coeffs' not in smplx_params and 'expression' in smplx_params:
        smplx_params['num_expression_coeffs'] = smplx_params['expression'].shape[-1]
    if 'gender' not in smplx_params:
        smplx_params['gender'] = 'male'
    if 'model_type' not in smplx_params:
        smplx_params['model_type'] = 'smplx'

    return smplx_params
