# SplattingAvatar Optimizer.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
import torch
import torch.nn as nn
from utils.general_utils import get_expon_lr_func
from tqdm import tqdm
from .loss_base import LossBase


def _is_cfg_group_active(cfg_group) -> bool:
    if cfg_group is None:
        return False
    try:
        s1 = bool(cfg_group.get("stage1_active", False))
        s2 = bool(cfg_group.get("stage2_active", False))
        base_lr = cfg_group.get("lr", None)
        s1_lr = cfg_group.get("stage1_lr", base_lr if base_lr is not None else 0.0)
        s2_lr = cfg_group.get("stage2_lr", base_lr if base_lr is not None else 0.0)
        return (s1 and float(s1_lr) > 0.0) or (s2 and float(s2_lr) > 0.0)
    except Exception:
        return False


def validate_litept_stage_config(*, litept_stage: str, deform_optim_config, added_litept_groups: dict[str, int]) -> None:
    stage = str(litept_stage).lower().strip()
    if stage not in {"p1", "p2", "ft"}:
        raise ValueError(f"Invalid optim.litept_stage='{litept_stage}'. Expected one of: p1, p2, ft.")
    if stage != "p1":
        return

    required = {
        "optim_litept_trunk_early",
        "optim_litept_trunk_late",
        "optim_litept_decoder",
        "optim_litept_pose_head",
        "optim_litept_gs_alpha_head",
        "optim_litept_gs_logscale_head",
    }
    missing_cfg = [k for k in sorted(required) if deform_optim_config.get(k, None) is None]
    if len(missing_cfg) > 0:
        raise ValueError(
            "LitePT P1 requires explicit deformation_optim groups in YAML. "
            f"Missing: {missing_cfg}"
        )
    empty_runtime = [k for k in sorted(required) if int(added_litept_groups.get(k, 0)) <= 0]
    if len(empty_runtime) > 0:
        raise ValueError(
            "LitePT P1 optimizer groups are configured but no parameters were collected. "
            f"Check backend wiring for: {empty_runtime}"
        )

    forbidden_active = []
    for k in ["optim_litept_interaction_head", "optim_litept_gate_head"]:
        if _is_cfg_group_active(deform_optim_config.get(k, None)):
            forbidden_active.append(k)
    if len(forbidden_active) > 0:
        raise ValueError(
            "LitePT P1 must keep interaction/gate frozen or inactive. "
            f"Disable these groups: {forbidden_active}"
        )

    legacy_forbidden = []
    for k in ["optim_gauss_mlp_pos", "optim_gauss_mlp_color", "optim_blendshapes_pos", "optim_blendshapes_color"]:
        if _is_cfg_group_active(deform_optim_config.get(k, None)):
            legacy_forbidden.append(k)
    if len(legacy_forbidden) > 0:
        raise ValueError(
            "LitePT P1 must not optimize legacy GS blendshape groups. "
            f"Disable/remove: {legacy_forbidden}"
        )


def _make_expon_lr_func(scheduler_args):
    return get_expon_lr_func(lr_init=scheduler_args.lr_init,
                             lr_final=scheduler_args.lr_final,
                             lr_delay_mult=scheduler_args.lr_delay_mult,
                             max_steps=scheduler_args.lr_max_steps)

# standard 3dgs
class SplattingAvatarOptimizer(LossBase):
    def __init__(self, gs_model, optimizer_config=None) -> None:
        super().__init__(gs_model, optimizer_config)
        self.gs_model = gs_model
        self.optimizer = None
        self.smplx_optim = None
        self.schedulers = []
        self.lr_schedulers = {}
        self.current_stage = 0

        if optimizer_config is not None:
            self.setup_optimizer(optimizer_config)

    def setup_optimizer(self, optimizer_config):
        self.optimizer_config = optimizer_config
        model = self.gs_model
        self.lr_schedulers = {}
        param_groups = []

        def _cfg_get(cfg, key: str, default=None):
            if cfg is None:
                return default
            try:
                if isinstance(cfg, dict) or hasattr(cfg, "get"):
                    v = cfg.get(key, default)
                    return default if v is None else v
            except Exception:
                pass
            try:
                v = getattr(cfg, key)
                return default if v is None else v
            except Exception:
                return default

        # --- Stage definitions from config ---
        self.stage1_end_iter = optimizer_config.get("stage1_end_iter", 0) # Default to 0 if not specified
                                                                        # Stage 1: Vert MLP. Stage 2: Gauss MLP + Blendshapes + GS params

        # --- FullbodyFix bake: per-GS-parameter optimization gating ---
        fbf_cfg = _cfg_get(optimizer_config, "fullbodyfix_bake", None)
        fbf_enabled = bool(_cfg_get(fbf_cfg, "enabled", False))
        fbf_opt = _cfg_get(fbf_cfg, "optimize", None) if fbf_enabled else None
        # Defaults (appearance-only) if optimize flags are missing.
        fbf_defaults = {
            "_features_dc": True,
            "_features_rest": True,
            "_opacity": False,
            "_xyz": False,
            "_scaling": False,
            "_rotation": False,
        }
        fbf_flag_keys = {
            "_features_dc": "features_dc",
            "_features_rest": "features_rest",
            "_opacity": "opacity",
            "_xyz": "xyz",
            "_scaling": "scaling",
            "_rotation": "rotation",
        }

        def _fbf_group_enabled(group_name: str) -> bool:
            if not fbf_enabled:
                return True
            key = fbf_flag_keys.get(group_name, None)
            if key is None:
                return True
            return bool(_cfg_get(fbf_opt, key, fbf_defaults.get(group_name, False)))

        # --- Original Gaussian Splatting Parameters ---
        gs_param_configs = [
            ('_xyz', 'optim_xyz', '_xyz', False, True),
            ('_features_dc', 'optim_features', '_features_dc', False, True),
            ('_features_rest', 'optim_features', '_features_rest', False, True), # separate config for rest
            ('_opacity', 'optim_opacity', '_opacity', False, True),
            ('_scaling', 'optim_scaling', '_scaling', False, True),
            ('_rotation', 'optim_rotation', '_rotation', False, True),
        ]

        enabled_gs_groups = []
        for param_name, config_key, group_name, s1_active_default, s2_active_default in gs_param_configs:
            group_enabled = _fbf_group_enabled(group_name)
            cfg_group = optimizer_config.get(config_key)
            if cfg_group and group_enabled:
                actual_param = getattr(model, param_name)
                # Ensure the parameter is a nn.Parameter and requires grad if it's going to be optimized.
                # The requires_grad will be managed by update_parameter_activity later.
                if not isinstance(actual_param, nn.Parameter):
                    setattr(model, param_name, nn.Parameter(actual_param.detach()))
                getattr(model, param_name).requires_grad_(True) # Initially set to True, will be controlled by stage
                
                lr_val = cfg_group.lr
                # Special handling for features_rest LR if main optim_features is used
                if config_key == 'optim_features_rest' and not optimizer_config.get('optim_features_rest'):
                    # If optim_features_rest is not defined, but optim_features is, use optim_features LR / 20.0
                    main_feat_cfg = optimizer_config.get('optim_features')
                    if main_feat_cfg:
                        lr_val = main_feat_cfg.lr / 20.0
                    else: # Should not happen if logic is correct, but as a fallback
                        print(f"Warning: optim_features_rest LR not found for {group_name}")
                        continue # Skip this param group if LR is undetermined
                elif config_key == 'optim_features_rest' and optimizer_config.get('optim_features_rest'):
                    lr_val = cfg_group.lr # Use its own LR if defined
                
                print(f'[SplattingAvatarOptim] Adding GS group: {group_name}, lr={lr_val}')
                stage1_lr = float(cfg_group.get('stage1_lr', lr_val))
                stage2_lr = float(cfg_group.get('stage2_lr', lr_val))
                param_groups.append({
                    'params': [getattr(model, param_name)],
                    'lr': lr_val,
                    'stage1_lr': stage1_lr,
                    'stage2_lr': stage2_lr,
                    'name': group_name,
                    'stage1_active': cfg_group.get('stage1_active', s1_active_default),
                    'stage2_active': cfg_group.get('stage2_active', s2_active_default)
                })
                if cfg_group.get('scheduler_args'):
                    self.lr_schedulers[group_name] = _make_expon_lr_func(cfg_group.scheduler_args)
                enabled_gs_groups.append(group_name)
            else:
                # Ensure param exists on model even if not optimized.
                if hasattr(model, param_name):
                    if not isinstance(getattr(model, param_name), nn.Parameter):
                        setattr(model, param_name, nn.Parameter(getattr(model, param_name).detach(), requires_grad=False))
                    else:
                        try:
                            getattr(model, param_name).requires_grad_(False)
                        except Exception:
                            pass
                elif not hasattr(model, param_name):
                    # This case should be handled by model initialization
                    print(f"Warning: GS parameter {param_name} not found on model and not configured for optimization.")

                if fbf_enabled and group_enabled and not cfg_group:
                    print(f"[FullbodyFixBake] Warning: optimize.{fbf_flag_keys.get(group_name, group_name)}=true but '{config_key}' is missing; skipping optimizer group for {group_name}.")

        if fbf_enabled:
            print(f"[FullbodyFixBake] Trainable GS groups: {enabled_gs_groups if enabled_gs_groups else '[]'}")


        # --- Deformation Module Parameters ---
        if model.deformation_module is not None:
            deform_optim_config = optimizer_config.get('deformation_optim', {})
            deform_backend = str(getattr(model.deformation_module, "backend", "mlp")).lower()

            deform_param_configs = []
            if deform_backend == "litept" and hasattr(model.deformation_module, "get_param_groups"):
                litept_stage = str(_cfg_get(optimizer_config, "litept_stage", "p1")).lower()
                if litept_stage not in {"p1", "p2", "ft"}:
                    raise ValueError(f"Invalid optim.litept_stage='{litept_stage}'. Expected one of: p1, p2, ft.")
                # Task-1 LitePT groups (future-compatible names).
                litept_groups = model.deformation_module.get_param_groups()
                litept_cfg_map = [
                    ("litept_trunk_early", "optim_litept_trunk_early", True, False),
                    ("litept_trunk_late", "optim_litept_trunk_late", True, False),
                    ("litept_decoder", "optim_litept_decoder", True, False),
                    ("litept_pose_head", "optim_litept_pose_head", True, False),
                    ("litept_interaction_head", "optim_litept_interaction_head", False, True),
                    ("litept_gate_head", "optim_litept_gate_head", False, True),
                    ("litept_gs_alpha_head", "optim_litept_gs_alpha_head", False, True),
                    ("litept_gs_logscale_head", "optim_litept_gs_logscale_head", False, True),
                    # Shared groups in existing naming convention.
                    ("deform_latent_codes", "optim_latent_codes", True, False),
                    ("deform_gauss_mlp_pos", "optim_gauss_mlp_pos", False, True),
                    ("deform_gauss_mlp_color", "optim_gauss_mlp_color", False, True),
                    ("deform_blendshapes_pos", "optim_blendshapes_pos", False, True),
                    ("deform_blendshapes_color", "optim_blendshapes_color", False, True),
                ]
                litept_added_counts = {}
                for group_name, config_key, s1_active_default, s2_active_default in litept_cfg_map:
                    cfg_group = deform_optim_config.get(config_key)
                    if not cfg_group:
                        continue
                    params_list = list(litept_groups.get(group_name, []))
                    if len(params_list) == 0:
                        litept_added_counts[config_key] = 0
                        continue
                    for p in params_list:
                        p.requires_grad_(True)
                    base_lr = cfg_group.get('lr', None)
                    if base_lr is None:
                        base_lr = cfg_group.get('stage1_lr', cfg_group.get('stage2_lr', 0.0))
                    base_lr = float(base_lr)
                    print(f'[SplattingAvatarOptim] Adding Deformation group: {group_name}, base_lr={base_lr}')
                    stage1_lr = float(cfg_group.get('stage1_lr', base_lr))
                    stage2_lr = float(cfg_group.get('stage2_lr', base_lr))
                    param_groups.append({
                        'params': params_list,
                        'lr': base_lr,
                        'stage1_lr': stage1_lr,
                        'stage2_lr': stage2_lr,
                        'name': group_name,
                        'stage1_active': cfg_group.get('stage1_active', s1_active_default),
                        'stage2_active': cfg_group.get('stage2_active', s2_active_default),
                        'stage2_start_offset': int(cfg_group.get('stage2_start_offset', 0)) if cfg_group.get('stage2_start_offset', None) is not None else 0,
                        'stage2_end_offset': (int(cfg_group.get('stage2_end_offset')) if cfg_group.get('stage2_end_offset', None) is not None else None),
                    })
                    if cfg_group.get('scheduler_args'):
                        self.lr_schedulers[group_name] = _make_expon_lr_func(cfg_group.scheduler_args)
                    litept_added_counts[config_key] = len(params_list)

                # Stage-aware validation for LitePT schedules.
                validate_litept_stage_config(
                    litept_stage=litept_stage,
                    deform_optim_config=deform_optim_config,
                    added_litept_groups=litept_added_counts,
                )
                # Runtime interaction flag is intentionally separate from optimizer freeze.
                try:
                    deform_cfg = _cfg_get(getattr(model, "config", None), "deformation", {})
                    litept_cfg = _cfg_get(deform_cfg, "litept", {})
                    interaction_cfg = _cfg_get(litept_cfg, "interaction", {})
                    interaction_enabled = bool(_cfg_get(interaction_cfg, "enabled", False))
                    if litept_stage == "p1" and interaction_enabled:
                        print("[SplattingAvatarOptim] Warning: optim.litept_stage=p1 but model.deformation.litept.interaction.enabled=true. "
                              "P1 recommends interaction.enabled=false.")
                except Exception:
                    pass
                # Lightweight schedule summary for debugging/config audit.
                print(f"[SplattingAvatarOptim][LitePT] stage={litept_stage}, stage1_end_iter={self.stage1_end_iter}")
                for pg in param_groups:
                    n = str(pg.get("name", ""))
                    if not n.startswith("litept_"):
                        continue
                    print(
                        "[SplattingAvatarOptim][LitePT] "
                        f"{n}: stage1_active={bool(pg.get('stage1_active', False))}, "
                        f"stage2_active={bool(pg.get('stage2_active', False))}, "
                        f"stage1_lr={float(pg.get('stage1_lr', 0.0))}, stage2_lr={float(pg.get('stage2_lr', 0.0))}, "
                        f"stage2_start_offset={int(pg.get('stage2_start_offset', 0) or 0)}, "
                        f"stage2_end_offset={pg.get('stage2_end_offset', None)}"
                    )
            else:
                deform_param_configs = [
                    ('vertex_deformation_net', 'optim_vert_mlp', 'deform_vert_mlp', True, False),
                    # Default: freeze per-frame latent codes Z in Stage 2 (Stage C) to avoid entanglement.
                    ('frame_latent_codes.weight', 'optim_latent_codes', 'deform_latent_codes', True, False),
                    # Allow separate scheduling for pos/color coefficient MLPs (back-compat: optim_gauss_mlp applies to both).
                    ('gaussian_blendshape_net.mlp_pos_blendshapes', 'optim_gauss_mlp_pos', 'deform_gauss_mlp_pos', False, True),
                    ('gaussian_blendshape_net.mlp_color_blendshapes', 'optim_gauss_mlp_color', 'deform_gauss_mlp_color', False, True),
                    ('pos_blendshapes', 'optim_blendshapes_pos', 'deform_blendshapes_pos', False, True),
                    ('color_blendshapes', 'optim_blendshapes_color', 'deform_blendshapes_color', False, True),
                ]

            # Back-compat: if user only specifies optim_gauss_mlp, apply it to both pos/color coefficient MLPs
            legacy_gauss_mlp = deform_optim_config.get('optim_gauss_mlp', None)
            if legacy_gauss_mlp is not None:
                if deform_optim_config.get('optim_gauss_mlp_pos', None) is None:
                    deform_optim_config['optim_gauss_mlp_pos'] = legacy_gauss_mlp
                if deform_optim_config.get('optim_gauss_mlp_color', None) is None:
                    deform_optim_config['optim_gauss_mlp_color'] = legacy_gauss_mlp

            for param_path, config_key, group_name, s1_active_default, s2_active_default in deform_param_configs:
                cfg_group = deform_optim_config.get(config_key)
                if cfg_group:
                    # Resolve parameter path (e.g., 'frame_latent_codes.weight')
                    current_obj = model.deformation_module
                    parts = param_path.split('.')
                    for part in parts[:-1]:
                        current_obj = getattr(current_obj, part)
                    final_param = getattr(current_obj, parts[-1])
                    
                    # For nn.Module parameters (like MLPs), use .parameters()
                    params_list = list(final_param.parameters()) if isinstance(final_param, nn.Module) else [final_param]
                    
                    # Ensure all fetched parameters require grad initially, stage logic will control it
                    for p in params_list:
                        p.requires_grad_(True)

                    # Resolve a base LR from (lr | stage1_lr | stage2_lr) for back-compat.
                    base_lr = cfg_group.get('lr', None)
                    if base_lr is None:
                        base_lr = cfg_group.get('stage1_lr', cfg_group.get('stage2_lr', 0.0))
                    base_lr = float(base_lr)

                    print(f'[SplattingAvatarOptim] Adding Deformation group: {group_name}, base_lr={base_lr}')
                    stage1_lr = float(cfg_group.get('stage1_lr', base_lr))
                    stage2_lr = float(cfg_group.get('stage2_lr', base_lr))
                    param_groups.append({
                        'params': params_list,
                        'lr': base_lr,
                        # Stage-specific static LRs (used when no scheduler_args).
                        'stage1_lr': stage1_lr,
                        'stage2_lr': stage2_lr,
                        'name': group_name,
                        'stage1_active': cfg_group.get('stage1_active', s1_active_default),
                        'stage2_active': cfg_group.get('stage2_active', s2_active_default),
                        # Fine-grain stage2 schedule (offsets relative to stage2 start, in iterations)
                        # - stage2_start_offset: delay enabling group for first M iters in stage2
                        # - stage2_end_offset: disable after N iters in stage2 (exclusive)
                        'stage2_start_offset': int(cfg_group.get('stage2_start_offset', 0)) if cfg_group.get('stage2_start_offset', None) is not None else 0,
                        'stage2_end_offset': (int(cfg_group.get('stage2_end_offset')) if cfg_group.get('stage2_end_offset', None) is not None else None),
                    })
                    if cfg_group.get('scheduler_args'):
                        self.lr_schedulers[group_name] = _make_expon_lr_func(cfg_group.scheduler_args)

        # --- TODO-4: LBS weight residual (cloth-only) ---
        lbs_cfg = optimizer_config.get("lbs_weight_residual_optim", None)
        if lbs_cfg is not None and getattr(model, "lbs_weight_residual", None) is not None:
            try:
                p = model.lbs_weight_residual.delta_l
                if not isinstance(p, nn.Parameter):
                    p = nn.Parameter(p.detach())
                    model.lbs_weight_residual.delta_l = p
                p.requires_grad_(True)

                base_lr = lbs_cfg.get("lr", lbs_cfg.get("stage2_lr", lbs_cfg.get("stage1_lr", 0.0)))
                base_lr = float(base_lr)
                stage1_lr = float(lbs_cfg.get("stage1_lr", base_lr))
                stage2_lr = float(lbs_cfg.get("stage2_lr", base_lr))
                print(f"[SplattingAvatarOptim] Adding LBS residual group: lbs_weight_residual, base_lr={base_lr}")
                param_groups.append({
                    "params": [p],
                    "lr": base_lr,
                    "stage1_lr": stage1_lr,
                    "stage2_lr": stage2_lr,
                    "name": "lbs_weight_residual",
                    "stage1_active": bool(lbs_cfg.get("stage1_active", False)),
                    "stage2_active": bool(lbs_cfg.get("stage2_active", True)),
                    "stage2_start_offset": int(lbs_cfg.get("stage2_start_offset", 0) or 0),
                    "stage2_end_offset": (int(lbs_cfg.get("stage2_end_offset")) if lbs_cfg.get("stage2_end_offset", None) is not None else None),
                })
            except Exception as e:
                print(f"[SplattingAvatarOptim] Warning: failed to set up LBS residual optimizer group: {e}")

        # --- TODO-5 (P2-1): base scale residual (static, per-Gaussian) ---
        scale_cfg = optimizer_config.get("deform_scale_base_optim", None)
        if scale_cfg is not None and hasattr(model, "delta_log_s_base") and getattr(model, "delta_log_s_base", None) is not None:
            try:
                p = model.delta_log_s_base
                if not isinstance(p, nn.Parameter):
                    model.delta_log_s_base = nn.Parameter(p.detach())
                    p = model.delta_log_s_base
                p.requires_grad_(True)

                base_lr = scale_cfg.get("lr", scale_cfg.get("stage2_lr", scale_cfg.get("stage1_lr", 0.0)))
                base_lr = float(base_lr)
                stage1_lr = float(scale_cfg.get("stage1_lr", base_lr))
                stage2_lr = float(scale_cfg.get("stage2_lr", base_lr))
                print(f"[SplattingAvatarOptim] Adding scale-base group: deform_scale_base, base_lr={base_lr}")
                param_groups.append({
                    "params": [p],
                    "lr": base_lr,
                    "stage1_lr": stage1_lr,
                    "stage2_lr": stage2_lr,
                    "name": "deform_scale_base",
                    "stage1_active": bool(scale_cfg.get("stage1_active", False)),
                    "stage2_active": bool(scale_cfg.get("stage2_active", True)),
                    "stage2_start_offset": int(scale_cfg.get("stage2_start_offset", scale_cfg.get("stage2_start_offset_scale_base", 0)) or 0),
                    "stage2_end_offset": (int(scale_cfg.get("stage2_end_offset")) if scale_cfg.get("stage2_end_offset", None) is not None else None),
                })
            except Exception as e:
                print(f"[SplattingAvatarOptim] Warning: failed to set up scale-base optimizer group: {e}")
        
        self.optimizer = torch.optim.Adam(param_groups, lr=optimizer_config.get("default_lr", 5e-4), eps=optimizer_config.get("adam_eps", 1e-15))
        
        # Initialize xyz_gradient_accum and denom on the model if they are not already present or have wrong size
        # This should ideally be managed by the model itself based on its _xyz state.
        if not hasattr(model, 'xyz_gradient_accum') or model.xyz_gradient_accum.shape[0] != model._xyz.shape[0]:
            model.xyz_gradient_accum = torch.zeros((model._xyz.shape[0], 1), device=model._xyz.device)
        if not hasattr(model, 'denom') or model.denom.shape[0] != model._xyz.shape[0]:
            model.denom = torch.zeros((model._xyz.shape[0], 1), device=model._xyz.device)
        
        model.percent_dense = optimizer_config.get('percent_dense', 0.01)

        # Global optimizer scheduler (e.g., MultiStepLR)
        if optimizer_config.get('scheduler'):
            scheduler_conf = optimizer_config.scheduler
            total_iteration = optimizer_config.total_iteration # This should be overall total iterations
            milestones = scheduler_conf.get('milestones', [self.stage1_end_iter] if self.stage1_end_iter > 0 else []) 
            
            # If milestones are relative to stages, they need careful handling.
            # For now, assume global milestones.
            if isinstance(milestones, str) and milestones.lower() == "auto_stage1":
                 milestones = [self.stage1_end_iter] if self.stage1_end_iter > 0 else []
            elif not isinstance(milestones, list):
                # default: milestone every X steps, up to total_iteration
                ms_interval = scheduler_conf.get('milestone_interval', 10000)
                milestones = [i for i in range(ms_interval, total_iteration + 1, ms_interval)]

            if milestones: # Only add scheduler if there are milestones
                decay = scheduler_conf.get('gamma', 0.33)
                self.schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=milestones,
                    gamma=decay,
                ))
                print(f"[SplattingAvatarOptim] Added MultiStepLR scheduler with milestones: {milestones}, gamma: {decay}")

        # Initial call to set requires_grad based on iteration (assume starting at iter 1 for safety)
        self.update_parameter_activity(1) 

    def update_parameter_activity(self, iteration):
        # Determine current stage: Stage 1 up to stage1_end_iter, Stage 2 afterwards.
        # If stage1_end_iter is 0, it's always Stage 2 (or a single combined stage).
        current_stage = 1 if self.stage1_end_iter > 0 and iteration <= self.stage1_end_iter else 2

        if self.current_stage != current_stage:
            print(f"[SplattingAvatarOptim] Iteration {iteration}: {'Switching to' if self.current_stage != 0 else 'Starting in'} Stage {current_stage}")
            self.current_stage = current_stage
            # Reset printed flags when stage changes to re-print grad changes if verbose
            self._printed_grad_flags = set()

        if not hasattr(self, '_printed_grad_flags'):
            self._printed_grad_flags = set()

        stage2_iter = None
        if current_stage == 2 and self.stage1_end_iter > 0:
            stage2_iter = int(iteration) - int(self.stage1_end_iter) - 1  # 0-based within stage2

        for param_group in self.optimizer.param_groups:
            group_name = param_group['name']
            is_active_in_stage1 = param_group.get('stage1_active', False)
            is_active_in_stage2 = param_group.get('stage2_active', False) # Default to False if not specified

            should_be_active_now = (current_stage == 1 and is_active_in_stage1) or \
                                   (current_stage == 2 and is_active_in_stage2)

            # Fine-grain stage2 schedule gates (only applies in stage2 and only if stage2_active)
            if current_stage == 2 and should_be_active_now and stage2_iter is not None:
                start_off = int(param_group.get('stage2_start_offset', 0) or 0)
                end_off = param_group.get('stage2_end_offset', None)
                if stage2_iter < start_off:
                    should_be_active_now = False
                if end_off is not None and stage2_iter >= int(end_off):
                    should_be_active_now = False
            
            # Update requires_grad for all params in the group
            for param in param_group['params']:
                if param.requires_grad != should_be_active_now:
                    param.requires_grad = should_be_active_now
                    log_key = f"{group_name}_{current_stage}_{should_be_active_now}"
                    if self.gs_model.verbose and log_key not in self._printed_grad_flags:
                        # print(f"[SplattingAvatarOptim] Iter {iteration} (Stage {current_stage}): Setting requires_grad={should_be_active_now} for group '{group_name}'")
                        self._printed_grad_flags.add(log_key)
            
            # Set LR for active groups based on stage, else 0.
            if should_be_active_now:
                if current_stage == 1:
                    param_group['lr'] = float(param_group.get('stage1_lr', param_group.get('lr', 0.0)))
                else:
                    param_group['lr'] = float(param_group.get('stage2_lr', param_group.get('lr', 0.0)))
            else:
                param_group['lr'] = 0.0

    def update_learning_rate(self, iteration):
        # First, update parameter activity which might change requires_grad status
        self.update_parameter_activity(iteration)

        # Then, apply per-parameter group LR schedulers
        final_lr = self.optimizer.param_groups[0]['lr'] # Fallback LR
        for param_group in self.optimizer.param_groups:
            if param_group['name'] in self.lr_schedulers:
                # Only apply scheduler if the group is active
                current_stage = 1 if self.stage1_end_iter > 0 and iteration <= self.stage1_end_iter else 2
                is_active_in_stage1 = param_group.get('stage1_active', False)
                is_active_in_stage2 = param_group.get('stage2_active', False)
                is_active_now = (current_stage == 1 and is_active_in_stage1) or \
                                (current_stage == 2 and is_active_in_stage2)

                # Also respect fine-grain stage2 schedule (lr==0 means inactive)
                if is_active_now and param_group['params'][0].requires_grad and float(param_group.get('lr', 0.0)) > 0:
                    lr = self.lr_schedulers[param_group['name']](iteration)
                    param_group['lr'] = lr
                    final_lr = lr # Keep track of one of the LRs for return
        
        # Then, step any global schedulers (like MultiStepLR)
        # These operate on LRs potentially modified by per-group schedulers.
        for scheduler in self.schedulers:
            scheduler.step() # Global schedulers step based on epoch/iteration count
        
        return final_lr # Return one of the LRs, e.g., for pbar display

    def step(self, enable_optim=True, enable_smplx=True):
        if enable_optim:
            self.optimizer.step()

        # if enable_smplx and self.smplx_optim is not None: # smplx_optim not used here
        #     self.smplx_optim.step()

        # Global schedulers are stepped in update_learning_rate if iteration-based
        # If they were epoch-based, they'd be stepped after an epoch. Here, assuming per-iteration step for MultiStepLR.
        # The `scheduler.step()` in original code was after optimizer.step(). Moved to update_learning_rate.

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

        if self.smplx_optim is not None:
            self.smplx_optim.zero_grad()

    ##################################################
    def reset_opacity(self):
        model = self.gs_model
        opacities_new = model.inverse_opacity_activation(torch.min(model.get_opacity, torch.ones_like(model.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, '_opacity')
        model._opacity = optimizable_tensors['_opacity']

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state['exp_avg'] = torch.zeros_like(tensor)
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group['params']) != 1:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)

            if group['name'] != 'xyz_comp':
                if stored_state is not None:
                    stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                    stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]

                    del self.optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter((group['params'][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group['name']] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    optimizable_tensors[group['name']] = group['params'][0]
            else:
                if stored_state is not None:
                    stored_state['exp_avg'] = stored_state['exp_avg'][:, mask]
                    stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][:, mask]

                    del self.optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter((group['params'][0][:, mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group['name']] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][:, mask].requires_grad_(True))
                    optimizable_tensors[group['name']] = group['params'][0]

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self.gs_model.prune_points(valid_points_mask, optimizable_tensors)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # assert len(group['params']) == 1
            if len(group['params']) != 1:
                continue

            extension_tensor = tensors_dict[group['name']]
            if extension_tensor is None:
                continue

            dd = 1 if group['name'] == 'xyz_comp' else 0
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=dd)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=dd)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=dd).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=dd).requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]

        return optimizable_tensors
    
    def densification_postfix(self, densify_out):
        d = {
            '_xyz': densify_out['new_xyz'],
            '_scaling' : densify_out['new_scaling'],
            '_rotation' : densify_out['new_rotation'],
        }

        d.update({
            '_features_dc': densify_out['new_features_dc'],
            '_features_rest': densify_out['new_features_rest'],
            '_opacity': densify_out['new_opacity'],
        })

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.gs_model.densification_postfix(optimizable_tensors, densify_out)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        selected_pts_mask, new_xyz = self.gs_model.prepare_densify_and_split(grads, grad_threshold, scene_extent, N=N)
        self.split_selected_to_new_xyz(selected_pts_mask, new_xyz, N)
        
    def split_selected_to_new_xyz(self, selected_pts_mask, new_xyz, N):
        splitout = self.gs_model.prepare_split_selected_to_new_xyz(selected_pts_mask, new_xyz, N)
        self.densification_postfix(splitout)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent=2.0):
        cloneout = self.gs_model.prepare_densify_and_clone(grads, grad_threshold, scene_extent)
        self.densification_postfix(cloneout)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        model = self.gs_model
        grads = model.xyz_gradient_accum / model.denom
        grads[grads.isnan()] = 0.0

        if model.config.get('max_n_gauss', -1) <= 0 or model.get_xyz.shape[0] < model.config.max_n_gauss:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        self.prune(min_opacity, extent, max_screen_size)

    def prune(self, min_opacity, extent, max_screen_size):
        model = self.gs_model

        opacity = model.get_opacity
        prune_mask = (opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = model.max_radii2D > max_screen_size
            big_points_ws = model.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.gs_model.add_densification_stats(viewspace_point_tensor, update_filter)
        
    def adaptive_density_control(self, render_pkg, iteration, cameras_extent=2.0):
        model = self.gs_model
        # Tao/deformation training expects a fixed set of pretrained Gaussians.
        # Disable any densification/pruning paths when deformation is enabled.
        if getattr(model, "use_deformation", False):
            return
        viewspace_point_tensor = render_pkg['viewspace_points']
        visibility_filter = render_pkg['visibility_filter']
        radii = render_pkg['radii']
        opt = self.optimizer_config
        opacity_reset_iter = opt.get('opacity_reset_start_iter', 300)
        opacity_reset_interval = opt.get('opacity_reset_interval', 0)

        # Densification
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            model.max_radii2D[visibility_filter] = torch.max(model.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = opt.size_threshold if iteration > opacity_reset_interval else None
                # size_threshold = None
                self.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, 
                                       cameras_extent, size_threshold)
            
            # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            if opacity_reset_interval > 0 and (iteration - opacity_reset_iter) % opacity_reset_interval == 0:
                self.reset_opacity()


    ########################################
    def update_trangle_walk(self, iteration):        
        triangle_walk_interval = self.optimizer_config.get('triangle_walk_interval', 100)
        if iteration % triangle_walk_interval == 0:
            # from model import libcore
            # libcore.startCudaTimer('walking_on_triangles')
            self.gs_model.walking_on_triangles()
            # libcore.stopCudaTimer('walking_on_triangles')
            self.reset_optimizer_uv()

    def reset_optimizer_uv(self):
        for group in self.optimizer.param_groups:
            if group["name"] == "_xyz":
                assert len(group["params"]) == 1
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"][..., :2] = torch.zeros_like(stored_state["exp_avg"][..., :2])
                    stored_state["exp_avg_sq"][..., :2] = torch.zeros_like(stored_state["exp_avg_sq"][..., :2])

                    del self.optimizer.state[group['params'][0]]
                    _xyz = group["params"][0]
                    group['params'][0] = nn.Parameter(torch.cat((torch.zeros_like(_xyz[..., :2]), 
                                                                 _xyz[..., 2:]), dim=-1).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    _xyz = group["params"][0]
                    group['params'][0] = nn.Parameter(torch.cat((torch.zeros_like(_xyz[..., :2]), 
                                                                 _xyz[..., 2:]), dim=-1).requires_grad_(True))


    ##################################################
    def save_checkpoint(self, model_path, iteration):
        pc_dir = os.path.join(model_path, f'point_cloud/iteration_{iteration}')
        os.makedirs(pc_dir, exist_ok=True)

        # to cano
        self.gs_model.update_to_cano_mesh()

        # save gaussian
        self.gs_model.save_ply(os.path.join(pc_dir, 'point_cloud.ply'))

        # save mesh embedding
        self.gs_model.save_embedding_json(os.path.join(pc_dir, 'embedding.json'))

        return pc_dir

