import torch
import torch.nn as nn
from .litept_backend import LitePTVertexDeformationNetwork
try:
    import tinycudann as tcnn
except ImportError:
    print("tinycudann not found. MLPs will use torch.nn.Linear, which will be slower if deformation network is used.")
    tcnn = None

class _TcnnCoreWithZeroHead(nn.Module):
    """tcnn core + torch head that starts at zero output.

    Important: tinycudann may run internal ops in fp16; we cast:
    - input -> fp32 for the tcnn core
    - tcnn output -> fp32 for the torch head (avoids Half/Float matmul mismatch)
    The caller can cast the final output to any dtype it wants.
    """
    def __init__(self, core: nn.Module, head: nn.Linear):
        super().__init__()
        self.core = core
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.core(x.float().contiguous())
        y = y.float()
        return self.head(y)

def _zero_last_linear(module: nn.Module) -> bool:
    """Zero-init the last nn.Linear found inside `module`.

    Returns True if a Linear layer was found and zeroed.
    """
    last_linear = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        return False
    with torch.no_grad():
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)
    return True

class PositionalEncoder(nn.Module):
    def __init__(self, input_dims, num_freqs, include_input=True):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.output_dims = input_dims * (1 + 2 * num_freqs) if include_input else input_dims * (2 * num_freqs)
        
        if self.num_freqs > 0:
            self.freq_bands = 2.0 ** torch.arange(num_freqs)

    def forward(self, x):
        # x: (..., input_dims)
        if self.num_freqs == 0:
            return x

        out = []
        if self.include_input:
            out.append(x)
        
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1) # (..., output_dims)

def create_mlp(
    input_dim,
    output_dim,
    hidden_layers,
    hidden_dim,
    use_tcnn=True,
    activation="ReLU",
    output_activation="None",
    *,
    tcnn_zero_head: bool = False,
    tcnn_head_dim: int = None,
):
    """Create an MLP using tinycudann if available, otherwise PyTorch.

    If `tcnn_zero_head=True`, we build:
      tcnn.FullyFusedMLP(input_dim -> head_dim) + torch.nn.Linear(head_dim -> output_dim)
    and zero-init the torch head so the initial output is ~0 (identity deformation).
    """
    if tcnn and use_tcnn:
        use_zero_head = bool(tcnn_zero_head)
        head_dim = int(tcnn_head_dim) if tcnn_head_dim is not None else int(hidden_dim)
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": activation,
            "output_activation": output_activation,
            "n_neurons": hidden_dim,
            "n_hidden_layers": hidden_layers,
        }
        # TinyCUDNN input/output dim includes all flattened dims
        core = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=(head_dim if use_zero_head else output_dim),
            network_config=network_config
        )
        if use_zero_head:
            head = nn.Linear(head_dim, output_dim)
            with torch.no_grad():
                nn.init.zeros_(head.weight)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)
            model = _TcnnCoreWithZeroHead(core, head)
        else:
            model = core
    else:
        layers = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        if output_activation.lower() != "none" and output_activation.lower() != "identity":
            # Add common activations if specified, extend if necessary
            if output_activation == "ReLU":
                layers.append(nn.ReLU())
            elif output_activation == "Sigmoid":
                layers.append(nn.Sigmoid())
            # Add more activations as needed
            else:
                print(f"Warning: Output activation {output_activation} not implemented for torch.nn MLP, using Identity.")

        model = nn.Sequential(*layers)
    return model

# --- Placeholder for VertexDeformationNetwork ---
class VertexDeformationNetwork(nn.Module):
    def __init__(
        self,
        pos_encoder_L=6,
        num_pose_params=63,
        latent_code_dim=32,
        mlp_hidden_layers=5,
        mlp_hidden_dim=64,
        use_tcnn=True,
        tcnn_zero_head=False,
        tcnn_head_dim=None,
        *,
        input_mode: str = "xyz",
        pe_L_rlocal: int = 6,
        smpl_embed_dim: int = 32,
        use_rnorm: bool = False,
        use_smpl_curvature: bool = False,
        smpl_num_verts: int | None = None,
        log_input_stats: bool = False,
        assert_same_cano_frame: bool = False,
    ):
        super().__init__()
        self.use_tcnn = use_tcnn
        self.input_mode = str(input_mode).lower()
        self.pe_L_rlocal = int(pe_L_rlocal)
        self.smpl_embed_dim = int(smpl_embed_dim)
        self.use_rnorm = bool(use_rnorm)
        self.use_smpl_curvature = bool(use_smpl_curvature)
        self.smpl_num_verts = int(smpl_num_verts) if smpl_num_verts is not None else None
        self.log_input_stats = bool(log_input_stats)
        self.assert_same_cano_frame = bool(assert_same_cano_frame)

        self.pos_encoder = PositionalEncoder(input_dims=3, num_freqs=pos_encoder_L, include_input=True)
        mlp_input_dim = self.pos_encoder.output_dims + num_pose_params + latent_code_dim

        self.mlp_all_verts = create_mlp(
            input_dim=mlp_input_dim,
            output_dim=3, # 3D offset
            hidden_layers=mlp_hidden_layers,
            hidden_dim=mlp_hidden_dim,
            use_tcnn=self.use_tcnn,
            activation="ReLU",
            output_activation="None", # Output raw offsets
            tcnn_zero_head=bool(tcnn_zero_head),
            tcnn_head_dim=tcnn_head_dim,
        )

        self.mlp_garment_verts = create_mlp(
            input_dim=mlp_input_dim, # Same input structure
            output_dim=3, # 3D offset
            hidden_layers=mlp_hidden_layers,
            hidden_dim=mlp_hidden_dim,
            use_tcnn=self.use_tcnn,
            activation="ReLU",
            output_activation="None", # Output raw offsets
            tcnn_zero_head=bool(tcnn_zero_head),
            tcnn_head_dim=tcnn_head_dim,
        )

        # Make the *initial* deformation exactly zero to avoid global drift at iteration 1.
        # This is important because we start from a pretrained canonical avatar and expect
        # the deformation module to begin as an identity mapping.
        if (not (self.use_tcnn and tcnn is not None)):
            _zero_last_linear(self.mlp_all_verts)
            _zero_last_linear(self.mlp_garment_verts)

        # --- smpl_anchor mode (tao_offset_input_enhancement.md) ---
        # We keep the legacy xyz path intact and add separate anchor MLPs + an SMPL-vertex embedding table.
        self.rlocal_encoder = None
        self.E_smpl = None
        self.mlp_all_verts_anchor = None
        self.mlp_garment_verts_anchor = None
        self._last_anchor_stats = None  # populated in forward when enabled

        if self.use_smpl_curvature:
            # Optional feature per spec; not implemented yet.
            raise NotImplementedError("use_smpl_curvature=True is not implemented (set it to false).")

        if self.input_mode == "smpl_anchor":
            if self.smpl_num_verts is None or self.smpl_num_verts <= 0:
                raise ValueError("smpl_anchor mode requires smpl_num_verts (pass it from the model).")

            # PE(r_local) without including raw input; we explicitly concat raw r_local.
            self.rlocal_encoder = PositionalEncoder(input_dims=3, num_freqs=int(self.pe_L_rlocal), include_input=False)

            # Per-SMPL-vertex learnable embedding table E_smpl.
            self.E_smpl = nn.Embedding(int(self.smpl_num_verts), int(self.smpl_embed_dim))
            with torch.no_grad():
                torch.nn.init.normal_(self.E_smpl.weight, mean=0.0, std=0.01)

            r_feat_dim = 3 + int(self.rlocal_encoder.output_dims)
            anchor_in_dim = int(self.smpl_embed_dim) + int(r_feat_dim) + int(num_pose_params) + int(latent_code_dim)
            if self.use_rnorm:
                anchor_in_dim += 1

            self.mlp_all_verts_anchor = create_mlp(
                input_dim=anchor_in_dim,
                output_dim=3,
                hidden_layers=mlp_hidden_layers,
                hidden_dim=mlp_hidden_dim,
                use_tcnn=self.use_tcnn,
                activation="ReLU",
                output_activation="None",
                tcnn_zero_head=bool(tcnn_zero_head),
                tcnn_head_dim=tcnn_head_dim,
            )
            self.mlp_garment_verts_anchor = create_mlp(
                input_dim=anchor_in_dim,
                output_dim=3,
                hidden_layers=mlp_hidden_layers,
                hidden_dim=mlp_hidden_dim,
                use_tcnn=self.use_tcnn,
                activation="ReLU",
                output_activation="None",
                tcnn_zero_head=bool(tcnn_zero_head),
                tcnn_head_dim=tcnn_head_dim,
            )
            if (not (self.use_tcnn and tcnn is not None)):
                _zero_last_linear(self.mlp_all_verts_anchor)
                _zero_last_linear(self.mlp_garment_verts_anchor)

    def forward(self, cano_verts, pose_params, frame_latent_code, garment_mask, *, smpl_anchor_cache=None):
        # cano_verts: (N_verts, 3)
        # pose_params: (63,)
        # frame_latent_code: (32,)
        # garment_mask: (N_verts,) boolean tensor

        N_verts = cano_verts.shape[0]

        if self.input_mode == "smpl_anchor":
            if smpl_anchor_cache is None:
                raise ValueError("smpl_anchor mode requires `smpl_anchor_cache` (tri_vidx + bary + r_local).")
            tri_vidx = smpl_anchor_cache.get("tri_vidx", None)
            bary = smpl_anchor_cache.get("bary", None)
            r_local = smpl_anchor_cache.get("r_local", None)
            q = smpl_anchor_cache.get("q", None)
            if tri_vidx is None or bary is None or r_local is None:
                raise ValueError("smpl_anchor_cache missing required keys: tri_vidx, bary, r_local.")

            tri_vidx = tri_vidx.to(device=cano_verts.device, dtype=torch.long)
            bary = bary.to(device=cano_verts.device, dtype=torch.float32)
            r_local = r_local.to(device=cano_verts.device, dtype=torch.float32)

            # Optional sanity check: cloth canonical and SMPL reference surface must be in the same frame.
            if self.assert_same_cano_frame and (q is not None):
                qv = q.to(device=cano_verts.device, dtype=torch.float32)
                d = (cano_verts.detach().to(torch.float32) - qv).norm(dim=-1)
                # Threshold is intentionally loose; this is a debugging guardrail.
                if torch.isfinite(d).all() and float(d.max().item()) > 1.0:
                    raise RuntimeError(f"smpl_anchor cache sanity failed: max ||p-q|| = {float(d.max().item()):.4g} (> 1.0).")

            if self.E_smpl is None or self.rlocal_encoder is None or self.mlp_all_verts_anchor is None:
                raise RuntimeError("smpl_anchor modules not initialized.")

            # E_q via barycentric interpolation of per-SMPL-vertex embedding table.
            Etri = self.E_smpl(tri_vidx)  # (Nv,3,D)
            E_q = (bary.unsqueeze(-1) * Etri).sum(dim=1)  # (Nv,D)

            # r_local feature: concat raw r_local + PE(r_local)
            r_pe = self.rlocal_encoder(r_local)  # (Nv, 3*2L)
            r_feat = torch.cat([r_local, r_pe], dim=-1)
            if self.use_rnorm:
                r_norm = r_local.norm(dim=-1, keepdim=True)
                r_feat = torch.cat([r_feat, r_norm], dim=-1)

            expanded_pose_params = pose_params.unsqueeze(0).expand(N_verts, -1)
            expanded_latent_code = frame_latent_code.unsqueeze(0).expand(N_verts, -1)
            mlp_in = torch.cat([E_q, r_feat, expanded_pose_params, expanded_latent_code], dim=-1)

            if self.use_tcnn and tcnn is not None:
                offsets_all = self.mlp_all_verts_anchor(mlp_in.contiguous().float()).to(cano_verts.dtype)
            else:
                offsets_all = self.mlp_all_verts_anchor(mlp_in.float()).to(cano_verts.dtype)

            total_offsets = offsets_all
            num_garment_verts = torch.sum(garment_mask)
            if num_garment_verts > 0:
                garment_mlp_input = mlp_in[garment_mask]
                if self.use_tcnn and tcnn is not None:
                    offsets_garment_specific = self.mlp_garment_verts_anchor(garment_mlp_input.contiguous().float()).to(cano_verts.dtype)
                else:
                    offsets_garment_specific = self.mlp_garment_verts_anchor(garment_mlp_input.float()).to(cano_verts.dtype)
                if offsets_garment_specific.shape[0] == num_garment_verts:
                    total_offsets[garment_mask] += offsets_garment_specific

            if self.log_input_stats:
                try:
                    rn = r_local.norm(dim=-1)
                    en = E_q.norm(dim=-1)
                    self._last_anchor_stats = {
                        "r_norm_mean": float(rn.mean().item()),
                        "r_norm_max": float(rn.max().item()),
                        "E_q_norm_mean": float(en.mean().item()),
                        "E_q_norm_max": float(en.max().item()),
                    }
                except Exception:
                    self._last_anchor_stats = None

            return total_offsets

        # 1. Encode vertex positions
        encoded_verts = self.pos_encoder(cano_verts) # (N_verts, 3 * (1 + 2*L))

        # 2. Prepare inputs for MLPs
        # Expand pose_params and frame_latent_code to match N_verts
        expanded_pose_params = pose_params.unsqueeze(0).expand(N_verts, -1) # (N_verts, 63)
        expanded_latent_code = frame_latent_code.unsqueeze(0).expand(N_verts, -1) # (N_verts, 32)

        mlp_input_all = torch.cat([encoded_verts, expanded_pose_params, expanded_latent_code], dim=-1)
        # (N_verts, D_all_mlp_in)

        # 3. Predict base deformation for all vertices
        # tcnn.Network expects float32. Ensure input is correct type.
        # Also, tcnn.Network might not like mixed precision in its fwd pass if we use amp.
        # For now, assume inputs are float32.
        if self.use_tcnn and tcnn is not None:
             # tcnn.Network expects contiguous inputs and might have specific shape requirements.
             # (Batch_size, Din) where Batch_size can be N_verts
            offsets_all = self.mlp_all_verts(mlp_input_all.contiguous().float()).to(cano_verts.dtype)
        else:
            # If upstream uses AMP and produces fp16 inputs, torch Linear weights may remain fp32.
            # Cast inputs to fp32 for stability, then cast outputs back to cano dtype.
            offsets_all = self.mlp_all_verts(mlp_input_all.float()).to(cano_verts.dtype)
        
        # 4. Predict garment-specific deformation
        total_offsets = offsets_all
        
        num_garment_verts = torch.sum(garment_mask)
        if num_garment_verts > 0:
            garment_mlp_input = mlp_input_all[garment_mask]
            
            if self.use_tcnn and tcnn is not None:
                offsets_garment_specific = self.mlp_garment_verts(garment_mlp_input.contiguous().float()).to(cano_verts.dtype)
            else:
                offsets_garment_specific = self.mlp_garment_verts(garment_mlp_input.float()).to(cano_verts.dtype)
            
            # 5. Add garment-specific offsets
            # Ensure correct broadcasting or indexing if using advanced indexing with tcnn outputs
            if offsets_garment_specific.shape[0] == num_garment_verts:
                total_offsets[garment_mask] += offsets_garment_specific
            else:
                 # This case should ideally not happen if inputs to mlp_garment_verts are correctly filtered
                print(f"Warning: Mismatch in garment offsets shape. Expected {num_garment_verts}, got {offsets_garment_specific.shape[0]}")

        return total_offsets

# --- Placeholder for GaussianBlendshapeNetwork ---
class GaussianBlendshapeNetwork(nn.Module):
    def __init__(self, num_pose_params=63, num_blendshapes=20,
                 mlp_hidden_layers=5, mlp_hidden_dim=64, use_tcnn=True, tcnn_zero_head=False, tcnn_head_dim=None):
        super().__init__()
        self.use_tcnn = use_tcnn
        self.num_blendshapes = num_blendshapes

        # MLP for position blendshape weights
        self.mlp_pos_blendshapes = create_mlp(
            input_dim=num_pose_params,
            output_dim=num_blendshapes,
            hidden_layers=mlp_hidden_layers,
            hidden_dim=mlp_hidden_dim,
            use_tcnn=self.use_tcnn,
            activation="ReLU",
            output_activation="None", # Raw weights, maybe softmax/normalize later if needed
            tcnn_zero_head=bool(tcnn_zero_head),
            tcnn_head_dim=tcnn_head_dim,
        )

        # MLP for color blendshape weights
        self.mlp_color_blendshapes = create_mlp(
            input_dim=num_pose_params,
            output_dim=num_blendshapes,
            hidden_layers=mlp_hidden_layers,
            hidden_dim=mlp_hidden_dim,
            use_tcnn=self.use_tcnn,
            activation="ReLU",
            output_activation="None", # Raw weights
            tcnn_zero_head=bool(tcnn_zero_head),
            tcnn_head_dim=tcnn_head_dim,
        )

        # Start blendshape weights at zero (no residual) for stability.
        if (not (self.use_tcnn and tcnn is not None)):
            _zero_last_linear(self.mlp_pos_blendshapes)
            _zero_last_linear(self.mlp_color_blendshapes)

    def forward(self, pose_params):
        # pose_params: (63,) or (B, 63) if batched in the future
        
        # Ensure input is at least 2D for tcnn (Batch_size, Din)
        if pose_params.ndim == 1:
            pose_params_mlp_input = pose_params.unsqueeze(0)
        else:
            pose_params_mlp_input = pose_params

        if self.use_tcnn and tcnn is not None:
            pos_blend_weights = self.mlp_pos_blendshapes(pose_params_mlp_input.contiguous()).to(pose_params.dtype)
            color_blend_weights = self.mlp_color_blendshapes(pose_params_mlp_input.contiguous()).to(pose_params.dtype)
        else:
            pos_blend_weights = self.mlp_pos_blendshapes(pose_params_mlp_input).to(pose_params.dtype)
            color_blend_weights = self.mlp_color_blendshapes(pose_params_mlp_input).to(pose_params.dtype)

        # If input was unsqueezed, squeeze output back
        if pose_params.ndim == 1:
            pos_blend_weights = pos_blend_weights.squeeze(0)
            color_blend_weights = color_blend_weights.squeeze(0)
        
        return pos_blend_weights, color_blend_weights

# --- Placeholder for DeformationModule ---
class DeformationModule(nn.Module):
    def __init__(self, config, num_training_frames, n_initial_gaussians, use_tcnn=True, *, smpl_num_verts: int | None = None):
        super().__init__()
        self.config = config # Store sub-configs for networks if needed
        self.n_initial_gaussians = n_initial_gaussians
        self.use_tcnn = use_tcnn
        self.smpl_num_verts = int(smpl_num_verts) if smpl_num_verts is not None else None
        self.backend = str(config.get("backend", "mlp")).lower()

        # Per-frame latent codes
        self.latent_code_dim = config.get('latent_code_dim', 32)
        self.frame_latent_codes = nn.Embedding(num_training_frames, self.latent_code_dim)
        latent_init_std = float(config.get('latent_init_std', 0.0))
        if latent_init_std <= 0:
            torch.nn.init.zeros_(self.frame_latent_codes.weight.data)
        else:
            torch.nn.init.normal_(self.frame_latent_codes.weight.data, 0.0, latent_init_std)

        # Vertex Deformation Network
        tcnn_zero_head = bool(config.get('tcnn_zero_head', True))
        tcnn_head_dim = config.get('tcnn_head_dim', None)
        # v2 (tao_offset_input_enhancement.md): support configurable vertex-MLP input modes.
        # Keep legacy key `pos_encoder_L` as the default for xyz PE length; accept `pe_L_xyz` alias.
        pe_L_xyz = config.get('pe_L_xyz', config.get('pos_encoder_L', 6))
        if self.backend == "litept":
            self.vertex_deformation_net = LitePTVertexDeformationNetwork(
                num_pose_params=config.get('num_pose_params', 63),
                latent_code_dim=self.latent_code_dim,
                litept_cfg=config.get('litept', {}),
                input_mode=config.get('input_mode', 'xyz'),
            )
        else:
            self.backend = "mlp"
            self.vertex_deformation_net = VertexDeformationNetwork(
                pos_encoder_L=pe_L_xyz,
                num_pose_params=config.get('num_pose_params', 63),
                latent_code_dim=self.latent_code_dim,
                mlp_hidden_layers=config.get('vert_mlp_hidden_layers', 5),
                mlp_hidden_dim=config.get('vert_mlp_hidden_dim', 64),
                use_tcnn=self.use_tcnn,
                tcnn_zero_head=tcnn_zero_head,
                tcnn_head_dim=tcnn_head_dim,
                input_mode=config.get('input_mode', 'xyz'),
                pe_L_rlocal=config.get('pe_L_rlocal', 6),
                smpl_embed_dim=config.get('smpl_embed_dim', 32),
                use_rnorm=config.get('use_rnorm', False),
                use_smpl_curvature=config.get('use_smpl_curvature', False),
                smpl_num_verts=self.smpl_num_verts,
                log_input_stats=config.get('log_input_stats', False),
                assert_same_cano_frame=config.get('assert_same_cano_frame', False),
            )

        # Gaussian Blendshape Network and Embeddings
        self.num_blendshapes = config.get('num_blendshapes', 20)
        self.gaussian_blendshape_net = GaussianBlendshapeNetwork(
            num_pose_params=config.get('num_pose_params', 63),
            num_blendshapes=self.num_blendshapes,
            mlp_hidden_layers=config.get('gauss_mlp_hidden_layers', 5),
            mlp_hidden_dim=config.get('gauss_mlp_hidden_dim', 64),
            use_tcnn=self.use_tcnn,
            tcnn_zero_head=tcnn_zero_head,
            tcnn_head_dim=tcnn_head_dim,
        )

        # Learnable blendshape embeddings (Shape: [num_blendshapes, N_gaussians, 3])
        # Initialize small to not drastically alter initial state
        self.pos_blendshapes = nn.Parameter(
            torch.randn(self.num_blendshapes, self.n_initial_gaussians, 3) * config.get('pos_blendshape_init_scale', 0.001)
        )
        self.color_blendshapes = nn.Parameter(
            torch.randn(self.num_blendshapes, self.n_initial_gaussians, 3) * config.get('color_blendshape_init_scale', 0.001)
        )

    def get_frame_latent_code(self, frame_idx):
        # frame_idx should be a scalar or a tensor that can index nn.Embedding
        return self.frame_latent_codes(frame_idx)

    def forward_vertex_deform(self, cano_verts, pose_params, frame_latent_code, garment_mask, **kwargs):
        """
        Deforms canonical vertices.
        cano_verts: (N_verts, 3)
        pose_params: (63,) --- SMPLX pose parameters for current frame
        frame_latent_code: (latent_dim,) --- Learned latent code for current frame
        garment_mask: (N_verts,) --- Boolean mask for garment vertices
        Returns: deformed_verts (N_verts, 3)
        """
        vertex_offsets = self.vertex_deformation_net(
            cano_verts,
            pose_params,
            frame_latent_code,
            garment_mask,
            **kwargs,
        )
        return cano_verts + vertex_offsets

    def forward_gaussian_deform(self, pose_params, *, return_weights: bool = False):
        """
        Computes deformations for the initial set of Gaussians based on pose.
        This returns deltas for position and color for the *initial* N_initial_gaussians.
        The calling code is responsible for applying these to the correct subset of active Gaussians.
        
        pose_params: (63,) --- SMPLX pose parameters for current frame
        Returns: 
            delta_xyz (N_initial_gaussians, 3)
            delta_rgb (N_initial_gaussians, 3)
        """
        pos_blend_weights, color_blend_weights = self.gaussian_blendshape_net(pose_params)
        # pos_blend_weights: (num_blendshapes,)
        # color_blend_weights: (num_blendshapes,)

        # Einsum for weighted sum of blendshapes:
        # 'b,bgs->gs'  b: num_blendshapes, g: n_initial_gaussians, s: 3 (xyz or rgb)
        delta_xyz = torch.einsum('b,bgs->gs', pos_blend_weights, self.pos_blendshapes)
        delta_rgb = torch.einsum('b,bgs->gs', color_blend_weights, self.color_blendshapes)

        if return_weights:
            return delta_xyz, delta_rgb, pos_blend_weights, color_blend_weights
        return delta_xyz, delta_rgb 

    def get_param_groups(self):
        groups = {
            "deform_latent_codes": [self.frame_latent_codes.weight],
            "deform_gauss_mlp_pos": list(self.gaussian_blendshape_net.mlp_pos_blendshapes.parameters()),
            "deform_gauss_mlp_color": list(self.gaussian_blendshape_net.mlp_color_blendshapes.parameters()),
            "deform_blendshapes_pos": [self.pos_blendshapes],
            "deform_blendshapes_color": [self.color_blendshapes],
        }
        if self.backend == "litept" and hasattr(self.vertex_deformation_net, "get_litept_param_groups"):
            groups.update(self.vertex_deformation_net.get_litept_param_groups())
        else:
            groups["deform_vert_mlp"] = list(self.vertex_deformation_net.parameters())
        return groups