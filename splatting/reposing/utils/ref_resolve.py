from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re


@dataclass(frozen=True)
class ReferenceAssets:
    reference_frame_idx_for_smpl: Optional[int]
    reference_smpl_params_path: Optional[str]
    lbs_weights_path: Optional[str]

    inferred_ref_frame_from_name: Optional[int]


_FRAME_SUFFIX_RE = re.compile(r"_(\d{3,})$")
_ACTOR_SEQ_RE = re.compile(r"(Actor\d+_Seq\d+)")
_SUBACTOR_SEQ_RE = re.compile(r"subActor(\d+)_Seq(\d+)", re.IGNORECASE)


def resolve_reference_assets_from_pretrained_gs_from(pretrained_gs_from: str) -> ReferenceAssets:

    p = Path(str(pretrained_gs_from))
    exp_name = _infer_exp_name_from_pretrained_path(p)
    ref_frame = _infer_frame_idx_from_name(exp_name or p.name)

    base_dir = _infer_base_dir_from_pretrained_path(p)
    seq_id = _infer_seq_id_from_exp_name(exp_name or p.name)

    ref_smpl_path = None
    ref_lbs_path = None

    if base_dir is not None and seq_id is not None and ref_frame is not None:

        ref_frame_str = _infer_frame_str_from_name(exp_name or p.name) or f"{ref_frame:06d}"
        ref_single = base_dir / f"{seq_id}_{ref_frame_str}"
        if ref_single.exists():
            cand_smpl = ref_single / "smpl_params.npz"
            cand_lbs = ref_single / "mesh" / "processed" / "smoothed_inpainted_weights.npy"
            if cand_smpl.exists():
                ref_smpl_path = str(cand_smpl)
            if cand_lbs.exists():
                ref_lbs_path = str(cand_lbs)
        else:

            pattern = f"*{seq_id}*_{ref_frame_str}"
            for cand in base_dir.glob(pattern):
                if cand.is_dir():
                    cand_smpl = cand / "smpl_params.npz"
                    cand_lbs = cand / "mesh" / "processed" / "smoothed_inpainted_weights.npy"
                    if ref_smpl_path is None and cand_smpl.exists():
                        ref_smpl_path = str(cand_smpl)
                    if ref_lbs_path is None and cand_lbs.exists():
                        ref_lbs_path = str(cand_lbs)
                    if ref_smpl_path is not None and ref_lbs_path is not None:
                        break

    return ReferenceAssets(
        reference_frame_idx_for_smpl=ref_frame,
        reference_smpl_params_path=ref_smpl_path,
        lbs_weights_path=ref_lbs_path,
        inferred_ref_frame_from_name=ref_frame,
    )


def apply_reference_assets_overrides_to_config(config, assets: ReferenceAssets) -> None:

    try:
        avatarrex_cfg = config.dataset.avatarrex_config
    except Exception:
        return

    def _is_missing(v) -> bool:
        return v is None or (isinstance(v, str) and v.strip() == "")

    if assets.reference_frame_idx_for_smpl is not None:
        try:
            cur = getattr(avatarrex_cfg, "reference_frame_idx_for_smpl", None)
            if _is_missing(cur):
                avatarrex_cfg.reference_frame_idx_for_smpl = int(assets.reference_frame_idx_for_smpl)
        except Exception:
            pass

    if assets.reference_smpl_params_path is not None:
        try:
            cur = getattr(avatarrex_cfg, "reference_smpl_params_path", None)
            if _is_missing(cur):
                avatarrex_cfg.reference_smpl_params_path = str(assets.reference_smpl_params_path)
        except Exception:
            pass

    if assets.lbs_weights_path is not None:
        try:
            cur = getattr(avatarrex_cfg, "lbs_weights_path", None)
            if _is_missing(cur):
                avatarrex_cfg.lbs_weights_path = str(assets.lbs_weights_path)
        except Exception:
            pass


def _infer_base_dir_from_pretrained_path(p: Path) -> Optional[Path]:

    parts = list(p.parts)
    if "output-splatting" in parts:
        i = parts.index("output-splatting")
        if i > 0:
            return Path(*parts[:i])

    if len(p.parents) >= 3:
        return p.parents[2]
    return None


def _infer_exp_name_from_pretrained_path(p: Path) -> Optional[str]:

    parts = list(p.parts)
    if "output-splatting" in parts:
        i = parts.index("output-splatting")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def _infer_frame_str_from_name(name: str) -> Optional[str]:
    m = _FRAME_SUFFIX_RE.search(str(name))
    return m.group(1) if m else None


def _infer_frame_idx_from_name(name: str) -> Optional[int]:
    s = _infer_frame_str_from_name(name)
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _infer_seq_id_from_exp_name(exp_name: str) -> Optional[str]:
    if not exp_name:
        return None
    m = _ACTOR_SEQ_RE.search(exp_name)
    if m:
        return m.group(1)
    m2 = _SUBACTOR_SEQ_RE.search(exp_name)
    if m2:
        actor_num = int(m2.group(1))
        seq_num = int(m2.group(2))
        return f"Actor{actor_num:02d}_Seq{seq_num}"
    return None
