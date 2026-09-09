from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.scripts.smplx_fitting.fit_mvhumannet_smplx_to_neus2 import run_finetune


def _expand_pipeline(pipeline: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for item in pipeline:
        if "stage" in item:
            out.append(str(item["stage"]))
            continue
        if "group" in item:
            repeat = int(item.get("repeat", 1))
            stages = item.get("stages", [])
            expanded = _expand_pipeline(stages)
            for _ in range(repeat):
                out.extend(expanded)
            continue
        raise ValueError(f"Unknown pipeline item: {item}")
    return out


def _stage_name_safe(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("/", "-").replace("\\", "-")


def _resolve_gender(subject_dir: Path, gender_arg: str) -> str:

    g = str(gender_arg).strip().lower()
    if g in {"male", "female", "neutral"}:
        return g
    if g in {"auto", ""}:
        gpath = subject_dir / "gender.txt"
        if gpath.exists():
            try:
                txt = gpath.read_text().strip().lower()
                if txt in {"m", "male"}:
                    return "male"
                if txt in {"f", "female"}:
                    return "female"
                if txt in {"n", "neutral"}:
                    return "neutral"
            except Exception:
                pass
    return "neutral"


def _exp_suffix(exp_name: str) -> str:
    exp_name_clean = str(exp_name).strip().replace("/", "-").replace("\\", "-").replace(" ", "_")
    return f"_{exp_name_clean}" if exp_name_clean else ""


def main() -> None:
    ap = argparse.ArgumentParser("Run SMPL-X fitting from YAML config")
    ap.add_argument("--config", type=str, required=True, help="YAML config path")
    ap.add_argument("--subject_dir", type=str, default=None, help="Override global.subject_dir in YAML")
    ap.add_argument("--neus_mesh_path", type=str, default=None, help="Override global.neus_mesh_path in YAML")
    ap.add_argument("--calib_path", type=str, default=None, help="Override global.calib_path in YAML")
    ap.add_argument("--smpl_model_path", type=str, default=None, help="Override global.smpl_model_path in YAML")
    ap.add_argument("--gender", type=str, default=None, help="Override global.gender in YAML (male|female|neutral|auto)")
    ap.add_argument("--exp_name", type=str, default=None, help="Override global.exp_name in YAML")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    global_cfg = dict(cfg.get("global", {}))
    stages_cfg = dict(cfg.get("stages", {}))
    pipeline_cfg = cfg.get("pipeline", [])


    if args.subject_dir:
        subject_dir = Path(args.subject_dir)
        global_cfg.pop("subject_dir", None)
    else:
        subject_dir = Path(global_cfg.pop("subject_dir"))

    if args.neus_mesh_path:
        neus_mesh_path = Path(args.neus_mesh_path)
        global_cfg.pop("neus_mesh_path", None)
    else:
        neus_mesh_path = Path(global_cfg.pop("neus_mesh_path"))

    if args.calib_path:
        calib_path = Path(args.calib_path)
        global_cfg.pop("calib_path", None)
    else:
        calib_path_val = global_cfg.pop("calib_path", None)
        calib_path = Path(calib_path_val) if calib_path_val else (subject_dir / "calibration_full.json")

    if args.smpl_model_path:
        smpl_model_path = Path(args.smpl_model_path)
        global_cfg.pop("smpl_model_path", None)
    else:
        smpl_model_path = Path(
            global_cfg.pop("smpl_model_path", str(Path(os.environ.get(
                "AVATARMIX_ASSET_ROOT", Path(__file__).resolve().parents[4] / "external_assets"
            )) / "smpl"))
        )


    for k in ("subject_dir", "neus_mesh_path", "calib_path", "smpl_model_path"):
        global_cfg.pop(k, None)

    if args.gender is not None:
        global_cfg["gender"] = str(args.gender)
    if args.exp_name is not None:
        global_cfg["exp_name"] = str(args.exp_name)


    smpl_params_path = subject_dir / "smpl_params.npz"
    smpl_params_old_path = subject_dir / "smpl_params_old.npz"
    if smpl_params_path.exists() and not smpl_params_old_path.exists():
        shutil.copy2(smpl_params_path, smpl_params_old_path)
    elif (not smpl_params_path.exists()) and smpl_params_old_path.exists():

        shutil.copy2(smpl_params_old_path, smpl_params_path)


    stage_names = _expand_pipeline(list(pipeline_cfg))
    pipeline: List[Dict[str, Any]] = []
    for sname in stage_names:
        sname = _stage_name_safe(sname)
        sdef = dict(stages_cfg.get(sname, {}))
        stype = sdef.get("type", "smplx_fitting")
        if stype != "smplx_fitting":
            raise ValueError(f"Unsupported stage type '{stype}' for stage '{sname}'")
        params = dict(sdef.get("params", {}))
        pipeline.append({"name": sname, **params})


    run_finetune(
        subject_dir=subject_dir,
        smpl_model_path=smpl_model_path,
        neus_mesh_path=neus_mesh_path,
        calib_path=calib_path,
        pipeline=pipeline,
        **global_cfg,
    )


    gender_used = _resolve_gender(subject_dir, str(global_cfg.get("gender", "auto")))
    exp_name = str(global_cfg.get("exp_name", "")).strip()
    finetuned_path = subject_dir / f"smpl_params_finetuned_{gender_used}{_exp_suffix(exp_name)}.npz"
    if not finetuned_path.exists():

        cands = list(subject_dir.glob(f"smpl_params_finetuned_{gender_used}*.npz"))
        if cands:
            finetuned_path = max(cands, key=lambda p: p.stat().st_mtime)
    if finetuned_path.exists():
        shutil.copy2(finetuned_path, smpl_params_path)
    else:
        raise FileNotFoundError(
            f"Fitting finished but finetuned params not found. Expected: {finetuned_path} "
            f"(subject_dir={subject_dir}, gender={gender_used}, exp_name={exp_name})"
        )


if __name__ == "__main__":
    main()
