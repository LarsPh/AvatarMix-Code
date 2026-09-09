import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _safe_std(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s = x.std(axis=0, keepdims=True)
    s[s < eps] = 1.0
    return s


def _to_theta63(body_pose_np: np.ndarray, joint_indices_21: Sequence[int]) -> np.ndarray:
    """Convert various SMPLX pose array shapes into (T,63) axis-angle feature.

    Supported:
    - (T,63): returned
    - (T,J,3): slice J using joint_indices_21 -> (T,21,3) -> (T,63)
    - (T,3*J): reshape to (T,J,3) then slice -> (T,63)
    """
    if body_pose_np.ndim == 2 and body_pose_np.shape[1] == 63:
        return body_pose_np.astype(np.float32, copy=False)

    if body_pose_np.ndim == 3 and body_pose_np.shape[-1] == 3:
        J = body_pose_np.shape[1]
        max_idx = max(joint_indices_21)
        if J <= max_idx:
            raise ValueError(f"body_pose has J={J}, but require index up to {max_idx}")
        x = body_pose_np[:, joint_indices_21, :].reshape(body_pose_np.shape[0], -1)
        return x.astype(np.float32, copy=False)

    if body_pose_np.ndim == 2 and (body_pose_np.shape[1] % 3 == 0):
        J = body_pose_np.shape[1] // 3
        x3 = body_pose_np.reshape(body_pose_np.shape[0], J, 3)
        max_idx = max(joint_indices_21)
        if J <= max_idx:
            raise ValueError(f"body_pose has J={J}, but require index up to {max_idx}")
        x = x3[:, joint_indices_21, :].reshape(body_pose_np.shape[0], -1)
        return x.astype(np.float32, copy=False)

    raise ValueError(f"Unsupported body_pose shape: {tuple(body_pose_np.shape)}")


def greedy_fps_select(
    feat: np.ndarray,
    K: int,
    seed_mode: str = "min_norm",
    candidate_stride: int = 1,
    rng: Optional[np.random.RandomState] = None,
) -> List[int]:
    """Greedy farthest point sampling on rows of feat.

    Returns selected row indices in increasing order (unique).
    """
    if rng is None:
        rng = np.random.RandomState(0)
    if feat.ndim != 2:
        raise ValueError(f"feat must be (T,D), got {tuple(feat.shape)}")
    T = feat.shape[0]
    if T == 0:
        return []
    stride = max(int(candidate_stride), 1)
    candidates = np.arange(0, T, stride, dtype=np.int64)
    if candidates.size == 0:
        candidates = np.arange(T, dtype=np.int64)

    X = feat[candidates]
    if K <= 0:
        return []
    K = min(int(K), X.shape[0])

    if seed_mode == "random":
        seed_local = int(rng.randint(0, X.shape[0]))
    else:
        # closest to rest: minimal L2 norm
        seed_local = int(np.argmin((X * X).sum(axis=1)))

    selected_local = [seed_local]
    # distances to nearest selected center (squared L2)
    D = ((X - X[seed_local : seed_local + 1]) ** 2).sum(axis=1)

    for _ in range(1, K):
        new_local = int(np.argmax(D))
        selected_local.append(new_local)
        d_new = ((X - X[new_local : new_local + 1]) ** 2).sum(axis=1)
        D = np.minimum(D, d_new)

    selected_global = np.unique(candidates[np.array(selected_local, dtype=np.int64)])
    return selected_global.tolist()


def greedy_fps_select_with_scores(
    feat: np.ndarray,
    K: int,
    seed_mode: str = "min_norm",
    candidate_stride: int = 1,
    min_interval: int = 0,
    rng: Optional[np.random.RandomState] = None,
) -> dict:
    """Greedy farthest point sampling that preserves selection order and novelty scores.

    Returns:
      {
        "selected_global": List[int],   # global indices into feat, in SELECTION ORDER
        "selected_local": List[int],    # local indices into X=feat[candidates]
        "candidates": np.ndarray[int],  # local->global mapping
        "novelty": List[float],         # novelty at selection time (squared L2); novelty[0]=0 for seed
      }
    """
    if rng is None:
        rng = np.random.RandomState(0)
    if feat.ndim != 2:
        raise ValueError(f"feat must be (T,D), got {tuple(feat.shape)}")
    T = feat.shape[0]
    if T == 0 or K <= 0:
        return {"selected_global": [], "selected_local": [], "candidates": np.arange(0, 0, dtype=np.int64), "novelty": []}

    stride = max(int(candidate_stride), 1)
    candidates = np.arange(0, T, stride, dtype=np.int64)
    if candidates.size == 0:
        candidates = np.arange(T, dtype=np.int64)

    X = feat[candidates]
    K = min(int(K), int(X.shape[0]))

    if seed_mode == "random":
        seed_local = int(rng.randint(0, X.shape[0]))
    else:
        seed_local = int(np.argmin((X * X).sum(axis=1)))

    selected_local: List[int] = [seed_local]
    novelty: List[float] = [0.0]

    # distances to nearest selected center (squared L2)
    D = ((X - X[seed_local : seed_local + 1]) ** 2).sum(axis=1)
    # Optional: enforce a minimum absolute interval in global frame index between selected points.
    # This is a hard constraint on `candidates[local]` values (indices into original feat rows).
    min_interval = max(int(min_interval), 0)
    allow = np.ones((X.shape[0],), dtype=bool)
    if min_interval > 0:
        seed_g = int(candidates[seed_local])
        allow &= (np.abs(candidates - seed_g) >= min_interval)
        allow[seed_local] = True  # always keep already selected valid

    for _ in range(1, K):
        if min_interval > 0:
            Dm = np.where(allow, D, -np.inf)
            new_local = int(np.argmax(Dm))
            if not np.isfinite(Dm[new_local]):
                break
        else:
            new_local = int(np.argmax(D))
        selected_local.append(new_local)
        novelty.append(float(D[new_local]))
        d_new = ((X - X[new_local : new_local + 1]) ** 2).sum(axis=1)
        D = np.minimum(D, d_new)
        if min_interval > 0:
            g = int(candidates[new_local])
            allow &= (np.abs(candidates - g) >= min_interval)
            allow[new_local] = True

    selected_global = candidates[np.array(selected_local, dtype=np.int64)].tolist()
    return {
        "selected_global": [int(i) for i in selected_global],
        "selected_local": [int(i) for i in selected_local],
        "candidates": candidates,
        "novelty": [float(x) for x in novelty],
    }

def build_s_div_and_w_win(
    frame_id_list: Sequence[str],
    body_pose_np: np.ndarray,
    joint_indices_21: Sequence[int],
    fps_k: int,
    fps_seed_mode: str,
    fps_candidate_stride: int,
    win_m: int,
    win_len: int = 3,
    win_min_gap: int = 0,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[List[str], List[str]]:
    """Build S_div (pose-diverse frames) and W_win (window start frames) as frame_id_str lists."""
    if rng is None:
        rng = np.random.RandomState(0)
    if win_len != 3:
        raise ValueError("This implementation currently assumes win_len=3.")

    theta63 = _to_theta63(body_pose_np, joint_indices_21)
    theta63 = np.clip(theta63, -math.pi, math.pi)
    mu = theta63.mean(axis=0, keepdims=True)
    sd = _safe_std(theta63)
    feat = (theta63 - mu) / sd

    sel_idx = greedy_fps_select(
        feat=feat,
        K=fps_k,
        seed_mode=fps_seed_mode,
        candidate_stride=fps_candidate_stride,
        rng=rng,
    )
    s_div = [frame_id_list[i] for i in sel_idx]
    s_div = sorted(list(dict.fromkeys(s_div)))

    # window starts: must be valid indices such that t,t+1,t+2 exist as frame IDs
    frame_ids_set = set(frame_id_list)
    candidates = []
    for fid in s_div:
        try:
            t = int(fid)
        except Exception:
            continue
        # require existence of fid, fid+1, fid+2 in this split
        f1 = f"{t+1:0{len(fid)}d}"
        f2 = f"{t+2:0{len(fid)}d}"
        if f1 in frame_ids_set and f2 in frame_ids_set:
            candidates.append(fid)

    rng.shuffle(candidates)
    starts = []
    min_gap = max(int(win_min_gap), 0)
    for fid in candidates:
        if len(starts) >= int(win_m):
            break
        if min_gap > 0:
            t = int(fid)
            ok = True
            for s in starts:
                if abs(int(s) - t) < min_gap:
                    ok = False
                    break
            if not ok:
                continue
        starts.append(fid)
    starts = sorted(starts)
    return s_div, starts


@dataclass
class MixedSamplerConfig:
    enable_pose_fps: bool = True
    fps_k: int = 1000
    fps_seed_mode: str = "min_norm"
    fps_candidate_stride: int = 1

    window_len: int = 3
    win_m: int = 200
    win_min_gap: int = 0

    p_div: float = 0.7
    p_win: float = 0.3
    sampler_seed: int = 0


class MixedDivWinSampler(torch.utils.data.Sampler[int]):
    """Stateful mixed sampler over existing (frame,cam) dataset indices.

    - yields a single dataset index each iteration
    - maintains an internal queue for 3-frame windows so they are yielded consecutively
    """

    def __init__(
        self,
        data_samples: Sequence[Tuple[str, str]],
        frame_id_list: Sequence[str],
        body_pose_np: np.ndarray,
        fullbody_cam_ids: Sequence[str],
        joint_indices_21: Sequence[int],
        cfg: MixedSamplerConfig,
        max_resample_tries: int = 50,
    ):
        super().__init__(None)
        self.data_samples = list(data_samples)
        self.frame_id_list = list(frame_id_list)
        self.body_pose_np = body_pose_np
        self.fullbody_cam_ids = list(fullbody_cam_ids)
        self.joint_indices_21 = list(joint_indices_21)
        self.cfg = cfg
        self.max_resample_tries = int(max_resample_tries)

        if len(self.fullbody_cam_ids) == 0:
            raise ValueError("fullbody_cam_ids is empty; cannot sample cameras.")

        self.sample_to_idx: Dict[Tuple[str, str], int] = {
            (f, c): i for i, (f, c) in enumerate(self.data_samples)
        }
        self.frame_to_cams: Dict[str, List[str]] = {}
        for f, c in self.data_samples:
            self.frame_to_cams.setdefault(f, []).append(c)

        self.rng = np.random.RandomState(int(cfg.sampler_seed))
        self.queue: List[Tuple[str, str]] = []

        self.s_div_frame_ids, self.win_start_frame_ids = build_s_div_and_w_win(
            frame_id_list=self.frame_id_list,
            body_pose_np=self.body_pose_np,
            joint_indices_21=self.joint_indices_21,
            fps_k=int(cfg.fps_k),
            fps_seed_mode=str(cfg.fps_seed_mode),
            fps_candidate_stride=int(cfg.fps_candidate_stride),
            win_m=int(cfg.win_m),
            win_len=int(cfg.window_len),
            win_min_gap=int(cfg.win_min_gap),
            rng=self.rng,
        )

    def __iter__(self) -> Iterable[int]:
        # Infinite sampler: training loop is iteration-limited.
        while True:
            idx = self._next_index()
            yield idx

    def __len__(self) -> int:
        # Large sentinel length; DataLoader won't exhaust in typical iteration-limited training.
        return 2**31 - 1

    def _pick_fullbody_cam(self) -> str:
        return self.fullbody_cam_ids[int(self.rng.randint(0, len(self.fullbody_cam_ids)))]

    def _resolve_pair_to_idx(self, frame_id: str, cam_id: str) -> Optional[int]:
        return self.sample_to_idx.get((frame_id, cam_id), None)

    def _fallback_any_cam_for_frame(self, frame_id: str) -> Optional[int]:
        cams = self.frame_to_cams.get(frame_id, None)
        if not cams:
            return None
        cam_id = cams[int(self.rng.randint(0, len(cams)))]
        return self.sample_to_idx.get((frame_id, cam_id), None)

    def _enqueue_window(self, start_fid: str, cam_id: str):
        t = int(start_fid)
        z = len(start_fid)
        f0 = start_fid
        f1 = f"{t+1:0{z}d}"
        f2 = f"{t+2:0{z}d}"
        self.queue.extend([(f0, cam_id), (f1, cam_id), (f2, cam_id)])

    def _next_index(self) -> int:
        # Drain queued window samples first (ensures consecutive iterations).
        if self.queue:
            f, c = self.queue.pop(0)
            idx = self._resolve_pair_to_idx(f, c)
            if idx is not None:
                return idx
            # If missing (frame/cam not on disk), fallback within frame.
            idx2 = self._fallback_any_cam_for_frame(f)
            if idx2 is not None:
                return idx2
            # As last resort, keep sampling a fresh draw.

        p_win = float(self.cfg.p_win)
        p_div = float(self.cfg.p_div)
        # normalize if user config is inconsistent
        s = max(p_win + p_div, 1e-6)
        p_win /= s

        for _ in range(self.max_resample_tries):
            if self.rng.rand() < p_win and len(self.win_start_frame_ids) > 0:
                start_fid = self.win_start_frame_ids[int(self.rng.randint(0, len(self.win_start_frame_ids)))]
                cam_id = self._pick_fullbody_cam()
                self._enqueue_window(start_fid, cam_id)
                f, c = self.queue.pop(0)
                idx = self._resolve_pair_to_idx(f, c)
                if idx is not None:
                    return idx
                idx2 = self._fallback_any_cam_for_frame(f)
                if idx2 is not None:
                    return idx2
            else:
                fid = self.s_div_frame_ids[int(self.rng.randint(0, len(self.s_div_frame_ids)))]
                cam_id = self._pick_fullbody_cam()
                idx = self._resolve_pair_to_idx(fid, cam_id)
                if idx is not None:
                    return idx
                idx2 = self._fallback_any_cam_for_frame(fid)
                if idx2 is not None:
                    return idx2

        # Final fallback: uniform over existing samples.
        return int(self.rng.randint(0, len(self.data_samples)))


def save_sampling_artifacts(
    out_dir: Path,
    fullbody_cam_ids: Sequence[str],
    s_div_frame_ids: Sequence[str],
    win_start_frame_ids: Sequence[str],
    cfg_dict: dict,
    *,
    frame_id_list: Optional[Sequence[str]] = None,
    body_pose_np: Optional[np.ndarray] = None,
    joint_indices_21: Optional[Sequence[int]] = None,
    export_pose_diverse_cfg: Optional[dict] = None,
):
    """Save sampler artifacts.

    Backward compatible: always writes the existing three JSONs + sampling_config.json to `out_dir`.

    Optional: when export_pose_diverse_cfg.enabled, also writes pose-diversity JSONs under
    `out_dir/<out_subdir>/` and returns a small dict for downstream quicklook dumping.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sampling_fullbody_cams.json").write_text(json.dumps({"fullbody_cam_ids": list(fullbody_cam_ids)}, indent=2))
    (out_dir / "sampling_s_div.json").write_text(json.dumps({"s_div_frame_ids": list(s_div_frame_ids)}, indent=2))
    (out_dir / "sampling_w_win.json").write_text(json.dumps({"win_start_frame_ids": list(win_start_frame_ids)}, indent=2))
    # Convert any OmegaConf containers (DictConfig/ListConfig) to plain Python types.
    def _to_basic(obj):
        try:
            from omegaconf import DictConfig, ListConfig
        except Exception:
            DictConfig = tuple()  # unused
            ListConfig = tuple()

        if isinstance(obj, dict):
            return {k: _to_basic(v) for k, v in obj.items()}
        # OmegaConf list-like
        if obj.__class__.__name__ in ("ListConfig",) or getattr(obj, "_is_list_config", False):
            return [_to_basic(v) for v in list(obj)]
        # OmegaConf dict-like
        if obj.__class__.__name__ in ("DictConfig",) or getattr(obj, "_is_dict_config", False):
            return {k: _to_basic(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_basic(v) for v in obj]
        return obj

    cfg_serializable = _to_basic(cfg_dict)
    (out_dir / "sampling_config.json").write_text(json.dumps(cfg_serializable, indent=2))

    # Optional pose-diverse export (top-k novelty) used for static recon planning.
    export_info = None
    try:
        epd = export_pose_diverse_cfg
        if epd is not None and hasattr(epd, "get"):
            enabled = bool(epd.get("enabled", False))
        elif isinstance(epd, dict):
            enabled = bool(epd.get("enabled", False))
        else:
            enabled = False
        if enabled:
            if frame_id_list is None or body_pose_np is None or joint_indices_21 is None:
                raise ValueError("export_pose_diverse.enabled requires frame_id_list, body_pose_np, joint_indices_21")

            # Config defaults
            fps_pool_k = int(epd.get("fps_pool_k", 32))
            export_topk = int(epd.get("export_topk", 12))
            seed_mode = str(epd.get("seed_mode", "min_norm"))
            cand_stride = int(epd.get("candidate_stride", 2))
            min_interval_fps = int(epd.get("min_interval_fps", 0) or 0)
            min_interval_topk = int(epd.get("min_interval_topk", 0) or 0)
            out_subdir = str(epd.get("out_subdir", "sampling_artifacts"))
            ranked_json_name = str(epd.get("ranked_json_name", "pose_diverse_ranked.json"))
            topk_json_name = str(epd.get("topk_json_name", "pose_diverse_topk.json"))

            fixed_cam_id = str(epd.get("fixed_cam_id", "auto"))
            auto_cam_policy = str(epd.get("auto_cam_policy", "first"))

            # Choose fixed camera id (string)
            cams_sorted = sorted([str(c) for c in list(fullbody_cam_ids)])
            fixed_cam_used = None
            if fixed_cam_id != "auto":
                fixed_cam_used = str(fixed_cam_id)
            else:
                if not cams_sorted:
                    fixed_cam_used = "cam"
                elif auto_cam_policy == "median":
                    fixed_cam_used = cams_sorted[len(cams_sorted) // 2]
                else:
                    fixed_cam_used = cams_sorted[0]

            # Compute theta63 feature exactly like build_s_div_and_w_win()
            theta63 = _to_theta63(body_pose_np, joint_indices_21)
            theta63 = np.clip(theta63, -math.pi, math.pi)
            mu = theta63.mean(axis=0, keepdims=True)
            sd = _safe_std(theta63)
            feat = (theta63 - mu) / sd

            rng = np.random.RandomState(int(epd.get("rng_seed", 0)))
            fps_pool_k = max(1, min(int(fps_pool_k), int(feat.shape[0])))
            export_topk = max(1, min(int(export_topk), int(fps_pool_k)))

            sel = greedy_fps_select_with_scores(
                feat=feat,
                K=int(fps_pool_k),
                seed_mode=seed_mode,
                candidate_stride=int(cand_stride),
                min_interval=int(min_interval_fps),
                rng=rng,
            )
            sel_idx = [int(i) for i in sel.get("selected_global", [])]
            novelty = [float(x) for x in sel.get("novelty", [])]

            fid_list = list(frame_id_list)
            entries_sel = []
            for i, idx in enumerate(sel_idx):
                if idx < 0 or idx >= len(fid_list):
                    continue
                entries_sel.append(
                    {
                        "rank_by_selection": int(i),
                        "frame_index": int(idx),
                        "frame_id": str(fid_list[idx]),
                        "novelty": float(novelty[i]) if i < len(novelty) else 0.0,
                    }
                )

            # Ranked by novelty (descending)
            entries_ranked = sorted(entries_sel, key=lambda e: float(e.get("novelty", 0.0)), reverse=True)
            if min_interval_topk > 0:
                kept = []
                kept_t = []
                for e in entries_ranked:
                    if len(kept) >= int(export_topk):
                        break
                    try:
                        t = int(e.get("frame_id", e.get("frame_index", 0)))
                    except Exception:
                        t = int(e.get("frame_index", 0))
                    ok = True
                    for tt in kept_t:
                        if abs(int(tt) - int(t)) < int(min_interval_topk):
                            ok = False
                            break
                    if ok:
                        kept.append(e)
                        kept_t.append(t)
                topk_entries = kept
            else:
                topk_entries = entries_ranked[: int(export_topk)]

            out_root = out_dir / str(out_subdir)
            out_root.mkdir(parents=True, exist_ok=True)

            ranked_payload = {
                "method": "greedy_fps_theta63",
                "fps_pool_k": int(fps_pool_k),
                "export_topk": int(export_topk),
                "seed_mode": str(seed_mode),
                "candidate_stride": int(cand_stride),
                "min_interval_fps": int(min_interval_fps),
                "min_interval_topk": int(min_interval_topk),
                "feature": {"type": "theta63", "clip": "[-pi,pi]", "normalize": "(x-mu)/std"},
                "fixed_cam_id_used": str(fixed_cam_used),
                "selection_order": [e["frame_id"] for e in entries_sel],
                "entries_selection_order": entries_sel,
                "entries_ranked_by_novelty": entries_ranked,
            }
            (out_root / ranked_json_name).write_text(json.dumps(ranked_payload, indent=2))

            topk_payload = {
                "fixed_cam_id_used": str(fixed_cam_used),
                "topk_ranked_by_novelty": topk_entries,
            }
            (out_root / topk_json_name).write_text(json.dumps(topk_payload, indent=2))

            export_info = {
                "fixed_cam_id_used": str(fixed_cam_used),
                "topk_entries": topk_entries,
                "out_root": str(out_root),
                "img_dir_topk": str(epd.get("img_dir_topk", "pose_diverse_topk_imgs")),
            }
    except Exception:
        export_info = None

    return export_info

