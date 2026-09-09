from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Set

import yaml


_SUBJECT_RE = re.compile(r"^\d{4}$")


def _norm_subject_id(x) -> str:

    if isinstance(x, int):
        return f"{x:04d}"
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            return f"{int(s):04d}"
    raise ValueError(f"Invalid subject id value in YAML: {x!r} (expected int or digit string)")


def _find_values_for_key(obj, key: str):

    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found.extend(_find_values_for_key(v, key))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_find_values_for_key(v, key))
    return found


def _load_female_subjects(yaml_path: Path) -> Set[str]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text())


    if isinstance(data, dict) and "female_subjects" in data:
        female_raw = data["female_subjects"]
        if not isinstance(female_raw, list):
            raise ValueError("'female_subjects' must be a list")
        return {_norm_subject_id(v) for v in female_raw}


    candidates = _find_values_for_key(data, "female_subjects")
    candidates = [c for c in candidates if isinstance(c, list)]
    if len(candidates) == 0:
        raise ValueError("Could not find a list-valued key 'female_subjects' anywhere in YAML")
    if len(candidates) > 1:
        raise ValueError(
            f"Found multiple list-valued 'female_subjects' keys in YAML ({len(candidates)} matches). "
            "Please keep only one."
        )

    female_raw = candidates[0]
    return {_norm_subject_id(v) for v in female_raw}


def _iter_subject_dirs(dataset_root: Path) -> Iterable[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and _SUBJECT_RE.match(p.name):
            yield p


def main() -> None:
    ap = argparse.ArgumentParser("Write THUman2 gender.txt files for all subjects")
    ap.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="THUman2 dataset root directory containing 4-digit subject folders",
    )
    ap.add_argument(
        "--female_yaml",
        type=str,
        required=True,
        help="YAML file containing female_subjects list",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, do not overwrite existing <subject_dir>/gender.txt",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, do not write any files; only print summary",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    female_yaml = Path(args.female_yaml)

    female_ids = _load_female_subjects(female_yaml)

    total = 0
    wrote = 0
    skipped = 0
    n_female = 0
    n_male = 0

    for subj_dir in _iter_subject_dirs(dataset_root):
        total += 1
        sid = subj_dir.name
        gender = "female" if sid in female_ids else "male"
        if gender == "female":
            n_female += 1
        else:
            n_male += 1

        out_path = subj_dir / "gender.txt"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        if not args.dry_run:
            out_path.write_text(gender + "\n")
        wrote += 1

    print(
        f"[write_thuman2_gender_txts] subjects={total} female={n_female} male={n_male} "
        f"wrote={wrote} skipped={skipped} dry_run={bool(args.dry_run)}"
    )


if __name__ == "__main__":
    main()
