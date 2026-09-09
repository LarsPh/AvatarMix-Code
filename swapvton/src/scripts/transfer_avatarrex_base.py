import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


def _should_include_file(rel_posix: str) -> bool:

    parts = rel_posix.split("/")
    if "mesh" in parts:
        return True
    name = parts[-1]
    if name == "smpl_params.npz":
        return True
    if name == "calibration_full.json":
        return True
    if name.endswith("_metadata.json"):
        return True
    if name == "gender.txt":
        return True
    if name.lower().endswith(".jpg") or name.lower().endswith(".png"):
        return True
    return False


def _iter_included_relpaths(src_root: Path, allowed_subject_dirs: Optional[Set[str]] = None) -> List[str]:
    relpaths: List[str] = []
    for root, _, files in os.walk(src_root):
        for fn in files:
            abs_path = Path(root) / fn
            rel = abs_path.relative_to(src_root).as_posix()
            if allowed_subject_dirs is not None:
                top = rel.split("/", 1)[0]
                if top not in allowed_subject_dirs:
                    continue
            if _should_include_file(rel):
                relpaths.append(rel)
    relpaths.sort()
    return relpaths


def _write_rsync_filter_file(
    filter_path: Path, allowed_subject_dirs: Optional[Set[str]] = None
) -> None:


    lines: List[str] = []


    if allowed_subject_dirs is not None:
        for subj in sorted(allowed_subject_dirs):

            lines.append(f"+ /{subj}/")
            lines.append(f"+ /{subj}/**/")

        lines.append("- /*/")
    else:

        lines.append("+ */")


    lines += [
        "+ **/mesh/***",
        "+ */smpl_params.npz",
        "+ */calibration_full.json",
        "+ **/*_metadata.json",
        "+ **/gender.txt",
        "+ **/*.jpg",
        "+ **/*.png",
        "- *",
        "",
    ]
    filter_path.write_text("\n".join(lines), encoding="utf-8")


def _run(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _extract_leading_int(name: str) -> Optional[int]:

    n = 0
    for ch in name:
        if ch.isdigit():
            n += 1
        else:
            break
    if n == 0:
        return None
    try:
        return int(name[:n])
    except Exception:
        return None


def _parse_subject_ranges(ranges: List[str]) -> List[Tuple[int, int]]:

    out: List[Tuple[int, int]] = []
    for r in ranges:
        r = r.strip()
        if not r:
            continue
        if "-" not in r:
            raise ValueError(f"Invalid --subject_range '{r}', expected START-END")
        a, b = r.split("-", 1)
        a, b = a.strip(), b.strip()
        if not (a.isdigit() and b.isdigit()):
            raise ValueError(f"Invalid --subject_range '{r}', START/END must be integers")
        lo, hi = int(a), int(b)
        if lo > hi:
            lo, hi = hi, lo
        out.append((lo, hi))
    return out


def _load_subject_list_file(path: Path) -> List[str]:

    if not path.exists():
        raise FileNotFoundError(f"--subject_list_file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    tokens: List[str] = []
    for part in text.replace("\n", " ").split(","):
        part = part.strip()
        if not part:
            continue
        tokens.extend(part.split())
    return tokens


def _load_pairs_yaml_subjects(path: Path) -> List[str]:

    if not path.exists():
        raise FileNotFoundError(f"--pairs_yaml not found: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("subject_a:") or s.startswith("subject_b:"):
            _, val = s.split(":", 1)
            val = val.strip()

            if len(val) >= 2 and ((val[0] == "'" and val[-1] == "'") or (val[0] == '"' and val[-1] == '"')):
                val = val[1:-1]
            if val:
                out.append(val)
    return out


def _resolve_allowed_subject_dirs(
    src_root: Path,
    subject_ranges: List[str],
    explicit_subjects: List[str],
    subject_list_file: Optional[str],
    pairs_yaml: Optional[str],
) -> Optional[Set[str]]:

    wanted: Set[str] = set()

    if subject_list_file:
        explicit_subjects = list(explicit_subjects) + _load_subject_list_file(Path(subject_list_file))
    if pairs_yaml:
        explicit_subjects = list(explicit_subjects) + _load_pairs_yaml_subjects(Path(pairs_yaml))

    explicit_subjects = [s.strip() for s in explicit_subjects if s and s.strip()]
    parsed_ranges = _parse_subject_ranges(subject_ranges) if subject_ranges else []

    if not explicit_subjects and not parsed_ranges:
        return None


    for s in explicit_subjects:
        wanted.add(s)


    if parsed_ranges:
        for p in src_root.iterdir():
            if not p.is_dir():
                continue
            v = _extract_leading_int(p.name)
            if v is None:
                continue
            for lo, hi in parsed_ranges:
                if lo <= v <= hi:
                    wanted.add(p.name)
                    break

    if not wanted:
        raise RuntimeError(
            "Subject filtering was requested, but no subject directories matched. "
            "Check --subject_range / --subjects / --subject_list_file."
        )
    return wanted


def _mode_rsync(
    src_root: Path,
    dst_ssh: str,
    ssh_port: int | None,
    ssh_key: str | None,
    allowed_subject_dirs: Optional[Set[str]],
    overwrite_existing: bool,
    delete_extra: bool,
    dry_run: bool,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        filt = Path(td) / "rsync.filter"
        _write_rsync_filter_file(filt, allowed_subject_dirs=allowed_subject_dirs)

        cmd = [
            "rsync",
            "-av",
            "--prune-empty-dirs",
            "--filter",
            f"merge {str(filt)}",
        ]

        if not overwrite_existing:
            cmd.append("--ignore-existing")

        if delete_extra:
            cmd.append("--delete")
        if dry_run:
            cmd.append("--dry-run")

        ssh_cmd = ["ssh"]
        if ssh_port is not None:
            ssh_cmd += ["-p", str(ssh_port)]
        if ssh_key is not None:
            ssh_cmd += ["-i", ssh_key]
        cmd += ["-e", " ".join(ssh_cmd)]


        cmd += [str(src_root) + "/", dst_ssh]
        _run(cmd)


def _mode_tar_scp(
    src_root: Path,
    dst_ssh: str,
    ssh_port: int | None,
    ssh_key: str | None,
    allowed_subject_dirs: Optional[Set[str]],
    compress: str,
    dry_run: bool,
) -> None:

    import tarfile

    relpaths = _iter_included_relpaths(src_root, allowed_subject_dirs=allowed_subject_dirs)
    if not relpaths:
        raise RuntimeError(f"No files matched filter under {src_root}")

    suffix = ".tar" if compress == "none" else ".tar.gz"
    tar_name = f"{src_root.name}_avatarrex_base{suffix}"
    tar_path = Path.cwd() / tar_name

    mode = "w" if compress == "none" else "w:gz"
    if dry_run:
        print(f"[dry-run] Would create {tar_path} with {len(relpaths)} files from {src_root}")
    else:
        print(f"Creating {tar_path} with {len(relpaths)} files...")
        with tarfile.open(tar_path, mode) as tf:
            for rel in relpaths:
                tf.add(src_root / rel, arcname=rel, recursive=False)


    scp_cmd = ["scp"]
    if ssh_port is not None:
        scp_cmd += ["-P", str(ssh_port)]
    if ssh_key is not None:
        scp_cmd += ["-i", ssh_key]
    scp_cmd += [str(tar_path), dst_ssh]

    if dry_run:
        print("[dry-run] Would run:", " ".join(scp_cmd))
    else:
        _run(scp_cmd)

    print("\nOn the remote, extract with:")
    if compress == "none":
        print(f"  tar -xf {tar_path.name} -C <DEST_DIR>")
    else:
        print(f"  tar -xzf {tar_path.name} -C <DEST_DIR>")
    print("Then you can delete the tarball if desired.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer the *base* AvatarREX outputs (rgb/mask/smpl/calib/metadata) and mesh/ contents.\n"
            "Supports rsync (incremental) or tar+scp (single archive)."
        )
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Local AvatarREX dataset root directory (contains per-subject subdirectories).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Remote destination in scp/rsync form, e.g. user@host:/abs/path/to/dst_dir",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rsync",
        choices=["rsync", "tar"],
        help="Transfer mode: rsync (incremental) or tar (create tarball then scp). Default: rsync",
    )
    parser.add_argument(
        "--subject_range",
        nargs="*",
        default=[],
        help=(
            "Only transfer subjects whose leading numeric ID falls in START-END (inclusive). "
            "Example: --subject_range 0000-0523 (THuman2) or 100001-100500 (MVHumanNet). "
            "Can specify multiple ranges."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=[],
        help=(
            "Explicit list of top-level subject directory names to transfer "
            "(e.g., 0036 0108 100001_1185)."
        ),
    )
    parser.add_argument(
        "--subject_list_file",
        type=str,
        default=None,
        help="Path to a text file containing subject dir names (comma/space/newline separated).",
    )
    parser.add_argument(
        "--pairs_yaml",
        type=str,
        default=None,
        help=(
            "Path to a pairs YAML file (e.g., thuman2_test_56.yaml). "
            "All subject_a/subject_b IDs in pairs will be transferred."
        ),
    )
    parser.add_argument(
        "--ssh_port",
        type=int,
        default=None,
        help="SSH port (optional).",
    )
    parser.add_argument(
        "--ssh_key",
        type=str,
        default=None,
        help="SSH private key path (optional).",
    )
    parser.add_argument(
        "--compress",
        type=str,
        default="gz",
        choices=["gz", "none"],
        help="Compression for tar mode. Default: gz",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run (for rsync: --dry-run; for tar: only prints what would happen).",
    )
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Overwrite existing files on the remote (rsync mode only). Default: skip existing.",
    )
    parser.add_argument(
        "--delete_extra",
        action="store_true",
        help=(
            "Delete remote files that don't exist locally (within the filtered set). "
            "rsync mode only. Default: off (safer)."
        ),
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    if not src_root.exists():
        raise FileNotFoundError(f"--src not found: {src_root}")
    if not src_root.is_dir():
        raise ValueError(f"--src must be a directory: {src_root}")


    has_subject_dirs = any(p.is_dir() for p in src_root.iterdir())
    if not has_subject_dirs:
        print(f"Warning: {src_root} has no subdirectories; are you pointing at the dataset root?")

    allowed_subject_dirs = _resolve_allowed_subject_dirs(
        src_root=src_root,
        subject_ranges=args.subject_range,
        explicit_subjects=args.subjects,
        subject_list_file=args.subject_list_file,
        pairs_yaml=args.pairs_yaml,
    )
    if allowed_subject_dirs is not None:
        print(f"Subject filter enabled: transferring {len(allowed_subject_dirs)} subjects.")

    if args.mode == "rsync":
        _mode_rsync(
            src_root=src_root,
            dst_ssh=args.dst,
            ssh_port=args.ssh_port,
            ssh_key=args.ssh_key,
            allowed_subject_dirs=allowed_subject_dirs,
            overwrite_existing=args.overwrite_existing,
            delete_extra=args.delete_extra,
            dry_run=args.dry_run,
        )
    else:
        _mode_tar_scp(
            src_root=src_root,
            dst_ssh=args.dst,
            ssh_port=args.ssh_port,
            ssh_key=args.ssh_key,
            allowed_subject_dirs=allowed_subject_dirs,
            compress=args.compress,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
