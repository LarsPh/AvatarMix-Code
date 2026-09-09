#!/usr/bin/env python3


import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Set
import yaml
from loguru import logger


class NoSAMOrganizer:


    def __init__(
        self,
        dataset_root: Path,
        pairs_yaml: Path,
        dry_run: bool = False,
        verbose: bool = False,
        use_copy: bool = True,
        skip_ply: bool = True
    ):

        self.dataset_root = Path(dataset_root)
        self.pairs_yaml = Path(pairs_yaml)
        self.dry_run = dry_run
        self.verbose = verbose
        self.use_copy = use_copy
        self.skip_ply = skip_ply


        self.image_dir = self.dataset_root / "no_sam_annot_image"
        self.ply_dir = self.dataset_root / "no_sam_annot_ply"


        self.stats = {
            "total_subjects": 0,
            "successful_links": 0,
            "skipped_missing": 0,
            "failed_links": 0,
        }


        logger.remove()
        if verbose:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")

    def load_pairs(self) -> Set[str]:

        logger.info(f"Loading pairs from: {self.pairs_yaml}")

        with open(self.pairs_yaml, 'r') as f:
            data = yaml.safe_load(f)

        pairs = data.get('pairs', [])
        logger.info(f"Found {len(pairs)} pairs")


        subjects = set()
        for pair in pairs:
            subjects.add(pair['subject_a'])
            subjects.add(pair['subject_b'])

        logger.info(f"Extracted {len(subjects)} unique subjects")
        self.stats["total_subjects"] = len(subjects)

        return subjects

    def get_source_files(self, subject: str) -> Tuple[Path, Path]:

        base_dir = self.dataset_root / subject / "Semantic" / "process" / "labels_auto_extended"

        png_path = base_dir / "label-f0000_no_sam.png"
        ply_path = base_dir / "vis-labeled-mesh-f0000_no_sam.ply"

        return png_path, ply_path

    def create_output_dirs(self):

        if self.dry_run:
            logger.info("[DRY RUN] Would create directories:")
            logger.info(f"  - {self.image_dir}")
            logger.info(f"  - {self.ply_dir}")
            return

        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.ply_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output directories:")
        logger.info(f"  - {self.image_dir}")
        logger.info(f"  - {self.ply_dir}")

    def create_symlink(self, src: Path, dst: Path) -> bool:

        if self.dry_run:
            action = "copy" if self.use_copy else "link"
            logger.debug(f"[DRY RUN] Would {action}: {src} -> {dst}")
            return True

        try:

            if dst.exists() or dst.is_symlink():
                logger.debug(f"Removing existing file: {dst}")
                dst.unlink()


            if self.use_copy:
                shutil.copy2(src, dst)
                logger.debug(f"Copied: {dst}")
                return True


            import os
            os.symlink(src, dst)
            logger.debug(f"Created symlink: {dst} -> {src}")
            return True

        except OSError as e:

            if sys.platform == "win32" and not self.use_copy:
                logger.warning(f"Symlink failed on Windows, falling back to copy: {e}")
                try:
                    shutil.copy2(src, dst)
                    logger.debug(f"Created copy (fallback): {dst}")
                    return True
                except Exception as copy_error:
                    logger.error(f"Copy fallback also failed: {copy_error}")
                    return False
            else:
                logger.error(f"Failed to create {'copy' if self.use_copy else 'symlink'}: {e}")
                return False

    def process_subject(self, subject: str) -> bool:

        logger.debug(f"Processing subject: {subject}")


        png_src, ply_src = self.get_source_files(subject)


        png_exists = png_src.exists()
        ply_exists = ply_src.exists()

        if not png_exists and not ply_exists:
            logger.debug(f"Subject {subject}: No source files found, skipping")
            self.stats["skipped_missing"] += 1
            return False

        if not png_exists:
            logger.warning(f"Subject {subject}: PNG missing: {png_src}")


        if self.skip_ply and ply_exists:
            logger.debug(f"Subject {subject}: Skipping PLY file (too large)")
            ply_exists = False


        png_dst = self.image_dir / f"{subject}_label-f0000.png"
        ply_dst = self.ply_dir / f"{subject}_vis-labeled-mesh-f0000.ply"


        success = True

        if png_exists:
            if not self.create_symlink(png_src, png_dst):
                success = False

        if ply_exists:
            if not self.create_symlink(ply_src, ply_dst):
                success = False

        if success and (png_exists or ply_exists):
            logger.debug(f"Subject {subject}: Successfully processed")
            self.stats["successful_links"] += 1
            return True
        else:
            logger.error(f"Subject {subject}: Failed to create links")
            self.stats["failed_links"] += 1
            return False

    def run(self):

        logger.info("=" * 60)
        logger.info("No-SAM Annotation Organizer")
        logger.info("=" * 60)
        logger.info(f"Dataset root: {self.dataset_root}")
        logger.info(f"Pairs YAML: {self.pairs_yaml}")
        logger.info(f"Mode: {'COPY' if self.use_copy else 'SYMLINK'}")
        logger.info(f"Include PLY: {'No (images only)' if self.skip_ply else 'Yes'}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("")


        subjects = self.load_pairs()


        self.create_output_dirs()


        logger.info("")
        logger.info(f"Processing {len(subjects)} subjects...")
        logger.info("")

        for subject in sorted(subjects):
            self.process_subject(subject)


        logger.info("")
        logger.info("=" * 60)
        logger.info("Summary")
        logger.info("=" * 60)
        logger.info(f"Total subjects: {self.stats['total_subjects']}")
        logger.info(f"Successfully linked: {self.stats['successful_links']}")
        logger.info(f"Skipped (missing files): {self.stats['skipped_missing']}")
        logger.info(f"Failed: {self.stats['failed_links']}")
        logger.info("")

        if not self.dry_run:
            logger.info("Output directories:")
            logger.info(f"  Images: {self.image_dir}")
            logger.info(f"  PLY files: {self.ply_dir}")
        else:
            logger.info("[DRY RUN] No files were created")

        logger.info("=" * 60)


def main():

    parser = argparse.ArgumentParser(
        description="Organize no_sam annotation files into flat directories"
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root directory (e.g., data/thuman2_avatarrex)"
    )

    parser.add_argument(
        "--pairs-yaml",
        type=Path,
        default=None,
        help="Pairs YAML file (default: {dataset_root}/pairs/thuman2_test_56.yaml)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually creating links"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        default=True,
        help="Copy files instead of symlinks (default: True, better for SSHFS/Windows)"
    )

    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying (overrides --copy)"
    )

    parser.add_argument(
        "--skip-ply",
        action="store_true",
        default=True,
        help="Skip PLY files (default: True, avoid time-consuming copies)"
    )

    parser.add_argument(
        "--include-ply",
        action="store_true",
        help="Include PLY files (overrides --skip-ply)"
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root


    if args.pairs_yaml is None:
        pairs_yaml = dataset_root / "pairs" / "thuman2_test_56.yaml"
    else:
        pairs_yaml = args.pairs_yaml


    if not dataset_root.exists():
        logger.error(f"Dataset root does not exist: {dataset_root}")
        sys.exit(1)

    if not pairs_yaml.exists():
        logger.error(f"Pairs YAML does not exist: {pairs_yaml}")
        sys.exit(1)


    use_copy = not args.symlink


    skip_ply = not args.include_ply


    organizer = NoSAMOrganizer(
        dataset_root=dataset_root,
        pairs_yaml=pairs_yaml,
        dry_run=args.dry_run,
        verbose=args.verbose,
        use_copy=use_copy,
        skip_ply=skip_ply
    )

    organizer.run()


if __name__ == "__main__":
    main()
