import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lightning_diffusers.models.refiner_adapter import export_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AvatarMix refiner parameters")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("seamfix", "fullbodyfix"), required=True)
    parser.add_argument("--base-revision", required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    counts = export_adapter(args.checkpoint, args.output, args.variant, args.base_revision)
    digest = hashlib.sha256()
    with args.output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    print(f"{args.output}: {counts}; sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()
