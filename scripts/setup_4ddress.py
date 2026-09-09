import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch pinned parsing dependencies and apply AvatarMix patches.")
    parser.add_argument("--output-dir", type=Path, help="Parsing component root; defaults to AvatarMix/4d-dress.")
    args = parser.parse_args()
    release = Path(__file__).resolve().parents[1]
    source_dir = release / "4d-dress"
    output = (args.output_dir or source_dir).resolve()
    destination = output / "4dhumanparsing"
    if destination.exists():
        parser.error(f"Refusing to overwrite {destination}. Keep local changes and choose an empty --output-dir.")
    manifest = json.loads((source_dir / "sources.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".avatarmix-parser-", dir=output) as work:
        work = Path(work)
        repos = {}
        for component in manifest["components"]:
            repo = work / component["name"]
            run("git", "init", "--quiet", str(repo))
            run("git", "remote", "add", "origin", component["url"], cwd=repo)
            run("git", "fetch", "--quiet", "--depth", "1", "origin", component["revision"], cwd=repo)
            actual = subprocess.check_output(["git", "rev-parse", "FETCH_HEAD"], cwd=repo, text=True).strip()
            if actual != component["revision"]:
                raise RuntimeError(f"Unexpected revision for {component['name']}: {actual}")
            run("git", "checkout", "--quiet", "--detach", actual, cwd=repo)
            if component["patch"]:
                patch = str(source_dir / "patches" / component["patch"])
                run("git", "apply", "--check", patch, cwd=repo)
                run("git", "apply", patch, cwd=repo)
            for relative, expected in component["expected_sha256"].items():
                if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Patched-file checksum mismatch: {component['name']}/{relative}")
            repos[component["name"]] = repo
            print(f"Verified {component['name']} at {actual}", flush=True)
        assembled = repos["4d-dress"] / "4dhumanparsing"
        for name in ("Graphonomy", "RAFT", "pygco"):
            shutil.copytree(repos[name], assembled / "lib" / name, ignore=shutil.ignore_patterns(".git"))
        (assembled / ".avatarmix-sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
        assembled.rename(destination)
    print(f"Parsing sources ready at {destination}. Install the environment and external model weights next.")


if __name__ == "__main__":
    main()
