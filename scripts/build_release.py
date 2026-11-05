"""Build and verify a ComfyUI install ZIP from a committed Git tree."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "ComfyUI-PowerShard/"
VERSION_PATTERN = r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}"
SMOKE_IMPORT = """
import importlib.util
import sys
from pathlib import Path

root = Path.cwd()
spec = importlib.util.spec_from_file_location(
    "powershard_release_plugin", root / "__init__.py", submodule_search_locations=[str(root)]
)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
if not plugin.NODE_CLASS_MAPPINGS or not (root / plugin.WEB_DIRECTORY).is_dir():
    raise SystemExit("Install ZIP has no node mappings or web directory")
if "torch" in sys.modules:
    raise SystemExit("Importing the custom node loaded torch")
"""


def git_text(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8")


def read_version(commit):
    version = tomllib.loads(git_text("show", f"{commit}:pyproject.toml"))["project"]["version"]
    tree = ast.parse(git_text("show", f"{commit}:powershard/__init__.py"))
    runtime_versions = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    ]
    if not re.fullmatch(VERSION_PATTERN, version) or runtime_versions != [version]:
        raise ValueError("Expected a stable X.Y.Z version shared by pyproject.toml and powershard/__init__.py")
    return version


def release_notes(commit, version):
    changelog = git_text("show", f"{commit}:CHANGELOG.md")
    match = re.search(rf"^## {re.escape(version)}[ \t]*\r?\n(.*?)(?=^## |\Z)", changelog, re.M | re.S)
    if not match or not match[1].strip():
        raise ValueError(f"Missing release notes under '## {version}' in CHANGELOG.md")
    return match[1].strip() + "\n"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build(output_dir, tag):
    if tag and not re.fullmatch("v" + VERSION_PATTERN, tag):
        raise ValueError(f"Expected a stable vX.Y.Z tag, got {tag!r}")
    ref = f"refs/tags/{tag}" if tag else "HEAD"
    commit = git_text("rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    version = read_version(commit)
    if tag and tag != f"v{version}":
        raise ValueError(f"Tag {tag} does not match version {version} at {commit}")
    notes = release_notes(commit, version)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=output_dir) as directory:
        staging = Path(directory)
        archive = staging / f"ComfyUI-PowerShard-{version}.zip"
        subprocess.run(
            ["git", "archive", "--format=zip", f"--prefix={PREFIX}", f"--output={archive}", commit],
            cwd=ROOT, check=True,
        )
        with ZipFile(archive) as package:
            if bad_entry := package.testzip():
                raise ValueError(f"Invalid ZIP member: {bad_entry}")
            files = {entry.filename.removeprefix(PREFIX): sha256(package.read(entry))
                     for entry in package.infolist() if not entry.is_dir()}
            required = {"__init__.py", "LICENSE", "NOTICE", "README.md", "README_RU.md", "web/powershard.js"}
            if missing := required - files.keys():
                raise ValueError(f"Install ZIP is missing: {', '.join(sorted(missing))}")
            package.extractall(staging / "smoke")
        subprocess.run([sys.executable, "-I", "-S", "-c", SMOKE_IMPORT], cwd=staging / "smoke" / PREFIX, check=True)

        manifest = staging / f"PowerShard-{version}-release-manifest.json"
        manifest.write_text(json.dumps({
            "version": version,
            "tag": tag,
            "commit": commit,
            "package": {"name": archive.name, "bytes": archive.stat().st_size, "sha256": sha256(archive.read_bytes())},
            "files_sha256": files,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        sums = staging / f"PowerShard-{version}-SHA256SUMS.txt"
        sums.write_text("".join(f"{sha256(p.read_bytes())}  {p.name}\n" for p in (archive, manifest)),
                        encoding="ascii", newline="\n")
        notes_file = staging / "RELEASE_NOTES.md"
        notes_file.write_text(notes, encoding="utf-8", newline="\n")
        for artifact in (archive, manifest, sums, notes_file):
            artifact.replace(output_dir / artifact.name)
    print(f"Built and verified {archive.name} from {commit} ({len(files)} files)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Build an existing local vX.Y.Z tag (default: committed HEAD)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    try:
        build(args.output_dir.resolve(), args.tag)
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Release build failed: {exc}\n")


if __name__ == "__main__":
    main()
