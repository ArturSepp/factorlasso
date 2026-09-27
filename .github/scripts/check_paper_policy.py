"""Check indexed paper publication boundaries, or preview the working tree."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import zipfile


PUBLIC_PAPERS = {"sign_pooling_2026"}

PROBES = (
    "papers/unapproved_paper/replication/reproduce.py",
    "papers/sign_pooling_2026/drafts/v1/paper.tex",
    "papers/sign_pooling_2026/private/referee.txt",
    "papers/sign_pooling_2026/agents/ROADMAP.md",
    "papers/sign_pooling_2026/paper/unapproved_probe.tex",
    "papers/sign_pooling_2026/paper/unapproved_probe.pdf",
    "papers/sign_pooling_2026/presentations/event/slides.pdf",
    "papers/sign_pooling_2026/replication/data/local/input.csv",
    "papers/sign_pooling_2026/replication/outputs/run.csv",
    "agents/ROADMAP.md",
)


def git(root: Path, *args: str, data: bytes | None = None) -> bytes:
    """Run read-only Git queries without relying on personal exclusions."""
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-c", "core.excludesFile=",
         "-C", str(root), *args], input=data, capture_output=True, check=False,
    )
    if result.returncode not in (0, 1) or (result.returncode and args[0] != "check-ignore"):
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def selected_files(root: Path, worktree: bool) -> list[str]:
    """Select indexed paths, or tracked and visible new working-tree files."""
    names = set(git(root, "ls-files", "-z").decode("utf-8").split("\0")) - {""}
    if worktree:
        names.update(git(root, "ls-files", "--others", "--exclude-standard", "-z")
                     .decode("utf-8").split("\0"))
        names = {name for name in names if name and (root / name).is_file()}
    return sorted(names)


def protected(name: str) -> bool:
    """Identify private sections even if a nested ignore rule reopens them."""
    parts = PurePosixPath(name).parts
    lower = [part.casefold() for part in parts]
    if lower[0] == "agents":
        return True
    if lower[0] != "papers":
        return False
    if len(parts) >= 3 and parts[1] not in PUBLIC_PAPERS:
        return True
    if len(parts) == 2 and parts[1] not in {"AGENTS.md", "README.md"}:
        return True
    directories = lower[1:-1]
    return bool(set(directories) & {"private", "drafts", "agents"}) or any(
        directories[i:i + 2] == ["data", "local"] for i in range(len(directories))
    )


def check_repository(root: Path, worktree: bool = False) -> list[str]:
    """Evaluate paths against an isolated copy of the selected ignore policy."""
    names = selected_files(root, worktree)
    errors = [f"protected material is tracked: {name}" for name in names if protected(name)]
    paper_names = [name for name in names if name.casefold().startswith("papers/")]
    approved_files = set()
    if "papers/AGENTS.md" not in names:
        errors.append("papers/AGENTS.md is missing from the selected files")
    with tempfile.TemporaryDirectory(prefix="fl-paper-policy-") as directory:
        fixture = Path(directory)
        git(fixture, "init", "-q")
        for name in names:
            if PurePosixPath(name).name != ".gitignore":
                continue
            contents = (root / name).read_bytes() if worktree else git(root, "show", f":{name}")
            destination = fixture / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            for line in contents.decode("utf-8-sig").splitlines():
                if line.startswith("!") and not line.endswith("/"):
                    if not any(character in line for character in "*?["):
                        approved_files.add((PurePosixPath(name).parent / line[1:].lstrip("/"))
                                           .as_posix())
                if line.startswith("!") and any(
                    section in line for section in ("paper/", "presentations/", "replication/data/")
                ):
                    if any(character in line for character in "*?["):
                        errors.append(f"use exact publication exceptions in {name}: {line}")
        candidates = paper_names + list(PROBES)
        data = ("\0".join(candidates) + "\0").encode("utf-8")
        ignored = set(git(fixture, "check-ignore", "--no-index", "-z", "--stdin", data=data)
                      .decode("utf-8").split("\0")) - {""}
        errors.extend(f"indexed paper file violates ignore policy: {name}"
                      for name in paper_names if name in ignored)
        errors.extend(f"missing default ignore protection: {name}"
                      for name in PROBES if name not in ignored)
    for name in paper_names:
        parts = PurePosixPath(name).parts
        if len(parts) >= 4 and (parts[2].casefold() in {"paper", "presentations"}
                                 or parts[2:4] == ("replication", "data")):
            if name not in approved_files:
                errors.append(f"publication file needs an exact exception: {name}")
        if len(parts) == 3 and name.endswith(".py"):
            if parts[-1] != "__init__.py":
                errors.append(f"paper code belongs in replication/: {name}")
    return errors


def check_artifacts(directory: Path) -> list[str]:
    """Reject paper or agent workspaces in both wheel and source archives."""
    wheels = sorted(directory.glob("*.whl"))
    sources = sorted(directory.glob("*.tar.gz"))
    errors = []
    if not wheels or not sources:
        errors.append("artifact check requires both a wheel and a .tar.gz source distribution")
    for path in wheels + sources:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        else:
            with tarfile.open(path) as archive:
                names = archive.getnames()
        for name in names:
            if {part.casefold() for part in PurePosixPath(name).parts} & {"papers", "agents"}:
                errors.append(f"research workspace shipped in {path.name}: {name}")
    return errors


def main() -> int:
    """Run the index, worktree-preview or archive check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", action="store_true", help="preview unstaged changes")
    parser.add_argument("--artifacts", type=Path, help="check wheels and source archives")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    errors = (check_artifacts(args.artifacts) if args.artifacts
              else check_repository(root, args.worktree))
    if errors:
        print("Paper policy failed:\n- " + "\n- ".join(errors))
        return 1
    mode = "archives" if args.artifacts else "working tree" if args.worktree else "Git index"
    print(f"PASS: paper publication policy ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
