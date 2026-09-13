"""Keep analysis results inside the repository.

Experiment results must never be written outside the working tree, and the
sealed runners refuse to continue once their own output makes the tree dirty,
so an output directory has to be both inside the repository and git-ignored.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def repository_output_dir(value):
    """argparse type: resolve a git-ignored directory inside the repository."""
    path = Path(value).resolve()
    if PROJECT_ROOT not in path.parents:
        raise argparse.ArgumentTypeError(
            f"output directory must be inside the repository: {path}"
        )
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", f"{path}/"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    if not ignored:
        raise argparse.ArgumentTypeError(
            "output directory must be git-ignored (e.g. under outputs/ or "
            f"analyses/archvied_analyses/) or the clean-tree check fails: {path}"
        )
    return path
