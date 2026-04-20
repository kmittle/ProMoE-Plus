#!/usr/bin/env python3
"""
Create a compact repository inventory for AGENTS.md drafting or refresh work.
"""

from __future__ import annotations

import argparse
from pathlib import Path


IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".claude",
    ".codex",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "outputs",
    "tmp",
    "temp",
}

DOC_NAMES = {
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "README.md",
    "README",
}

MANIFEST_NAMES = {
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "package-lock.json",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "environment.yml",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "Cargo.toml",
    "go.mod",
    "justfile",
    "Taskfile.yml",
    "Taskfile.yaml",
    "CMakeLists.txt",
}

ENTRYPOINT_EXTS = {
    ".py",
    ".sh",
    ".js",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
}

ENTRYPOINT_PREFIXES = (
    "app",
    "build",
    "eval",
    "infer",
    "main",
    "run",
    "sample",
    "serve",
    "test",
    "train",
)

INTERESTING_EXTS = {
    ".py",
    ".sh",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".rs",
}


def should_skip_dir(path: Path) -> bool:
    return path.name in IGNORE_DIRS or path.name.startswith(".")


def top_level_files(root: Path, allowed_names: set[str]) -> list[str]:
    items = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.is_file() and child.name in allowed_names:
            items.append(child.name)
    return items


def top_level_readmes(root: Path) -> list[str]:
    items = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.is_file() and child.name.upper().startswith("README"):
            items.append(child.name)
    return items


def root_entrypoints(root: Path) -> list[str]:
    items = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_file() or child.suffix not in ENTRYPOINT_EXTS or child.name.startswith("."):
            continue
        if looks_like_entrypoint(child):
            items.append(child.name)
    return items


def looks_like_entrypoint(path: Path) -> bool:
    if path.suffix == ".sh":
        return True

    if path.stem.startswith(ENTRYPOINT_PREFIXES):
        return True

    try:
        content = path.read_text(errors="ignore")
    except OSError:
        return False

    return (
        "__main__" in content
        or "argparse.ArgumentParser" in content
        or "@click.command" in content
        or "click.command(" in content
    )


def collect_directory_examples(root: Path, directory: Path, max_depth: int, limit: int) -> list[str]:
    examples: list[str] = []

    def visit(current: Path, depth: int) -> None:
        if len(examples) >= limit:
            return
        if depth > max_depth:
            return

        children = sorted(current.iterdir(), key=lambda p: p.name.lower())
        for child in children:
            if child.is_dir():
                if should_skip_dir(child):
                    continue
                visit(child, depth + 1)
                if len(examples) >= limit:
                    return
                continue

            if child.suffix in INTERESTING_EXTS or child.name in DOC_NAMES or child.name in MANIFEST_NAMES:
                examples.append(str(child.relative_to(root)))
                if len(examples) >= limit:
                    return

    visit(directory, 1)
    return examples


def top_level_directories(root: Path, max_depth: int, limit: int) -> list[tuple[str, list[str]]]:
    results = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or should_skip_dir(child):
            continue
        examples = collect_directory_examples(root, child, max_depth=max_depth, limit=limit)
        results.append((child.name, examples))
    return results


def emit_section(title: str, items: list[str]) -> None:
    print(f"{title}:")
    if not items:
        print("- none")
        print()
        return
    for item in items:
        print(f"- {item}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a compact repository inventory for AGENTS.md generation.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository root to inspect. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum directory depth to sample beneath each top-level directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Maximum number of example files to show per top-level directory.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Repository root does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Repository root is not a directory: {root}")

    docs = sorted(set(top_level_files(root, DOC_NAMES) + top_level_readmes(root)))
    manifests = top_level_files(root, MANIFEST_NAMES)
    entrypoints = root_entrypoints(root)
    directories = top_level_directories(root, max_depth=args.max_depth, limit=args.limit)

    print(f"Repository: {root}")
    print()
    emit_section("Key docs", docs)
    emit_section("Build manifests", manifests)
    emit_section("Root entrypoints", entrypoints)

    print("Top-level directories:")
    if not directories:
        print("- none")
        return

    for name, examples in directories:
        print(f"- {name}/")
        if not examples:
            print("  - no sampled files")
            continue
        for example in examples:
            print(f"  - {example}")


if __name__ == "__main__":
    main()
