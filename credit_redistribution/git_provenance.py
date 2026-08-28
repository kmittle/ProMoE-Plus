"""Read Git provenance without inheriting repository overrides."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


_HEXADECIMAL = frozenset("0123456789abcdef")
AUTHORITATIVE_REMOTE_URL = "git@github.com:kmittle/ProMoE-Plus.git"
AUTHORITATIVE_REMOTE_REF = "refs/heads/repa"
REMOTE_QUERY_TIMEOUT_SECONDS = 30


def sanitized_git_environment():
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_SSH_COMMAND"] = (
        "ssh -oBatchMode=yes -oConnectTimeout=15"
    )
    environment["GIT_SSH_VARIANT"] = "ssh"
    return environment


def run_git(project_root, *arguments, check=False, text=True):
    return subprocess.run(
        ["git", "--no-replace-objects", *arguments],
        cwd=Path(project_root),
        env=sanitized_git_environment(),
        check=check,
        capture_output=True,
        text=text,
    )


def git_output(project_root, *arguments):
    return run_git(
        project_root,
        *arguments,
        check=True,
        text=True,
    ).stdout.strip()


def _git_common_dir(project_root):
    project_root = Path(project_root).resolve()
    common_dir = Path(
        git_output(project_root, "rev-parse", "--git-common-dir")
    )
    if not common_dir.is_absolute():
        common_dir = project_root / common_dir
    return common_dir.resolve()


def reject_history_overrides(project_root):
    project_root = Path(project_root)
    replace_refs = git_output(
        project_root,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace",
    )
    if replace_refs:
        raise RuntimeError("Git replace refs are forbidden for provenance checks")

    common_dir = _git_common_dir(project_root)
    grafts = common_dir / "info" / "grafts"
    if grafts.is_file() and grafts.stat().st_size:
        raise RuntimeError("Git grafts are forbidden for provenance checks")


def reject_index_overrides(project_root):
    result = run_git(
        project_root,
        "ls-files",
        "-v",
        "-z",
        check=True,
        text=False,
    )
    hidden = []
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        if len(entry) < 3 or entry[1:2] != b" ":
            raise RuntimeError("Git index metadata is malformed")
        marker = entry[:1]
        if marker == b"S" or marker.islower():
            hidden.append(os.fsdecode(entry[2:]))
    if hidden:
        raise RuntimeError(
            "Git assume-unchanged/skip-worktree flags are forbidden for "
            "provenance checks: " + ", ".join(sorted(hidden))
        )


def authoritative_remote_tip(remote_url=None, remote_ref=None):
    remote_url = remote_url or AUTHORITATIVE_REMOTE_URL
    remote_ref = remote_ref or AUTHORITATIVE_REMOTE_REF
    environment = sanitized_git_environment()
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_DIR"] = os.devnull
    with tempfile.TemporaryDirectory(
        prefix="promoe-authoritative-git-"
    ) as isolated_directory:
        query_directory = Path(isolated_directory) / "query"
        query_directory.mkdir()
        try:
            result = subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "ls-remote",
                    "--exit-code",
                    remote_url,
                    remote_ref,
                ],
                cwd=query_directory,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=REMOTE_QUERY_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "Authoritative Git remote query timed out"
            ) from error
    if result.returncode != 0:
        raise RuntimeError(
            "Could not read the authoritative Git remote branch"
        )
    rows = [line.split("\t") for line in result.stdout.splitlines()]
    if (
        len(rows) != 1
        or len(rows[0]) != 2
        or rows[0][1] != remote_ref
        or len(rows[0][0]) != 40
        or not set(rows[0][0]).issubset(_HEXADECIMAL)
    ):
        raise RuntimeError("Authoritative Git remote response is malformed")
    return rows[0][0]


def fresh_worktree_status(project_root):
    """Compare the worktree with ``HEAD`` without trusting Git stat data.

    CubeFS can preserve all metadata fields consulted by Git after a file is
    rewritten.  Rebuilding an index is insufficient in that case: Git may
    populate the new index with the same poisoned metadata and skip hashing
    the file again.  Hash every tracked blob explicitly and compare it with
    the object ID recorded by ``HEAD`` instead.
    """

    project_root = Path(project_root).resolve()
    tree = run_git(
        project_root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        "HEAD",
        check=True,
        text=False,
    )
    expected_blobs = {}
    changed = []
    hash_paths = []
    for raw_entry in tree.stdout.split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, encoded_path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = metadata.split(b" ", 2)
        except ValueError as error:
            raise RuntimeError("Git HEAD tree entry is malformed") from error
        relative = os.fsdecode(encoded_path)
        relative_path = PurePosixPath(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or b"\n" in encoded_path
        ):
            raise RuntimeError(f"Git HEAD path cannot be verified: {relative!r}")
        if object_type != b"blob" or mode not in (b"100644", b"100755", b"120000"):
            raise RuntimeError(
                f"Git HEAD entry type cannot be verified: {relative}"
            )

        worktree_path = project_root.joinpath(*relative_path.parts)
        try:
            observed_stat = worktree_path.lstat()
        except FileNotFoundError:
            changed.append((relative, f" D {relative}\n"))
            continue
        except OSError as error:
            raise RuntimeError(
                f"Could not inspect tracked worktree path: {relative}"
            ) from error

        if mode == b"120000":
            expected_type = stat.S_ISLNK(observed_stat.st_mode)
        else:
            expected_type = stat.S_ISREG(observed_stat.st_mode)
        if not expected_type:
            changed.append((relative, f" T {relative}\n"))
            continue
        if mode in (b"100644", b"100755"):
            observed_executable = bool(observed_stat.st_mode & 0o111)
            expected_executable = mode == b"100755"
            if observed_executable != expected_executable:
                changed.append((relative, f" M {relative}\n"))

        expected_blobs[relative] = object_id.decode("ascii")
        if mode == b"120000":
            try:
                link_target = os.fsencode(os.readlink(worktree_path))
            except OSError as error:
                raise RuntimeError(
                    f"Could not read tracked worktree symlink: {relative}"
                ) from error
            link_hash = subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "hash-object",
                    "--no-filters",
                    "--stdin",
                ],
                cwd=project_root,
                env=sanitized_git_environment(),
                input=link_target,
                check=False,
                capture_output=True,
                text=False,
            )
            if link_hash.returncode != 0:
                raise RuntimeError(
                    f"Could not hash tracked worktree symlink: {relative}"
                )
            observed = link_hash.stdout.strip().decode("ascii")
            if observed != expected_blobs[relative]:
                changed.append((relative, f" M {relative}\n"))
        else:
            hash_paths.append(relative)

    if hash_paths:
        path_input = b"".join(
            os.fsencode(relative) + b"\n" for relative in hash_paths
        )
        hashes = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "hash-object",
                "--no-filters",
                "--stdin-paths",
            ],
            cwd=project_root,
            env=sanitized_git_environment(),
            input=path_input,
            check=False,
            capture_output=True,
            text=False,
        )
        if hashes.returncode != 0:
            raise RuntimeError("Could not hash every tracked worktree blob")
        observed_blobs = hashes.stdout.splitlines()
        if len(observed_blobs) != len(hash_paths):
            raise RuntimeError("Git returned an incomplete worktree hash set")
        for relative, observed_blob in zip(hash_paths, observed_blobs):
            try:
                observed = observed_blob.decode("ascii")
            except UnicodeDecodeError as error:
                raise RuntimeError("Git returned a malformed worktree hash") from error
            if observed != expected_blobs[relative]:
                changed.append((relative, f" M {relative}\n"))

    untracked = run_git(
        project_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        check=True,
        text=False,
    )
    for encoded_path in untracked.stdout.split(b"\0"):
        if encoded_path:
            relative = os.fsdecode(encoded_path)
            changed.append((relative, f"?? {relative}\n"))

    index_difference = run_git(
        project_root,
        "diff-index",
        "--cached",
        "--quiet",
        "HEAD",
        "--",
        text=True,
    )
    if index_difference.returncode not in (0, 1):
        raise RuntimeError("Could not verify the real Git index against HEAD")
    status = "".join(line for _, line in sorted(set(changed)))
    if index_difference.returncode == 1:
        return status + "[real index differs from HEAD]\n"
    return status


def verify_worktree_source_manifest(project_root, commit, source_hashes):
    project_root = Path(project_root).resolve()
    reject_history_overrides(project_root)
    reject_index_overrides(project_root)
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("Git source manifest is absent or invalid")

    for relative, expected in sorted(source_hashes.items()):
        if not isinstance(relative, str):
            raise RuntimeError(f"Git source path is invalid: {relative!r}")
        source_path = PurePosixPath(relative)
        if (
            not relative
            or source_path.is_absolute()
            or ".." in source_path.parts
            or ":" in relative
        ):
            raise RuntimeError(f"Git source path is invalid: {relative!r}")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or not set(expected).issubset(_HEXADECIMAL)
        ):
            raise RuntimeError(f"Git source hash is invalid: {relative}")

        blob = run_git(
            project_root,
            "cat-file",
            "blob",
            f"{commit}:{relative}",
            text=False,
        )
        if blob.returncode != 0:
            raise RuntimeError(
                f"Git source is absent from the recorded commit: {relative}"
            )
        if hashlib.sha256(blob.stdout).hexdigest() != expected:
            raise RuntimeError(
                f"Git source manifest differs from the recorded commit: {relative}"
            )

        worktree_path = project_root.joinpath(*source_path.parts)
        if worktree_path.is_symlink() or not worktree_path.is_file():
            raise RuntimeError(
                f"Working-tree source is absent or indirect: {relative}"
            )
        if hashlib.sha256(worktree_path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(
                f"Working-tree source differs from the recorded commit: {relative}"
            )


def repository_state(
    project_root,
    verify_remote=True,
    authoritative_remote_url=None,
    authoritative_remote_ref=None,
):
    project_root = Path(project_root)
    reject_history_overrides(project_root)
    reject_index_overrides(project_root)
    status = fresh_worktree_status(project_root)
    remote_url = None
    remote_ref = None
    remote_tip = None
    if verify_remote:
        remote_url = authoritative_remote_url or AUTHORITATIVE_REMOTE_URL
        remote_ref = authoritative_remote_ref or AUTHORITATIVE_REMOTE_REF
        configured = run_git(
            project_root,
            "config",
            "--local",
            "--get-all",
            "remote.origin.url",
            text=True,
        )
        configured_urls = configured.stdout.splitlines()
        if configured.returncode not in (0, 1) or configured_urls != [remote_url]:
            raise RuntimeError(
                "Configured origin URL differs from the authoritative Git remote"
            )
        remote_tip = authoritative_remote_tip(remote_url, remote_ref)
    return {
        "branch": git_output(project_root, "branch", "--show-current"),
        "commit": git_output(project_root, "rev-parse", "HEAD"),
        "origin_repa": git_output(
            project_root,
            "rev-parse",
            "--verify",
            "refs/remotes/origin/repa^{commit}",
        ),
        "authoritative_remote_url": remote_url,
        "authoritative_remote_ref": remote_ref,
        "authoritative_remote_tip": remote_tip,
        "status": status,
    }
