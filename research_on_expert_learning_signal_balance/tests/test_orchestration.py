from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from credit_redistribution.controller import BRANCHES
from credit_redistribution.git_provenance import (
    authoritative_remote_tip,
    repository_state,
    run_git,
    sanitized_git_environment,
    verify_worktree_source_manifest,
)
from credit_redistribution.heldout import canonical_json_sha256
from credit_redistribution.orchestration import (
    _git_blob_sha256,
    _git_commit_is_ancestor,
    _revalidate_before_aggregation,
    _verify_continuation_git_provenance,
    _verify_cross_checkpoint_git_provenance,
    verify_prerequisites,
)
from credit_redistribution.serialization import sha256_file


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_protocol(path, payload):
    _write_json(path, payload)
    digest = canonical_json_sha256(payload)
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def _write_summary(root, name, protocol_sha256):
    payload = {
        "name": name,
        "protocol_sha256": protocol_sha256,
        "passed": True,
    }
    path = root / f"{name}-summary.json"
    _write_json(path, payload)
    _write_json(Path(str(path) + ".seal.json"), {
        "protocol_sha256": protocol_sha256,
        "result_sha256": canonical_json_sha256(payload),
    })


def _run_git(root, *args):
    return run_git(
        root,
        "-c",
        "commit.gpgSign=false",
        "-c",
        "core.hooksPath=/dev/null",
        *args,
        check=True,
        text=True,
    ).stdout.strip()


class OrchestrationTest(unittest.TestCase):
    def test_authoritative_remote_query_ignores_ancestor_url_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worktree = root / "worktree"
            attacker = root / f"attacker{os.pathsep}repo"
            authoritative = root / "authoritative.git"
            forged = root / "forged.git"
            for path in (worktree, attacker, authoritative, forged):
                path.mkdir()
            _run_git(worktree, "init", "--quiet")
            _run_git(attacker, "init", "--quiet")
            _run_git(authoritative, "init", "--bare", "--quiet")
            _run_git(forged, "init", "--bare", "--quiet")
            _run_git(worktree, "config", "user.email", "test@example.com")
            _run_git(worktree, "config", "user.name", "Test User")

            source = worktree / "source.py"
            source.write_text("authoritative\n", encoding="utf-8")
            _run_git(worktree, "add", "--", "source.py")
            _run_git(worktree, "commit", "--quiet", "-m", "authoritative")
            authoritative_tip = _run_git(worktree, "rev-parse", "HEAD")
            _run_git(
                worktree,
                "push",
                "--quiet",
                str(authoritative),
                f"{authoritative_tip}:refs/heads/repa",
            )

            source.write_text("forged\n", encoding="utf-8")
            _run_git(worktree, "add", "--", "source.py")
            _run_git(worktree, "commit", "--quiet", "-m", "forged")
            forged_tip = _run_git(worktree, "rev-parse", "HEAD")
            _run_git(
                worktree,
                "push",
                "--quiet",
                str(forged),
                f"{forged_tip}:refs/heads/repa",
            )

            authoritative_url = authoritative.as_uri()
            forged_url = forged.as_uri()
            self.assertIn(os.pathsep, str(attacker))
            _run_git(
                attacker,
                "config",
                f"url.{forged_url}.insteadOf",
                authoritative_url,
            )
            child = attacker / "child"
            child.mkdir()
            environment = sanitized_git_environment()
            environment["GIT_CONFIG_NOSYSTEM"] = "1"
            environment["GIT_CONFIG_GLOBAL"] = os.devnull
            rewritten = subprocess.run(
                [
                    "git",
                    "ls-remote",
                    "--exit-code",
                    authoritative_url,
                    "refs/heads/repa",
                ],
                cwd=child,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()[0]
            self.assertEqual(rewritten, forged_tip)

            with mock.patch(
                "credit_redistribution.git_provenance.tempfile.tempdir",
                str(attacker),
            ):
                observed = authoritative_remote_tip(
                    authoritative_url,
                    "refs/heads/repa",
                )
            self.assertEqual(observed, authoritative_tip)

    def test_repository_state_uses_exact_origin_tracking_ref(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _run_git(root, "init", "--quiet")
            _run_git(root, "config", "user.email", "test@example.com")
            _run_git(root, "config", "user.name", "Test User")

            source = root / "source.py"
            source.write_text("tracked\n", encoding="utf-8")
            _run_git(root, "add", "--", "source.py")
            _run_git(root, "commit", "--quiet", "-m", "tracked")
            tracking_tip = _run_git(root, "rev-parse", "HEAD")
            _run_git(
                root,
                "update-ref",
                "refs/remotes/origin/repa",
                tracking_tip,
            )

            source.write_text("local branch\n", encoding="utf-8")
            _run_git(root, "add", "--", "source.py")
            _run_git(root, "commit", "--quiet", "-m", "local branch")
            local_tip = _run_git(root, "rev-parse", "HEAD")
            _run_git(root, "branch", "origin/repa", local_tip)

            self.assertNotEqual(local_tip, tracking_tip)
            state = repository_state(root, verify_remote=False)
            self.assertEqual(state["origin_repa"], tracking_tip)

    def test_repository_state_rejects_poisoned_stat_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _run_git(root, "init", "--quiet")
            _run_git(root, "config", "user.email", "test@example.com")
            _run_git(root, "config", "user.name", "Test User")
            _run_git(root, "config", "core.trustctime", "false")
            _run_git(root, "config", "core.checkStat", "minimal")

            fsmonitor = root / ".git" / "malicious-fsmonitor"
            fsmonitor.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = 2 ]; then\n"
                "    printf 'unchanged-token\\0'\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fsmonitor.chmod(0o755)
            _run_git(root, "config", "core.fsmonitor", str(fsmonitor))

            source = root / "source.py"
            source.write_text("original\n", encoding="utf-8")
            _run_git(root, "add", "--", "source.py")
            _run_git(root, "commit", "--quiet", "-m", "original")
            commit = _run_git(root, "rev-parse", "HEAD")
            _run_git(
                root,
                "update-ref",
                "refs/remotes/origin/repa",
                commit,
            )
            _run_git(root, "update-index", "--fsmonitor")
            _run_git(
                root,
                "update-index",
                "--fsmonitor-valid",
                "--",
                "source.py",
            )
            fsmonitor_marker = _run_git(
                root,
                "ls-files",
                "-f",
                "--",
                "source.py",
            )
            self.assertTrue(fsmonitor_marker.startswith("h "))

            original_stat = source.stat()
            source.write_text("tampered\n", encoding="utf-8")
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            cached_status = _run_git(
                root,
                "status",
                "--porcelain",
                "--untracked-files=all",
            )
            self.assertEqual(cached_status, "")

            state = repository_state(root, verify_remote=False)
            self.assertIn("source.py", state["status"])

    def test_repository_state_hashes_symlink_target_without_following_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _run_git(root, "init", "--quiet")
            _run_git(root, "config", "user.email", "test@example.com")
            _run_git(root, "config", "user.name", "Test User")

            link = root / "tracked-link"
            link.symlink_to("missing-target-a")
            _run_git(root, "add", "--", "tracked-link")
            _run_git(root, "commit", "--quiet", "-m", "add symlink")
            commit = _run_git(root, "rev-parse", "HEAD")
            _run_git(
                root,
                "update-ref",
                "refs/remotes/origin/repa",
                commit,
            )
            self.assertEqual(
                repository_state(root, verify_remote=False)["status"],
                "",
            )

            link.unlink()
            link.symlink_to("missing-target-b")
            state = repository_state(root, verify_remote=False)
            self.assertIn("tracked-link", state["status"])

    def test_repository_state_hashes_raw_bytes_without_clean_filters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _run_git(root, "init", "--quiet")
            _run_git(root, "config", "user.email", "test@example.com")
            _run_git(root, "config", "user.name", "Test User")
            _run_git(
                root,
                "config",
                "filter.mask.clean",
                "sed s/tampered/original/g",
            )
            _run_git(root, "config", "filter.mask.smudge", "cat")
            _run_git(root, "config", "filter.mask.required", "true")

            (root / ".gitattributes").write_text(
                "source.py filter=mask\n",
                encoding="utf-8",
            )
            source = root / "source.py"
            source.write_text("original\n", encoding="utf-8")
            _run_git(root, "add", "--", ".gitattributes", "source.py")
            _run_git(root, "commit", "--quiet", "-m", "add filtered source")
            commit = _run_git(root, "rev-parse", "HEAD")
            _run_git(
                root,
                "update-ref",
                "refs/remotes/origin/repa",
                commit,
            )

            source.write_text("tampered\n", encoding="utf-8")
            filtered_hash = subprocess.run(
                ["git", "hash-object", "--stdin-paths"],
                cwd=root,
                env=sanitized_git_environment(),
                input=b"source.py\n",
                check=True,
                capture_output=True,
            ).stdout.decode("ascii").strip()
            self.assertEqual(
                filtered_hash,
                _run_git(root, "rev-parse", "HEAD:source.py"),
            )

            state = repository_state(root, verify_remote=False)
            self.assertIn("source.py", state["status"])

    def test_repository_state_rejects_index_only_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _run_git(root, "init", "--quiet")
            _run_git(root, "config", "user.email", "test@example.com")
            _run_git(root, "config", "user.name", "Test User")
            _run_git(
                root,
                "commit",
                "--allow-empty",
                "--quiet",
                "-m",
                "empty",
            )
            commit = _run_git(root, "rev-parse", "HEAD")
            _run_git(
                root,
                "update-ref",
                "refs/remotes/origin/repa",
                commit,
            )

            staged = root / "staged-only.py"
            staged.write_text("staged\n", encoding="utf-8")
            _run_git(root, "add", "--", "staged-only.py")
            staged.unlink()

            state = repository_state(root, verify_remote=False)
            self.assertIn("real index differs from HEAD", state["status"])

    def test_git_provenance_helpers_use_real_history_and_binary_blobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            root = temporary_root / "worktree"
            remote = temporary_root / "authoritative.git"
            root.mkdir()
            remote.mkdir()
            _run_git(root, "init", "--quiet")
            _run_git(remote, "init", "--bare", "--quiet")
            _run_git(root, "config", "user.email", "test@example.com")
            _run_git(root, "config", "user.name", "Test User")

            payload = b"binary-source\x00\xff\x80\n"
            source = root / "analyses" / "probe.bin"
            source.parent.mkdir(parents=True)
            source.write_bytes(payload)
            _run_git(root, "add", "--", "analyses/probe.bin")
            _run_git(root, "commit", "--quiet", "-m", "add binary probe")
            ancestor = _run_git(root, "rev-parse", "HEAD")

            (root / "next.txt").write_text("next\n", encoding="utf-8")
            _run_git(root, "add", "--", "next.txt")
            _run_git(root, "commit", "--quiet", "-m", "add descendant")
            descendant = _run_git(root, "rev-parse", "HEAD")
            _run_git(root, "remote", "add", "origin", str(remote))
            _run_git(root, "update-ref", "refs/remotes/origin/repa", descendant)
            _run_git(
                root,
                "push",
                "--quiet",
                "origin",
                f"{descendant}:refs/heads/repa",
            )

            with mock.patch(
                "credit_redistribution.orchestration.PROJECT_ROOT", root
            ), mock.patch(
                "credit_redistribution.git_provenance."
                "AUTHORITATIVE_REMOTE_URL",
                str(remote),
            ):
                self.assertTrue(_git_commit_is_ancestor(ancestor, descendant))
                self.assertFalse(_git_commit_is_ancestor(descendant, ancestor))
                self.assertEqual(
                    _git_blob_sha256(ancestor, "analyses/probe.bin"),
                    hashlib.sha256(payload).hexdigest(),
                )
                with self.assertRaisesRegex(
                    RuntimeError, "Could not verify cross-checkpoint Git ancestry"
                ):
                    _git_commit_is_ancestor("0" * 40, descendant)
                with self.assertRaisesRegex(
                    RuntimeError, "source is absent from its recorded Git commit"
                ):
                    _git_blob_sha256(ancestor, "analyses/missing.bin")
                with mock.patch.dict(os.environ, {
                    "GIT_DIR": str(root / "redirected.git"),
                    "GIT_OBJECT_DIRECTORY": str(root / "redirected-objects"),
                }):
                    self.assertTrue(
                        _git_commit_is_ancestor(ancestor, descendant)
                    )
                    self.assertEqual(
                        _git_blob_sha256(ancestor, "analyses/probe.bin"),
                        hashlib.sha256(payload).hexdigest(),
                    )
                manifest = {
                    "analyses/probe.bin": hashlib.sha256(payload).hexdigest(),
                }
                _verify_continuation_git_provenance(descendant, manifest)

                _run_git(
                    root,
                    "update-ref",
                    "refs/remotes/origin/repa",
                    ancestor,
                )
                with self.assertRaisesRegex(RuntimeError, "not pushed"):
                    _verify_continuation_git_provenance(descendant, manifest)
                _run_git(
                    root,
                    "update-ref",
                    "refs/remotes/origin/repa",
                    descendant,
                )
                (root / "next.txt").write_text("dirty\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "not clean"):
                    _verify_continuation_git_provenance(descendant, manifest)

                (root / "next.txt").write_text("next\n", encoding="utf-8")
                _run_git(
                    root,
                    "commit",
                    "--allow-empty",
                    "--quiet",
                    "-m",
                    "unpushed",
                )
                unpushed = _run_git(root, "rev-parse", "HEAD")
                _run_git(
                    root,
                    "update-ref",
                    "refs/remotes/origin/repa",
                    unpushed,
                )
                with self.assertRaisesRegex(
                    RuntimeError, "authoritative remote"
                ):
                    _verify_continuation_git_provenance(unpushed, manifest)

    def test_git_provenance_rejects_hidden_index_and_source_changes(self):
        for index_flag in ("--assume-unchanged", "--skip-worktree"):
            with self.subTest(index_flag=index_flag), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _run_git(root, "init", "--quiet")
                _run_git(root, "config", "user.email", "test@example.com")
                _run_git(root, "config", "user.name", "Test User")
                source = root / "source.py"
                source.write_text("original\n", encoding="utf-8")
                _run_git(root, "add", "--", "source.py")
                _run_git(root, "commit", "--quiet", "-m", "add source")
                commit = _run_git(root, "rev-parse", "HEAD")
                _run_git(root, "update-ref", "refs/remotes/origin/repa", commit)
                manifest = {
                    "source.py": hashlib.sha256(b"original\n").hexdigest(),
                }
                verify_worktree_source_manifest(root, commit, manifest)

                _run_git(root, "update-index", index_flag, "--", "source.py")
                source.write_text("hidden change\n", encoding="utf-8")
                self.assertEqual(
                    _run_git(root, "status", "--porcelain", "--untracked-files=all"),
                    "",
                )
                with self.assertRaisesRegex(
                    RuntimeError, "assume-unchanged/skip-worktree"
                ):
                    repository_state(root, verify_remote=False)
                with self.assertRaisesRegex(
                    RuntimeError, "assume-unchanged/skip-worktree"
                ):
                    verify_worktree_source_manifest(root, commit, manifest)

                clear_flag = (
                    "--no-assume-unchanged"
                    if index_flag == "--assume-unchanged"
                    else "--no-skip-worktree"
                )
                _run_git(root, "update-index", clear_flag, "--", "source.py")
                with self.assertRaisesRegex(
                    RuntimeError, "Working-tree source differs"
                ):
                    verify_worktree_source_manifest(root, commit, manifest)

    def test_cross_checkpoint_provenance_rejects_local_history_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _run_git(root, "init", "--quiet")
            _run_git(root, "config", "user.email", "test@example.com")
            _run_git(root, "config", "user.name", "Test User")
            empty_tree = _run_git(root, "hash-object", "-t", "tree", "/dev/null")
            root_a = _run_git(root, "commit-tree", empty_tree, "-m", "root a")
            root_b = _run_git(root, "commit-tree", empty_tree, "-m", "root b")
            protocol = {
                "git": {
                    "commit": root_a,
                    "origin_repa_divergence": "0\t0",
                },
                "project_source_sha256": {"probe.py": "c" * 64},
            }

            _run_git(root, "replace", "--graft", root_b, root_a)
            default_environment = sanitized_git_environment()
            default_environment.pop("GIT_NO_REPLACE_OBJECTS")
            default_ancestry = subprocess.run(
                ["git", "merge-base", "--is-ancestor", root_a, root_b],
                cwd=root,
                env=default_environment,
            )
            self.assertEqual(default_ancestry.returncode, 0)
            with mock.patch(
                "credit_redistribution.orchestration.PROJECT_ROOT", root
            ), self.assertRaisesRegex(RuntimeError, "replace refs are forbidden"):
                _verify_cross_checkpoint_git_provenance(protocol, root_b)

            _run_git(root, "replace", "--delete", root_b)
            grafts = root / ".git" / "info" / "grafts"
            grafts.write_text(f"{root_b} {root_a}\n", encoding="ascii")
            with mock.patch(
                "credit_redistribution.orchestration.PROJECT_ROOT", root
            ), self.assertRaisesRegex(RuntimeError, "grafts are forbidden"):
                _verify_cross_checkpoint_git_provenance(protocol, root_b)

    def test_cross_checkpoint_provenance_rejects_invalid_metadata(self):
        valid = {
            "git": {
                "commit": "a" * 40,
                "origin_repa_divergence": "0\t0",
            },
            "project_source_sha256": {"analyses/probe.py": "c" * 64},
        }
        invalid_cases = (
            ("short cross commit", {"git": {"commit": "abc"}}, "b" * 40,
             "cross-checkpoint Git commit is not a full SHA-1"),
            ("short current commit", valid, "abc",
             "continuation Git commit is not a full SHA-1"),
            ("unpushed cross commit", {
                **valid,
                "git": {
                    "commit": "a" * 40,
                    "origin_repa_divergence": "1\t0",
                },
            }, "b" * 40, "was not run from pushed code"),
        )
        for name, protocol, current_commit, message in invalid_cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                RuntimeError, message
            ):
                _verify_cross_checkpoint_git_provenance(
                    protocol, current_commit
                )

        invalid_sources = (
            ({}, "has no source binding"),
            ({"": "c" * 64}, "source path is invalid"),
            ({"/absolute.py": "c" * 64}, "source path is invalid"),
            ({"analyses/../probe.py": "c" * 64}, "source path is invalid"),
            ({"analyses:probe.py": "c" * 64}, "source path is invalid"),
            ({"analyses/probe.py": "short"}, "source hash is invalid"),
        )
        for source_hashes, message in invalid_sources:
            protocol = copy.deepcopy(valid)
            protocol["project_source_sha256"] = source_hashes
            with self.subTest(
                source_hashes=source_hashes
            ), mock.patch(
                "credit_redistribution.orchestration._git_commit_is_ancestor",
                return_value=True,
            ), self.assertRaisesRegex(RuntimeError, message):
                _verify_cross_checkpoint_git_provenance(protocol, "b" * 40)

    def test_cross_checkpoint_provenance_accepts_pushed_ancestor(self):
        cross_commit = "a" * 40
        current_commit = "b" * 40
        source_sha256 = "c" * 64
        protocol = {
            "git": {
                "commit": cross_commit,
                "origin_repa_divergence": "0\t0",
            },
            "project_source_sha256": {"analyses/probe.py": source_sha256},
        }
        with mock.patch(
            "credit_redistribution.orchestration._git_commit_is_ancestor",
            return_value=True,
        ) as ancestry, mock.patch(
            "credit_redistribution.orchestration._git_blob_sha256",
            return_value=source_sha256,
        ) as blob_sha256:
            _verify_cross_checkpoint_git_provenance(protocol, current_commit)
        ancestry.assert_called_once_with(cross_commit, current_commit)
        blob_sha256.assert_called_once_with(cross_commit, "analyses/probe.py")

    def test_cross_checkpoint_provenance_rejects_unrelated_commit(self):
        protocol = {
            "git": {
                "commit": "a" * 40,
                "origin_repa_divergence": "0\t0",
            },
            "project_source_sha256": {"analyses/probe.py": "c" * 64},
        }
        with mock.patch(
            "credit_redistribution.orchestration._git_commit_is_ancestor",
            return_value=False,
        ), self.assertRaisesRegex(RuntimeError, "not an ancestor"):
            _verify_cross_checkpoint_git_provenance(protocol, "b" * 40)

    def test_cross_checkpoint_provenance_rejects_git_blob_drift(self):
        protocol = {
            "git": {
                "commit": "a" * 40,
                "origin_repa_divergence": "0\t0",
            },
            "project_source_sha256": {"analyses/probe.py": "c" * 64},
        }
        with mock.patch(
            "credit_redistribution.orchestration._git_commit_is_ancestor",
            return_value=True,
        ), mock.patch(
            "credit_redistribution.orchestration._git_blob_sha256",
            return_value="d" * 64,
        ), self.assertRaisesRegex(RuntimeError, "Git source binding changed"):
            _verify_cross_checkpoint_git_provenance(protocol, "b" * 40)

    def test_preaggregation_revalidates_every_completion_binding(self):
        protocol_sha256 = "a" * 64
        protocol = {"heldout_evaluation_output": "/sealed/evaluation"}
        checkpoint_specs = {
            branch: {"path": f"/{branch}.pth", "sha256": "b" * 64}
            for branch in BRANCHES
        }
        transcripts = {branch: "c" * 64 for branch in BRANCHES}
        integrity = {branch: {"ledger": "d" * 64} for branch in BRANCHES}
        trainer = {branch: "e" * 64 for branch in BRANCHES}
        completion = {
            "checkpoint_file_sha256": {
                branch: spec["sha256"] for branch, spec in checkpoint_specs.items()
            },
            "transcript_final_chain_digests": transcripts,
            "branch_integrity": integrity,
            "trainer_state_sha256": trainer,
        }
        validated = (
            protocol,
            protocol_sha256,
            {},
            checkpoint_specs,
            transcripts,
            integrity,
            trainer,
        )
        with mock.patch(
            "credit_redistribution.orchestration._load_sealed",
            return_value=completion,
        ), mock.patch(
            "credit_redistribution.orchestration.validate_protocol_for_evaluation",
            return_value=validated,
        ):
            _revalidate_before_aggregation(protocol, protocol_sha256)
            changed = copy.deepcopy(completion)
            changed["branch_integrity"][BRANCHES[0]] = {"ledger": "f" * 64}
            with mock.patch(
                "credit_redistribution.orchestration._load_sealed",
                return_value=changed,
            ), self.assertRaisesRegex(RuntimeError, "branch_integrity"):
                _revalidate_before_aggregation(protocol, protocol_sha256)

    def test_prerequisite_chain_binds_both_preregistrations_and_stage_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_root = root / "base"
            cross_root = root / "cross"
            base_protocol_path = base_root / "protocol.json"
            base_protocol_sha256 = _write_protocol(
                base_protocol_path, {"name": "base-protocol"}
            )
            _write_summary(base_root, "plumbing", base_protocol_sha256)

            v1 = root / "v1.json"
            v2 = root / "v2.json"
            base_preregister = root / "base-preregister.json"
            for path, version in ((v1, 1), (v2, 2), (base_preregister, 1)):
                _write_json(path, {"version": version})

            stage_order = ["plumbing", "confirmatory_credit"]
            legacy_v1 = (
                "/home/dev/promoe-probes/"
                "credit-balance-lossfree-s0-200k-v1-preregister.json"
            )
            legacy_v2 = (
                "/home/dev/promoe-probes/"
                "credit-balance-lossfree-s0-200k-v2-preregister.json"
            )
            cross_protocol = {
                "effective_preregistrations": [
                    {"version": 1, "path": legacy_v1, "sha256": sha256_file(v1)},
                    {"version": 2, "path": legacy_v2, "sha256": sha256_file(v2)},
                ],
                "stage_order": stage_order,
                "base_protocol": {
                    "canonical_json_sha256": base_protocol_sha256,
                },
                "git": {
                    "commit": "a" * 40,
                    "origin_repa_divergence": "0\t0",
                },
                "project_source_sha256": {
                    "analyses/probe.py": "c" * 64,
                },
                "checkpoints": {},
            }
            cross_protocol_path = cross_root / "protocol.json"
            cross_protocol_sha256 = _write_protocol(
                cross_protocol_path, cross_protocol
            )
            _write_summary(cross_root, "confirmatory", cross_protocol_sha256)

            protocol = {
                "git": {"commit": "b" * 40},
                "prerequisites": {
                    "base_gate": {
                        "preregister_path": str(base_preregister),
                        "preregister_file_sha256": sha256_file(base_preregister),
                        "protocol_path": str(base_protocol_path),
                        "protocol_canonical_sha256": base_protocol_sha256,
                        "output_root": str(base_root),
                        "required_summaries": ["plumbing"],
                    },
                    "cross_checkpoint_gate": {
                        "preregister_v1_path": str(v1),
                        "preregister_v1_file_sha256": sha256_file(v1),
                        "preregister_v2_path": str(v2),
                        "preregister_v2_file_sha256": sha256_file(v2),
                        "protocol_path": str(cross_protocol_path),
                        "output_root": str(cross_root),
                        "required_stage_order": stage_order,
                        "required_summaries": ["confirmatory"],
                    },
                },
            }
            archived_paths = {
                legacy_v1: v1,
                legacy_v2: v2,
            }
            with mock.patch(
                "credit_redistribution.orchestration."
                "_verify_continuation_git_provenance"
            ) as verify_current, mock.patch(
                "credit_redistribution.orchestration."
                "_verify_cross_checkpoint_git_provenance"
            ) as verify_git, mock.patch(
                "credit_redistribution.orchestration."
                "resolve_archived_artifact_path",
                side_effect=lambda path: archived_paths[path],
            ):
                result = verify_prerequisites(protocol)
            verify_current.assert_called_once_with(
                "b" * 40,
                protocol.get("project_source_file_sha256"),
            )
            verify_git.assert_called_once_with(cross_protocol, "b" * 40)
            self.assertEqual(result["base_protocol_sha256"], base_protocol_sha256)
            self.assertEqual(
                result["cross_checkpoint_protocol_sha256"],
                cross_protocol_sha256,
            )

            v2.write_text("changed", encoding="utf-8")
            with mock.patch(
                "credit_redistribution.orchestration."
                "resolve_archived_artifact_path",
                side_effect=lambda path: archived_paths[path],
            ), self.assertRaisesRegex(RuntimeError, "preregistration changed"):
                verify_prerequisites(protocol)


if __name__ == "__main__":
    unittest.main()
