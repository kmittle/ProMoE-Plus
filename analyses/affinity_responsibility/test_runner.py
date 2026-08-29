"""Tests for manifest sealing and worker lifecycle safeguards."""

import copy
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analyses.affinity_responsibility import runner
from analyses.run_rcl_responsibility_probe_batch import _result_exit_code


class CanonicalManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(runner.GATE_MANIFEST.read_text(encoding="utf-8"))

    def _load(self, payload):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "manifest.json"
            path.write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
            with mock.patch.object(runner, "GATE_MANIFEST", path):
                return runner._canonical_gate_manifest()

    def test_canonical_manifest_is_accepted(self):
        self.assertEqual(self._load(copy.deepcopy(self.payload)), self.payload)

    def test_prior_observation_tampering_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["prior_observation"]["fresh_result_seen"] = True
        with self.assertRaisesRegex(ValueError, "canonical schema"):
            self._load(payload)

    def test_extra_top_level_field_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["post_hoc_note"] = "result looked promising"
        with self.assertRaisesRegex(ValueError, "canonical schema"):
            self._load(payload)

    def test_extra_nested_split_field_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["splits"]["confirmatory"]["result_seen"] = True
        with self.assertRaisesRegex(ValueError, "canonical schema"):
            self._load(payload)

    def test_numeric_type_coercion_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["protocol"]["support_batch_size"] = 64.0
        with self.assertRaisesRegex(ValueError, "canonical schema"):
            self._load(payload)

    def test_boolean_integer_coercion_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["locked_before_any_fresh_rcl_responsibility_result"] = 1
        with self.assertRaisesRegex(ValueError, "canonical schema"):
            self._load(payload)

    def test_source_hash_set_covers_direct_model_dependencies_without_duplicates(self):
        self.assertIn("models/modules.py", runner.STATIC_SOURCE_PATHS)
        self.assertIn("models/phase_metric.py", runner.STATIC_SOURCE_PATHS)
        self.assertEqual(
            len(runner.STATIC_SOURCE_PATHS),
            len(set(runner.STATIC_SOURCE_PATHS)),
        )


class CliExitCodeTest(unittest.TestCase):
    def test_negative_gate_result_is_failure(self):
        self.assertEqual(_result_exit_code({"passed": False}), 1)

    def test_positive_or_non_gate_result_is_success(self):
        self.assertEqual(_result_exit_code({"passed": True}), 0)
        self.assertEqual(_result_exit_code({"protocol_sha256": "abc"}), 0)


class _FakeCommandQueue:
    def __init__(self):
        self.values = []

    def put(self, value):
        self.values.append(value)


class _KillRequiredProcess:
    def __init__(self):
        self.pid = 1234
        self.exitcode = None
        self.alive = True
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.alive = False
        self.exitcode = -9


class WorkerLifecycleTest(unittest.TestCase):
    def test_stop_workers_kills_a_terminate_resistant_process(self):
        process = _KillRequiredProcess()
        command_queue = _FakeCommandQueue()
        runner._stop_workers({"cuda:0": (process, command_queue)})
        self.assertEqual(command_queue.values, [None])
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_message_collection_has_an_absolute_deadline(self):
        process = _KillRequiredProcess()
        workers = {"cuda:0": (process, _FakeCommandQueue())}
        with mock.patch.object(runner, "LOCKED_DEVICES", ("cuda:0",)):
            with self.assertRaises(TimeoutError):
                runner._collect_worker_messages(
                    workers,
                    queue.Queue(),
                    "run_split",
                    timeout_seconds=0.001,
                )

    def test_worker_error_is_reported_before_split_metadata(self):
        process = _KillRequiredProcess()
        workers = {"cuda:0": (process, _FakeCommandQueue())}
        messages = queue.Queue()
        messages.put({
            "device": "cuda:0",
            "phase": "run_split",
            "error": "synthetic traceback",
        })
        with mock.patch.object(runner, "LOCKED_DEVICES", ("cuda:0",)):
            with self.assertRaisesRegex(RuntimeError, "synthetic traceback"):
                runner._collect_worker_messages(
                    workers,
                    messages,
                    "run_split",
                    timeout_seconds=1,
                    split="discovery",
                )

    def test_start_failure_stops_workers_started_before_the_failure(self):
        class StartFailureProcess:
            def __init__(self, should_fail):
                self.should_fail = should_fail
                self.pid = 100 if not should_fail else None
                self.alive = False
                self.stopped = False

            def start(self):
                if self.should_fail:
                    raise RuntimeError("synthetic process-start failure")
                self.alive = True

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                self.alive = False

            def terminate(self):
                self.stopped = True
                self.alive = False

            def kill(self):
                self.stopped = True
                self.alive = False

        class StartFailureContext:
            def __init__(self):
                self.processes = []

            def Queue(self):
                return _FakeCommandQueue()

            def Process(self, target, args):
                process = StartFailureProcess(bool(self.processes))
                self.processes.append(process)
                return process

        context = StartFailureContext()
        with mock.patch.object(
            runner.multiprocessing,
            "get_context",
            return_value=context,
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                runner._start_workers({}, "output")
        self.assertEqual(len(context.processes), 2)
        self.assertFalse(context.processes[0].alive)


if __name__ == "__main__":
    unittest.main()
