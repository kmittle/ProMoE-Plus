from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

from credit_redistribution.transcript import (
    FIELD_ORDER,
    JsonlLedger,
    TranscriptOnlyRecorder,
    TrainingInputTranscript,
    build_step_record,
    iter_replayed_local_records,
    persisted_record_digest,
    persisted_identity_field_hashes,
    validate_local_transcript_replay,
    _replay_get_rng_state,
    _replay_hash_dataset_samples,
)


def _tensors(offset=0.0):
    image = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) + offset
    return {
        "latent_parameters": image,
        "realized_z": image + 1,
        "sampled_u": torch.tensor([0.1], dtype=torch.float32),
        "timestep": torch.tensor([100.0], dtype=torch.float32),
        "sigma": torch.tensor([0.1], dtype=torch.float32),
        "diffusion_noise": image + 2,
        "noised_model_input": image + 3,
        "denoising_target": image + 4,
        "effective_labels": torch.tensor([9], dtype=torch.int64),
    }


def _make_replay_fixture(root, start_step=10, final_step=11):
    dataset = root / "dataset"
    paths = []
    for class_name, offset in (("0000", 0.0), ("0001", 1.0)):
        class_dir = dataset / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        path = class_dir / f"sample-{class_name}.latent.npz"
        latent = np.full((8, 2, 2), offset, dtype=np.float32)
        np.savez(path, latent=latent, latent_flip=latent + 2.0)
        paths.append(str(path.resolve()))

    runtime_cfg = SimpleNamespace(
        latent_data_path=str(dataset),
        world_size=1,
        global_seed=0,
        total_train_batch_size=2,
        img_num_workers=0,
        prefetch_factor=2,
        weighting_scheme="uniform",
        logit_mean=0.0,
        logit_std=1.0,
        sigmoid_scale=1.0,
        mode_scale=1.29,
        shift=1.0,
        num_train_timesteps=1000,
        DiT_B_config={"class_dropout_prob": 0.25},
    )
    labels = [int(Path(path).parent.name) for path in paths]
    dataset_hash = _replay_hash_dataset_samples(
        paths,
        labels,
        "train.LatentFolder",
        dataset,
    )
    torch.manual_seed(1234)
    checkpoint = {
        "step": start_step - 1,
        "trainer_state": {
            "next_step": start_step,
            "world_size": 1,
            "grad_mix": 1,
            "sampler_epoch": 0,
            "sampler_batch_offset": 0,
            "batches_per_epoch": 1,
            "sampler_contract": {
                "version": 1,
                "type": "distributed",
                "global_seed": 0,
                "per_rank_batch_size": 2,
                "drop_last": False,
                "case1_prob": None,
                "dataset": {
                    "version": 1,
                    "type": "train.LatentFolder",
                    "num_samples": len(paths),
                    "ordered_samples_sha256": dataset_hash,
                },
            },
            "rank_states": [
                {"rank": 0, "rng_state": _replay_get_rng_state(torch.device("cpu"))}
            ],
        },
    }
    return dataset, runtime_cfg, checkpoint, start_step, final_step


def _materialize_replay_transcript(
    artifact_root, branch, dataset, runtime_cfg, checkpoint, start_step, final_step
):
    path = (
        Path(artifact_root)
        / "transcripts"
        / branch
        / "rank-00.jsonl"
    )
    ledger = JsonlLedger(path, start_step)
    records = list(
        iter_replayed_local_records(
            checkpoint,
            runtime_cfg,
            dataset,
            start_step,
            final_step,
            rank=0,
            device="cpu",
        )
    )
    for record in records:
        ledger.append_or_verify(record)
    return path, records


class TranscriptTest(unittest.TestCase):
    def test_deterministic_replay_matches_transcript_and_restores_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, runtime_cfg, checkpoint, start_step, final_step = (
                _make_replay_fixture(root)
            )
            checkpoint_path = root / "frozen.pth"
            torch.save(checkpoint, checkpoint_path)
            artifact_root = root / "artifacts"
            transcript_path, records = _materialize_replay_transcript(
                artifact_root,
                "measure_only_control",
                dataset,
                runtime_cfg,
                checkpoint,
                start_step,
                final_step,
            )
            self.assertEqual(len(records), 2)
            self.assertTrue(
                all(
                    not Path(path).is_absolute()
                    for path in records[0]["relative_latent_paths"]
                )
            )

            random.seed(91)
            np.random.seed(92)
            torch.manual_seed(93)
            before_python = random.getstate()
            before_numpy = np.random.get_state()
            before_torch = torch.get_rng_state().clone()
            validate_local_transcript_replay(
                artifact_root=artifact_root,
                branch="measure_only_control",
                initial_checkpoint_path=checkpoint_path,
                runtime_cfg=runtime_cfg,
                dataset_root=dataset,
                start_step=start_step,
                final_step=final_step,
                device="cpu",
            )
            self.assertEqual(random.getstate(), before_python)
            after_numpy = np.random.get_state()
            self.assertEqual(after_numpy[0], before_numpy[0])
            np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
            self.assertEqual(after_numpy[2:], before_numpy[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
            self.assertTrue(transcript_path.is_file())

    def test_replay_rejects_synchronized_tensor_digest_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, runtime_cfg, checkpoint, start_step, final_step = (
                _make_replay_fixture(root)
            )
            checkpoint_path = root / "frozen.pth"
            torch.save(checkpoint, checkpoint_path)
            artifact_root = root / "artifacts"
            transcript_path, _ = _materialize_replay_transcript(
                artifact_root,
                "measure_only_control",
                dataset,
                runtime_cfg,
                checkpoint,
                start_step,
                final_step,
            )

            record = json.loads(transcript_path.read_text(encoding="utf-8").splitlines()[0])
            for field_name in FIELD_ORDER[2:]:
                record["field_sha256"][field_name] = "f" * 64
            record["step_digest"] = "e" * 64
            record["record_digest"] = persisted_record_digest(record)
            record["chain_digest"] = JsonlLedger._chain("0" * 64, record)
            transcript_path.write_text(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Replay tensor content differs"):
                validate_local_transcript_replay(
                    artifact_root=artifact_root,
                    branch="measure_only_control",
                    initial_checkpoint_path=checkpoint_path,
                    runtime_cfg=runtime_cfg,
                    dataset_root=dataset,
                    start_step=start_step,
                    final_step=final_step,
                    device="cpu",
                )

    def test_persisted_commitments_are_recomputable(self):
        record = build_step_record(
            301001,
            0,
            ["0009/sample.latent.npz"],
            torch.tensor([9], dtype=torch.int64),
            _tensors(),
        )
        self.assertEqual(
            persisted_identity_field_hashes(record),
            {
                key: record["field_sha256"][key]
                for key in ("relative_latent_paths", "original_labels")
            },
        )
        tampered = dict(record)
        tampered["original_labels"] = [10]
        self.assertNotEqual(
            persisted_identity_field_hashes(tampered)["original_labels"],
            record["field_sha256"]["original_labels"],
        )

    def test_reference_match_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            latent = dataset / "0009" / "sample.latent.npz"
            latent.parent.mkdir()
            latent.touch()
            labels = torch.tensor([9], dtype=torch.int64)
            reference_root = root / "reference"
            reference = TrainingInputTranscript(
                artifact_root=reference_root,
                branch="measure_only_control",
                start_step=301001,
                dataset_root=dataset,
            )
            digest = reference.record(
                301001, [str(latent)], labels, _tensors()
            )

            matched = TrainingInputTranscript(
                artifact_root=root / "matched",
                branch="matched_credit_rate_redistribution",
                start_step=301001,
                dataset_root=dataset,
                reference_artifact_root=reference_root,
            )
            self.assertEqual(
                matched.record(301001, [str(latent)], labels, _tensors()),
                digest,
            )

            mismatched = TrainingInputTranscript(
                artifact_root=root / "mismatched",
                branch="matched_credit_rate_redistribution",
                start_step=301001,
                dataset_root=dataset,
                reference_artifact_root=reference_root,
            )
            with self.assertRaisesRegex(RuntimeError, "Reference transcript mismatch"):
                mismatched.record(
                    301001, [str(latent)], labels, _tensors(offset=0.5)
                )

    def test_transcript_only_recorder_requires_every_locked_update(self):
        class Embedder(nn.Module):
            def forward(self, labels):
                return torch.zeros(labels.shape[0], 2), labels

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.y_embedder = Embedder()

        protocol = {
            "branches": {"start_step": 301001},
            "checkpoint": {
                "frozen_path": "/tmp/frozen.pth",
                "sha256": "0" * 64,
            },
            "source_anchor": {
                "training_facts": {"global_batch_size": 256},
            },
        }
        runtime_cfg = SimpleNamespace(
            num_steps=301021,
            model_name="ProMoE_TC_B",
            total_train_batch_size=256,
            global_seed=0,
            grad_mix=1,
            use_gradient_checkpointing=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            latent = dataset / "0004" / "sample.latent.npz"
            latent.parent.mkdir(parents=True)
            latent.touch()
            runtime_cfg.latent_data_path = str(dataset)
            cfg = {
                "enabled": True,
                "branch": "measure_only_control",
                "execution_mode": "deterministic_replay_baseline",
                "initial_checkpoint_path": "/tmp/frozen.pth",
                "preregister_v3_path": "/tmp/v3.json",
                "preregister_v4_path": "/tmp/v4.json",
                "artifact_root": str(root / "artifacts"),
            }
            with mock.patch(
                "credit_redistribution.transcript.load_effective_protocol",
                return_value=protocol,
            ), mock.patch(
                "credit_redistribution.transcript._dist_world_size",
                return_value=4,
            ):
                recorder = TranscriptOnlyRecorder(Model(), runtime_cfg, cfg)
            recorder.transcript.world_size = 1
            labels = torch.tensor([4], dtype=torch.int64)
            for offset in range(20):
                recorder.begin_step(301001 + offset)
                recorder.model.y_embedder(labels)
                inputs = {
                    "paths": [str(latent)],
                    "original_labels": labels,
                    "tensors": {
                        key: value
                        for key, value in _tensors().items()
                        if key != "effective_labels"
                    },
                }
                recorder.record_before_optimizer(inputs)
                recorder.after_optimizer_step()
            recorder.close()


if __name__ == "__main__":
    unittest.main()
