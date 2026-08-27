import copy
import datetime
import io
import pickle
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

from train import (
    AUGMENTATION_SEED_VERSION,
    LatentFolder,
    ResumableBatchSampler,
    _augmentation_seed,
    _build_latent_class_to_idx,
    _capture_rng_state,
    _dataset_sampler_identity,
    _legacy_resume_state,
    _resume_state_from_checkpoint,
    _restore_rng_state,
    load_latest_checkpoint,
    save_checkpoint,
)


def _checkpoint_save_failure_worker(rank, world_size, init_file, blocker, queue):
    dist.init_process_group(
        backend='gloo',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=10),
    )
    try:
        try:
            save_checkpoint(
                model=None,
                ema_model=None,
                optimizer=None,
                step=7,
                trainer_progress={},
                checkpoint_dir=blocker,
                global_seed=0,
                sampler_contract={'global_seed': 0},
            )
        except RuntimeError as error:
            queue.put((rank, 'Checkpoint persistence failed' in str(error)))
        else:
            queue.put((rank, False))
    finally:
        dist.destroy_process_group()


class TrainResumeTests(unittest.TestCase):
    @staticmethod
    def _sampler_contract(global_seed=0, sampler_type='distributed'):
        return {
            'version': 1,
            'type': sampler_type,
            'global_seed': global_seed,
            'per_rank_batch_size': 3,
            'drop_last': False,
            'case1_prob': (
                0.5 if sampler_type == 'structured_distributed' else None
            ),
            'dataset': {
                'version': 1,
                'type': 'tests.FixedDataset',
                'num_samples': 9,
                'ordered_samples_sha256': 'fixed-test-digest',
            },
        }

    @staticmethod
    def _trainer_state(step, grad_mix=1, batches_per_epoch=3):
        next_step = step + 1
        data_batches_seen = next_step * grad_mix
        sampler_epoch, sampler_batch_offset = divmod(
            data_batches_seen,
            batches_per_epoch,
        )
        return {
            'version': 2,
            'augmentation_seed_version': AUGMENTATION_SEED_VERSION,
            'global_seed': 0,
            'sampler_contract': TrainResumeTests._sampler_contract(),
            'world_size': 1,
            'next_step': next_step,
            'data_batches_seen': data_batches_seen,
            'sampler_epoch': sampler_epoch,
            'sampler_batch_offset': sampler_batch_offset,
            'grad_mix': grad_mix,
            'batches_per_epoch': batches_per_epoch,
            'rank_states': [{
                'rank': 0,
                'rng_state': _capture_rng_state(),
            }],
        }

    @staticmethod
    def _make_loader(dataset, epoch, start_batch, loader_seed):
        distributed = DistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=41,
        )
        batch_sampler = BatchSampler(
            distributed,
            batch_size=3,
            drop_last=False,
        )
        resumable = ResumableBatchSampler(batch_sampler, seed=43, rank=0)
        resumable.set_epoch(epoch)
        resumable.set_start_batch(start_batch)
        loader_generator = torch.Generator().manual_seed(loader_seed)
        return DataLoader(
            dataset,
            batch_sampler=resumable,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            generator=loader_generator,
        )

    @staticmethod
    def _materialize_batches(loader):
        return [
            (tuple(paths), labels.tolist(), latents.clone())
            for paths, labels, latents in loader
        ]

    @staticmethod
    def _seed_rng_streams():
        random.seed(7)
        np.random.seed(11)
        torch.manual_seed(13)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(17)

    @staticmethod
    def _draw_rng_streams():
        values = {
            'python': random.random(),
            'numpy': float(np.random.random()),
            'torch': torch.rand(4),
        }
        if torch.cuda.is_available():
            values['cuda'] = torch.rand(4, device='cuda').cpu()
        return values

    @staticmethod
    def _disturb_rng_streams():
        random.seed(101)
        np.random.seed(103)
        torch.manual_seed(107)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(109)

    def _assert_rng_continuation(self, state):
        expected = self._draw_rng_streams()
        self._disturb_rng_streams()
        _restore_rng_state(state)
        actual = self._draw_rng_streams()
        self.assertEqual(actual['python'], expected['python'])
        self.assertEqual(actual['numpy'], expected['numpy'])
        torch.testing.assert_close(
            actual['torch'], expected['torch'], rtol=0, atol=0
        )
        if 'cuda' in expected:
            torch.testing.assert_close(
                actual['cuda'], expected['cuda'], rtol=0, atol=0
            )

    def test_rng_state_round_trip(self):
        self._seed_rng_streams()
        state = _capture_rng_state()
        self.assertEqual(state['numpy']['state'].dtype, torch.int64)
        self._assert_rng_continuation(state)

    def test_rank_zero_checkpoint_failure_reaches_every_rank(self):
        context = torch.multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            init_file = str(root / 'gloo-init')
            blocker = root / 'checkpoint-dir-is-a-file'
            blocker.write_text('block directory creation', encoding='utf-8')
            queue = context.Queue()
            processes = [
                context.Process(
                    target=_checkpoint_save_failure_worker,
                    args=(rank, 2, init_file, str(blocker), queue),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=20)
            alive = [process for process in processes if process.is_alive()]
            for process in alive:
                process.terminate()
                process.join(timeout=5)
            self.assertEqual(alive, [])
            results = sorted(queue.get(timeout=2) for _ in range(2))
            self.assertEqual(results, [(0, True), (1, True)])
            self.assertTrue(all(process.exitcode == 0 for process in processes))

    def test_rng_state_pickle_round_trip(self):
        self._seed_rng_streams()
        state = _capture_rng_state()
        deserialized = pickle.loads(pickle.dumps(state))
        self._assert_rng_continuation(deserialized)

    def test_rng_state_torch_save_round_trip(self):
        self._seed_rng_streams()
        state = _capture_rng_state()
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        deserialized = torch.load(buffer, weights_only=False)
        self._assert_rng_continuation(deserialized)

    def test_rng_state_restores_legacy_uint32_numpy_state(self):
        self._seed_rng_streams()
        state = _capture_rng_state()
        state['numpy']['state'] = state['numpy']['state'].to(torch.uint32)
        self._assert_rng_continuation(state)

    def test_v2_checkpoint_accepts_legacy_uint32_numpy_state(self):
        checkpoint = {
            'step': 8,
            'trainer_state': self._trainer_state(8),
        }
        rng_state = checkpoint['trainer_state']['rank_states'][0]['rng_state']
        rng_state['numpy']['state'] = rng_state['numpy']['state'].to(
            torch.uint32
        )
        resumed = _resume_state_from_checkpoint(
            checkpoint,
            rank=0,
            world_size=1,
            grad_mix=1,
            batches_per_epoch=3,
            fallback_seed=0,
            sampler_contract=self._sampler_contract(),
        )
        self.assertEqual(resumed['next_step'], 9)

    def test_legacy_checkpoint_resumes_after_saved_step(self):
        state = _legacy_resume_state(
            checkpoint_step=300000,
            grad_mix=2,
            batches_per_epoch=5005,
            fallback_seed=17,
        )
        self.assertEqual(state['next_step'], 300001)
        self.assertEqual(state['data_batches_seen'], 600002)
        self.assertEqual(
            (state['sampler_epoch'], state['sampler_batch_offset']),
            divmod(600002, 5005),
        )
        self.assertTrue(state['legacy_checkpoint'])

    def test_resumable_sampler_matches_uninterrupted_suffix(self):
        dataset = list(range(31))
        distributed = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=1,
            shuffle=True,
            seed=23,
        )
        batch_sampler = BatchSampler(distributed, batch_size=3, drop_last=False)
        resumable = ResumableBatchSampler(batch_sampler, seed=29, rank=1)
        resumable.set_epoch(4)
        uninterrupted = list(resumable)

        resumable.set_epoch(4)
        resumable.set_start_batch(2)
        self.assertEqual(list(resumable), uninterrupted[2:])
        self.assertEqual(len(resumable), len(uninterrupted) - 2)

    def test_augmentation_seed_is_bound_to_absolute_sampler_position(self):
        expected = _augmentation_seed(29, 1, 4, 2, 0, 7)
        self.assertEqual(expected, _augmentation_seed(29, 1, 4, 2, 0, 7))
        self.assertNotEqual(expected, _augmentation_seed(29, 1, 5, 2, 0, 7))
        self.assertNotEqual(expected, _augmentation_seed(29, 1, 4, 2, 1, 7))

    def test_latent_flip_uses_seeded_index_instead_of_worker_rng(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            latent_path = Path(temporary_dir) / 'example.latent.npz'
            np.savez(
                latent_path,
                latent=np.zeros((8, 2, 2), dtype=np.float32),
                latent_flip=np.ones((8, 2, 2), dtype=np.float32),
            )
            dataset = LatentFolder.__new__(LatentFolder)
            dataset.latent_paths = [str(latent_path)]
            dataset.class_to_idx = {Path(temporary_dir).name: 3}
            torch.manual_seed(1)
            original = dataset[(0, 0)][2]
            torch.manual_seed(999)
            flipped = dataset[(0, 1)][2]
            self.assertTrue(torch.equal(original, torch.zeros_like(original)))
            self.assertTrue(torch.equal(flipped, torch.ones_like(flipped)))

    def test_prefetched_loader_resume_matches_mid_epoch_and_boundary(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            latent_paths = []
            for index in range(12):
                latent_path = Path(temporary_dir) / f'{index:02d}.latent.npz'
                np.savez(
                    latent_path,
                    latent=np.full((8, 2, 2), index, dtype=np.float32),
                    latent_flip=np.full((8, 2, 2), -index - 1, dtype=np.float32),
                )
                latent_paths.append(str(latent_path))
            dataset = LatentFolder.__new__(LatentFolder)
            dataset.latent_paths = latent_paths
            dataset.class_to_idx = {Path(temporary_dir).name: 3}

            full_loader = self._make_loader(dataset, 5, 0, loader_seed=101)
            full = self._materialize_batches(full_loader)
            del full_loader

            resumed_loader = self._make_loader(dataset, 5, 2, loader_seed=999)
            resumed = self._materialize_batches(resumed_loader)
            del resumed_loader
            self.assertEqual(len(resumed), len(full) - 2)
            for actual, expected in zip(resumed, full[2:]):
                self.assertEqual(actual[:2], expected[:2])
                torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

            boundary_loader = self._make_loader(dataset, 6, 0, loader_seed=202)
            boundary = self._materialize_batches(boundary_loader)
            del boundary_loader
            recreated_loader = self._make_loader(dataset, 6, 0, loader_seed=303)
            recreated = self._materialize_batches(recreated_loader)
            del recreated_loader
            for actual, expected in zip(recreated, boundary):
                self.assertEqual(actual[:2], expected[:2])
                torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_checkpoint_state_requires_matching_training_geometry(self):
        checkpoint = {
            'step': 8,
            'trainer_state': {
                'version': 1,
                'world_size': 1,
                'next_step': 9,
                'data_batches_seen': 18,
                'sampler_epoch': 3,
                'sampler_batch_offset': 0,
                'grad_mix': 2,
                'batches_per_epoch': 6,
                'rank_states': [{
                    'rank': 0,
                    'rng_state': _capture_rng_state(),
                }],
            },
        }
        state = _resume_state_from_checkpoint(
            checkpoint,
            rank=0,
            world_size=1,
            grad_mix=2,
            batches_per_epoch=6,
            fallback_seed=0,
        )
        self.assertEqual(state['sampler_epoch'], 3)
        self.assertEqual(state['sampler_batch_offset'], 0)
        self.assertTrue(state['legacy_augmentation_state'])
        with self.assertRaisesRegex(ValueError, "grad_mix"):
            _resume_state_from_checkpoint(
                checkpoint,
                rank=0,
                world_size=1,
                grad_mix=1,
                batches_per_epoch=6,
                fallback_seed=0,
            )

    def test_v2_checkpoint_requires_augmentation_seed_contract(self):
        checkpoint = {
            'step': 0,
            'trainer_state': {
                'version': 2,
                'augmentation_seed_version': AUGMENTATION_SEED_VERSION + 1,
            },
        }
        with self.assertRaisesRegex(ValueError, "augmentation seed version"):
            _resume_state_from_checkpoint(
                checkpoint,
                rank=0,
                world_size=1,
                grad_mix=1,
                batches_per_epoch=1,
                fallback_seed=0,
            )

    def test_v2_checkpoint_rejects_progress_and_rng_gaps(self):
        checkpoint = {
            'step': 8,
            'trainer_state': self._trainer_state(8),
        }
        checkpoint['trainer_state']['data_batches_seen'] += 1
        with self.assertRaisesRegex(ValueError, "data_batches_seen"):
            _resume_state_from_checkpoint(
                checkpoint,
                rank=0,
                world_size=1,
                grad_mix=1,
                batches_per_epoch=3,
                fallback_seed=0,
                sampler_contract=self._sampler_contract(),
            )

        checkpoint['trainer_state'] = self._trainer_state(8)
        checkpoint['trainer_state']['rank_states'][0]['rng_state'] = None
        with self.assertRaisesRegex(ValueError, "RNG state"):
            _resume_state_from_checkpoint(
                checkpoint,
                rank=0,
                world_size=1,
                grad_mix=1,
                batches_per_epoch=3,
                fallback_seed=0,
                sampler_contract=self._sampler_contract(),
            )

    def test_v2_checkpoint_rejects_different_global_seed(self):
        checkpoint = {
            'step': 8,
            'trainer_state': self._trainer_state(8),
        }
        with self.assertRaisesRegex(ValueError, "global_seed"):
            _resume_state_from_checkpoint(
                checkpoint,
                rank=0,
                world_size=1,
                grad_mix=1,
                batches_per_epoch=3,
                fallback_seed=0,
                global_seed=123,
                sampler_contract=self._sampler_contract(global_seed=123),
            )

    def test_v2_checkpoint_rejects_different_sampler_contract(self):
        checkpoint = {
            'step': 8,
            'trainer_state': self._trainer_state(8),
        }
        with self.assertRaisesRegex(ValueError, "sampler contract"):
            _resume_state_from_checkpoint(
                checkpoint,
                rank=0,
                world_size=1,
                grad_mix=1,
                batches_per_epoch=3,
                fallback_seed=0,
                sampler_contract=self._sampler_contract(
                    sampler_type='structured_distributed'
                ),
            )

        structured_contract = self._sampler_contract(
            sampler_type='structured_distributed'
        )
        checkpoint['trainer_state']['sampler_contract'] = structured_contract
        changed_probability = copy.deepcopy(structured_contract)
        changed_probability['case1_prob'] = 0.75
        with self.assertRaisesRegex(ValueError, "sampler contract"):
            _resume_state_from_checkpoint(
                checkpoint,
                rank=0,
                world_size=1,
                grad_mix=1,
                batches_per_epoch=3,
                fallback_seed=0,
                sampler_contract=changed_probability,
            )

    def test_dataset_identity_tracks_order_but_allows_root_relocation(self):
        first = LatentFolder.__new__(LatentFolder)
        first.latent_dir = '/data/first'
        first.latent_paths = [
            '/data/first/0000/a.latent.npz',
            '/data/first/0001/b.latent.npz',
        ]
        first.class_to_idx = {'0000': 0, '0001': 1}

        relocated = LatentFolder.__new__(LatentFolder)
        relocated.latent_dir = '/data/relocated'
        relocated.latent_paths = [
            '/data/relocated/0000/a.latent.npz',
            '/data/relocated/0001/b.latent.npz',
        ]
        relocated.class_to_idx = dict(first.class_to_idx)
        self.assertEqual(
            _dataset_sampler_identity(first),
            _dataset_sampler_identity(relocated),
        )

        relocated.latent_paths.reverse()
        self.assertNotEqual(
            _dataset_sampler_identity(first),
            _dataset_sampler_identity(relocated),
        )

    def test_incompatible_explicit_checkpoint_does_not_mutate_model(self):
        model = nn.Linear(2, 1, bias=False)
        ema_model = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with torch.no_grad():
            model.weight.fill_(1.0)
            ema_model.weight.fill_(2.0)
        checkpoint_model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            checkpoint_model.weight.fill_(7.0)

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / 'ckpt_step_8.pth'
            torch.save({
                'step': 8,
                'model_state_dict': checkpoint_model.state_dict(),
                'ema_model_state_dict': checkpoint_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainer_state': {
                    **self._trainer_state(8),
                    'world_size': 2,
                },
            }, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "world size"):
                load_latest_checkpoint(
                    model,
                    ema_model,
                    optimizer,
                    str(temporary_dir),
                    resume_checkpoint_step=8,
                    rank=0,
                    world_size=1,
                    grad_mix=1,
                    batches_per_epoch=1,
                    fallback_seed=0,
                    sampler_contract=self._sampler_contract(),
                )

        torch.testing.assert_close(
            model.weight,
            torch.ones_like(model.weight),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            ema_model.weight,
            torch.full_like(ema_model.weight, 2.0),
            rtol=0,
            atol=0,
        )

    def test_non_tensor_checkpoint_value_is_rejected_before_commit(self):
        model = nn.Sequential(
            nn.Linear(2, 2, bias=False),
            nn.Linear(2, 1, bias=False),
        )
        ema_model = nn.Sequential(
            nn.Linear(2, 2, bias=False),
            nn.Linear(2, 1, bias=False),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1.0)
            for parameter in ema_model.parameters():
                parameter.fill_(2.0)
        saved_model = model.state_dict()
        saved_model['0.weight'] = np.full((2, 2), 7.0, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / 'ckpt_step_8.pth'
            torch.save({
                'step': 8,
                'model_state_dict': saved_model,
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainer_state': self._trainer_state(8),
            }, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "not Tensor"):
                load_latest_checkpoint(
                    model,
                    ema_model,
                    optimizer,
                    str(temporary_dir),
                    resume_checkpoint_step=8,
                    rank=0,
                    world_size=1,
                    grad_mix=1,
                    batches_per_epoch=3,
                    fallback_seed=0,
                    sampler_contract=self._sampler_contract(),
                )

        for parameter in model.parameters():
            torch.testing.assert_close(
                parameter,
                torch.ones_like(parameter),
                rtol=0,
                atol=0,
            )

    def test_meta_checkpoint_tensor_is_rejected_before_commit(self):
        model = nn.Sequential(
            nn.Linear(2, 2, bias=False),
            nn.Linear(2, 1, bias=False),
        )
        ema_model = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1.0)
            for parameter in ema_model.parameters():
                parameter.fill_(2.0)
        original_model = copy.deepcopy(model.state_dict())
        original_ema = copy.deepcopy(ema_model.state_dict())
        original_optimizer = copy.deepcopy(optimizer.state_dict())
        saved_model = copy.deepcopy(model.state_dict())
        saved_model['0.weight'].fill_(7.0)
        saved_model['1.weight'] = torch.empty_like(
            saved_model['1.weight'],
            device='meta',
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / 'ckpt_step_8.pth'
            torch.save({
                'step': 8,
                'model_state_dict': saved_model,
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainer_state': self._trainer_state(8),
            }, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "materialized CPU tensor"):
                load_latest_checkpoint(
                    model,
                    ema_model,
                    optimizer,
                    str(temporary_dir),
                    resume_checkpoint_step=8,
                    rank=0,
                    world_size=1,
                    grad_mix=1,
                    batches_per_epoch=3,
                    fallback_seed=0,
                    sampler_contract=self._sampler_contract(),
                )

        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original_model[key], rtol=0, atol=0)
        for key, value in ema_model.state_dict().items():
            torch.testing.assert_close(value, original_ema[key], rtol=0, atol=0)
        current_optimizer = optimizer.state_dict()
        self.assertEqual(
            current_optimizer['param_groups'],
            original_optimizer['param_groups'],
        )
        self.assertEqual(
            set(current_optimizer['state']),
            set(original_optimizer['state']),
        )
        for parameter_id, state in current_optimizer['state'].items():
            self.assertEqual(
                set(state),
                set(original_optimizer['state'][parameter_id]),
            )
            for key, value in state.items():
                expected = original_optimizer['state'][parameter_id][key]
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                else:
                    self.assertEqual(value, expected)

    def test_bad_optimizer_moment_shape_is_rejected_before_commit(self):
        model = nn.Linear(2, 1, bias=False)
        ema_model = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        parameter_id = next(iter(optimizer_state['state']))
        optimizer_state['state'][parameter_id]['exp_avg'] = torch.zeros(3)
        with torch.no_grad():
            model.weight.fill_(1.0)
            ema_model.weight.fill_(2.0)
        saved_model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            saved_model.weight.fill_(7.0)

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / 'ckpt_step_8.pth'
            torch.save({
                'step': 8,
                'model_state_dict': saved_model.state_dict(),
                'ema_model_state_dict': saved_model.state_dict(),
                'optimizer_state_dict': optimizer_state,
                'trainer_state': self._trainer_state(8),
            }, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "exp_avg shape"):
                load_latest_checkpoint(
                    model,
                    ema_model,
                    optimizer,
                    str(temporary_dir),
                    resume_checkpoint_step=8,
                    rank=0,
                    world_size=1,
                    grad_mix=1,
                    batches_per_epoch=3,
                    fallback_seed=0,
                    sampler_contract=self._sampler_contract(),
                )

        torch.testing.assert_close(
            model.weight,
            torch.ones_like(model.weight),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            ema_model.weight,
            torch.full_like(ema_model.weight, 2.0),
            rtol=0,
            atol=0,
        )

    def test_incompatible_optimizer_group_is_rejected_before_commit(self):
        model = nn.Linear(2, 1, bias=False)
        ema_model = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        optimizer_state['param_groups'][0]['amsgrad'] = True

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / 'ckpt_step_8.pth'
            torch.save({
                'step': 8,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer_state,
                'trainer_state': self._trainer_state(8),
            }, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "group 0 options differ"):
                load_latest_checkpoint(
                    model,
                    ema_model,
                    optimizer,
                    str(temporary_dir),
                    resume_checkpoint_step=8,
                    rank=0,
                    world_size=1,
                    grad_mix=1,
                    batches_per_epoch=3,
                    fallback_seed=0,
                    sampler_contract=self._sampler_contract(),
                )

    def test_missing_optimizer_parameter_state_is_rejected_before_commit(self):
        model = nn.Linear(2, 2, bias=True)
        ema_model = nn.Linear(2, 2, bias=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        removed_parameter_id = next(iter(optimizer_state['state']))
        del optimizer_state['state'][removed_parameter_id]

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / 'ckpt_step_8.pth'
            torch.save({
                'step': 8,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer_state,
                'trainer_state': self._trainer_state(8),
            }, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "coverage differs"):
                load_latest_checkpoint(
                    model,
                    ema_model,
                    optimizer,
                    str(temporary_dir),
                    resume_checkpoint_step=8,
                    rank=0,
                    world_size=1,
                    grad_mix=1,
                    batches_per_epoch=3,
                    fallback_seed=0,
                    sampler_contract=self._sampler_contract(),
                )

    def test_complete_checkpoint_commits_all_training_states(self):
        model = nn.Linear(2, 1, bias=False)
        ema_model = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved_model = nn.Linear(2, 1, bias=False)
        saved_ema = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
            ema_model.weight.fill_(2.0)
            saved_model.weight.fill_(7.0)
            saved_ema.weight.fill_(8.0)

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / 'ckpt_step_8.pth'
            torch.save({
                'step': 8,
                'model_state_dict': saved_model.state_dict(),
                'ema_model_state_dict': saved_ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainer_state': self._trainer_state(8),
            }, checkpoint_path)
            resume_state = load_latest_checkpoint(
                model,
                ema_model,
                optimizer,
                str(temporary_dir),
                resume_checkpoint_step=8,
                rank=0,
                world_size=1,
                grad_mix=1,
                batches_per_epoch=3,
                fallback_seed=0,
                sampler_contract=self._sampler_contract(),
            )

        self.assertEqual(resume_state['next_step'], 9)
        torch.testing.assert_close(
            model.weight,
            torch.full_like(model.weight, 7.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            ema_model.weight,
            torch.full_like(ema_model.weight, 8.0),
            rtol=0,
            atol=0,
        )

    def test_empty_optimizer_state_is_rejected(self):
        model = nn.Linear(2, 1, bias=False)
        ema_model = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / 'ckpt_step_8.pth'
            torch.save({
                'step': 8,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainer_state': self._trainer_state(8),
            }, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "state mapping is empty"):
                load_latest_checkpoint(
                    model,
                    ema_model,
                    optimizer,
                    str(temporary_dir),
                    resume_checkpoint_step=8,
                    rank=0,
                    world_size=1,
                    grad_mix=1,
                    batches_per_epoch=3,
                    fallback_seed=0,
                    sampler_contract=self._sampler_contract(),
                )

    def test_synset_mapping_uses_all_root_directories(self):
        root_classes = [f"n{index:08d}" for index in range(1000)]
        observed = root_classes[1:]
        mapping = _build_latent_class_to_idx(observed, root_classes)
        self.assertEqual(mapping[root_classes[1]], 1)
        self.assertEqual(mapping[root_classes[-1]], 999)

        with self.assertRaisesRegex(ValueError, "exactly 1000"):
            _build_latent_class_to_idx(observed, root_classes[1:])

    def test_numeric_mapping_does_not_shift_when_a_class_is_absent(self):
        mapping = _build_latent_class_to_idx(
            ["0000", "0002"],
            ["0000", "0001", "0002"],
        )
        self.assertEqual(mapping, {"0000": 0, "0002": 2})


if __name__ == '__main__':
    unittest.main()
