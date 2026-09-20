"""Synthetic-only HD2C checkpoint adapter checks; no training or backpropagation."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def context_payload(index):
    limbs, rows = 12, 6
    obs_mask = torch.tensor([False, False] + [True] * 10) if index % 2 == 0 else torch.tensor([False] + [True] * 11)
    act_mask = torch.repeat_interleave(obs_mask, 2)
    return {
        'obs': torch.zeros(rows, limbs * 17),
        'act': torch.zeros(rows, limbs * 2),
        'act_mean': torch.zeros(rows, limbs * 2),
        'episode_id': torch.arange(rows) // 2,
        'context': torch.full((limbs * 35,), index / 100.),
        'obs_padding_mask': obs_mask,
        'act_padding_mask': act_mask,
        'adjacency_matrix': torch.eye(limbs),
        'teacher_obs_rms': {'mean': torch.zeros(limbs * 17), 'var': torch.ones(limbs * 17), 'count': torch.tensor(1.)},
        'manifest': {
            'context_version': 1,
            'proprio_features_per_limb': 17,
            'max_limbs': limbs,
            'normalization': 'teacher_obs_rms_then_selected',
            'walker_id': f'ood{index}',
            'teacher_provenance': {'sha256': {'config': 'CONFIG_SHA_PLACEHOLDER', 'checkpoint': 'TEACHER_SHA_PLACEHOLDER'}},
        },
    }


@unittest.skipUnless(torch.cuda.is_available(), 'HD2C requires CUDA for the official model')
class Hd2cExportTests(unittest.TestCase):
    def test_preflight_and_pinned_ood_export(self):
        from tools.hd2c_export import (
            CHECKPOINT_EPOCHS, OOD6, SAMPLES_PER_EPOCH, STEPS_PER_EPOCH, CONFIG_SHA, TEACHER_SHA,
            _hd2c_dummy_model, context_sha256, export_hd2c, preflight_hd2c, validate_hd2b_checkpoint,
        )
        from tools.hd1_student import sha256_file

        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
            temporary = Path(temporary)
            checkpoints = temporary / 'checkpoints'
            checkpoints.mkdir()
            model = _hd2c_dummy_model()
            state_dict = {key: value.detach().cpu().clone() for key, value in model.mu_net.state_dict().items()}
            for epoch in CHECKPOINT_EPOCHS:
                torch.save({
                    'mu_net': state_dict,
                    'optimizer': {'must_not_be_loaded': epoch},
                    'seed': 1409,
                    'completed_epoch': epoch,
                    'cumulative_optimizer_steps': epoch * STEPS_PER_EPOCH,
                    'cumulative_samples_seen': epoch * SAMPLES_PER_EPOCH,
                }, checkpoints / f'checkpoint_{epoch:03d}.pt')
            contexts = temporary / 'contexts'
            contexts.mkdir()
            entries = []
            for index in range(6):
                payload = context_payload(index)
                payload['manifest']['walker_id'] = OOD6[index]
                payload['manifest']['teacher_provenance']['sha256'] = {'config': CONFIG_SHA, 'checkpoint': TEACHER_SHA}
                with (contexts / f'{OOD6[index]}.pkl').open('wb') as handle:
                    pickle.dump(payload, handle)
                entries.append({'walker_id': OOD6[index], 'family': 'floor' if index < 2 else ('mvt' if index < 4 else 'vt')})
            selection = {'train': [], 'train_smoke': [], 'ood_smoke': entries}
            selection_path = temporary / 'selection.json'
            selection_path.write_text(json.dumps(selection))
            preflight = preflight_hd2c(checkpoints, temporary / 'preflight')
            self.assertEqual([row['epoch'] for row in preflight['checkpoints']], list(CHECKPOINT_EPOCHS))
            self.assertTrue((temporary / 'preflight' / 'checkpoint_hashes.json').is_file())
            final = preflight['checkpoints'][-1]
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                validate_hd2b_checkpoint(final['path'], 150, model, '0' * 64)
            with self.assertRaises(FileNotFoundError):
                validate_hd2b_checkpoint(temporary / 'missing.pt', 150, model)
            invalid_metadata = torch.load(final['path'], map_location='cpu')
            invalid_metadata['cumulative_optimizer_steps'] -= 1
            invalid_metadata_path = temporary / 'invalid_metadata.pt'
            torch.save(invalid_metadata, invalid_metadata_path)
            with self.assertRaisesRegex(ValueError, 'optimizer-step count'):
                validate_hd2b_checkpoint(invalid_metadata_path, 150, model)
            invalid_finite = torch.load(final['path'], map_location='cpu')
            first_key = next(iter(invalid_finite['mu_net']))
            invalid_finite['mu_net'][first_key] = invalid_finite['mu_net'][first_key].clone()
            invalid_finite['mu_net'][first_key].reshape(-1)[0] = float('inf')
            invalid_finite_path = temporary / 'invalid_finite.pt'
            torch.save(invalid_finite, invalid_finite_path)
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                validate_hd2b_checkpoint(invalid_finite_path, 150, model)
            invalid_shape = torch.load(final['path'], map_location='cpu')
            invalid_shape['mu_net']['unexpected.weight'] = torch.zeros(1)
            invalid_shape_path = temporary / 'invalid_shape.pt'
            torch.save(invalid_shape, invalid_shape_path)
            with self.assertRaises(RuntimeError):
                validate_hd2b_checkpoint(invalid_shape_path, 150, model)
            stats = export_hd2c(
                final['path'], 150, final['sha256'], [contexts], selection_path, temporary / 'exports',
            )
            self.assertFalse(stats['optimizer_loaded'])
            self.assertEqual(stats['student_checkpoint_sha256'], sha256_file(final['path']))
            self.assertEqual(len(stats['exports']), 6)
            self.assertEqual(stats['policy_count'], 6)
            by_walker = {row['walker_id']: row for row in stats['exports']}
            with (contexts / f'{OOD6[0]}.pkl').open('rb') as handle:
                payload = pickle.load(handle)
            self.assertEqual(by_walker[OOD6[0]]['context_sha256'], context_sha256(payload['context']))
            self.assertEqual(by_walker[OOD6[0]]['policy_sha256'], sha256_file(by_walker[OOD6[0]]['path']))
            self.assertEqual(by_walker[OOD6[0]]['torchscript_reload'], 'PASS')
            policy = torch.jit.load(by_walker[OOD6[0]]['path']).cuda().eval()
            output = policy(torch.zeros(1, payload['obs'].shape[1], device='cuda'))
            self.assertTrue(torch.equal(
                output[0, payload['act_padding_mask'].cuda()],
                torch.zeros(int(payload['act_padding_mask'].sum()), device='cuda'),
            ))


if __name__ == '__main__':
    unittest.main()
