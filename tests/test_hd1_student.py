"""Synthetic-only HD1 checks; no teacher or rollout is launched here."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def make_payload(walker_index):
    limbs, rows = 3, 6
    obs_mask = torch.tensor([False, False, True]) if walker_index % 2 == 0 else torch.tensor([False, True, True])
    act_mask = torch.repeat_interleave(obs_mask, 2)
    obs = torch.randn(rows, limbs * 17)
    target = torch.zeros(rows, limbs * 2)
    target[:, act_mask] = 0.
    return {
        'obs': obs,
        'act': target.clone(),
        'act_mean': target,
        'episode_id': torch.arange(rows) // 2,
        'context': torch.full((limbs * 35,), walker_index / 100.),
        'obs_padding_mask': obs_mask,
        'act_padding_mask': act_mask,
        'adjacency_matrix': torch.eye(limbs),
        'teacher_obs_rms': {'mean': torch.zeros(limbs * 17), 'var': torch.ones(limbs * 17), 'count': torch.tensor(1.)},
        'manifest': {
            'context_version': 1,
            'proprio_features_per_limb': 17,
            'max_limbs': limbs,
            'normalization': 'teacher_obs_rms_then_selected',
            'walker_id': f'w{walker_index:02d}',
        },
    }


@unittest.skipUnless(torch.cuda.is_available(), 'HD1 student requires CUDA')
class Hd1StudentTests(unittest.TestCase):
    def test_mixed_context_masks_and_prescribed_split(self):
        from tools.hd1_student import (
            _bank, _make_model, _masked_row_mean_mse, configure_hd1,
            context_sha256, export_hd1, load_hd1_records, sha256_file, train_hd1,
        )

        torch.manual_seed(5)
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
            temporary = Path(temporary)
            dataset = temporary / 'dataset'
            dataset.mkdir()
            train = []
            for index in range(18):
                walker_id = f'w{index:02d}'
                with (dataset / f'{walker_id}.pkl').open('wb') as handle:
                    pickle.dump(make_payload(index), handle)
                train.append({'walker_id': walker_id, 'family': 'even' if index % 2 == 0 else 'odd'})
            selection = {
                'train': train,
                'train_smoke': [{'walker_id': f'w{index:02d}', 'family': 'even'} for index in range(0, 12, 2)],
                'ood_smoke': [{'walker_id': f'w{index:02d}', 'family': 'odd'} for index in range(1, 12, 2)],
            }
            selection_path = temporary / 'selection.json'
            selection_path.write_text(json.dumps(selection))
            records = load_hd1_records(dataset, selection)
            self.assertEqual(len(records), 18)
            self.assertEqual(records[0]['episode_ids'], [0, 1, 2])
            self.assertEqual(int(records[0]['train_rows'].sum()), 4)
            self.assertEqual(int(records[0]['validation_rows'].sum()), 2)
            bank = _bank(records, torch.device('cuda'))
            self.assertFalse(torch.equal(bank['context'][0], bank['context'][1]))
            self.assertTrue(bank['act_mask'][0, 4:].all())
            prediction = torch.tensor([[1., 1., 0., 0., 99., 99.], [2., 0., 99., 99., 99., 99.]], device='cuda')
            target = torch.zeros_like(prediction)
            masks = torch.stack((bank['act_mask'][0], bank['act_mask'][1]))
            self.assertAlmostEqual(_masked_row_mean_mse(prediction, target, masks).item(), 1.25)
            configure_hd1(str(ROOT / 'configs' / 'ft.yaml'), 1409)
            model = _make_model(records).eval()
            with torch.no_grad():
                obs = torch.cat((records[0]['obs'][:1], records[1]['obs'][:1])).cuda()
                context, obs_mask, act_mask, adjacency = (
                    bank['context'][:2], bank['obs_mask'][:2], bank['act_mask'][:2], bank['adjacency'][:2],
                )
                model.mu_net.generate_params(context, obs_mask)
                model({
                    'proprioceptive': obs, 'context': context, 'obs_padding_mask': obs_mask,
                    'act_padding_mask': act_mask, 'adjacency_matrix': adjacency,
                }, compute_val=False)
                batched = model.action_mu.clone()
                oracle = []
                for index in range(2):
                    model.mu_net.generate_params(context[index:index + 1], obs_mask[index:index + 1])
                    model({
                        'proprioceptive': obs[index:index + 1], 'context': context[index:index + 1],
                        'obs_padding_mask': obs_mask[index:index + 1], 'act_padding_mask': act_mask[index:index + 1],
                        'adjacency_matrix': adjacency[index:index + 1],
                    }, compute_val=False)
                    oracle.append(model.action_mu.clone())
                self.assertTrue(torch.allclose(batched, torch.cat(oracle), rtol=1e-5, atol=1e-6))
                model.mu_net.generate_params(context[:1], obs_mask[:1])
                model({
                    'proprioceptive': obs[:1], 'context': context[:1], 'obs_padding_mask': obs_mask[:1],
                    'act_padding_mask': act_mask[:1], 'adjacency_matrix': adjacency[:1],
                }, compute_val=False)
                native = model.action_mu.clone()
                model.mu_net.generate_params(context[1:2], obs_mask[1:2])
                model({
                    'proprioceptive': obs[:1], 'context': context[1:2], 'obs_padding_mask': obs_mask[1:2],
                    'act_padding_mask': act_mask[1:2], 'adjacency_matrix': adjacency[1:2],
                }, compute_val=False)
                self.assertFalse(torch.allclose(native, model.action_mu, rtol=1e-5, atol=1e-6))
            metrics = train_hd1(dataset, selection_path, temporary / 'student', epochs=1, batch_size=64)
            self.assertEqual(metrics['HD1_STUDENT_TRAIN'], 'PASS')
            self.assertEqual(metrics['HD1_VALIDATION_FINITE'], 'PASS')
            self.assertEqual(metrics['train_transitions'], 72)
            self.assertEqual(metrics['valid_transitions'], 36)
            self.assertEqual(len(metrics['final_validation_per_morphology_mse']), 18)
            export_dir = temporary / 'exports'
            exports = export_hd1(
                temporary / 'student' / 'hd1_student_checkpoint.pt', [dataset],
                selection_path, export_dir,
            )
            self.assertEqual(len(exports), 12)
            stats = json.loads((export_dir / 'hd1_export_stats.json').read_text())
            self.assertEqual(stats['student_checkpoint_sha256'], sha256_file(temporary / 'student' / 'hd1_student_checkpoint.pt'))
            by_walker = {row['walker_id']: row for row in stats['exports']}
            self.assertFalse(torch.equal(records[0]['act_mask'], records[1]['act_mask']))
            for walker_id in ('w00', 'w01'):
                with (dataset / f'{walker_id}.pkl').open('rb') as handle:
                    payload = pickle.load(handle)
                row = by_walker[walker_id]
                self.assertEqual(row['context_sha256'], context_sha256(payload['context']))
                self.assertEqual(row['policy_sha256'], sha256_file(row['path']))
                policy = torch.jit.load(row['path']).cuda().eval()
                output = policy(torch.zeros(1, payload['obs'].shape[1], device='cuda'))
                self.assertTrue(torch.equal(
                    output[0, payload['act_padding_mask'].cuda()],
                    torch.zeros(int(payload['act_padding_mask'].sum()), device='cuda'),
                ))


if __name__ == '__main__':
    unittest.main()
