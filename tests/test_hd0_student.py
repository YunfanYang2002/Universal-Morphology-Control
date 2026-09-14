"""Synthetic-only checks for the offline HD0 HNMLP entrypoint."""

import pickle
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), 'HD0 student requires CUDA')
class Hd0StudentTests(unittest.TestCase):
    def test_one_epoch_writes_frozen_student_and_metrics(self):
        from metamorph.algos.distill.distill import distill_hd0_policy
        from metamorph.config import cfg

        torch.manual_seed(3)
        limbs, rows = 3, 20
        obs_mask = torch.tensor([False, False, True])
        act_mask = torch.tensor([False, False, False, False, True, True])
        payload = {
            'obs': torch.randn(rows, limbs * 17),
            'act': torch.zeros(rows, limbs * 2),
            'act_mean': torch.zeros(rows, limbs * 2),
            'episode_id': torch.arange(rows) // 2,
            'context': torch.randn(limbs * 35),
            'obs_padding_mask': obs_mask,
            'act_padding_mask': act_mask,
            'adjacency_matrix': torch.eye(limbs),
            'teacher_obs_rms': {'mean': torch.zeros(limbs * 17), 'var': torch.ones(limbs * 17), 'count': torch.tensor(1.)},
            'manifest': {
                'context_version': 1,
                'proprio_features_per_limb': 17,
                'max_limbs': limbs,
                'normalization': 'teacher_obs_rms_then_selected',
            },
        }
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            temp = Path(temp)
            dataset, output = temp / 'walker.pkl', temp / 'student'
            with dataset.open('wb') as handle:
                pickle.dump(payload, handle)
            cfg.merge_from_file(ROOT / 'configs' / 'ft.yaml')
            cfg.merge_from_list(['MODEL.TYPE', 'hnmlp', 'DISTILL.VALUE_NET', False, 'ENV.KEYS_TO_KEEP', []])
            metrics = distill_hd0_policy(str(dataset), str(output), epochs=1, batch_size=8, seed=1409)
            self.assertEqual(metrics['finite_checks'], 'PASS')
            self.assertEqual(metrics['STATIC_CONTEXT_FREEZE'], 'PASS')
            self.assertTrue((output / 'hd0_student_checkpoint.pt').is_file())
            self.assertTrue((output / 'hd0_student_metrics.json').is_file())
            student = torch.jit.load(str(output / 'hd0_student.ts')).cuda().eval()
            result = student(torch.zeros(1, limbs * 17, device='cuda'))
            self.assertTrue(torch.equal(result[0, act_mask.cuda()], torch.zeros(2, device='cuda')))


if __name__ == '__main__':
    unittest.main()
