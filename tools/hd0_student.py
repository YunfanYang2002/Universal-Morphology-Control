"""HD0 one-walker student overfit entrypoint."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metamorph.algos.distill.distill import distill_hd0_policy
from metamorph.config import cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Overfit one HNMLP student to exported HD0 teacher data.',
    )
    parser.add_argument('--dataset', required=True, help='Per-walker exported pickle path.')
    parser.add_argument('--walker', required=True, help='Training walker ID; episodes, not walkers, are held out.')
    parser.add_argument('--output', required=True, help='Output directory for the HD0 artifacts.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1409)
    parser.add_argument('--cfg', default=str(ROOT / 'configs' / 'ft.yaml'), help='Base training config.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError('--epochs must be positive')
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError('--batch-size must be positive')
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list([
        'RNG_SEED', args.seed,
        'MODEL.TYPE', 'hnmlp',
        'MODEL.MLP.LAYER_NUM', 2,
        'DISTILL.VALUE_NET', False,
        'DISTILL.LOSS_TYPE', 'KL',
        'DISTILL.KL_TARGET', 'act_mean',
        'DISTILL.IMITATION_TARGET', 'act_mean',
        'DISTILL.BALANCED_LOSS', True,
        'ENV.KEYS_TO_KEEP', [],
    ])
    metrics = distill_hd0_policy(
        os.path.abspath(args.dataset), os.path.abspath(args.output), args.epochs,
        args.batch_size, args.seed, args.walker,
    )
    print('HD0_METRICS=' + str(metrics))


if __name__ == '__main__':
    main()
