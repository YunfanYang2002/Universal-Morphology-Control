"""Read-only HD2C checkpoint audit and per-morphology TorchScript export."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metamorph.algos.distill.distill import _hd0_export_torchscript  # noqa: E402
from tools.hd1_student import (  # noqa: E402
    _load_walker_dataset_path,
    _make_model,
    context_sha256,
    read_selection,
    same_teacher_rms,
    sha256_file,
)
from tools.hd2_student import configure_hd2  # noqa: E402
from tools.hd2c_protocol import OOD6  # noqa: E402
from tools.run_hyperdistill_hd1 import CONFIG_SHA, TEACHER_SHA  # noqa: E402


CHECKPOINT_EPOCHS = (30, 60, 90, 120, 150)
STEPS_PER_EPOCH = 1563
SAMPLES_PER_EPOCH = 8_000_000


def _checkpoint_path(checkpoints_dir, epoch):
    return Path(checkpoints_dir) / f'checkpoint_{epoch:03d}.pt'


def _finite_state_dict(state_dict, label):
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f'{label} mu_net must be a nonempty state_dict')
    for key, value in state_dict.items():
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f'{label} mu_net has a non-finite or non-tensor value at {key!r}')


def _validate_teacher_rms(rms, label):
    if not isinstance(rms, dict) or set(rms) != {'mean', 'var', 'count'}:
        raise ValueError(f'{label} teacher_obs_rms must contain exactly mean, var, and count')
    mean, var, count = (torch.as_tensor(rms[key]) for key in ('mean', 'var', 'count'))
    if mean.shape != (204,) or var.shape != (204,) or count.numel() != 1:
        raise ValueError(f'{label} teacher_obs_rms shape differs from frozen 204-column mapper')
    if not (torch.isfinite(mean).all() and torch.isfinite(var).all() and torch.isfinite(count).all()):
        raise ValueError(f'{label} teacher_obs_rms contains non-finite values')
    if (var < 0).any() or count.item() <= 0:
        raise ValueError(f'{label} teacher_obs_rms has invalid variance or count')


def _assert_model_matches_state(model, state_dict, label):
    loaded = model.mu_net.state_dict()
    if loaded.keys() != state_dict.keys():
        raise AssertionError(f'{label} model state keys differ after strict load')
    for key in loaded:
        if loaded[key].dtype != state_dict[key].dtype or not torch.equal(loaded[key].detach().cpu(), state_dict[key].detach().cpu()):
            raise AssertionError(f'{label} model parameter differs from loaded checkpoint: {key}')


def _validate_torchscript_policy(path, record):
    policy = torch.jit.load(str(path), map_location='cuda').eval()
    with torch.no_grad():
        output = policy(torch.zeros(1, record['obs'].shape[1], dtype=torch.float32, device='cuda'))
    if output.shape != (1, record['act_mask'].numel()) or not torch.isfinite(output).all():
        raise ValueError(f'HD2C reloaded TorchScript policy is non-finite or mis-shaped: {path}')
    mask = record['act_mask'].to(device='cuda').reshape(1, -1)
    if not torch.equal(output.masked_select(mask), torch.zeros(int(mask.sum().item()), device='cuda')):
        raise ValueError(f'HD2C reloaded TorchScript policy emits nonzero padded actions: {path}')
    for name, buffer in policy.named_buffers():
        if buffer.is_floating_point() and not torch.isfinite(buffer).all():
            raise ValueError(f'HD2C reloaded TorchScript policy has a non-finite buffer {name!r}: {path}')
    return {'torchscript_reload': 'PASS', 'output_finite': 'PASS', 'padded_actions_zero': 'PASS'}


def _validate_checkpoint_metadata(state, epoch, label):
    required = {
        'mu_net', 'optimizer', 'seed', 'completed_epoch',
        'cumulative_optimizer_steps', 'cumulative_samples_seen',
    }
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError(f'{label} checkpoint schema differs from frozen HD2B schema')
    if state['seed'] != 1409 or state['completed_epoch'] != epoch:
        raise ValueError(f'{label} checkpoint seed or completed_epoch is invalid')
    if state['cumulative_optimizer_steps'] != epoch * STEPS_PER_EPOCH:
        raise ValueError(f'{label} cumulative optimizer-step count is invalid')
    if state['cumulative_samples_seen'] != epoch * SAMPLES_PER_EPOCH:
        raise ValueError(f'{label} cumulative sample count is invalid')
    _finite_state_dict(state['mu_net'], label)


def load_hd2c_ood_records(context_dirs, selection_path):
    """Load exactly six OOD converter pickles, each from one named directory."""
    selection = read_selection(selection_path)
    if selection['train'] or selection['train_smoke']:
        raise ValueError('HD2C selection must have empty train and train_smoke lists')
    entries = selection['ood_smoke']
    ids = [entry.get('walker_id') if isinstance(entry, dict) else None for entry in entries]
    if ids != list(OOD6):
        raise ValueError('HD2C selection OOD walker order differs from the frozen OOD6 tuple')
    if any(not isinstance(entry.get('family'), str) or not entry['family'] for entry in entries):
        raise ValueError('HD2C OOD entries require explicit family provenance')
    directories = [Path(path) for path in context_dirs]
    if not directories:
        raise ValueError('HD2C requires at least one context directory')
    records = []
    for entry in entries:
        walker_id = entry['walker_id']
        matches = [directory / f'{walker_id}.pkl' for directory in directories if (directory / f'{walker_id}.pkl').is_file()]
        if len(matches) != 1:
            raise ValueError(f'HD2C OOD context for {walker_id!r} must appear in exactly one context directory, found {len(matches)}')
        records.append(_load_walker_dataset_path(matches[0], entry))
    first_shape = (records[0]['obs'].shape[1], records[0]['context'].numel(), records[0]['act_mask'].numel())
    if any((record['obs'].shape[1], record['context'].numel(), record['act_mask'].numel()) != first_shape for record in records[1:]):
        raise ValueError('HD2C OOD contexts do not share a padded HNMLP input shape')
    reference_rms = records[0]['teacher_obs_rms']
    _validate_teacher_rms(reference_rms, f'HD2C OOD context {records[0]["walker_id"]!r}')
    for record in records[1:]:
        if not same_teacher_rms(reference_rms, record['teacher_obs_rms']):
            raise ValueError('HD2C OOD context teacher RMS differs across walkers')
        _validate_teacher_rms(record['teacher_obs_rms'], f'HD2C OOD context {record["walker_id"]!r}')
    for record in records:
        provenance = record['manifest'].get('teacher_provenance')
        if not isinstance(provenance, dict) or provenance.get('sha256', {}).get('config') != CONFIG_SHA or provenance.get('sha256', {}).get('checkpoint') != TEACHER_SHA:
            raise ValueError(f'HD2C context {record["walker_id"]!r} lacks pinned teacher provenance')
    return selection, records, reference_rms


def validate_hd2b_checkpoint(path, epoch, model, expected_sha256=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'HD2C required checkpoint is missing: {path}')
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f'HD2C checkpoint SHA-256 mismatch for epoch {epoch}')
    state = torch.load(path, map_location='cpu')
    _validate_checkpoint_metadata(state, epoch, f'HD2C epoch {epoch}')
    model.mu_net.load_state_dict(state['mu_net'], strict=True)
    _assert_model_matches_state(model, state['mu_net'], f'HD2C epoch {epoch}')
    return state, {
        'epoch': epoch,
        'path': str(path.resolve()),
        'sha256': actual_sha256,
        'cumulative_optimizer_steps': state['cumulative_optimizer_steps'],
        'cumulative_samples_seen': state['cumulative_samples_seen'],
        'strict_model_load': 'PASS',
        'model_state_exact': 'PASS',
        'state_finite': 'PASS',
    }


def validate_hd2b_checkpoints(checkpoints_dir, model):
    """Strictly load all frozen HD2B checkpoints without constructing an optimizer."""
    audits = []
    for epoch in CHECKPOINT_EPOCHS:
        _, audit = validate_hd2b_checkpoint(_checkpoint_path(checkpoints_dir, epoch), epoch, model)
        audits.append(audit)
    return audits


def _hd2c_dummy_model():
    configure_hd2(str(ROOT / 'configs' / 'ft.yaml'), 1409)
    record = {
        'manifest': {'max_limbs': 12},
        'obs': torch.zeros(1, 204),
        'context': torch.zeros(420),
        'act_mask': torch.zeros(24, dtype=torch.bool),
    }
    return _make_model([record])


def preflight_hd2c(checkpoint_dir, output_dir):
    """Hash, finite-check, and strict-load every HD2B checkpoint before rollout."""
    if not torch.cuda.is_available():
        raise RuntimeError('HD2C checkpoint preflight requires CUDA for the official model')
    model = _hd2c_dummy_model()
    audits = validate_hd2b_checkpoints(checkpoint_dir, model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        'stage': 'HD2C_PREFLIGHT',
        'checkpoint_epochs': list(CHECKPOINT_EPOCHS),
        'checkpoints': audits,
        'checkpoint_schema_validation': 'PASS',
        'checkpoint_state_finite': 'PASS',
        'strict_model_load': 'PASS',
        'optimizer_loaded': False,
    }
    (output_dir / 'checkpoint_hashes.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report


def export_hd2c(checkpoint_path, epoch, expected_sha256, context_dirs, selection_path, output_dir, cfg_path=None):
    """Strict-load one preflight-pinned HD2B checkpoint and export six OOD policies.

    Only state dictionaries are strictly loaded into the official HNMLP.  The
    checkpoint optimizer payload is inspected for schema only and never loaded.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('HD2C TorchScript export requires CUDA')
    configure_hd2(str(cfg_path or ROOT / 'configs' / 'ft.yaml'), 1409)
    selection, records, teacher_obs_rms = load_hd2c_ood_records(context_dirs, selection_path)
    model = _make_model(records)
    if epoch not in CHECKPOINT_EPOCHS:
        raise ValueError(f'HD2C export epoch must be one of {CHECKPOINT_EPOCHS}')
    final_state, checkpoint_audit = validate_hd2b_checkpoint(checkpoint_path, epoch, model, expected_sha256)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    exported = []
    for record in records:
        output_path = output_dir / f'{record["walker_id"]}.ts'
        _hd0_export_torchscript(
            model, record['context'].reshape(1, -1).cuda(), record['obs_mask'].reshape(1, -1).cuda(),
            record['act_mask'].reshape(1, -1).cuda(), str(output_path), torch.device('cuda'), record['obs'].shape[1],
        )
        policy_validation = _validate_torchscript_policy(output_path, record)
        exported.append({
            'walker_id': record['walker_id'],
            'family': record['family'],
            'path': str(output_path.resolve()),
            'policy_sha256': sha256_file(output_path),
            'context_sha256': context_sha256(record['context']),
            'context_source': record['dataset_path'],
            'episode_ids': record['episode_ids'],
            **policy_validation,
        })
    _assert_model_matches_state(model, final_state['mu_net'], 'HD2C post-export')
    stats = {
        'stage': 'HD2C',
        'selection': str(Path(selection_path).resolve()),
        'checkpoint': checkpoint_audit['path'],
        'student_checkpoint_sha256': checkpoint_audit['sha256'],
        'checkpoint_audit': checkpoint_audit,
        'checkpoint_epoch': epoch,
        'checkpoint_schema_validation': 'PASS',
        'checkpoint_state_finite': 'PASS',
        'optimizer_loaded': False,
        'ood_training_samples': 0,
        'teacher_obs_rms_bound': 'PASS',
        'policy_count': len(exported),
        'exports': exported,
    }
    (output_dir / 'hd1_export_stats.json').write_text(json.dumps(stats, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    preflight = subparsers.add_parser('preflight', help='Validate and hash epochs 30/60/90/120/150 before any teacher evaluation.')
    preflight.add_argument('--checkpoint-dir', required=True)
    preflight.add_argument('--output', required=True)
    export = subparsers.add_parser('export', help='Export one SHA-pinned checkpoint after OOD contexts are available.')
    export.add_argument('--checkpoint', required=True)
    export.add_argument('--epoch', required=True, type=int)
    export.add_argument('--expected-sha256', required=True)
    export.add_argument('--contexts', required=True, nargs='+')
    export.add_argument('--selection', required=True)
    export.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == 'preflight':
        report = preflight_hd2c(args.checkpoint_dir, args.output)
        print('HD2C_PREFLIGHT=' + json.dumps(report, sort_keys=True))
    else:
        stats = export_hd2c(args.checkpoint, args.epoch, args.expected_sha256, args.contexts, args.selection, args.output)
        print('HD2C_EXPORT=' + json.dumps(stats, sort_keys=True))


if __name__ == '__main__':
    main()
