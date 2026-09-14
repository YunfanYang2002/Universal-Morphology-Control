"""HD1 multi-morphology HNMLP student training and frozen smoke export."""

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from gym import spaces
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metamorph.algos.distill.distill import (  # noqa: E402
    _hd0_check_finite,
    _hd0_export_torchscript,
    _hd0_tensor,
)
from metamorph.algos.ppo.model import ActorCritic  # noqa: E402
from metamorph.config import cfg  # noqa: E402


REQUIRED_DATASET_KEYS = {
    'obs', 'act', 'act_mean', 'episode_id', 'context', 'obs_padding_mask',
    'act_padding_mask', 'adjacency_matrix', 'teacher_obs_rms', 'manifest',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def context_sha256(context):
    value = context.detach().cpu().to(dtype=torch.float32).contiguous().view(-1).numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def same_teacher_rms(left, right):
    for key in ('mean', 'var', 'count'):
        if key not in left or key not in right or not torch.equal(torch.as_tensor(left[key]), torch.as_tensor(right[key])):
            return False
    return True


def configure_hd1(cfg_path, seed):
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list([
        'RNG_SEED', seed,
        'MODEL.TYPE', 'hnmlp',
        'MODEL.MLP.LAYER_NUM', 2,
        'DISTILL.VALUE_NET', False,
        'DISTILL.LOSS_TYPE', 'KL',
        'DISTILL.KL_TARGET', 'act_mean',
        'DISTILL.IMITATION_TARGET', 'act_mean',
        'DISTILL.BALANCED_LOSS', True,
        'ENV.KEYS_TO_KEEP', [],
    ])
    if cfg.PPO.TANH is not None:
        raise ValueError('HD1 supports raw teacher action means only; PPO.TANH must be None')


def read_selection(path):
    with open(path, 'r', encoding='utf-8') as handle:
        selection = json.load(handle)
    for key in ('train', 'train_smoke', 'ood_smoke'):
        if key not in selection or not isinstance(selection[key], list):
            raise ValueError(f'HD1 selection must contain a list named {key!r}')
    return selection


def _entry_id(entry, section):
    if not isinstance(entry, dict) or not isinstance(entry.get('walker_id'), str) or not entry['walker_id']:
        raise ValueError(f'HD1 {section} entries require a non-empty walker_id')
    return entry['walker_id']


def _masked_row_mean_mse(prediction, target, act_padding_mask):
    valid = (~act_padding_mask.bool()).to(dtype=prediction.dtype)
    valid_counts = valid.sum(dim=1)
    if (valid_counts == 0).any():
        raise ValueError('HD1 batch contains a walker with no valid action dimensions')
    return (((prediction - target).square() * valid).sum(dim=1) / valid_counts).mean()


def _load_walker_dataset_path(dataset_path, entry):
    walker_id = _entry_id(entry, 'train')
    family = entry.get('family')
    if not isinstance(family, str) or not family:
        raise ValueError(f'HD1 train entry {walker_id!r} requires a non-empty family')
    with dataset_path.open('rb') as handle:
        payload = pickle.load(handle)
    missing = sorted(REQUIRED_DATASET_KEYS.difference(payload))
    if missing:
        raise KeyError(f'HD1 dataset {walker_id!r} missing required fields: {missing}')
    manifest = payload['manifest']
    if manifest.get('walker_id') != walker_id:
        raise ValueError(f'HD1 dataset manifest walker_id does not match selection for {walker_id!r}')
    if (manifest.get('context_version'), manifest.get('proprio_features_per_limb'), manifest.get('normalization')) != (
        1, 17, 'teacher_obs_rms_then_selected',
    ):
        raise ValueError(f'HD1 dataset {walker_id!r} does not use the HD0 converter schema')
    max_limbs = manifest.get('max_limbs')
    if not isinstance(max_limbs, int) or max_limbs <= 0:
        raise ValueError(f'HD1 dataset {walker_id!r} has invalid max_limbs')
    obs = _hd0_tensor(payload['obs'], f'{walker_id}.obs')
    target = _hd0_tensor(payload['act_mean'], f'{walker_id}.act_mean')
    episode_id = _hd0_tensor(payload['episode_id'], f'{walker_id}.episode_id', dtype=torch.long)
    context = _hd0_tensor(payload['context'], f'{walker_id}.context').reshape(-1)
    obs_mask = _hd0_tensor(payload['obs_padding_mask'], f'{walker_id}.obs_padding_mask', dtype=torch.bool).reshape(-1)
    act_mask = _hd0_tensor(payload['act_padding_mask'], f'{walker_id}.act_padding_mask', dtype=torch.bool).reshape(-1)
    adjacency = _hd0_tensor(payload['adjacency_matrix'], f'{walker_id}.adjacency_matrix')
    if obs.ndim != 2 or obs.shape[1] != max_limbs * 17:
        raise ValueError(f'HD1 dataset {walker_id!r} obs shape is incompatible with max_limbs')
    if target.shape != (len(obs), max_limbs * 2) or len(episode_id) != len(obs):
        raise ValueError(f'HD1 dataset {walker_id!r} action/episode shape is invalid')
    if context.numel() != max_limbs * 35 or obs_mask.numel() != max_limbs or act_mask.numel() != max_limbs * 2:
        raise ValueError(f'HD1 dataset {walker_id!r} context or mask shape is invalid')
    if adjacency.shape != (max_limbs, max_limbs) or act_mask.all():
        raise ValueError(f'HD1 dataset {walker_id!r} adjacency or action mask is invalid')
    if (target[:, act_mask] != 0).any():
        raise ValueError(f'HD1 dataset {walker_id!r} has nonzero padded act_mean targets')
    episode_values = torch.unique(episode_id, sorted=True)
    if len(episode_values) != 3:
        raise ValueError(f'HD1 dataset {walker_id!r} must contain exactly three episode ids')
    train_rows = (episode_id == episode_values[0]) | (episode_id == episode_values[1])
    validation_rows = episode_id == episode_values[2]
    if not train_rows.any() or not validation_rows.any():
        raise ValueError(f'HD1 dataset {walker_id!r} has an empty prescribed episode split')
    if payload['teacher_obs_rms'] is None:
        raise ValueError(f'HD1 dataset {walker_id!r} lacks teacher_obs_rms')
    return {
        'walker_id': walker_id,
        'dataset_path': os.path.abspath(dataset_path),
        'family': family,
        'obs': obs,
        'target': target,
        'context': context,
        'obs_mask': obs_mask,
        'act_mask': act_mask,
        'adjacency_matrix': adjacency,
        'train_rows': train_rows,
        'validation_rows': validation_rows,
        'episode_ids': episode_values.tolist(),
        'teacher_obs_rms': payload['teacher_obs_rms'],
        'manifest': manifest,
    }


def _load_walker_dataset(dataset_dir, entry):
    return _load_walker_dataset_path(Path(dataset_dir) / f'{_entry_id(entry, "train")}.pkl', entry)


def load_hd1_records(dataset_dir, selection):
    train = selection['train']
    if len(train) != 18:
        raise ValueError(f'HD1 requires exactly 18 train walkers, got {len(train)}')
    walker_ids = [_entry_id(entry, 'train') for entry in train]
    if len(set(walker_ids)) != len(walker_ids):
        raise ValueError('HD1 train selection contains duplicate walker_ids')
    records = [_load_walker_dataset(dataset_dir, entry) for entry in train]
    reference = records[0]
    shape = (reference['obs'].shape[1], reference['context'].numel(), reference['act_mask'].numel())
    for record in records[1:]:
        current = (record['obs'].shape[1], record['context'].numel(), record['act_mask'].numel())
        if current != shape:
            raise ValueError('HD1 requires a common padded observation, context, and action shape')
    reference_rms = records[0]['teacher_obs_rms']
    if not same_teacher_rms(reference_rms, reference_rms):
        raise ValueError('HD1 reference teacher_obs_rms lacks mean, var, or count')
    for record in records[1:]:
        if not same_teacher_rms(reference_rms, record['teacher_obs_rms']):
            raise ValueError('HD1 teacher_obs_rms differs across selected train walkers')
    return records


def _make_model(records):
    first = records[0]
    max_limbs = first['manifest']['max_limbs']
    cfg.MODEL.MAX_LIMBS = max_limbs
    obs_space = spaces.Dict({
        'proprioceptive': spaces.Box(-np.inf, np.inf, shape=(first['obs'].shape[1],), dtype=np.float32),
        'context': spaces.Box(-np.inf, np.inf, shape=(first['context'].numel(),), dtype=np.float32),
        'obs_padding_mask': spaces.Box(0, 1, shape=(max_limbs,), dtype=np.bool_),
        'act_padding_mask': spaces.Box(0, 1, shape=(max_limbs * 2,), dtype=np.bool_),
        'adjacency_matrix': spaces.Box(-np.inf, np.inf, shape=(max_limbs, max_limbs), dtype=np.float32),
    })
    action_space = spaces.Box(-np.inf, np.inf, shape=(max_limbs * 2,), dtype=np.float32)
    return ActorCritic(obs_space, action_space).cuda()


def _bank(records, device):
    return {
        'context': torch.stack([record['context'] for record in records]).to(device),
        'obs_mask': torch.stack([record['obs_mask'] for record in records]).to(device),
        'act_mask': torch.stack([record['act_mask'] for record in records]).to(device),
        'adjacency': torch.stack([record['adjacency_matrix'] for record in records]).to(device),
    }


def _forward_train(model, obs, target, walker_index, bank):
    obs_dict = {
        'proprioceptive': obs,
        'context': bank['context'][walker_index],
        'obs_padding_mask': bank['obs_mask'][walker_index],
        'act_padding_mask': bank['act_mask'][walker_index],
        'adjacency_matrix': bank['adjacency'][walker_index],
    }
    model(obs_dict, compute_val=False)
    _hd0_check_finite('HD1 training prediction', model.action_mu)
    return _masked_row_mean_mse(model.action_mu, target, bank['act_mask'][walker_index])


def evaluate_hd1(model, records, split, batch_size, device):
    """Evaluation is grouped by morphology and generates each HN parameter set once."""
    model.eval()
    per_morphology = {}
    all_row_losses = []
    family_row_losses = {}
    with torch.no_grad():
        for record in records:
            rows = record[f'{split}_rows']
            context = record['context'].reshape(1, -1).to(device)
            obs_mask = record['obs_mask'].reshape(1, -1).to(device)
            act_mask = record['act_mask'].reshape(1, -1).to(device)
            adjacency = record['adjacency_matrix'].reshape(1, *record['adjacency_matrix'].shape).to(device)
            model.mu_net.generate_params(context, obs_mask)
            row_losses = []
            selected_obs, selected_target = record['obs'][rows], record['target'][rows]
            for start in range(0, len(selected_obs), batch_size):
                obs = selected_obs[start:start + batch_size].to(device, non_blocking=True)
                target = selected_target[start:start + batch_size].to(device, non_blocking=True)
                count = obs.shape[0]
                obs_dict = {
                    'proprioceptive': obs,
                    'context': context.expand(count, -1),
                    'obs_padding_mask': obs_mask.expand(count, -1),
                    'act_padding_mask': act_mask.expand(count, -1),
                    'adjacency_matrix': adjacency.expand(count, -1, -1),
                }
                model(obs_dict, compute_val=False)
                _hd0_check_finite('HD1 evaluation prediction', model.action_mu)
                valid = (~act_mask).to(dtype=model.action_mu.dtype)
                row_losses.extend(
                    (((model.action_mu - target).square() * valid).sum(dim=1) / valid.sum(dim=1)).cpu().tolist()
                )
            if not row_losses:
                raise ValueError(f'HD1 {split} set is empty for {record["walker_id"]!r}')
            per_morphology[record['walker_id']] = float(np.mean(row_losses))
            all_row_losses.extend(row_losses)
            family_row_losses.setdefault(record['family'], []).extend(row_losses)
    overall = float(np.mean(all_row_losses))
    per_family = {}
    for record in records:
        per_family.setdefault(record['family'], []).append(per_morphology[record['walker_id']])
    return overall, per_morphology, {
        family: float(np.mean(values)) for family, values in per_family.items()
    }, {
        family: float(np.mean(values)) for family, values in family_row_losses.items()
    }


def train_hd1(dataset_dir, selection_path, output_dir, epochs=50, batch_size=64, seed=1409, cfg_path=None):
    if not torch.cuda.is_available():
        raise RuntimeError('HD1 student distillation requires CUDA; no CUDA device is visible')
    if epochs <= 0 or batch_size <= 0:
        raise ValueError('HD1 epochs and batch_size must be positive')
    configure_hd1(cfg_path or str(ROOT / 'configs' / 'ft.yaml'), seed)
    selection = read_selection(selection_path)
    records = load_hd1_records(dataset_dir, selection)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda')
    model = _make_model(records)
    optimizer = optim.Adam(model.parameters(), lr=cfg.DISTILL.BASE_LR, eps=cfg.DISTILL.EPS, weight_decay=cfg.DISTILL.WEIGHT_DECAY)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'hd1_student_config.yaml'), 'w') as handle:
        cfg.dump(stream=handle)
    bank = _bank(records, device)
    train_obs = torch.cat([record['obs'][record['train_rows']] for record in records])
    train_target = torch.cat([record['target'][record['train_rows']] for record in records])
    train_walker_index = torch.cat([
        torch.full((int(record['train_rows'].sum().item()),), index, dtype=torch.long)
        for index, record in enumerate(records)
    ])
    loader = DataLoader(
        TensorDataset(train_obs, train_target, train_walker_index), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed), pin_memory=True,
    )
    initial_train_mse, _, _, _ = evaluate_hd1(model, records, 'train', batch_size, device)
    initial_validation_mse, initial_per_morphology, initial_per_family, initial_per_family_transition = evaluate_hd1(model, records, 'validation', batch_size, device)
    train_curve, validation_curve = [], []
    progress_path = os.path.join(output_dir, 'hd1_student_progress.json')
    for epoch in range(epochs):
        model.train()
        row_loss_sum = 0.
        row_count = 0
        for obs, target, walker_index in loader:
            obs, target, walker_index = obs.to(device), target.to(device), walker_index.to(device)
            mse = _forward_train(model, obs, target, walker_index, bank)
            loss = 0.5 * mse
            _hd0_check_finite('HD1 training loss', loss)
            optimizer.zero_grad()
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    _hd0_check_finite('HD1 optimizer gradient', parameter.grad)
            if cfg.DISTILL.GRAD_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.DISTILL.GRAD_NORM)
            optimizer.step()
            for parameter in model.parameters():
                _hd0_check_finite('HD1 model parameter', parameter)
            row_loss_sum += mse.item() * len(obs)
            row_count += len(obs)
        train_curve.append(row_loss_sum / row_count)
        validation_curve.append(evaluate_hd1(model, records, 'validation', batch_size, device)[0])
        progress = {
            'completed_epochs': epoch + 1,
            'train_masked_action_mean_mse': train_curve,
            'validation_masked_action_mean_mse': validation_curve,
            'train_official_half_mse': [0.5 * value for value in train_curve],
            'validation_official_half_mse': [0.5 * value for value in validation_curve],
            'finite_checks_so_far': 'PASS',
        }
        with open(progress_path, 'w') as handle:
            json.dump(progress, handle, indent=2, sort_keys=True)
        with open(os.path.join(output_dir, 'hd1_student_curves.json'), 'w') as handle:
            json.dump({
                'train_masked_action_mean_mse': train_curve,
                'validation_masked_action_mean_mse': validation_curve,
                'train_official_half_mse': [0.5 * value for value in train_curve],
                'validation_official_half_mse': [0.5 * value for value in validation_curve],
            }, handle, indent=2, sort_keys=True)
        print(f'HD1_EPOCH={epoch + 1}/{epochs} TRAIN_MSE={train_curve[-1]:.8g} VALID_MSE={validation_curve[-1]:.8g}', flush=True)
    final_train_mse, final_train_per_morphology, final_train_per_family, final_train_per_family_transition = evaluate_hd1(model, records, 'train', batch_size, device)
    final_validation_mse, final_per_morphology, final_per_family, final_per_family_transition = evaluate_hd1(model, records, 'validation', batch_size, device)
    checkpoint_path = os.path.join(output_dir, 'hd1_student_checkpoint.pt')
    torch.save([model.mu_net.state_dict(), records[0]['teacher_obs_rms']], checkpoint_path)
    with open(os.path.join(output_dir, 'hd1_student_config.yaml'), 'w') as handle:
        cfg.dump(stream=handle)
    curves = {
        'train_masked_action_mean_mse': train_curve,
        'validation_masked_action_mean_mse': validation_curve,
        'train_official_half_mse': [0.5 * value for value in train_curve],
        'validation_official_half_mse': [0.5 * value for value in validation_curve],
    }
    with open(os.path.join(output_dir, 'hd1_student_curves.json'), 'w') as handle:
        json.dump(curves, handle, indent=2, sort_keys=True)
    provenance = {
        'selection': os.path.abspath(selection_path),
        'dataset_dir': os.path.abspath(dataset_dir),
        'train_walkers': [{
            'walker_id': record['walker_id'], 'family': record['family'],
            'episode_ids': record['episode_ids'], 'train_episode_ids': record['episode_ids'][:2],
            'validation_episode_id': record['episode_ids'][2],
        } for record in records],
        'train_smoke_walkers': [_entry_id(entry, 'train_smoke') for entry in selection['train_smoke']],
        'ood_smoke_walkers': [_entry_id(entry, 'ood_smoke') for entry in selection['ood_smoke']],
        'ood_training_samples': 0,
        'train_transitions': int(len(train_obs)),
        'valid_transitions': int(sum(int(record['validation_rows'].sum().item()) for record in records)),
    }
    metrics = {
        **provenance,
        'seed': seed,
        'epochs': epochs,
        'batch_size': batch_size,
        'initial_train_mse': initial_train_mse,
        'initial_validation_mse': initial_validation_mse,
        'final_train_mse': final_train_mse,
        'final_validation_mse': final_validation_mse,
        'initial_train_official_loss': 0.5 * initial_train_mse,
        'initial_validation_official_loss': 0.5 * initial_validation_mse,
        'final_train_official_loss': 0.5 * final_train_mse,
        'final_validation_official_loss': 0.5 * final_validation_mse,
        'initial_validation_per_morphology_mse': initial_per_morphology,
        'initial_validation_per_family_mse': initial_per_family,
        'initial_validation_per_family_transition_weighted_mse': initial_per_family_transition,
        'final_train_per_morphology_mse': final_train_per_morphology,
        'final_train_per_family_mse': final_train_per_family,
        'final_train_per_family_transition_weighted_mse': final_train_per_family_transition,
        'final_validation_per_morphology_mse': final_per_morphology,
        'final_validation_per_family_mse': final_per_family,
        'final_validation_per_family_transition_weighted_mse': final_per_family_transition,
        'finite_checks': 'PASS',
        'HD1_STUDENT_TRAIN': 'PASS',
        'HD1_VALIDATION_FINITE': 'PASS',
        'rollout_validation': 'NOT_RUN',
    }
    with open(os.path.join(output_dir, 'hd1_student_metrics.json'), 'w') as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics


def export_hd1(checkpoint_path, context_dirs, selection_path, output_dir, cfg_path=None, seed=1409):
    if not torch.cuda.is_available():
        raise RuntimeError('HD1 TorchScript export requires CUDA; no CUDA device is visible')
    configure_hd1(cfg_path or str(ROOT / 'configs' / 'ft.yaml'), seed)
    selection = read_selection(selection_path)
    smoke_entries = selection['train_smoke'] + selection['ood_smoke']
    smoke_ids = [_entry_id(entry, 'smoke') for entry in smoke_entries]
    if len(smoke_ids) != 12 or len(set(smoke_ids)) != 12:
        raise ValueError('HD1 export requires exactly 12 distinct train_smoke + ood_smoke walkers')
    if not context_dirs:
        raise ValueError('HD1 export requires at least one --contexts directory')
    context_dirs = [Path(directory) for directory in context_dirs]
    records = []
    for entry in smoke_entries:
        walker_id = _entry_id(entry, 'smoke')
        matches = [directory / f'{walker_id}.pkl' for directory in context_dirs if (directory / f'{walker_id}.pkl').is_file()]
        if len(matches) != 1:
            raise ValueError(f'HD1 smoke context for {walker_id!r} must appear in exactly one --contexts directory, found {len(matches)}')
        records.append(_load_walker_dataset_path(matches[0], dict(entry, family=entry.get('family', 'smoke'))))
    first_shape = (records[0]['obs'].shape[1], records[0]['context'].numel(), records[0]['act_mask'].numel())
    if any((record['obs'].shape[1], record['context'].numel(), record['act_mask'].numel()) != first_shape for record in records[1:]):
        raise ValueError('HD1 smoke contexts do not share the trained padded shape')
    model = _make_model(records)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, (list, tuple)) or len(checkpoint) != 2:
        raise ValueError('HD1 checkpoint must be [mu_net.state_dict(), teacher_obs_rms]')
    for record in records:
        if not same_teacher_rms(checkpoint[1], record['teacher_obs_rms']):
            raise ValueError(f'HD1 smoke context RMS does not match checkpoint for {record["walker_id"]!r}')
    model.mu_net.load_state_dict(checkpoint[0], strict=True)
    os.makedirs(output_dir, exist_ok=True)
    exported = []
    for record in records:
        output_path = os.path.join(output_dir, f'{record["walker_id"]}.ts')
        _hd0_export_torchscript(
            model, record['context'].reshape(1, -1).cuda(), record['obs_mask'].reshape(1, -1).cuda(),
            record['act_mask'].reshape(1, -1).cuda(), output_path, torch.device('cuda'), record['obs'].shape[1],
        )
        exported.append({
            'walker_id': record['walker_id'], 'family': record['family'], 'path': os.path.abspath(output_path),
            'episode_ids': record['episode_ids'], 'context_source': record['dataset_path'],
            'context_sha256': context_sha256(record['context']), 'policy_sha256': sha256_file(output_path),
        })
    with open(os.path.join(output_dir, 'hd1_export_stats.json'), 'w') as handle:
        json.dump({
            'checkpoint': os.path.abspath(checkpoint_path),
            'student_checkpoint_sha256': sha256_file(checkpoint_path),
            'exports': exported, 'ood_training_samples': 0,
        }, handle, indent=2)
    return exported


def parse_args():
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        parser = argparse.ArgumentParser(description='Export frozen HD1 smoke policies.')
        parser.add_argument('--checkpoint', required=True)
        parser.add_argument('--contexts', required=True, nargs='+', help='Converter-schema pickle directories; each smoke walker must occur in exactly one.')
        parser.add_argument('--selection', required=True)
        parser.add_argument('--output', required=True)
        args = parser.parse_args(sys.argv[2:])
        args.command = 'export'
        return args
    parser = argparse.ArgumentParser(description='Train the fixed-budget HD1 multi-morphology student. Use `export --help` for frozen smoke export.')
    parser.add_argument('--dataset', required=True, help='Directory containing the 18 converter-schema train pickles.')
    parser.add_argument('--selection', required=True, help='Selection JSON with train/train_smoke/ood_smoke lists.')
    parser.add_argument('--output', required=True)
    train_args = sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1] == 'train' else sys.argv[1:]
    args = parser.parse_args(train_args)
    args.command = 'train'
    return args


def main():
    args = parse_args()
    if args.command == 'train':
        print('HD1_METRICS=' + json.dumps(train_hd1(
            args.dataset, args.selection, args.output, 50, 64, 1409,
        ), sort_keys=True))
    else:
        print('HD1_EXPORTS=' + json.dumps(export_hd1(
            args.checkpoint, args.contexts, args.selection, args.output, seed=1409,
        ), sort_keys=True))


if __name__ == '__main__':
    main()
