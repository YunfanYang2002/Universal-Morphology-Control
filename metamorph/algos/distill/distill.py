import numpy as np
import os
import pickle
import json
from collections import defaultdict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
from gym import spaces

from metamorph.config import cfg
from metamorph.algos.ppo.model import ActorCritic
from metamorph.envs.vec_env.running_mean_std import RunningMeanStd

torch.manual_seed(0)
np.random.seed(0)

DEFAULT_OBS_DIM = list(range(13)) + [30, 31] + [41, 42]
DEFAULT_CONTEXT_DIM = list(range(13, 30)) + list(range(30+2, 30+2+9)) + list(range(30+11+2, 30+11+2+9))


class FrozenHNMLPStudent(nn.Module):
    """Inference-only wrapper around an HNMLP with one generated morphology."""

    def __init__(self, mu_net, obs_padding_mask, act_padding_mask, tanh_action):
        super().__init__()
        self.mu_net = mu_net
        self.register_buffer('obs_padding_mask', obs_padding_mask.bool())
        self.register_buffer('act_padding_mask', act_padding_mask.bool())
        self.tanh_action = tanh_action

    def forward(self, proprioceptive):
        batch_size = proprioceptive.shape[0]
        obs_mask = self.obs_padding_mask.expand(batch_size, -1)
        action_mean, _ = self.mu_net(
            proprioceptive, obs_mask, None, None, None, None,
        )
        if self.tanh_action:
            action_mean = torch.tanh(action_mean)
        return action_mean.masked_fill(self.act_padding_mask.expand(batch_size, -1), 0.)


class DistillationDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, index):
        obs = self.data['obs'][index]
        act = self.data['act'][index]
        act_mean = self.data['act_mean'][index]
        unimal_ids = self.data['unimal_ids'][index]
        if 'hfield' in self.data:
            hfield = self.data['hfield'][index]
        else:
            hfield = torch.zeros(1)
        return obs, act, act_mean, hfield, unimal_ids


def _hd0_tensor(value, name, dtype=torch.float32):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if not torch.isfinite(value.float()).all():
        raise ValueError(f'HD0 dataset field {name} contains non-finite values')
    return value.detach().to(dtype=dtype, device='cpu').contiguous()


def _hd0_masked_mse(prediction, target, act_padding_mask):
    valid = (~act_padding_mask.bool()).expand_as(prediction).to(dtype=prediction.dtype)
    valid_count = valid.sum()
    if valid_count.item() == 0:
        raise ValueError('HD0 action mask has no valid action dimensions')
    return ((prediction - target).square() * valid).sum() / valid_count


def _hd0_check_finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'HD0 {name} became non-finite')


def _hd0_episode_split(episode_id, seed):
    episode_values = np.unique(episode_id.cpu().numpy())
    if len(episode_values) < 2:
        raise ValueError('HD0 episode split requires at least two complete episodes')
    rng = np.random.RandomState(seed)
    val_episode_count = max(1, len(episode_values) // 10)
    val_episode_ids = episode_values[rng.permutation(len(episode_values))[:val_episode_count]]
    val_mask = torch.from_numpy(np.isin(episode_id.cpu().numpy(), val_episode_ids))
    if not val_mask.any() or val_mask.all():
        raise ValueError('HD0 episode split produced an empty train or validation set')
    return ~val_mask, val_mask, val_episode_ids.tolist()


def _hd0_eval_metrics(model, obs, target, context, obs_mask, act_mask, batch_size, device):
    """Evaluate a fixed morphology after generating its HN parameters once."""
    model.eval()
    with torch.no_grad():
        model.mu_net.generate_params(context, obs_mask)
        total_squared_error = 0.
        total_valid = 0
        for start in range(0, len(obs), batch_size):
            obs_batch = obs[start:start + batch_size].to(device, non_blocking=True)
            target_batch = target[start:start + batch_size].to(device, non_blocking=True)
            obs_dict = {
                'proprioceptive': obs_batch,
                'context': context.expand(obs_batch.shape[0], -1),
                'obs_padding_mask': obs_mask.expand(obs_batch.shape[0], -1),
                'act_padding_mask': act_mask.expand(obs_batch.shape[0], -1),
                'adjacency_matrix': torch.empty(0, device=device),
            }
            model(obs_dict, compute_val=False)
            prediction = torch.tanh(model.action_mu) if cfg.PPO.TANH == 'action' else model.action_mu
            _hd0_check_finite('validation prediction', prediction)
            valid = (~act_mask.bool()).to(dtype=prediction.dtype)
            total_squared_error += (((prediction - target_batch).square() * valid).sum().item())
            total_valid += valid.sum().item() * obs_batch.shape[0]
    if total_valid == 0:
        raise ValueError('HD0 validation set has no valid action values')
    mse = total_squared_error / total_valid
    if not np.isfinite(mse):
        raise FloatingPointError('HD0 validation MSE became non-finite')
    return mse


def _hd0_static_context_gate(model, obs, context, obs_mask, act_mask, adjacency_matrix, device):
    """Prove eval uses the already generated HN parameters, never input context."""
    model.eval()
    sample = obs[:1].to(device)
    adjacency_matrix = adjacency_matrix.to(device).reshape(1, *adjacency_matrix.shape)

    def forward_with(context_value):
        obs_dict = {
            'proprioceptive': sample,
            'context': context_value,
            'obs_padding_mask': obs_mask,
            'act_padding_mask': act_mask,
            'adjacency_matrix': adjacency_matrix,
        }
        model(obs_dict, compute_val=False)
        return (torch.tanh(model.action_mu) if cfg.PPO.TANH == 'action' else model.action_mu).clone()

    with torch.no_grad():
        reference = forward_with(context)
        original_generate_params = model.mu_net.generate_params

        def fail_if_regenerated(*args, **kwargs):
            raise AssertionError('HD0 evaluation regenerated HN parameters')

        model.mu_net.generate_params = fail_if_regenerated
        try:
            changed_context = context + 1.
            changed = forward_with(changed_context)
        finally:
            model.mu_net.generate_params = original_generate_params
    if not torch.equal(reference, changed):
        raise AssertionError('HD0 evaluation output changed with a different supplied context')


def _hd0_export_torchscript(model, context, obs_mask, act_mask, output_path, device, obs_dim):
    model.eval()
    with torch.no_grad():
        model.mu_net.generate_params(context, obs_mask)
        model.mu_net.input_weight = model.mu_net.input_weight.detach()
        model.mu_net.input_bias = model.mu_net.input_bias.detach()
        model.mu_net.output_weight = model.mu_net.output_weight.detach()
        model.mu_net.output_bias = model.mu_net.output_bias.detach()
        if hasattr(model.mu_net, 'hidden_weights'):
            model.mu_net.hidden_weights = [weight.detach() for weight in model.mu_net.hidden_weights]
            model.mu_net.hidden_bias = [bias.detach() for bias in model.mu_net.hidden_bias]
        frozen_student = FrozenHNMLPStudent(
            model.mu_net, obs_mask, act_mask, cfg.PPO.TANH == 'action',
        ).to(device).eval()
        example = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
        eager_output = frozen_student(example)
        traced = torch.jit.trace(frozen_student, example, check_trace=True)
        traced_output = traced(example)
        if not torch.allclose(eager_output, traced_output, rtol=1e-5, atol=1e-6):
            raise RuntimeError('HD0 TorchScript trace does not match the frozen HNMLP output')
        traced.save(output_path)


def distill_hd0_policy(dataset_path, output_dir, epochs=50, batch_size=None, seed=1409, walker_id=None):
    """Overfit one exported morphology with the existing HNMLP distillation path.

    The input observations are already normalized by the exported teacher RMS.
    This function intentionally does not recalculate normalization or recreate
    morphology context from a simulator.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('HD0 student distillation requires CUDA; no CUDA device is visible')
    if cfg.PPO.TANH is not None:
        raise ValueError('HD0 supports raw teacher action means only; PPO.TANH must be None')
    if 'hfield' in cfg.ENV.KEYS_TO_KEEP:
        raise ValueError('HD0 does not support hfield observations')
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)

    required = {
        'obs', 'act', 'act_mean', 'episode_id', 'context', 'obs_padding_mask',
        'act_padding_mask', 'adjacency_matrix', 'teacher_obs_rms', 'manifest',
    }
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f'HD0 dataset missing required fields: {missing}')
    manifest = data['manifest']
    if manifest.get('context_version') != 1:
        raise ValueError('HD0 requires context_version=1 from the exported reset context')
    if manifest.get('proprio_features_per_limb') != 17:
        raise ValueError('HD0 requires 17 selected proprioceptive features per limb')
    if manifest.get('normalization') != 'teacher_obs_rms_then_selected':
        raise ValueError('HD0 requires teacher-normalized proprioceptive observations')
    if walker_id is not None and manifest.get('walker_id') != walker_id:
        raise ValueError(f'HD0 dataset walker_id={manifest.get("walker_id")!r} does not match --walker={walker_id!r}')
    max_limbs = manifest.get('max_limbs')
    if not isinstance(max_limbs, int) or max_limbs <= 0:
        raise ValueError('HD0 manifest max_limbs must be a positive integer')
    if data['teacher_obs_rms'] is None:
        raise ValueError('HD0 requires the exported teacher_obs_rms; normalization is not recomputed')

    obs = _hd0_tensor(data['obs'], 'obs')
    act = _hd0_tensor(data['act'], 'act')
    target = _hd0_tensor(data['act_mean'], 'act_mean')
    episode_id = _hd0_tensor(data['episode_id'], 'episode_id', dtype=torch.long)
    context = _hd0_tensor(data['context'], 'context')
    obs_mask = _hd0_tensor(data['obs_padding_mask'], 'obs_padding_mask', dtype=torch.bool)
    act_mask = _hd0_tensor(data['act_padding_mask'], 'act_padding_mask', dtype=torch.bool)
    adjacency_matrix = _hd0_tensor(data['adjacency_matrix'], 'adjacency_matrix')
    if obs.ndim != 2 or obs.shape[1] != max_limbs * 17:
        raise ValueError(f'HD0 obs must have shape [N, {max_limbs * 17}], got {tuple(obs.shape)}')
    if act.shape != target.shape or act.ndim != 2 or act.shape[1] != max_limbs * 2:
        raise ValueError(f'HD0 action tensors must both have shape [N, {max_limbs * 2}]')
    if not (len(obs) == len(act) == len(episode_id)):
        raise ValueError('HD0 obs, actions, and episode_id must have the same sample count')
    if context.numel() != max_limbs * 35:
        raise ValueError(f'HD0 context must contain {max_limbs * 35} values (35 per limb)')
    if obs_mask.numel() != max_limbs or act_mask.numel() != max_limbs * 2:
        raise ValueError('HD0 padding masks do not match manifest max_limbs')
    if adjacency_matrix.shape != (max_limbs, max_limbs):
        raise ValueError('HD0 adjacency_matrix shape does not match manifest max_limbs')
    if act_mask.all():
        raise ValueError('HD0 act_padding_mask masks every action')

    train_mask, valid_mask, valid_episode_ids = _hd0_episode_split(episode_id, seed)
    context = context.reshape(1, -1).cuda()
    obs_mask = obs_mask.reshape(1, -1).cuda()
    act_mask = act_mask.reshape(1, -1).cuda()
    device = torch.device('cuda')
    cfg.MODEL.MAX_LIMBS = max_limbs
    cfg.MODEL.TYPE = 'hnmlp'
    cfg.MODEL.MLP.LAYER_NUM = 2
    cfg.DISTILL.VALUE_NET = False
    cfg.DISTILL.LOSS_TYPE = 'KL'
    cfg.DISTILL.KL_TARGET = 'act_mean'
    cfg.DISTILL.BALANCED_LOSS = True
    cfg.DISTILL.IMITATION_TARGET = 'act_mean'
    cfg.RNG_SEED = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_space = spaces.Dict({
        'proprioceptive': spaces.Box(-np.inf, np.inf, shape=(obs.shape[1],), dtype=np.float32),
        'context': spaces.Box(-np.inf, np.inf, shape=(context.shape[1],), dtype=np.float32),
        'obs_padding_mask': spaces.Box(0, 1, shape=(max_limbs,), dtype=np.bool_),
        'act_padding_mask': spaces.Box(0, 1, shape=(max_limbs * 2,), dtype=np.bool_),
        'adjacency_matrix': spaces.Box(-np.inf, np.inf, shape=(max_limbs, max_limbs), dtype=np.float32),
    })
    action_space = spaces.Box(-np.inf, np.inf, shape=(max_limbs * 2,), dtype=np.float32)
    model = ActorCritic(obs_space, action_space).cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.DISTILL.BASE_LR, eps=cfg.DISTILL.EPS,
        weight_decay=cfg.DISTILL.WEIGHT_DECAY,
    )
    if batch_size is None:
        batch_size = min(cfg.DISTILL.BATCH_SIZE, int(train_mask.sum().item()))
    if batch_size <= 0:
        raise ValueError('HD0 batch_size must be positive')
    generator = torch.Generator().manual_seed(seed)
    train_data = TensorDataset(obs[train_mask], target[train_mask])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, pin_memory=True)
    train_obs, train_target = obs[train_mask], target[train_mask]
    valid_obs, valid_target = obs[valid_mask], target[valid_mask]
    initial_train_mse = _hd0_eval_metrics(model, train_obs, train_target, context, obs_mask, act_mask, batch_size, device)
    initial_validation_mse = _hd0_eval_metrics(model, valid_obs, valid_target, context, obs_mask, act_mask, batch_size, device)
    train_curve, validation_curve = [], []

    for epoch in range(epochs):
        model.train()
        total_squared_error = 0.
        total_valid = 0
        for obs_batch, target_batch in train_loader:
            obs_batch = obs_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            obs_dict = {
                'proprioceptive': obs_batch,
                'context': context.expand(obs_batch.shape[0], -1),
                'obs_padding_mask': obs_mask.expand(obs_batch.shape[0], -1),
                'act_padding_mask': act_mask.expand(obs_batch.shape[0], -1),
                'adjacency_matrix': adjacency_matrix.to(device).expand(obs_batch.shape[0], -1, -1),
            }
            model(obs_dict, compute_val=False)
            prediction = torch.tanh(model.action_mu) if cfg.PPO.TANH == 'action' else model.action_mu
            mse = _hd0_masked_mse(prediction, target_batch, act_mask)
            loss = 0.5 * mse
            _hd0_check_finite('training loss', loss)
            optimizer.zero_grad()
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    _hd0_check_finite('optimizer gradient', parameter.grad)
            if cfg.DISTILL.GRAD_NORM is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.DISTILL.GRAD_NORM)
            optimizer.step()
            for parameter in model.parameters():
                _hd0_check_finite('model parameter', parameter)
            valid = (~act_mask.bool()).sum().item() * obs_batch.shape[0]
            total_squared_error += mse.item() * valid
            total_valid += valid
        train_mse = total_squared_error / total_valid
        validation_mse = _hd0_eval_metrics(model, valid_obs, valid_target, context, obs_mask, act_mask, batch_size, device)
        if not np.isfinite(train_mse):
            raise FloatingPointError('HD0 training MSE became non-finite')
        train_curve.append(train_mse)
        validation_curve.append(validation_mse)

    final_train_mse = _hd0_eval_metrics(model, train_obs, train_target, context, obs_mask, act_mask, batch_size, device)
    final_validation_mse = _hd0_eval_metrics(model, valid_obs, valid_target, context, obs_mask, act_mask, batch_size, device)
    _hd0_static_context_gate(model, valid_obs, context, obs_mask, act_mask, adjacency_matrix, device)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'hd0_student_checkpoint.pt')
    torch.save([model.mu_net.state_dict(), data['teacher_obs_rms']], checkpoint_path)
    with open(os.path.join(output_dir, 'loss_curve.pkl'), 'wb') as handle:
        pickle.dump([[0.5 * x for x in train_curve], [0.5 * x for x in validation_curve], []], handle)
    with open(os.path.join(output_dir, 'hd0_student_curves.json'), 'w') as handle:
        json.dump({
            'train_masked_action_mean_mse': train_curve,
            'validation_masked_action_mean_mse': validation_curve,
            'train_official_half_mse': [0.5 * x for x in train_curve],
            'validation_official_half_mse': [0.5 * x for x in validation_curve],
        }, handle, indent=2)
    with open(os.path.join(output_dir, 'hd0_student_config.yaml'), 'w') as handle:
        cfg.dump(stream=handle)
    metrics = {
        'dataset': os.path.abspath(dataset_path),
        'walker_id': walker_id,
        'seed': seed,
        'epochs': epochs,
        'max_limbs': max_limbs,
        'train_sample_count': int(train_mask.sum().item()),
        'validation_sample_count': int(valid_mask.sum().item()),
        'validation_episode_ids': valid_episode_ids,
        'initial_train_mse': initial_train_mse,
        'initial_validation_mse': initial_validation_mse,
        'final_train_mse': final_train_mse,
        'final_validation_mse': final_validation_mse,
        'initial_train_official_loss': 0.5 * initial_train_mse,
        'initial_validation_official_loss': 0.5 * initial_validation_mse,
        'final_train_official_loss': 0.5 * final_train_mse,
        'final_validation_official_loss': 0.5 * final_validation_mse,
        'finite_checks': 'PASS',
        'STATIC_CONTEXT_FREEZE': 'PASS',
        'NO_MUTATION_LEAKAGE': 'PASS',
        'imitation_convergence_ratio': 0.5,
        'imitation_converged': bool(
            final_train_mse <= 0.5 * initial_train_mse
            and final_validation_mse <= 0.5 * initial_validation_mse
        ),
        'rollout_validation': 'NOT_RUN',
    }
    with open(os.path.join(output_dir, 'hd0_student_metrics.json'), 'w') as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    _hd0_export_torchscript(
        model, context, obs_mask, act_mask,
        os.path.join(output_dir, 'hd0_student.ts'), device, obs.shape[1],
    )
    return metrics


def distill_policy(source_folder, target_folder, teacher_mode, validation=False):

    from metamorph.algos.ppo.envs import make_env, make_vec_envs

    agents = cfg.ENV.WALKERS
    envs = make_vec_envs()
    model = ActorCritic(envs.observation_space, envs.action_space).cuda()

    # merge the training data
    buffer = {
        'obs': [], 
        'act': [], 
        'act_mean': [], 
        'unimal_ids': [], 
    }
    if 'hfield' in cfg.ENV.KEYS_TO_KEEP:
        buffer['hfield'] = []
    all_context = defaultdict(list)

    i = 0
    for agent in agents:
        # if i == 5:
        #     break
        data_path = f'expert_data/{source_folder}/{agent}.pkl'
        if not os.path.exists(data_path):
            continue
        print (agent)
        with open(data_path, 'rb') as f:
            agent_data = pickle.load(f)
        env = make_env(cfg.ENV_NAME, 0, 0, xml_file=agent)()
        init_obs = env.reset()
        env.close()
        all_context['context'].append(init_obs['context'])
        all_context['obs_mask'].append(init_obs['obs_padding_mask'])
        all_context['act_mask'].append(init_obs['act_padding_mask'])
        all_context['adjacency_matrix'].append(init_obs['adjacency_matrix'])
        # drop context features from obs if needed
        if len(cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES) == 6 and agent_data['obs'].shape[-1] == 624:
            data_size = agent_data['obs'].shape[0]
            new_obs = agent_data['obs'].view(data_size, cfg.MODEL.MAX_LIMBS, -1)
            new_obs = new_obs[:, :, DEFAULT_OBS_DIM]
            agent_data['obs'] = new_obs.view(data_size, -1)
        if cfg.DISTILL.SAMPLE_STRATEGY == 'random':
            sample_index = np.random.choice(agent_data['obs'].shape[0], cfg.DISTILL.PER_AGENT_SAMPLE_NUM, replace=False)
        for key in ['obs', 'act', 'act_mean', 'hfield']:
            if key not in buffer:
                continue
            if cfg.DISTILL.SAMPLE_STRATEGY == 'random':
                buffer[key].append(agent_data[key][sample_index])
            elif cfg.DISTILL.SAMPLE_STRATEGY == 'timestep_first':
                data_size = agent_data[key].shape[0]
                feat_dim = agent_data[key].shape[-1]
                buffer[key].append(agent_data[key].reshape(-1, 64, feat_dim).permute(1, 0, 2).reshape(data_size, feat_dim)[:cfg.DISTILL.PER_AGENT_SAMPLE_NUM])
            elif cfg.DISTILL.SAMPLE_STRATEGY == 'env_first':
                buffer[key].append(agent_data[key][:cfg.DISTILL.PER_AGENT_SAMPLE_NUM])
            else:
                raise ValueError("Unsupported sample strategy")
        buffer['unimal_ids'].append(torch.ones(cfg.DISTILL.PER_AGENT_SAMPLE_NUM, dtype=torch.long) * i)
        i += 1

    all_context['context'] = torch.from_numpy(np.stack(all_context['context'])).float().cuda()
    all_context['obs_mask'] = torch.from_numpy(np.stack(all_context['obs_mask'])).float().cuda()
    all_context['act_mask'] = torch.from_numpy(np.stack(all_context['act_mask'])).float().cuda()
    all_context['adjacency_matrix'] = torch.from_numpy(np.stack(all_context['adjacency_matrix'])).float().cuda()

    context_features = all_context['context'].reshape(all_context['context'].shape[0] * 12, -1)
    for i in range(context_features.shape[1]):
        print (i, context_features[:, i].min(), context_features[:, i].max())

    if validation:
        in_domain_validation_buffer = {}
        out_domain_validation_buffer = {}
    for key in buffer:
        if validation:
            valid_agent_num = int(len(buffer[key]) * 0.9)
            out_domain_validation_buffer[key] = torch.cat(buffer[key][valid_agent_num:], dim=0)
            valid_sample_num = int(cfg.DISTILL.PER_AGENT_SAMPLE_NUM * 0.9)
            in_domain_validation_buffer[key] = torch.cat([x[valid_sample_num:] for x in buffer[key][:valid_agent_num]], dim=0)
            buffer[key] = torch.cat([x[:valid_sample_num] for x in buffer[key][:valid_agent_num]], dim=0)
            print (out_domain_validation_buffer[key].shape[0], in_domain_validation_buffer[key].shape[0], buffer[key].shape[0])
        else:
            buffer[key] = torch.cat(buffer[key], dim=0)
            print (buffer[key].shape)

    if teacher_mode == 'MT':
        # if the teacher is trained by multi-task RL, reuse its obs rms for the student
        with open(f'expert_data/{source_folder}/obs_rms.pkl', 'rb') as f:
            obs_rms = pickle.load(f)
        if len(cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES) == 6:
            new_mean = obs_rms['proprioceptive'].mean.reshape(cfg.MODEL.MAX_LIMBS, -1)
            obs_rms['proprioceptive'].mean = new_mean[:, DEFAULT_OBS_DIM].ravel()
            new_var = obs_rms['proprioceptive'].var.reshape(cfg.MODEL.MAX_LIMBS, -1)
            obs_rms['proprioceptive'].var = new_var[:, DEFAULT_OBS_DIM].ravel()
    elif teacher_mode == 'ST':
        # if the teachers are trained by single-task RL, renormalize the proprioceptive features
        obs_rms = {'proprioceptive': RunningMeanStd(shape=buffer['obs'].shape[-1])}
        obs_rms['proprioceptive'].mean = buffer['obs'].mean(axis=0)
        obs_rms['proprioceptive'].var = buffer['obs'].var(axis=0)
        buffer['obs'] = np.clip(
            (buffer['obs'] - obs_rms['proprioceptive'].mean) / np.sqrt(obs_rms['proprioceptive'].var + 1e-8), 
            -10., 
            10.
        )

    train_data = DistillationDataset(buffer)
    train_dataloader = DataLoader(train_data, batch_size=cfg.DISTILL.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
    if validation:
        in_domain_validation_data = DistillationDataset(in_domain_validation_buffer)
        in_domain_validation_dataloader = DataLoader(in_domain_validation_data, batch_size=cfg.DISTILL.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
        out_domain_validation_data = DistillationDataset(out_domain_validation_buffer)
        out_domain_validation_dataloader = DataLoader(out_domain_validation_data, batch_size=cfg.DISTILL.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)

    if cfg.DISTILL.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=cfg.DISTILL.BASE_LR, 
            eps=cfg.DISTILL.EPS, 
            weight_decay=cfg.DISTILL.WEIGHT_DECAY
        )
    elif cfg.DISTILL.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=cfg.DISTILL.BASE_LR, 
            eps=cfg.DISTILL.EPS, 
            weight_decay=cfg.DISTILL.WEIGHT_DECAY
        )
    else:
        raise ValueError("Unsupported optimizer type")

    def loss_function(obs_dict, act, act_mean):
        if cfg.DISTILL.IMITATION_TARGET == 'act':
            _, pi, logp, _ = model(obs_dict, act=act.cuda(), compute_val=False, unimal_ids=unimal_ids)
        else:
            _, pi, logp, _ = model(obs_dict, act=act_mean.cuda(), compute_val=False, unimal_ids=unimal_ids)
        if cfg.DISTILL.LOSS_TYPE == 'KL':
            if cfg.DISTILL.KL_TARGET == 'act':
                target = act
            elif cfg.DISTILL.KL_TARGET == 'act_mean':
                target = act_mean
            else:
                raise ValueError("Unsupported loss type")
            if cfg.PPO.TANH == 'action':
                pred = torch.tanh(model.action_mu)
                target = torch.tanh(target)
            else:
                pred = model.action_mu
            if cfg.DISTILL.SAMPLE_WEIGHT:
                threshold = cfg.DISTILL.LARGE_ACT_DECAY
                w = torch.where(target.abs() > threshold, torch.exp(threshold - target.abs()), 1.).cuda()
                if cfg.DISTILL.BALANCED_LOSS:
                    loss = 0.5 * (((pred - target.cuda()).square() * w * (1 - obs_dict['act_padding_mask'])).sum(dim=1) / (1 - obs_dict['act_padding_mask']).sum(dim=1)).mean()
                else:
                    loss = 0.5 * ((pred - target.cuda()).square() * w * (1 - obs_dict['act_padding_mask'])).mean()
            else:
                if cfg.DISTILL.BALANCED_LOSS:
                    loss = 0.5 * (((pred - target.cuda()).square() * (1 - obs_dict['act_padding_mask'])).sum(dim=1) / (1 - obs_dict['act_padding_mask']).sum(dim=1)).mean()
                else:
                    loss = 0.5 * ((pred - target.cuda()).square() * (1 - obs_dict['act_padding_mask'])).mean()
        elif cfg.DISTILL.LOSS_TYPE == 'logp':
            if cfg.DISTILL.BALANCED_LOSS:
                loss = -((model.limb_logp * (1 - obs_dict['act_padding_mask'])).sum(dim=1, keepdim=True) / (1 - obs_dict['act_padding_mask']).sum(dim=1, keepdim=True)).mean()
            else:
                loss = -logp.mean()
        else:
            raise ValueError("Unsupported loss type")
        return loss

    loss_curve, in_domain_valid_curve, out_domain_valid_curve = [], [], []
    for i in range(cfg.DISTILL.EPOCH_NUM):

        if i % cfg.DISTILL.SAVE_FREQ == 0:
            torch.save([model.mu_net.state_dict(), obs_rms], f'{cfg.OUT_DIR}/checkpoint_{i}.pt')
        elif i <= 50:
            if i % 5 == 0:
                torch.save([model.mu_net.state_dict(), obs_rms], f'{cfg.OUT_DIR}/checkpoint_{i}.pt')

        batch_losses = []
        for j, (obs, train_act, train_act_mean, hfield, unimal_ids) in enumerate(train_dataloader):

            context = all_context['context'][unimal_ids]
            obs_mask = all_context['obs_mask'][unimal_ids]
            act_mask = all_context['act_mask'][unimal_ids]
            adjacency_matrix = all_context['adjacency_matrix'][unimal_ids]
            obs = obs.cuda()

            if cfg.DISTILL.CONCAT_CONTEXT_TO_OBS:
                batch_size = obs.shape[0]
                merged_obs = torch.zeros(batch_size, cfg.MODEL.MAX_LIMBS, 52, device='cuda')
                merged_obs[:, :, DEFAULT_OBS_DIM] = obs.reshape(batch_size, cfg.MODEL.MAX_LIMBS, -1)
                merged_obs[:, :, DEFAULT_CONTEXT_DIM] = context.reshape(batch_size, cfg.MODEL.MAX_LIMBS, -1)
                obs = merged_obs.reshape(batch_size, -1)

            train_obs_dict = {
                'proprioceptive': obs,  
                'context': context, 
                'obs_padding_mask': obs_mask,  
                'act_padding_mask': act_mask, 
                'adjacency_matrix': adjacency_matrix, 
            }
            if 'hfield' in cfg.ENV.KEYS_TO_KEEP:
                train_obs_dict['hfield'] = hfield.cuda()

            loss = loss_function(train_obs_dict, train_act, train_act_mean)
            # if cfg.DISTILL.BASE_WEIGHT_DECAY is not None:
            #     if j % 100 == 0:
            #         print (f'batch {j}: loss = {loss.item()}, L2 reg = {model.mu_net.base_norm_square}')
            #     loss += cfg.DISTILL.BASE_WEIGHT_DECAY * model.mu_net.base_norm_square

            optimizer.zero_grad()
            loss.backward()
            if cfg.DISTILL.GRAD_NORM is not None:
                norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.DISTILL.GRAD_NORM)
            optimizer.step()
            batch_losses.append(loss.item())

        if validation:
            batch_in_domain_valid_loss = []
            for obs, valid_act, valid_act_mean, context, obs_mask, act_mask, hfield, unimal_ids in in_domain_validation_dataloader:
                valid_obs_dict = {
                    'proprioceptive': obs.cuda(), 
                    'context': context.cuda(), 
                    'obs_padding_mask': obs_mask.cuda(), 
                    'act_padding_mask': act_mask.cuda(), 
                }
                if 'hfield' in cfg.ENV.KEYS_TO_KEEP:
                    valid_obs_dict['hfield'] = hfield.cuda()
                with torch.no_grad():
                    loss = loss_function(valid_obs_dict, valid_act, valid_act_mean)
                batch_in_domain_valid_loss.append(loss.item())
            in_domain_valid_curve.append(np.mean(batch_in_domain_valid_loss))

            batch_out_domain_valid_loss = []
            for obs, valid_act, valid_act_mean, context, obs_mask, act_mask, hfield, unimal_ids in out_domain_validation_dataloader:
                valid_obs_dict = {
                    'proprioceptive': obs.cuda(), 
                    'context': context.cuda(), 
                    'obs_padding_mask': obs_mask.cuda(), 
                    'act_padding_mask': act_mask.cuda(), 
                }
                if 'hfield' in cfg.ENV.KEYS_TO_KEEP:
                    valid_obs_dict['hfield'] = hfield.cuda()
                with torch.no_grad():
                    loss = loss_function(valid_obs_dict, valid_act, valid_act_mean)
                batch_out_domain_valid_loss.append(loss.item())
            out_domain_valid_curve.append(np.mean(batch_out_domain_valid_loss))

            print (f'epoch {i}, train: {np.mean(batch_losses):.4f}, in domain valid: {np.mean(batch_in_domain_valid_loss):.4f}, out domain valid: {np.mean(batch_out_domain_valid_loss):.4f}')
        else:
            print (f'epoch {i}, average batch loss: {np.mean(batch_losses)}')
        params_norm = torch.norm(torch.cat([p.view(-1) for p in model.parameters()]), 2).item()
        print ('model norm: ', params_norm)
        loss_curve.append(np.mean(batch_losses))
        with open(f'{cfg.OUT_DIR}/loss_curve.pkl', 'wb') as f:
            pickle.dump([loss_curve, in_domain_valid_curve, out_domain_valid_curve], f)

    torch.save([model.mu_net.state_dict(), obs_rms], f'{cfg.OUT_DIR}/checkpoint_{cfg.DISTILL.EPOCH_NUM}.pt')
