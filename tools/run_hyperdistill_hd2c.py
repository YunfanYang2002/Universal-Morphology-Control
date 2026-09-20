"""Evaluate five frozen HD2B checkpoints on HD1 OOD6; never train or select a checkpoint."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.hd1_protocol import sha256
from tools.hd2c_protocol import EPOCHS, OOD6, PROTOCOL, prepare_selection, summarize
from tools.run_hyperdistill_hd1 import CONFIG_SHA, TEACHER_SHA
from tools.run_hyperdistill_hd2b import HD2B_PROTOCOL

DEFAULT_RUN = ROOT / 'tmp/hyperdistill_hd2b_s1409_20260916T062120_921323Z'
MAPPER_SHA = '5aae11e48a05f57bca706085939d3ed47985f8c80e2e37299766e8d5f309917d'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify_hd2b(run, config, teacher):
    """Read completed-run provenance only; do not collect, resume, or repair anything."""
    if sha256(config) != CONFIG_SHA or sha256(teacher) != TEACHER_SHA:
        raise ValueError('Teacher/config differs from frozen HD0/HD1 pair')
    if sha256(run / 'teacher_config.yaml') != CONFIG_SHA:
        raise ValueError('HD2B teacher config differs from the frozen pair')
    if json.loads((run / 'frozen_protocol.json').read_text()) != HD2B_PROTOCOL:
        raise ValueError('HD2B protocol differs from completed paper-scale training')
    summary = json.loads((run / 'student/hd2b_training_summary.json').read_text())
    required = {'HD2_DROP_LAST': False, 'HD2_FULL_BATCH_SIZE': 5120,
                'HD2_FULL_BATCH_COUNT_PER_EPOCH': 1562, 'HD2_FINAL_BATCH_SIZE': 2560,
                'HD2_STEPS_PER_EPOCH': 1563, 'HD2_SAMPLES_PER_EPOCH': 8000000,
                'HD2B_RESUMABLE_TRAINING': 'PASS'}
    if any(summary.get(key) != value for key, value in required.items()):
        raise ValueError('HD2B completion summary does not match the frozen training contract')
    metrics = run / 'student/epoch_metrics.jsonl'
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    if not rows or rows[-1]['epoch'] != 150:
        raise ValueError('HD2B epoch metrics do not end at epoch 150')
    if any(row['optimizer_steps'] != 1563 or row['samples_seen_this_epoch'] != 8000000
           or row['last_batch_size'] != 2560 for row in rows):
        raise ValueError('HD2B epoch metrics violate the frozen sample/batch contract')
    return {'checkpoint_path': str(teacher), 'checkpoint_sha256': TEACHER_SHA,
            'config_path': str(config), 'config_sha256': CONFIG_SHA,
            'hd2b_config_sha256': sha256(run / 'teacher_config.yaml'),
            'hd2b_teacher_binding': 'HD2B launcher enforces the identical frozen HD0/HD1 hash pair; copied config verified',
            'hd2b_run': str(run), 'epoch_metrics_sha256': sha256(metrics)}


def package(output, manifest):
    """Only this invocation's artifacts enter the audit; HD2B root failure.txt is never read."""
    suffixes = {'.json', '.jsonl', '.yaml', '.txt', '.log', '.csv', '.md'}
    files = sorted(p for p in output.rglob('*') if p.is_file()
                   and p.name != 'manifest.json' and 'temp' not in p.relative_to(output).parts)
    # Nested native reference manifests are essential provenance.
    files += sorted(p for p in output.rglob('manifest.json') if p != output / 'manifest.json'
                    and 'temp' not in p.relative_to(output).parts)
    manifest['artifacts'] = [{'relative_path': p.relative_to(output).as_posix(),
                              'bytes': p.stat().st_size, 'sha256': sha256(p),
                              'included_in_package': p.suffix in suffixes} for p in files]
    write_json(output / 'manifest.json', manifest)
    archive = Path(str(output) + '_audit.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in [*files, output / 'manifest.json']:
            if path.suffix in suffixes:
                bundle.write(path, path.relative_to(output))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hd2b-run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--rmamorph-root', type=Path, default=Path.home() / 'Workspace/Code/rmamorph')
    args = parser.parse_args()
    os.chdir(ROOT)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / 'tmp' / f'hyperdistill_hd2c_ood6_{stamp}'
    for name in ('provenance', 'summary', 'logs', 'temp'):
        (output / name).mkdir(parents=True, exist_ok=False)
    for key in ('TMP', 'TEMP', 'TMPDIR'):
        os.environ[key] = str(output / 'temp')
    status = {key: 'NOT_RUN' for key in (
        'HD2C_OOD6_SELECTION_FROZEN', 'HD2C_CHECKPOINT_5', 'HD2C_EXPORT_REUSE',
        'HD2C_ROLLOUT_REUSE', 'HD2C_GATE_REUSE', 'HD2_RUNTIME_FINITE',
        'HD2_CROSS_MORPH_CONTROL', 'HD2_NORMALIZED_RETURN_GATE', 'HD2_FINAL')}
    status.update(HD2C_NO_TRAINING='PASS', OOD_TRAINING_SAMPLES=0)
    manifest = {'protocol': PROTOCOL, 'commands': [], 'stages': [], 'host': platform.node(),
                'python': sys.executable, 'python_version': sys.version,
                'conda_prefix': os.environ.get('CONDA_PREFIX'),
                'execution_kind': 'evaluation only; no optimizer calls',
                'source': {}, 'status': status}
    write_json(output / 'protocol.json', PROTOCOL)

    def stage(name, script, arguments, cwd=ROOT):
        command = [sys.executable, str(ROOT / 'tools' / script), *map(str, arguments)]
        manifest['active_stage'] = name
        manifest['commands'].append({'stage': name, 'argv': command, 'cwd': str(cwd)})
        print(f'STAGE={name}', flush=True)
        env = {**os.environ, 'PYTHONPATH': str(cwd), 'PYTHONUNBUFFERED': '1'}
        with (output / 'logs' / f'{name}.log').open('w', encoding='utf-8') as log:
            log.write(json.dumps(command) + '\n'); log.flush()
            with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
        manifest['stages'].append(name)
        manifest['active_stage'] = None

    code = 1
    try:
        run = args.hd2b_run.expanduser().resolve()
        if not run.is_relative_to((ROOT / 'tmp').resolve()) or not run.is_dir():
            raise FileNotFoundError(f'HD2B completed run must exist below project ./tmp: {run}')
        teacher_root = args.rmamorph_root.expanduser().resolve()
        manifest['environment_versions'] = {
            name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'gym', 'PyYAML')}
        manifest['implementation_hashes'] = {
            name: sha256(ROOT / 'tools' / name) for name in (
                'run_hyperdistill_hd2c.py', 'hd2c_export.py', 'hd2c_protocol.py',
                'hd0_teacher_export.py', 'hd1_protocol.py', 'hd1_student.py',
                'hd2_student.py', 'convert_rmamorph_teacher_to_hyperdistill.py')}
        config = teacher_root / 'output/metamorph_dr_matched_s1415_100m/config.yaml'
        teacher = config.with_name('Unimal-v0.pt')
        for label, repo in (('hyperdistill', ROOT), ('rmamorph', teacher_root)):
            manifest['source'][label] = {
                'root': str(repo),
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                'tracked_changes': subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'], cwd=repo, text=True)}
        teacher_binding = verify_hd2b(run, config, teacher)
        write_json(output / 'provenance/teacher_hashes.json', teacher_binding)
        shutil.copyfile(config, output / 'provenance/teacher_config.yaml')
        for name in ('epoch_metrics.jsonl', 'hd2b_training_summary.json'):
            destination = output / 'provenance/hd2b/student' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / 'student' / name, destination)
        pd_manifest = run / 'pd_inventory/provenance/mutation_manifest.json'
        prepare_selection(config, teacher_root / 'configs/morphadapt_metamorph_dr_formal_ood98.txt',
                          teacher_root / 'output/unimals_100/test', pd_manifest, output / 'provenance')
        status['HD2C_OOD6_SELECTION_FROZEN'] = 'PASS'
        stage('checkpoint_preflight', 'hd2c_export.py', ['preflight', '--checkpoint-dir', run / 'student/checkpoints',
                                                       '--output', output / 'provenance/checkpoints'])
        checkpoint_rows = json.loads((output / 'provenance/checkpoints/checkpoint_hashes.json').read_text())['checkpoints']
        if [row['epoch'] for row in checkpoint_rows] != list(EPOCHS):
            raise ValueError('Checkpoint preflight did not bind the exact five frozen epochs')
        status['HD2C_CHECKPOINT_5'] = 'PASS'
        selection = output / 'provenance/selection.json'
        common = ['--rmamorph-root', teacher_root, '--config', config, '--checkpoint', teacher,
                  '--hd1-selection', selection, '--hd1-set', 'ood_smoke', '--seed', 1409,
                  '--episodes', 3, '--max-steps', 3000]
        reference, contexts = output / 'teacher_reference', output / 'ood_context'
        stage('teacher_reference', 'hd0_teacher_export.py', [*common, '--output', reference], teacher_root)
        stage('ood_context', 'convert_rmamorph_teacher_to_hyperdistill.py',
              ['--source', reference, '--output', contexts, '--expected-walkers', json.dumps(list(OOD6))])
        columns = contexts / 'student_columns.json'
        if sha256(columns) != MAPPER_SHA or sha256(ROOT / 'tools/hd1_student_columns.json') != MAPPER_SHA:
            raise ValueError('OOD mapper differs from frozen HD1/HD2 204-column mapping')
        write_json(output / 'provenance/mapper.json', {'sha256': MAPPER_SHA, 'OOD_TRAINING_SAMPLES': 0})
        evaluations = {}
        for row in checkpoint_rows:
            epoch = row['epoch']
            policies = output / 'policies' / f'epoch_{epoch:03d}'
            destination = output / 'evaluation' / f'epoch_{epoch:03d}'
            stage(f'export_{epoch:03d}', 'hd2c_export.py',
                  ['export', '--checkpoint', row['path'], '--epoch', epoch, '--expected-sha256', row['sha256'],
                   '--contexts', contexts, '--selection', selection, '--output', policies])
            if sha256(row['path']) != row['sha256']:
                raise ValueError(f'Checkpoint changed after export: epoch {epoch}')
            stage(f'evaluation_{epoch:03d}', 'hd0_teacher_export.py',
                  [*common, '--teacher-reference', reference, '--student-policy-dir', policies,
                   '--student-columns', columns, '--output', destination], teacher_root)
            evaluations[epoch] = json.loads((destination / 'rollout_metrics.json').read_text())
        if tuple(evaluations) != tuple(EPOCHS):
            raise ValueError('Evaluation epoch order differs from frozen five checkpoints')
        learning_curve, formal, comparison = summarize(evaluations)
        write_json(output / 'summary/learning_curve.json', learning_curve)
        write_json(output / 'summary/hd2_epoch150_scientific_metrics.json', formal)
        write_json(output / 'summary/hd1_hd2_comparison.json', comparison)
        for key in status:
            if key in formal:
                status[key] = formal[key]
        status.update(HD2C_EXPORT_REUSE='PASS', HD2C_ROLLOUT_REUSE='PASS', HD2C_GATE_REUSE='PASS')
        code = 0 if status['HD2_FINAL'] == 'PASS' else 1
    except Exception:
        # The external orchestration boundary preserves the traceback and packages partial evidence.
        manifest['failure'] = traceback.format_exc()
        (output / 'failure.txt').write_text(manifest['failure'], encoding='utf-8')
        print(manifest['failure'], file=sys.stderr)
    finally:
        manifest['exit_code'] = code
        manifest['missing_evidence_semantics'] = 'NOT_RUN is missing evidence, never a scientific FAIL'
        lines = '\n'.join(f'{key}={value}' for key, value in status.items()) + '\n'
        (output / 'status.txt').write_text(lines, encoding='utf-8')
        write_json(output / 'status.json', status)
        print(lines, end='')
        print(f'OUTPUT_ZIP={package(output, manifest)}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
