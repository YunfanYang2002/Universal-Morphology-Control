# HyperDistill (MetaMorph teacher): HD0

Update: the user subsequently confirmed real-server **HD0_FINAL=PASS**.
The local-only limitations below describe the original implementation handoff.
The next reduced protocol is documented in `HYPERDISTILL_HD1.md`.

This is teacher substitution, **not original HyperDistill reproduction**. The
official `HNMLP` architecture is unchanged. RMAMorph alone loads and runs the
teacher; the two repositories run in separate processes because both import
`metamorph`. No teacher training, MorphAdapt changes, or HD1 run is included.

## Selected teacher and current evidence

- Local pair: `D:/CODES/master/rmamorph/output/morphadapt_dr_s1415_evidence_20260829_103504/training/{config.yaml,Unimal-v0.pt}`.
- Original server pair, recorded by its training completion manifest:
  `~/Workspace/Code/rmamorph/output/metamorph_dr_matched_s1415_100m/{config.yaml,Unimal-v0.pt}`.
- Checkpoint SHA256: `ed5bfbcaa57a04aafd58411b8735faa392377d1cd9e16c784a8ddc0ce8fd3d84`.
- Local config SHA256: `90764721ea3474e1e9b2f177b479d74e1fdc5b61d4a0e292cf06f32c2fb08f51`.
- Native CPU deserialization was executed: `ActorCritic`, `TransformerModel`,
  `adapt_enabled=False`, RMS shape `(624,)`, count `99942432.0001`.
- This is not proof of native rollout correctness. The real local launch stops
  at missing RMAMorph `output/unimals_100/train/xml` assets. `mujoco_py` is also
  unavailable locally. Server runtime and real-data overfit are **NOT_RUN**.
- `HD0_FINAL=FAIL` currently means required evidence is missing, not that
  HyperDistill failed to imitate the teacher. Do not enter HD1 yet.

The default three walkers are the first three **training** entries in that config:

1. `vt-5506-0-12-02-19-26-55`
2. `floor-5506-2-16-01-10-58-23`
3. `mvt-5506-15-15-17-10-52-43`

## Mapping and static-context contract

| Item | Binding |
|---|---|
| Teacher observation | Native 12 limbs x 52 features, flattened limb first |
| Student observation | 12 x 17 = 204 features: body_xpos(3), body_xvelp(3), body_xvelr(3), body_xquat(4), then qpos/qvel for each of two joint slots |
| Feature selection | Per-limb columns `0..12,30,31,41,42` for this checkpoint; converter derives these by explicit field labels |
| Normalization | Native padded observation -> saved teacher RMS `(x-mean)/sqrt(var+1e-8)` -> clip to [-10,10] -> selected columns; no second normalization or RMS fit |
| Joint qpos | Native normalized joint-range coordinate, not raw joint angle; qvel is native velocity |
| Action | Identity mapping, 12 x 2 = 24 policy-side slots; raw distribution mean is the imitation target; canonical clipped command is separately exported; invalid slots exactly zero |
| Context | Official supported `context_version=1`: 17 body values + 9 values per joint slot = 35 per limb, 420 total |
| Context body fields | body_pos, body_ipos, body_iquat, geom_quat, body_mass, body_shape |
| Context joint fields | jnt_pos, joint_range, joint_axis, gear |
| Context scaling | Official `Agent.get_context` fixed affine ranges, independently checked against official source; absent joint slots and padded limbs are zero after scaling |
| Masks | True means invalid/padded; observation mask `(12,)`, action mask `(24,)` |
| Adjacency/order | Native child-parent edges `(12,12)` and native body/joint/actuator names and slot masks recorded; HN transformer context encoder does not consume adjacency |

At each teacher reset the exporter takes one nominal context snapshot. Reset
motor/friction/mass multipliers are fixed to 1 and mid-episode mutation is off.
The converter rejects different context across episodes of the same walker.
Offline training regenerates differentiable HN parameters per batch, as in the
official model. Evaluation generates them once, then `eval()` uses the frozen
parameters. The deployment TorchScript takes only selected proprioception and
has no context or MuJoCo model input. For the same nominal walker this frozen
policy is reused across resets; it cannot reread changed mass/gear at t=250.

The local synthetic check forbids calling `generate_params` during evaluation
and changes supplied context while asserting identical outputs. This establishes
the adapter's freeze contract; an actual mutation rollout has not been run.
An adaptation-enabled privileged teacher candidate was explicitly rejected.

## Files and bounded protocol

- `hd0_teacher_export.py`: native loading/inference, reset snapshots, provenance,
  three-walker expert export and one-walker student deployment.
- `convert_rmamorph_teacher_to_hyperdistill.py`: explicit feature mapper and
  official `obs` / `act` / `act_mean` pickle fields, selected RMS and reload check.
- `metamorph/algos/distill/distill.py`: opt-in offline HD0 gate using official
  ActorCritic/HNMLP, Adam and balanced half-MSE semantics. Original distillation
  entry remains available; its preexisting broken one-walker validation split
  is not used by HD0.
- `hd0_student.py`: small gate CLI, checkpoint, curves and frozen deployment.
- `run_hyperdistill_hd0.py` and `run_hyperdistill_hd0_server.sh`: fail-fast stage
  execution and evidence ZIP on success or failure.
- `tests/test_hd0_bridge.py`, `tests/test_hd0_student.py`: synthetic engineering
  checks, not experimental evidence.

Defaults: 3 completed episodes per exported walker, at most 5000 steps per
walker, first walker only for training, seed 1409, 50 epochs, batch 64, official
two-hidden-layer HNMLP. A deterministic episode-level split holds out at least
one whole trajectory. No transitions from that episode enter training.

Report raw valid-slot MSE and the official balanced loss `0.5 * MSE` (the code
calls it KL; it is not an estimated full-distribution KL here). Before seeing
real data, the engineering gate requires final train **and** held-out MSE at
most half their initialization values, plus finite real student rollout.
The 0.5 reduction ratio is a pragmatic overfit gate, not a significance test or
a locomotion performance claim. Teacher/student returns, lengths and action
MSE on student-visited observations are also reported; return equality is not
required. HD1 is only a future decision after real HD0 PASS.

## Server command

Use the existing RMAMorph Python environment with working MuJoCo/CUDA and
training assets. No guessed Conda environment is activated. The script records
the interpreter and fails if dependencies or the selected checkpoint are absent.
Assuming this checkout is at `~/Workspace/Code/Universal-Morphology-Control`:

```bash
cd ~/Workspace/Code/Universal-Morphology-Control && git pull --ff-only && bash tools/run_hyperdistill_hd0_server.sh
```

It pins the selected checkpoint and paired config SHA256. If the same files live
elsewhere, supply `--config PATH --checkpoint PATH` (relative paths resolve
against `--rmamorph-root`). `--walkers ID ID ID` must remain within its training
config. `--no-keep-open` as the first shell argument disables interactive terminal
retention for automation. Server performs no commit.

All generated files and subprocess temporary files go below the checkout's
`./tmp/`. Final output is `OUTPUT_ZIP=.../tmp/hyperdistill_hd0_<timestamp>_package.zip`.
It contains stage logs, original failure traceback if any, selected config and
provenance, all artifacts produced by completed stages (arrays, RMS, student
checkpoint/TorchScript, metrics and curves), and all 14 gate lines.
Missing evidence prints FAIL with `gate.json.not_run` distinguishing it
from a measured negative result. A failed stage prevents dependent stages.

## 验证方式

Executed locally from this checkout:

```powershell
python -m unittest discover -s tests -p 'test_hd0_*.py'
python -m py_compile tools/hd0_teacher_export.py tools/convert_rmamorph_teacher_to_hyperdistill.py tools/hd0_student.py tools/run_hyperdistill_hd0.py metamorph/algos/distill/distill.py
```

Expected: tests report OK (student test requires CUDA), compilation exits 0.
Git Bash `bash -n tools/run_hyperdistill_hd0_server.sh` also exits 0.
The real local preflight is expected to fail on missing RMAMorph assets and still
print `OUTPUT_ZIP`. Only the server command above can establish real-data HD0.
