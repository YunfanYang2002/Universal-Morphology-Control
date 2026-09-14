# HD1 reduced cross-morphology gate

Baseline name: **HyperDistill (MetaMorph-DR teacher)**. The user's real-server
HD0 result is frozen as PASS: train MSE 2.8209 -> 0.02065, validation MSE
2.6840 -> 0.06253, mean teacher/student returns 3218.81/3172.93, student lengths
1000/1000/1000. This implementation does not rerun or reinterpret HD0.

## Frozen experiment

- Same `metamorph_dr_matched_s1415_100m/Unimal-v0.pt` and paired config, pinned
  to their HD0 SHA256 values. No teacher substitution or additional training.
- Same native 624 -> selected 204 observation mapping, saved teacher RMS,
  420-dimensional static context, 24 canonical action slots and strict zero
  invalid slots. Original HNMLP and HD0 optimizer/loss defaults remain unchanged.
- Training: 18 morphologies, six each floor/mvt/vt, 3 episodes each, horizon
  1000, nominal motor/friction/mass multipliers 1, no mid-episode mutation.
- Deterministic split per morphology: first two sorted episode IDs train,
  third validates. Global episode IDs are recorded alongside this mapping.
- Student: seed 1409, 50 epochs, batch 64, same two-hidden-layer official
  HNMLP, Adam and conditional gradient clipping as HD0. No sweep or tuning CLI.
- Loss remains `0.5 * mean_rows(sum_valid_squared_error / valid_action_count)`.
  Report MSE and this official KL-labelled half-MSE; do not present it as an
  independently estimated full-distribution KL.
- Final-epoch checkpoint is used. No best-checkpoint selection. Per-family
  validation is reported both as unweighted mean of per-morphology MSE and as
  explicitly labelled transition-weighted MSE; neither changes training loss.

## Selection and provenance

Take first six IDs per family in the frozen teacher training config, then the
first two selected IDs per family for train-smoke. Take first two per family
in `configs/morphadapt_metamorph_dr_formal_ood98.txt` for OOD-smoke. This is a
nominal six-walker smoke drawn from the frozen disjoint pool, not a full
Strict-OOD97 run. Family is the explicit floor/mvt/vt naming prefix.

The runner verifies whole-pool teacher/OOD disjointness, exact 18/6/6 counts,
family balance, XML/JSON parseability and hashes. A missing chosen asset fails;
it does not substitute another morphology or select by performance. It writes:

- `provenance/hd1_train_walkers.txt` and `hd1_train_provenance.json`
- `provenance/hd1_train_smoke_walkers.txt` and corresponding provenance
- `provenance/hd1_ood_smoke_walkers.txt` and corresponding provenance
- `provenance/selection.json` with source paths/hashes and membership flags

Local selection inspection found all 24 distinct selected XML+metadata pairs
in this repository's official data folders. This does not certify the server's
native asset paths; those are validated again by the runner.

## Evaluation and predeclared gates

Each of 12 smoke walkers has 3 teacher and 3 student episodes, horizon 1000.
Teacher-reference results are reused, not rolled out a second time. OOD
reference export happens **after student training** and cannot enter its optimizer.

HN parameters are generated separately from each smoke walker's frozen nominal
context. The named TorchScript policy is reused for that same morphology;
every reset compares actual raw context against its reference. Policy bytes,
generated-context bytes and checkpoint are hashed. No context enters the
per-step TorchScript interface. Real mutation leakage remains an HD2 check.

| Gate | Predeclared requirement |
|---|---|
| Runtime | All six OOD student initializations/rollouts complete; no NaN/Inf or simulator numerical failure |
| Cross-morphology control | At least 4/6 OOD walkers have student median episode length >=500 AND median forward displacement >=1.0 m |
| Normalized return | Median of defined OOD mean-student/mean-teacher return ratios >=0.50 |
| Final | All three gates above PASS |

Displacement is native terminal `info['x_pos'] - reset sim.data.qpos[0]`, before
vector-environment auto-reset. Teacher return must exceed 1e-6 to define a
ratio; otherwise JSON uses null and it is excluded from the ratio median.
If no ratio is defined, the normalized gate fails. All six walkers remain
part of runtime and locomotion gates. Early termination and time limit are
reported separately; early termination is not invented into a diagnosed fall.

Per-walker reports include teacher/student returns and lengths, displacement,
termination reasons, ratio, numerical failures and valid-slot action MSE on
student-visited states. A failed scientific gate writes ordered diagnostic
evidence without changing any thresholds, budget, architecture or optimizer.
No HD2/Strict-OOD/mutation experiment starts automatically.

## Implementation and evidence

- `hd1_protocol.py`: deterministic split selection, provenance and frozen gates.
- `hd0_teacher_export.py`: opt-in HD1 selection/evaluation extensions; existing
  native teacher loader, rollout, normalization and static-context mapper reused.
- `convert_rmamorph_teacher_to_hyperdistill.py`: optional selected-ID list for
  HD1 counts/order; HD0 still defaults to exactly three walkers.
- `hd1_student.py`: multi-morphology row indexing over the official HNMLP,
  episode split, metrics and per-morphology frozen export; no new network.
- `run_hyperdistill_hd1.py` and shell launcher: existing staged execution and
  packaging workflow extended for HD1, with fatal stage errors stopping dependencies.

Every output lives in `./tmp/hyperdistill_hd1_<timestamp>/`. The package contains
JSON/YAML/TXT/CSV/log evidence, git SHAs, environment versions, commands,
configs, selection provenance, curves and per-morphology metrics. Raw expert
arrays, pickle datasets, model checkpoints, TorchScript and video are excluded
by default; their absolute paths, sizes and SHA256 remain in `manifest.json`.
The main artifact is `..._package.zip`; when larger than 25 MiB an additional
`..._audit_slim.zip` limits large logs to their last 64 KiB while preserving
small audit files. Failures also package traceback/status/config evidence that
was available before the failing stage.

All required HD1 status and OOD summary keys are printed. Unmeasured numeric
quantities use NOT_RUN, not zero; a measured but undefined ratio prints UNDEFINED.
FAIL with `manifest.unmeasured` is missing
evidence, not a measured scientific failure. HD1 has not been run on the server
by this local implementation task.

## 验证方式

Local checks (synthetic engineering evidence, not HD1 scientific results):

```powershell
python -m unittest discover -s tests -p 'test_hd*.py'
python -m py_compile tools/hd1_protocol.py tools/hd1_student.py tools/run_hyperdistill_hd1.py tools/hd0_teacher_export.py tools/convert_rmamorph_teacher_to_hyperdistill.py
```

Expected: tests report OK; CUDA-dependent tests skip if CUDA is unavailable.
Compilation exits 0. `bash -n tools/run_hyperdistill_hd1_server.sh` exits 0.
Tests cover mixed-context/mask batching against a per-morphology oracle,
episode separation, RMS checks, padded targets, gate boundaries, undefined
ratios, split provenance and exclusion of raw artifacts from audit ZIPs.

Reuse the already working `hyperdistill` environment. The user confirmed HD0
needed no compatibility shim; none is installed or requested. Server command:

```bash
cd ~/Workspace/Code/Universal-Morphology-Control && git pull --ff-only && conda activate hyperdistill && bash tools/run_hyperdistill_hd1_server.sh
```

Interactive terminal stays open on success/failure; shell argument
`--no-keep-open` disables this for automation. Default RMAMorph root is
`~/Workspace/Code/rmamorph`; `--config`/`--checkpoint` can relocate the same
hash-pinned pair. Server only pulls and runs; it makes no commit.
