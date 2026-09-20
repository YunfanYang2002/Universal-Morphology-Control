# HD2C frozen OOD6 evaluation

Use the same active Python/CUDA/MuJoCo environment that completed HD2B. From this
repository on the server, after pulling the committed changes:

```bash
bash tools/run_hyperdistill_hd2c_server.sh
```

The launcher changes into the repository, records the environment, and retains
the terminal after success or failure. Automation can pass `--no-keep-open`.
It reads `tmp/hyperdistill_hd2b_s1409_20260916T062120_921323Z` by default;
`--hd2b-run` accepts another completed run under this repository's `tmp`.
`--rmamorph-root` defaults to `$HOME/Workspace/Code/rmamorph`.

Before any rollout, all five checkpoints (30/60/90/120/150) must exist, load
strictly into the unchanged official HD2B HN-MLP, contain finite parameters and
matching training counters, and receive pinned SHA256 hashes. The adapter never
loads the optimizer into an optimizer object or changes weights. HD2B checkpoints
have no RMS; the six evaluation contexts supply the identical frozen RMS from
the hash-locked HD0/HD1 teacher. The 204-column mapper is hash-checked.

The runner creates one fresh evaluation-only teacher reference using the existing
HD0 exporter, then exports six fixed TorchScript policies per epoch using its
existing exporter and executes the existing closed-loop evaluator. There are no
training, resumption, Strict-OOD97, or dynamics-mutation stages. Three episodes
per walker, seed 1409 plus walker index, horizon 1000, and nominal multipliers
remain the HD1 protocol. No OOD samples enter an optimizer.

Epochs 30/60/90/120 are diagnostic only. The unchanged HD1 gate function evaluates
epoch 150 as the only formal result, regardless of earlier performance. The
curve's median action MSE is the median of the six per-walker closed-loop MSEs;
individual episode MSEs remain in rollout records. `early_termination` is the
native termination category, not a separately validated fall classifier.

Artifacts are written to `tmp/hyperdistill_hd2c_ood6_<timestamp>/`; the printed
`OUTPUT_ZIP` points to its `_audit.zip`. The ZIP includes provenance, five
checkpoint bindings, teacher/config and asset hashes, export and rollout
metrics, the learning curve, epoch-150 gates, HD1 comparison, and HD2B epoch
history. Raw arrays and policy binaries stay on disk with hashes. Old HD2B root
failure files are never copied. A current infrastructure error preserves its
traceback and leaves unmeasured scientific gates `NOT_RUN`; a measured gate
failure is `HD2_FINAL=FAIL`. Either failure exits nonzero.

The HD1 reference (18 morphologies / approximately 52k samples / 50 epochs /
ratio approximately 0.105) is user-supplied, not reconstructed from this run.
HD2 changes morphology count, samples, epochs, and batch/loss fidelity together;
this comparison cannot isolate robot count or establish statistical significance.
The user-reported 405-PD coverage audit also found a nontrivial fraction of weak
teacher controllers. A failed gate is limited to HyperDistill (MetaMorph-DR
teacher) under this paper-scale distillation setup.

Local focused validation (no training or MuJoCo rollout):

```bash
python -m unittest discover -s tests -p 'test_hd2c_*.py'
python -m unittest tests.test_hd1_export
```

Synthetic checkpoint/export tests validate infrastructure only. Real checkpoint,
server MuJoCo, and scientific results require the server invocation above.
