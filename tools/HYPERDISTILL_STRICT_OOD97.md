# HyperDistill Strict-OOD97 formal baseline

This is evaluation-only. It loads HD2B `checkpoint_150.pt` as the formal
result and `checkpoint_030.pt` as supplementary training-horizon sensitivity.
Both use the current MorphAdapt formal contract: the frozen Strict-OOD97
identity, nominal dynamics, evaluation seed 1409, one episode per walker, and
a 1000-step horizon. No optimizer or dynamics-mutation stage is invoked.

The adapter is `tools/hyperdistill_morphadapt_adapter.py`; it reuses the
namespace-independent HD2 student constructor used by HD2B/HD2C and never
mutates rmamorph's teacher `cfg`. At each walker reset it binds the static
morphology context through the HD0 `base_env`/`native_raw_context` helpers;
the per-step formal observation remains context-free. The rollout process
calls `rmamorph/tools/evaluate_dynamics.py` through
`tools/hyperdistill_morphadapt_evaluator.py`; it supplies only the frozen
HyperDistill action and leaves environment, termination, return, tracking, and
aggregation semantics in the canonical evaluator.

On the server, after the repository and frozen MorphAdapt assets are present:

```bash
bash tools/run_hyperdistill_strict_ood97_nominal_server.sh
```

The runner writes the formal evidence package below project `./tmp/` and
fails closed if the 97-ID identity, train overlap, PD1000 exact-ID overlap,
checkpoints, or evaluator inputs are unavailable or inconsistent.
