# JAX harness — Boltz-2 trunk interpretability

Mechanistic interpretability of the Boltz-2 trunk under mutation, built on
**joltz** (the JAX/equinox reimplementation of Boltz-1/2 shipped inside
`mosaic_setup/images/mosaic.sif`) rather than the PyTorch stack in this repo's
`src/`. joltz keeps the Pairformer and MSA stacks as a stacked parameter pytree
run by `jax.lax.scan`, which makes three things possible that hooks cannot do
cleanly: per-layer capture by re-running the scan with `ys` populated, component
ablation as a single `eqx.tree_at`, and path patching as a pytree field swap.

**Methods write-up:** `vault/Lab/protein-interp/methods_pairformer_interp.md`
(harness validation, units conventions, cohort construction, every experiment
with its controls). **Chronological log incl. corrections:**
`vault/Lab/protein-interp/log/2026-07-30-jax-pairformer-harness.md`.

## Running

Everything model-side goes through SLURM; nothing heavier than file parsing runs
on a login node.

```bash
sbatch run.sbatch <script.py> [args...]      # kempner_h100, via mosaic-exec.sh
```

## Layout

| file | role |
|---|---|
| `pi_core.py` | model/feature loading, instrumented Pairformer + MSA scans, `iteration` / `run_trunk`, distogram logit lens |
| `pi_paths.py` | five-route decomposition, `patch()`, MSA depth truncation |
| `geom.py` | structure comparison — **tmtools only**, see docstring for why |
| `test_equivalence.py` | proves `pi_core.iteration` is bit-identical to joltz |
| `build_dataset.py` | GFP core/surface/random/scramble cohorts |
| `predict_wt.py` | wild-type structure → CA-only cif (for burial) |
| `exp_paths.py` | route necessity / sufficiency |
| `exp_layers.py`, `exp_sublayers.py` | per-layer and per-operation attribution |
| `exp_ablate.py` | causal ablation of `transition_z` |
| `exp_subspace.py`, `exp_kl.py` | readout-subspace test; KL-vs-Ångström units check |
| `exp_relative.py` | normalisation against a scrambled-sequence control |
| `exp_matrix.py`, `analyze_onset.py` | layer-resolved matrices; spatial propagation |
| `exp_bench.py`, `score_bench.py` | structure vs confidence vs internal state |
| `exp_gym.py`, `exp_gym2.py`, `probe_gym.py` | ProteinGym features and probe |
| `analyze_rsa.py`, `exp_ensemble.py`, `analyze_ensemble.py` | RSA; ensemble spread and β |
| `compare_internal_output.py` | like-for-like internal vs output |
| `make_figures.py`, `fig_*.py` | figures |

## Two conventions that are not optional

1. **Divergence, not Ångström**, for anything compared across depth. The
   distogram sharpens with depth (entropy 2.16 → 0.81 nats), so `E[d]` in Å is
   not a valid cross-layer yardstick and manufactures false "suppression".
2. **Every quantity needs a denominator or a control.** Five conclusions in this
   project were withdrawn for want of one. The controls are built into the
   scripts; do not strip them.
