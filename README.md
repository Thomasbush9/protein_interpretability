# Protein interpretability

Mechanistic interpretability of protein structure-prediction trunks — Boltz-2,
OpenFold3 and Protenix — with the analysis kept strictly separable from the
models.

The central result: **Boltz-2's internal state predicts mutational stability far
better than anything it emits.** A probe on the trunk reaches a within-assay
Spearman well above the richest description of the structure the model actually
produces, and a single shared direction (PC2) transfers across assays, survives
on a held-out cohort, and steers the model causally when injected.

---

## Layout

```
src/protein_interpretability/     the library
├── analysis/      statistics, basis, chemistry, metrics   — imports NO backend
├── collection/    cohorts, records                        — backends, lazily
├── experiments/   protocol
├── artifacts.py   write_result, load_capture, Capture
└── cli/           pi reproduce | verify | inspect | cohort

jax_harness/       experiment entry points: exp_* collect, analyze_*/probe_*
                   analyse, fig_* plot, build_*_report assemble,
                   launch_*.sh and *.sbatch submit
configs/cohorts/   four checksummed cohort manifests
experiments/       readable programs written against the library
docs/API.md        how to write your own scripts   ← start here
tests/             model-free
```

One rule shapes it, and a test enforces it rather than a convention:

> `analysis/` may not import a model backend or load weights. It **may** use
> jax as a numerics library.

That distinction is narrower than "no jax" deliberately — three report producers
run their linear algebra on the GPU, and the basis needs a float64 `jnp` SVD
because float32 once corrupted an archived basis.

---

## Install

Python 3.12, `uv`:

```bash
uv sync
uv run pytest tests/test_records.py tests/test_boundaries.py -q
```

That is the whole analysis environment — it installs no model. Running a model
needs the mosaic container, which the submitters below know how to reach.

---

## Running things

Nothing that loads a model runs on the login node. This account has no CPU
partition, so analysis jobs go to a GPU partition too and are written to use the
device.

```bash
# the git checkout — use this for anything touching src/
sbatch jax_harness/checkout.sbatch analyze_svd.py --out $W/runs/mine.json

# prot_interp_files/harness/, a COPY that does not carry src/
sbatch jax_harness/analysis.sbatch analyze_svd.py --out $W/runs/mine.json
```

If you changed the library, use `checkout.sbatch`; the mirror will not have your
change and the job will quietly run the old code.

---

## The `pi` command

```bash
pi cohort                                   # the four cohorts
pi cohort basis_assays --list --verify      # and check their inputs on disk
pi inspect jax_harness/my_script.py         # before you submit
pi reproduce $W/runs/svd_dz_v3.json --out /tmp/check --checkout --submit
pi verify $W/runs/svd_dz_v3.json /tmp/check/svd_dz_v3.json
```

Every result file records the exact `argv` that produced it, so `pi reproduce`
replays it rather than asking you to reconstruct the command. `pi verify`
applies the measured non-determinism bands — three producers are not bit-stable,
and each band records how it was measured.

---

## Two things that will save you an afternoon

**Run the unchanged code twice before calling a difference a regression.** Three
apparent regressions during the refactor turned out to be run-to-run variation.
One apparent MSA effect turned out to be the diffusion sampler: coordinates come
from a stochastic sampler keyed by the same seed, so comparing structures across
different keys measures the sampler, not what you changed.

**Choose the MSA regime explicitly.** `pi_models.load(name, msa=...)` has no
default. `subsample` draws 1024 alignment rows per key — fast, and not
bit-reproducible. `full` uses the whole alignment and is what every archived
Boltz-2 capture was produced with. The two agree on the science; they differ in
whether a rerun gives you the same number.

---

## Writing your own scripts

Read **`docs/API.md`**, then copy
`experiments/analysis/example_transfer_probe.py` — cohort → verify → read
captures → statistic → archived result, in about 80 lines, runnable on the login
node.

Results are written through one seam, which refuses a result with no protocol
block. That block states what a number is comparable to, and it exists because a
page once quoted an archive that recorded neither its layer convention nor its
orientation rule.

---

## Data

Datasets, captures, weights and reports live outside this repository under
`prot_interp_files/` — captures in `runs/`, assay tables and alignments in
`data/gym/`, the master report in `report_master/`. Nothing large belongs in
git.
