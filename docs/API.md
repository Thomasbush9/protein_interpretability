# Writing your own scripts

Two runnable scripts to start from:

- **`experiments/analysis/reproduce_headline_transfer.py`** — reproduces the
  report's headline (+0.758) exactly, in ~130 lines. If you want to see what a
  real result is actually made of, read this one.
- **`experiments/analysis/example_transfer_probe.py`** — the same shape at
  minimum size: cohort → verify → captures → statistic → archived result.
  Copy this one.


Everything here runs from a checkout. The library lives in
`src/protein_interpretability/`, the experiment entry points in `jax_harness/`.

One rule shapes the whole layout, and it is enforced by a test rather than by
convention:

> **`analysis/` may not import a model backend or load weights.** It may use
> numpy, scipy, and jax as a numerics library. Anything that loads a model lives
> in `collection/`.

That is narrower than "no jax" on purpose — three report producers run their
linear algebra on the GPU, and `analysis/basis.py` needs a float64 `jnp` SVD
because float32 corrupted an archived basis once. `tests/test_boundaries.py`
asserts the real rule.

---

## Two environments, and which to use

| | how to run | what it can do |
|---|---|---|
| **analysis** | `uv run python …` on the login node | read artifacts, fit bases, compute statistics, make figures |
| **model** | `sbatch jax_harness/checkout.sbatch script.py …` | load Boltz-2 / OpenFold3 / Protenix, capture, intervene |

Never load a model on the login node. Both submitters run on a GPU node because
this account has no CPU partition:

```bash
# runs the git checkout — use this for anything touching the library
sbatch jax_harness/checkout.sbatch analyze_svd.py --out $W/runs/mine.json

# runs prot_interp_files/harness/, a copy that does NOT carry src/
sbatch jax_harness/analysis.sbatch analyze_svd.py --out $W/runs/mine.json
```

If you changed anything under `src/`, use `checkout.sbatch`. The mirror will not
have your change and the job will silently run the old code.

---

## Cohorts: which assays, and are they still the same ones

```python
from protein_interpretability.collection import Cohort

basis = Cohort.load("basis_assays")        # 12 stability assays
held  = Cohort.load("heldout_assays")      # 16, disjoint from the above

basis.verify()                             # raises if any input moved
basis.assert_disjoint(held)                # raises if they overlap

for assay in basis:
    print(assay.id, assay.wt_length, assay.msa_path, assay.msa_rows)
```

`Cohort.available()` lists them; `pi cohort --list --verify` does the same from
a shell.

**Call `verify()` before you submit.** Every input carries a sha256 taken when
the manifest was written. An alignment regenerated in place has the same path
and a plausible size, so a run against it does not fail — it returns a number
computed from an input nobody chose. That has happened here before.

Cohorts used to be whatever `runs/gym2_*.npz` matched at the time, which meant
deleting a capture silently shrank the cohort. Regenerate the manifests
deliberately, after you meant to change something:

```bash
uv run python jax_harness/build_cohort_manifests.py --out configs/cohorts
```

---

## Reading captures

Captures are `.npz` archives under `$W/runs/`. Read them through `Capture`
rather than `np.load`, so a missing field fails where you can see it:

```python
from protein_interpretability.artifacts import load_capture

cap = load_capture("$W/runs/gym2s_ARGR_ECOLI_Tsuboyama_2023_1AOY.npz")
cap.n_layers()             # 64
cap.has_vectors()          # does it carry directions, or only magnitudes?
dz = cap.field("dz_site")  # [n_variants, n_layers, 128]
y  = cap.field("score")    # the assay score, one row per variant
```

The assay score travels **with** the capture, so you don't re-read the CSV and
re-align by mutant name; `mutant` and `pos` are there too if you need to join.
A wrong field name gets you the list of what the archive actually holds, which
is faster than guessing.

`has_vectors()` is worth checking. Some archives hold only magnitudes, and a
basis fitted on a magnitude is not the same object as one fitted on a direction
— `load_capture(..., require_vectors=True)` refuses rather than lets that pass.

---

## The shared basis

```python
from protein_interpretability.analysis import basis

B = basis.fit(blocks, layer=-1, orient_on="kl_glob", n_boot=2000, seed=0)
B.V              # components  [n_pc, 128]
B.ev             # explained variance
B.project(dz)    # coordinates in the basis
B.protocol       # the protocol fields describing this fit — merge into your block
B.save(path)     # and basis.load(path) to freeze one
```

The defaults are the published protocol: per-assay z-score, pool, subtract the
pooled mean, SVD, then fix the sign by `orient_on`. **Sign orientation is not
cosmetic** — PC2's direction is what the steering result is defined against, so
changing `orient_on`, `orient_k` or `n_boot` gives you a different scientific
object, not a re-run of the same one. `analyze_basis` orients at `n_boot=500`
against `analyze_pc2`'s 2000, and that divergence is recorded rather than fixed.

The basis rotates with depth. `pc2_v2.npz` is a **last-layer** object; do not
project a depth profile onto it.

---

## Statistics

```python
from protein_interpretability.analysis import statistics as st

st.spearman(x, y)                     # tie-aware
st.partial_spearman(x, y, control)
st.block_permutation(...)

# Clusters are assays. Returns (point, lo, hi, n_clusters) -- four values.
point, lo, hi, k = st.cluster_bootstrap({"ASSAY_A": [...], "ASSAY_B": [...]},
                                        n_boot=10000, seed=0)
```

`cluster_bootstrap` resamples **clusters, not observations**, and the point
estimate is the mean of the per-cluster statistics — so an assay with 10 splits
does not outvote one with 5.

Use these rather than `scipy.stats` directly. Everything reported in this
project is a within-assay Spearman averaged over assays, with intervals that
respect the clustering — three bugs came from getting exactly that wrong, which
is why the module exists.

`protein_interpretability.analysis.metrics` has `auc`, `mcc`, `ndcg_top`,
`recall_top`, `all_metrics` for checking a conclusion is not metric-bound.

---

## Writing a result

Results go through one seam, and it will refuse you:

```python
from protein_interpretability import artifacts
from protein_interpretability.experiments import protocol as P

res = {"spearman": 0.42, "n": 1287}
proto = P.protocol(
    script="my_script.py",                                 # required
    design="within-assay, position-grouped splits",        # required
    layer=P.layers("final"),                               # required
    features=P.features("dz_site, final-layer pair row", 128),   # required
    source="$W/runs/gym2s_<assay>.npz",                    # required
    n_assays=12,                                           # required
    seeds=5, frac=0.25, n_perm=2000,                       # anything else you set
)
artifacts.write_result("$W/runs/my_result.json", res, protocol=proto)
```

Two guards, both of which raise rather than warn. `write_result` refuses a
missing `protocol`, and `protocol()` refuses a block missing any of
`script, design, layer, features, source, n_assays`. The block states what the
number is comparable to, and it exists because a page once quoted an archive
that recorded neither its layer convention nor its orientation rule.

`P.layers(which, n_layers=None, window=None)` and
`P.features(name, width, kept=None, note=None)` build the two structured fields.

`write_result` also stamps provenance: `argv`, git commit, host, SLURM job. That
is what makes any result re-runnable later:

```bash
pi reproduce $W/runs/my_result.json --out /tmp/check --checkout --submit
pi verify $W/runs/my_result.json /tmp/check/my_result.json
```

---

## Loading a model

```python
import pi_models

wrapper = pi_models.load("boltz2", msa="full")   # or "subsample"
ex = pi_models.run_one(wrapper, seq, a3m_path, recycles=3, name="boltz2")
ex.logits, ex.ed, ex.plddt, ex.ca                # normalised across all models
```

`msa` is **required and has no default**:

- `subsample` — 1024 alignment rows drawn per PRNG key. Fast, mosaic's own
  default, and **not bit-reproducible**.
- `full` — the whole alignment. Bit-reproducible, and the regime every archived
  Boltz-2 capture was produced with.

Measured over two assays: the two agree on `dz_site` as well as two subsample
draws agree with each other, so the choice costs accuracy nothing. It costs
*exactness*. Use `full` for anything going in the paper.

Record it with `pi_models.regime_block(name, wrapper)`, which reads the flag off
the built model rather than echoing your argument.

Trunk internals — per-layer `z`, logit lens, path patching — are in `pi_core`,
which also loads Boltz-2 directly:

```python
import pi_core as pi

model = pi.load_model(subsample_msa=False)          # joltz, the default
model = pi.load_model(backend="mosaic")             # via pi_models, verified equal on dz_site
feats, handle = pi.load_features(yaml_text)         # keep `handle` alive
out = pi.trunk_capture(model, feats, recycling_steps=3, key=jax.random.key(0))
out["distogram"], out["trunk_state"]
```

`load_features` returns a temporary-directory handle as its second value. Boltz's
processed inputs live inside it, so keep it alive for as long as you use
`feats` — letting it go out of scope deletes the features out from under you.

Validate anything a model returns against the shared schema:

```python
from protein_interpretability.collection import records
records.validate(ex)                  # shapes, finiteness, probabilities sum to 1
records.assert_comparable(a, b)       # same bin GRID, not just the same bin count
```

`assert_comparable` is the check a cross-model KL needs: two models can each be
valid and still be incomparable, and a KL across two different distance grids is
a well-formed number that means nothing.

---

## Before you submit

```bash
pi inspect jax_harness/my_script.py
```

Catches a module-scope backend import (the file cannot then be inspected on a
login node) and a result written with no protocol block (which otherwise fails
at the *end* of a job).

Import backends lazily so your script stays inspectable:

```python
def build():
    import mosaic          # inside the function, not at module scope
    ...
```

---

## When numbers differ

Three producers are not bit-stable, each established by running the *unchanged*
code twice:

| producer | band | why |
|---|---|---|
| `svd_ds_v1` | ~9e-4 | prediction-ordered curves, per-assay entries |
| `svd_dz_v3` | ~3e-14 | float noise in a batched SVD |
| `gate_probe` | ~9e-7 | `jax.vmap` over Pairformer transitions in float32 |

`pi verify OLD NEW` applies these automatically and requires anything else to be
exact.

**Run the unchanged code twice before calling a difference a regression.** Three
apparent regressions during this refactor turned out to be run-to-run variation,
and one apparent MSA effect turned out to be the diffusion sampler — coordinates
come from a stochastic sampler keyed by the same seed, so never compare
structures across different keys and attribute the difference to what you
changed.

---

## Where things are

```
src/protein_interpretability/
├── analysis/          statistics, basis, chemistry, metrics   (no backends)
├── collection/        cohorts, records                        (backends, lazily)
├── experiments/       protocol
├── artifacts.py       write_result, load_capture, Capture
└── cli/               pi reproduce | verify | inspect | cohort

jax_harness/           exp_* collect · analyze_*/probe_* analyse · fig_* plot
                       build_*_report assemble · launch_*.sh, *.sbatch submit
                       pi_*.py are aliases into the package — import either name

configs/cohorts/       the four checksummed cohort manifests
tests/                 model-free; `uv run pytest tests/ -q`, under a second
```

Don't add a new `pi_*` module. New library code goes in the package.
