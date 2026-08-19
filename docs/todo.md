# Refactoring TODO

This backlog turns `docs/refactoring_plan.html` into concrete implementation
work. The immediate target is a short, inspectable workflow that can:

1. identify a scientific task and model;
2. run that model over a checksummed cohort and collect specified activations;
3. compute offline statistics and metrics from the resulting artifacts; and
4. save results with enough identity and provenance to reproduce them.

The current repository already has useful foundations: checksummed cohorts,
model capabilities, `CaptureSpec`, backend-free analysis modules, atomic artifact
writers, protocol records, site configuration, SLURM rendering, and one Boltz-2
pair-layer vertical slice. The work below connects those pieces without
rewriting validated numerical kernels prematurely.

## Working rules

- [ ] Keep `analysis/` artifact-only. It may use JAX for numerics, but it must
  not import model backends or load weights.
- [ ] Import model backends only inside adapter factories or compute-time
  methods. `inspect` and `render` must remain login-node safe.
- [ ] Preserve the numerical behavior of existing capture kernels until an
  old-versus-new equivalence test has passed.
- [ ] Make invalid or incomplete work fail before model execution whenever the
  condition is knowable on the login node.
- [ ] Make tests assert refusals: missing inputs, unsupported fields, wrong
  layers, mismatched dtypes, incomplete cohorts, and incompatible artifacts.
- [ ] Do not restore `torch` as a project dependency. Model containers supply
  their own backends.
- [ ] Use `jax_harness/checkout.sbatch` for validation involving library code.
- [ ] Run unchanged code twice before assigning a numerical difference to a
  refactor.

## P0: correct the reference vertical slice

These are correctness issues in the current example and should be fixed before
using it as the template for more models.

### Fix `kl_site` — DONE

Both reductions now live in `collection/reductions.py` (pure numpy, no backend)
and are returned together from one `kl_reductions()` call, so the two fields
cannot be wired to the same expression without editing a function the tests
cover.

- [x] In `experiments/collection/collect_pairformer_layers.py`, compute
  `kl_glob` over all sampled token pairs.
- [x] Compute `kl_site` only over sampled pairs for which either token is the
  mutated token, matching `jax_harness/exp_gym2.py`.
- [x] Use the resolved token index, not an assumed residue index, when building
  the site mask. The residue-to-token identity is now CHECKED against the pad
  mask and refused if the valid tokens are not a prefix of the grid.
- [x] Refuse the run if the pair sample contains no pair touching the mutation
  site; do not write a plausible zero. Checked for every variant before the
  first trunk pass, since the sample and the sites are both known by then.
- [x] Avoid recomputing the full KL tensor twice for the two reductions.
- [x] Add a small model-free test with synthetic logits that proves global and
  site reductions produce distinct, expected values (`tests/test_reductions.py`).
- [x] Add a refusal test for an empty site mask.
- [x] Compare the corrected arrays against an unchanged `exp_gym2` collection
  on the same variants, pair indices, model seed, and input materialization.

Two further corrections the fix required, neither of them in this list
originally:

- The divergence itself was wrong. The new script computed a one-sided
  `KL(mut || wt)` in float64; every archived `kl_glob` and `kl_site` is the
  SYMMETRIC KL (Jeffreys) of `exp_gym.skl`, in float32. Same field names, a
  different quantity. `reductions.symmetric_kl` is `skl` expression for
  expression, and does not promote dtype.
- The pair sample was drawn differently: `integers(0, n_tok, 1500)` against
  `choice(valid, 1500)` with self-pairs dropped. The diagonal is a delta at
  distance zero for every variant, so keeping it diluted both reductions.

Validation, offline (`tests/test_kl_matches_the_archive.py`, no GPU required):
`gym2s_*` stores the final-layer sampled logits it reduced, so both fields can
be recomputed from the archive's own numbers. Over 250 archived variants of
`ARGR_ECOLI_Tsuboyama_2023_1AOY`, the corrected `kl_glob` is BIT-IDENTICAL to
the recorded curve and `kl_site` matches to 9.5e-07 (float32 summation over a
subset). The pair indices, which the archive does not store, were recovered by
replaying the producer's RNG stream and accepted only because they reproduce
the archived pair count and the archived `kl_glob` exactly. Those indices are
now archived at `$W/runs/pairs_gym2s_ARGR_ECOLI_Tsuboyama_2023_1AOY.npz`.

Validation, live (SLURM job 40203202, commit 36a0408 + this change, H100,
2026-08-18): eight archived variants recollected through the corrected script
with `--mutants-from` and `--pairs-from`, so the comparison is the same
variants over the same pairs.

| field | mean abs diff | relative | corr |
|---|---|---|---|
| `dz_site` | 3.84e-02 | 0.94% | 0.999952 |
| `kl_site` | 1.38e-03 | 0.56% | 0.999990 |
| `kl_glob` | 3.41e-04 | 2.65% | 0.999671 |

Seven of the eight variants land at 0.7-1.1x the measured `dz_site` run-to-run
band of 3.9e-02. The eighth is A10E, the archive row already investigated in
commit 523f525 and attributed to a partially written alignment in the producer;
it is 7.1x the band here, against 7.77% archive-versus-fresh and 1.07%
fresh-versus-fresh recorded there. The table above excludes it for that reason.
Its `kl_glob` is also the worst of the eight at 12.5%, which is the same finding
in a second field.

For scale: in the new capture `kl_site` runs 4.3x to 31.4x `kl_glob` at the
final layer. That range is the size of the error the bug was writing.

Acceptance criteria:

- [x] `kl_site` agrees with the legacy implementation within the measured
  same-code run-to-run band — exactly, in fact, from the archived logits.
- [x] `kl_glob` remains unchanged apart from that same band — bit-identical.
- [x] A test would fail if both fields were wired to the same reduction again.
  In this archive `kl_site` is ~8x `kl_glob`, which is the size of the error the
  bug was writing.

Left open by this correction, and now recorded above under the artifact schema:
captures do not store their sampled pair indices, so two runs' KL fields cannot
be compared pair-for-pair without replaying an RNG stream. New captures from
this script write `pair_i`/`pair_j`, and `--pairs-from` reuses them.

### Make the example analysis refuse incomplete cohorts — DONE

- [x] Change `experiments/analysis/example_transfer_probe.py` so a missing
  required capture raises instead of printing `skip` and continuing. The check
  runs over the whole cohort before any array is read, so the refusal happens
  before the first per-assay number exists.
- [x] If partial-cohort exploratory analysis is useful, require an explicit
  `--allow-partial` flag and record the missing assay IDs in both the result and
  protocol (`partial`, `cohort_size`, `missing_assays` in each).
- [x] Assert that the number of loaded artifact groups equals the declared
  cohort size before computing the primary result — against the cohort, not
  against the table's own length.
- [x] Add a test where one capture is missing and verify that the default path
  refuses to produce a result (`tests/test_example_transfer_probe.py`, which
  also checks that a refusal leaves no output file behind).

Acceptance criteria:

- [x] A named 12-assay task cannot silently turn into an 11-assay result.
- [x] Any deliberately partial result is visibly and machine-readably partial.

## P1 STATUS, 2026-08-19

The API spine is in and validated on a GPU. What landed:

- `collection/task.py` — `ModelSpec`, `CollectionTask`, `ResolvedTask`,
  `resolve_layers`. Model-free; `inspect()` resolves and prices a whole sweep on
  a login node and verifies every cohort checksum.
- `collection/models/base.py` — `ModelAdapter` protocol, `ModelIdentity`,
  `ResolvedInput`.
- `collection/models/__init__.py` — lazy registry; listing it imports no backend.
- `collection/models/trunk.py` — one adapter for boltz2/of3/protenix, wrapping
  `exp_gym_deep.collect_assay`.
- `jax_harness/exp_gym_deep.py` — per-assay body extracted to `collect_assay()`
  as a PURE MOVE. The diff removes only the argparse block and the write tail;
  no expression in the numerical body changed.
- `experiments/collection/collect_xmodel_layers.py` and
  `jax_harness/launch_xmodel_layers.sh` — tonight's declaration and its sweep.

Corrections found on the way, none of them in the backlog before:

- `capabilities.REGISTRY` declared `of3` and `protenix` as NOT producing
  `kl_site`/`kl_glob`, and `boltz2` as not producing `dz_vec`/`ds_vec`, while
  every archive on disk carries them. `check_spec` therefore refused specs for
  captures that demonstrably work. Fixed, and
  `tests/test_capture_fields_match_archives.py` now reads the archives and fails
  if the declaration drifts from them again.
- `tests/test_capabilities.py` asserted that of3 + `kl_glob` must be refused,
  which is the same wrong belief in the test suite. Corrected, with the reason
  recorded in the test.
- Two spellings of the chain-mean pLDDT exist across families (`plddt` in
  gym2s, `plddt_mean` in xm). Both are now declared rather than one renamed.

Verified on GPU (job 40357913, protenix, 4 variants, layers 0/7/15): the
artifact holds a 3-layer axis labelled `[0, 7, 15]`, loads under
`require_meta=True, require_vectors=True`, and embeds the resolved task, the
task id, requested-versus-resolved layers, input checksums, and a model identity
READ OFF THE LOADED MODEL -- including `msa_regime: "n/a (no subsampling switch
on this model)"` where the task had asked for `subsample`.

Still open in P1: the versioned artifact schema and writer-seam validation, and
the input reproduction contract (portable `${PROT_INTERP_DATA}` roots,
post-materialization checksums). Job-local work directories ARE done, since the
A10E race made them urgent.

### Cohort coverage, measured 2026-08-19

All 217 ProteinGym substitution assays are on disk as CSVs. Only 31 have an
alignment, and 26 of those are under 100 aa:

| length | assays | with an alignment |
|---|---|---|
| 0-100 aa | 70 | 26 |
| 100-200 | 26 | 2 |
| 200-400 | 44 | 2 |
| 400-800 | 52 | 1 |
| 800+ | 25 | 0 |

Median assay length across all 217 is 245 aa; the cohorts in `configs/` top out
at 118. **Three assays have alignments and are in no cohort: PTEN_HUMAN
(403 aa), TPMT_HUMAN (245 aa), ESTA_BACSU (212 aa).** Adding them extends the
length range 3.4x with no MSA work.

The cost caveat I wrote here first was WRONG, and the correction is worth
keeping. I argued from the Pairformer's O(N^3) triangle operations that a
403-residue protein would cost ~240x a 65-residue one. Measured on the first
wave (2026-08-19, 100 variants): protenix costs 8.3 s/variant at 40 aa and
10.1 s/variant at 118 aa — **1.2x for a 3x longer protein**. boltz2 is
20.7 s/variant and of3 21.7 s/variant at 40 aa.

Length is not the driver at these sizes. Per-variant cost is dominated by
length-independent work: the alignment is re-parsed for every variant, and the
diffusion sampler runs a fixed 200 steps. `exp_gym2`'s docstring already said so
about the MSA — "Boltz re-parses it per variant and that, not the GPU, sets the
wall-clock" — and the scaling argument talked me out of believing it.

Consequence: the full sweep is ~26 GPU-hours with no job over ~50 min, and the
`length_ladder` cohort is much more affordable than the cubic estimate implied.
403 aa is still 3.4x beyond anything measured and the pair tensor does grow as
N^2 (boltz2's per-layer z stack is ~5.3 GB there), so run PTEN alone first.

The bottleneck for the other 186 assays is alignment generation, not data. The
pipeline for it exists and is local: `jax_harness/msa_panel3.sbatch` runs
`colabfold_search` against an on-disk MMseqs2 database inside `msa.sif`, no
network. `jax_harness/select_panel.py` chooses the next panel and
`msa_panel5.sbatch` aligns it.

### Capture fidelity versus protein size, measured 2026-08-19

Probes on PTEN (403 aa) against the 40 aa assays, all with `DRIFT_TOL = 2e-3`:

The tolerance is `pi_capture.DRIFT_TOL = 2e-3` for all three models. It is a
RELATIVE error on the final-layer pair representation,
`max|z_captured - z_trunk| / max|z_trunk|` in float64, against the model's own
NON-capturing entry point (`boltz2_trunk`, `of3.run_trunk`, `protenix.recycle`).
`MIN_SIGNAL_TO_DRIFT = 50` then requires a single mutation to move z at least
50x further than that drift. Both are enforced ceilings: a run that exceeds
either aborts rather than writing the capture.

| model | drift @ 40 aa | drift @ 403 aa | msa_depth | headroom at 403 aa |
|---|---|---|---|---|
| boltz2 | 0.0 | 0.0 | 1114 / 2049 | see caveat |
| of3 | 5.10e-04 | 6.69e-04 | 2050 | 33% of tolerance |
| protenix | 4.10e-04 | 1.40e-03 | 1114 / 2049 | **70% of tolerance** |

CAVEAT ON THE BOLTZ-2 ZEROS, and it corrects an earlier claim in this file.
They are NOT evidence that the Boltz-2 capture is exact. `pi_capture` records
that this exact belief was held before, was written into that module, and was
refuted: the supporting evidence had been produced on a degenerate input where
boltz silently substituted a one-row dummy MSA, and both paths agreed at 0.0
with an infinite signal/drift ratio. With real alignments the composed
3-recycle trunk drifts 2.5e-04 (RCRO) to 6.2e-04 (RS15, PSAE), protein-
dependent in the way accumulated rounding is, because `boltz2_trunk` scans its
recycles while the capture must unroll them and XLA fuses the two differently.
What `test_equivalence.py` proves is that a SINGLE iteration is exact, which is
a weaker statement.

The zeros measured here are not that artefact -- the alignments were healthy
(1114 and 2049 rows) and Protenix on the identical input drifted 4.1e-04 -- so
the two paths really did agree bit-for-bit on these inputs. But agreement on
two proteins is an observation about those shapes, not a property of the model,
and the archived `xm_boltz2_r1` captures carry 1.8e-04 on a third. Do not plan
around Boltz-2 being drift-free at other sizes.

Two consequences worth carrying:

- **Protenix is the model closest to its ceiling** (70% at 403 aa), so it is
  the one to watch as protein size grows. That much stands.
- **At drift 0 the signal/drift test is vacuous** -- any signal over zero is
  infinite, so the ratio criterion the module calls "the operative criterion for
  all three" provides no assurance on those runs. The drift check alone is
  carrying them.

Per-variant cost at 403 aa (sublinear in length, because the alignment re-parse
and the fixed 200-step sampler dominate): boltz2 38 s (from 20.7 s at 40 aa),
protenix 24 s (from 8.3 s). The `panel5` capture sweep is therefore roughly 80
GPU-hours across three models, not one night.

## P1: define the minimal end-to-end public API

Do this for Boltz-2 first. The first API should be small enough to understand
from one experiment script and concrete enough that it cannot describe a run
the implementation ignores.

### Add an explicit task and model declaration

- [x] Decide and document the stable name: `CollectionTask` is preferred here
  because it directly represents the user's requested operation.
- [x] Add a model-free `ModelSpec` containing at least:
  - canonical model name;
  - backend/implementation name;
  - checkpoint identity or logical checkpoint reference;
  - recycle count;
  - model PRNG seed;
  - MSA regime and cap;
  - network policy; and
  - any backend-specific options that affect the scientific result.
- [x] Add a model-free `CollectionTask` containing:
  - task name or logical run ID;
  - cohort/inputs;
  - `ModelSpec`;
  - `CaptureSpec`;
  - output location;
  - resume policy; and
  - optional resource overrides.
- [x] Give the task an `inspect()` or `resolve()` method that validates inputs,
  model capabilities, capture fields, layer selection, output identity, and
  estimated storage without loading a backend.
- [x] Keep the resolved task serializable so the inspected object is exactly
  what the compute job consumes.
- [x] Detect conflicting task declarations, such as different seeds in the
  task and adapter arguments.

Target surface:

```python
task = CollectionTask(
    name="boltz2_pair_layers",
    cohort=Cohort.load("smoke_pairformer"),
    model=ModelSpec(
        name="boltz2",
        backend="joltz",
        recycles=3,
        seed=0,
        msa="full",
    ),
    capture=CaptureSpec(
        model="boltz2",
        fields=("dz_site", "kl_site", "kl_glob"),
        layers=(15, 31, 63),
        reduction="vector",
    ),
    output="runs/boltz2_pair_layers",
)

task.inspect()
task.run()
```

Acceptance criteria:

- The model, inputs, fields, layers, reductions, seeds, and output are readable
  near the top of the experiment file.
- Constructing and inspecting the task imports no model backend.
- The compute path consumes a serialized/resolved task rather than rebuilding
  scientific choices from unrelated CLI defaults.

### Define the adapter protocol

- [x] Add `collection/models/base.py` with a small `ModelAdapter` protocol.
- [x] Require adapter metadata for architecture, backend, checkpoint, trunk
  depth, representation widths, output grids, and supported capture sites.
- [x] Require a login-node-safe capability/inspection path.
- [x] Require compute-time methods to report the resolved inputs and actual
  model identity they consumed.
- [x] Keep model-specific capture semantics explicit. Do not create a fictional
  common layer type across Pairformer, AlphaFold, and ESMFold.
- [x] Add an adapter registry/factory whose module-level imports remain
  backend-free.
- [x] Cross-check adapter metadata against `collection/capabilities.py` so the
  registry cannot drift independently.
- [x] Refuse unsupported site/layer/field combinations during inspection.

Suggested minimal protocol:

```python
class ModelAdapter(Protocol):
    def identity(self) -> ModelIdentity: ...
    def capabilities(self) -> ModelCapabilities: ...
    def materialize_inputs(self, record, work_dir) -> ResolvedInput: ...
    def collect(self, resolved_input, capture: ResolvedCaptureSpec): ...
```

### Add the Boltz-2 adapter without changing its numerical kernel

- [ ] Wrap the current Joltz/JAX loader and `exp_gym2.trunk_capture` behavior
  behind the adapter protocol.
- [ ] Decide where the validated capture kernel should permanently live. If it
  is promoted into the package, keep a thin harness alias and prove the move
  does not change outputs.
- [x] Keep the backend import and weight load inside the adapter factory.
- [x] Preserve explicit `full` versus `subsample` MSA behavior.
- [x] Keep network blocking on by default for fixed-input/reproduction tasks.
- [x] Record the actual loaded checkpoint and backend version rather than
  echoing only requested values.
- [x] Verify capabilities against the loaded model on the compute node.
- [x] Add a smoke test in the model environment for one short sequence and a
  minimal recycle count.

Acceptance criteria:

- The existing vertical-slice experiment no longer contains model loading,
  feature construction, mutation looping, and artifact assembly itself.
- The adapter uses the same numerical capture path as the archived producer.
- Old and new `dz_site` agree within the measured same-code band.

### Make selected layers real

- [x] Resolve negative indices and `"final"` to canonical non-negative layer
  indices during inspection.
- [x] Reject duplicate or out-of-order layer declarations unless their meaning
  is explicitly supported.
- [x] Pass the resolved layer selection into the collection runner.
- [x] At minimum, write only requested layers to the artifact. If the backend
  must internally traverse all blocks, make that implementation detail clear.
- [ ] Where practical, avoid retaining unrequested per-layer tensors in device
  memory.
- [x] Record both requested layer expressions and resolved absolute indices.
- [ ] Make array descriptors attach a physical layer index to every stored
  layer axis.
- [x] Validate that returned layers exactly match the resolved request before
  committing the artifact.
- [x] Add tests for `"all"`, `"final"`, positive indices, negative indices,
  cross-model depth differences, duplicates, and out-of-range indices.

Acceptance criteria:

- Requesting `(15, 31, 63)` produces an artifact whose layer axis has length
  three and is labeled `[15, 31, 63]`.
- A backend that returns all 64 layers against that request is refused before
  the artifact is committed.

## P1: make representation artifacts self-validating

### Define a versioned representation artifact schema

- [ ] Add an explicit schema version independent of the Python package version.
- [ ] Include a stable artifact/run ID and completion state.
- [ ] Store the fully resolved `CollectionTask` or its canonical serialized
  form.
- [ ] Store model identity:
  - architecture and canonical name;
  - backend and backend version;
  - checkpoint path/logical ID and checksum where feasible;
  - actual trunk depth and widths;
  - output grids and semantics; and
  - deterministic settings and seeds.
- [ ] Store input identity for every record:
  - logical input and reference IDs;
  - source sequence/checksum;
  - source MSA/checksum/depth;
  - materialized YAML/A3M/checksums;
  - feature-generation version;
  - residue-to-token mapping;
  - chain mapping; and
  - actual token and MSA counts consumed by the model.
- [ ] Store the resolved capture specification, including layer indices,
  reduction, recycle selection, dtype, sampled-pair count, and sampled-pair
  seed or indices.
- [ ] Store array descriptors with semantic field name, shape, dtype, axes,
  units where applicable, and layer coordinates.
- [ ] Preserve run provenance: argv, Git commit/dirty state, host, SLURM job,
  environment/container identity, device, and resolved site profile.
- [ ] Keep the first implementation compatible with atomic `.npz` writes if
  possible; do not introduce a larger storage system before it is needed.

### Strengthen artifact writes and reads

- [ ] Make `write_result` and `write_npz` validate required protocol fields at
  the writer seam; a merely non-empty dictionary is not enough.
- [ ] Make `CaptureSpec.validate_capture()` check declared dtype as well as
  shape.
- [ ] Compare an artifact's embedded capture specification with the spec used
  by the caller; refuse semantic disagreement even when shapes happen to fit.
- [ ] On load, cross-check every embedded array descriptor against the actual
  array shape and dtype.
- [ ] Refuse metadata entries for missing arrays and unexpected arrays unless
  the schema explicitly marks them optional.
- [ ] Add a context manager or `close()` method to `Capture` so repeated reads
  do not retain open `NpzFile` handles.
- [ ] Keep legacy captures readable through an explicit compatibility path,
  while requiring complete metadata for new scientific results.
- [ ] Add corruption/refusal tests for stale shape metadata, stale dtype
  metadata, changed capture specs, incomplete protocols, and truncated writes.

Acceptance criteria:

- Editing an array's dtype, layer count, or metadata independently causes a
  load failure.
- A new artifact can explain what model and exact inputs produced every row.
- A completed artifact is atomically visible; partial output is never accepted
  as resumable work.

## P1: complete the input reproduction contract

- [ ] Introduce the planned per-model input/materialization manifest rather
  than relying only on assay-level CSV and MSA paths.
- [ ] Decide whether model inputs live in a separate `model_inputs.yaml` or as
  structured entries referenced from each cohort. Document the choice.
- [ ] Replace committed user-specific absolute paths with logical roots such as
  `${PROT_INTERP_DATA}` or site-profile resolution.
- [ ] Add controlled variable expansion to cohort/input loading; refuse unknown
  variables instead of leaving them as literal paths.
- [ ] Verify checksums before materialization.
- [ ] Record checksums after materialization so generated variant A3Ms and YAML
  files become part of the run identity.
- [ ] Verify expected MSA rows, sequence length, token count, chain mapping,
  residue-token mapping, model grid, and representation width.
- [ ] Ensure feature generation never silently falls back to a remote MSA or
  regenerates a missing input in reproduction mode.
- [x] Use job-local, run-unique work directories so concurrent jobs cannot read
  each other's generated alignment or YAML.
- [ ] Add refusal tests for changed MSAs, changed assay CSVs, mapping mismatches,
  MSA-depth mismatches, missing logical roots, and shared-work-directory
  collisions.

Acceptance criteria:

- A live job stops before model execution if any declared scientific input has
  moved.
- The artifact records both source inputs and the exact model-specific files
  consumed.
- The same manifest can resolve under a different user/site profile without
  editing tracked YAML.

## P2: add an artifact-native analysis dataset

### Implement `RepresentationDataset`

- [ ] Add `analysis/datasets.py`; it must import no backend.
- [ ] Open one artifact or a directory/index of per-assay artifacts.
- [ ] Require an expected cohort/task by default and refuse missing, duplicate,
  or unexpected assay records.
- [ ] Validate schema versions and compatible capture semantics across all
  constituent artifacts.
- [ ] Expose fields and coordinates without callers constructing filename
  conventions.
- [ ] Provide explicit accessors for vectors versus magnitudes so a norm cannot
  masquerade as a direction.
- [ ] Provide targets and identity columns (`assay`, `input_id`, `reference_id`,
  `mutant`, `position`, layer index) with alignment checked centrally.
- [ ] Add `difference_from_reference()` only where the artifact contains enough
  identity to pair rows safely.
- [ ] Preserve per-assay grouping for downstream bootstrap and transfer probes.
- [ ] Allow deliberately partial datasets only through an explicit option that
  records omissions.
- [ ] Add synthetic tests covering missing references, duplicated variants,
  mismatched widths, layer mismatches, and incomplete cohorts.

Target surface:

```python
reps = RepresentationDataset.open(
    "runs/boltz2_pair_layers",
    cohort=Cohort.load("basis_assays"),
)

dz = reps.vectors("dz_site", layer="final")
y = reps.target("score")
per_assay = reps.group_by_assay(dz, y)
result = leave_one_group_out(per_assay, lam=10.0)
```

### Make analysis results easy and safe to save

- [ ] Add a small result helper that derives source artifact IDs/checksums,
  cohort size, layers, and feature descriptors from the dataset.
- [ ] Keep the scientific protocol visible in the experiment script: design,
  statistic, normalization, split unit, permutation/bootstrap unit, seeds, and
  analysis-specific parameters must remain explicit.
- [ ] Do not make the helper infer scientific choices it cannot know.
- [ ] Write through `artifacts.write_result` and include source artifact hashes,
  not only a filename pattern.
- [ ] Add a read-side `ResultArtifact` validation path for report producers.

Acceptance criteria:

- A short analysis script contains the statistic and protocol choices, not
  file-globbing and row-alignment plumbing.
- Every output can identify the exact capture artifacts it analyzed.

## P2: finish the command-line lifecycle

Implement these after `CollectionTask` is stable so the CLI delegates to the
same objects as Python rather than creating a second execution system.

- [ ] `pi collect inspect EXPERIMENT.py`: resolve task, verify inputs and
  capabilities, display selected layers and resource/storage estimates, and
  load no backend.
- [ ] `pi collect render EXPERIMENT.py`: render the exact resolved task and site
  configuration used by the compute job.
- [ ] `pi collect submit EXPERIMENT.py`: submit the rendered checkout job and
  record the task ID/output location.
- [ ] `pi collect run EXPERIMENT.py`: execute only on an allowed compute node,
  unless an explicit development override is present.
- [ ] `pi analyze run EXPERIMENT.py`: execute artifact-only analysis without a
  model environment.
- [ ] Implement resume only after validating the completed artifact's task ID,
  metadata, input hashes, and arrays.
- [ ] Refuse overwriting an artifact produced by a different resolved task
  unless the user explicitly chooses a new output or replacement policy.
- [ ] Keep the existing `pi reproduce`, `pi verify`, `pi inspect`, `pi cohort`,
  `pi site`, `pi render`, and `pi models` behavior compatible during migration.

Acceptance criteria:

- Python and CLI paths resolve to the same serialized task.
- Inspection, rendering, and submission do not initialize CUDA/JAX or import a
  model backend.

## P3: migrate additional models through the same workflow

### OpenFold3

- [ ] Implement the adapter using the Mosaic wrapper with lazy imports.
- [ ] Record its per-atom pLDDT semantics explicitly.
- [ ] Measure and record its real distogram grid before allowing binwise
  cross-model comparisons.
- [ ] Verify trunk depth and widths against the loaded model.
- [ ] Implement requested-layer pair capture with the same artifact schema.
- [ ] Verify exact local-MSA injection and network blocking.
- [ ] Compare old and new captures on the cross-model cohort.

### Protenix

- [ ] Implement the adapter using the Mosaic wrapper with lazy imports.
- [ ] Preserve the wrapper's distinct inner-network layout rather than assuming
  the Boltz-2/OpenFold3 attribute path.
- [ ] Measure and record its distogram grid.
- [ ] Verify trunk depth and widths against the loaded model.
- [ ] Implement requested-layer pair capture with the same artifact schema.
- [ ] Compare old and new captures on the cross-model cohort.

### Cross-model comparison gate

- [ ] Require identical logical cohort members and variant identities.
- [ ] Verify actual sequences, MSAs, mappings, and operating regimes consumed by
  each model.
- [ ] Compare at declared matched relative depths rather than assuming absolute
  layer numbers correspond.
- [ ] Call `records.assert_comparable` before binwise output comparisons.
- [ ] Refuse comparisons whose representation semantics cannot be normalized
  honestly.

### AlphaFold2 and ESMFold

- [ ] Keep the current audited exclusions visible.
- [ ] Revisit AlphaFold2 only after choosing a comparable single-sequence/MSA
  operating point or obtaining an MSA-capable wrapper.
- [ ] Add ESMFold only after defining whether “layers” means encoder layers,
  folding-trunk layers, or both.
- [ ] Do not add either model to the supported registry until a real capture
  validates its declared capabilities.

## P3: prediction and intervention paths

- [ ] Define a normalized prediction artifact separately from representation
  captures.
- [ ] Record architecture-specific output semantics and grids.
- [ ] Add a `Predictor`/prediction task only after common outputs and refusal
  conditions are explicit.
- [ ] Keep intervention definitions separate from measurements collected after
  them.
- [ ] Connect `PairDirectionIntervention` to the live Boltz-2 adapter without
  moving its numpy-testable algebra into backend code.
- [ ] Store intervention direction artifact identity, component, scale, layers,
  mode, doses, sites, controls, and seeds.
- [ ] Preserve and test the current symmetric diagonal behavior.
- [ ] Require alpha/dose zero as the deterministic baseline check.
- [ ] Migrate `steer_pooled` only after old/new response metrics agree under the
  recorded protocol.

## P3: scientific regression and migration accounting

### Regression checks

- [ ] Keep `uv run pytest tests/ -q` green after every library change.
- [ ] Run harness self-tests directly where applicable.
- [ ] For changes under `src/`, rerun all 17 report producers from
  `prot_interp_files/runs/check_20260817/` through `checkout.sbatch`.
- [ ] Diff against the archived results and against a second unchanged run.
- [ ] Apply only the measured instability bands documented in `AGENTS.md`.
- [ ] Record commands, commits, scheduler jobs, tolerances, and comparison
  outputs in the migration ledger.
- [ ] Add small compute integration tests for each supported adapter: short
  sequence, minimal recycles, selected layers, fixed local input, network off.

### Migration ledger and archive

- [ ] Create `MIGRATION_LEDGER.csv` with legacy path/commit/hash, scientific
  role, inputs/outputs, replacement, validation command/result, disposition,
  and archive location/checksum.
- [ ] Account for every active `exp_*`, `analyze_*`, `probe_*`, `fig_*`, report,
  launcher, and scheduler file.
- [ ] Create the external timestamped pre-refactor code archive required by the
  plan.
- [ ] Include source, commit, environments, scheduler files, README, ledger,
  and SHA-256 checksums.
- [ ] Treat the archive as immutable after verification.
- [ ] Do not treat “recoverable with `git show`” as a substitute for the
  explicit external archive.
- [ ] Remove additional legacy files only after their ledger disposition and
  archive checksum are recorded.

## Documentation cleanup

- [ ] Update `docs/API.md` when the executable task/adapter/dataset API lands.
- [ ] Clearly label examples in `docs/refactoring_plan.html` as target API until
  they are importable.
- [ ] Ensure the supported-model statement agrees across `docs/API.md`,
  `docs/MODEL_AUDIT.md`, and the capability registry.
- [ ] Document the artifact schema and compatibility policy.
- [ ] Document how logical data roots resolve through the site profile.
- [ ] Add one runnable collection example and one runnable analysis example
  that form a complete chain.
- [ ] Correct package metadata (`pyproject.toml` description/version versus
  `protein_interpretability.__version__`) when choosing the first stable API
  release.

## Suggested implementation sequence

Each item should be a focused change with its own tests and validation record.

1. [x] Correct `kl_site` and add its refusal/regression tests.
2. [x] Make missing cohort artifacts a refusal by default.
3. [ ] Strengthen embedded artifact descriptor and protocol validation.
4. [ ] Introduce `ModelSpec`, `CollectionTask`, and resolved task serialization.
5. [ ] Introduce the adapter protocol and Boltz-2 adapter.
6. [ ] Connect real selected-layer resolution to the Boltz-2 runner.
7. [ ] Complete per-model input identity and portable path resolution.
8. [ ] Implement `RepresentationDataset` with cohort completeness checks.
9. [ ] Rewrite the pair-layer collection and analysis examples on the public
   API and repeat old/new validation.
10. [ ] Add the collection/analyze CLI lifecycle as a thin wrapper over those
    same task objects.
11. [ ] Migrate OpenFold3 and Protenix one at a time.
12. [ ] Migrate prediction and intervention workflows.
13. [ ] Complete the migration ledger, immutable archive, 17-producer
    regression record, and legacy removal gate.

## Definition of done for the requested workflow

- [ ] A user can select a checksummed task/cohort and supported model from a
  short Python file.
- [ ] A user can request named fields from `"all"`, `"final"`, or an explicit
  set of validated layers.
- [ ] The same file can be inspected and rendered on a login node without
  importing a model backend.
- [ ] The compute job records the actual model, checkpoint, inputs, mappings,
  seeds, layers, dtypes, and environment it consumed.
- [ ] The capture is atomic, versioned, self-describing, and validated on read.
- [ ] Offline analysis opens captures without filename/glob conventions and
  refuses incomplete cohorts or incompatible semantics.
- [ ] Statistics and metrics remain explicit in a short analysis program.
- [ ] Results are atomically saved with complete protocol, provenance, and
  source artifact identities.
- [ ] Boltz-2, OpenFold3, and Protenix pass their declared old-versus-new
  scientific equivalence gates.
- [ ] The 17 report producers remain within exact or measured same-code
  tolerances.
