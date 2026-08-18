# Model audit: what is supported, and what is excluded and why

§10 of the plan asks that AlphaFold2 and ESMFold be *either* supported through
the same contracts *or* explicitly documented as audited exclusions with
reasons. This is that document. It records the state as of 2026-08-18 and the
evidence behind each line, so a later reader can tell a decision from an
oversight.

## Supported

| model | trunk blocks | pair / single | pLDDT | distogram grid | verified |
|---|---|---|---|---|---|
| Boltz-2 | 64 | 128 / 384 | per token | 2–22 Å, 64 bins, centres 2.15625…21.84375 | against a loaded model |
| OpenFold3 | 48 | 128 / 384 | **per atom** | not recorded — read it from the model | against a loaded model |
| Protenix | 16 | 128 / 384 | per token | not recorded — read it from the model | against a loaded model |

All three resolve through `pi_models.load(name, msa=...)`, produce the
normalised `Extraction` record that `collection/records.py` validates, and have
their declared trunk depth confirmed against a real loaded model by
`jax_harness/probe_capabilities.py` (`capabilities.json`: `all_verified: true`).

Two differences are recorded as properties rather than smoothed away, because
smoothing them is how a wrong number gets produced quietly:

- **pLDDT granularity.** Per-atom in OpenFold3, per-token in the other two. The
  arrays have different lengths for the same protein.
- **Distogram grids.** Only Boltz-2's is recorded here. A KL computed across two
  different distance grids is a well-formed number that means nothing, so
  cross-model comparisons go through `records.assert_comparable`, which checks
  the bin *grid* and not merely the bin *count*.

## Excluded, with reasons

### AlphaFold2 — audited, partially supported, not usable for the current results

`pi_models.load("af2")` builds, and the mosaic wrapper is present. It is
excluded from every result in the report for one reason that is a property of
the wrapper, not a gap in this project:

> **It is single-sequence only.** `mosaic.models.af2` asserts `not use_msa` and
> pins `max_msa_clusters=1`.

Every result here is computed at full MSA depth. Single-sequence input is a
different operating point, not a variant of the same one — measured ~4.4×
more mutation-sensitive at full depth — so an AF2 row would not be comparable
with the other three. Including it in a cross-model panel would produce a
number that looks like a fourth model and is answering a different question.

`capabilities.check_msa("af2", use_msa=True)` raises with that reasoning rather
than letting it happen.

Its trunk depth and widths are **not recorded**, because no capture in this
project has ever read them. The registry carries `None` and raises when asked,
rather than a plausible default: a guessed depth silently changes which layer
"final" means.

**To support it properly** would need: a decision that single-sequence is the
comparison point for *all* models, or a wrapper that accepts an alignment; then
its depth and grid measured from a real run and recorded.

### ESMFold — no code, no adapter, out of scope

ESMFold has **no implementation in this repository**. The extractor that existed
belonged to the ESM/Boltz-torch generation removed on 2026-08-17 (`git show
fcceb3f:src/protein_interpretability/extractor_esmfold.py`), and nothing in the
current harness references it. `docs/esmfold_extraction_sites.md` and
`docs/esmfold_extraction_howto.md` are retained as notes from that work; they
describe code that is no longer here.

It is not in `capabilities.REGISTRY` at all, so asking for it raises `unknown
model` rather than half-resolving.

**To support it** would need a Transformers/PyTorch adapter producing the same
normalised record, and a decision about what its "trunk" means — ESM encoder
layers and the folding trunk are different objects from a Pairformer stack, so
`n_trunk_blocks` and "matched relative depth" would need defining before a
cross-model comparison could be honest.

## What would change this document

- AF2 gaining an MSA path, or the project adopting single-sequence as the shared
  operating point.
- An ESMFold adapter, with its depth semantics decided.
- Any of the "not recorded" grids being measured — record them in
  `capabilities.REGISTRY` with the evidence, and `probe_capabilities.py` will
  check them against the model from then on.
