# Mutation signal: spatial localisation, depth, and causal use

**Status:** proposed experiment specification, not a record of completed runs.
**Purpose:** implementation and launch handoff for a bounded extension of the paper.

Start with Boltz-2. Reuse the existing cohorts, collection infrastructure, and
analysis conventions; add other models only after the within-model experiment
answers its question. This document specifies tests and decision gates. It does
not submit jobs or imply that the proposed capture fields and interventions are
already implemented.

## 1. Questions and claim boundaries

Distinguish three questions throughout the figures and result artifacts:

1. **Sensitivity:** where does changing the sequence change an activation?
2. **Accessibility:** where can a fixed-capacity readout predict experimental
   mutation effects on held-out proteins?
3. **Causal use:** which interventions change a specified downstream readout,
   denoiser response, or emitted structural endpoint?

A nonzero mutant-minus-WT activation difference is not necessarily a biological
signal. A predictive probe is not evidence that the native decoder uses its
feature. Lower prediction performance establishes reduced accessibility under
that readout, not information erasure.

The existing [audit response](../../prot_interp_files/audit_response_20260906/REPORT.md),
under “tracing the signal to the decoder entrance,” reports that the paired
trunk-minus-conditioning probe gap at k=32 was −0.005 [−0.037, +0.027]. Use that
as motivation to investigate decoder response, not as proof of equivalence or
of where information is destroyed. The same response documents the negative
natural-PC2-deletion result and withdrawal of the old steering significance.
This campaign does not assume PC2 is a stability-control knob.

**Target contribution:** identify where context-dependent mutation effects become
readable, test which spatial computation constructs that readout, and separately
test whether the relevant conditioning changes influence denoising.

## 2. Freeze the experiment before production collection

Use development proteins for mask choices, precision estimates, layer selection,
and debugging. Freeze those decisions before evaluating confirmation proteins.
Previously inspected held-out cohorts can support exploratory reuse, but are not
an untouched confirmation set for decisions informed by their results.

Record the following in the campaign specification:

- Exact cohort, assay and variant IDs, input checksums, exclusions, and phenotype
  orientation. Primary analysis: stability assays; other phenotypes reported
  separately. Check protein/family overlap as well as assay-ID overlap.
- Development/confirmation roles and a fixed variant selection rule. Include
  tolerated and damaging variants without selecting on new intervention success.
- Model/checkpoint, MSA contents and regime, recycles, numerical precision,
  sampler schedule, and separate random seeds for each stochastic source.
- Reference structure, contact definition, distance threshold, treatment of
  sequence neighbours, and policy for sites with no eligible contacts.
- Feature reductions, dimensionality, readout fitting/tuning, primary contrasts,
  resampling unit, practical effect threshold, and interval precision target.
- Pilot repeat count and a bounded rule for increasing it. A reasonable starting
  point is eight independent diffusion draws per selected WT and mutant; this is
  a pilot allocation, not a claim that eight draws establish a null.

Use a fixed MSA/full regime for the primary isolation experiment. Treat MSA
subsampling as a separate sensitivity condition. Keep WT and mutant homolog rows
aligned and document which query-dependent inputs change.

**Gate:** the above choices are recorded before confirmation results are read.
A pilot may change the protocol; freeze a new version and preserve the old one.

## 3. Test N: establish the appropriate noise floors

### N1. Numerical and capture fidelity

**Run:** repeat the same input with identical randomness; compare the instrumented
forward with the native forward. Verify the selected intermediate capture and
resume boundaries, not just final array shapes.

**Measure:** absolute and relative activation drift, native output drift, and
same-state/no-op intervention drift in the metrics used below.

**Acceptance:** differences stay inside an empirically established numerical
band. Establish that band with unchanged code. A mismatch blocks scientific
interpretation of later interventions until its source is fixed.

### N2. Trunk stochasticity

**Run:** repeat WT and selected mutants with only the MSA subsampling key changed,
if subsampling is being studied. Hold all other inputs and settings fixed.

**Measure:** within-sequence variation in each proposed spatial/depth feature;
compare against mutant-minus-WT differences under the same regime.

**Deliverable:** a separate trunk-variability estimate, not folded into the
fixed-conditioning diffusion floor. A deterministic full-MSA primary run should
still receive the N1 fidelity check.

### N3. Decoder stochasticity

**Run:** hold each sequence's trunk and conditioning fixed and sample independent
diffusion draws. Repeat both WT and selected mutants.

**Measure:** WT–WT, mutant–mutant, and WT–mutant distributions for prespecified
structural features and trajectory summaries. Use aligned coordinates or
rigid-motion-invariant distances. Report location and dispersion effects
separately; a mutation can affect either.

**Report:** absolute effect, within-input spread, and uncertainty. An effect/spread
ratio is descriptive and is undefined or unstable when the floor is near zero.
A mean effect smaller than single-draw spread can still be resolved with repeats.
Failure to exceed mean WT–WT distance is not a valid universal detection rule.

Resample independent draws, then proteins/assays as appropriate. Pairwise distances
sharing a draw are dependent; eight draws do not yield 28 independent replicates.
Preserve shared-WT and shared-seed dependencies in paired contrasts.

**Gate:** the pilot establishes metric-specific uncertainty and a production
repeat allocation. If uncertainty remains too large to exclude a meaningful
effect, report “unresolved at this precision,” not “no mutation signal.”

**Atom correspondence:** equal random seeds do not couple all-atom WT and mutant
noise when substitutions change atom count or indexing. Natural WT/mutant runs
are independent unless a validated atom-correspondence coupling is implemented.
The fixed-recipient conditioning intervention in Test C avoids that requirement.

## 4. Test S: spatially resolved pair features

For mutation site m and block l, define:

    delta_z_l[i,j,:] = z_mut_l[i,j,:] - z_wt_l[i,j,:]

Confirm token correspondence before subtraction. Capture channel vectors, not
only their norms. The input state before the first Pairformer block is a distinct
capture boundary, not an assumed existing block index.

Define N(m) once from the WT reference structure. Keep the same mask for WT,
mutant, all layers, and all interventions. Record the reference and the actual
mask, so the analysis cannot silently change when a structure is regenerated.

| Feature family | Entries and reduction | Question |
|---|---|---|
| Global row | Mean delta_z_l[m,j,:] over valid j other than m | Existing global information |
| Global column | Mean delta_z_l[i,m,:] over valid i other than m | Directional asymmetry |
| Contact row/column | Same reductions restricted to N(m), separately | Local-context concentration |
| Matched non-contact row/column | Same number of non-contact partners, with a fixed sampling rule | Locality versus pooling size/noise |
| Distal–distal | Mean over i != j, with both i and j outside N(m) and neither equal to m | Information away from the mutation neighbourhood |

Retain the exact legacy global feature as a separately labelled comparator if its
diagonal convention differs. Do not silently redefine an archived feature.
Keep row and column separate initially; their concatenation is a secondary
feature with explicitly matched readout capacity. The pair tensor need not be
symmetric. Off-diagonal alone does not mean distant from the mutation.

Match non-contact controls for partner count and, where feasible, sequence
separation. Record any matching failures. Use a prespecified exclusion or
missing-feature policy for empty neighbourhoods; never silently switch to global
pooling. Keep comparisons on identical eligible variants.

**Deliverable:** spatial feature vectors at every selected boundary, masks and
counts, and a contact-versus-matched-non-contact comparison. Norms can diagnose
sensitivity/noise but do not replace the biological readout.

**Interpretation:** a positive distal result establishes distributed predictive
representation, not physical allostery. A negative pooled result does not exclude
information in individual entries that cancel under averaging.

## 5. Test D: depth-resolved experimental-effect prediction

**Run:** fit the same linear-probe procedure separately at each block and at the
pre-Pairformer input boundary. Use identical proteins, variants, folds, phenotype
orientation, and fitting rules across depths and spatial regions.

- Use protein/assay-held-out evaluation; use family-grouped sensitivity where
  relevant. Keep development selection separate from confirmation.
- Fit scaling, PCA, and any tuning on training proteins only. Use an intercept
  and one frozen regularisation rule or identical nested training-only tuning.
- Match readout capacity across feature families, especially row+column versus
  row alone. Retain full-width results as secondary context.
- Fit separate coefficients at each depth. A fixed coefficient vector across
  layers confounds predictive content with rotation of the representation.
- State recycle identity explicitly. Prefer all blocks of one specified recycle
  for the primary curve; do not merge recycle and block indices ambiguously.
- Compute within-assay Spearman on held-out predictions, then the prespecified
  assay aggregate. Report paired uncertainty across proteins/assays.
- Run a label-randomisation control through the fitting procedure, respecting
  the chosen assay/group structure. Keep masks and selection rules frozen.

**Primary figure:** correlation versus block depth, including the input boundary;
curves for global, contact-local, matched non-contact, and distal features. Show
paired contrasts and uncertainty rather than highlighting the maximum layer.
If testing an entire curve, use simultaneous uncertainty or a prespecified
aggregate contrast; pointwise intervals do not correct a search across layers.

### D-context. Is the signal more than substitution identity?

Reuse the project's specified chemistry/MSA/burial baseline. Compare baseline
against baseline plus the spatial internal feature on the same held-out rows.
Add within-substitution comparisons where there are enough repeated substitution
types across contexts; report eligibility and precision rather than pooling
incomparable small groups.

**Deliverable:** the depth curves, input-to-later-depth contrasts, context
increments, and a small development-selected set of patch sites/layers frozen
before confirmation. A rising curve means increasing accessibility, not by
itself the creation of information at that block.

## 6. Test T: trunk activation patching

**Question:** which spatial computation constructs the downstream predictive
feature? This endpoint is distinct from native decoder use.

1. Cache corresponding natural WT and mutant states at the selected boundaries.
2. Run the mutant recipient, replace one selected activation region with its WT
   donor counterpart, and resume the unmodified remaining trunk computation.
3. Reverse donor and recipient for the complementary intervention.
4. Compare contact-local, size-matched non-contact, and selected distal patches.
   Record the exact tensors replaced and the single/pair states held fixed.
5. Evaluate a frozen final-layer readout trained without the evaluated protein.
   Keep its WT reference, scaling, basis, and coefficients fixed across patches.

**Controls:** same-state patch; empty patch; matched control region; reverse
patch; and continuation fidelity. A complete-state replacement should reproduce
the donor continuation only when all relevant state and downstream inputs also
match. Partial patches have no such endpoint-equality guarantee. Handle overlap
between row and column masks once, with explicit diagonal treatment.

**Primary endpoint:** signed movement of the fixed downstream prediction toward
the donor's prediction, compared with matched controls. Also report absolute
changes and generic-disruption measures. Report normalised recovery only where
the unpatched donor–recipient gap is sufficiently resolved; keep small-gap cases
in absolute-effect summaries.

Inspect activation scale, finite values, structural validity, and broader output
disruption. A hybrid natural-donor patch can still be off-distribution. A broad
collapse is not selective removal of a biological feature.

**Deliverable:** a region-by-layer causal-effect plot on confirmation proteins,
with bidirectional effects and matched controls. Success is reproducible
selectivity, not merely any changed probe score.

Removing a final probe direction and observing that same probe vanish is algebra,
not this test. Here the intervention is upstream and the readout is independent
and fixed. Even a selective positive result establishes construction of a
predictive representation, not control of folding stability.

## 7. Test C: conditioning response and paired trajectories

### C1. Fixed-state denoiser response

Start with a token-indexed block such as `token_trans_bias`. Hold the recipient's
atom layout, masks, remaining conditioning, noisy coordinates x, noise level
sigma, and model randomness fixed. Replace the selected conditioning block or
region with the corresponding WT/mutant donor values.

Evaluate both branches at the identical state:

    delta_D(x, sigma) = D(x, sigma; c_patched) - D(x, sigma; c_original)

Here D is the model's denoised-coordinate prediction. Report displacement first.
Convert it to a diffusion score only after establishing the actual model's
preconditioning, noise convention, units, and alignment behaviour; the sampler's
update direction is not automatically the score.

Evaluate several prespecified noise levels and states from independent recipient
trajectories. Freeze state selection before reading patch responses. Use both WT
and mutant recipient backgrounds, each retaining its own atom layout.

**Controls:** same conditioning twice, same-state patch, matched non-contact
conditioning patch, and reverse donor direction. Use identical rigid-frame
handling and stochastic augmentation within each branch pair.

**Endpoints:** per-atom/backbone response magnitude and localisation, with a
prespecified directional structural metric where justified. Compare conditions
at fixed sigma; raw score norms at different sigma have different scaling.

Natural conditioning swaps test mutation-related conditioning influence. They do
not isolate experimental stability. Relate response to the frozen biological
readout and assay target, with substitution/context controls, before calling the
response stability-related.

Atom-indexed q/c and windowed biases require explicit correspondence and validated
intervention semantics before use. Equal tensor shape is insufficient. Token
aggregation used for a probe is not automatically an invertible decoder patch.
A pair-bias-only swap is a defined hybrid intervention, not a full WT/MT exchange.

**Deliverable:** response-versus-noise-level curves at identical input states,
with numerical baseline and matched-region contrasts.

### C2. Paired trajectory branching

From the same recipient state, branch original and patched runs with identical
future random draws, atom layout, and augmentation. Specify whether the patch is
applied for one denoising evaluation or held for the remaining trajectory; these
are different interventions. Use persistent conditioning as the primary branch
comparison if the question is its cumulative influence.

Record the actual noise level, raw noisy state, denoised estimate, and aligned or
rigid-invariant divergence at selected steps. Repeat across independent starting
trajectories. Analyse divergence, final structural features, confidence, and
structural-quality checks separately.

Compare against no-op paired branches and the independent-draw output spread
from N3. Independent-draw spread contextualises practical magnitude; it is not
the significance threshold for a controlled paired intervention.

**Deliverable:** paired trajectories and endpoint effects with seed-level and
protein-level uncertainty. Reconvergence shows attenuation in the measured
metric, not thermodynamic attraction, information erasure, or necessarily the
absence of a score-field response.

**Gate:** carry an intervention into large trajectory collection only after
fixed-state fidelity is established and its scientific contrast is specified.
A well-resolved negative C1 result is useful; an imprecise C1 result needs a
precision decision, not a claim that the conditioning is ignored.

## 8. Execution order and stopping rules

| Stage | Run | Completion criterion |
|---|---|---|
| Specification | Freeze cohorts, masks, fitting, endpoints, seeds, precision targets | Versioned campaign specification and explicit development/confirmation roles |
| Noise pilot | N1–N3 | Validated capture/continuation and a defensible repeat allocation |
| Observational collection | S and D, including D-context | Comparable spatial depth curves and frozen patch hypotheses |
| Trunk causality | T | Controlled downstream effects or a precision-bounded null |
| Decoder causality | C1 | Valid fixed-state response contrasts or a precision-bounded null |
| Trajectory extension | C2 when warranted by C1/design | Paired trajectories, endpoint effects, and quality controls |
| Evidence freeze | Assemble artifacts and figures | Every claim tied to its protocol, cohort, uncertainty, and control |

After observational collection, T and C1 can proceed independently. Neither needs
a positive result from the other. A negative native-decoder result does not
invalidate a reproducible account of how the predictive representation forms.

Stop expanding after the frozen confirmation panel answers the primary contrasts
at the specified precision, including negative answers. A change of hypothesis
or endpoints becomes a separately labelled exploratory analysis. A statistically
unresolved result remains unresolved; do not increase doses or search regions
until an output moves.

Defer sparse autoencoders, another architecture, broad head searches, and a
compensatory-double-mutant campaign. Epistasis is a worthwhile later direction
if suitable experimental data exist, but is not a prerequisite for this plan.

## 9. Implementation and launch handoff

Read [API.md](API.md) before adding scripts. Existing entry points provide pieces,
not a turnkey implementation of this specification:

| Existing file | Reuse | Required extension or caution |
|---|---|---|
| [collect_pairformer_layers.py](../experiments/collection/collect_pairformer_layers.py) | Collection pattern | Check support for input-boundary and spatial vector reductions before declaring them |
| [exp_gym_deep.py](../jax_harness/exp_gym_deep.py) | Layer capture and fidelity checks | Existing per-layer magnitudes are not the proposed spatial channel vectors |
| [exp_ensemble.py](../jax_harness/exp_ensemble.py) | Fixed-trunk repeated diffusion collection | Preserve separate WT and mutant variability and correct seed semantics |
| [exp_conditioning.py](../jax_harness/exp_conditioning.py) | Conditioning extraction and token correspondence | Captures are not conditioning interventions |
| [probe_conditioning.py](../jax_harness/probe_conditioning.py) | Shape/axis inspection | Validate any additional decoder-side tensor before swapping |
| [exp_paths.py](../jax_harness/exp_paths.py), [pi_paths.py](../jax_harness/pi_paths.py) | Natural donor/recipient input-route patching | Extend at actual intermediate boundaries; existing route patches are not layer-local patching |
| [exp_trajectory.py](../jax_harness/exp_trajectory.py) | Trajectory capture and aligned repeat comparisons | Replace the “must exceed WT floor” detection rule; add fixed-recipient branching and fixed-state denoiser evaluations |

Before production submission:

- Verify manifests and checksums; inspect existing captures for vector-versus-norm
  semantics and boundary coverage. Reuse valid data and collect only missing
  measurements. Name all exclusions and preserve row identity across analyses.
- Implement missing collection/intervention support in the existing package
  structure. Keep backend loading in collection; analysis reads artifacts.
  Add no new `pi_*` module and no alternate result-writing path.
- Reduce spatial features during collection where practical instead of archiving
  every full pair tensor for every variant and block. Retain the exact selected
  states required for patches and the masks needed to reproduce reductions.
- Write results through the sanctioned artifact/protocol seam. Record feature
  formulas, dimensions, layer/recycle convention, WT reference, patch masks,
  donor/recipient IDs, state/noise IDs, pairing semantics, seeds, and exclusions
  in addition to automatic source provenance.
- Exercise one real forward, no-op patch, donor patch, and paired diffusion branch
  on a GPU before scaling up. Keep regression guards for correspondence refusals,
  invalid masks, and other scientifically dangerous silent failures.
- If library code changes, perform the repository-required scientific regression
  against the archived report producers, allowing only established numerical
  variation. Diagnose unexpected drift by rerunning unchanged code.

Model jobs use [checkout.sbatch](../jax_harness/checkout.sbatch), not the deployed
mirror: it executes scripts from the checkout's `jax_harness/`. Set job resources
from the pilot rather than assuming its default time limit is adequate. Model
loading never runs on the login node. Keep large captures under the external
`prot_interp_files/` data root, not in Git.

**Launch deliverable for implementation:** provide the exact executable commands
and manifests for the noise pilot first; after its gate passes, provide production
commands for each remaining stage. Record argv in each result artifact. There is
intentionally no fabricated one-command launcher here for features not yet built.

## 10. Expected paper outputs

1. **Noise panel:** numerical drift, trunk stochasticity, and decoder spread as
   distinct quantities; effect estimates with uncertainty.
2. **Spatial depth figure:** input plus all selected blocks, contact/global/distal
   curves, matched controls, and context-baseline increments.
3. **Trunk patch figure:** where natural donor patches selectively change a frozen
   downstream biological readout.
4. **Decoder figure:** fixed-state denoiser response and, where run, paired
   trajectory persistence/attenuation and final structural effects.

A positive result can support a spatial account of predictive-feature construction
and a separately measured decoder response. A negative decoder result can support
limited use under the tested interventions and precision. Neither outcome alone
establishes a thermodynamic mechanism or absence of mutation information from the
complete output distribution.

Methodological references: [activation-patching guidance](https://arxiv.org/abs/2404.15255)
for intervention interpretation; [causal tracing in language models](https://arxiv.org/abs/2202.05262)
for the donor/recipient localisation idea. These motivate the method, not claims
about protein-model results.
