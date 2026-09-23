# Publication audit and finishing plan

> **Follow-up correction — this is the original audit, not the current acceptance checklist.**
> The subsequent [audit response](../../prot_interp_files/audit_response_20260906/REPORT.md)
> and [claim ledger](../../prot_interp_files/audit_response_20260906/CLAIM_LEDGER.md)
> supersede resolved findings and proposed gates below. I withdraw the OpenFold3
> per-atom pLDDT finding: the wrapper already returned per-token confidence; my
> premise came from stale documentation, not an error in the captures. The
> unaligned 12–24 Å coordinate comparison is also not evidence of degenerate
> structures; the separate question of poor aligned archived Boltz-2 structures
> remains open. The follow-up reports a residual internal-versus-four-draw-output
> gap on eight Boltz-2 assays, frozen-PC2 transfer concentrated in stability
> assays, and incremental prediction beyond the specified chemistry/MSA/burial
> baseline. These strengthen the observational case. The old steering
> significance is withdrawn, and the existing natural-mutation ablation is
> negative; neither establishes a selective stability-control mechanism.

## Verdict

**Yes: this is worth writing up. The strongest paper is about experimentally anchored, transferable mutation-effect decodability in frozen folding-model representations—not a general claim that folding models ignore mutations, a new state-of-the-art stability predictor, or a demonstrated thermodynamic mechanism.**

My assessment: there is a credible computational biology / representation-analysis paper here. Its publication strength depends more on precise claims, a defensible output comparator, and a small number of decisive controls than on adding more architectures or interpretability machinery. Acceptance or novelty cannot be guaranteed by this focused audit.

Recommended central claim:

> Across Boltz-2, OpenFold3, and Protenix, mutation-induced changes in final Pairformer representations support protein-held-out prediction of experimental mutation effects better than prespecified summaries of sampled C-alpha geometry and confidence. Much of this predictive performance is accessible in a low-dimensional subspace. In Boltz-2, a frozen direction associated with stability also modulates distogram width and confidence, without establishing control of folding stability or conformation.

The last sentence should remain secondary and explicitly include the negative ablation evidence. The three models are separately trained members of an AF3-derived architectural family; this is not three independent architectural families.

## Audit scope and verification

Reviewed the supplied audit index, master report, SVD audit trail, mechanism and both Jacobian reports, original report overview/corrections/open questions and relevant methods/results sections, both August cross-model results pages, publication plan, current evidence review, and prior independent audit. Followed key claims into the checkout's source and authoritative JSON/NPZ archives. Conducted a focused primary-literature comparison, including relevant 2026 work.

**Executed, without loading any folding model:**

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python experiments/analysis/reproduce_headline_transfer.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python experiments/analysis/reproduce_xmodel_transfer.py --all
```

- Original headline: mean Spearman **0.757827**, worst per-assay archive difference **0.00e+00**.
- Heldout16: both the 128-channel internal predictor and the ten-feature output predictor reproduced in **all three models**, worst per-assay archive difference **0.00e+00** in each.
- Independently recomputed the six two-PC-minus-geometry mean gaps and win counts from archived per-assay values; they agree with the JSON summaries.
- Reconstructed all eight steering control vectors in each of twelve NPZs from the same seed-0 standardized random draws, allowing for assay-specific channel scaling. Worst coordinate residual: **2.02e-14**. This reveals a dependence problem in the reported steering null; see below.
- Inspected signed frozen-PC2 heldout results: all sixteen inductive correlations have the same negative orientation, so the reported absolute-value mean is not being inflated by opposite-sign assays in this particular archive.

These checks establish reproducibility of the exercised calculations, not the correctness of the model captures or every scientific interpretation. I did not rerun model inference, the 17-producer GPU regression campaign, panel5 capture-level reproduction, or the full geometry/PCA producers. The latest geometry numbers below were checked from archived per-assay results, not independently recollected.

No code, captures, or scientific result archives were changed. The known pre-x64 master-report copies were accepted as the user-reported provenance issue, not re-hashed to reconfirm it.

Paths below use `runs/` for `../prot_interp_files/runs/`, and `audit_data/` for `../prot_interp_files/audit_data/`, relative to the checkout root.

## 1. What is actually strong

### 1.1 The original headline is real, but it is a development result

`runs/transfer_full.json` reports:

| Predictor | Mean held-out-assay Spearman |
|---|---:|
| Final-layer pair vector, 128 channels | 0.758 |
| Per-layer internal magnitudes | 0.665 |
| Substitution chemistry, 17 features | 0.455 |
| Sampled-output summaries, 10 features | 0.328 |

The **score** is 0.758; the internal-minus-output **gain** is 0.430, with reported assay-bootstrap CI [0.353, 0.505], positive in 12/12 assays. These are not interchangeable numbers.

The assay-held-out ridge implementation genuinely excludes the test assay's labels from fitting. Feature selection, when used, is training-only. The full 128-channel headline is not truncated. However, the August 21 report explicitly identifies these twelve assays as the development set where the method, layer, and hyperparameters were chosen. Lead the paper with the later evaluation cohorts; keep 0.758 as the development benchmark.

Sources: `experiments/analysis/reproduce_headline_transfer.py:11-26,54-103`; `src/protein_interpretability/analysis/probes.py:71-104`; `audit_data/results_20260821.html`, section 1.

### 1.2 The cross-model result is more persuasive than the headline alone

The later cohorts provide 16 additional assays and 25 analyzed panel5 assays. The original twelve were not evaluated in all three models under this same later protocol; do not write “53 proteins in each of three models.”

| Cohort | Model | Full internal | Geometry, 37 features | Full internal − geometry | Two PCs − geometry, reported 95% CI | Two-PC wins |
|---|---|---:|---:|---:|---:|---:|
| heldout16 | Boltz-2 | 0.552 | 0.252 | 0.300 | 0.200 [0.118, 0.287] | 14/16 |
| heldout16 | OpenFold3 | 0.676 | 0.348 | 0.329 | 0.062 [0.003, 0.126] | 9/16 |
| heldout16 | Protenix | 0.655 | 0.367 | 0.288 | 0.175 [0.103, 0.240] | 14/16 |
| panel5 | Boltz-2 | 0.360 | 0.160 | 0.201 | 0.147 [0.082, 0.213] | 20/25 |
| panel5 | OpenFold3 | 0.490 | 0.242 | 0.249 | 0.204 [0.152, 0.256] | 24/25 |
| panel5 | Protenix | 0.484 | 0.267 | 0.217 | 0.104 [0.045, 0.170] | 18/25 |

These are paired assay-level comparisons. They make a pure “128 features beat 10 because there are more of them” objection substantially less convincing. OpenFold3's two-PC heldout16 result is marginal, wins only 9/16 assays, and uses the confidence fields requiring correction. It should not carry a universal claim by itself.

The comparisons are still about accessible predictive performance under this estimator—not a measurement of information content. The two-PC choice followed inspection of the dimensionality curves and is exploratory in that respect.

Sources: `runs/geometry_heldout16.json`, `runs/geometry_panel5.json`; `experiments/analysis/geometry_baseline.py:82-138`; `src/protein_interpretability/analysis/transfer.py:78-105`.

### 1.3 Frozen PC2 is a valuable, narrower interpretability result

In `runs/heldout_v1.json`, PC2 with training-statistics normalization has mean signed Spearman −0.695 on twelve held-out stability assays, CI [−0.734, −0.649]. All sixteen held-out assays have the same sign; a single development-frozen DMS orientation is sufficient. This is stronger evidence than a post-hoc absolute correlation with independently chosen signs.

It is still a stability-associated mixture, not a thermodynamic coordinate. The SVD report's own attribution analysis finds substantial substitution and site contributions. The within-position advantage of the full internal predictor over chemistry is **0.083 [0.025, 0.135]**, smaller than its between-position advantage. Preserve that distinction: much of the advantage is contextual sensitivity, with a more modest substitution-specific increment.

The newer cross-phenotype transfer is also useful, but not uniformly beyond chemistry: Boltz-2 stability→panel5 has internal-minus-chemistry 0.034 [−0.044, 0.112], and OpenFold3 panel5→stability 0.076 [−0.017, 0.169]. Do not summarize every cross-phenotype transfer as a demonstrated chemistry-independent signal.

Sources: `runs/heldout_v1.json`, `runs/cross_phenotype.json`; `jax_harness/analyze_heldout.py:149-267`; SVD report sections on attribution and within-position performance.

## 2. Conceptual corrections that matter to the paper

### A. Start from an empirical question, not universal mutation invariance

Predicted native-state geometry can change little while folding free energy changes substantially. Folding stability depends on relative free energies of folded and unfolded ensembles; a coordinate predictor need not produce an unfolded chain to be useful or internally sensitive. Distogram width and pLDDT are model predictive quantities, not calibrated thermodynamic fluctuations or experimental stability.

“Known to be invariant to single/multiple mutations” is too broad. Pak et al. found weak confidence-based mutation prediction, whereas McBride et al. found informative local structural deformation using effective strain. Your own collection contains poorly behaved mutant coordinate predictions in some settings. Separate native-fold robustness, numerical/sampling instability, and phenotype decodability.

Most of the DMS evidence audited here concerns **single substitutions**; the cross-model collector explicitly removes multi-mutants. GFP stress tests and XCL1 do not establish general prediction of double-mutant effects or epistasis.

### B. Replace “more information” with “more readily decodable signal”

A superior linear probe establishes better prediction for a specified target, population, and probe class. It does not establish a mutual-information ordering or absence of the target from the full output. Nor does a deterministic representation add information beyond its complete inputs; it can make information more accessible.

This applies symmetrically. The SVD report correctly warns that the full pair row's poor high-dimensional probe does not mean the full row contains less information than its mean. The same restraint is needed when a high-dimensional coordinate probe performs poorly.

Avoid “91% of the information,” “four PCs contain 90% of stability,” or interpreting ratios of Spearman correlations as fractions of information. Report prediction differences and uncertainty. Similar performance is not equivalence unless a practical margin was specified and evaluated.

### C. Define output categories without moving the boundary

Use separate columns:

1. Hidden Pairformer representation.
2. Distogram-head predictions.
3. Sampled coordinate geometry.
4. Confidence-head outputs.

Excluding the distogram is legitimate for a **trunk-versus-sampled-coordinates** comparison. It is not legitimate evidence for “better than anything the model emits.” The current strongest steering endpoint is the distogram itself, so it cannot simultaneously be treated as irrelevant to output predictability and decisive evidence of structure-module use.

The 37-feature block is useful but not exhaustive. It excludes full all-atom chemistry, the complete output distribution, and spatial confidence detail available in some older captures. It also is **not a strict concatenation of the original ten features plus 27 new ones**: the two original local RMSDs are absent as standalone columns. Consequently, “tripling features buys almost nothing” is not a nested-model proof that no useful geometry remains.

A supplementary within-protein analysis already tried coordinate/distance vectors up to approximately 1,831 features and per-residue confidence. Keep it; do not pretend these controls were never run. It nevertheless cannot close every nonlinear or cross-protein output-readout alternative.

Sources: `src/protein_interpretability/analysis/emitted_geometry.py:65-74,116-185`; `exp_steer.py:173-199`; master and SVD output-comparison sections.

### D. Distinguish three different meanings of “shared”

- The original Boltz-2 PC2 is a particular frozen direction in a particular standardized channel space.
- A two-PC ridge in the new analyses is fitted separately within each model and training fold.
- The direction-agreement analysis compares stability versus panel5 **inside each model**. It does not align Boltz-2 channels to OpenFold3 or Protenix channels.

Thus “the same two dimensions in three architectures” is unsupported. Supported: a low-dimensional predictive subspace occurs in each tested model, with evidence of cross-phenotype sharing within models. Cross-model identity requires an independently trained correspondence, tested on held-out matched variants; equal dimension and high within-model cosine do not establish it.

The two-PC predictor also does not uniformly reproduce the full model: on heldout16 OpenFold3 it scores 0.410 versus 0.676 for all channels. Do not call it “essentially all” without reporting that gap.

Source: `experiments/analysis/direction_agreement.py:139-160,246-269`.

### E. Negative mechanism findings do not establish “no mechanism”

“No operation preferentially engages PC2 under these measurements” is supported. “The stability axis is not constructed by a mechanism” is not: distributed computation, changing bases, nonlinear interactions, and unmeasured pathways remain possible. Rotation of a representation is not evidence that its predictive content originated at that layer.

The Jacobian results are valuable as local descriptions of sampled operating points. The compressed channel operator is not the full spatial Jacobian. Also, “14% live units” is an effective participation statistic, not a literal count of exactly zero versus nonzero SwiGLU units.

Keep the Jacobian/rotation work in supplementary material unless it provides a necessary link in the final argument. Do not infer absence of useful denoiser representations from small coordinate-trajectory differences: that repeats the representation-versus-output inference the paper is challenging.

## 3. Statistical and implementation issues to resolve

### P0: OpenFold3 confidence extraction

`docs/MODEL_AUDIT.md` declares OpenFold3 pLDDT **per atom**. `jax_harness/exp_gym_deep.py:252` stores `e.plddt[pos]`, where `pos` is a residue index. The site confidence is therefore not the intended site's value. The derived mean-minus-site feature is affected too. Define whether chain confidence is atom-weighted or residue-weighted; do not silently mix them.

Correct the mapping and regenerate or recover the affected confidence fields and all dependent output comparisons. The 128-channel internal score does not use site pLDDT, but the reported paired advantage depends on the baseline. Dropping the affected columns is a useful diagnostic, not a replacement for a corrected comparator. Do not predict the direction of the correction.

### P0 for causal significance: steering controls are shared across assays

`launch_steer.sh` does not supply an assay-specific seed. `exp_steer.py:105,129-136` defaults to seed 0 and generates random vectors in standardized space before applying assay-specific channel scales. The NPZ reconstruction described above confirms this actually happened, rather than merely being a possible source path.

`analyze_steer_pool.py:129-142` uses a binomial rank-first null and independently sampled uniform ranks across assays. These are not the actual randomization design: the same random orientations pass through the same model across proteins. The resulting cross-assay dependence is not preserved. A uniform rank also requires exchangeability of the selected PC2 and random directions under the intended null, which is not automatically established for a direction discovered and characterized on the same development cohort.

**Do not use p=0.000957 or the reported near-zero rank p as decisive confirmation unchanged.** Keep 6/12 rank-first and mean normalized rank 0.896 as descriptive observations. All twelve archived PC2 width slopes are positive and all twelve pLDDT slopes negative; these signed observations remain useful.

There is a second conceptual error: random perturbations need not produce an even response. Locally, for any direction v,

\[
f(z+\alpha v)-f(z-\alpha v)=2\alpha\nabla f(z)^Tv+O(\alpha^3).
\]

Odd response is generic first-order sensitivity, not proof of semantic use. The pooled statistic ranks **absolute** odd responses, so it is not itself a test of the biologically expected sign.

For stronger inference, freeze direction, orientation, sites, mode, dose range, and endpoint; evaluate on proteins outside discovery, with independently generated matched controls or a joint null preserving shared directions. Include controls matched for empirical covariance and, for distogram specificity, head sensitivity—not just raw norm. Report actual signed dose-response, separating near-mutation-scale doses from 10–30× perturbations. Use a finite Monte Carlo correction rather than p=0.

### Preserve the existing PC2-deletion null

`audit_data/reports/report_svd/data/ablate_v1.json` reports four assays: PC2 deletion yields distogram recovery **−0.016 [−0.109, 0.082]**, approximately zero pLDDT change, and mean C-alpha shift **0.018 Å**. No PC2-minus-random recovery interval excludes zero.

This does not prove the direction is never used, but it does prevent a simple necessity claim. Synthetic steering and deletion of the natural mutation component are different interventions. Present both. Removing a probe direction and thereby eliminating that probe's prediction is algebra, not independent evidence of biological causality.

### Normalization and metadata need a clean comparison

The main LOAO and cross-phenotype implementations standardize held-out assay features using their own unlabeled variants. This is transductive, not label leakage. Test-assay labels do not enter ridge fitting; the stored standardized test target is not used for training.

However:

- The original `transfer_inductive.json` uses **k=16** for the 128- and 256-wide blocks, unlike the untruncated primary archive. Its magnitudes score is 0.566 and vector score 0.692. The master's “under inductive normalization the gap grows” paragraph is not a controlled full-width normalization comparison.
- The master heldout-PC2 table labels two rows as bases fitted with/without held-out proteins. In the inspected producer the basis is frozen in both; the two variants differ in feature normalization.
- New geometry and cross-phenotype results also use held-out feature statistics. “Blind transfer” there means training-only PCA, not fully inductive preprocessing.
- `analyze_transfer.py:319` hard-codes “train on 11 assays” for later 16/25-assay runs.

Report both normalization regimes under otherwise identical feature width, PCA, targets, and folds. Retain the original setting for continuity; avoid implying it is a calibrated single-variant deployment predictor.

### Chemistry controls are useful, not exhaustive

The existing 17-descriptor baseline, conditional comparisons, identity attribution, and within-position decomposition are substantial evidence. Do not restart the project as if chemistry had never been tested.

But the chemistry increments in `analyze_chem.py` are within-assay position-split analyses, not the new cross-cohort claim. Their PC2 basis is fitted globally on the development features before the position splits. They should be labeled accordingly.

The attribution code also claims that separate WT and mutant one-hot vectors can represent any substitution function. With a linear fit they represent an additive function a(WT)+b(mutant), not all 380 ordered substitution effects or their context interactions. A small residual after this adjustment does not isolate novel physics, and a residual after linear adjustment can still contain nonlinear chemistry.

Better test: compare a prespecified chemistry/context baseline against that same baseline plus frozen PC2 or a training-fitted reduced trunk representation on held-out proteins. Include ordered substitution identity with regularization, conservation/MSA compatibility, and WT burial/contact context and a small prespecified set of interactions. Report a sequence-model score as contextual evidence if available; it need not be beaten for a within-model interpretability claim.

Sources: `jax_harness/analyze_chem.py:96-134,197-213`; `jax_harness/analyze_attrib.py`, `resid` and the 40-column identity block.

### Other statistical boundaries

- Assay-level paired bootstrap is a major improvement over bootstrapping overlapping splits. These intervals still do not account for the entire adaptive research process or fully refit every shared training set. Use a train/refit resampling sensitivity if a marginal result is central.
- Cohort disjointness checks assay IDs; panel selection also deduplicates UniProt IDs. Neither alone establishes sequence/fold independence. Audit cross-cohort homology, use family-grouped sensitivity where needed, and distinguish “unseen by the probe” from “absent from the folding model's pretraining.” No actual homology leakage was established in this audit.
- Cross-model variants are selected at evenly spaced ranks of observed DMS scores (`exp_gym_deep.py:137-147`). This is outcome-informed sampling, not direct test-label leakage into fitting. It does not automatically prove inflated correlation: evenly spaced ranks approximately cover the empirical score distribution. A label-blind sample or full-assay sensitivity would clarify the estimand.
- The phenotype contrast remains confounded with assay technology, protein population, and length. ENVZ shows length alone is insufficient; it does not show length and MSA depth have no effect.
- `metrics.mcc` uses the true held-out positive count to set its prediction threshold. Call this oracle-prevalence/rank-based MCC, not an independently calibrated classifier. Spearman and AUC are unaffected by that threshold choice.
- Distogram Gaussian shift/spread shares are approximate for non-Gaussian distributions. A nonsignificant KL-versus-spread gap is not equivalence. “Confidence, not geometry” remains too strong.

## 4. Prior work and the remaining novelty

This is a focused comparison, not a systematic literature review or a claim of priority over every current paper.

| Primary work | What overlaps | Your defensible distinction |
|---|---|---|
| [AlphaInterp, Feldman & Skolnick, 2026 preprint, May 29 version](https://pmc.ncbi.nlm.nih.gov/articles/PMC13131572/) | AF3 checkpoint probing, PCA, confidence/distogram manipulation, adversarial mutation robustness, MSA dependence. Particularly close to the current framing. | Experimental single-mutation DMS/stability prediction, paired output-readout comparisons, disjoint-protein and cross-phenotype transfer, quantitative low-dimensional mutation differences across three related models. |
| [AFToolkit, Sindeeva et al., 2025](https://academic.oup.com/bib/article/26/4/bbaf324/8190210) | WT/mutant AlphaFold-derived embeddings with lightweight supervised adapters for stability and affinity; no folding-backbone fine-tuning. | Unmodified frozen-model representation analysis under your declared inference conditions; explicit output comparison and a characterized mutation subspace rather than primarily a prediction pipeline. |
| [Mutate Everything, Ouyang-Zhang et al., NeurIPS 2023](https://arxiv.org/abs/2310.12979) | AlphaFold/ESM representations support stability prediction, including multiple mutations. | A representation audit rather than an efficient trained mutation-effect predictor. Do not claim that usefulness of folding embeddings for stability is new. |
| [McBride et al., AlphaFold2 Can Predict Single-Mutation Effects, PRL 2023](https://arxiv.org/abs/2204.06860) | Local predicted structural deformation can correlate with mutation phenotypes; effective strain is a relevant comparator. | Quantify how much experimental signal is more accessible internally under matched protocols, rather than assert coordinates cannot be informative. |
| [Pak et al., PLOS ONE 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10019719/) | Weak confidence-based stability/function prediction. | Move beyond confidence scalars, using paired controlled representations and transferable probes. |
| [AF2BIND, Gazizov et al., Nature Methods 2026](https://www.nature.com/articles/s41592-026-03011-2) | Linear readout of pretrained AF2 pair features reveals useful functional signal. | Mutation-difference, phenotype-transfer, and output-readout questions; not generic emergence of functional information in structure-model features. |

AlphaInterp deserves a direct paragraph in the Introduction/Discussion. Its severe multi-mutation, whole-representation-distance analysis and your small single-mutation, localized-difference decodability analysis need not disagree: a representation can remain globally similar while a low-amplitude direction predicts experimental effects. Do not claim a contradiction without matched perturbations and metrics.

For causal language, [Heimersheim & Nanda's activation-patching guidance](https://arxiv.org/abs/2404.15255) is useful methodological context. Distinguish manipulability, necessity, sufficiency, and mediation; these are not synonyms.

**Novelty judgment:** the combination of experimentally grounded mutation differences, frozen protein transfer, low-dimensional organization, and explicit representation/output comparison is the strongest contribution. Generic “hidden representations encode biology” and “PC injection changes confidence” are not sufficient novelty on their own.

## 5. Finishing plan, in priority order

### Gate 0 — Correctness and one authoritative evidence release

Before drawing final conclusions:

1. Correct OpenFold3 atom-to-residue confidence handling and recompute dependent comparisons.
2. Repair or downgrade steering significance; include the existing deletion null and explicitly separate the distogram head from diffusion outputs.
3. Regenerate clean, versioned result artifacts from an identifiable committed checkout, preserving historical outputs. `git_dirty: true` is not evidence that results are wrong, but a commit alone cannot reproduce unrecorded edits.
4. Rebuild master figures/report with adopted x64 inputs. Fix normalization labels, cohort-dependent protocol text, mutant-versus-WT confidence labels, and analyzed phenotype counts (panel5: 14 fitness, 7 abundance, 4 activity after exclusions).
5. Make one claim ledger: claim, estimand, cohort/variants, normalization, input artifact, figure, and status (development, evaluation, exploratory, corrected/withdrawn). Historical reports should not all remain equally authoritative.

**Acceptance:** every manuscript number points to a single adopted artifact and exact protocol; corrected confidence comparisons are available; causal p-values are justified or explicitly descriptive. No new biological claim is required to pass this gate.

### Gate 1 — Close the central offline interpretation gaps

**A. Frozen PC2 bridge.** Use the original development-frozen Boltz-2 basis and DMS sign on the exact variants in the new heldout16/panel5 comparisons. Compare PC2, two training-fold PCs, and full 128 channels against geometry/confidence on identical rows. Report both transductive and training-statistics versions; never choose a new sign on test labels.

**B. Conditional signal, not just a race against weak baselines.** Compare B versus B+internal, where B includes the prespecified substitution/context controls above. Fit nuisance regressions, PCA, and hyperparameters on training proteins only. Residualized subspace plots may be secondary; the primary test should be a held-out incremental predictive gain. The August 23 suggestion to residualize within every assay is not the fully inductive version of this test.

**C. Baseline fairness sensitivity.** Preserve fixed lambda=10 as the original protocol, but allow a small identical inner protein-held-out lambda search for each feature block. The same numeric ridge penalty does not imply the same effective capacity under different covariance spectra. Include a genuinely nested union of the 10- and 37-feature blocks and a prespecified local strain feature. If claiming more than linear accessibility, add one modest nonlinear output/context decoder with training-only tuning, not an unrestricted architecture search.

**Interpretation:** if PC2 transfers poorly outside stability while the full representation remains useful, write that distinction. If chemistry/context explains the two-PC transfer, retain the broader decodability result but stop calling the shared core novel stability-specific information. If a better output readout closes the gap, the original result was about readout accessibility, not loss of information.

### Gate 2 — One targeted inference-quality and ensemble challenge

This is the highest-value remaining GPU campaign for a representation-versus-output paper. Do not launch the old full combinatorial grid.

- Use heldout16, or a balanced subset selected before looking at new outcomes. Include stability and non-stability, differing confidence, and more than only the easiest proteins. A subset establishes sensitivity, not full-cohort confirmation.
- Fix each WT/mutant's MSA and trunk state across diffusion draws. Vary diffusion noise only; changing a wrapper key must not silently redraw the MSA.
- Prespecify 4–8 diffusion draws. Compare one-draw summaries with feature-wise ensemble means; mean+SD is a secondary block. Aggregate geometry features rather than averaging unaligned coordinates.
- **Same seed is not sufficient to claim common random numbers across mutations.** Mutations can change atom counts/order, as the existing corrections page already records. Either implement a verified atom correspondence/noise coupling or use independently sampled WT/mutant ensembles and distributional summaries without claiming paired atom noise.
- Include a small convergence check at the model's documented recommended inference settings, with fixed variants. A fixed 200-step setting across different models is not itself proof of equal convergence.
- For Boltz-2's known degenerate mutant-coordinate outputs, verify a small WT/conservative/deleterious subset against the native/recommended inference pipeline before treating weak coordinates as a biological result. Trunk capture fidelity does not validate downstream coordinates.
- On a small fixed subset, compare the controlled WT-homolog MSA regime with ordinary per-variant search at matched depth/settings. Existing route experiments establish that MSA handling matters, but do not establish that the new DMS ranking gap is invariant to it.

**Primary endpoint:** paired internal-minus-ensemble-output Spearman, with assay as the uncertainty unit, reporting both full and prespecified reduced internal representations. Prespecify which model/cohort comparison is primary rather than requiring every exploratory cell to be nominally significant.

**Interpretation:** a surviving gap supports a claim against a better estimate of sampled output. Closure by seeds, convergence, or ordinary inference narrows the claim honestly. Neither outcome makes the representation measurements worthless.

### Gate 3 — Optional only if a causal bottleneck is the paper's headline

If the intended claim remains “the sampler discards stability information,” then the conditioning experiment is not optional. Probe Pairformer output and the actual diffusion-conditioning tensors on identical variants with matched evaluation and adequately expressive, fairly tuned decoders. A decodability drop localizes a readout gap; causal localization additionally needs targeted patching/ablation/rescue and quality controls.

Do not make this a prerequisite for the narrower recommended paper. Existing synthetic steering is an informative case study, not a complete explanation of where information is lost. If stronger mechanistic novelty is desired, the best extension is held-out, natural-perturbation-scale intervention with matched controls and an observable downstream endpoint—not more Jacobian plots or eliminating a probe's own direction.

## 6. What I would do differently

1. **Organize by questions, not chronology.** The reports repeatedly revise broad claims. The paper should contain only the final answers plus a compact robustness/corrections supplement.
2. **Freeze a claim and comparator before each new analysis.** Separate development, evaluation, and post-evaluation exploration. Do not describe a post-hoc two-PC choice as prespecified.
3. **Treat the input/context baseline as central.** A mutation representation can encode substitution identity, evolutionary surprise, and environment without being a distinct physical free-energy variable. Quantify that mixture instead of trying to purge all chemistry and call the remainder physics.
4. **Use one matched figure for each inferential question.** Do not combine within-protein position splits, within-cohort LOAO, and completely frozen cross-cohort transfer on an unlabeled common axis.
5. **Prioritize falsification over breadth.** A robust output comparator and ordinary-inference sensitivity are more decisive than a fourth related model, more PCA variants, or a new SAE campaign.
6. **Avoid declaring negative results universal.** The XCL1 result is no demonstrated directional conformational steering in that regime; the PC2 deletion result is no demonstrated necessity under that intervention; the Jacobian result is no identified PC2-specific operation under that analysis.

## 7. Suggested manuscript structure

Working title: **Low-dimensional mutation-effect signals in protein structure model representations**.

- **Figure 1 — Setup and principal paired result.** Mutation-induced pair-row difference, frozen models, exact protocols; heldout16/panel5 paired per-assay comparisons with corrected baselines. Development headline as a clearly labeled inset or supplement.
- **Figure 2 — Low-dimensional organization and transfer.** Training-only dimensionality curves, frozen Boltz-2 PC2, separate cross-phenotype transfer panels. No claim of cross-model axis identity without alignment evidence.
- **Figure 3 — What explains the signal?** Chemistry/context increments, within-position versus between-position performance, ensemble/inference sensitivity. Explain what the residual does and does not imply.
- **Figure 4 — Scoped Boltz-2 intervention.** Signed distogram/confidence responses, matched controls, and deletion null side by side. If the corrected evidence remains only descriptive, make this supplementary instead of stretching it into a causal headline.
- **Supplement —** Jacobian/gate/rotation analyses, route experiments, XCL1, older full-dimensional output controls, numerical validation, exclusions, and provenance ledger.

Discussion should distinguish native-state structure prediction from folding thermodynamics, biological information from model confidence, and readout accessibility from information destruction. It should address AlphaInterp and AFToolkit explicitly.

## 8. Stopping rule

Start writing now around the narrowed claim. Finish Gate 0, the targeted offline comparisons in Gate 1, and one prespecified inference/output challenge in Gate 2. Then freeze the evidence and write the observed outcome, including narrower or negative conclusions where necessary.

Do not require a positive PC2 result in every phenotype or every model to finish. Do require that the central claim survive its corrected comparator. If it does not, change the claim rather than add analyses until a preferred number reappears.

Do **not** add, for this paper by default: new SAEs, a large MSA-depth grid, a full denoiser capture campaign, many more folds/models, or an epistasis project. Double-mutant non-additivity is a worthwhile separate question, but needs matched singles, the correct experimental scale, an additive/global-epistasis baseline, and held-out pair/protein design. It is not a small extension of the current single-mutant result.

**Bottom line:** the experimental representation signal is substantially better supported than the strongest prose claims. Tightening the prose is not retreat; it identifies the contribution that can survive review. The remaining effort should establish what the signal adds and what comparator it truly beats, not expand the number of analyses.
