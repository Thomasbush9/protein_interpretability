# Capturing inside the diffusion module: what it would take, and what we already know

Scoped 2026-08-19, not started. This exists so the decision can be made with the
cost and the prior evidence in front of you rather than from memory.

## What the question is

"Internal versus output" in this project has a specific boundary:

- **internal** — read off the Pairformer trunk's representation tensors
- **output** — what the structure module *emits*: CA coordinates and pLDDT

`compare_internal_output.py` deliberately excludes the trunk distogram from the
output side, and says why: *"It is a head on the Pairformer, not a product of the
structure module, and including it would blur the internal/output distinction the
whole comparison rests on."*

Diffusion-module activations are therefore **not more of the output side**. They
are a third category — the machinery between the two — and adding them to the
"output" column would break the distinction the headline gap is defined by. If
they are captured, they need their own column and their own claim.

## What we already measured, and it is discouraging

`exp_trajectory.py` re-runs joltz's `AtomDiffusion2.sample` scan with `ys`
populated, so the per-step coordinate state is already recoverable, bit-identical
to the library's dynamics. Three captures exist (`traj_*`, `traj3_*`).

From `logs/tr3_RCRO_36679756.out` — RCRO, 63 residues, 200 steps, 100 variants,
against a wild-type-versus-wild-type noise floor over 4 keys:

| step | sigma | mut-vs-WT | WT floor | ratio |
|---|---|---|---|---|
| 0 | 2560.000 | 1981.7 | 3213.2 | 0.617 |
| 100 | 45.771 | 31.1 | 46.2 | 0.674 |
| 180 | 0.051 | 0.330 | 0.305 | 1.082 |
| 199 | 0.002 | 0.331 | 0.305 | 1.085 |

**The mutation is indistinguishable from sampler noise for about 80% of the
schedule, and peaks at 1.085x the floor at the very last step.** The sampler is
not, on this evidence, where mutation information lives.

`exp_amplify.py`'s docstring states the chain more sharply — the conditioning
carries the mutation (‖dq‖/‖q‖ = 0.285), the sampled structure does not (TM-to-WT
rho 0.214), and the sampler ignores the conditioning while the global fold is
decided. **Caveat: that 0.285 appears only in a docstring.** No script, archived
JSON, or log in this project produces it. It is an unarchived measurement and
should be re-derived before being quoted.

## The cheap version, if we do anything

The trunk reaches the sampler **only** as three bias tensors, not as raw `(s, z)`:
`atom_enc_bias`, `token_trans_bias`, `atom_dec_bias`, plus `q`, `c`, `to_keys`.
`exp_trajectory.py:170` already computes all six via
`model.diffusion_conditioning(...)` and then throws them away.

Saving those answers the actual question — *does trunk knowledge reach the
sampler, and does it carry the mutation?* — for roughly the cost of adding array
names to an existing capture. It requires no new capture site, no new kernel and
no re-validation of the sampler, because the tensors are already materialized on
the path that produced validated trajectories. This is the version worth doing.

## The expensive version, and its prerequisites

Capturing score-network hidden activations (atom encoder, token transformer,
atom decoder) needs all of the following, none of which exists:

1. **Recorded widths.** No atom-encoder / token-transformer / atom-decoder
   dimension is written down anywhere in this repo. `capabilities.py`'s policy is
   that anything unmeasured is `None` and raises, so these must be measured off a
   loaded model before a spec can name them.
2. **A fidelity gate.** `pi_capture.verify_capture()` compares a captured trunk
   against the model's own output and `check_signal_to_drift()` refuses a run
   whose signal is not far above that drift. Nothing equivalent exists for the
   sampler, and without it a diffusion capture cannot be told from noise — which,
   given the table above, is the failure mode most likely to occur.
3. **Schema fields.** `capture_spec.FIELDS` and `capabilities.ModelCapabilities`
   have no diffusion entries; a capture would be unshapeable and uncheckable.
4. **A port of the design.** `docs/diffusion_boltz.md` §5.6 is the only shape
   table, and it is written against **PyTorch Boltz** `diffusionv2.py` line
   numbers. The runtime here is **joltz/JAX**, where `sample` is a `jax.lax.scan`
   and the mechanism is `exp_trajectory`'s populate-the-`ys` trick, not a module
   hook. The doc is a design sketch for a different codebase.

## Cost

A 200-step trajectory on a 63-mer is ~25 s per variant (measured: 2668 s for 104
trajectories). Storing per-step, per-atom tensors instead of CA-only would
multiply the per-variant artifact by roughly the atom count over the residue
count times the retained channels — the reason the existing capture reduces to CA
frames and then to scalars.

## Recommendation

Do the conditioning-tensor version if the question is live, and re-derive the
0.285 figure while doing it since nothing archives it. Do not build a
score-network capture site until the trajectory evidence is contradicted — the
measurement we already have says the signal is not there to find.
