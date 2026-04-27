# Boltz-2 diffusion + steering: how it works and how to interpret it

Companion to [`boltz2_info.md`](boltz2_info.md). That doc covers the trunk
(input embedder → MSA → pairformer → distogram). This one covers what happens
**after** the trunk: the structure module, which is a conditional diffusion
process over atom coordinates, and the **steering** machinery that biases that
process toward physically plausible / user-constrained structures.

Code under reference (Boltz repo at `~/Documents/ML/boltz`):

- `src/boltz/model/modules/diffusionv2.py` — Boltz-2 atom diffusion
- `src/boltz/model/modules/diffusion_conditioning.py` — pre-computed conditioning
- `src/boltz/model/potentials/potentials.py` — guidance potentials (energies)
- `src/boltz/model/potentials/schedules.py` — time-dependent potential schedules
- `src/boltz/model/models/boltz2.py` — top-level wiring of trunk → diffusion
- `src/boltz/main.py` — CLI flags, default `BoltzSteeringParams`

---

## 1. The diffusion process at a glance

Boltz-2's structure module is an EDM-style score-based diffusion model on atom
coordinates `x ∈ ℝ^{L×3}`, conditioned on the trunk outputs `(s_inputs, s_trunk)`
and the pre-computed pair biases derived from `z_trunk`.

```
x_T ~ N(0, σ_max² I)   ──►  iterative denoising (N=200 default)  ──►  x_0 (final coords)
                              │
                              ├── score network  D_θ(x_t, σ_t; s, z-derived biases)
                              └── steering: potentials modify trajectory
```

Important: the diffusion happens **per recycling step** of the trunk. The trunk
runs `(s, z)` recycling, and at the very end of the last recycling step it
hands `(s_trunk, z_trunk)` to `AtomDiffusion.sample(...)` which runs the full
denoising loop (default 200 steps). So one Boltz prediction = one trunk forward
(× recycling) + one diffusion run (× particles, see §4).

### 1.1 Karras noise schedule

Defined in `AtomDiffusion.sample_schedule` (`diffusionv2.py:276`). Defaults
(`BoltzDiffusionParams`, `main.py:130`):

| Parameter | Value | Meaning |
|---|---|---|
| `sigma_min` | `1e-4` | floor noise level at the end of denoising |
| `sigma_max` | `160` | initial noise level (essentially Gaussian noise on coordinates) |
| `sigma_data` | `16.0` | empirical std of clean data, used in EDM preconditioning |
| `rho` | `7` | shape of the schedule (Karras et al.) |
| `gamma_0` | `0.8` | extra stochasticity above `gamma_min` |
| `gamma_min` | `1.0` | threshold below which step is deterministic |
| `step_scale` | `1.5` | overshoot factor on the Euler step |
| `noise_scale` | `1.003` | small additional injected noise |

### 1.2 EDM preconditioning

`AtomDiffusion.preconditioned_network_forward` (`diffusionv2.py:239–274`)
implements Karras et al.'s `c_skip / c_out / c_in / c_noise`:

```
x_in   = c_in(σ) · x_noisy                       # ≈ unit variance
F_θ    = score_network(x_in, c_noise(σ), …)      # raw network output
x_0    = c_skip(σ) · x_noisy + c_out(σ) · F_θ    # denoised x_0 prediction
```

So **the network always predicts a clean `x_0`**, not noise — at every step,
intermediate or final. This is important for interpretability: every step has
an interpretable "current best guess of the structure", not a hard-to-read
noise prediction.

### 1.3 Per-step update (the loop body)

The denoising loop is `AtomDiffusion.sample` (`diffusionv2.py:295–530`). The
core pattern at each step `step_idx ∈ [0, N-1]`:

```
1. Random global rotation + translation of x  (coord augmentation)         (351–363)
2. t_hat = σ_{t-1} · (1 + γ);  add stochastic kick eps ~ N(0, σ_kick² I)   (374–378)
3. x_0_hat = preconditioned_network_forward(x + eps, t_hat, conditioning)   (387–396)
4. [optional] FK steering: compute energy of x_0_hat, prep resample weights (398–441)
5. [optional] Gradient guidance: x_0_hat ← x_0_hat - α · ∇E(x_0_hat)        (443–473)
6. [optional] FK steering: resample particles by importance weights         (475–510)
7. [optional] Rigid-align noisy → denoised before Euler step                (512–521)
8. Euler step: x ← x_noisy + step_scale · (σ_t - t_hat) · (x_noisy - x_0)/t_hat   (523–528)
```

The score network call is gated by `torch.no_grad()` — the diffusion loop is
inference-only, so any gradient interpretability work has to enable grad
explicitly or use hooks on activations.

---

## 2. Conditioning: how the trunk talks to the diffusion

The trunk doesn't enter the per-step score evaluation as raw `(s, z)` — instead
it enters as **pre-computed attention biases** to keep diffusion cheap.

`DiffusionConditioning` (`diffusion_conditioning.py:13–117`) runs once before
the denoising loop and produces:

| Output | Shape | Used by |
|---|---|---|
| `q`, `c`, `to_keys` | various | atom encoder pre-projection |
| `atom_enc_bias` | `(B, num_atoms, atom_enc_depth · num_heads)` | atom encoder attention |
| `token_trans_bias` | `(B, N, token_trans_depth · num_heads)` | token transformer attention |
| `atom_dec_bias` | `(B, num_atoms, atom_dec_depth · num_heads)` | atom decoder attention |

These biases are **what carry trunk knowledge into the diffusion score**. The
score network itself sees per-step `(x_noisy, σ)` plus these constant biases.

**Interpretability hook**: zero out, scale, or perturb `atom_enc_bias` /
`token_trans_bias` / `atom_dec_bias` to ablate the trunk's influence at
different points in the score network — analogue of "do we still get the right
structure if the trunk only conditions the encoder, not the decoder?"

---

## 3. Steering, mode 1: gradient-based potential guidance

This is classifier-style guidance. At each step you compute an energy
`E(x_0_hat)` over the current denoised prediction and push `x_0_hat` downhill
by gradient descent before doing the Euler step.

Code: `diffusionv2.py:443–473`.

```
guidance_update = 0
for k in range(num_gd_steps):                            # default 20
    g = Σ_potential w_p(t) · ∇_x E_p(x_0_hat + guidance_update)
    guidance_update -= g
x_0_hat += guidance_update
scaled_guidance_update = guidance_update · -step_scale · (σ_t - t_hat)/t_hat
```

The "scaled_guidance_update" is rescaled into noise-space coordinates and used
later to correct the FK importance ratio (so the two steering modes compose
correctly).

### 3.1 The potentials catalog

Defined in `potentials.py`. `get_potentials(steering_args, boltz2=True)`
(approx. `potentials.py:670` and `:756`) returns a list. The standard ones:

| Potential | What it penalises |
|---|---|
| `VDWOverlap` | Atoms closer than van der Waals contact (steric clash) |
| `Connections` | Bonded atoms drifting apart |
| `PoseBusters` | Geometric sanity (PoseBusters checks: bond/angle/clash) |
| `Chiral` | Wrong chirality at chiral centers |
| `StereoBond` | Wrong cis/trans |
| `PlanarBond` | Non-planar planar bonds (e.g. peptide bond, aromatic rings) |
| `SymmetricChainCOM` | Symmetric homomers whose chain COMs collapse |
| `ContactPotential` | User-provided contact constraints (Boltz-2) |
| `TemplateReferencePotential` | Pull toward provided template (Boltz-2) |

Each potential subclasses `Potential` (`potentials.py:15–229`) and implements
`compute_variable` (extract a geometric quantity), `compute_function` (map it
to energy + gradient), `compute_args` (pick the relevant atom pairs from
`feats`). All are differentiable.

### 3.2 Time-varying potential weights

`schedules.py:1–37` defines two schedule classes that map the normalised
denoising progress `t ∈ [0, 1]` (where `t = 1.0 - step_idx / N`) to a scalar
weight:

- `ExponentialInterpolation` — smoothly ramps a weight from `start` to `end`
  with curvature `α`. Used e.g. for the symmetric-chain buffer (1 Å → 5 Å as
  noise drops) and the ContactPotential `union_lambda` (8 → 0 over time).
- `PiecewiseStepFunction` — turns a potential on/off at fixed thresholds.
  E.g. VDW overlap penalty engages at `t > 0.4`; contact guidance ramps
  `0 → 0.5 → 1` at `t = 0.25, 0.75`.

**Why this matters for interpretation**: the model uses *strong* trunk
conditioning + *weak/no* geometry potentials at high noise (early steps), then
gradually hands control to the potentials as the structure crystallises. So
"what is steering doing at step k?" depends sharply on `k`. Any sensitivity
analysis must be step-resolved.

---

## 4. Steering, mode 2: Feynman-Kac (particle-filter) steering

This is the more interesting mechanism for interpretability — it's not just
gradient guidance, it's **sequential Monte Carlo on the diffusion trajectory**.
Idea: run `M = num_particles` (default 3) noised trajectories in parallel,
periodically resample them in proportion to how good they look under a
potential `E`. Particles that are heading toward bad geometry die; ones with
good `x_0_hat` get cloned.

Code: `diffusionv2.py:398–441` (weight computation) and `:475–510` (resample).

```
every fk_resampling_interval steps:                       # default 3
    E_i = Σ_potential w_p_resample(t) · E_p(x_0_hat_i)    # per-particle energy
    log G_i = E_prev_i - E_curr_i                         # energy drop this interval
    ll_diff_i = (||eps||² - ||eps + scaled_guidance||²) / (2 σ²)   # IS correction
    w_i = softmax( ll_diff + fk_lambda · log G ) over particles   # default λ = 4
    indices = multinomial(w_i, num_particles, replacement=True)
    resample x, x_noisy, atom_mask, energy_traj, ... by indices
```

Notes:
- The `fk_lambda` knob is the temperature on the energy-vs-likelihood tradeoff.
  Higher = more aggressive selection toward low-energy particles.
- `--use_potentials` (CLI flag) sets `fk_steering = True` *and*
  `physical_guidance_update = True` (so the two modes stack).
- `contact_guidance_update` defaults to `True` in Boltz-2 even without
  `--use_potentials` — contacts always guide if you provide them.

### 4.1 The default user-facing knobs

`BoltzSteeringParams` in `main.py:148–157`:

```python
fk_steering: bool = False
num_particles: int = 3
fk_lambda: float = 4.0
fk_resampling_interval: int = 3
physical_guidance_update: bool = False
contact_guidance_update: bool = True
num_gd_steps: int = 20
```

CLI flag: `--use_potentials` toggles both `fk_steering` and
`physical_guidance_update` (`main.py:1309–1312`). All other knobs are
hard-coded defaults — you'd need to either patch `main.py` or call the model
directly to sweep them.

---

## 5. Concrete intervention points (what to hook for interpretability)

The diffusion loop is a forward-only `for` loop in Python — every step is a
clean monkey-patch / hook target. Below: the points worth tapping, organised
by what question they answer.

### 5.1 "What does the model think the structure is at step k?"

Hook on `AtomDiffusion.preconditioned_network_forward` (returns
`atom_coords_denoised_chunk`, `diffusionv2.py:388`). Each call returns the
current best `x_0` prediction. Capture all N=200 of them per particle to get
the **denoising trajectory** — a movie of the model "deciding" the structure.

Useful aggregates per step:
- RMSD(`x_0_hat_step_k`, `x_0_hat_final`) → "when did the model commit?"
- Per-residue confidence over time (you can compute pseudo-pLDDT by feeding
  intermediate `x_0` through the confidence head).
- Per-residue position variance across the M FK particles → "where is the
  model uncertain at step k?"

### 5.2 "Which residues / atoms steer the trajectory?"

The energy gradient `energy_gradient` at `diffusionv2.py:450–464` is
per-atom `(B, num_atoms, 3)`. Capturing it gives the **steering force field**
at every step. You can:
- Sum per-residue magnitude → which residues are being pushed by potentials.
- Decompose by potential (`for potential in potentials: log
  potential.compute_gradient(...)` separately) → which physics is driving the
  push at this step.
- Compare `||energy_gradient|| × num_gd_steps` to `||score||` → is steering
  dominating or perturbing?

### 5.3 "Which particles win?"

Hook `resample_indices` at `diffusionv2.py:482`. This is a
`(num_particles,)` integer tensor per resampling round. Track:
- Which initial particle is the ancestor of the final survivor (genealogy).
- Effective sample size: `1 / Σ w_i²` — collapses to 1 when one particle
  dominates → the FK is doing strong selection.
- Per-particle `energy_traj` (`diffusionv2.py:416`) is already retained inside
  the loop. Just expose it.

### 5.4 "How do trunk perturbations propagate into the structure?"

This is the closest thing to the project's existing mutation-divergence
analysis but on the *diffusion* output. Two clean knobs:

- **Perturb `(s_trunk, z_trunk)`** before they reach
  `DiffusionConditioning.forward`. Add Gaussian noise, swap residue rows, zero
  rows. Measure: per-residue Cα displacement in the final structure, and
  divergence in `x_0_hat` per diffusion step. Combined with the existing
  pairformer-layer divergence story this gives a complete chain
  `Δseq → Δlayer_s → Δstructure` curve.
- **Perturb the conditioning biases** (`atom_enc_bias`, `token_trans_bias`,
  `atom_dec_bias`) after `DiffusionConditioning` but before the loop. Lets
  you isolate which conditioning channel matters for which atoms.

Two perturbations of the *same* sequence that give the same trunk should also
give similar diffusion trajectories — divergences are signal that the
diffusion is amplifying or dampening trunk signal.

### 5.5 "Is the steering actually doing anything?"

Three lesion experiments that fall right out of the existing knobs:
- `fk_steering=False, physical_guidance_update=False` → vanilla unsteered
  diffusion. Compare RMSD / clash-score / PoseBusters to steered.
- `fk_steering=True, num_particles=1` → no resampling possible, just energy
  logging. Use as a free observation channel: get `energy_traj` without it
  influencing the result.
- Replace one potential with a no-op (return zero energy/gradient) → is that
  potential load-bearing for this input?

### 5.6 Capture sites at a glance

| Site | Lives at | Shape | Per |
|---|---|---|---|
| `atom_coords_denoised` (x_0_hat) | `diffusionv2.py:388–396` | `(M, num_atoms, 3)` | step × particle |
| `atom_coords_noisy` (x_t) | `diffusionv2.py:378` | `(M, num_atoms, 3)` | step × particle |
| `eps` (injected noise) | `diffusionv2.py:377` | `(M, num_atoms, 3)` | step × particle |
| `energy` (per-particle) | `diffusionv2.py:406–416` | `(M,)` | resample tick |
| `energy_traj` (history) | `diffusionv2.py:313, 416` | `(M, ticks)` | live |
| `energy_gradient` | `diffusionv2.py:450–464` | `(M, num_atoms, 3)` | step × gd_step × potential |
| `resample_weights` | `diffusionv2.py:436–441` | `(M / num_particles, num_particles)` | resample tick |
| `resample_indices` | `diffusionv2.py:482` | `(M,)` | resample tick |
| `atom_enc_bias`, `atom_dec_bias`, `token_trans_bias` | `diffusion_conditioning.py:13–117` | see §2 | once |

All of these are pure tensors in a Python loop — easiest extraction is to
subclass `AtomDiffusion`, override `sample`, and yield/store. Hooks work too
for the network-forward call but the steering bookkeeping happens at the loop
level, not inside a `nn.Module.forward`, so monkey-patching `sample` (à la the
existing `attention_weights` patch in the extractor) is the most natural fit.

---

## 6. Research ideas this enables

A short list of interpretability questions that map directly onto the hooks
above. Ordered roughly by tractability.

1. **Denoising movies for mutants.** Run the same WT / mutant pair through the
   trunk, then capture every `x_0_hat` from diffusion. RMSD-vs-step curves
   should diverge at a characteristic step — early divergence = the trunk
   already disagrees, late divergence = same trunk but diffusion samples
   different basins. This separates "the model knows it's different" from
   "the diffusion is stochastic enough to explore both options".

2. **Per-residue steering force maps.** Sum `energy_gradient` magnitudes over
   the full trajectory, per residue, per potential. Produces a per-mutation
   "where did physics correct the model?" heatmap. Mutations that the trunk
   handles cleanly should have flat steering maps; mutations that confuse the
   trunk should show steering effort concentrated at the perturbed site.

3. **Particle genealogy under perturbation.** Run with
   `num_particles ∈ {3, 8, 32}` and log resample indices. Effective sample
   size collapse profiles for WT vs mutant — does the mutant trajectory
   require more aggressive selection? When/where does the FK "give up" on
   particles?

4. **Conditioning-channel ablation.** Zero each of `atom_enc_bias`,
   `token_trans_bias`, `atom_dec_bias` independently, measure final-structure
   degradation. Plausible result: encoder bias controls global topology,
   decoder bias controls local geometry. Verifies (or breaks) the
   "trunk-as-blueprint, decoder-as-builder" reading.

5. **Steering vs no-steering divergence as a confidence proxy.** RMSD between
   the unsteered and `--use_potentials` predictions per residue might track
   uncertainty better than pLDDT — high divergence = model needed external
   physics to commit. Cross-check against pLDDT and PAE per residue.

6. **Cross-mutation CKA on diffusion trajectories.** The existing
   layer-of-pairformer CKA story extends naturally: do CKA between
   `x_0_hat_step_k` of mutant A and mutant B, across `k`. Mutants whose
   diffusion trajectories converge (high late-step CKA) but whose early
   trajectories disagree are candidates for "the trunk reads them differently
   but the structure module collapses them to the same answer" — exactly the
   information-bottleneck pattern we see in the trunk.

7. **Custom potentials as interpretability probes.** Adding a hand-written
   `Potential` subclass with `compute_gradient` returning a chosen vector
   field lets you steer the diffusion in arbitrary directions and observe
   which directions the conditioning resists vs. allows. Concretely: push
   residue *i* toward residue *j* — does the model pull it back? At which
   step does it give in?

---

## 7. Cheatsheet

- Diffusion = 200-step EDM denoising of atom coords, runs *once* after the
  trunk (per recycling).
- Score network always predicts `x_0`, not noise (EDM preconditioning).
- Trunk → diffusion via *pre-computed attention biases*, not raw `(s, z)`.
- Two steering modes, both gated by `--use_potentials`:
  - **Gradient guidance** (mode 1): GD on `x_0_hat` against potential energies,
    20 inner steps per diffusion step.
  - **FK / SMC** (mode 2): `num_particles=3` parallel trajectories, resample
    every 3 steps weighted by `softmax(λ · ΔE)`.
- `contact_guidance_update` is *always on* in Boltz-2 if you provide contacts.
- All loop state is plain tensors at known line numbers — subclass
  `AtomDiffusion.sample` to capture, perturb, or steer arbitrarily.
