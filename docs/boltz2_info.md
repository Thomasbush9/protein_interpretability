# Boltz-2: architecture, forward pass, and extraction sites

Reference for the Boltz-2 model as used in this project: what happens during a
forward pass, what each module computes, and which hidden representations we
extract for interpretability analyses.

For a dedicated reference table of capture sites (shapes, fire points, how the
extractor hooks them), see also
[`boltz2_extraction_sites.md`](boltz2_extraction_sites.md).

---

## 1. High-level overview

Boltz-2 is a structure-prediction model in the AlphaFold-2/3 lineage. Given a
sequence (optionally with MSA, templates, ligands, and conditioning flags), it
predicts 3D atom coordinates plus per-residue and pairwise confidence.

Representation flow:

```
tokens ──► input_embedder ──► MSA module ──► pairformer stack (×48) ──► structure module
                                         ▲                           │
                                         └───── recycling (×N) ──────┘
```

Two main tracks are carried through the trunk:

- **Single track** `s ∈ (N, 384)` — one vector per residue.
- **Pair track**  `z ∈ (N, N, 128)` — one vector per residue pair.

The pairformer alternates updates on `z` (geometry / contact reasoning) and `s`
(per-residue, conditioned on `z`). After the trunk, the structure module
decodes atom coordinates by diffusion.

---

## 2. Forward pass, step by step

### 2.1 Input featurization

- Tokenizer builds residue-level features (residue-type one-hot, method /
  molecule-type conditioning, optional modified residues).
- An atom-level encoder provides atom coordinates and atom types.
- MSA profile adds evolutionary conservation (can be empty → "MSA-free" mode).
- Relative-position encoding seeds the pair track.

### 2.2 `InputEmbedder`

- Combines the above into the **initial single track** `s_inputs`.
- **Extraction site**: `input_embedder` — `(B, N, 384)`. Captures what the model
  knows about each residue *before any pairwise reasoning*. Good baseline for
  measuring how much later layers add.

### 2.3 Trunk initialisation

From `s_inputs` and relative-position features, the trunk produces the initial
`(s₀, z₀)` used as the starting point for the first recycling iteration.

### 2.4 Recycling loop (default 3 iterations)

At the start of each iteration, previously recycled `(s, z)` are added back to
the initial state:

```
s_next = s_init + s_recycle_proj(LN(s_prev))
z_next = z_init + z_recycle_proj(LN(z_prev))
```

Then inside the loop:

1. **MSA module** — attends over MSA rows/columns; writes updates into `z` via
   outer-product mean so evolutionary signal reaches the pair track.
2. **Template module** (if templates are provided) — injects structural priors
   into `z`.
3. **Pairformer stack** — 48 blocks that update `s` and `z` (see §2.5).

Our hidden-rep files store one snapshot per recycling step
(`step_0 … step_{N-1}`). Later steps are more refined; earlier steps are closer
to sequence/MSA input.

### 2.5 Pairformer block (×48)

Each block applies these sublayers in order. Unless noted, updates are residual
(`x ← x + sublayer(x)`).

1. **`AttentionPairBias`** — single-track self-attention over residues, with
   per-head logits biased by a projection of `z`. This is the step where pair
   geometry conditions the single track.
   - Extraction: `attention` (the value after gating + output projection),
     `attention_weights` (the post-softmax attention probabilities; monkey-patched
     because softmax output is a local inside `AttentionPairBias.forward`).
2. **`TriangleMultiplication` outgoing** — `z[i,j] += Σ_k a[i,k] * b[k,j]`.
   Enforces triangle-like consistency along shared "outgoing" endpoints.
   - Extraction: `tri_mul_out`.
3. **`TriangleMultiplication` incoming** — `z[i,j] += Σ_k a[i,k] * b[j,k]`.
   Complementary direction.
   - Extraction: `tri_mul_in`.
4. **`TriangleAttention` starting node** — attention along rows `z[i, :, :]`.
   - Extraction: `tri_att_start`.
5. **`TriangleAttention` ending node** — attention along columns `z[:, j, :]`.
   - Extraction: `tri_att_end`.
6. **`Transition` on `z`** — 2-layer gated MLP (SiLU, ~4× expansion) applied
   position-wise.
   - Extraction: `transition_z`.
7. **`Transition` on `s`** — same, for the single track.
   - Extraction: `transition_s`.

At the end of the block, `s` and `z` are the updated per-residue and per-pair
representations.

- Extraction: `layer_s` — `(B, N, 384)` and `layer_z` — `(B, N, N, 128)` at the
  **output** of each pairformer block (after all sublayers + residuals).
  Named `pairformer_{i}_layer_s` / `pairformer_{i}_layer_z` with `i ∈ [0, 47]`.

After the last block, the trunk outputs:

- `pairformer_s` — `(B, N, 384)` final single track.
- `pairformer_z` — `(B, N, N, 128)` final pair track.

These feed both into the next recycling iteration and into downstream heads.

### 2.6 Distogram head

A linear head on `z + z^T` (symmetrised) produces:

- **Extraction: `distogram`** — `(B, N, N, 64)` unnormalised logits over 64
  distance bins (roughly 0–50 Å). Softmax gives the predicted distance
  distribution; used as an auxiliary loss and for confidence metrics.

### 2.7 Structure module (atom decoder)

- Diffusion-based: conditioned on `(s, z)`, it iteratively denoises atom
  coordinates to produce the final 3D structure.
- Confidence heads (pLDDT, PAE, pTM) read from trunk features + predicted
  structure.

### 2.8 Auxiliary heads (Boltz-2-specific)

- **Affinity head**: consumes trunk features for protein–ligand affinity
  prediction when ligand inputs are provided.

---

## 3. Extraction sites — what each one is good for

| Site | Shape | Fires | Intuition / use |
|---|---|---|---|
| `input_embedder` | `(B, N, 384)` | once per forward | Pre-trunk per-residue embedding. Difference between two sequences here ≈ sequence-identity difference, untouched by structural reasoning. Baseline. |
| `pairformer_s` | `(B, N, 384)` | once per recycling step | Final single track after all 48 blocks. Best per-residue summary the model commits to. |
| `pairformer_z` | `(B, N, N, 128)` | once per recycling step | Final pair track. Encodes predicted inter-residue geometry. Feeds distogram + structure module. |
| `distogram` | `(B, N, N, 64)` | once (last recycling step) | Logits over distance bins. Softmax → distance distribution. Use for direct contact / distance comparisons. |
| `layer_s` (`pairformer_i_layer_s`) | `(B, N, 384)` | per block | Single-track trajectory across depth. Early = local sequence/secondary-structure features; mid = long-range propagation; late = refinement. Our default for divergence analyses. |
| `layer_z` (`pairformer_i_layer_z`) | `(B, N, N, 128)` | per block | Pair-track trajectory. Use to separate "local sequence change" from "long-range geometry change". |
| `attention` | `(B, N, 384)` | per block | Output of `AttentionPairBias` (the single-track delta from attention only). Useful for isolating attention's contribution vs MLP/triangle updates. |
| `attention_weights` | `(B, 16, N, N)` | per block | Post-softmax attention probabilities, 16 heads. Direct view of "who attends to whom". Supports attention-rollout analyses and per-head specialisation studies. Monkey-patched to capture (not a module return value). |
| `tri_mul_out` | `(B, N, N, 128)` | per block | Outgoing triangle multiplication update. Implements triangle-inequality consistency along outgoing edges. |
| `tri_mul_in` | `(B, N, N, 128)` | per block | Incoming triangle multiplication update. Complementary direction. |
| `tri_att_start` | `(B, N, N, 128)` | per block | Triangle attention over rows (starting-node axis). Learned, data-dependent geometry routing. |
| `tri_att_end` | `(B, N, N, 128)` | per block | Triangle attention over columns (ending-node axis). |
| `transition_s` | `(B, N, 384)` | per block | Post-MLP single-track update. Separates "attention+triangle update" from "MLP refinement". |
| `transition_z` | `(B, N, N, 128)` | per block | Post-MLP pair-track update. |

Dimension names: `token_s = 384`, `token_z = 128`, `heads = 16`, `distance bins = 64`.

---

## 4. Relevance for mutation-divergence analyses

- **Default site**: `layer_s`. Lives in residue space, is contextual, and is
  what the structure module ultimately consumes. Gives per-layer × per-step
  divergence curves.
- **Why divergence curves have the shape they do**:
  - *Early layers*: `s` is close to the input embedding → mutants differ mainly
    at mutated positions → low, locally concentrated divergence.
  - *Mid layers*: pair-biased attention + triangle ops propagate mutation
    effects via `z` → divergence spreads across residues; cohorts separate.
  - *Late layers*: representations converge toward structure-grounded features.
    When structure is conserved (TM ≈ 1), late-layer `s` also converges —
    which is why global divergence can *decrease* for function-only
    disruptions.
  - *Recycling steps*: each step re-reads the previous prediction as a prior.
    Later steps commit more strongly to a structural solution, so per-step
    divergence patterns can shift.
- **`layer_z` vs `layer_s`**: running the same divergence analysis on `layer_z`
  tells you whether a mutation perturbs *geometry* (contacts, distances)
  beyond the single-residue feature — complementary signal to `layer_s`.
- **`tri_*` outputs**: isolate the geometry-update step. If divergence appears
  there but is damped at `transition_s`, the MLP is compressing the change
  (information-bottleneck signal).
- **`attention_weights`**: ask "which residues shift their attention under
  mutation?" — candidate for "how does the model route mutation information?"
- **`input_embedder`**: useful control — subtracting input-embedder divergence
  from layer-wise divergence separates "it's just sequence identity" from
  "the model is reasoning about this mutation".

---

## 5. Quick cheatsheet

- 48 pairformer blocks.
- 16 attention heads in `AttentionPairBias`.
- `s`: 384-dim per residue; `z`: 128-dim per pair.
- Recycling: default 3 iterations; our extractor can store all steps
  (`step_0 … step_{N-1}`).
- Use `get_step(-1)` for the most refined step; `--recycling_step -2` to save
  all steps for depth-of-refinement analyses.
