# Boltz2 Hidden Representation Extraction Sites

Overview of every activation site captured by `Boltz2Extractor`, what it encodes,
and where it sits in the forward pass.

---

## Forward Pass Overview

```
Input tokens
    │
    ▼
┌──────────────────┐
│  InputEmbedder   │  ──►  s_inputs  (B, N, 384)
└────────┬─────────┘
         │
    ┌────▼────┐
    │  Init   │  s₀, z₀ from relative-position encoding + recycled projections
    └────┬────┘
         │
    ╔════▼════════════════════════════════╗
    ║  Recycling loop (default 3 iters)  ║
    ║                                    ║
    ║   MSA Module  ──► updates z        ║
    ║   Template Module (if available)   ║
    ║                                    ║
    ║   ┌────────────────────────────┐   ║
    ║   │  Pairformer (48 layers)    │   ║
    ║   │                            │   ║
    ║   │  for each layer:           │   ║
    ║   │    attention(s, z)  → s    │   ║
    ║   │    tri_mul_out(z)   → z    │   ║
    ║   │    tri_mul_in(z)    → z    │   ║
    ║   │    tri_att_start(z) → z    │   ║
    ║   │    tri_att_end(z)   → z    │   ║
    ║   │    transition_s(s)  → s    │   ║
    ║   │    transition_z(z)  → z    │   ║
    ║   └────────────┬───────────────┘   ║
    ║                │                   ║
    ║   s, z recycled back to Init       ║
    ╚════════════════╪═══════════════════╝
                     │
                     ▼
           ┌─────────────────┐
           │  Distogram      │  ──►  (B, N, N, 64)
           └─────────────────┘
                     │
                     ▼
             Structure Module
           (diffusion sampling)
```

---

## Top-Level Sites (`SITES`)

### `input_embedder`

| | |
|---|---|
| **Shape** | `(B, N, 384)` |
| **Fires** | Once per forward pass (before recycling) |

Raw per-token embeddings built from:

- Atom-level encoder (coordinates, atom types)
- Residue-type one-hot embeddings
- MSA profile (evolutionary conservation)
- Method/molecule-type conditioning flags

This is what the model knows about each residue **before any pairwise reasoning**.
Useful as a baseline to measure how much information subsequent layers add.

---

### `pairformer_s`

| | |
|---|---|
| **Shape** | `(B, N, 384)` |
| **Fires** | Once per recycling step |

Final per-token representation after all 48 pairformer layers.
Has integrated information from every other token via attention, and from
pairwise features via the attention bias term. Encodes the model's
best per-residue structural prediction after full processing.

---

### `pairformer_z`

| | |
|---|---|
| **Shape** | `(B, N, N, 128)` |
| **Fires** | Once per recycling step |

Final pairwise representation. Each entry `z[i,j]` encodes the predicted
relationship between tokens `i` and `j` — inter-residue distances, orientations,
and contact likelihood. This matrix feeds directly into the distogram and
the structure module's coordinate generation.

---

### `distogram`

| | |
|---|---|
| **Shape** | `(B, N, N, 64)` |
| **Fires** | Once (after last recycling step) |

Predicted pairwise distance distribution logits (64 distance bins, typically
spanning 0–50 A). Computed as `linear(z + z^T)` to enforce symmetry.
These are unnormalized logits — apply softmax to get probabilities.
Used as an auxiliary training loss and for confidence metrics (pAE, pLDDT).

---

## Per-Layer Sites (`LAYER_SITES`)

Captured for each selected pairformer layer (0–47).
Key naming: `pairformer_{layer_idx}_{site_name}`.

### `layer_s` / `layer_z`

| | |
|---|---|
| **Shape** | `(B, N, 384)` / `(B, N, N, 128)` |

The sequence and pairwise representations at the **output** of each
pairformer layer (after all sublayers + residual connections). Tracking
these across layers reveals how the model progressively refines its
predictions:

- **Early layers (0–10):** local contact patterns, secondary structure
- **Middle layers (10–30):** long-range information propagation
- **Late layers (30–47):** final refinement, resolving ambiguities

---

### `attention`

| | |
|---|---|
| **Shape** | `(B, N, 384)` |

Output of the `AttentionPairBias` module (the value after gating and
output projection). This is the sequence-track update contributed by
the attention mechanism at each layer — the delta added to `s` via
the residual connection.

---

### `attention_weights`

| | |
|---|---|
| **Shape** | `(B, H, N, N)` where `H = 16` heads |

Post-softmax attention probabilities. Each head's matrix sums to 1 along
the key dimension. Shows which tokens each position attends to.

Captured via monkey-patching (the softmax output is a local variable inside
`AttentionPairBias.forward`, not a module return value).

Useful for:
- Identifying predicted contacts and interactions
- Attention rollout analysis
- Per-head specialisation studies

---

### `tri_mul_out` / `tri_mul_in`

| | |
|---|---|
| **Shape** | `(B, N, N, 128)` |

Triangle multiplication modules. These enforce geometric consistency
by reasoning over triplets of residues:

- **Outgoing:** `z[i,j] += sum_k  a[i,k] * b[k,j]`
  "If residue i is close to k, and k is close to j, then i and j
  should be aware of each other."

- **Incoming:** `z[i,j] += sum_k  a[i,k] * b[j,k]`
  Complementary direction — edges sharing a common endpoint.

Together they implement the **triangle inequality constraint** from
AlphaFold2/3: if distances (i,k) and (k,j) are both short, then
distance (i,j) must also be bounded.

---

### `tri_att_start` / `tri_att_end`

| | |
|---|---|
| **Shape** | `(B, N, N, 128)` |

Triangle attention modules. Multi-head attention applied to the pairwise
representation along one axis at a time:

- **Starting node:** each row `z[i, :, :]` attends across other rows
  (information flows from source residues)
- **Ending node:** each column `z[:, j, :]` attends across other columns
  (information flows from target residues)

These complement the triangle multiplications by allowing learned,
data-dependent information routing in pairwise space.

---

### `transition_s` / `transition_z`

| | |
|---|---|
| **Shape** | `(B, N, 384)` / `(B, N, N, 128)` |

Two-layer gated MLPs (SiLU activation, 4x expansion factor) applied
independently to each position. These add nonlinear capacity after
the attention and triangle updates, acting as per-position "refinement"
steps. Applied residually.

---

## Recycling

The pairformer runs multiple times with the same weights. After each pass,
`s` and `z` are projected and added back to the initial representations:

```
s_next = s_init + s_recycle_proj(norm(s_prev))
z_next = z_init + z_recycle_proj(norm(z_prev))
```

This iterative refinement lets the model progressively sharpen its
predictions. The extractor stores one activation dict per recycling step —
use `get_step(-1)` for the final (most refined) step, or `--recycling_step -2`
to save all steps for analysing how representations evolve.

---

## Dimension Reference

| Site | Shape | Dim name |
|------|-------|----------|
| `input_embedder` | `(B, N, 384)` | token_s |
| `pairformer_s` | `(B, N, 384)` | token_s |
| `pairformer_z` | `(B, N, N, 128)` | token_z |
| `distogram` | `(B, N, N, 64)` | distance bins |
| `layer_s` | `(B, N, 384)` | token_s |
| `layer_z` | `(B, N, N, 128)` | token_z |
| `attention` | `(B, N, 384)` | token_s |
| `attention_weights` | `(B, 16, N, N)` | heads |
| `tri_mul_out` | `(B, N, N, 128)` | token_z |
| `tri_mul_in` | `(B, N, N, 128)` | token_z |
| `tri_att_start` | `(B, N, N, 128)` | token_z |
| `tri_att_end` | `(B, N, N, 128)` | token_z |
| `transition_s` | `(B, N, 384)` | token_s |
| `transition_z` | `(B, N, N, 128)` | token_z |
