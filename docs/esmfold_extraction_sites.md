# ESMFold Hidden Representation Extraction Sites

Overview of every activation site captured by `ESMFoldExtractor`, what it
encodes, and where it sits in the forward pass.

---

## Forward Pass Overview

ESMFold has **two distinct stages**: a pure language model (ESM-2) followed by
an AlphaFold2-style folding trunk. Only the trunk produces pairwise `z`
representations; the ESM-2 backbone is a standard transformer that outputs
per-residue embeddings only.

```
Input sequence (amino acid string)
    |
    v
+---------------------+
|  ESM-2 Backbone     |   36 transformer layers (standard LM)
|  (frozen weights)   |   per-layer output: (B, L, 2560)
+----------+----------+
           |
    esm_s_combine (learned weighted sum across all 36 layers)
           |
           v
+---------------------+
|  esm_s_mlp          |   projects combined ESM2 repr to trunk dim
+----------+----------+   (B, L, 2560) -> (B, L, 1024)
           |
           v
   s_0 = esm_s_mlp(combined)
   z_0 = outer_product(s_0) + positional_embedding
           |
    +======v==============================+
    |  Recycling loop (default 4 iters)   |
    |                                     |
    |   +---------------------------+     |
    |   |  Folding Trunk (48 blks)  |     |
    |   |                           |     |
    |   |  for each block:          |     |
    |   |    seq_attention(s, z) -> s     |
    |   |    tri_mul_out(z)     -> z     |
    |   |    tri_mul_in(z)      -> z     |
    |   |    tri_att_start(z)   -> z     |
    |   |    tri_att_end(z)     -> z     |
    |   |    mlp_seq(s)         -> s     |
    |   |    mlp_pair(z)        -> z     |
    |   +-------------+-------------+     |
    |                 |                   |
    |   Structure Module (IPA)            |
    |     -> frames, positions            |
    |                 |                   |
    |   s, z, distogram recycled back     |
    +================|====================+
                     |
                     v
         +-------------------+
         |  Prediction Heads |
         +-------------------+
           distogram_head  -> (B, L, L, 64)
           lddt_head       -> (B, L, 50)
           ptm_head        -> (B, L, L, bins)
```

---

## Architecture Dimensions

| Component | Hidden dim | Notes |
|-----------|-----------|-------|
| ESM-2 backbone | 2560 | 36 layers, 40 attention heads |
| Trunk sequence state (`s`) | 1024 | = `sequence_state_dim` |
| Trunk pair state (`z`) | 128 | = `pairwise_state_dim` |
| Trunk blocks | 48 | evoformer-like |
| Distogram bins | 64 | distance bins |
| pLDDT bins | 50 | confidence bins |

---

## Top-Level Sites (`SITES`)

### `esm_s_combined`

| | |
|---|---|
| **Shape** | `(B, L, 2560)` |
| **Fires** | Once per forward pass (before recycling) |

Weighted combination of all 36 ESM-2 layer hidden states, **before** the
MLP projection to trunk dimension. This is the language model's final
representation of each residue — what it "knows" from sequence context
alone, without any structure-aware processing.

Captured by hooking the input to `model.esm_s_mlp`.

---

### `trunk_s`

| | |
|---|---|
| **Shape** | `(B, L, 1024)` |
| **Fires** | Once (after all recycling) |

Final per-residue representation from the folding trunk, after all 48
blocks and all recycling iterations. Has integrated pairwise geometric
information via the attention bias from `z`. Analogous to Boltz's
`pairformer_s`.

---

### `trunk_z`

| | |
|---|---|
| **Shape** | `(B, L, L, 128)` |
| **Fires** | Once (after all recycling) |

Final pairwise representation. Each entry `z[i,j]` encodes the predicted
spatial relationship between residues `i` and `j`. This does **not** exist
in the ESM-2 backbone — it is constructed by the folding trunk from outer
products of `s` and refined through triangle modules.

Analogous to Boltz's `pairformer_z`.

---

### `structure_out`

| | |
|---|---|
| **Type** | `dict` of tensors |
| **Fires** | Once per recycling step |

Full output of the structure module (IPA + backbone update), including
frames, sidechain frames, angles, and intermediate states. Stored as a
dict — use selectively as it can be large.

---

### `distogram`

| | |
|---|---|
| **Shape** | `(B, L, L, 64)` |
| **Fires** | Once (after model forward) |

Predicted pairwise distance distribution logits (64 bins). Computed from
the final pairwise state via `distogram_head(z + z^T)`. Apply softmax
to get probabilities.

---

### `lddt_logits`

| | |
|---|---|
| **Shape** | `(B, L, 50)` |
| **Fires** | Once |

Per-residue confidence logits from the lDDT head. The argmax gives the
predicted local distance difference test bin; the expected value gives
pLDDT. Higher pLDDT = more confident in local structure accuracy.

---

### `ptm_logits`

| | |
|---|---|
| **Shape** | `(B, L, L, bins)` |
| **Fires** | Once |

Predicted TM-score logits from the pTM head. Used to compute the overall
predicted TM-score and the predicted aligned error (PAE) matrix.

---

### `positions`

| | |
|---|---|
| **Shape** | `(num_recycles+1, B, L, 14, 3)` |
| **Source** | Model output (not a hook) |

Predicted 3D atom coordinates across all recycling iterations. 14 atoms
per residue (backbone N, CA, C, O + up to 10 sidechain atoms). Available
directly from `model(**inputs).positions` — included in `model_outputs.pt`
rather than `hidden_reps.pt`.

---

## Per-Layer Sites (`LAYER_SITES`)

### Folding Trunk (per-block)

Captured for each selected trunk block (0--47).
Key naming: `trunk_{block_idx}_{site_name}`.

#### `layer_s` / `layer_z`

| | |
|---|---|
| **Shape** | `(B, L, 1024)` / `(B, L, L, 128)` |

The sequence and pairwise representations at the **output** of each trunk
block (after all submodules + residual connections). Tracking these across
blocks shows how the model progressively builds structural understanding:

- **Early blocks (0--10):** local contacts, secondary structure signals
- **Middle blocks (10--30):** long-range information propagation
- **Late blocks (30--47):** final refinement, geometric consistency

#### `seq_attention`

| | |
|---|---|
| **Shape** | `(B, L, 1024)` |

Output of the gated self-attention module on the sequence track.
This is the delta added to `s` via the residual connection.

#### `tri_mul_out` / `tri_mul_in`

| | |
|---|---|
| **Shape** | `(B, L, L, 128)` |

Triangle multiplication modules (same role as in Boltz/AlphaFold2).
Enforce geometric consistency via triplet reasoning:
- **Outgoing:** `z[i,j] += sum_k a[i,k] * b[k,j]`
- **Incoming:** `z[i,j] += sum_k a[i,k] * b[j,k]`

#### `tri_att_start` / `tri_att_end`

| | |
|---|---|
| **Shape** | `(B, L, L, 128)` |

Triangle attention along rows (starting node) or columns (ending node).
Learned, data-dependent information routing in pairwise space.

#### `transition_s` / `transition_z`

| | |
|---|---|
| **Shape** | `(B, L, 1024)` / `(B, L, L, 128)` |

Per-position MLPs applied after attention and triangle updates.
Mapped from `mlp_seq` and `mlp_pair` submodules respectively.

---

### ESM-2 Backbone (per-layer)

#### `esm_layer`

| | |
|---|---|
| **Shape** | `(B, L, 2560)` |
| **Key** | `esm_{layer_idx}_layer` |

Hidden state output from each ESM-2 transformer layer (0--35).
This is a standard transformer LM representation — **no pairwise `z`
exists at this stage**. The folding trunk constructs `z` downstream.

These are useful for:
- Studying what the LM learns vs. what the structure module adds
- Comparing to other protein LMs (ESM-2 standalone, ProtBERT, etc.)
- Layer-wise probing for sequence properties (secondary structure,
  solvent accessibility) before structure-aware processing

#### `esm_attention_out`

| | |
|---|---|
| **Shape** | `(B, L, 2560)` |
| **Key** | `esm_{layer_idx}_attention_out` |

Output of the attention sublayer (EsmAttention) — after the QKV
projection, multi-head attention, output projection, and residual
connection. This is the representation **after attention but before
the FFN**, useful for isolating what attention contributes at each layer.

Hooked via `model.esm.encoder.layer[i].attention`.

#### `esm_attention_weights`

| | |
|---|---|
| **Shape** | `(B, H, L, L)` where `H = 40` heads |
| **Key** | `esm_{layer_idx}_attention_weights` |

Post-softmax attention probabilities from each ESM-2 layer. Each head's
matrix sums to 1 along the key dimension. In eval mode these are
pre-dropout (dropout is a no-op).

Captured via monkey-patching `EsmSelfAttention.forward` — the original
forward is wrapped to force `output_attentions=True` internally, the
attention probs are captured, and the return value is restored to the
original format so downstream code is unaffected.

Useful for:
- Attention rollout / head-level analysis
- Identifying which residues the LM attends to (contacts, motifs)
- Per-head specialization studies
- Comparing attention patterns between WT and mutants

**Memory warning:** `(B, 40, L, L)` per layer is O(L^2). For L=500
across 36 layers this is ~1.4 GB. Select specific `esm_layers` to limit.

#### `esm_ffn_out`

| | |
|---|---|
| **Shape** | `(B, L, 2560)` |
| **Key** | `esm_{layer_idx}_ffn_out` |

Output of the FFN sublayer (EsmOutput) — after the two-layer MLP
(expansion to 10240, GELU, contraction back to 2560) plus the residual
connection from the attention output. This is the same tensor as
`esm_layer` for non-decoder models, but captured at the FFN submodule
level for completeness.

Hooked via `model.esm.encoder.layer[i].output`.

---

## Recycling

The folding trunk runs `num_recycles + 1` times (the first pass is the
"standard" forward, subsequent passes are recycling iterations). After each
pass the structure module predicts coordinates, a distogram is computed from
those, and `(s, z, distogram_bins)` are recycled back:

```
s_next = s_init + recycle_s_norm(s_prev)
z_next = z_init + recycle_z_norm(z_prev) + recycle_disto(dist_bins)
```

The extractor stores one activation dict per recycling step. Use
`--recycling_steps_to_save last` (default) for the final iteration, or
`all` to study convergence across recycles.

---

## Dimension Reference

| Site | Shape | Dim |
|------|-------|-----|
| `esm_s_combined` | `(B, L, 2560)` | ESM-2 hidden |
| `esm_layer` | `(B, L, 2560)` | ESM-2 hidden |
| `esm_attention_out` | `(B, L, 2560)` | ESM-2 hidden |
| `esm_attention_weights` | `(B, 40, L, L)` | ESM-2 heads |
| `esm_ffn_out` | `(B, L, 2560)` | ESM-2 hidden |
| `trunk_s` | `(B, L, 1024)` | trunk seq |
| `trunk_z` | `(B, L, L, 128)` | trunk pair |
| `layer_s` | `(B, L, 1024)` | trunk seq |
| `layer_z` | `(B, L, L, 128)` | trunk pair |
| `seq_attention` | `(B, L, 1024)` | trunk seq |
| `tri_mul_out` | `(B, L, L, 128)` | trunk pair |
| `tri_mul_in` | `(B, L, L, 128)` | trunk pair |
| `tri_att_start` | `(B, L, L, 128)` | trunk pair |
| `tri_att_end` | `(B, L, L, 128)` | trunk pair |
| `transition_s` | `(B, L, 1024)` | trunk seq |
| `transition_z` | `(B, L, L, 128)` | trunk pair |
| `distogram` | `(B, L, L, 64)` | dist bins |
| `lddt_logits` | `(B, L, 50)` | lDDT bins |
| `ptm_logits` | `(B, L, L, bins)` | TM bins |

---

## Comparison with Boltz2

| | ESMFold | Boltz2 |
|---|---|---|
| Backbone | ESM-2 (36 layers, 2560d) | InputEmbedder (384d) |
| Pairwise origin | Constructed in trunk from s | Initialized from MSA + templates |
| Trunk blocks | 48 (evoformer-like) | 48 (pairformer) |
| Seq dim | 1024 | 384 |
| Pair dim | 128 | 128 |
| Recycling | 4 (default) | 3 (default) |
| Structure module | IPA | Diffusion |
| MSA required | No | Yes |
| Attention weights | Not monkey-patched (yet) | Monkey-patched |
