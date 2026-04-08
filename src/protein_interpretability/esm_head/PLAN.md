# ESM-Guided Refinement of ProteinMPNN Predictions

## Goal

Use ESM3 as a post-hoc refinement step for ProteinMPNN AR predictions.
ProteinMPNN knows structure, ESM3 knows sequence evolution. Combining
both should produce sequences that are structurally valid AND evolutionarily
plausible.

## Pipeline Overview

```
Input: protein backbone structure (PDB)
                    │
                    ▼
        ┌───────────────────┐
        │   ProteinMPNN AR  │  → initial sequence S₀ + per-position log_probs
        └────────┬──────────┘
                 │
                 ▼
        ┌───────────────────┐
        │   ESM3 scoring    │  → per-position pseudo-log-likelihood for S₀
        └────────┬──────────┘
                 │
                 ▼
        ┌───────────────────┐
        │   Identify weak   │  → positions where ESM3 disagrees with MPNN
        │   positions       │     (low ESM PLL or high entropy)
        └────────┬──────────┘
                 │
                 ▼
        ┌───────────────────┐
        │  Refine           │  → Option A: product of experts (blend logits)
        │                   │  → Option B: re-run MPNN with weak positions
        │                   │     designable, rest fixed
        └────────┬──────────┘
                 │
                 ▼
        Output: refined sequence S₁ (optionally iterate N rounds)
```

## Step-by-Step Implementation

### Step 1: Generate initial sequences with ProteinMPNN

Use the existing `protein_mpnn_run.py` or write a lightweight wrapper.
For each input PDB, generate K sequences with temperature sampling.
Save: sequences, per-position log_probs.

```python
# Pseudocode
mpnn_model = load_proteinmpnn(checkpoint_path)
for pdb in input_pdbs:
    sequences, log_probs = mpnn_sample(mpnn_model, pdb, num_seqs=8, temperature=0.1)
    # sequences: list of str
    # log_probs: (K, L, 21) tensor
```

### Step 2: Score with ESM3

For each generated sequence, compute ESM3's masked marginal log-likelihoods.
Two approaches:

**Fast (1 forward pass per sequence):** Mask ALL positions, get ESM3's one-shot
prediction. Quick but doesn't capture positional context.

**Accurate (L forward passes per sequence):** Mask one position at a time,
compute conditional log-probability. Expensive but proper pseudo-likelihood.

**Compromise:** Mask positions in batches of ~15% (like MLM), average over
multiple random masks. ~7 forward passes for full coverage.

```python
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein

esm_model = ESM3.from_pretrained("esm3-open").to("cuda")

def esm3_score(model, sequence):
    """Compute per-position pseudo-log-likelihood using masked marginals."""
    protein = ESMProtein(sequence=sequence)
    tokens = model.encode(protein)
    L = len(sequence)
    pll = torch.zeros(L)

    # Mask each position, predict
    for i in range(L):
        masked_tokens = tokens.clone()
        masked_tokens.sequence[0, i+1] = MASK_TOKEN  # +1 for BOS
        logits = model.forward(masked_tokens)
        log_probs = F.log_softmax(logits.sequence_logits[0, i+1], dim=-1)
        pll[i] = log_probs[tokens.sequence[0, i+1]].item()

    return pll  # (L,) — higher = more plausible
```

### Step 3: Product of Experts (simplest refinement)

Blend ProteinMPNN and ESM3 logits at each position:

```python
def refine_product_of_experts(mpnn_log_probs, esm_log_probs, alpha=0.5):
    """
    mpnn_log_probs: (L, 21) from ProteinMPNN
    esm_log_probs:  (L, 20 or 21) from ESM3 (may need vocab alignment)
    alpha: weight for ESM3 contribution
    """
    # NOTE: ESM3 uses a different vocab (32 tokens including special).
    # Need to map ESM3's amino acid logits to MPNN's 21-class vocab.
    esm_mapped = map_esm3_to_mpnn_vocab(esm_log_probs)

    combined = mpnn_log_probs + alpha * esm_mapped
    refined_sequence = combined.argmax(dim=-1)
    return refined_sequence
```

### Step 4: Iterative Refinement (stronger)

```python
def iterative_refine(mpnn_model, esm_model, structure, n_rounds=3, k_remask=0.15):
    # Initial prediction
    seq, mpnn_logprobs = mpnn_sample(mpnn_model, structure)

    for round in range(n_rounds):
        # Score with ESM3
        esm_pll = esm3_score(esm_model, seq)

        # Find worst positions
        n_remask = int(len(seq) * k_remask)
        worst_positions = esm_pll.argsort()[:n_remask]

        # Re-run MPNN with worst positions as designable, rest fixed
        seq, mpnn_logprobs = mpnn_sample(
            mpnn_model, structure,
            fixed_positions=all_except(worst_positions),
            initial_sequence=seq
        )

    return seq
```

## Vocab Mapping: ESM3 ↔ ProteinMPNN

ESM3 uses a 64-token vocabulary (including structure/function tokens).
The sequence tokens are at specific indices. ProteinMPNN uses 21 classes
(20 standard AAs + X). Need a mapping table.

```python
MPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'  # 21 classes
# ESM3 sequence token indices need to be extracted from the tokenizer
# esm.tokenization.sequence_tokenizer.SequenceTokenizer
```

This mapping is critical — get it wrong and the logit blending is meaningless.

## Scripts to Create

| Script | Purpose |
|--------|---------|
| `vocab_mapping.py` | ESM3 ↔ MPNN vocab alignment utilities |
| `mpnn_predict.py` | Run ProteinMPNN on input PDBs, save sequences + logits |
| `esm_score.py` | Score sequences with ESM3 masked marginals |
| `refine.py` | Product-of-experts or iterative refinement |
| `evaluate.py` | Compare refined vs original: recovery, ESM PPL, diversity |

## Evaluation Plan

Test on CATH validation set or a small PDB test set:

| Method | Description |
|--------|-------------|
| MPNN baseline | Standard AR ProteinMPNN |
| ESM3 filter | Generate 8 seqs, pick best ESM3 score |
| Product of experts | Blend MPNN + ESM3 logits (sweep alpha) |
| Iterative refinement | N=1,3,5 rounds of ESM-guided re-masking |

Metrics:
- Sequence recovery (vs native)
- ESM3 pseudo-perplexity
- ProteinMPNN log-probability
- Diversity (among generated sequences)
- scTM score (AlphaFold2 foldability, if compute allows)

## Dependencies

- ProteinMPNN (this repo's parent project)
- ESM3: `pip install esm` (already installed on cluster)
- huggingface_hub (for ESM3 model download)

## Notes

- ESM3 is ~1.4B parameters. Single A100/H100 should be fine for inference.
- Per-position masked marginal scoring is O(L) forward passes — expensive
  for long proteins. Batch-masking approach reduces this to ~7 passes.
- Start with product-of-experts (simplest, 1 ESM forward pass per sequence).
  Move to iterative refinement if results look promising.
