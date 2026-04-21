# ESMFold Hidden Representation Extraction — How To Launch

## Prerequisites

**Cluster environment** — the ESMFold env needs:
- `transformers` (HuggingFace)
- `torch` (with CUDA)
- `pyyaml`
- `sentencepiece`

The SLURM script activates the env at:
```
/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/esmfold
```
Edit `scripts/run_esmfold_extract.slrm` if your env path differs.

**Model weights** — either:
- A local checkout (set `model.name` in config to the absolute path), or
- The HuggingFace hub id `facebook/esmfold_v1` (downloads on first run).

---

## Quick Start

### 1. Edit the config

```bash
cp scripts/esmfold_extract_config.yaml scripts/my_esmfold_config.yaml
vim scripts/my_esmfold_config.yaml
```

Key fields to set:
```yaml
input:
  sequences_dir: /path/to/fastas     # directory with .fasta files

output:
  out_dir: /path/to/output           # results go here

model:
  name: /path/to/local/esmfold_v1    # or facebook/esmfold_v1
  num_recycles: 4
  fp16: false

extraction:
  sites: [trunk_s]
  trunk_layers: all
  esm_layers: all
  layer_sites: [layer_s, esm_layer]
  recycling_steps_to_save: last

runtime:
  num_gpus: 1
```

### 2a. Launch via SLURM (recommended for cluster)

```bash
# Default config
sbatch scripts/run_esmfold_extract.slrm

# Custom config
sbatch scripts/run_esmfold_extract.slrm /abs/path/to/my_esmfold_config.yaml

# Multi-GPU: set both --gpus-per-node AND runtime.num_gpus
sbatch --gpus-per-node=4 scripts/run_esmfold_extract.slrm my_config.yaml
# (and set runtime.num_gpus: 4 in the config)
```

### 2b. Launch directly with Python

```bash
# Activate env first
mamba activate /path/to/esmfold/env
export PYTHONPATH=/path/to/protein_interpretability/src:$PYTHONPATH

# Via orchestrator (multi-GPU chunking, logging)
python scripts/run_esmfold_extract.py --config scripts/esmfold_extract_config.yaml

# Direct single-process (no orchestrator)
python -m protein_interpretability.extract_hidden_reps_esmfold \
    /path/to/fastas \
    --out_dir ./esmfold_reps \
    --model_name /path/to/esmfold_v1 \
    --num_recycles 4 \
    --sites trunk_s \
    --layer_sites layer_s,esm_layer \
    --trunk_layers all \
    --esm_layers all
```

---

## Input Format

The orchestrator discovers FASTA files recursively under `sequences_dir`.
Supported extensions: `.fasta`, `.fa`, `.faa`.

Works with both layouts:

**Flat directory:**
```
sequences/
  protein_A.fasta
  protein_B.fasta
```

**Nested (Boltz-style) directory:**
```
sequences/
  seq_001/
    seq_001.yaml    # ignored by ESMFold
    seq_001.fasta   # picked up
  seq_002/
    seq_002.fasta
```

Each FASTA can contain one or many sequences. The record ID is taken from
the FASTA header (first word after `>`).

---

## Output Layout

```
<out_dir>/
  <record_id>/
    model_outputs.pt    — positions, plddt, ptm, distogram_logits, etc.
    hidden_reps.pt      — all extractor activations (selected recycling steps)
    metadata.json       — shapes, sizes, config parameters
  _staging/             — temporary symlink chunks (safe to delete)
  _log_chunk_0.out      — per-GPU subprocess log
```

### Loading outputs

```python
import torch, json

# Model predictions
out = torch.load("output/protein_A/model_outputs.pt")
out["plddt"]        # (1, L) per-residue confidence
out["positions"]    # (num_recycles+1, 1, L, 14, 3) atom coords

# Hidden representations
reps = torch.load("output/protein_A/hidden_reps.pt")
reps["trunk_s"]         # (1, L, 1024)  final trunk sequence state
reps["trunk_0_layer_s"] # (1, L, 1024)  block 0 sequence output
reps["esm_0_layer"]     # (1, L, 2560)  ESM2 layer 0 hidden state

# Metadata
meta = json.load(open("output/protein_A/metadata.json"))
meta["sequence_length"]
meta["hidden_rep_shapes"]
```

---

## Extraction Sites Reference

See `docs/esmfold_extraction_sites.md` for a detailed description of every
site, what it encodes, expected shapes, and the forward pass diagram.

**Short summary:**

| Site | Shape | What |
|------|-------|------|
| `esm_s_combined` | `(B, L, 2560)` | Combined ESM2 output (before trunk) |
| `trunk_s` | `(B, L, 1024)` | Final trunk sequence state |
| `trunk_z` | `(B, L, L, 128)` | Final trunk pair state |
| `layer_s` | `(B, L, 1024)` | Per-block sequence output |
| `layer_z` | `(B, L, L, 128)` | Per-block pair output |
| `esm_layer` | `(B, L, 2560)` | Per-ESM2-layer hidden state |
| `esm_attention_out` | `(B, L, 2560)` | Per-ESM2-layer attention output |
| `esm_attention_weights` | `(B, 40, L, L)` | Per-ESM2-layer attention matrix |
| `esm_ffn_out` | `(B, L, 2560)` | Per-ESM2-layer FFN output |
| `distogram` | `(B, L, L, 64)` | Distance distribution logits |
| `lddt_logits` | `(B, L, 50)` | pLDDT head logits |

---

## Memory Considerations

ESMFold is memory-hungry due to the `O(L^2)` pairwise representation.
Guidelines:

| Sequence length | `trunk_z` per block | Approx VRAM (all sites) |
|----------------|--------------------|-----------------------|
| 200 | 5 MB | ~8 GB |
| 500 | 32 MB | ~20 GB |
| 1000 | 128 MB | ~50 GB |

**Tips:**
- Use `--fp16` to halve VRAM for the model itself.
- Set `chunk_size: 64` (or lower) to reduce peak memory at the cost of speed.
- Use `--max_length 800` to skip very long sequences.
- Capture only `layer_s` (not `layer_z`) to avoid O(L^2) storage per block.
- Use `--trunk_layers 0,11,23,35,47` to sample a few blocks instead of all 48.
- Set `recycling_steps_to_save: last` to store only the final iteration.
- The extractor moves tensors to CPU by default (`to_cpu=True`), so GPU
  memory is freed after each hook fires.

---

## Comparison with Boltz Pipeline

| | ESMFold | Boltz |
|---|---|---|
| Config | `esmfold_extract_config.yaml` | `boltz_extract_config.yaml` |
| Orchestrator | `run_esmfold_extract.py` | `run_boltz_extract.py` |
| SLURM | `run_esmfold_extract.slrm` | `run_boltz_extract.slrm` |
| Extractor class | `ESMFoldExtractor` | `Boltz2Extractor` |
| Extraction module | `extract_hidden_reps_esmfold` | `extract_hidden_reps` |
| Input format | FASTA files | YAML files (with MSA) |
| Env | `esmfold` | `boltz` |
| Model input | Sequence only | Sequence + MSA + templates |
