# Protein Interpretability

Tools for extracting and analysing hidden representations from protein structure
prediction models (Boltz2, ESM3).

## Installation

Requires Python 3.12+.

```bash
# Core install (scoring, analysis)
uv sync

# With ESM3 support
uv sync --extra esm
```

On a cluster where the Boltz conda env already exists, you can skip the install
and set `PYTHONPATH` to point at `src/` instead (the orchestrator script does
this automatically). Boltz2 is intentionally not managed by this project's
local `uv` environment because its dependency pins conflict with the core
analysis stack; use the dedicated cluster environment for Boltz runs.

## Boltz2 Hidden Representation Extraction

### Prerequisites

1. **Boltz2 model cache.** The checkpoint and CCD database are downloaded
   automatically on first run. Set the cache path in the config or via
   `--cache`:

   ```
   ~/.boltz/                    # default local
   /n/holylfs06/.../boltz_db    # cluster shared cache
   ```

2. **Input YAMLs.** Each sequence needs a YAML file in Boltz2's input format:

   ```yaml
   version: 1

   sequences:
     - protein:
         id: "P12345"
         sequence: "MKTLLILAVF..."
         msa: /absolute/path/to/sequence.a3m
   ```

   MSA paths **must be absolute** — the orchestrator symlinks YAMLs into
   staging directories, so relative paths would break MSA resolution.

   These YAMLs are produced by
   [ProtForge](https://github.com/...)&rsquo;s `organize_msa_outputs.py`.

3. **Directory layout.** The extraction scripts accept either a single YAML or
   a directory. For batch runs, organise sequences as:

   ```
   sequences_dir/
     seq_00132/
       msa/
       seq_00132.yaml
     seq_00318/
       msa/
       seq_00318.yaml
     ...
   ```

   A flat directory of YAMLs also works (discovery is recursive via
   `rglob("*.yaml")`).

### Option A: Batch extraction with the orchestrator (recommended)

The orchestrator splits YAMLs across GPUs and launches one extraction
subprocess per GPU in parallel.

**1. Edit the config:**

```bash
cp scripts/boltz_extract_config.yaml my_config.yaml
```

```yaml
input:
  sequences_dir: /path/to/sequences
  subset_file: null              # optional: file with one yaml stem per line

output:
  out_dir: /path/to/hidden_reps

boltz:
  cache: /path/to/boltz_db
  recycling_steps: 3
  sampling_steps: 200
  diffusion_samples: 1
  step_scale: 1.5
  seed: null
  no_kernels: false

extraction:
  sites: [input_embedder, pairformer_s]
  layers: all
  layer_sites: [layer_s]
  recycling_steps_to_save: last  # "last" | "all" | "every:N" | "0,2"

runtime:
  num_gpus: 4
  accelerator: gpu
  num_workers: 2
  python: python
  env:
    CUEQ_DEFAULT_CONFIG: "1"
    CUEQ_DISABLE_AOT_TUNING: "1"
```

See [docs/boltz2_extraction_sites.md](docs/boltz2_extraction_sites.md) for
what each site captures and its tensor shape.

**2. Run:**

```bash
python scripts/run_boltz_extract.py --config my_config.yaml
```

The orchestrator will print a summary and launch. Per-GPU logs are written to
`<out_dir>/_log_chunk_N.out`.

### Option B: Single sequence / directory via CLI

```bash
python -m protein_interpretability.extract_hidden_reps input.yaml \
    --out_dir ./output \
    --cache ~/.boltz \
    --recycling_steps 3 \
    --sites input_embedder,pairformer_s \
    --layers all \
    --layer_sites layer_s \
    --recycling_steps_to_save last
```

Pass a directory instead of a single YAML to process all YAMLs in it.

Full CLI options:

```
python -m protein_interpretability.extract_hidden_reps --help
```

### Option C: SLURM job

```bash
sbatch scripts/run_extract_hidden_reps.slrm /path/to/input /path/to/output
```

Override defaults via environment variables:

```bash
sbatch --export=RECYCLING_STEPS=5,SITES=pairformer_s,LAYERS=all,RECYCLING_STEPS_SAVE=every:2 \
    scripts/run_extract_hidden_reps.slrm /path/to/input /path/to/output
```

Available overrides: `BOLTZ_CACHE`, `BOLTZ_ENV_PATH`, `RECYCLING_STEPS`,
`SAMPLING_STEPS`, `DIFFUSION_SAMPLES`, `LAYERS`, `SITES`, `LAYER_SITES`,
`RECYCLING_STEPS_SAVE`.

### Output structure

```
out_dir/
  seq_00132/
    model_outputs.pt          # Boltz2 predictions (coords, confidence)
    hidden_reps.pt            # extracted activations
    metadata.json             # config, shapes, file sizes
  seq_00318/
    ...
  _log_chunk_0.out            # per-GPU logs (orchestrator only)
  _staging/                   # temporary symlinks (can delete after)
```

**`model_outputs.pt`** contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `sample_atom_coords` | (1, N_atoms, 3) | predicted 3D coordinates |
| `s` | (1, N, 384) | final sequence representation |
| `z` | (1, N, N, 128) | final pairwise representation |
| `plddt` | (1, N) | per-residue confidence |
| `pae` | (1, N, N) | predicted alignment error |
| `ptm` | scalar | predicted TM-score |
| `token_pad_mask` | (1, N) | True = valid token |

**`hidden_reps.pt`** depends on `recycling_steps_to_save`:

With `last` (default) — a flat dict:
```python
{
    "input_embedder":       Tensor(1, N, 384),
    "pairformer_s":         Tensor(1, N, 384),
    "pairformer_0_layer_s": Tensor(1, N, 384),
    "pairformer_1_layer_s": Tensor(1, N, 384),
    ...
    "pairformer_47_layer_s": Tensor(1, N, 384),
}
```

With `all`, `every:N`, or comma-separated indices — nested by step:
```python
{
    "step_0": {"input_embedder": ..., "pairformer_0_layer_s": ..., ...},
    "step_1": {...},
    "step_2": {...},
}
```

### Extraction sites reference

| Site | Shape | Description |
|------|-------|-------------|
| `input_embedder` | (B, N, 384) | initial per-token embeddings (pre-pairformer) |
| `pairformer_s` | (B, N, 384) | final per-token representation |
| `pairformer_z` | (B, N, N, 128) | final pairwise representation |
| `distogram` | (B, N, N, 64) | distance bin logits |
| `layer_s` | (B, N, 384) | per-layer sequence output |
| `layer_z` | (B, N, N, 128) | per-layer pairwise output |
| `attention` | (B, N, 384) | attention module output |
| `attention_weights` | (B, 16, N, N) | post-softmax attention (16 heads) |
| `tri_mul_out` | (B, N, N, 128) | triangle multiplication outgoing |
| `tri_mul_in` | (B, N, N, 128) | triangle multiplication incoming |
| `tri_att_start` | (B, N, N, 128) | triangle attention starting node |
| `tri_att_end` | (B, N, N, 128) | triangle attention ending node |
| `transition_s` | (B, N, 384) | sequence transition MLP |
| `transition_z` | (B, N, N, 128) | pairwise transition MLP |

Full architecture descriptions in
[docs/boltz2_extraction_sites.md](docs/boltz2_extraction_sites.md).

### `recycling_steps_to_save` options

| Value | Saves | Example with 6 recycling steps |
|-------|-------|-------------------------------|
| `last` | final step only | step 5 |
| `all` | every step | 0, 1, 2, 3, 4, 5 |
| `every:2` | every 2nd + last | 0, 2, 4, 5 |
| `every:3` | every 3rd + last | 0, 3, 5 |
| `0,3,-1` | specific indices | 0, 3, 5 |

### Storage estimates

Per sequence of length N, per recycling step saved:

| Config | Size formula | N=150 |
|--------|-------------|-------|
| `layer_s` only, 48 layers | 48 x N x 384 x 4B | ~10 MB |
| + `input_embedder` + `pairformer_s` | +2 x N x 384 x 4B | ~10.5 MB |
| + `layer_z`, 48 layers | +48 x N x N x 128 x 4B | ~4.7 GB |
| + `attention_weights`, 48 layers | +48 x 16 x N x N x 4B | ~3.2 GB |

For the distance-to-WT analysis, `layer_s` + `input_embedder` + `pairformer_s`
is sufficient and storage-friendly (~10 MB per sequence per step).

## Boltz2 Attention Extraction

Standalone script for capturing only attention weight matrices:

```bash
python -m protein_interpretability.extract_attention input.yaml \
    --out_dir ./attention_output \
    --layers all \
    --average_heads \
    --save_format pt
```

Outputs are saved to `<out_dir>/boltz_results_<stem>/attention/`.

## Scoring

Structure comparison utilities for evaluating mutations:

```python
from protein_interpretability.scoring import path_tm_score, path_rmsd

tm = path_tm_score("wildtype.cif", "mutant.cif")
rmsd = path_rmsd("wildtype.cif", "mutant.cif")
```

Supports both PDB and mmCIF formats. For multi-chain structures, pass
`chain_id` explicitly.

### Batch scoring a directory of predictions

Use `score_sequences` to score every `.cif` or `.pdb` under a results directory
against a single reference and write the results to CSV:

```bash
python -m protein_interpretability.score_sequences \
    --ref /path/to/reference.cif \
    --predicted-dir /path/to/results \
    --output-dir /path/to/output \
    --output-name structure_scores.csv \
    --normalize-by reference
    # --chain-id A        # optional, for multi-chain structures
```

The script searches `--predicted-dir` recursively for `.cif` and `.pdb`, so
Boltz2 output layouts like `results/seq_00132/predictions/seq_00132_model_0.cif`
are picked up automatically. Each filename must contain a `seq_<N>` token
so the sequence index can be parsed.

`--normalize-by` controls which structure length normalizes the TM-score
(`reference` or `predicted`); RMSD is length-independent.

The output CSV has columns:

| Column | Description |
|--------|-------------|
| `sequence_idx` | integer parsed from `seq_<N>` in the filename |
| `predicted_path` | absolute path to the scored `.cif` or `.pdb` |
| `tm_score` | TM-score vs. reference |
| `rmsd` | C-alpha RMSD vs. reference |

## Tests

```bash
uv run pytest tests/
```
