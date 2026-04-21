"""Extract hidden representations from ESMFold v1 during inference.

Runs a full ESMFold forward pass, captures activations via ``ESMFoldExtractor``,
and saves model outputs (coords, plddt, distogram, etc.) plus per-layer
hidden reps to disk.

Inputs:
    - FASTA file (one or many sequences), or
    - directory of FASTA files, or
    - plain text file with one sequence per line.

Usage::

    python -m protein_interpretability.extract_hidden_reps_esmfold seqs.fasta \
        --out_dir ./esmfold_reps --num_recycles 3

Output layout (mirrors the Boltz extractor)::

    <out_dir>/<record_id>/
        model_outputs.pt      — positions, plddt, distogram logits, etc.
        hidden_reps.pt        — all extractor activations (last recycling step
                                by default)
        metadata.json         — shapes, sizes, config
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path

import torch

from protein_interpretability.extractor_esmfold import ESMFoldExtractor


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract hidden representations from ESMFold v1"
    )
    p.add_argument(
        "data",
        type=str,
        help="FASTA file, directory of FASTAs, or text file (one seq per line).",
    )
    p.add_argument("--out_dir", type=str, default="./esmfold_reps")
    p.add_argument(
        "--model_name",
        type=str,
        default="facebook/esmfold_v1",
        help="HuggingFace model id (default: facebook/esmfold_v1).",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HF cache root (e.g. /path/to/esm_models_cache). "
             "Enables offline load via cache_dir/hub. Same pattern as ProtForge.",
    )
    p.add_argument("--num_recycles", type=int, default=3)
    p.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="Trunk chunk size; lower = less memory, slower. Set -1 to disable.",
    )
    p.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--fp16", action="store_true", help="Cast trunk to fp16.")
    p.add_argument(
        "--trunk_layers",
        type=str,
        default="all",
        help="Comma-separated trunk block indices, or 'all'.",
    )
    p.add_argument(
        "--esm_layers",
        type=str,
        default="all",
        help="Comma-separated ESM2 encoder layer indices, or 'all'.",
    )
    p.add_argument(
        "--sites",
        type=str,
        default="all",
        help=f"Top-level sites, comma-separated or 'all'. Options: "
             f"{', '.join(ESMFoldExtractor.SITES)}",
    )
    p.add_argument(
        "--layer_sites",
        type=str,
        default="layer_s,layer_z",
        help=f"Per-layer sites, comma-separated or 'all'. Options: "
             f"{', '.join(ESMFoldExtractor.LAYER_SITES)}",
    )
    p.add_argument(
        "--recycling_steps_to_save",
        type=str,
        default="last",
        help="'last', 'all', 'every:N', or comma-separated indices.",
    )
    p.add_argument("--max_length", type=int, default=None,
                   help="Skip sequences longer than this (memory safety).")
    return p.parse_args()


def _parse_list_arg(value: str, valid: tuple[str, ...]) -> list[str] | None:
    if value == "all":
        return None
    if value in ("none", ""):
        return []
    items = [x.strip() for x in value.split(",") if x.strip()]
    for item in items:
        if item not in valid:
            raise ValueError(f"Unknown site '{item}'. Valid: {valid}")
    return items


def _parse_int_list(value: str) -> list[int] | None:
    if value == "all":
        return None
    if value in ("none", ""):
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _read_sequences(path: Path) -> list[tuple[str, str]]:
    """Return list of (record_id, sequence). Supports FASTA, dir of FASTA, or
    a plain text file with one sequence per line."""
    records: list[tuple[str, str]] = []

    def _parse_fasta(fp: Path) -> list[tuple[str, str]]:
        out, cur_id, cur_seq = [], None, []
        for line in fp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    out.append((cur_id, "".join(cur_seq)))
                cur_id = line[1:].split()[0] or fp.stem
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            out.append((cur_id, "".join(cur_seq)))
        return out

    if path.is_dir():
        for fp in sorted(path.iterdir()):
            if fp.suffix.lower() in {".fasta", ".fa", ".faa"}:
                records.extend(_parse_fasta(fp))
        return records

    if path.suffix.lower() in {".fasta", ".fa", ".faa"}:
        return _parse_fasta(path)

    # Plain text: one sequence per line, id = filename_<index>
    stem = path.stem
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if line:
            records.append((f"{stem}_{i}", line))
    return records


def _tensor_size_mb(t: torch.Tensor) -> float:
    return t.nelement() * t.element_size() / (1024 * 1024)


def _flatten_shapes(obj, prefix: str = "") -> dict:
    """Recursively collect shape/dtype/size for tensors inside nested dicts."""
    out = {}
    if isinstance(obj, torch.Tensor):
        out[prefix or "<tensor>"] = {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "size_mb": round(_tensor_size_mb(obj), 3),
        }
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten_shapes(v, f"{prefix}/{k}" if prefix else str(k)))
    return out


def main():
    args = parse_args()

    warnings.filterwarnings("ignore")
    torch.set_grad_enabled(False)

    # Import here so that module import doesn't require transformers installed
    # for users who only want the extractor class.
    from transformers import AutoTokenizer, EsmForProteinFolding

    sites = _parse_list_arg(args.sites, ESMFoldExtractor.SITES)
    if sites is None:
        sites = list(ESMFoldExtractor.SITES)
    layer_sites = _parse_list_arg(args.layer_sites, ESMFoldExtractor.LAYER_SITES)
    if layer_sites is None:
        layer_sites = list(ESMFoldExtractor.LAYER_SITES)
    trunk_layers = _parse_int_list(args.trunk_layers)
    esm_layers = _parse_int_list(args.esm_layers)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _read_sequences(Path(args.data).expanduser())
    if not records:
        print("No sequences found.")
        return
    if args.max_length is not None:
        records = [(rid, s) for rid, s in records if len(s) <= args.max_length]
        if not records:
            print(f"All sequences exceed max_length={args.max_length}.")
            return

    device = torch.device(
        "cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )

    # Load model — use cache_dir pattern (same as ProtForge) when provided,
    # otherwise treat model_name as a direct HF id or local path.
    hub_cache = str(Path(args.cache_dir).expanduser() / "hub") if args.cache_dir else None
    offline = hub_cache is not None
    print(f"Loading {args.model_name} (cache={hub_cache}, offline={offline}) ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=hub_cache, local_files_only=offline,
    )
    model = EsmForProteinFolding.from_pretrained(
        args.model_name, cache_dir=hub_cache, local_files_only=offline,
        low_cpu_mem_usage=True,
    )
    model = model.to(device).eval()
    if args.fp16 and device.type == "cuda":
        model.esm = model.esm.half()
        model.trunk = model.trunk.to(dtype=torch.float16)
    if args.chunk_size is not None and args.chunk_size > 0:
        model.trunk.set_chunk_size(args.chunk_size)

    extractor = ESMFoldExtractor(
        model,
        sites=sites,
        trunk_layers=trunk_layers,
        esm_layers=esm_layers,
        layer_sites=layer_sites,
    )
    extractor.install()

    n_trunk = len(model.trunk.blocks)
    n_esm = len(model.esm.encoder.layer)
    print(
        f"Extractor installed: {len(sites)} top-level sites, "
        f"{len(layer_sites)} layer sites | "
        f"trunk={len(trunk_layers) if trunk_layers else n_trunk}/{n_trunk}, "
        f"esm={len(esm_layers) if esm_layers else n_esm}/{n_esm}"
    )

    for idx, (record_id, seq) in enumerate(records):
        print(f"[{idx + 1}/{len(records)}] {record_id} (L={len(seq)})")

        record_dir = out_dir / record_id
        record_dir.mkdir(parents=True, exist_ok=True)

        tokenized = tokenizer(
            [seq], return_tensors="pt", add_special_tokens=False
        )
        input_ids = tokenized["input_ids"].to(device)

        extractor.clear()
        output = model(input_ids, num_recycles=args.num_recycles)

        # ---- Save model outputs ----
        output_keys = [
            "positions", "frames", "sidechain_frames", "unnormalized_angles",
            "angles", "s_s", "s_z", "distogram_logits", "lddt_head",
            "plddt", "ptm_logits", "ptm",
            "aligned_confidence_probs", "predicted_aligned_error",
            "max_predicted_aligned_error",
        ]
        model_outputs = {}
        for k in output_keys:
            v = getattr(output, k, None)
            if isinstance(v, torch.Tensor):
                model_outputs[k] = v.detach().cpu()
        model_outputs["input_ids"] = input_ids.cpu()
        model_outputs["sequence"] = seq

        model_out_path = record_dir / "model_outputs.pt"
        torch.save(model_outputs, model_out_path)
        model_out_size = model_out_path.stat().st_size / (1024 * 1024)

        # ---- Select recycling steps to save ----
        n_steps = len(extractor.activations)
        save_arg = args.recycling_steps_to_save
        if save_arg == "all":
            step_indices = list(range(n_steps))
        elif save_arg == "last":
            step_indices = [-1]
        elif save_arg.startswith("every:"):
            stride = int(save_arg.split(":")[1])
            step_indices = list(range(0, n_steps, stride))
            if step_indices and step_indices[-1] != n_steps - 1:
                step_indices.append(n_steps - 1)
        else:
            step_indices = [int(x.strip()) for x in save_arg.split(",")]

        if len(step_indices) == 1:
            hidden_reps = extractor.get_step(step_indices[0])
        else:
            hidden_reps = {
                f"step_{i % n_steps}": extractor.get_step(i) for i in step_indices
            }

        hidden_reps_path = record_dir / "hidden_reps.pt"
        torch.save(hidden_reps, hidden_reps_path)
        hidden_reps_size = hidden_reps_path.stat().st_size / (1024 * 1024)

        # ---- Metadata ----
        metadata = {
            "record_id": record_id,
            "sequence_length": len(seq),
            "config": {
                "model_name": args.model_name,
                "num_recycles": args.num_recycles,
                "chunk_size": args.chunk_size,
                "fp16": args.fp16,
                "trunk_layers": args.trunk_layers,
                "esm_layers": args.esm_layers,
                "sites": args.sites,
                "layer_sites": args.layer_sites,
                "recycling_steps_saved": args.recycling_steps_to_save,
                "num_recycling_steps_captured": n_steps,
            },
            "file_sizes_mb": {
                "model_outputs.pt": round(model_out_size, 2),
                "hidden_reps.pt": round(hidden_reps_size, 2),
                "total": round(model_out_size + hidden_reps_size, 2),
            },
            "model_output_shapes": _flatten_shapes(model_outputs),
            "hidden_rep_shapes": _flatten_shapes(hidden_reps),
        }
        with open(record_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"  -> {record_dir} | outputs={model_out_size:.1f}MB, "
            f"hidden={hidden_reps_size:.1f}MB, steps={n_steps}"
        )

        del output, model_outputs, hidden_reps
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    extractor.remove()
    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
