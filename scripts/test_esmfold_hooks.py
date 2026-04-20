"""Test ESMFold hook extractor end-to-end.

Loads the HuggingFace EsmForProteinFolding model, runs a short sequence,
and verifies that hooks capture activations with the expected shapes at
every extraction site.

Usage::

    python scripts/test_esmfold_hooks.py
    python scripts/test_esmfold_hooks.py --model_path /path/to/local/esmfold_v1
    python scripts/test_esmfold_hooks.py --device cpu
    python scripts/test_esmfold_hooks.py --num_recycles 1  # faster test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from protein_interpretability.extractor_esmfold import ESMFoldExtractor


def parse_args():
    p = argparse.ArgumentParser(description="Test ESMFold hook extractor")
    p.add_argument(
        "--model_path",
        type=str,
        default="facebook/esmfold_v1",
        help="HuggingFace model id or local path.",
    )
    p.add_argument("--device", type=str, default=None,
                   help="Device (auto-detect if omitted).")
    p.add_argument("--num_recycles", type=int, default=2,
                   help="Number of recycling iterations.")
    p.add_argument("--chunk_size", type=int, default=64,
                   help="Trunk chunk size (-1 to disable).")
    p.add_argument("--sequence", type=str, default="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL",
                   help="Test protein sequence.")
    p.add_argument("--trunk_layers", type=str, default="0,1,23,47",
                   help="Trunk block indices to hook (comma-sep or 'all').")
    p.add_argument("--esm_layers", type=str, default="0,1,35",
                   help="ESM2 layer indices to hook (comma-sep or 'all').")
    return p.parse_args()


def _parse_ints(s: str) -> list[int] | None:
    if s == "all":
        return None
    return [int(x.strip()) for x in s.split(",")]


def main():
    args = parse_args()
    seq = args.sequence
    L = len(seq)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    trunk_layers = _parse_ints(args.trunk_layers)
    esm_layers = _parse_ints(args.esm_layers)

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    from transformers import AutoTokenizer, EsmForProteinFolding

    print(f"Loading model from {args.model_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = EsmForProteinFolding.from_pretrained(
        args.model_path, low_cpu_mem_usage=True
    )
    model = model.to(device).eval()
    if args.chunk_size > 0:
        model.trunk.set_chunk_size(args.chunk_size)
    print(f"  loaded in {time.time() - t0:.1f}s on {device}")

    n_trunk = len(model.trunk.blocks)
    n_esm = len(model.esm.encoder.layer)
    print(f"  trunk blocks: {n_trunk}, ESM2 layers: {n_esm}")

    # ----------------------------------------------------------------
    # Tokenize
    # ----------------------------------------------------------------
    tokenized = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized["input_ids"].to(device)
    print(f"\nSequence: {seq[:40]}{'...' if len(seq) > 40 else ''}")
    print(f"Length: {L}, token shape: {tuple(input_ids.shape)}")

    # ----------------------------------------------------------------
    # Test 1: Minimal hooks (layer_s, layer_z only)
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 1: Minimal hooks (layer_s + layer_z on selected trunk blocks)")
    print("=" * 70)

    extractor = ESMFoldExtractor(
        model,
        sites=["trunk_s", "trunk_z"],
        trunk_layers=trunk_layers,
        layer_sites=["layer_s", "layer_z"],
    )

    with extractor:
        t0 = time.time()
        with torch.no_grad():
            output = model(input_ids, num_recycles=args.num_recycles)
        elapsed = time.time() - t0
        print(f"  forward pass: {elapsed:.2f}s")

        n_steps = len(extractor.activations)
        # num_recycles + 1 because the trunk internally adds 1
        expected_steps = args.num_recycles + 1
        print(f"  recycling steps captured: {n_steps} (expected ~{expected_steps})")

        # Check final step
        final = extractor.get_step(-1)
        print(f"  keys in final step: {sorted(final.keys())}")

        passed = True

        if "trunk_s" in final:
            shape = tuple(final["trunk_s"].shape)
            print(f"  trunk_s shape: {shape}")
            assert shape[1] == L, f"Expected L={L} in dim 1, got {shape[1]}"
        else:
            print("  WARNING: trunk_s not captured")
            passed = False

        if "trunk_z" in final:
            shape = tuple(final["trunk_z"].shape)
            print(f"  trunk_z shape: {shape}")
            assert shape[1] == L and shape[2] == L, \
                f"Expected (*, {L}, {L}, *), got {shape}"
        else:
            print("  WARNING: trunk_z not captured")
            passed = False

        tl = trunk_layers if trunk_layers else list(range(n_trunk))
        for i in tl:
            key_s = f"trunk_{i}_layer_s"
            key_z = f"trunk_{i}_layer_z"
            if key_s not in final:
                print(f"  WARNING: {key_s} missing from final step")
                passed = False
            if key_z not in final:
                print(f"  WARNING: {key_z} missing from final step")
                passed = False

        print(f"  TEST 1: {'PASSED' if passed else 'FAILED'}")

    # ----------------------------------------------------------------
    # Test 2: All top-level sites
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 2: All top-level sites")
    print("=" * 70)

    extractor2 = ESMFoldExtractor(
        model,
        sites=list(ESMFoldExtractor.SITES),
        trunk_layers=[0],
        layer_sites=["layer_s"],
    )

    with extractor2:
        with torch.no_grad():
            output = model(input_ids, num_recycles=args.num_recycles)

        final = extractor2.get_step(-1)
        first = extractor2.get_step(0)
        print(f"  keys in first step: {sorted(first.keys())}")
        print(f"  keys in final step: {sorted(final.keys())}")

        passed = True
        for site in ESMFoldExtractor.SITES:
            if site == "positions":
                # positions come from model output, not hooks
                continue
            found_in_any = any(
                site in step for step in extractor2.activations
            )
            if found_in_any:
                for step in extractor2.activations:
                    if site in step:
                        val = step[site]
                        if isinstance(val, torch.Tensor):
                            print(f"  {site}: shape={tuple(val.shape)}, dtype={val.dtype}")
                        else:
                            print(f"  {site}: type={type(val).__name__}")
                        break
            else:
                print(f"  WARNING: {site} not captured in any step")
                passed = False

        print(f"  TEST 2: {'PASSED' if passed else 'FAILED'}")

    # ----------------------------------------------------------------
    # Test 3: ESM2 backbone layer hooks
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 3: ESM2 backbone per-layer hooks")
    print("=" * 70)

    extractor3 = ESMFoldExtractor(
        model,
        sites=[],
        trunk_layers=[],
        esm_layers=esm_layers,
        layer_sites=["esm_layer"],
    )

    with extractor3:
        with torch.no_grad():
            output = model(input_ids, num_recycles=args.num_recycles)

        # ESM2 runs once (before recycling), so esm_layer keys should be in
        # the first step dict.
        first = extractor3.get_step(0)
        print(f"  keys in first step: {sorted(first.keys())}")

        passed = True
        el = esm_layers if esm_layers else list(range(n_esm))
        for i in el:
            key = f"esm_{i}_layer"
            found = any(key in step for step in extractor3.activations)
            if found:
                for step in extractor3.activations:
                    if key in step:
                        print(f"  {key}: shape={tuple(step[key].shape)}")
                        break
            else:
                print(f"  WARNING: {key} not captured")
                passed = False

        print(f"  TEST 3: {'PASSED' if passed else 'FAILED'}")

    # ----------------------------------------------------------------
    # Test 4: Per-trunk-block submodule hooks
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 4: Trunk submodule hooks (tri_mul, tri_att, transitions)")
    print("=" * 70)

    sub_sites = ["seq_attention", "tri_mul_out", "tri_mul_in",
                 "tri_att_start", "tri_att_end", "transition_s", "transition_z"]

    extractor4 = ESMFoldExtractor(
        model,
        sites=[],
        trunk_layers=[0],  # just block 0 to keep output small
        layer_sites=sub_sites,
    )

    with extractor4:
        with torch.no_grad():
            output = model(input_ids, num_recycles=args.num_recycles)

        final = extractor4.get_step(-1)
        print(f"  keys in final step: {sorted(final.keys())}")

        passed = True
        for site in sub_sites:
            key = f"trunk_0_{site}"
            if key in final:
                val = final[key]
                if isinstance(val, torch.Tensor):
                    print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
                else:
                    print(f"  {key}: type={type(val).__name__}")
            else:
                print(f"  WARNING: {key} not captured")
                passed = False

        print(f"  TEST 4: {'PASSED' if passed else 'FAILED'}")

    # ----------------------------------------------------------------
    # Test 5: Clear / re-run / remove lifecycle
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 5: Lifecycle (install, clear, re-run, remove)")
    print("=" * 70)

    extractor5 = ESMFoldExtractor(
        model,
        sites=["trunk_s"],
        trunk_layers=[0],
        layer_sites=["layer_s"],
    )
    extractor5.install()

    with torch.no_grad():
        model(input_ids, num_recycles=1)
    n1 = len(extractor5.activations)
    keys1 = extractor5.step_keys()

    extractor5.clear()
    assert len(extractor5.activations) == 0, "clear() didn't reset activations"

    with torch.no_grad():
        model(input_ids, num_recycles=1)
    n2 = len(extractor5.activations)
    keys2 = extractor5.step_keys()

    extractor5.remove()

    # After remove, a forward pass should not add activations
    extractor5.clear()
    with torch.no_grad():
        model(input_ids, num_recycles=1)
    n3 = len(extractor5.activations)

    passed = n1 == n2 and n1 > 0 and n3 == 0 and keys1 == keys2
    print(f"  steps after 1st run: {n1}")
    print(f"  steps after clear+re-run: {n2}")
    print(f"  steps after remove+run: {n3} (expected 0)")
    print(f"  TEST 5: {'PASSED' if passed else 'FAILED'}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Sequence length: {L}")
    print(f"Trunk blocks: {n_trunk}, ESM2 layers: {n_esm}")
    print(f"Recycles: {args.num_recycles}")


if __name__ == "__main__":
    main()
