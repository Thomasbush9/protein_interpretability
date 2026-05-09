#!/usr/bin/env python3
"""Extract per-protein deep MSAs from the Porter et al. Zenodo deposit and
populate <out_dir>/msa/<seq_stem>.a3m, then regenerate YAMLs to reference them.

Zenodo layout (after download from DOI 10.5281/zenodo.13221957):

    <zenodo_root>/AFcluster_MSAs/
        info_all_runs.txt              # bulk index: '<sub_X> <Y>_msas <pdb_chain>'
        sub_{0,4,5,6,8}.tar.gz{aa,ab,ac,ad}   # split archives — concat first
        sub_{1,2,3,7,9}.tgz                    # single-piece archives
        <pdb_chain>.tar.gz | .tgz       # 13 standalone case studies

Inside each archive:
    Standalone:    <pdb_chain>/0.a3m       (deep MSA)  + <pdb_chain>/0_msas/...
    Sub_X bulk:    <sub_X>/<Y>.a3m         (deep MSA)  + <sub_X>/<Y>_msas/...

We use the **deep MSA** (top-level `<pdb_chain>/0.a3m` or `<sub_X>/<Y>.a3m`),
which is Porter's Step-1 ColabFold MSA — i.e. Phase-1-grade default input,
NOT the AF-Cluster shallow ablation.

Usage::

    python scripts/foldswitch_extract_zenodo_msa.py \\
        --foldswitch_dir /Users/.../data/foldswitch \\
        --zenodo_root    /Users/.../data/foldswitch/porterll-AF2_benchmark-40a57d7

Output (under <foldswitch_dir>):
    msa/seq_NNNN_<f1>_<f2>.a3m
    yamls/seq_NNNN_<f1>_<f2>.yaml      (regenerated with `msa: msa/<stem>.a3m`)
    manifest_with_msa.csv               (manifest + msa columns)
    _extracted/<archive>/               (cache, kept for re-runs)
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

SUBS_SPLIT = {0, 4, 5, 6, 8}  # archives that arrive as <sub_X>.tar.gz{aa,ab,..}
SUBS_SINGLE = {1, 2, 3, 7, 9}  # archives that arrive as <sub_X>.tgz


def parse_info_all_runs(path: Path) -> dict[str, tuple[str, str]]:
    """Returns {pdb_chain_norm (e.g. '1H38D'): (sub_dir, member_index)}."""
    lookup: dict[str, tuple[str, str]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            sub_dir, y_msas, pdb_chain = parts[0], parts[1], parts[2]
            # 'y_msas' looks like '0_msas' / '10_msas' — the deep MSA is '<Y>.a3m'
            y = y_msas.split("_")[0]
            norm = pdb_chain.replace("_", "").upper()
            lookup[norm] = (sub_dir, y)
    return lookup


def find_standalone_archive(zenodo_root: Path, pdb_chain_norm: str) -> Path | None:
    """Look for `<pdb>_<chain>.tar.gz` / `.tgz` in AFcluster_MSAs/."""
    # restore '_' between pdb id and chain
    base = f"{pdb_chain_norm[:4].lower()}_{pdb_chain_norm[4:].upper()}"
    afdir = zenodo_root / "AFcluster_MSAs"
    for ext in (".tar.gz", ".tgz"):
        cand = afdir / f"{base}{ext}"
        if cand.exists():
            return cand
    return None


def ensure_sub_archive_concatenated(zenodo_root: Path, sub_idx: int) -> Path | None:
    """For sub_X that arrives split, concat .aa/.ab/... into a single .tar.gz.

    Returns the path to the consolidated archive, or None if pieces are missing.
    """
    afdir = zenodo_root / "AFcluster_MSAs"
    if sub_idx in SUBS_SINGLE:
        cand = afdir / f"sub_{sub_idx}.tgz"
        return cand if cand.exists() else None

    consolidated = afdir / f"sub_{sub_idx}.tar.gz"
    if consolidated.exists() and consolidated.stat().st_size > 0:
        return consolidated

    pieces = sorted(afdir.glob(f"sub_{sub_idx}.tar.gz??"))
    if not pieces:
        return None
    print(f"  [concat] sub_{sub_idx}: {len(pieces)} pieces -> {consolidated.name}")
    with consolidated.open("wb") as out:
        for p in pieces:
            out.write(p.read_bytes())
    return consolidated


def extract_archive(archive: Path, cache_dir: Path) -> Path:
    """Extract to cache_dir, return the directory containing the extracted tree."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / f".extracted__{archive.name}"
    if marker.exists():
        return cache_dir
    print(f"  [extract] {archive.name} -> {cache_dir}")
    subprocess.run(
        ["tar", "-xzf", str(archive), "-C", str(cache_dir)],
        check=True,
    )
    marker.touch()
    return cache_dir


def find_a3m_after_extract(cache_dir: Path,
                           archive: Path,
                           sub_dir: str | None,
                           y: str | None,
                           pdb_chain_norm: str) -> Path | None:
    """Return the path to the deep MSA after extraction.

    For sub_X archives:    cache_dir / sub_X / <Y>.a3m
    For standalone:        cache_dir / <pdb>_<chain> / 0.a3m
    """
    if sub_dir is not None and y is not None:
        cand = cache_dir / sub_dir / f"{y}.a3m"
        return cand if cand.exists() else None
    # Standalone: archive name like '2oug_C.tgz' -> dir '2oug_C'.
    # The deep MSA filename is `<Y>.a3m` where Y varies per protein.
    # Pick the unique top-level *.a3m (not the ones inside '<Y>_msas/').
    base = archive.name
    for ext in (".tar.gz", ".tgz"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    proto_dir = cache_dir / base
    if not proto_dir.is_dir():
        return None
    top_level = [p for p in proto_dir.glob("*.a3m") if p.is_file()]
    if len(top_level) == 1:
        return top_level[0]
    if len(top_level) > 1:
        print(f"  [warn] multiple top-level a3m in {proto_dir}: {[p.name for p in top_level]}")
        return top_level[0]
    return None


def read_msa_query(a3m_path: Path) -> tuple[str, str] | None:
    """Return (header, query_sequence) for the first record in the a3m."""
    header = None
    body: list[str] = []
    with a3m_path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    return header, "".join(body)
                header = line[1:]
            elif header is not None:
                body.append(line)
    if header is not None:
        return header, "".join(body)
    return None


def write_yaml(path: Path, sequence: str, msa_rel: str | None) -> None:
    lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        "      id: A",
        f"      sequence: {sequence}",
    ]
    if msa_rel is not None:
        lines.append(f"      msa: {msa_rel}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--foldswitch_dir", type=Path, required=True,
                    help="Dir containing manifest.csv, yamls/. msa/ will be (re)populated here.")
    ap.add_argument("--zenodo_root", type=Path, required=True,
                    help="Path to extracted Zenodo dir (porterll-AF2_benchmark-XXXXXXX)")
    ap.add_argument("--no_yaml_rewrite", action="store_true",
                    help="Skip regenerating yamls/ (just populate msa/)")
    ap.add_argument("--strip_query_only_msa", action="store_true",
                    help="Skip pairs whose MSA contains only the query sequence")
    args = ap.parse_args()

    fs_dir: Path = args.foldswitch_dir
    manifest_path = fs_dir / "manifest.csv"
    yamls_dir = fs_dir / "yamls"
    msa_dir = fs_dir / "msa"
    cache_dir = fs_dir / "_extracted"
    msa_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    info_path = args.zenodo_root / "AFcluster_MSAs" / "info_all_runs.txt"
    if not info_path.exists():
        sys.exit(f"missing index: {info_path}")
    bulk_lookup = parse_info_all_runs(info_path)
    print(f"[info] info_all_runs.txt: {len(bulk_lookup)} bulk pdb_chain entries")

    with manifest_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("seq_len") and int(r["seq_len"]) > 0]
    print(f"[info] manifest: {len(rows)} usable pairs")

    out_rows: list[dict] = []
    n_used_f1 = n_used_f2 = n_missing = n_seq_mismatch = 0

    for r in rows:
        idx = int(r["idx"])
        f1 = r["fold1"].upper()
        f2 = r["fold2"].upper()
        stem_yaml = f"seq_{idx:04d}_{r['fold1']}_{r['fold2']}"

        # Try Fold1 first, then Fold2.
        chosen, source_kind, archive, sub_y = None, None, None, None
        for fold_id in (f1, f2):
            if fold_id in bulk_lookup:
                sub_dir, y = bulk_lookup[fold_id]
                sub_idx = int(sub_dir.split("_")[1])
                arc = ensure_sub_archive_concatenated(args.zenodo_root, sub_idx)
                if arc is None:
                    continue  # archive not yet downloaded / pieces missing
                chosen, source_kind, archive, sub_y = fold_id, "bulk", arc, (sub_dir, y)
                break
            arc = find_standalone_archive(args.zenodo_root, fold_id)
            if arc is not None:
                chosen, source_kind, archive, sub_y = fold_id, "standalone", arc, None
                break

        out = dict(r)
        if chosen is None:
            print(f"  [{idx:>3}] {r['fold1']:>6}/{r['fold2']:<6}  MISS (no archive)")
            out.update(msa_status="archive_missing", msa_source="", msa_path="")
            out_rows.append(out)
            n_missing += 1
            continue

        # Extract archive to cache (idempotent)
        try:
            extract_archive(archive, cache_dir)
        except subprocess.CalledProcessError as e:
            print(f"  [{idx:>3}] {chosen}: tar failed -> {e}")
            out.update(msa_status="extract_error", msa_source="", msa_path="")
            out_rows.append(out)
            n_missing += 1
            continue

        if source_kind == "bulk":
            sub_dir, y = sub_y
            a3m_src = find_a3m_after_extract(cache_dir, archive, sub_dir, y, chosen)
        else:
            a3m_src = find_a3m_after_extract(cache_dir, archive, None, None, chosen)

        if a3m_src is None or not a3m_src.exists():
            print(f"  [{idx:>3}] {chosen}: extracted but a3m not found")
            out.update(msa_status="a3m_not_found", msa_source=str(archive),
                       msa_path="")
            out_rows.append(out)
            n_missing += 1
            continue

        # Read query and validate against manifest sequence
        rec = read_msa_query(a3m_src)
        if rec is None:
            print(f"  [{idx:>3}] {chosen}: a3m unreadable")
            out.update(msa_status="a3m_unreadable", msa_source=str(archive),
                       msa_path="")
            out_rows.append(out)
            n_missing += 1
            continue
        msa_header, msa_query = rec
        msa_query_clean = msa_query.upper().replace("-", "")
        manifest_seq = r["sequence"].upper()

        seq_to_use = msa_query_clean
        if msa_query_clean == manifest_seq:
            seq_match = "exact"
        elif manifest_seq.startswith(msa_query_clean) or msa_query_clean.startswith(manifest_seq):
            seq_match = "prefix"
        elif manifest_seq in msa_query_clean or msa_query_clean in manifest_seq:
            seq_match = "substring"
        else:
            seq_match = "diverge"
            n_seq_mismatch += 1

        # Optionally skip query-only MSAs (single seq doesn't help)
        n_seqs = sum(1 for ln in a3m_src.read_text().splitlines() if ln.startswith(">"))
        if args.strip_query_only_msa and n_seqs <= 1:
            print(f"  [{idx:>3}] {chosen}: only {n_seqs} seqs; skipping")
            out.update(msa_status="query_only", msa_source=str(archive), msa_path="")
            out_rows.append(out)
            n_missing += 1
            continue

        # Copy MSA to msa/<stem>.a3m
        stem_msa = f"seq_{idx:04d}_{r['fold1']}_{r['fold2']}"
        msa_out = msa_dir / f"{stem_msa}.a3m"
        shutil.copy(a3m_src, msa_out)

        # Regenerate YAML
        if not args.no_yaml_rewrite:
            yaml_out = yamls_dir / f"{stem_yaml}.yaml"
            write_yaml(yaml_out, seq_to_use, f"msa/{stem_msa}.a3m")

        if chosen == f1:
            n_used_f1 += 1
        else:
            n_used_f2 += 1

        out.update(
            msa_status=f"ok_{source_kind}_{seq_match}",
            msa_source=str(archive.relative_to(args.zenodo_root)) if archive else "",
            msa_path=f"msa/{stem_msa}.a3m",
            msa_chain_used=chosen,
            msa_n_seqs=n_seqs,
            msa_query_match=seq_match,
            sequence_used=seq_to_use,
        )
        out_rows.append(out)
        print(f"  [{idx:>3}] {r['fold1']:>6}/{r['fold2']:<6}  {chosen:>6} "
              f"src={source_kind:<10} n_seqs={n_seqs:>5}  match={seq_match}")

    # Write augmented manifest
    out_csv = fs_dir / "manifest_with_msa.csv"
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    # Summary
    print("\n[summary]")
    print(f"  total pairs:               {len(rows)}")
    print(f"  MSA from Fold1:            {n_used_f1}")
    print(f"  MSA from Fold2 (fallback): {n_used_f2}")
    print(f"  no MSA available:          {n_missing}")
    print(f"  seq divergence (Fold1):    {n_seq_mismatch}")
    print(f"  manifest_with_msa.csv ->   {out_csv}")
    print(f"  msa dir:                   {msa_dir}")
    print(f"  yaml dir:                  {yamls_dir}")
    print()
    print("To launch boltz:  cd foldswitch_dir && boltz predict yamls/seq_X.yaml --out_dir predictions/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
