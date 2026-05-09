#!/usr/bin/env python3
"""Finalize foldswitch yaml/msa naming for cluster predictions.

Renames `yamls/seq_NNNN_<f1>_<f2>.yaml` → `yamls/seq_NNNNN.yaml` (5-digit
zero-padded, keeping the original TableS1 idx so seq_00014 is reserved for the
PDB-1FZP error row, etc.). Same for `msa/*.a3m`. Writes `identities.tsv`
mapping each new seq_id back to its provenance.

Optionally rewrites every yaml's `msa:` field to an absolute path so Boltz can
be launched from any cwd on the cluster:

    msa: msa/seq_00001.a3m       # default (relative; needs cd into foldswitch)
    msa: <abs_root>/msa/seq_00001.a3m   # with --abs_root /n/holylfs06/.../foldswitch

Idempotent: re-running on already-renamed files is a no-op for the files but
will refresh `identities.tsv` and (optionally) the absolute paths inside the
yamls.

Usage::

    # Local (before upload, or on cluster) — rename + write identities.tsv
    python scripts/foldswitch_finalize.py \\
        --foldswitch_dir /Users/.../data/foldswitch

    # On the cluster — rename (no-op if already done) + write absolute msa paths
    python scripts/foldswitch_finalize.py \\
        --foldswitch_dir /n/holylfs06/.../foldswitch \\
        --abs_root       /n/holylfs06/.../foldswitch
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

OLD_YAML_RE = re.compile(r"^seq_(\d{4})_([^.]+)\.yaml$")
OLD_A3M_RE = re.compile(r"^seq_(\d{4})_([^.]+)\.a3m$")
NEW_RE = re.compile(r"^seq_(\d{5})\.(yaml|a3m)$")
MSA_LINE_RE = re.compile(r"^(\s*msa:\s*).*$", re.MULTILINE)


def find_files_by_idx(directory: Path, suffix: str) -> dict[int, Path]:
    """Return {idx: path} for both old-format (seq_NNNN_*.{suffix}) and new
    (seq_NNNNN.{suffix}) files in `directory`."""
    out: dict[int, Path] = {}
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if suffix == "yaml":
            m_old = OLD_YAML_RE.match(p.name)
        elif suffix == "a3m":
            m_old = OLD_A3M_RE.match(p.name)
        else:
            raise ValueError(suffix)
        m_new = NEW_RE.match(p.name)
        if m_old:
            out[int(m_old.group(1))] = p
        elif m_new and m_new.group(2) == suffix:
            out[int(m_new.group(1))] = p
    return out


def rewrite_yaml_msa_line(yaml_path: Path, new_msa_path: str) -> None:
    txt = yaml_path.read_text()
    new_txt, n = MSA_LINE_RE.subn(rf"\1{new_msa_path}", txt, count=1)
    if n == 0:
        # No msa line — append one
        if not new_txt.endswith("\n"):
            new_txt += "\n"
        new_txt += f"      msa: {new_msa_path}\n"
    yaml_path.write_text(new_txt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--foldswitch_dir", type=Path, required=True)
    ap.add_argument("--abs_root", type=Path, default=None,
                    help="If set, rewrite every yaml's `msa:` to "
                         "<abs_root>/msa/seq_NNNNN.a3m. Use the cluster path.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Path to manifest_with_msa.csv (default: "
                         "<foldswitch_dir>/manifest_with_msa.csv)")
    args = ap.parse_args()

    fs_dir: Path = args.foldswitch_dir
    yamls_dir = fs_dir / "yamls"
    msa_dir = fs_dir / "msa"
    if not yamls_dir.is_dir() or not msa_dir.is_dir():
        sys.exit(f"missing yamls/ or msa/ in {fs_dir}")

    manifest_path = args.manifest or (fs_dir / "manifest_with_msa.csv")
    if not manifest_path.exists():
        # Fall back to manifest.csv (no msa columns) for identity info
        manifest_path = fs_dir / "manifest.csv"
        if not manifest_path.exists():
            sys.exit(f"no manifest in {fs_dir}")
    print(f"[info] manifest: {manifest_path.name}")

    with manifest_path.open() as f:
        rows = {int(r["idx"]): r for r in csv.DictReader(f)
                if r.get("idx") and r.get("idx").isdigit()}

    yaml_by_idx = find_files_by_idx(yamls_dir, "yaml")
    a3m_by_idx = find_files_by_idx(msa_dir, "a3m")

    # ------- Rename pass (idempotent) -------
    n_renamed_y = n_renamed_a = 0
    for idx, p in sorted(yaml_by_idx.items()):
        new_name = f"seq_{idx:05d}.yaml"
        if p.name != new_name:
            target = p.with_name(new_name)
            p.rename(target)
            yaml_by_idx[idx] = target
            n_renamed_y += 1

    for idx, p in sorted(a3m_by_idx.items()):
        new_name = f"seq_{idx:05d}.a3m"
        if p.name != new_name:
            target = p.with_name(new_name)
            p.rename(target)
            a3m_by_idx[idx] = target
            n_renamed_a += 1

    print(f"[rename] yamls renamed: {n_renamed_y}/{len(yaml_by_idx)}")
    print(f"[rename] a3m renamed:   {n_renamed_a}/{len(a3m_by_idx)}")

    # ------- Rewrite msa: lines inside yamls -------
    n_rewritten = 0
    for idx, yp in sorted(yaml_by_idx.items()):
        a3m_name = f"seq_{idx:05d}.a3m"
        if args.abs_root is not None:
            new_msa = str(args.abs_root / "msa" / a3m_name)
        else:
            new_msa = f"msa/{a3m_name}"
        rewrite_yaml_msa_line(yp, new_msa)
        n_rewritten += 1
    print(f"[yaml] msa lines rewritten: {n_rewritten}  "
          f"(target: {'absolute' if args.abs_root else 'relative'})")

    # ------- Write identities.tsv -------
    id_path = fs_dir / "identities.tsv"
    cols = ["seq_id", "idx_tableS1", "fold1", "fold2",
            "chain_used", "msa_status", "seq_len", "yaml", "msa"]
    with id_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for idx in sorted(yaml_by_idx.keys()):
            row = rows.get(idx, {})
            seq_id = f"seq_{idx:05d}"
            w.writerow([
                seq_id,
                idx,
                row.get("fold1", ""),
                row.get("fold2", ""),
                row.get("msa_chain_used", row.get("fold1", "")),
                row.get("msa_status", ""),
                row.get("seq_len", ""),
                f"yamls/{seq_id}.yaml",
                f"msa/{seq_id}.a3m",
            ])
    print(f"[identities] wrote -> {id_path} ({len(yaml_by_idx)} rows)")

    # Summary
    print()
    print(f"[done]  {len(yaml_by_idx)} yamls / {len(a3m_by_idx)} msas in {fs_dir}")
    print(f"        identities: {id_path}")
    if args.abs_root:
        print(f"        msa paths in yamls: ABSOLUTE under {args.abs_root}")
    else:
        print(f"        msa paths in yamls: RELATIVE (run boltz from {fs_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
