"""Unit tests for the query-occlusion helpers.

The full forward sweep needs Boltz weights, so these tests cover the pure
helpers — variant naming, KL math, valid-pair aggregation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

# Load the script as a module without requiring it on PYTHONPATH.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_query_occlusion.py"
_spec = importlib.util.spec_from_file_location("run_query_occlusion", _SCRIPT)
occ = importlib.util.module_from_spec(_spec)
sys.modules["run_query_occlusion"] = occ
# Avoid importing boltz at test time — patch the heavy imports out.
# The helpers we test don't need them, so skip executing the full module.
# Read the source and exec only the pure helpers we need.
_SRC = _SCRIPT.read_text()
_NS: dict = {"__name__": "run_query_occlusion_helpers"}
exec(  # noqa: S102 - test-only sandbox
    "\n".join(
        [
            "from __future__ import annotations",
            "import torch",
            "BASELINE_SUFFIX = '__baseline'",
            "REVERT_PREFIX = '__revert_'",
        ]
    ),
    _NS,
)
# Pull just the helper definitions out of the source by string match.
for fn in (
    "revert_stem",
    "parse_variant_id",
    "kl_per_pair",
    "aggregate_divergence",
    "write_variant_a3m",
):
    start = _SRC.index(f"def {fn}(")
    # End at the next top-level def/decorator or '# ---' comment block.
    rest = _SRC[start:]
    end = len(rest)
    for marker in ("\ndef ", "\n# ---", "\nclass "):
        idx = rest.find(marker, 1)
        if idx != -1:
            end = min(end, idx)
    exec(rest[:end], _NS)  # noqa: S102

revert_stem = _NS["revert_stem"]
parse_variant_id = _NS["parse_variant_id"]
kl_per_pair = _NS["kl_per_pair"]
aggregate_divergence = _NS["aggregate_divergence"]
write_variant_a3m = _NS["write_variant_a3m"]


def test_revert_stem_format() -> None:
    assert revert_stem("seq_00132", 23, "K") == "seq_00132__revert_0023_K"


def test_parse_variant_id_baseline() -> None:
    kind, pos, aa = parse_variant_id("seq_00132__baseline", "seq_00132")
    assert kind == "baseline"
    assert pos is None and aa is None


def test_parse_variant_id_revert() -> None:
    kind, pos, aa = parse_variant_id("seq_00132__revert_0023_K", "seq_00132")
    assert kind == "revert"
    assert pos == 23
    assert aa == "K"


def test_parse_variant_id_unknown() -> None:
    kind, pos, aa = parse_variant_id("something_else", "seq_00132")
    assert kind == "unknown"
    assert pos is None and aa is None


def test_kl_zero_when_identical() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 4, 8)
    kl = kl_per_pair(logits, logits)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_kl_positive_when_different() -> None:
    torch.manual_seed(0)
    a = torch.randn(1, 4, 4, 8)
    b = torch.randn(1, 4, 4, 8)
    kl = kl_per_pair(a, b)
    # KL is non-negative
    assert (kl >= -1e-6).all()
    # And not all zero for random distinct logits
    assert kl.abs().sum() > 0


def test_aggregate_excludes_diag_and_padding() -> None:
    B, N = 1, 4
    kl_map = torch.arange(B * N * N, dtype=torch.float32).view(B, N, N)
    # Last token padded
    mask = torch.tensor([[True, True, True, False]])
    agg = aggregate_divergence(kl_map, mask, exclude_diag=True)
    # Valid (i,j) pairs: i,j in {0,1,2}, i != j -> 6 pairs
    assert agg["n_pairs"] == 6
    # Pairs touching index 3 must not contribute
    assert agg["kl_max"] < float(kl_map[0, 2, 3])  # touches padded token


def test_aggregate_no_valid_pairs_returns_nan() -> None:
    kl_map = torch.zeros(1, 3, 3)
    mask = torch.tensor([[False, False, False]])
    agg = aggregate_divergence(kl_map, mask)
    assert agg["n_pairs"] == 0
    assert agg["kl_mean"] != agg["kl_mean"]  # NaN check


def test_write_variant_a3m_replaces_first_sequence_only(tmp_path) -> None:
    src = tmp_path / "src.a3m"
    src.write_text(
        ">query\n"
        "MKLAVTGEDDQ\n"
        ">homolog_1\n"
        "MKLAVTGEAAQ\n"
        ">homolog_2 with insertion\n"
        "MKLav-TGEDDQ\n"
    )
    out = tmp_path / "patched.a3m"
    write_variant_a3m(src, "MKLAVTGEDQQ", out)
    text = out.read_text().splitlines()
    # Header + new query, then unchanged homolog rows.
    assert text[0] == ">query"
    assert text[1] == "MKLAVTGEDQQ"
    assert text[2] == ">homolog_1"
    assert text[3] == "MKLAVTGEAAQ"
    assert text[4] == ">homolog_2 with insertion"
    assert text[5] == "MKLav-TGEDDQ"


def test_write_variant_a3m_length_mismatch_raises(tmp_path) -> None:
    src = tmp_path / "src.a3m"
    src.write_text(">q\nMKLA\n>h\nMKLA\n")
    out = tmp_path / "out.a3m"
    try:
        write_variant_a3m(src, "MKLAA", out)  # wrong length
    except ValueError as e:
        assert "length" in str(e)
        return
    raise AssertionError("expected ValueError on length mismatch")
