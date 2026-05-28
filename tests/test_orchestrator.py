"""Tests for the shared chunked-extraction orchestrator.

Subprocess launch itself is not covered here — that needs a real entrypoint
module and is exercised by the integration runs. These tests pin the
mechanical pieces: chunking, discovery, subset filtering, config helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from protein_interpretability.orchestrator import (
    discover_fastas,
    discover_yamls,
    format_list_arg,
    load_config,
    load_subset,
    make_chunks,
    required,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def test_load_config_round_trips_yaml(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    import yaml as _yaml

    p = tmp_path / "cfg.yaml"
    p.write_text(_yaml.safe_dump({"a": 1, "b": {"c": "two"}}))
    assert load_config(p) == {"a": 1, "b": {"c": "two"}}


def test_required_returns_nested_value() -> None:
    cfg = {"input": {"sequences_dir": "/x/y"}}
    assert required(cfg, "input", "sequences_dir") == "/x/y"


@pytest.mark.parametrize(
    "cfg",
    [
        {},
        {"input": {}},
        {"input": {"sequences_dir": None}},
        {"input": {"sequences_dir": ""}},
        {"input": "not-a-dict"},
    ],
)
def test_required_raises_on_missing_or_empty(cfg: dict) -> None:
    with pytest.raises(ValueError, match="Missing required config key"):
        required(cfg, "input", "sequences_dir")


def test_load_subset_ignores_comments_and_blanks(tmp_path: Path) -> None:
    p = tmp_path / "subset.txt"
    p.write_text("seq_00001\n# a comment\n\nseq_00002\n  \nseq_00003\n")
    assert load_subset(p) == {"seq_00001", "seq_00002", "seq_00003"}


def test_load_subset_none_returns_none() -> None:
    assert load_subset(None) is None


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "all"),
        ("all", "all"),
        ("  ALL  ", "all"),
        ("0,1,2", "0,1,2"),
        ([0, 1, 2], "0,1,2"),
        (("layer_s", "layer_z"), "layer_s,layer_z"),
        ([], "none"),
    ],
)
def test_format_list_arg_accepts_known_shapes(value, expected: str) -> None:
    assert format_list_arg(value) == expected


def test_format_list_arg_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Cannot parse list-style arg"):
        format_list_arg(42)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    return p


def test_discover_yamls_walks_recursively(tmp_path: Path) -> None:
    _touch(tmp_path / "seq_00001" / "seq_00001.yaml")
    _touch(tmp_path / "seq_00002" / "seq_00002.yaml")
    _touch(tmp_path / "seq_00003" / "msa" / "ignore.a3m")
    _touch(tmp_path / "notes.txt")

    found = discover_yamls(tmp_path, subset=None)
    assert [p.name for p in found] == ["seq_00001.yaml", "seq_00002.yaml"]


def test_discover_yamls_subset_filters_by_stem(tmp_path: Path) -> None:
    _touch(tmp_path / "seq_00001.yaml")
    _touch(tmp_path / "seq_00002.yaml")
    _touch(tmp_path / "seq_00003.yaml")

    found = discover_yamls(tmp_path, subset={"seq_00001", "seq_00003"})
    assert [p.name for p in found] == ["seq_00001.yaml", "seq_00003.yaml"]


def test_discover_fastas_matches_known_extensions(tmp_path: Path) -> None:
    _touch(tmp_path / "a.fasta")
    _touch(tmp_path / "b.fa")
    _touch(tmp_path / "c.faa")
    _touch(tmp_path / "nested" / "d.FASTA")  # case-insensitive
    _touch(tmp_path / "e.txt")
    _touch(tmp_path / "f.yaml")

    found = discover_fastas(tmp_path, subset=None)
    assert {p.name for p in found} == {"a.fasta", "b.fa", "c.faa", "d.FASTA"}


def test_discover_fastas_subset_filters_by_stem(tmp_path: Path) -> None:
    _touch(tmp_path / "seq_00001.fasta")
    _touch(tmp_path / "seq_00002.fa")
    _touch(tmp_path / "seq_00003.faa")

    found = discover_fastas(tmp_path, subset={"seq_00002"})
    assert [p.name for p in found] == ["seq_00002.fa"]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _make_items(dirpath: Path, n: int, ext: str = ".yaml") -> list[Path]:
    items = [_touch(dirpath / f"item_{i:03d}{ext}") for i in range(n)]
    return items


def test_make_chunks_even_split(tmp_path: Path) -> None:
    items = _make_items(tmp_path / "src", 8)
    chunks = make_chunks(items, tmp_path / "stage", num_chunks=4)
    assert len(chunks) == 4
    sizes = [sum(1 for _ in d.iterdir()) for d in chunks]
    assert sizes == [2, 2, 2, 2]


def test_make_chunks_uneven_split_puts_extras_in_early_chunks(tmp_path: Path) -> None:
    items = _make_items(tmp_path / "src", 10)
    chunks = make_chunks(items, tmp_path / "stage", num_chunks=3)
    sizes = [sum(1 for _ in d.iterdir()) for d in chunks]
    assert sizes == [4, 3, 3]


def test_make_chunks_more_chunks_than_items_drops_empty(tmp_path: Path) -> None:
    items = _make_items(tmp_path / "src", 2)
    chunks = make_chunks(items, tmp_path / "stage", num_chunks=8)
    assert len(chunks) == 2
    for d in chunks:
        assert sum(1 for _ in d.iterdir()) == 1


def test_make_chunks_zero_or_negative_clamped_to_one(tmp_path: Path) -> None:
    items = _make_items(tmp_path / "src", 3)
    chunks = make_chunks(items, tmp_path / "stage", num_chunks=0)
    assert len(chunks) == 1
    assert sum(1 for _ in chunks[0].iterdir()) == 3


def test_make_chunks_uses_symlinks_pointing_at_sources(tmp_path: Path) -> None:
    items = _make_items(tmp_path / "src", 3)
    chunks = make_chunks(items, tmp_path / "stage", num_chunks=2)
    for d in chunks:
        for child in d.iterdir():
            assert child.is_symlink()
            assert child.resolve() == (tmp_path / "src" / child.name).resolve()


def test_make_chunks_idempotent_clears_stale_symlinks(tmp_path: Path) -> None:
    items = _make_items(tmp_path / "src", 5)
    stage = tmp_path / "stage"

    make_chunks(items, stage, num_chunks=2)

    # Drop two source files and rerun — stale symlinks must be cleared.
    items[3].unlink()
    items[4].unlink()
    remaining = items[:3]
    chunks = make_chunks(remaining, stage, num_chunks=2)

    seen = {child.name for d in chunks for child in d.iterdir()}
    assert seen == {p.name for p in remaining}


def test_make_chunks_preserves_real_files_in_staging(tmp_path: Path) -> None:
    items = _make_items(tmp_path / "src", 2)
    stage = tmp_path / "stage"
    make_chunks(items, stage, num_chunks=1)

    real_log = stage / "chunk_0" / "do_not_delete.log"
    real_log.write_text("important")

    make_chunks(items, stage, num_chunks=1)
    assert real_log.exists()
    assert real_log.read_text() == "important"
