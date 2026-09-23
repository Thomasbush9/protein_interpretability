"""Does the CLI refuse what it claims to refuse?

The three commands exist to catch things late-failing jobs otherwise catch at
the end, so what matters is the bad path: a result with no recorded command, a
difference above a producer's known band, a script that cannot be imported
without a GPU.

    uv run pytest tests/test_cli.py
"""

from __future__ import annotations

import json

import pytest

from protein_interpretability.cli.compare import KNOWN_BANDS, compare, verdict
from protein_interpretability.cli.main import main


# ---- compare / verdict -----------------------------------------------------

def test_identical_results_are_exact():
    doc = {"a": 1.0, "b": [1, 2, 3], "protocol": {"design": "x"}}
    d = compare(doc, json.loads(json.dumps(doc)))
    ok, why = verdict("anything", d)
    assert ok and why == "exact"
    assert d.n_numbers == 4


def test_provenance_and_timestamp_are_ignored():
    old = {"x": 1.0, "provenance": {"host": "a", "argv": ["p"]},
           "protocol": {"recorded": "2026-08-12 22:03 UTC"}}
    new = {"x": 1.0, "provenance": {"host": "b", "argv": ["q"]},
           "protocol": {"recorded": "2026-08-17 16:50 UTC"}}
    ok, _ = verdict("anything", compare(old, new))
    assert ok, "host, argv and the protocol timestamp differ between any two runs"


def test_unknown_producer_must_be_exact():
    ok, why = verdict("some_new_producer", compare({"x": 1.0}, {"x": 1.0001}))
    assert not ok
    assert "run the unchanged code twice" in why


def test_known_band_absorbs_its_own_noise():
    """gate_probe moves ~1e-6 between identical runs; that must not read as a
    regression, which is the mistake this table exists to prevent."""
    ok, why = verdict("gate_probe", compare({"x": 1.0}, {"x": 1.0 + 1.7e-6}))
    assert ok and "known band" in why


def test_known_band_is_not_a_blank_cheque():
    ok, _ = verdict("gate_probe", compare({"x": 1.0}, {"x": 1.5}))
    assert not ok, "a band absorbs that producer's noise, not any difference"


def test_structural_difference_fails_even_inside_a_band():
    d = compare({"x": 1.0, "curve": [1, 2, 3]}, {"x": 1.0, "curve": [1, 2]})
    ok, why = verdict("svd_ds_v1", d)
    assert not ok and "structural" in why


def test_missing_and_extra_keys_are_reported():
    d = compare({"a": 1, "gone": 2}, {"a": 1, "added": 3})
    assert any("gone" in i for i in d.issues)
    assert any("added" in i for i in d.issues)


def test_nan_matches_nan():
    ok, _ = verdict("anything", compare({"x": float("nan")}, {"x": float("nan")}))
    assert ok, "a NaN in both results is agreement, not a difference"


def test_every_band_records_why():
    for name, (tol, why) in KNOWN_BANDS.items():
        assert tol > 0 and len(why) > 20, (
            f"{name}: a tolerance without its measurement is a number someone "
            "will later widen")


# ---- reproduce -------------------------------------------------------------

def test_reproduce_refuses_a_result_with_no_recorded_command(tmp_path):
    bad = tmp_path / "nameless.json"
    bad.write_text(json.dumps({"value": 1.0}))
    with pytest.raises(SystemExit, match="provenance.argv"):
        main(["reproduce", str(bad), "--out", str(tmp_path)])


def test_reproduce_prints_without_submitting(tmp_path, capsys):
    res = tmp_path / "thing_v1.json"
    res.write_text(json.dumps({
        "v": 1, "provenance": {"argv": ["/somewhere/harness/analyze_thing.py",
                                        "--glob", "x_*.npz",
                                        "--out", "/old/thing_v1.json"]}}))
    assert main(["reproduce", str(res), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "sbatch" in out and "analyze_thing.py" in out
    assert "--glob x_*.npz" in out, "arguments must replay verbatim"
    assert str(tmp_path / "o" / "thing_v1.json") in out, "--out is redirected"
    assert "/old/thing_v1.json" not in out


def test_reproduce_selects_the_checkout_submitter(tmp_path, capsys):
    res = tmp_path / "t.json"
    res.write_text(json.dumps({"provenance": {"argv": ["h/a.py", "--out", "o"]}}))
    main(["reproduce", str(res), "--out", str(tmp_path), "--checkout"])
    assert "checkout.sbatch" in capsys.readouterr().out


# ---- inspect ---------------------------------------------------------------

def test_inspect_flags_a_module_scope_backend_import(tmp_path, capsys):
    f = tmp_path / "collector.py"
    f.write_text("import joltz\n\ndef main():\n    pass\n")
    assert main(["inspect", str(f)]) == 1
    assert "login node" in capsys.readouterr().out


def test_inspect_allows_a_backend_imported_inside_a_function(tmp_path):
    f = tmp_path / "lazy.py"
    f.write_text("def build():\n    import mosaic\n    return mosaic\n")
    assert main(["inspect", str(f)]) == 0, (
        "importing a backend lazily is the pattern that keeps inspection "
        "runnable without a GPU")


def test_inspect_flags_a_result_written_with_no_protocol(tmp_path, capsys):
    f = tmp_path / "writer.py"
    f.write_text("import pi_archive\n\ndef main():\n"
                 "    pi_archive.write_result(p, {}, protocol={})\n")
    assert main(["inspect", str(f)]) == 1
    assert "PROTOCOL" in capsys.readouterr().out


def test_inspect_accepts_the_inline_protocol_idiom(tmp_path):
    f = tmp_path / "inline.py"
    f.write_text("import pi_archive, pi_protocol\n\ndef main():\n"
                 "    res = {'protocol': pi_protocol.protocol(design='x')}\n"
                 "    pi_archive.write_result(p, res, protocol=res['protocol'])\n")
    assert main(["inspect", str(f)]) == 0, (
        "the producers build the block inside main; only checking for a "
        "module-level constant flagged four that have always carried one")


def test_inspect_accepts_an_aliased_protocol_import(tmp_path):
    """`from ... import protocol as P` then `P.protocol(...)`.

    Matching the text `pi_protocol.protocol(` missed this and flagged a script
    whose block is complete -- the second false positive of the same shape, one
    import alias later. Hence matching the call, not the spelling.
    """
    f = tmp_path / "aliased.py"
    f.write_text(
        "from protein_interpretability import artifacts\n"
        "from protein_interpretability.experiments import protocol as P\n\n"
        "def main():\n"
        "    artifacts.write_result(p, {}, protocol=P.protocol(design='x'))\n")
    assert main(["inspect", str(f)]) == 0


def test_inspect_still_flags_a_result_with_no_protocol_call_at_all(tmp_path):
    f = tmp_path / "bare.py"
    f.write_text("from protein_interpretability import artifacts\n\n"
                 "def main():\n"
                 "    artifacts.write_result(p, {}, protocol=SOME_DICT)\n")
    assert main(["inspect", str(f)]) == 1, (
        "the guard must still catch a block that never went through protocol()")


def test_inspect_ignores_write_result_mentioned_in_prose(tmp_path):
    """A module that only NAMES write_result in an error message is not a writer.

    The substring test flagged pi_report, whose sole mention is inside a string
    telling the reader to go through the seam.
    """
    f = tmp_path / "library.py"
    f.write_text('def check(x):\n'
                 '    raise ValueError("rerun it through '
                 'pi_archive.write_result, or accept the truncation")\n')
    assert main(["inspect", str(f)]) == 0


def test_inspect_reports_a_syntax_error_rather_than_raising(tmp_path, capsys):
    f = tmp_path / "broken.py"
    f.write_text("def (:\n")
    assert main(["inspect", str(f)]) == 1
    assert "SYNTAX ERROR" in capsys.readouterr().out
