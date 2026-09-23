"""Site resolution and SLURM rendering: what they refuse, and that they agree.

The renderer's job is not to be clever. It is to describe the SAME job the
hand-written submitters describe, from one place instead of three, and to say so
on a login node before anything is queued. So the load-bearing test is the last
one here: rendered against `analysis.sbatch`, `run.sbatch` and
`checkout.sbatch`, field by field.

    uv run pytest tests/test_site.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from protein_interpretability.experiments.site import Site, SiteError, _split_refs
from protein_interpretability.experiments.slurm import (
    JobSpec,
    equivalent_to,
    render,
    sbatch_fields,
)

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "jax_harness"


def site_from(tmp_path, text) -> Site:
    p = tmp_path / "profile.yaml"
    p.write_text(text)
    from protein_interpretability.experiments.site import _parse
    return Site(_parse(text), [p])


# ---- variable expansion ----------------------------------------------------

def test_nested_variables_in_a_default_survive():
    """`${A:-/x/${USER}/y}` -- the inner brace closes the outer default under
    any `[^}]*` regex. The first version rendered `/x/${USER/y}`, a path that is
    wrong in a way a reader skims straight past."""
    pieces = list(_split_refs("${A:-/x/${USER}/y}"))
    assert pieces == [("ref", "A", "/x/${USER}/y")]


def test_expansion_prefers_the_environment_over_the_default(tmp_path, monkeypatch):
    s = site_from(tmp_path, "roots:\n  work: ${PI_TEST_ROOT:-/fallback}\n")
    assert s.get("roots.work") == "/fallback"
    monkeypatch.setenv("PI_TEST_ROOT", "/from/env")
    assert s.get("roots.work") == "/from/env"


def test_a_nested_default_expands_the_inner_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "someone")
    monkeypatch.delenv("PI_SCRATCH", raising=False)
    s = site_from(tmp_path, "c:\n  scratch: ${PI_SCRATCH:-/net/${USER}/mosaic}\n")
    assert s.get("c.scratch") == "/net/someone/mosaic"


def test_a_variable_with_no_value_and_no_default_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("PI_UNSET_THING", raising=False)
    s = site_from(tmp_path, "roots:\n  work: ${PI_UNSET_THING}\n")
    with pytest.raises(SiteError, match="not set and has no"):
        s.get("roots.work")


def test_a_section_reference_expands(tmp_path):
    s = site_from(tmp_path,
                  "roots:\n  setup: /opt/setup\ncontainer:\n"
                  "  image: ${roots.setup}/images/m.sif\n")
    assert s.get("container.image") == "/opt/setup/images/m.sif"


def test_a_circular_reference_is_refused(tmp_path):
    s = site_from(tmp_path, "a:\n  x: ${a.y}\n  y: ${a.x}\n")
    with pytest.raises(SiteError, match="circular"):
        s.get("a.x")


def test_a_missing_key_says_which_profiles_were_read(tmp_path):
    s = site_from(tmp_path, "roots:\n  work: /w\n")
    with pytest.raises(SiteError, match="profile.yaml"):
        s.get("roots.nonexistent")


# ---- verification ----------------------------------------------------------

def test_verify_refuses_a_root_that_does_not_exist(tmp_path):
    s = site_from(tmp_path,
                  f"roots:\n  work: {tmp_path}\n  repo: /no/such/place\n"
                  "scheduler:\n  account: a\n  partition: p\n")
    with pytest.raises(SiteError, match="not a directory"):
        s.verify()


def test_verify_reports_every_problem_at_once(tmp_path):
    s = site_from(tmp_path,
                  "roots:\n  work: /nope/a\n  repo: /nope/b\n"
                  "scheduler:\n  account: a\n  partition: p\n")
    with pytest.raises(SiteError) as exc:
        s.verify()
    assert "/nope/a" in str(exc.value) and "/nope/b" in str(exc.value)


def test_the_committed_profile_resolves_and_is_real():
    """A committed example that does not work teaches nothing."""
    Site.load().verify()


# ---- overriding ------------------------------------------------------------

def test_a_later_profile_overrides_key_by_key():
    from protein_interpretability.experiments.site import _merge
    base = {"scheduler": {"account": "a", "partition": "p", "mem_mb": 1}}
    over = {"scheduler": {"partition": "other"}}
    out = _merge(base, over)
    assert out["scheduler"] == {"account": "a", "partition": "other", "mem_mb": 1}, (
        "an override must be able to set one field without restating the block")


# ---- rendering -------------------------------------------------------------

def test_render_puts_the_args_outside_the_quoted_script_path():
    """`python "…/analyze_svd.py --out x"` looks for a file with spaces in its
    name. The first version did exactly that."""
    text = render(JobSpec(script="analyze_svd.py", args=("--out", "/tmp/x.json")))
    exec_line = [l for l in text.splitlines() if l.startswith("exec ")][0]
    assert exec_line.endswith('analyze_svd.py" --out /tmp/x.json')


def test_render_quotes_an_argument_containing_spaces():
    text = render(JobSpec(script="s.py", args=("--glob", "a b*.npz")))
    assert "'a b*.npz'" in text


def test_source_selects_the_directory_the_job_runs_from():
    mirror = render(JobSpec(script="s.py", source="mirror"))
    checkout = render(JobSpec(script="s.py", source="checkout"))
    assert "/harness/s.py" in mirror
    assert "/jax_harness/s.py" in checkout


def test_an_unknown_source_is_refused():
    with pytest.raises(SiteError, match="source must be"):
        render(JobSpec(script="s.py", source="somewhere_else"))


def test_per_job_resources_override_the_profile():
    text = render(JobSpec(script="s.py", mem_mb=4096, time_min=7))
    fields = sbatch_fields(text)
    assert fields["mem"] == "4096" and fields["time"] == "7"


def test_exclusive_is_emitted_only_when_asked():
    assert "--exclusive" not in render(JobSpec(script="s.py"))
    assert "--exclusive" in render(JobSpec(script="s.py", exclusive=True))


# ---- the test that matters -------------------------------------------------

@pytest.mark.parametrize("submitter", ["analysis.sbatch", "run.sbatch",
                                       "checkout.sbatch"])
def test_the_renderer_describes_the_same_job_as_the_scripts_in_use(submitter):
    """Rendered against the files that have produced every archived result.

    A renderer that drifts from them is worse than none: it would look
    authoritative while describing a job nobody runs.
    """
    path = HARNESS / submitter
    source = "checkout" if submitter == "checkout.sbatch" else "mirror"
    text = render(JobSpec(script="analyze_svd.py", source=source))
    assert equivalent_to(text, path) == []


def test_equivalence_actually_notices_a_difference(tmp_path):
    """Guard the guard: a comparison that cannot fail proves nothing."""
    other = tmp_path / "other.sbatch"
    other.write_text("#!/bin/bash\n#SBATCH --partition=some_other_partition\n")
    assert equivalent_to(render(JobSpec(script="s.py")), other)
