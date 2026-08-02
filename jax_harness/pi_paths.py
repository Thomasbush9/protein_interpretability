"""Path patching for the Boltz-2 trunk.

The question this answers: when the query sequence and the MSA disagree (a
mutant sequence carrying its wild-type alignment), which input route actually
determines the predicted structure?

Four separable routes carry information into the pair representation z:

  P1  "z_direct"   s_inputs -> z_init_1/z_init_2 outer sum
  P2  "msa_bcast"  s_inputs -> msa_module.s_proj, added to every MSA row
  P3  "msa_query"  MSA row 0 (the query itself) -> OuterProductMean
  P4  "msa_prior"  MSA rows 1..S -> OuterProductMean            <- the "cheat sheet"

plus  S1 "s_direct"  s_inputs -> s_init (the single representation)

Each can be independently sourced from a *donor* run while everything else
comes from the *recipient*. Running the mutant while patching one route back to
wild-type isolates that route's causal contribution to the prediction.

Note on OuterProductMean: it divides by the number of unmasked MSA rows, so P3
enters z at roughly 1/S the weight of P4. P2 is not diluted this way -- it is
broadcast to every row before the mean. Whether the *net* query influence
decays with MSA depth is an empirical question, and `sweep_msa_depth` in
pi_experiments.py is what measures it.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

import joltz
from joltz import TrunkState

import pi_core as pi

# Route names accepted by `patch`.
ROUTES = ("z_direct", "s_direct", "msa_bcast", "msa_query", "msa_prior")


def build_hybrid(emb_recipient, emb_donor, feats_recipient, feats_donor, routes):
    """Return (emb, feats) with `routes` sourced from the donor.

    `routes` is an iterable of names from ROUTES. Everything not named comes
    from the recipient.
    """
    routes = set(routes)
    unknown = routes - set(ROUTES)
    if unknown:
        raise ValueError(f"unknown route(s): {sorted(unknown)}; valid: {ROUTES}")

    emb = emb_recipient
    if "z_direct" in routes:
        emb = eqx.tree_at(lambda e: e.z_init, emb, emb_donor.z_init)
    if "s_direct" in routes:
        emb = eqx.tree_at(lambda e: e.s_init, emb, emb_donor.s_init)
    if "msa_bcast" in routes:
        # s_inputs is consumed by msa_module.s_proj inside trunk_iteration.
        emb = eqx.tree_at(lambda e: e.s_inputs, emb, emb_donor.s_inputs)

    feats = dict(feats_recipient)
    if "msa_query" in routes or "msa_prior" in routes:
        msa_r, msa_d = feats_recipient["msa"], feats_donor["msa"]
        # axis 1 is DEPTH and is allowed to differ (handled below); everything
        # else -- batch, tokens, encoding -- must match
        if (msa_r.shape[0] != msa_d.shape[0]
                or msa_r.shape[2:] != msa_d.shape[2:]):
            raise ValueError(
                f"MSA shapes differ beyond depth ({msa_r.shape} vs {msa_d.shape}); "
                "patching requires the same tokens and encoding."
            )
        if msa_r.shape[1] != msa_d.shape[1]:
            # Depth may differ by a few rows even from identically-capped a3m
            # files, because Boltz-2 dedups against the QUERY and each variant's
            # query differs. With grafted alignments this never happens; with
            # genuinely re-searched ones it does, by ~1-15 rows out of ~300-500.
            #
            # Patch the common prefix. Legitimate here ONLY because the homolog
            # sets are ~98 % identical by UniRef ID (measured), so the truncated
            # tail is a handful of low-ranked hits, not a different alignment.
            # Do NOT use this to paper over genuinely different MSAs.
            n = min(msa_r.shape[1], msa_d.shape[1])
            # `msa` is not alone on the depth axis: deletion_value, has_deletion
            # and friends are aligned to it and are concatenated with it
            # downstream. Truncating `msa` by itself produces a concat error, so
            # every feature whose axis 1 matches the old depth is truncated too.
            def _trim(fd, old, n):
                out = {}
                for k, v in fd.items():
                    if hasattr(v, "shape") and v.ndim >= 2 and v.shape[1] == old:
                        out[k] = v[:, :n]
                    else:
                        out[k] = v
                return out
            feats_recipient = _trim(feats_recipient, msa_r.shape[1], n)
            feats_donor = _trim(feats_donor, msa_d.shape[1], n)
            msa_r, msa_d = feats_recipient["msa"], feats_donor["msa"]
        msa = msa_r
        if "msa_query" in routes:
            msa = msa.at[:, 0].set(msa_d[:, 0])
        if "msa_prior" in routes:
            msa = msa.at[:, 1:].set(msa_d[:, 1:])
        feats["msa"] = msa

    return emb, feats


# `iteration` and `run_trunk` live in pi_core -- one definition only.
# See the note in pi_core.iteration about why a second variant is a trap.
iteration = pi.iteration
run_trunk = pi.run_trunk


def patch(
    model,
    feats_recipient,
    feats_donor,
    routes,
    *,
    recycling_steps=3,
    key,
    deterministic=True,
    capture_last=False,
):
    """Run the recipient with `routes` sourced from the donor.

    routes=() reproduces the plain recipient run; routes=ROUTES reproduces the
    donor. Both are worth asserting as sanity checks on any new pair of inputs.
    """
    emb_r = model.embed_inputs(feats_recipient)
    emb_d = model.embed_inputs(feats_donor)
    emb, feats = build_hybrid(emb_r, emb_d, feats_recipient, feats_donor, routes)
    return run_trunk(
        model, emb, feats,
        recycling_steps=recycling_steps, key=key,
        deterministic=deterministic, capture_last=capture_last,
    )


# --------------------------------------------------------------------------
# MSA depth control
# --------------------------------------------------------------------------
def truncate_msa(feats, depth: int, *, keep_query: bool = True):
    """Return feats with the MSA cut to `depth` rows (row 0 = the query).

    OuterProductMean normalises by the number of unmasked rows, so this is the
    knob that sets the query's share of the MSA -> pair write. Rows are taken in
    file order, which for a ColabFold a3m is roughly by decreasing similarity;
    `sweep_msa_depth` reports that caveat alongside its numbers.
    """
    out = dict(feats)
    for k in ("msa", "msa_mask", "has_deletion", "deletion_value", "msa_paired"):
        if k in out:
            out[k] = out[k][:, :depth]
    if keep_query and depth < 1:
        raise ValueError("depth must be >= 1 to retain the query row")
    return out


def effective_depth(feats) -> float:
    """Mean number of unmasked MSA rows -- the divisor OPM actually applies."""
    return float(jnp.asarray(feats["msa_mask"]).sum(axis=1).mean())
