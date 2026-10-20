"""The AUTHORITATIVE training-pool fingerprint for the held-out leakage gate.

The gate in ``heldout_common.check_corpus_leakage`` needs the set of
normalised-exact-match shas of every prompt the stage-3 defense was TRAINED on.
This module produces that set from the AUTHORITATIVE source — the realised
labelled store the config declares — and refuses (never silently returns an
empty set) when the source is unavailable, exactly like
``build_two_si_pool._scoring_prompt_shas`` (a silently empty exclusion set would
let a leaked family through, which is worse than no family).

WHY the store, not a re-derivation. The training pool is generated once and
persisted to ``pool.store_dir`` as one JSONL per attack family, each row
carrying the exact training prompt in ``attack_prompt``. That is the AUTHORITATIVE
record of what was trained on (``sources.build_inventory`` would RE-derive the
composition, but it is not on this branch and re-derivation can drift from what
was realised). The set of ``normalized_text_sha(attack_prompt)`` over the
declared ``pool.train_families`` files IS the training fingerprint.

PORTABILITY. The full sha SET can be large (BoN alone is hundreds of thousands
of augmented rows), so it is NOT committed. Instead the gate runs where the store
lives (the grading host), and the per-family MANIFEST records the resulting
``training_fingerprint_sha`` + counts; a replay elsewhere trusts that manifest
and re-verifies the corpus content hash rather than re-reading the store. A small
committed META attestation (:func:`write_fingerprint_meta` /
:func:`load_fingerprint_meta`) pins the expected ``fingerprint_sha`` so drift is
caught loudly.

This module is imported only by the gate preflight / the rewriter/ingested build steps /
the tests — NEVER by a family module at registry-import time — so importing an
attack for the registry never reads a 600 MB store.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from ..jailbreak_data import normalized_text_sha
from . import eval_only
from .heldout_common import fingerprint_sha, training_fingerprint


def _default_store() -> Path:
    """The labeled store under the grading host private root
    (:func:`sisteer.eval.jailbreak_artifacts.grading_host_private_root`)."""
    from ..jailbreak_artifacts import grading_host_private_root
    return grading_host_private_root() / "labeled"


def _store_dir(explicit: Optional[str] = None) -> Path:
    """Resolve the store dir: explicit > ``$TT_TWO_SI_STORE_DIR`` > the module default
    (:func:`_default_store`). A config's ``pool.store_dir`` is in placeholder form
    (``$STATE/...``, ``$WORK/...``) and is expanded through ``sisteer.site``, which refuses
    an unset placeholder by name rather than reading a literal ``$STATE`` directory."""
    cand = explicit or os.environ.get("TT_TWO_SI_STORE_DIR")
    if not cand:
        return _default_store()
    from sisteer import site as _site
    return _site.expand_path(cand, key="pool.store_dir")


def _iter_store_prompts(path: Path, field: str = "attack_prompt") -> Iterable[str]:
    """Yield every ``field`` value from a store JSONL. Tolerates a torn final
    line; skips rows missing the field (an ungraded/placeholder row) rather than
    crashing the whole read."""
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        val = rec.get(field)
        if isinstance(val, str) and val:
            yield val


def load_training_fingerprint(
    train_families: Iterable[str],
    *,
    store_dir: Optional[str] = None,
    benign_families: Iterable[str] = (),
) -> frozenset[str]:
    """The normalised-exact-match fingerprint of the training pool: the union of
    ``normalized_text_sha(attack_prompt)`` over each declared training family's
    store file (and any ``benign_families``, if a run wants the benign side
    included too). REFUSES loudly if the store dir or a declared family's file is
    absent — an authoritative source that is missing is surfaced as an error, not
    degraded to an empty (vacuously-passing) gate."""
    root = _store_dir(store_dir)
    families = list(train_families) + list(benign_families)
    if not families:
        raise ValueError(
            "no training families declared — the leakage gate needs a non-empty "
            "train_families to check against (a vacuous gate is worse than none)")
    if not root.exists():
        raise FileNotFoundError(
            f"training store {root} not found — the held-out leakage gate is "
            "authoritative and cannot pass without the training pool to check "
            "against. Set $TT_TWO_SI_STORE_DIR to the realised labelled store, "
            "or run the gate where it lives (the grading host).")
    shas: set[str] = set()
    missing: list[str] = []
    for fam in families:
        fp = root / f"{fam}.jsonl"
        if not fp.exists():
            missing.append(fam)
            continue
        def _fingerprinted(prompts):
            for p in prompts:
                shas.add(normalized_text_sha(p))
                yield p

        # one streamed pass: a training store that carries an evaluation-only corpus's prompts (the
        # ReNeLLM paper corpus, under its no-training terms) refuses here, whatever family file holds them
        eval_only.refuse_texts(_fingerprinted(_iter_store_prompts(fp)),
                               consumer=f"the held-out leakage gate's training fingerprint ({fp})")
    if missing:
        raise FileNotFoundError(
            f"training store {root} is missing declared family file(s) "
            f"{missing} — refusing a partial fingerprint (a silently-missing "
            "family would let its prompts leak into a held-out corpus).")
    return frozenset(shas)


def load_stage3_training_fingerprint_from_config(cfg: dict) -> frozenset[str]:
    """Load the fingerprint from a run config's declared pool: the store dir +
    ``pool.train_families``. This reads what the config AUTHORITATIVELY names as
    the training pool, so the gate tracks whatever families a branch actually
    trains on (e.g. ``[bon]`` on unified-sweep, the eight static families on the
    stage-3 lineage).

    The store dir resolves ``$TT_TWO_SI_STORE_DIR`` (env) BEFORE ``pool.store_dir``,
    matching ``build_two_si_pool._store_dir`` exactly. This is load-bearing: the
    training pool the defense was fit on is built by that pool builder, which lets
    the env var override the config default (on the grading host the config default
    belongs to a DIFFERENT lane; the stage-3 store lives at the env-pointed dir).
    If the gate ignored the env and read only the config default, it would check
    a held-out corpus against the WRONG store — a vacuously-passing gate on a lane
    the defense never trained on. So the gate MUST resolve the store the same way
    the pool builder did."""
    pool = cfg.get("pool", {})
    env = os.environ.get("TT_TWO_SI_STORE_DIR")
    return load_training_fingerprint(
        pool.get("train_families", []),
        store_dir=env or pool.get("store_dir"),
    )


# --------------------------------------------------------------------------- #
# Committed META attestation (leak-free: shas/counts only, never prompt text)  #
# --------------------------------------------------------------------------- #


def fingerprint_meta(fingerprint: Iterable[str], *, provenance: dict) -> dict:
    """A small, committable attestation of a fingerprint: its ``fingerprint_sha``,
    ``n_shas``, the build provenance, and the primitive used. No prompt text —
    just the one-way hash of the set and counts."""
    fp = frozenset(fingerprint)
    return {
        "fingerprint_sha": fingerprint_sha(fp),
        "n_shas": len(fp),
        "primitive": "normalized_text_sha",
        "provenance": dict(provenance),
    }


def write_fingerprint_meta(fingerprint, path, *, provenance: dict) -> dict:
    meta = fingerprint_meta(fingerprint, provenance=provenance)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=1, sort_keys=True))
    return meta


def load_fingerprint_meta(path) -> dict:
    return json.loads(Path(path).read_text())


def verify_fingerprint(fingerprint, meta_path) -> None:
    """Raise if a loaded fingerprint's sha does not match the committed META —
    the store drifted from what was attested, and the gate would be checking
    against a different pool than the one on record."""
    got = fingerprint_sha(fingerprint)
    meta = load_fingerprint_meta(meta_path)
    if got != meta["fingerprint_sha"]:
        raise ValueError(
            f"training fingerprint drifted: store yields {got}, but "
            f"{meta_path} attests {meta['fingerprint_sha']} "
            f"({meta['n_shas']} shas). Regenerate the attestation if the "
            "training pool legitimately changed.")
