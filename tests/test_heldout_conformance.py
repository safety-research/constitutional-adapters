"""Conformance suite EVERY registered held-out family must pass.

Data-driven over ``attacks.HELD_OUT_FAMILIES`` so adding a family is one module
+ one registry line + one fixture — this suite then covers it automatically.
For each family it exercises the invariants the held-out scaffolding guarantees
(``heldout_common``):

- determinism (same seed -> byte-identical materialised corpus),
- variant count (every covered goal yields >= 1 variant),
- no leakage (no prompt normalised-exact-matches the training pool),
- manifest completeness + servability (numbers/enum-ids only),
- content-hash stability (invariant to row/goal ordering),
- replay-resume idempotence (a second run adds no rows; an interrupted goal's
  partial rows are dropped and the goal cleanly redone) — via the shared static
  contract check.

CPU-only, model-free, no network, no API (rewriter build kind uses a StubRewriter), and no
training store (a SYNTHETIC fingerprint) — so the whole matrix runs in CI.

Everything is SYNTHETIC (invented benign goals, no real prompts) per the repo's
dual-use containment rule.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from attacks_static_common import StubDefense, read_jsonl  # noqa: E402

import sisteer.eval.attacks as A  # noqa: E402
from sisteer.eval.attacks.heldout_common import (  # noqa: E402
    conformance_report,
    materialize_corpus,
    training_fingerprint,
    write_family_manifest,
)
from sisteer.eval.jailbreak_data import Goal  # noqa: E402

# A synthetic training fingerprint: strings the held-out transforms will never
# reproduce, so the gate passes for a correct family and the test needs no store.
_FINGERPRINT = training_fingerprint([
    "an unrelated training prompt about gardening",
    "another training row about bicycle maintenance",
])

# Generic synthetic eval goals (invented, benign) — used for the deterministic / rewriter kinds.
_GOALS = [
    Goal("hc_g0", "assemble the widget carefully", "illegal", "harmbench", "rev"),
    Goal("hc_g1", "prepare the sample thoroughly", "cybercrime_intrusion", "jbb", "rev"),
]


@pytest.fixture(autouse=True)
def _isolate_env():
    """Snapshot/restore the corpus env vars and clear module caches so the
    parametrized families don't leak corpus state into one another."""
    import os

    keys = ("SISTEER_HELDOUT_CORPUS_DIR", "SISTEER_FROZEN_CORPUS_PATH",
            "SISTEER_FROZEN_CORPUS_SHA", "SISTEER_FROZEN_CORPUS_PROMPT_COL")
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    from sisteer.eval.attacks import frozen_corpus as fc
    fc._ENV_CACHE.clear()


class _StubRewriter:
    """Deterministic offline rewriter: a benign transform that cannot collide
    with the synthetic fingerprint."""

    model = "stub-rewriter"

    def batch(self, items):
        return ["STUBOUT[" + p.strip().split("\n")[-1][:40] + "]" for (p, s) in items]


def _prepare(name, spec, tmp: Path):
    """Return the goals to evaluate this family over, doing any corpus setup
    (rewriter build kind builds a stub corpus; the generic ingested-kind entry gets a temp corpus).
    Deterministic build kind needs no setup; frozen_demo uses its committed fixture goals."""
    import os

    if spec.build_kind == "deterministic":
        return _GOALS

    if spec.build_kind == "rewriter":
        os.environ["SISTEER_HELDOUT_CORPUS_DIR"] = str(tmp)
        # rebuild module caches so the just-built corpus is picked up
        mod = sys.modules[A.HELD_OUT_ATTACKS[name][0]]
        mod._CORPUS_CACHE.clear()
        assert spec.build_fn is not None, f"rewriter-kind {name} must set build_fn"
        spec.build_fn(_GOALS, rewriter=_StubRewriter(), fingerprint=_FINGERPRINT)
        mod._CORPUS_CACHE.clear()
        return _GOALS

    # ingested build kind
    if name == "frozen_demo":
        return [Goal(g, "(covered by corpus)", "illegal", "d", "r")
                for g in ("demo_sourdough", "demo_kite", "demo_origami")]
    if name == "frozen_corpus":
        # write a temp canonical corpus over the generic goals and point the
        # generic env-configured adapter at it
        corpus = tmp / "generic_corpus.jsonl"
        rows = [{"goal_id": g.goal_id, "sample_idx": 0, "variant_name": "frozen",
                 "prompt": f"[synthetic ingested artifact] {g.text}"} for g in _GOALS]
        corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        os.environ["SISTEER_FROZEN_CORPUS_PATH"] = str(corpus)
        os.environ.pop("SISTEER_FROZEN_CORPUS_SHA", None)
        from sisteer.eval.attacks import frozen_corpus as fc
        fc._ENV_CACHE.clear()
        return _GOALS
    raise AssertionError(f"unhandled ingested-kind family {name}")


@pytest.mark.parametrize("name", sorted(A.HELD_OUT_FAMILIES))
def test_family_conformance(name, tmp_path):
    spec = A.get_heldout_spec(name)
    assert spec.name == name
    assert spec.build_kind in ("deterministic", "rewriter", "ingested")

    goals = _prepare(name, spec, tmp_path)
    report = conformance_report(spec, goals, _FINGERPRINT)
    for check in ("determinism", "variant_count", "no_leakage",
                  "manifest_complete", "content_hash_stable"):
        assert report[check] is True, f"{name} failed conformance: {check} ({report})"
    assert report["n_rows"] > 0, f"{name} materialised no prompts (corpus wiring?)"

    # manifest: servable and complete for the family's build-kind
    # contract, and the two extra axes (goal-holdout + defense-object) break out
    covered = {r["goal_id"] for r in materialize_corpus(spec.variants_fn, goals)}
    group_of = {gid: ("held_out" if i % 2 else "held_in")
                for i, gid in enumerate(sorted(covered))}
    man = write_family_manifest(spec, goals,
                                manifest_path=tmp_path / f"{name}.manifest.json",
                                fingerprint=_FINGERPRINT,
                                category_group_of=group_of,
                                object_class="si_distilled",
                                cell="safety_refusal__psr_kl_all", arm="psr_kl_all")
    assert man["family"] == name
    assert man["leakage_check"]["passed"] is True
    assert man["corpus_content_hash"]  # non-empty content hash
    assert man["row_count"] == len(materialize_corpus(spec.variants_fn, goals))
    assert sum(v["n_behaviors"] for v in man["goal_axis"].values()) == len(covered)
    assert man["object_class"] == "si_distilled"
    assert man["goal_holdout_coverage"] == "categorised_harmbench_jbb_only"


@pytest.mark.parametrize("name", sorted(A.HELD_OUT_FAMILIES))
def test_family_replay_resume_idempotence(name, tmp_path):
    """A second run adds no rows; an interrupted goal's partial rows are dropped
    and the goal cleanly redone — the static replay contract, per family."""
    spec = A.get_heldout_spec(name)
    goals = _prepare(name, spec, tmp_path)
    run = spec.run_fn

    out = tmp_path / "run"
    r1 = run(StubDefense(), goals, out_dir=out, max_new_tokens=32, seed=0)
    assert not r1.get("skipped"), f"{name} unexpectedly skipped in the built path"
    assert oct(out.stat().st_mode)[-3:] == "700", f"{name} run dir not 0700"
    rows1 = read_jsonl(out / "responses.jsonl")
    assert rows1, f"{name} wrote no responses"
    for row in rows1:
        for k in ("goal_id", "sample_idx", "trial", "variant_key",
                  "variant_name", "family", "response"):
            assert k in row, f"{name} response row missing {k!r}"
        assert row["variant_key"] == f"{row['sample_idx']}:{row['trial']}"

    # resume: a second identical run adds nothing
    r2 = run(StubDefense(), goals, out_dir=out, max_new_tokens=32, seed=0)
    assert r2["n_goals_run"] == 0
    assert len(read_jsonl(out / "responses.jsonl")) == len(rows1)

    # an interrupted goal (partial row, no terminal progress record) is dropped
    extra = Goal("hc_interrupted", "an extra invented goal", "illegal", "d", "r")
    if extra.goal_id in {g.goal_id for g in goals}:
        return  # (won't happen for these fixtures)
    with (out / "responses.jsonl").open("a") as f:
        f.write(json.dumps({"goal_id": extra.goal_id, "sample_idx": 0,
                            "response": "PARTIAL_INTERRUPTED"}) + "\n")
    run(StubDefense(), list(goals) + [extra], out_dir=out, max_new_tokens=32, seed=0)
    assert "PARTIAL_INTERRUPTED" not in (out / "responses.jsonl").read_text()
