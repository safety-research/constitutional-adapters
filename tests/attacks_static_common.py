"""Shared stubs + contract checks for the STATIC jailbreak-family tests
(PastTense, PAP, Encoding, AIM, GCG-replay, AmpleGCG-plus).

NOT a test module (no ``test_`` prefix) — imported by the per-family test files.
Everything here is CPU-only and model-free: the ``defense`` is a stub, so no
family test ever loads a model or hits the network.
"""

import json
import sys
import tempfile
from pathlib import Path

from sisteer.eval.jailbreak_data import Goal

REPO = Path(__file__).resolve().parents[1]


class StubDefense:
    """The ``defense`` surface the attack contract needs — ``.generate``,
    ``.model``, ``.tok``, ``.n_target_forwards`` — with a canned refusal reply
    and a boundary-accurate forward counter. Touches no model, no network."""

    def __init__(self, text="I'm sorry, I can't help with that."):
        self.n_target_forwards = 0
        self.model = object()
        self.tok = object()
        self._text = text
        self.seen: list[str] = []

    def generate(self, prompts, **kw):
        self.n_target_forwards += len(prompts)
        self.seen.extend(prompts)
        return [self._text for _ in prompts]


def goals(n=2):
    return [Goal(f"g{i}", f"benign goal {i}", "illegal", "harmbench", "rev")
            for i in range(n)]


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def pool_builder_variant_key(record):
    """The variant_key the REAL two-SI pool builder derives for ``record`` —
    proves the family keys correctly under (family, goal_id, variant_key)."""
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from build_two_si_pool import _variant_key
    return _variant_key(record)


def assert_static_contract(run_fn, family, *, expected_variant_names=None,
                           n_goals=2, run_kwargs=None):
    """Exercise the shared static-family contract against a stub defense:

    (e) responses.jsonl rows carry the expected fields incl. a ``variant_key``
        that equals ``f"{sample_idx}:{trial}"`` AND what the pool builder derives;
        progress.jsonl has one terminal record per goal (``capped`` False);
    (f) the run dir is mode-0700;
    (d) a second run adds no rows (resume) and an interrupted goal's partial row
        is dropped and the goal cleanly redone.

    Returns ``(rows, progress)`` from the first run for family-specific asserts.
    """
    run_kwargs = run_kwargs or {}
    gs = goals(n_goals)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"

        r1 = run_fn(StubDefense(), gs, out_dir=out, **run_kwargs)
        assert not r1.get("skipped"), "unexpected skip in the available path"
        # (f) private dir mode
        assert oct(out.stat().st_mode)[-3:] == "700", "run dir is not 0700"

        rows = read_jsonl(out / "responses.jsonl")
        prog = read_jsonl(out / "progress.jsonl")
        assert rows, "no responses.jsonl rows written"

        # (e) response fields + variant_key convention
        for row in rows:
            for k in ("goal_id", "sample_idx", "trial", "variant_key",
                      "variant_name", "family", "variant", "attack_prompt",
                      "response"):
                assert k in row, f"response row missing field {k!r}"
            assert row["family"] == family
            assert row["variant_key"] == f"{row['sample_idx']}:{row['trial']}"
            assert pool_builder_variant_key(row) == row["variant_key"], \
                "record does not key the way build_two_si_pool._variant_key does"

        # deterministic ordered sub-variant names, matched across goals
        by_goal: dict[str, list[str]] = {}
        for row in rows:
            by_goal.setdefault(row["goal_id"], []).append(row["variant_name"])
        if expected_variant_names is not None:
            for names in by_goal.values():
                assert names == list(expected_variant_names), \
                    f"variant order {names} != {list(expected_variant_names)}"

        # progress: one terminal record per goal, forward_cap-exempt
        assert {p["goal_id"] for p in prog} == {g.goal_id for g in gs}
        for p in prog:
            assert p["family"] == family
            assert p["capped"] is False

        # (d) resume — a second identical run adds nothing
        n_before = len(rows)
        r2 = run_fn(StubDefense(), gs, out_dir=out, **run_kwargs)
        assert r2["n_goals_run"] == 0
        assert len(read_jsonl(out / "responses.jsonl")) == n_before

        # (d) an interrupted goal (partial row, no progress record) is dropped
        extra = Goal(f"g{n_goals}", "benign goal extra", "illegal",
                     "harmbench", "rev")
        with (out / "responses.jsonl").open("a") as f:
            f.write(json.dumps({"goal_id": extra.goal_id, "sample_idx": 0,
                                "response": "PARTIAL_INTERRUPTED"}) + "\n")
        r3 = run_fn(StubDefense(), gs + [extra], out_dir=out, **run_kwargs)
        assert r3["n_goals_run"] == 1
        assert "PARTIAL_INTERRUPTED" not in (out / "responses.jsonl").read_text()

        return rows, prog
