"""Review round 2, P2 — BoN coverage is accepted only after its seed policy is verified.

Every served response row persists `sampling_seed` and the named `seed_rule` (bon_seed_v1); the manifest names the
rule. Before a (goal, sample_idx, trial) is accepted as covered, `bon_seed(goal_id, sample_idx, trial)` is recomputed
and a mismatch (a foreign producer's or a corrupted row with the same key, geometry and preset) is REFUSED before any
manifest write and never counted as covered; a current-format row without a seed or a rule is unverifiable and
refused. Historical seedless rows (no decode, no seed) are accepted ONLY under the explicit legacy rule, by a
harness-default run, and counted in the manifest — absence is never read as compatibility.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sisteer.eval.attacks import bon as bon_mod
from sisteer.eval.attacks import bon_served as bs

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "tests"):
    sys.path.insert(0, str(p))

PRESET = {"temperature": 0.6, "top_p": 0.9}
#: the superseded grandfathered T=1 row BY NAME (since 2026-09-17 best-of-N has no model-independent default: a run
#: names its served model and draws at the vendor preset, or names this row to finish a store drawn under it)
HARNESS = dict(decode_label=bs.BON_HARNESS_DEFAULT)


def _world(tmp_path):
    import test_bon_served as tb
    backend = tb.RecordingBackend()
    proxy, routes = tb._proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(proxy, virtual_model="petri__base", cell_id="none")
    return tb, backend, routes, defense


def _rows(out):
    return [json.loads(l) for l in (out / "bon_responses.jsonl").read_text().splitlines() if l.strip()]


def _write(out, rows):
    (out / "bon_responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_every_row_persists_its_seed_and_the_rule_and_the_manifest_names_the_policy(tmp_path):
    tb, backend, routes, defense = _world(tmp_path)
    goals = tb._goals(2)
    out = tmp_path / "store"
    bs.bon_generate_served(defense, goals, out_dir=out, routes=routes, n_max=3, max_concurrency=4,
                           decode=PRESET, decode_label="vendor")
    rows = _rows(out)
    assert len(rows) == 6
    for r in rows:
        assert r["seed_rule"] == bon_mod.BON_SEED_RULE == "bon_seed_v1"
        assert r["sampling_seed"] == bon_mod.bon_seed(r["goal_id"], r["sample_idx"], r["trial"])
    man = json.loads((out / bs.BON_SERVED_MANIFEST).read_text())
    assert man["seed_rule"] == "bon_seed_v1" and man["n_legacy_seedless_rows"] == 0 and man["legacy_rows_rule"] is None
    # the same run resumes idempotently over its own validated rows
    n = backend.n_calls
    bs.bon_generate_served(defense, goals, out_dir=out, routes=routes, n_max=3, max_concurrency=4,
                           decode=PRESET, decode_label="vendor")
    assert backend.n_calls == n


def test_a_row_with_another_seed_under_the_same_key_geometry_and_preset_is_refused_before_any_write(tmp_path):
    tb, backend, routes, defense = _world(tmp_path)
    goals = tb._goals(2)
    out = tmp_path / "store"
    kw = dict(out_dir=out, routes=routes, n_max=2, max_concurrency=4, decode=PRESET, decode_label="vendor")
    bs.bon_generate_served(defense, goals, **kw)
    rows = _rows(out)
    manifest_before = (out / bs.BON_SERVED_MANIFEST).read_text()
    # a foreign producer's (or corrupted) row: identical (goal, sample_idx, trial), decode, preset, geometry — another seed
    rows[1]["sampling_seed"] = rows[1]["sampling_seed"] + 1
    _write(out, rows)
    n = backend.n_calls
    with pytest.raises(bs.BonServedError, match="records sampling_seed .* but bon_seed_v1 assigns") as ei:
        bs.bon_generate_served(defense, goals, **kw)
    assert "different sample identity" in str(ei.value)
    assert backend.n_calls == n                                             # no replacement request was made
    assert (out / bs.BON_SERVED_MANIFEST).read_text() == manifest_before    # the manifest was not rewritten
    assert _rows(out) == rows                                               # the store is untouched
    # and coverage is never accepted for it either: the validated coverage set refuses the whole store
    with pytest.raises(bs.BonServedError, match="different sample identity"):
        bs._load_covered_validated(out / "bon_responses.jsonl", PRESET, "vendor")


def test_a_current_format_row_without_a_seed_or_a_rule_is_unverifiable_and_refused(tmp_path):
    tb, backend, routes, defense = _world(tmp_path)
    goals = tb._goals(1)
    out = tmp_path / "store"
    kw = dict(out_dir=out, routes=routes, n_max=2, max_concurrency=2, decode=PRESET, decode_label="vendor")
    bs.bon_generate_served(defense, goals, **kw)
    rows = _rows(out)
    for drop in ("sampling_seed", "seed_rule"):
        broken = [dict(r) for r in rows]
        broken[0].pop(drop)
        _write(out, broken)
        with pytest.raises(bs.BonServedError, match="UNVERIFIABLE"):
            bs.bon_generate_served(defense, goals, **kw)
    # a row naming ANOTHER seed rule is another producer's sample identity
    other = [dict(r) for r in rows]
    other[0]["seed_rule"] = "bon_seed_v0"
    _write(out, other)
    with pytest.raises(bs.BonServedError, match="names the seed rule 'bon_seed_v0'"):
        bs.bon_generate_served(defense, goals, **kw)
    # a seed without a decode block (torn / foreign) is unverifiable too
    torn = [dict(r) for r in rows]
    torn[0].pop("decode"); torn[0].pop("bon_decode_preset")
    _write(out, torn)
    with pytest.raises(bs.BonServedError, match="UNVERIFIABLE"):
        bs.bon_generate_served(defense, goals, **kw)


def test_legacy_seedless_rows_are_accepted_only_under_the_explicit_rule_by_a_harness_default_run(tmp_path):
    tb, backend, routes, defense = _world(tmp_path)
    goals = tb._goals(1)
    out = tmp_path / "legacy"
    kw = dict(out_dir=out, routes=routes, n_max=2, max_concurrency=2)
    bs.bon_generate_served(defense, goals, **HARNESS, **kw)            # the grandfathered row, by name
    rows = _rows(out)
    for r in rows:                                                     # a pre-2026-09-10 store: no sampler, no seed, no rule
        for k in ("decode", "bon_decode_preset", "sampling_seed", "seed_rule"):
            r.pop(k)
    _write(out, rows)
    with pytest.raises(bs.BonServedError, match="LEGACY"):             # a preset never relabels them
        bs.bon_generate_served(defense, goals, decode=PRESET, decode_label="vendor", **kw)
    n = backend.n_calls
    bs.bon_generate_served(defense, goals, **HARNESS, **kw)            # the harness default resumes under the named rule
    assert backend.n_calls == n
    man = json.loads((out / bs.BON_SERVED_MANIFEST).read_text())
    assert man["legacy_rows_rule"] == bon_mod.LEGACY_SEEDLESS_RULE == "legacy_seedless_pre_2026-09-10"
    assert man["n_legacy_seedless_rows"] == 2 == man["n_legacy_unrecorded_rows"] and man["seed_rule"] == "bon_seed_v1"
    assert bs._load_covered_validated(out / "bon_responses.jsonl", None, bs.BON_HARNESS_DEFAULT) == {("g0", 0, 0), ("g0", 1, 0)}


def test_the_loop_reads_the_validated_coverage_not_the_raw_keys():
    src = (REPO / "src/sisteer/eval/attacks/bon_served.py").read_text()
    body = src[src.index("async def _bon_generate_served_async"):src.index("def bon_generate_served(")]
    # the loop reads the VALIDATED coverage for THIS run's resolved sampler (a BonSampler since 2026-09-17), never raw keys;
    # qwen-base-seed-replicates: the validated loader ALSO takes the decode-seed policy + base (None = bon_seed_v1, the
    # path every row before the policy existed took); the row's rule is the policy's (bon_seed_v1 under the default)
    assert "covered = _load_covered_validated(jsonl, sampler=sampler, decode_seed_policy=policy, decode_seed_base=seed_base)" in body
    assert "covered = _load_covered(jsonl)" not in body
    assert '"sampling_seed": int(seed)' in body and '"seed_rule": seed_rule,' in body
    assert "seed_rule = seed_rule_for_policy(policy)" in body and bon_mod.seed_rule_for_policy(None) == bon_mod.BON_SEED_RULE == "bon_seed_v1"
