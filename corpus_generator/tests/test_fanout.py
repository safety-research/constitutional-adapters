"""Fanout bookkeeping: deterministic custom_ids + request_set_hash across two
builds; lineage propagation A1->E; deterministic ~20% no-system selection."""

from __future__ import annotations


from conftest import make_engine
from corpus_generator.advice import advice_stages, include_system_for
from corpus_generator.batch import request_set_hash
from corpus_generator.pipeline import load_union_records, make_engine as make_cfg_engine, run_corpus
from corpus_generator.stages import run_stage
from corpus_generator.templates import repo_root


def test_builds_are_deterministic(ctx_factory, fake_client):
    ctx = ctx_factory("advice")
    specs = advice_stages()
    engine = make_engine(fake_client)
    ctx.upstream["a1"] = run_stage(engine, ctx, specs[0])
    ctx.upstream["a2"] = run_stage(engine, ctx, specs[1])
    ctx.upstream["a3"] = run_stage(engine, ctx, specs[2])

    b1 = specs[3]
    reqs1 = b1.build_requests(ctx)
    reqs2 = b1.build_requests(ctx)
    assert [r.custom_id for r in reqs1] == [r.custom_id for r in reqs2]
    h1 = request_set_hash(reqs1, {})
    h2 = request_set_hash(reqs2, {})
    assert h1 == h2
    assert all(cid.startswith("adv-b1-p0") for cid in (r.custom_id for r in reqs1))


def test_no_system_selection_deterministic_and_about_20pct():
    keys = [f"p{i:02d}-t{j:02d}-s{k:02d}" for i in range(10) for j in range(5) for k in range(4)]
    picks1 = [include_system_for(k, 5) for k in keys]
    picks2 = [include_system_for(k, 5) for k in keys]
    assert picks1 == picks2
    frac_no_system = 1 - sum(picks1) / len(picks1)
    assert 0.10 < frac_no_system < 0.30  # hash-uniform ~= 1/5


def test_no_system_property_preserved_through_pipeline(tmp_path):
    root = repo_root()
    from test_run_lifecycle import _cfg, _make_client

    cfg = _cfg("nosys-test", tmp_path, scale=5)  # 2x2x5 = 20 scenarios
    cfg.advice.no_system_mod = 2  # ~half no-system, so both kinds appear
    client = _make_client()
    run_corpus(cfg, "advice", make_cfg_engine(cfg, client, sleep=lambda s: None),
               root=root, on_event=lambda m: None)
    recs = load_union_records(cfg.run_dir(root), "advice")
    with_sys = [r for r in recs if r["system"]]
    without = [r for r in recs if not r["system"]]
    assert with_sys and without
    for r in without:
        assert r["draft_system"] is None
        assert include_system_for(r["scenario_id"], 2) is False
    for r in with_sys:
        assert include_system_for(r["scenario_id"], 2) is True


def test_lineage_propagates_a1_to_e(ctx_factory, fake_client):
    ctx = ctx_factory("advice")
    engine = make_engine(fake_client)
    for spec in advice_stages():
        rows = run_stage(engine, ctx, spec)
        ctx.upstream[spec.name] = rows
    e_rows = [r for r in ctx.upstream["e"] if r.status == "ok"]
    assert e_rows
    for r in e_rows:
        assert r.lineage["scenario_key"].startswith("p0")
        assert r.lineage["batch_id"].startswith("batch_")
