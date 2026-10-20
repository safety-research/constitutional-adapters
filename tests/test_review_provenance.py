"""The adversarial review's four findings on geom-sweep (2026-09-04), pinned:

R1  a resumed families run must not relabel old generations: the run-identity
    sidecar is written on a fresh out dir, VERIFIED on a reused one, and any
    mismatch (or a pre-sidecar dir holding generations) REFUSES the resume; the
    geometry stamp rides on every served row and a mixed store is refused.
R2  the sweep's resume is keyed on the measurement identity: a fingerprint is
    persisted before the first probe; foreign / pre-fingerprint probes and rows
    are refused; every probe carries its own job / node; paste_record carries
    the ORIGINATING probe's; sweep points complete on (L, arm, B, actual S).
R3  a truncated ladder is not a complete measurement: incomplete groups carry
    no definitive knee / speedup (provisional only), two S at one (L, arm) are
    two groups, missing arms are named; the analyzer rebuilds from rows.
R4  the Modal launchers pass --eval / --model-key with the dimensions, and the
    standup's own resolution of that argv EQUALS the outer one (as_dict).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.serving import run_identity as ri

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_decode_sweep as sw  # noqa: E402
from test_sweep_window_retry import _budget_json, _ctx, sweep_env  # noqa: E402,F401


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _identity(**over):
    base = dict(model="org/model", revision="a" * 40, tree_sha="b" * 40, model_key="llama8b",
                max_model_len=11520, max_num_seqs=39, client_concurrency=39, budget=458_752,
                budget_source="measured", serving_versions={"vllm": "0.25.1"},
                sampling={"max_new_tokens": 512}, input_fingerprints={"goal_ids_sha16": "c" * 16})
    base.update(over)
    return ri.RunIdentity(**base)


# --------------------------------------------------------------------------- #
# R1                                                                           #
# --------------------------------------------------------------------------- #


def test_sidecar_is_written_once_verified_and_refuses_a_mismatch(tmp_path):
    out = tmp_path / "run"
    ident = _identity()
    p = ri.write_or_verify(out, ident)
    assert p == out / ri.RUN_IDENTITY_FILE and p.is_file()
    rec = json.loads(p.read_text())
    assert rec["geometry_id"] == ident.geometry_id and rec["run_id"] == ident.run_id
    assert rec["max_num_seqs"] == 39 and rec["budget"] == 458_752 and "written_at" in rec
    # the same identity resumes (the file is untouched)
    before = p.read_text()
    assert ri.write_or_verify(out, _identity()) == p and p.read_text() == before
    # a different S (the per-model budget widened it) REFUSES, naming the field
    with pytest.raises(ri.RunIdentityMismatch) as ei:
        ri.write_or_verify(out, _identity(max_num_seqs=61, client_concurrency=61))
    msg = str(ei.value)
    assert "max_num_seqs: recorded 39 != current 61" in msg and "fresh out dir" in msg
    # so do a new tree, a new stack, new sampling, new inputs, another model key
    for over in (dict(tree_sha="d" * 40), dict(serving_versions={"vllm": "0.26.0"}),
                 dict(sampling={"max_new_tokens": 1024}),
                 dict(input_fingerprints={"goal_ids_sha16": "e" * 16}), dict(model_key="llama3b"),
                 dict(budget=393_216, budget_source="default_llama_certified")):
        with pytest.raises(ri.RunIdentityMismatch):
            ri.write_or_verify(out, _identity(**over))
    # geometry_id is the serving-relevant subset: inputs / tree do not move it, S does
    assert _identity(tree_sha="d" * 40).geometry_id == ident.geometry_id
    assert _identity(input_fingerprints={}).geometry_id == ident.geometry_id
    assert _identity(max_num_seqs=61).geometry_id != ident.geometry_id
    assert _identity(serving_versions={"vllm": "0.26.0"}).geometry_id != ident.geometry_id


def test_pre_sidecar_dir_with_generations_is_refused(tmp_path):
    out = tmp_path / "old_run"
    (out / "pasttense").mkdir(parents=True)
    (out / "pasttense" / "responses.jsonl").write_text('{"goal_id": "g1", "engine": "vllm_served"}\n')
    assert ri.has_prior_generations(out) == ["pasttense/responses.jsonl"]
    with pytest.raises(ri.RunIdentityMismatch) as ei:
        ri.write_or_verify(out, _identity())
    assert "pre-sidecar run" in str(ei.value) and not (out / ri.RUN_IDENTITY_FILE).exists()
    # an empty pre-existing dir is fine (nothing to relabel)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "conditions.tsv").write_text("x")
    assert ri.write_or_verify(empty, _identity()).is_file()
    # a zero-byte responses file is not a generation
    z = tmp_path / "zero"
    (z / "fam").mkdir(parents=True)
    (z / "fam" / "bon_responses.jsonl").write_text("")
    assert ri.has_prior_generations(z) == []


def test_geometry_stamp_rides_on_served_rows_and_mixed_stores_are_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(ri.GEOMETRY_ID_ENV, raising=False)
    assert ri.geometry_stamp() == {} and ri.current_geometry_id() is None
    assert ri.geometry_stamp_problem({"engine": "vllm_served"}) is None       # no stamp in force
    ident = _identity()
    gid = ri.stamp_env(ident)
    assert ri.current_geometry_id() == gid == ident.geometry_id
    stamp = ri.geometry_stamp()
    assert stamp == {"geometry_id": gid, "run_id": ident.run_id}
    assert ri.geometry_stamp_problem(dict(stamp)) is None
    assert "pre-sidecar row" in ri.geometry_stamp_problem({"engine": "vllm_served"})
    assert "!= this run's" in ri.geometry_stamp_problem({"geometry_id": "0" * 16})
    # the static store check refuses a row of another geometry (and a pre-stamp row)
    from sisteer.eval.attacks import static_common as sc
    from sisteer.eval.attacks import bon_served as bs
    p = tmp_path / "responses.jsonl"
    p.write_text(json.dumps({"engine": bs.SERVED_ENGINE, **stamp}) + "\n")
    sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE)                  # same run: fine
    p.write_text(json.dumps({"engine": bs.SERVED_ENGINE, "geometry_id": "0" * 16}) + "\n")
    with pytest.raises(sc.StaticStoreIncompleteError) as ei:
        sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE)
    assert "serving geometry" in str(ei.value)
    p.write_text(json.dumps({"engine": bs.SERVED_ENGINE}) + "\n")             # pre-stamp row
    with pytest.raises(sc.StaticStoreIncompleteError):
        sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE)
    # the BoN store check does the same
    b = tmp_path / "bon_responses.jsonl"
    b.write_text(json.dumps({"engine": bs.SERVED_ENGINE, **stamp}) + "\n")
    bs._assert_engine_consistent(b)
    b.write_text(json.dumps({"engine": bs.SERVED_ENGINE, "geometry_id": "0" * 16}) + "\n")
    with pytest.raises(bs.BonServedError):
        bs._assert_engine_consistent(b)
    # the row writers merge the stamp
    assert bs._geometry_stamp() == stamp
    src = (REPO / "src/sisteer/eval/attacks/bon_served.py").read_text()
    assert src.count("**_geometry_stamp()") == 3                               # rows, error rows, manifest
    src = (REPO / "src/sisteer/eval/attacks/static_common.py").read_text()
    assert "row.update(geometry_stamp())" in src
    # the families drivers write/verify BEFORE any generation and export the stamp
    for rel in ("scripts/petri_families_prompted.py", "scripts/petri_families_served_integration.py"):
        s = (REPO / rel).read_text()
        assert "ri.write_or_verify(out, identity)" in s and "ri.stamp_env(identity)" in s
        assert s.index("ri.write_or_verify(out, identity)") < s.index("launch_server(")
        assert 'rec["run_identity"] = identity.as_dict()' in s
        assert "RunIdentityMismatch" in s


# --------------------------------------------------------------------------- #
# R2                                                                           #
# --------------------------------------------------------------------------- #


def _fake_probe_pass(ctx, *, max_model_len, max_num_seqs, expect_object=None):
    # E54 #5: the real probe_geometry takes the phase fingerprint's
    # ``steered_object`` and re-attests it against the bytes the standup loaded,
    # and run_budget has passed it since. A stub that merely SWALLOWED the kwarg
    # would let the walk hand the attestation anything at all, so assert the walk
    # forwards the fingerprint's own value — None here: these contexts are
    # passthrough-only (``steered_cell=None``).
    assert expect_object == sw.object_identity(ctx.objects_dir, ctx.steered_cell)
    return {"product": max_num_seqs * max_model_len, "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len, "tag": f"budget_L{max_model_len}xS{max_num_seqs}",
            "status": "pass" if max_num_seqs <= 64 else "kv_under_coverage",
            "kv_tokens": max_num_seqs * max_model_len + 1000, "kv_kind": "uniform_blocks_x_block_size"}


def test_budget_walk_persists_a_fingerprint_and_refuses_foreign_probes(monkeypatch, tmp_path):
    ctx = _ctx(tmp_path, hybrid=False)
    monkeypatch.setattr(sw, "probe_geometry", _fake_probe_pass)
    monkeypatch.setattr(sw, "_nvidia_smi", lambda *a: "H200")
    monkeypatch.setattr(sw, "wait_gpu_free", lambda *a, **k: None)
    doc = sw.run_budget(ctx, budget_len=8192, ladder=sw.BUDGET_LADDER)
    fp, mid = doc["fingerprint"], doc["measurement_id"]
    assert mid == sw.fingerprint_id(fp) and fp["phase"] == "budget" and fp["budget_len"] == 8192
    for key in ("hf", "rev", "config_sha16", "manifest_sha16", "repo_sha", "requirements_file",
                "versions", "gpu_name", "gpu_mem_util", "hybrid", "seqs_cap", "ladder"):
        assert key in fp, key
    assert fp["config_sha16"] and fp["manifest_sha16"]
    # every probe carries its own provenance; the largest passing probe's job/node
    # (not the doc's) is what the paste record cites
    for p in doc["probes"]:
        assert p["job_id"] == "373464" and p["node"] == "TESTHOST" and p["measurement_id"] == mid
    assert doc["largest_passing"]["max_num_seqs"] == 64          # 48, 64 pass; 80 fails; bisect 72 fails
    assert doc["paste"]["basis"]["probe_job_id"] == "373464"
    # a paste from a probe of an EARLIER job cites that job / node
    br = {"largest_passing": {"product": 56 * 8192, "max_num_seqs": 56, "kv_tokens": 1,
                              "job_id": "111", "node": "node-old"},
          "first_refused": 64 * 8192, "n_probes": 3, "wall_reached": True}
    rec = sw.paste_record(model_key="llama8b", bracket=br, max_model_len=8192, gpu_mem_util=0.9,
                          job_id="222", measured_on="d", results_dir="/r", node="node-new")
    assert rec["basis"]["probe_job_id"] == "111" and "(node node-old)" in rec["basis"]["source"]
    # the SAME fingerprint resumes (no live probe: the bracket is closed) ...
    calls = []
    monkeypatch.setattr(sw, "probe_geometry",
                        lambda ctx_, **kw: calls.append(kw) or _fake_probe_pass(ctx_, **kw))
    doc2 = sw.run_budget(ctx, budget_len=8192, ladder=sw.BUDGET_LADDER)
    assert calls == [] and doc2["measurement_id"] == mid and len(doc2["probes"]) == 4   # 48 64 80 72
    # ... a different one REFUSES the recorded probes (here: another gpu_mem_util)
    ctx2 = sw.SweepContext(**{**ctx.__dict__, "gpu_mem_util": "0.85"})
    with pytest.raises(SystemExit) as ei:
        sw.run_budget(ctx2, budget_len=8192, ladder=sw.BUDGET_LADDER)
    assert "DIFFERENT measurement" in str(ei.value) and "gpu_mem_util" in str(ei.value)
    # ... and so does a pre-fingerprint budget.json (probes, no measurement_id)
    stale = json.loads((ctx.results_dir / "budget.json").read_text())
    stale.pop("measurement_id"); stale.pop("fingerprint")
    (ctx.results_dir / "budget.json").write_text(json.dumps(stale))
    with pytest.raises(SystemExit) as ei:
        sw.run_budget(ctx, budget_len=8192, ladder=sw.BUDGET_LADDER)
    assert "pre-fingerprint" in str(ei.value)


def test_sweep_resume_is_keyed_on_the_measurement_and_the_actual_S(sweep_env, tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, hybrid=True)
    _budget_json(ctx)
    monkeypatch.setattr(sw, "_nvidia_smi", lambda *a: "H200")
    # first run: 1536 refused at 256, measured at 192 (9 points)
    sweep_env.policy["refuse"] = lambda L, S: (L == 1536 and S > 200)
    summ = sw.run_sweep(ctx, windows=(1536,), budget=None, prompt_tokens=1000,
                        max_tokens=512, warmup_s=0.0, measure_s=1.0)
    mid = summ["measurement_id"]
    assert mid == sw.fingerprint_id(summ["fingerprint"]) and summ["fingerprint"]["phase"] == "sweep"
    rows = [json.loads(l) for l in (ctx.results_dir / "sweep.jsonl").read_text().splitlines()]
    assert all(r["measurement_id"] == mid for r in rows)          # points AND the refused row
    assert summ["groups"]["1536:passthrough"]["S"] == 192 and summ["groups"]["1536:passthrough"]["complete"]
    # a second run under the same fingerprint: the 192-S points are NOT the 256-S target's
    # points, so the window stands up again (at 256 -> now accepted) and measures the
    # missing 256-S rung set; the two S become two groups, never one curve
    sweep_env.policy["refuse"] = lambda L, S: False
    sweep_env.calls.clear()
    summ2 = sw.run_sweep(ctx, windows=(1536,), budget=None, prompt_tokens=1000,
                         max_tokens=512, warmup_s=0.0, measure_s=1.0)
    assert [(L, S) for L, S, _, _ in sweep_env.calls] == [(1536, 256)]
    assert set(summ2["groups"]) == {"1536:passthrough@S192", "1536:passthrough@S256"}
    assert summ2["groups"]["1536:passthrough@S256"]["n_points"] == 10       # ladder 8..256
    assert all(r["S"] in (192, 256) for r in rows + [json.loads(l) for l in
               (ctx.results_dir / "sweep.jsonl").read_text().splitlines()] if sw.is_point_row(r))
    # a third run with a different measure_s: the rows are of ANOTHER measurement -> refused
    with pytest.raises(SystemExit) as ei:
        sw.run_sweep(ctx, windows=(1536,), budget=None, prompt_tokens=1000,
                     max_tokens=512, warmup_s=0.0, measure_s=2.0)
    assert "DIFFERENT measurement" in str(ei.value) and "measure_s" in str(ei.value)
    assert sw.point_key({"window_L": 1536, "arm": "passthrough", "B": 8, "S": 192}) == (1536, "passthrough", 8, 192)


# --------------------------------------------------------------------------- #
# R3                                                                           #
# --------------------------------------------------------------------------- #


def _pt(L, arm, S, B, tok):
    return {"kind": "point", "window_L": L, "window_labels": list(sw.window_labels(L)), "S": S,
            "arm": arm, "B": B, "tok_s": tok, "sm_mean": 50.0, "lat_p50_s": 1.0, "preemptions_delta": 0}


def test_incomplete_groups_get_no_definitive_knee_and_missing_arms_are_named():
    # a full ladder at S=24 (8,16,24) is complete; two rungs of a 12-rung ladder are not
    rows = [_pt(16384, "passthrough", 24, b, 100.0 * b) for b in (8, 16, 24)]
    rows += [_pt(1536, "passthrough", 256, 8, 400.0), _pt(1536, "passthrough", 256, 16, 546.0)]
    rows += [_pt(1536, "steered", 256, 8, 91.0), _pt(1536, "steered", 256, 16, 91.0)]
    summ = sw.summarize_rows(rows, windows_requested=sw.DEFAULT_WINDOWS)
    full = summ["groups"]["16384:passthrough"]
    assert full["complete"] and full["knee_B"] == 24 and full["tok_s_at_knee"] == 2400.0
    assert full["missing_B"] == [] and full["n_expected"] == 3
    part = summ["groups"]["1536:passthrough"]
    assert part["complete"] is False and part["missing_B"] == [24, 32, 48, 64, 96, 128, 192, 256]
    assert part["knee_B"] is None and part["tok_s_at_knee"] is None
    assert part["speedup_knee_vs_B24"] is None and part["duty_at_knee"] is None
    assert part["knee_provisional_B"] == 16 and part["tok_s_at_knee_provisional"] == 546.0
    assert part["n_points"] == 2 and part["n_expected"] == 10
    assert summ["groups"]["1536:steered"]["knee_provisional_B"] == 8
    assert sorted(summ["groups_incomplete"]) == ["1536:passthrough", "1536:steered"]
    # the steered arm exists in the run, so the 16384 window is missing it
    assert summ["arms_missing"] == {"16384": ["steered"]}
    assert summ["expected_arms"] == ["passthrough", "steered"]
    assert summ["partial"] is True and summ["complete"] is False
    assert summ["windows_not_measured"] == [8704, 7424, 11520]
    # two S at one (L, arm) are two groups
    rows2 = [_pt(1536, "passthrough", 192, b, 10.0 * b) for b in sw.batch_ladder(192)]
    rows2 += [_pt(1536, "passthrough", 256, b, 10.0 * b) for b in sw.batch_ladder(256)]
    summ2 = sw.summarize_rows(rows2)
    assert set(summ2["groups"]) == {"1536:passthrough@S192", "1536:passthrough@S256"}
    assert all(g["complete"] for g in summ2["groups"].values()) and summ2["complete"]
    # a passthrough-only run expects no steered arm
    summ3 = sw.summarize_rows(rows2, expected_arms=["passthrough"])
    assert summ3["arms_missing"] == {}


def test_analyzer_rebuilds_from_rows_and_flags_the_truncated_window(tmp_path):
    an = _load("_geom_sweep_analyze_r3", "scripts/geom_sweep_analyze.py")
    root = tmp_path / "geom-sweep"
    d = root / "qwen35_9b"
    d.mkdir(parents=True)
    rows = [_pt(7424, "passthrough", 79, b, 100.0 * b ** 0.5) for b in sw.batch_ladder(79)]
    rows += [_pt(7424, "steered", 79, b, 90.0) for b in sw.batch_ladder(79)]
    rows += [_pt(1536, "passthrough", 256, 8, 400.0), _pt(1536, "passthrough", 256, 16, 546.0)]
    rows += [_pt(1536, "steered", 256, 8, 91.0), _pt(1536, "steered", 256, 16, 91.0)]
    (d / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # a STORED summary from the old code, marking the truncated window complete with a knee
    (d / "sweep_summary.json").write_text(json.dumps({
        "groups": {"1536:passthrough": {"window_L": 1536, "arm": "passthrough", "S": 256,
                                        "knee_B": 16, "table": []}},
        "old_S": 24, "windows_requested": [7424, 1536], "steered_cell": "si_constitution__psr_kl_all",
        "model_key": "qwen35_9b", "transformers_version": "5.15.0", "partial": False}))
    (d / "budget.json").write_text(json.dumps({
        "model_key": "qwen35_9b", "max_model_len": 8192, "job_id": "65727", "node": "gpu-6",
        "measured_on": "2026-09-04", "complete": True, "first_refused": 655_360, "gpu_mem_util": "0.90",
        "largest_passing": {"product": 589_824, "max_num_seqs": 72, "kv_tokens": 720_033,
                            "kv_kind": "hybrid_max_concurrency_x_L", "job_id": "65700", "node": "gpu-9"},
        "probes": [], "paste": {"model_key": "qwen35_9b", "batched_token_budget": 589_824,
                                "certified": False,
                                "basis": {"method": "m", "max_model_len": 8192, "max_num_seqs": 72,
                                          "gpu_memory_utilization": 0.9, "kv_tokens_reported": 720_033,
                                          "probe_job_id": "65727", "measured_on": "2026-09-04",
                                          "source": "/workspace/someone/exp/geom-sweep/qwen35_9b/budget.json (node gpu-6)",
                                          "first_refused": 655_360, "note": "n"}}}))
    res = an.load_results(root)
    s = res["qwen35_9b"]["summary"]
    assert s["rebuilt_from_rows"] and s["stored_summary_present"]
    assert s["steered_cell"] == "si_constitution__psr_kl_all" and s["transformers_version"] == "5.15.0"
    g = s["groups"]["1536:passthrough"]
    assert g["complete"] is False and g["knee_B"] is None and g["knee_provisional_B"] == 16
    assert s["groups"]["7424:passthrough"]["complete"] and s["groups"]["7424:passthrough"]["knee_B"]
    assert sorted(s["groups_incomplete"]) == ["1536:passthrough", "1536:steered"] and s["partial"]
    assert s["windows_not_measured"] == []
    # the paste cites the ORIGINATING probe's job / node, scrubbed
    b = res["qwen35_9b"]["budget"]
    assert b["paste"]["basis"]["probe_job_id"] == "65700"
    assert b["paste"]["basis"]["source"] == "<SHARED>/exp/geom-sweep/qwen35_9b/budget.json (node gpu-9)"
    md = an.markdown_tables(res)
    assert "(prov.) 16" in md and "2/10 INCOMPLETE" in md
    assert "INCOMPLETE: 1536:passthrough (S=256) has 2 of 10 rungs" in md
    assert "| 7424 | capability,families_prompted,bon | passthrough | 79 | 52 | " in md
    doc = an.write_doc(res, tmp_path / "doc.md")
    txt = doc.read_text()
    assert "### qwen35_9b  (**incomplete groups**: 1536:passthrough (missing B=" in txt
    assert "| 1536 | capability:short | passthrough | 256 | (prov.) 16 | - | - | - | - | 2/10 INCOMPLETE |" in txt
    assert "/home/" not in txt and "/workspace" not in txt and "someone" not in txt


# --------------------------------------------------------------------------- #
# R4                                                                           #
# --------------------------------------------------------------------------- #


def _modal_stub():
    from test_grokmt_served_resume import _stub_modal
    sys.modules.setdefault("modal", _stub_modal())


def _inner(up, argv):
    """Parse a launcher's standup argv EXACTLY as the standup does and resolve."""
    ns = up.build_parser().parse_args(argv[2:])
    return up.resolve_from_args(ns)


def test_modal_launchers_hand_the_standup_the_same_geometry_they_resolved(tmp_path):
    _modal_stub()
    up = _load("_standup_e2e", "scripts/capctrl_serving_up.py")
    capctrl = _load("_capctrl_e2e", "scripts/modal/capctrl_served.py")
    grokmt = _load("_grokmt_e2e", "scripts/modal/grokmt_served.py")
    # capctrl: every registered model's cell spec
    for spec in capctrl.cell_specs():
        outer = capctrl.cell_geometry(spec)
        argv = capctrl._standup_cmd("/tree", "cfg.yaml", spec["cell"], "ddir", tmp_path,
                                    tmp_path / "STOP", outer, 64)
        assert argv[argv.index("--eval") + 1] == "capability"
        assert argv[argv.index("--model-key") + 1] == spec["model"]
        assert "--allow-over-budget" not in argv
        inner = _inner(up, argv)
        assert inner.as_dict() == outer.as_dict(), spec["model"]
        assert inner.seqs_source == "model_budget"
        # the model's OWN row either way — measured, or the 2026-09-06 provisional
        # ceiling on llama8b — and the label travels with the geometry
        assert inner.budget_source in sg.PER_MODEL_BUDGET_SOURCES
        assert inner.budget_source == outer.budget_source
        assert inner.eval == "capability" and inner.model_key == spec["model"]
    # a row override that keeps the budget but breaks the identity travels with the opt-in
    outer = capctrl.cell_geometry({"model": "glm46vflash", "seqs": 100, "allow_over_budget": True})
    argv = capctrl._standup_cmd("/tree", "c", "cell", "d", tmp_path, tmp_path / "STOP", outer, 64)
    assert "--allow-over-budget" in argv
    inner = _inner(up, argv)
    assert inner.as_dict() == outer.as_dict()
    assert inner.seqs_source == "override" and inner.window_source == "table"
    # a bare registry row (no model): the Llama default, identically on both sides
    outer = capctrl.cell_geometry(capctrl.MODELS["llama8b"])
    argv = capctrl._standup_cmd("/tree", "c", "cell", "d", tmp_path, tmp_path / "STOP", outer, 64)
    assert "--model-key" not in argv and _inner(up, argv).as_dict() == outer.as_dict()
    # S3: EQUAL-VALUED explicit overrides keep their "override" source on both sides —
    # equal window, equal S, equal both; a dense and a hybrid model
    # (qwen35_4b's derived S at 7424 is 99 under the 2026-09-13 row 741,888 (97 under the swept 720,896) — it was 48 under
    # the 2026-09-06 provisional ceiling 360,448, lifted 2026-09-11)
    for model, window, seqs in (("glm46vflash", 7424, None), ("glm46vflash", None, 105),
                                ("glm46vflash", 7424, 105), ("qwen35_4b", 7424, None),
                                ("qwen35_4b", None, 99), ("qwen35_4b", 7424, 99)):
        spec = {"model": model}
        if window is not None:
            spec["window"] = window
        if seqs is not None:
            spec["seqs"] = seqs
        outer = capctrl.cell_geometry(spec)
        derived = capctrl.cell_geometry({"model": model})
        assert (outer.max_model_len, outer.max_num_seqs) == (derived.max_model_len, derived.max_num_seqs)
        assert outer.window_source == ("override" if window is not None else "table")
        assert outer.seqs_source == ("override" if seqs is not None else "model_budget")
        argv = capctrl._standup_cmd("/tree", "c", "cell", "d", tmp_path, tmp_path / "STOP", outer, 64)
        assert ("--max-model-len" in argv) == (window is not None)
        assert ("--max-num-seqs" in argv) == (seqs is not None)
        inner = _inner(up, argv)
        assert inner.as_dict() == outer.as_dict(), (model, window, seqs)
        assert (inner.window_source, inner.seqs_source) == (outer.window_source, outer.seqs_source)
    # ... and through grokmt
    for kw in ({"target_window": 8704}, {"target_seqs": 67}, {"target_window": 8704, "target_seqs": 67}):
        outer = grokmt.resolve_target_geometry(None, **kw)
        inner = _inner(up, grokmt._target_standup_cmd("/tree", tmp_path, tmp_path / "STOP", outer))
        assert inner.as_dict() == outer.as_dict(), kw
        assert inner.window_source == ("override" if "target_window" in kw else "table")
        assert inner.seqs_source == ("override" if "target_seqs" in kw else "model_budget")
    # the standup never folds a value back to "derived"
    src = (REPO / "scripts/capctrl_serving_up.py").read_text()
    assert "PER-FACTOR COLLAPSE" not in src and "== pure.max_model_len" not in src
    # grokmt: the default target, a seqs override, a hybrid target at the short window
    for kw in ({}, {"target_seqs": 24}, {"target_window": 1536, "target_key": "qwen35_4b"},
               {"target_window": 16384, "target_seqs": 60, "allow_over_budget": True}):
        outer = grokmt.resolve_target_geometry(None, **kw)
        argv = grokmt._target_standup_cmd("/tree", tmp_path, tmp_path / "STOP", outer)
        assert argv[argv.index("--eval") + 1] == "multiturn_attacks"
        assert argv[argv.index("--model-key") + 1] == kw.get("target_key", "gemma4e4b")
        inner = _inner(up, argv)
        assert inner.as_dict() == outer.as_dict(), kw
        d = inner.as_dict()
        key = kw.get("target_key", "gemma4e4b")
        assert d["model_key"] == key
        # gemma4e4b and qwen35_4b are measured (qwen35_4b's 2026-09-06 provisional ceiling
        # was lifted 2026-09-11) — the standup resolves the MODEL's own row, and says which
        assert d["budget_source"] == sg.BUDGET_BY_MODEL[key].source
        assert d["budget_source"] in sg.PER_MODEL_BUDGET_SOURCES
    # the default's provenance is what endpoint.json will now record
    d = _inner(up, grokmt._target_standup_cmd("/tree", tmp_path, tmp_path / "STOP",
                                              grokmt.resolve_target_geometry(None))).as_dict()
    assert (d["eval"], d["max_num_seqs"], d["seqs_source"], d["budget"], d["over_budget"],
            d["identity"]) == ("multiturn_attacks", 67, "model_budget", 589_824, False, True)


# --------------------------------------------------------------------------- #
# S1: the intervention and the frozen corpus are part of the run identity      #
# --------------------------------------------------------------------------- #


def _table(tmp_path, ihash):
    from sisteer.capeval.serving.proxy import RouteEntry
    return {"routes": [
        RouteEntry(virtual_model="petri__base", cell_id="none", cell_class="passthrough",
                   serving_identity="s", intervention_hash="base0000"),
        RouteEntry(virtual_model="petri__c", cell_id="si_constitution__psr_kl_all",
                   cell_class="object", serving_identity="s", intervention_hash=ihash)]}


def test_refit_in_place_and_rebuilt_corpus_refuse_the_resume(tmp_path, monkeypatch):
    # the fitted object's ACTUAL intervention hash, from the routing table
    ih = ri.intervention_hashes(_table(tmp_path, "aaaa1111"), ["none", "si_constitution__psr_kl_all"])
    assert ih == {"none": "base0000", "si_constitution__psr_kl_all": "aaaa1111"}
    assert ri.intervention_hashes(_table(tmp_path, "x"), ["missing_cell"]) == {"missing_cell": None}
    # a frozen corpus + its manifest under the held-out corpus dir
    cdir = tmp_path / "corpora"
    cdir.mkdir()
    rows = [{"goal_id": "g1", "sample_idx": 0, "variant_name": "v", "prompt": "p1"}]
    (cdir / "renellm.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    from sisteer.eval.attacks.frozen_corpus import corpus_content_hash
    h1 = corpus_content_hash(rows)
    (cdir / "renellm.jsonl.manifest.json").write_text(json.dumps({"corpus_content_hash": h1}))
    monkeypatch.setenv("SISTEER_HELDOUT_CORPUS_DIR", str(cdir))
    assert ri.default_corpus_dir() == cdir
    hashes = ri.frozen_corpus_hashes(["bon", "pasttense", "renellm", "codeattack"])
    assert hashes == {"renellm": h1}                        # only frozen replays appear
    # a corpus without a manifest is hashed from its rows
    (cdir / "renellm.jsonl.manifest.json").unlink()
    assert ri.frozen_corpus_hashes(["renellm"]) == {"renellm": h1}
    out = tmp_path / "run"
    base = dict(intervention_hashes=ih, frozen_corpora=hashes, cell="si_constitution__psr_kl_all")
    ident = _identity(input_fingerprints=base)
    ri.write_or_verify(out, ident)
    assert ri.write_or_verify(out, _identity(input_fingerprints=dict(base))) .is_file()
    # REFIT IN PLACE: same defenses dir, same cell id, new object -> new hash -> refused
    refit = ri.intervention_hashes(_table(tmp_path, "bbbb2222"), ["none", "si_constitution__psr_kl_all"])
    with pytest.raises(ri.RunIdentityMismatch) as ei:
        ri.write_or_verify(out, _identity(input_fingerprints=dict(base, intervention_hashes=refit)))
    assert "input_fingerprints" in str(ei.value) and "bbbb2222" in str(ei.value)
    # CORPUS REBUILT IN PLACE: same path, new rows -> new content hash -> refused
    rows2 = rows + [{"goal_id": "g2", "sample_idx": 0, "variant_name": "v", "prompt": "p2"}]
    (cdir / "renellm.jsonl").write_text("\n".join(json.dumps(r) for r in rows2) + "\n")
    h2 = ri.frozen_corpus_hashes(["renellm"])["renellm"]
    assert h2 != h1
    with pytest.raises(ri.RunIdentityMismatch):
        ri.write_or_verify(out, _identity(input_fingerprints=dict(base, frozen_corpora={"renellm": h2})))
    # the run id moves with either; the geometry id does not (it is the serving subset)
    assert _identity(input_fingerprints=dict(base, intervention_hashes=refit)).run_id != ident.run_id
    assert _identity(input_fingerprints=dict(base, intervention_hashes=refit)).geometry_id == ident.geometry_id


def test_rows_carry_the_run_id_and_a_foreign_run_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(ri.GEOMETRY_ID_ENV, raising=False)
    monkeypatch.delenv(ri.RUN_ID_ENV, raising=False)
    ident = _identity()
    ri.stamp_env(ident)
    assert ri.current_run_id() == ident.run_id
    assert ri.geometry_stamp() == {"geometry_id": ident.geometry_id, "run_id": ident.run_id}
    from sisteer.eval.attacks import bon_served as bs
    from sisteer.eval.attacks import static_common as sc
    assert bs._geometry_stamp()["run_id"] == ident.run_id
    # same geometry, DIFFERENT run (a refit): the store is refused
    other = _identity(input_fingerprints={"intervention_hashes": {"c": "new"}})
    assert other.geometry_id == ident.geometry_id and other.run_id != ident.run_id
    p = tmp_path / "responses.jsonl"
    p.write_text(json.dumps({"engine": bs.SERVED_ENGINE, "geometry_id": other.geometry_id,
                             "run_id": other.run_id}) + "\n")
    with pytest.raises(sc.StaticStoreIncompleteError) as ei:
        sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE)
    assert "run_id" in str(ei.value)
    # a geometry-stamped but run-less row (an earlier stamp scheme) is refused too
    p.write_text(json.dumps({"engine": bs.SERVED_ENGINE, "geometry_id": ident.geometry_id}) + "\n")
    with pytest.raises(sc.StaticStoreIncompleteError):
        sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE)
    p.write_text(json.dumps({"engine": bs.SERVED_ENGINE, **ri.geometry_stamp()}) + "\n")
    sc._assert_static_engine_consistent(p, bs.SERVED_ENGINE)
    # the drivers build the identity AFTER the routing table exists and put both
    # identities into it
    for rel, tbl in (("scripts/petri_families_prompted.py", "tbl"),
                     ("scripts/petri_families_served_integration.py", "table_path")):
        s = (REPO / rel).read_text()
        assert s.index("export_routing_table(") < s.index("ri.write_or_verify(out, identity)") \
            < s.index("launch_server(")
        assert f"ri.intervention_hashes({tbl}," in s and '["none", PROMPTED])' in s
        assert "ri.frozen_corpus_hashes(" in s and 'rec["run_id"] = identity.run_id' in s


# --------------------------------------------------------------------------- #
# S2: status-only foreign stores are refused; analysis is partitioned          #
# --------------------------------------------------------------------------- #


def test_status_only_foreign_sweep_store_is_refused_and_analysis_partitions(sweep_env, tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, hybrid=True)
    _budget_json(ctx)
    monkeypatch.setattr(sw, "_nvidia_smi", lambda *a: "H200")
    # a pre-existing sweep.jsonl holding ONLY window events (no points), pre-fingerprint
    ev = {"kind": "window_refused", "model": "qwen35_4b", "window_L": 1536, "S_tried": 469,
          "status": "kv_under_coverage", "error": "KV cache holds", "attempt": 1}
    sk = {"kind": "window_skipped", "model": "qwen35_4b", "window_L": 1536, "S_tried": [469],
          "reason": "retries exhausted", "last_status": "kv_under_coverage"}
    (ctx.results_dir / "sweep.jsonl").write_text(json.dumps(ev) + "\n" + json.dumps(sk) + "\n")
    with pytest.raises(SystemExit) as ei:
        sw.run_sweep(ctx, windows=(1536,), budget=None, prompt_tokens=1000,
                     max_tokens=512, warmup_s=0.0, measure_s=1.0)
    assert "DIFFERENT measurement" in str(ei.value)
    assert "window_refused" in str(ei.value) and "window_skipped" in str(ei.value)
    assert "pre-fingerprint" in str(ei.value)
    # ... and a status-only store of ANOTHER fingerprint likewise
    ev2 = dict(ev, measurement_id="0123456789abcdef")
    (ctx.results_dir / "sweep.jsonl").write_text(json.dumps(ev2) + "\n")
    with pytest.raises(SystemExit) as ei:
        sw.run_sweep(ctx, windows=(1536,), budget=None, prompt_tokens=1000,
                     max_tokens=512, warmup_s=0.0, measure_s=1.0)
    assert "0123456789abcdef" in str(ei.value)
    # summarize_rows never folds two measurements: partition first
    a = dict(_pt(1536, "passthrough", 256, 8, 1.0), measurement_id="A")
    b = dict(_pt(1536, "passthrough", 256, 8, 2.0), measurement_id="B")
    evA = dict(ev, measurement_id="A")
    with pytest.raises(ValueError):
        sw.summarize_rows([a, b, evA])
    parts = sw.partition_by_measurement([a, b, evA])
    assert set(parts) == {"A", "B"} and len(parts["A"]) == 2
    sA = sw.summarize_rows([a, b, evA], measurement_id="A")
    assert sA["measurement_id"] == "A" and len(sA["window_refusals"]) == 1
    assert sA["groups"]["1536:passthrough"]["table"][0]["tok_s"] == 1.0
    sB = sw.summarize_rows([a, b, evA], measurement_id="B")
    assert sB["window_refusals"] == [] and sB["groups"]["1536:passthrough"]["table"][0]["tok_s"] == 2.0
    # the analyzer picks the partition with the most points and names the others
    an = _load("_geom_sweep_analyze_s2", "scripts/geom_sweep_analyze.py")
    root = tmp_path / "geom-sweep"
    d = root / "qwen35_4b"
    d.mkdir(parents=True)
    rowsB = [dict(_pt(7424, "passthrough", 97, bb, 10.0 * bb), measurement_id="B")
             for bb in sw.batch_ladder(97)]
    (d / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in [a, evA] + rowsB) + "\n")
    res = an.load_results(root)
    s = res["qwen35_4b"]["summary"]
    assert s["measurement_id"] == "B" and s["other_measurements"] == ["A"]
    assert set(s["groups"]) == {"7424:passthrough"} and s["window_refusals"] == []
    assert s["measurements"] == {"A": {"n_rows": 2, "n_points": 1}, "B": {"n_rows": 7, "n_points": 7}}
    md = an.markdown_tables(res)
    assert "rows of 2 measurements" in md and "measurement B only" in md


# --------------------------------------------------------------------------- #
# S1 (second pass): a stale manifest never blesses changed rows                #
# --------------------------------------------------------------------------- #


def test_rows_replaced_under_an_old_manifest_are_refused(tmp_path, monkeypatch):
    from sisteer.eval.attacks.frozen_corpus import corpus_content_hash
    cdir = tmp_path / "corpora"
    cdir.mkdir()
    rows_a = [{"goal_id": "g1", "sample_idx": 0, "variant_name": "v", "prompt": "corpus A"}]
    rows_b = [{"goal_id": "g1", "sample_idx": 0, "variant_name": "v", "prompt": "corpus B"}]
    h_a, h_b = corpus_content_hash(rows_a), corpus_content_hash(rows_b)
    assert h_a != h_b
    p = cdir / "renellm.jsonl"
    m = cdir / "renellm.jsonl.manifest.json"
    p.write_text(json.dumps(rows_a[0]) + "\n")
    m.write_text(json.dumps({"corpus_content_hash": h_a}))
    # agreeing manifest: the COMPUTED hash is what the identity carries
    assert ri.frozen_corpus_hashes(["renellm"], cdir) == {"renellm": h_a}
    # the rows are replaced (an interrupted rebuild) while the old manifest stands
    p.write_text(json.dumps(rows_b[0]) + "\n")
    with pytest.raises(ri.FrozenCorpusMismatch) as ei:
        ri.frozen_corpus_hashes(["renellm"], cdir)
    msg = str(ei.value)
    assert str(p) in msg and str(m) in msg and h_a in msg and h_b in msg
    assert isinstance(ei.value, ri.RunIdentityMismatch)      # the drivers' SystemExit path catches it
    # a manifest without a hash, or none at all: the rows decide
    m.write_text(json.dumps({"n_rows": 1}))
    assert ri.frozen_corpus_hashes(["renellm"], cdir) == {"renellm": h_b}
    m.unlink()
    assert ri.frozen_corpus_hashes(["renellm"], cdir) == {"renellm": h_b}
    # an unreadable manifest refuses too (the corpus cannot be identified)
    m.write_text("{not json")
    with pytest.raises(ri.FrozenCorpusMismatch):
        ri.frozen_corpus_hashes(["renellm"], cdir)
    # the drivers resolve the corpora INSIDE the refusal path
    for rel in ("scripts/petri_families_prompted.py", "scripts/petri_families_served_integration.py"):
        s = (REPO / rel).read_text()
        i_try = s.index("    from sisteer.capeval.serving import run_identity as ri\n    try:\n")
        i_hash = s.index("ri.frozen_corpus_hashes(", i_try)
        i_exc = s.index("except ri.RunIdentityMismatch as e:", i_try)
        assert i_try < i_hash < i_exc, rel


# --------------------------------------------------------------------------- #
# S2 (second pass): one owner per results dir; the store re-validated at summary#
# --------------------------------------------------------------------------- #


def test_a_held_results_lock_refuses_with_exit_2_naming_the_holder(tmp_path, capsys, monkeypatch):
    ctx = _ctx(tmp_path, hybrid=False)
    _budget_json(ctx)
    holder = sw.ResultsLock(ctx.results_dir, phase="sweep", job_id="111", node="node-A")
    with holder:
        owner = json.loads(holder.owner_path.read_text())
        assert owner["job_id"] == "111" and owner["node"] == "node-A" and owner["pid"] == os.getpid()
        # a competing budget walk on the same dir: refused before anything runs
        with pytest.raises(SystemExit) as ei:
            sw.run_budget(ctx, budget_len=8192, ladder=sw.BUDGET_LADDER)
        assert ei.value.code == sw.EXIT_LOCKED == 2
        out = capsys.readouterr().out
        assert "REFUSED" in out and "job 111" in out and "node-A" in out and "sweep phase" in out
        # ... and a competing sweep
        with pytest.raises(SystemExit) as ei:
            sw.run_sweep(ctx, windows=(1536,), budget=None, prompt_tokens=1000, max_tokens=512,
                         warmup_s=0.0, measure_s=1.0)
        assert ei.value.code == 2
    # released: the owner sidecar is gone and the dir can be taken again
    assert not holder.owner_path.exists()
    with sw.ResultsLock(ctx.results_dir, phase="budget", job_id="222", node="node-B") as again:
        assert json.loads(again.owner_path.read_text())["job_id"] == "222"


import os  # noqa: E402


def test_a_foreign_row_appearing_mid_run_refuses_the_summary(sweep_env, tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, hybrid=True)
    _budget_json(ctx)
    monkeypatch.setattr(sw, "_nvidia_smi", lambda *a: "H200")
    rows_path = ctx.results_dir / "sweep.jsonl"
    real_loop = sw._closed_loop
    injected = {"done": False}

    async def injecting_loop(*a, **k):
        # another process appends a row of ANOTHER measurement while we measure
        if not injected["done"]:
            with rows_path.open("a") as fh:
                fh.write(json.dumps({"kind": "window_refused", "window_L": 1536, "S_tried": 300,
                                     "status": "kv_under_coverage", "measurement_id": "feedfacefeedface"})
                         + "\n")
            injected["done"] = True
        return await real_loop(*a, **k)

    monkeypatch.setattr(sw, "_closed_loop", injecting_loop)
    with pytest.raises(SystemExit) as ei:
        sw.run_sweep(ctx, windows=(1536,), budget=None, prompt_tokens=1000, max_tokens=512,
                     warmup_s=0.0, measure_s=1.0)
    msg = str(ei.value)
    assert "gained 1 row(s) of another measurement" in msg and "feedfacefeedface" in msg
    assert "window_refused" in msg and "MIXED" in msg
    assert not (ctx.results_dir / "sweep_summary.json").exists()
    # the points this run measured are on disk (nothing lost), the lock released
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    assert sum(1 for r in rows if sw.is_point_row(r)) == len(sw.batch_ladder(256))
    assert not (ctx.results_dir / sw.OWNER_FILE).exists()
    # a torn tail line counts as foreign (unknown measurement)
    assert sw.foreign_rows_on_disk(rows_path, "x")[0]["measurement_id"] in (None, "feedfacefeedface")
    with rows_path.open("a") as fh:
        fh.write('{"torn": tru')
    assert any(r.get("kind") == "torn_line" for r in sw.foreign_rows_on_disk(rows_path, "x"))


# --------------------------------------------------------------------------- #
# third pass: a refused resume leaves the old run's serving artifacts intact   #
# --------------------------------------------------------------------------- #

ARTIFACTS = ["chat_template.jinja", "lens_registry", "routing_table.json"]


def _write_artifacts(root, tag, *, readonly=True):
    """A run's serving artifacts as the drivers produce them: a chat template,
    a READ-ONLY registry directory (lens_steer exports it 0444), a routing table."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "chat_template.jinja").write_text(f"template {tag}")
    reg = root / "lens_registry"
    reg.mkdir()
    (reg / "registry.json").write_text(json.dumps({"tag": tag}))
    (reg / "cells").mkdir()
    (reg / "cells" / "c.json").write_text(f"cell {tag}")
    if readonly:
        for p in reg.rglob("*"):
            if p.is_file():
                os.chmod(p, 0o444)
    (root / "routing_table.json").write_text(json.dumps({"routes": [{"cell_id": "c",
                                                                      "intervention_hash": tag}]}))


def _snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_rejected_refit_leaves_the_existing_artifacts_byte_identical(tmp_path):
    out = tmp_path / "run"
    _write_artifacts(out, "A")
    ident_a = _identity(input_fingerprints={"intervention_hashes": {"c": "A"}})
    ri.write_or_verify(out, ident_a)
    before = _snapshot(out)
    # run B (a refit in place): its artifacts are generated INTO A STAGING DIR
    stage = ri.stage_dir(out)
    assert stage.parent == out and stage.name.startswith(ri.STAGING_PREFIX)
    _write_artifacts(stage, "B")
    # (the fixture is not a versioned proxy export, so hand the helper the loaded
    # mapping — the drivers pass the proxy's real routing_table.json path)
    hashes_b = ri.intervention_hashes(json.loads((stage / "routing_table.json").read_text()))
    assert hashes_b == {"c": "B"}
    with pytest.raises(ri.RunIdentityMismatch):
        ri.write_or_verify(out, _identity(input_fingerprints={"intervention_hashes": hashes_b}))
    ri.discard_stage(stage)                                    # the driver's refusal branch
    # run A's routing table, registry (read-only files included) and template are untouched
    assert _snapshot(out) == before
    assert json.loads((out / "routing_table.json").read_text())["routes"][0]["intervention_hash"] == "A"
    assert ri.staging_leftovers(out) == [] and not stage.exists()
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTIFACTS + [ri.RUN_IDENTITY_FILE])


def test_verified_identity_installs_staged_artifacts_by_rename(tmp_path):
    # a FRESH out dir: everything installed, no staging leftovers
    out = tmp_path / "fresh"
    stage = ri.stage_dir(out)
    _write_artifacts(stage, "A")
    ri.write_or_verify(out, _identity())
    status = ri.install_staged(stage, out, ARTIFACTS)
    assert status == {a: "installed" for a in ARTIFACTS}
    assert not stage.exists() and ri.staging_leftovers(out) == []
    assert (out / "lens_registry" / "cells" / "c.json").read_text() == "cell A"
    snap_a = _snapshot(out)
    # a same-identity resume regenerating IDENTICAL artifacts: kept, staged copy dropped
    stage2 = ri.stage_dir(out)
    _write_artifacts(stage2, "A")
    ri.write_or_verify(out, _identity())
    assert ri.install_staged(stage2, out, ARTIFACTS) == {a: "unchanged" for a in ARTIFACTS}
    assert _snapshot(out) == snap_a and ri.staging_leftovers(out) == []
    # ... regenerating DIFFERENT bytes under the same identity (nonces): replaced whole,
    # the read-only old registry removed, nothing partial left behind
    stage3 = ri.stage_dir(out)
    _write_artifacts(stage3, "A2")
    assert ri.install_staged(stage3, out, ARTIFACTS) == {a: "replaced" for a in ARTIFACTS}
    assert (out / "lens_registry" / "cells" / "c.json").read_text() == "cell A2"
    assert (out / "chat_template.jinja").read_text() == "template A2"
    assert ri.staging_leftovers(out) == []
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTIFACTS + [ri.RUN_IDENTITY_FILE])
    # a missing staged artifact is an error, never a partial install
    stage4 = ri.stage_dir(out)
    (stage4 / "chat_template.jinja").write_text("x")
    with pytest.raises(FileNotFoundError):
        ri.install_staged(stage4, out, ARTIFACTS)


def test_drivers_stage_then_verify_then_install_then_launch():
    for rel, tbl in (("scripts/petri_families_prompted.py", "tbl"),
                     ("scripts/petri_families_served_integration.py", "table_path")):
        s = (REPO / rel).read_text()
        i_stage = s.index("stage = ri.stage_dir(out)")
        # every generated artifact targets the staging dir ...
        assert 'tpath = stage / "chat_template.jinja"' in s
        assert 'reg_dir = stage / "lens_registry"' in s
        assert f'{tbl} = stage / "routing_table.json"' in s
        # ... the hashes come from the STAGED table, the identity is verified, THEN installed
        i_hash = s.index(f"ri.intervention_hashes({tbl},")
        i_verify = s.index("ri.write_or_verify(out, identity)")
        i_discard = s.index("ri.discard_stage(stage)")
        i_install = s.index('ri.install_staged(stage, out, ["chat_template.jinja", "lens_registry", '
                            '"routing_table.json"])')
        i_launch = s.index("launch_server(")
        assert i_stage < i_hash < i_verify < i_discard < i_install < i_launch, rel
        # the refusal branch discards the stage before exiting
        exc = s.index("except ri.RunIdentityMismatch as e:")
        assert exc < i_discard < s.index("raise SystemExit", exc)
        # after the install the launch uses the OUT dir's artifacts
        assert 'tpath = out / "chat_template.jinja"' in s[i_install:i_launch]
        assert 'reg_dir = out / "lens_registry"' in s[i_install:i_launch]
        assert f'{tbl} = out / "routing_table.json"' in s[i_install:i_launch]
        assert 'rec["artifacts_install"]' in s


# --------------------------------------------------------------------------- #
# fourth pass: concurrent launches cannot bypass the sidecar                   #
# --------------------------------------------------------------------------- #


def _race_worker(out_dir, over, q):
    """One racer: build its identity, try to create/verify the sidecar, report."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
    from sisteer.capeval.serving import run_identity as _ri
    ident = _ri.RunIdentity(
        model="org/model", revision="a" * 40, tree_sha="b" * 40, model_key="llama8b",
        max_model_len=11520, max_num_seqs=39, client_concurrency=39, budget=458_752,
        budget_source="measured", serving_versions={"vllm": "0.25.1"},
        sampling={"max_new_tokens": 512}, input_fingerprints={"intervention_hashes": {"c": over}})
    try:
        _ri.write_or_verify(out_dir, ident)
        q.put(("won", over, ident.run_id, None))
    except _ri.RunIdentityMismatch as e:
        q.put(("refused", over, ident.run_id, str(e)))


def test_two_process_race_on_a_fresh_out_dir_has_exactly_one_winner(tmp_path):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    out = tmp_path / "race"
    out.mkdir()
    for attempt in range(3):                       # the race itself is nondeterministic; the OUTCOME is not
        for f in out.iterdir():
            f.unlink()
        q = ctx.Queue()
        procs = [ctx.Process(target=_race_worker, args=(str(out), tag, q)) for tag in ("A", "B")]
        for p in procs:
            p.start()
        results = [q.get(timeout=60) for _ in procs]
        for p in procs:
            p.join(timeout=60)
        outcomes = sorted(r[0] for r in results)
        assert outcomes == ["refused", "won"], results
        winner = next(r for r in results if r[0] == "won")
        loser = next(r for r in results if r[0] == "refused")
        sidecar = json.loads((out / ri.RUN_IDENTITY_FILE).read_text())
        assert sidecar["run_id"] == winner[2]                          # the file is the winner's, whole
        assert sidecar["input_fingerprints"]["intervention_hashes"] == {"c": winner[1]}
        assert winner[2] in loser[3] and "DIFFERENT run identity" in loser[3]   # names the winner
        assert loser[2] in loser[3]
        assert not any(n.startswith(".run_identity.json.tmp") for n in os.listdir(out))


def test_lost_race_is_verified_against_the_winner_never_the_memory(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    ident_a = _identity(input_fingerprints={"intervention_hashes": {"c": "A"}})
    ident_b = _identity(input_fingerprints={"intervention_hashes": {"c": "B"}})
    real_link = os.link

    def racing_link(src, dst):
        # B wins between A's tmp write and A's link: the final name appears first
        Path(dst).write_text(json.dumps(dict(ident_b.as_dict(), written_at="t")))
        return real_link(src, dst)                                     # -> FileExistsError

    monkeypatch.setattr(ri.os, "link", racing_link)
    with pytest.raises(ri.RunIdentityMismatch) as ei:
        ri.write_or_verify(out, ident_a)
    assert ident_b.run_id in str(ei.value) and "intervention_hashes" in str(ei.value)
    assert json.loads((out / ri.RUN_IDENTITY_FILE).read_text())["run_id"] == ident_b.run_id
    monkeypatch.setattr(ri.os, "link", real_link)
    # the same identity as the winner proceeds (a legitimate same-identity resume)
    assert ri.write_or_verify(out, ident_b).is_file()
    # a lost race to an IDENTICAL identity also proceeds (verified against the file)
    out2 = tmp_path / "run2"
    out2.mkdir()

    def racing_link_same(src, dst):
        Path(dst).write_text(json.dumps(dict(ident_a.as_dict(), written_at="t")))
        return real_link(src, dst)

    monkeypatch.setattr(ri.os, "link", racing_link_same)
    assert ri.write_or_verify(out2, ident_a).is_file()
    assert not any(n.startswith(".run_identity.json.tmp") for n in os.listdir(out2))


def test_out_dir_lock_refuses_a_live_holder_and_recovers_a_stale_owner(tmp_path, capsys):
    out = tmp_path / "run"
    lock = ri.OutDirLock(out, phase="families", job_id="777", node="node-X").acquire()
    try:
        owner = json.loads(lock.owner_path.read_text())
        assert owner == {**owner, "pid": os.getpid(), "job_id": "777", "node": "node-X",
                         "phase": "families"}
        with pytest.raises(SystemExit) as ei:
            ri.OutDirLock(out, phase="families", job_id="778", node="node-Y").acquire()
        assert ei.value.code == ri.EXIT_LOCKED == 2
        err = capsys.readouterr().err
        assert "REFUSED" in err and "job 777" in err and "node-X" in err and "pid " in err
    finally:
        lock.release()
    assert not lock.owner_path.exists() and not lock.held
    # a STALE owner (a crashed run: sidecar left, no flock) is recovered: taken over, rewritten
    lock.owner_path.write_text(json.dumps({"pid": 999_999_999, "job_id": "dead", "node": "gone",
                                           "phase": "families", "started": "long ago"}))
    with ri.OutDirLock(out, phase="families", job_id="779", node="node-Z") as again:
        assert json.loads(again.owner_path.read_text())["job_id"] == "779"
    assert not (out / ri.RUN_OWNER_FILE).exists()
    # the drivers acquire it BEFORE staging / the sidecar, register release at exit,
    # and release explicitly on a refusal
    for rel in ("scripts/petri_families_prompted.py", "scripts/petri_families_served_integration.py"):
        s = (REPO / rel).read_text()
        i_lock = s.index("out_lock = ri.OutDirLock(out, phase=")
        assert i_lock < s.index("stage = ri.stage_dir(out)") < s.index("ri.write_or_verify(out, identity)")
        assert "atexit.register(out_lock.release)" in s
        exc = s.index("except ri.RunIdentityMismatch as e:")
        assert exc < s.index("out_lock.release()", exc) < s.index("raise SystemExit", exc)


def test_install_with_a_missing_later_artifact_moves_nothing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    stage = ri.stage_dir(out)
    (stage / "chat_template.jinja").write_text("t")
    (stage / "lens_registry").mkdir()
    (stage / "lens_registry" / "registry.json").write_text("{}")
    # routing_table.json is missing: the LAST artifact of the contract
    with pytest.raises(FileNotFoundError) as ei:
        ri.install_staged(stage, out, ARTIFACTS)
    assert "routing_table.json" in str(ei.value) and "nothing installed" in str(ei.value)
    assert not (out / "chat_template.jinja").exists() and not (out / "lens_registry").exists()
    assert (stage / "chat_template.jinja").exists() and (stage / "lens_registry").exists()
    # the stage is still there for the caller to discard
    assert stage.exists()
    ri.discard_stage(stage)
    assert ri.staging_leftovers(out) == []


# --------------------------------------------------------------------------- #
# fifth pass: no wrapper mutates a run's out dir before the driver owns it     #
# --------------------------------------------------------------------------- #

_OUT_DIR_VARS = ("WAVE_OUT_DIR", "RESUME_OUT_DIR", "OUT_DIR", "OUT_ROOT", "RUN_DIR", "OUT_PATH",
                 "WAVE_OUT", "$OUT", "--out")
_MUTATORS = re.compile(r"(?:^|[\s;&|(])(chmod|rm|mv|find)\s", re.M)
#: the ONE exception, applied to this exact line shape and nothing wider: serving_probe.sbatch's attempt record
#: is the WRAPPER's own file — every attempt writes $OUT/probe_attempt.json at START, tmp + mv, so a stalled record resumes
#: from its last attempt's shape (hpo_probe_resume.py reads it; the driver never writes it). The atomic rename is a write of a
#: wrapper-owned file, not a touch of the driver's artifacts; any other mutator on $OUT in that script is still an offender.
_WRAPPER_OWNED_WRITES = {
    "serving_probe.sbatch": '> "$OUT/probe_attempt.json.tmp.$$" && mv -f "$OUT/probe_attempt.json.tmp.$$" "$OUT/probe_attempt.json" \\',
}


def _active_lines(text: str):
    for i, ln in enumerate(text.splitlines(), 1):
        code = ln.split("#", 1)[0] if not ln.lstrip().startswith("#") else ""
        if code.strip():
            yield i, code


def test_no_sbatch_wrapper_mutates_its_out_dir_before_the_driver():
    """qwen_wave.sbatch used to `chmod -R u+w` the resume dir and `rm -rf` its
    lens_registry BEFORE the driver ran — before OutDirLock could refuse a
    duplicate launch — so a rejected resubmission still deleted the incumbent
    run's installed registry. The driver owns the out dir under its lock and
    replaces read-only artifacts itself; no wrapper may touch it."""
    offenders = []
    for path in sorted((REPO / "scripts" / "sbatch").iterdir()):
        if not path.is_file():
            continue
        for lineno, code in _active_lines(path.read_text()):
            if not _MUTATORS.search(code):
                continue
            if code.strip() == _WRAPPER_OWNED_WRITES.get(path.name):
                continue
            if any(v in code for v in _OUT_DIR_VARS) or "find " in code and "-delete" in code \
                    and any(v in code for v in _OUT_DIR_VARS):
                offenders.append(f"{path.name}:{lineno}: {code.strip()}")
    assert offenders == [], offenders
    # the exception is REAL (the line is in the script, so the allowlist cannot rot into a silent pass) and narrow (one line)
    sp_lines = [c.strip() for _, c in _active_lines((REPO / "scripts/sbatch/serving_probe.sbatch").read_text())]
    assert sp_lines.count(_WRAPPER_OWNED_WRITES["serving_probe.sbatch"]) == 1, "serving_probe.sbatch's attempt-record write moved; re-read D97 / D98 ii before widening"
    sb = (REPO / "scripts/sbatch/qwen_wave.sbatch").read_text()
    active = "\n".join(code for _, code in _active_lines(sb))
    assert "chmod -R" not in active and "rm -rf" not in active
    assert 'export WAVE_OUT_DIR="$RESUME_OUT_DIR"' in sb          # the resume seam itself stays
    assert "OutDirLock" in sb                                       # the comment names the owner
    # the driver replaces a read-only registry on its own (the staged install)
    src = (REPO / "src/sisteer/capeval/serving/run_identity.py").read_text()
    assert "_chmod_writable" in src and 'os.replace(dst, old)' in src
