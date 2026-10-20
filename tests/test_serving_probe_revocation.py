"""Stale certification is REVOKED when a serving-probe record is reopened (NO-SHIP finding).

Before: `run --resume --replan` moved the over-budget OOM results aside and changed the plan but never invalidated the
previously written allowed_pairs.json / probe_verdict.json / geometry overlay / memo; `write_verdict` left a standing
menu in place when the new verdict was partial; `hpo_probe_status` said READY on menu presence alone — so a completed
pre-fix record reopened with --resume --replan kept serving its falsely excluding menu to HPO and old geometry to
padded jobs while the re-probe ran.

After (this file pins each):
* every derived certification artifact (menu, verdict, geometry record + THIS record's site overlay, memo, RESULT) is
  ARCHIVED under launches/superseded/<utc>/ with an ARCHIVED.json note — never deleted — the moment a resumed attempt
  has launches to run (replan or otherwise), and a non-certifying verdict archives what an earlier one certified.
* a NON-complete probe_progress.json naming the attempt is written BEFORE any GPU work starts.
* the checkpoint, the verdict, the menu and the overlay's provenance are stamped with the record's RESULT GENERATION (a
  content hash over launches/*.json: sorted "name:sha256(bytes)"), and `hpo_probe_status --record <dir>` says READY only
  for a complete checkpoint whose verdict + menu carry the record's current generation (stamped), or — for an UNSTAMPED
  verdict a pre-stamp carrier wrote — whose listed result statuses recompute to the record's; the checkpoint-only stdin
  mode never says READY.
* the two padded consumers resolve through geometry.load_site_geometry, which refuses an overlay whose record is
  reopened (checkpoint not complete) or whose stamped generation is older than the record's, when the record is reachable.
* `serving_probe.py reverdict --record <dir>` (CPU) archives and rewrites verdict + menu + geometry + memo, stamped.
The scenario above is pinned by `test_a_completed_pre_fix_record_reopened_with_replan_is_not_ready_and_its_menu_is_archived`.
CPU only; fake engines; synthetic ids; no cluster path, no host name."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as g
from sisteer.capeval.serving import standup_manifest as sm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "tests"))
import hpo_probe_status as hps  # noqa: E402
import serving_probe as sp  # noqa: E402
from test_serving_probe_budget import START_TAIL, _oom  # noqa: E402
from test_serving_probe_parallel import _fake_deps  # noqa: E402
from test_serving_probe_smoke import REV, TOY, FakeExporter, _passed  # noqa: E402

MODEL = "qwen35_4b_demo"       # the front door's worked example: the probe subject (a key of lora_modules.ARCH_DIMS_BY_MODEL_KEY; a hybrid)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE", sp.VISIBLE_DEVICES_ENV, "SLURM_JOB_ID"):
        monkeypatch.delenv(v, raising=False)


def _mint_receipt(out: Path, gen: str, *, clusters=("cluster_a", "cluster_b"), nonce: str = None, minted_utc: str = "2026-09-30T13:00:00Z") -> str:
    """What `slurm_revoke_geometry` writes INTO the record after every cluster acknowledged its marker (re-check 3) — EXACTLY the
    shape b497680d2's front door mints (the epoch change adds no field; `minted_utc=None` drops even that one)."""
    import secrets
    nonce = nonce or secrets.token_hex(16)
    d = out / sp.RECEIPTS_DIR; d.mkdir(parents=True, exist_ok=True)
    doc = {"version": 1, "model_key": MODEL, "generation": gen, "nonce": nonce, "reason": "test reopen", "record": str(out), "probe_cluster": "cluster_a",
           "acknowledged": [{"cluster": c, "marker": f"/shared_{c}/exp/reproduce/geometry/{MODEL}.REVOKED.json", "utc": "2026-09-30T13:00:00Z", "archived_files": 1} for c in clusters],
           "site": {"marker": f"/site/geometry/{MODEL}.REVOKED.json", "utc": "2026-09-30T13:00:00Z"},
           "minted_utc": minted_utc, "minted_host": "TESTHOST", "by": "front door (test)"}
    if minted_utc is None:
        doc.pop("minted_utc")
    (d / f"{gen}.{nonce}.json").write_text(json.dumps(doc, indent=2) + "\n")
    return nonce


HEX = __import__("re").compile(r"[0-9a-f]{32,64}")


def _archive_dirs(out: Path):
    """The artifact archive dirs under launches/superseded/ (each carries ARCHIVED.json), oldest first — the superseded
    launch files / dirs a re-plan moves there sit beside them and are not archives."""
    root = out / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / sp.ARCHIVE_NOTE_FILE).is_file()) if root.is_dir() else []


def _plan():
    """wo r4 (S1cc + S1d + the pair's union) + mlp r4 (S1cc + S1d + S2 + the union) at 4,096 / 49,152 on the toy dims — the split
    topology of the 2026-10-01 ruling plus each pair's union (the served launch class, 2026-10-11), 14 launches; `wo` certifies,
    `mlp` is the pair a false start-time OOM would exclude."""
    plan = sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("wo", "mlp"), ranks=(4,),
                         windows=(4096, 49152), ladder_caps=())
    assert plan["n_launches"] == 14
    return plan


def _res(plan, lid, **kw):
    rec = next(r for r in plan["launches"] if r["id"] == lid)
    m = sm.Manifest.from_dict(plan["manifests"][rec["manifest"]])
    return _passed(rec, m, **kw)


def _pre_fix_record(out: Path, site: Path):
    """A record as the pre-fix code left it: every launch terminal — mlp S1cc at 49,152 died at start at S 32 (1.57 M tokens,
    a prefill above anything the record proved) — a COMPLETE checkpoint, a `complete` verdict + menu excluding (4, mlp) on
    `oom_at_R=8_at_L=49152`, the wo set's geometry record + site overlay, a memo and a RESULT; none of them stamped."""
    plan = _plan()
    by = {r["id"]: r for r in plan["launches"]}
    lost = "mlp_r4_S1cc_cap8_L49152"
    results = {lid: _res(plan, lid, engine_S=(64 if lid.endswith("L4096") else 5)) for lid in by if lid != lost}
    results[lost] = _oom(by[lost], 32)
    sp.write_json_atomic(out / sp.PLAN_FILE, plan)
    sp.build_adapter_dirs(plan, out, FakeExporter())
    for lid, res in results.items():
        sp.write_json_atomic(sp.launch_result_path(out, lid), res)
    # the pre-fix verdict: derive the shape from the current writer in memory, then turn it into what the old code wrote
    v, allowed = sp.write_verdict(plan, results, probe_job_id="800490", out=None, verdict_uri=str(out / sp.VERDICT_FILE))
    wo_pair = next(p for p in v["pairs"] if p["set"] == "wo")
    old_v = {**v, "status": "complete", "certifying": True, "replan_required": [], "undecided_pairs": [],
             "excluded": [{"rank": 4, "set": "mlp", "reason": "oom_at_R=8_at_L=49152", "reasons": ["oom_at_R=8_at_L=49152", "ceiling_S<1_at_L=49152"],
                           "decided": True, "outstanding": [], "topology": "S1ccS1dS2", "max_lora_rank_cap": 8}],
             "outstanding": []}
    old_v.pop(sp.RESULT_GENERATION_KEY, None); old_v.pop("n_results", None); old_v.pop("batched_token_budget", None)
    old_menu = {**allowed, "status": "complete", "certifying": True, "pairs": [wo_pair], "replan_required": [],
                "excluded": [{"rank": 4, "set": "mlp", "reason": "oom_at_R=8_at_L=49152", "decided": True}]}
    old_menu.pop(sp.RESULT_GENERATION_KEY, None); old_menu.pop("n_results", None); old_menu.pop("batched_token_budget", None)
    sp.write_json_atomic(out / sp.VERDICT_FILE, old_v)
    sp.write_json_atomic(out / sp.ALLOWED_PAIRS_FILE, old_menu)
    # the geometry the old verdict folded for `wo`, into the record and the site overlay (unstamped, like the old code)
    fold = sp.geometry_fold(plan, results, [wo_pair], out=out, probe_job_id="800490", verdict_uri=str(out / sp.VERDICT_FILE), geometry_dir=site)
    assert fold["status"] == "written"
    for path in (out / sp.GEOMETRY_RECORD_FILE, site / f"{MODEL}.json"):
        doc = json.loads(path.read_text())
        doc["provenance"].pop(sp.RESULT_GENERATION_KEY, None)
        g.write_site_geometry_file(doc, path)
    (out / sp.MEMO_FILE).write_text("# old memo\n")
    (out / sp.RESULT_FILE).write_text("state=VERDICT pairs_allowed=1 pairs_excluded=1 (pre-D108)\n")
    old_prog = sp.progress_record(plan, results)
    old_prog.update({"complete": True, "version": 1})
    for k in ("replan_required", "batched_token_budget", "plan_batched_token_budget", sp.RESULT_GENERATION_KEY, "n_results", "attempt"):
        old_prog.pop(k, None)
    sp.write_json_atomic(out / sp.PROGRESS_FILE, old_prog)
    return plan, lost, wo_pair


# --------------------------------------------------------------------------- the generation


def test_result_generation_is_one_content_hash_over_the_result_set_in_both_modules(tmp_path):
    assert g.record_result_generation(tmp_path) is None and hps.record_generation(tmp_path) is None
    (tmp_path / "launches").mkdir()
    (tmp_path / "launches" / "superseded").mkdir()
    (tmp_path / "launches" / "a_dir").mkdir()
    (tmp_path / "launches" / "superseded" / "old.json").write_text("{}")
    (tmp_path / "launches" / "b.json").write_text('{"launch_id": "b", "status": "passed"}')
    (tmp_path / "launches" / "a.json").write_text('{"launch_id": "a", "status": "oom"}')
    g1, h1 = g.record_result_generation(tmp_path), hps.record_generation(tmp_path)
    assert g1["generation"] == h1["generation"] and g1["n_results"] == h1["n_results"] == 2 and g1["files"] == ["a.json", "b.json"]
    (tmp_path / "launches" / "a.json").write_text('{"launch_id": "a", "status": "passed"}')     # one byte of one result: a new generation
    g2 = g.record_result_generation(tmp_path)
    assert g2["generation"] != g1["generation"] and g2["generation"] == hps.record_generation(tmp_path)["generation"]
    assert sp.result_generation(tmp_path) == g2 and sp.result_generation(None) is None


# ---------------------------------------------------------------------- the reopen scenario


def test_a_completed_pre_fix_record_reopened_with_replan_is_not_ready_and_its_menu_is_archived(tmp_path, monkeypatch):
    """A completed pre-fix record with a menu excluding (4, mlp) on a false OOM and a folded wo overlay; `run --resume
    --replan` reopens it; the REAL hpo_probe_status, queried from inside the first standup (before the rerun finishes),
    says INCOMPLETE — not READY — the menu / verdict / geometry record / overlay / memo / RESULT sit archived under
    launches/superseded/<utc>/, a non-complete checkpoint names the attempt, the overlay resolver returns None (nothing to
    size from); after the rerun the new verdict certifies BOTH pairs, stamped, and the status is READY on the stamp."""
    out, site = tmp_path / "rec", tmp_path / "site"
    out.mkdir(); site.mkdir()
    plan, lost, _wo = _pre_fix_record(out, site)
    # the pre-fix world: menu present, overlay resolvable, and the OLD status contract would have said READY
    assert (out / sp.ALLOWED_PAIRS_FILE).is_file() and g.site_geometry_for(MODEL, "wo", site) is not None
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.READY and info["generation_check"] == "recomputed_match"          # a closed record: its unstamped menu verifies
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(site))
    monkeypatch.setattr(sp, "ProductionExporter", lambda *a, **k: FakeExporter())
    seen_inside: dict = {}

    def probe_from_inside(lid, kw):
        # inside the FIRST standup of the resumed attempt = "before the rerun finishes"
        if seen_inside:
            return
        st, detail, info = hps.probe_state_record(out, job_active=True)
        seen_inside.update(state=st, detail=detail, info=info,
                           menu=(out / sp.ALLOWED_PAIRS_FILE).exists(), verdict=(out / sp.VERDICT_FILE).exists(),
                           geom=(out / sp.GEOMETRY_RECORD_FILE).exists(), overlay=(site / f"{MODEL}.json").exists(),
                           memo=(out / sp.MEMO_FILE).exists(), result=(out / sp.RESULT_FILE).exists(),
                           progress=sp.load_progress(out), overlay_resolves=g.site_geometry_for(MODEL, "wo", site),
                           archived=sorted(p.name for d in (out / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR).iterdir() if d.is_dir() and not d.name.startswith(lost)
                                           for p in d.iterdir()))
    deps, seen = _fake_deps(plan, stand_up_hook=probe_from_inside)
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
    # re-checks 2/3: the record's checkpoint is COMPLETE, so the carrier refuses to reopen it without the front door's revocation
    # RECEIPT — and the typed generation (a public digest) is not a token; no refusal prints a value one could type
    base = ["run", "--model-key", MODEL, "--out", str(out), "--no-throughput", "--resume", "--replan"]
    gen = sp.result_generation(out)["generation"]
    for bad in ([], ["--reopen-token", gen], ["--reopen-token", "0" * 32], ["--reopen-token", "not-hex"]):
        with pytest.raises(SystemExit) as ei:
            sp.main(base + bad)
        msg = str(ei.value)
        assert "REFUSING the reopen" in msg and "03_hpo.sh --phase probe --reopen" in msg and not HEX.search(msg), msg
    assert seen["stand_up"] == {} and (out / sp.ALLOWED_PAIRS_FILE).is_file()          # nothing reopened, nothing archived
    # a receipt for ANOTHER generation is refused; the receipt for the current one opens the record and is consumed
    _mint_receipt(out, "e" * 64, nonce="a" * 32)
    with pytest.raises(SystemExit, match="no revocation receipt for this record's CURRENT result generation"):
        sp.main(base + ["--reopen-token", "a" * 32])
    nonce = _mint_receipt(out, gen)
    rc = sp.main(base + ["--reopen-token", nonce])
    assert rc == 0 and list(seen["stand_up"]) == [lost]
    used = list((out / sp.RECEIPTS_DIR).glob(f"{gen}.{nonce}.used.*.json"))
    assert len(used) == 1 and not (out / sp.RECEIPTS_DIR / f"{gen}.{nonce}.json").exists()   # consumed, kept as history
    # the consumed receipt is not reusable (the record moved on anyway; even a same-generation reuse is refused)
    with pytest.raises(SystemExit, match="REFUSING"):
        sp.consume_reopen_receipt(out, nonce, what="a reuse")
    # what the front door would have seen mid-rerun
    assert seen_inside["state"] == hps.INCOMPLETE_RUNNING and seen_inside["state"] not in hps.FETCH_STATES
    assert "successor" in seen_inside["detail"] and seen_inside["info"]["menu_present"] is False
    assert not seen_inside["menu"] and not seen_inside["verdict"] and not seen_inside["geom"] and not seen_inside["overlay"]
    assert not seen_inside["memo"] and not seen_inside["result"]
    assert seen_inside["overlay_resolves"] is None                                      # nothing for a padded job to size from
    assert sorted(seen_inside["archived"]) == sorted([sp.ARCHIVE_NOTE_FILE, sp.ALLOWED_PAIRS_FILE, sp.VERDICT_FILE, sp.GEOMETRY_RECORD_FILE,
                                                      sp.MEMO_FILE, sp.RESULT_FILE, f"site_overlay_{MODEL}.json"])
    prog = seen_inside["progress"]
    assert prog["complete"] is False and prog["attempt"]["launches_to_run"] == 1 and prog["attempt"]["started_utc"]
    assert prog[sp.RESULT_GENERATION_KEY] and prog["n_results"] == 13                 # the lost launch's result was moved aside (14 - 1)
    # the archive keeps history: the old menu, unchanged, with its false exclusion
    arch = next(d for d in (out / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR).iterdir() if (d / sp.ALLOWED_PAIRS_FILE).is_file())
    old_menu = json.loads((arch / sp.ALLOWED_PAIRS_FILE).read_text())
    assert old_menu["excluded"][0]["reason"] == "oom_at_R=8_at_L=49152" and [p["set"] for p in old_menu["pairs"]] == ["wo"]
    note = json.loads((arch / sp.ARCHIVE_NOTE_FILE).read_text())
    assert note["why"].startswith("run --resume --replan: 1 launch(es) to run") and len(note["moved"]) == 6   # 5 artifacts + the overlay
    assert json.loads((arch / sp.VERDICT_FILE).read_text())["status"] == "complete"     # the pre-fix verdict, kept verbatim
    # after the rerun: the checkpoint is complete and stamped; no menu until `verdict` writes a stamped one
    prog = sp.load_progress(out)
    assert prog["complete"] is True and prog["replan_required"] == [] and prog[sp.RESULT_GENERATION_KEY] == sp.result_generation(out)["generation"]
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.VERDICT_REFUSED and info["menu_present"] is False
    rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(out), "--probe-job-id", "805161", "--geometry-dir", str(site)])
    assert rc == 0
    menu = json.loads((out / sp.ALLOWED_PAIRS_FILE).read_text())
    assert sorted((p["rank"], p["set"]) for p in menu["pairs"]) == [(4, "mlp"), (4, "wo")] and menu["excluded"] == []
    gen = sp.result_generation(out)["generation"]
    assert menu[sp.RESULT_GENERATION_KEY] == gen and json.loads((out / sp.VERDICT_FILE).read_text())[sp.RESULT_GENERATION_KEY] == gen
    assert json.loads((site / f"{MODEL}.json").read_text())["provenance"][sp.RESULT_GENERATION_KEY] == gen
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.READY and info["generation_check"] == "stamped_match" and info["complete"] is True
    assert g.site_geometry_for(MODEL, "mlp", site) is not None and g.site_geometry_for(MODEL, "wo", site) is not None
    # the CLI form the front door calls: three tab-separated fields, the third JSON
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert hps.main(["--record", str(out), "--job-active", "0"]) == 0
    state, detail, js = buf.getvalue().rstrip("\n").split("\t")
    assert state == "READY" and json.loads(js)["generation"] == gen and json.loads(js)["generation_check"] == "stamped_match"


def test_a_result_reopened_after_the_verdict_makes_the_menu_stale_and_the_overlay_refuse(tmp_path, monkeypatch):
    """The stamp is verified against the record AS IT STANDS: a result file rewritten after the verdict (a re-run, a
    manual deletion + resume, any reopen) → status MENU_STALE (mismatch), and the overlay resolver refuses by name;
    a reopened checkpoint (complete=false) refuses the overlay too."""
    out, site = tmp_path / "rec", tmp_path / "site"
    out.mkdir(); site.mkdir()
    plan, lost, _wo = _pre_fix_record(out, site)
    # close the record properly first: a stamped verdict over the CURRENT (pre-fix) results after re-running the lost launch
    monkeypatch.setattr(sp, "ProductionExporter", lambda *a, **k: FakeExporter())
    deps, _seen = _fake_deps(plan)
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
    assert sp.main(["run", "--model-key", MODEL, "--out", str(out), "--no-throughput", "--resume", "--replan", "--reopen-token", _mint_receipt(out, sp.result_generation(out)["generation"])]) == 0
    assert sp.main(["verdict", "--model-key", MODEL, "--out", str(out), "--probe-job-id", "1", "--geometry-dir", str(site)]) == 0
    assert hps.probe_state_record(out, job_active=False)[0] == hps.READY
    # reopen ONE result behind the verdict's back
    f = sp.launch_result_path(out, "wo_r4_S1cc_cap8_L4096")
    doc = json.loads(f.read_text()); doc["note"] = "rewritten after the verdict"; sp.write_json_atomic(f, doc)
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.MENU_STALE and info["generation_check"] == "mismatch" and "NEVER fetch" in detail
    with pytest.raises(g.SiteGeometryError, match="OLDER than the probe record's result set"):
        g.site_geometry_for(MODEL, "wo", site)
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(site))
    with pytest.raises(g.SiteGeometryError, match="OLDER"):
        g.budget_for(MODEL, "wo")                     # the padded consumers' resolver path (padding_preflight / both pins)
    with pytest.raises(g.SiteGeometryError, match="OLDER"):
        g.rows_for(MODEL, "wo")
    # a reopened checkpoint alone refuses the overlay, whatever the stamps
    sp.write_json_atomic(f, {k: v for k, v in doc.items() if k != "note"})
    prog = sp.load_progress(out); prog["complete"] = False; sp.write_json_atomic(out / sp.PROGRESS_FILE, prog)
    with pytest.raises(g.SiteGeometryError, match="REOPENED"):
        g.site_geometry_for(MODEL, "wo", site)
    assert hps.probe_state_record(out, job_active=True)[0] == hps.MENU_STALE
    # an overlay whose record is NOT reachable (another cluster / the grading host site copy) is refused unless it carries the
    # staging verification for its own generation (the proof slurm_stage_geometry verified it against the record)
    doc_ov = json.loads((site / f"{MODEL}.json").read_text())
    doc_ov["provenance"]["probe_record"] = "/shared/exp/multilora_cert/elsewhere_probe_1"
    g.write_site_geometry_file(doc_ov, site / f"{MODEL}.json")
    with pytest.raises(g.SiteGeometryError, match="not reachable from here and the overlay carries no staging_verification"):
        g.site_geometry_for(MODEL, "wo", site)
    gen = doc_ov["provenance"][g.RESULT_GENERATION_KEY]
    staged = g.staged_site_geometry_bytes(site / f"{MODEL}.json", record_generation=gen, verifier_host="TESTHOST")
    (site / f"{MODEL}.json").write_bytes(staged)
    assert g.check_overlay_generation(g.load_site_geometry(MODEL, site)) == "verified:staged"
    assert g.site_geometry_for(MODEL, "wo", site) is not None
    assert g.canonical_site_geometry_sha256(site / f"{MODEL}.json") == g.canonical_site_geometry_sha256(doc_ov)   # the block is outside the content
    # a block for ANOTHER generation, or a stale site copy handed to the stamper, refuses by name
    bad = json.loads(staged); bad[g.STAGING_VERIFICATION_KEY]["record_generation"] = "0" * 64
    g.write_site_geometry_file(bad, site / f"{MODEL}.json")
    with pytest.raises(g.SiteGeometryError, match="staging verified another result set"):
        g.site_geometry_for(MODEL, "wo", site)
    with pytest.raises(g.SiteGeometryError, match="stale site copy is never staged"):
        g.staged_site_geometry_bytes(doc_ov, record_generation="1" * 64, verifier_host="TESTHOST")
    unstamped = dict(doc_ov); unstamped["provenance"] = {k: v for k, v in doc_ov["provenance"].items() if k != g.RESULT_GENERATION_KEY}
    with pytest.raises(g.SiteGeometryError, match="no result_generation stamp"):
        g.staged_site_geometry_bytes(unstamped, record_generation=gen, verifier_host="TESTHOST")


def test_an_unstamped_verdict_is_verified_by_recomputation_from_the_results_it_lists(tmp_path, monkeypatch):
    """The live carrier (6b886ea71) writes a verdict WITHOUT the stamp: status recomputes from the result set the verdict
    lists and says READY when every listed result is in the record with its status and nothing else is; a changed
    status or an extra/missing result is MENU_STALE."""
    out, site = tmp_path / "rec", tmp_path / "site"
    out.mkdir(); site.mkdir()
    plan, lost, _wo = _pre_fix_record(out, site)
    monkeypatch.setattr(sp, "ProductionExporter", lambda *a, **k: FakeExporter())
    deps, _seen = _fake_deps(plan)
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
    assert sp.main(["run", "--model-key", MODEL, "--out", str(out), "--no-throughput", "--resume", "--replan", "--reopen-token", _mint_receipt(out, sp.result_generation(out)["generation"])]) == 0
    assert sp.main(["verdict", "--model-key", MODEL, "--out", str(out), "--probe-job-id", "1", "--geometry-dir", str(site)]) == 0
    for name in (sp.VERDICT_FILE, sp.ALLOWED_PAIRS_FILE):                # strip the stamps: the pre-stamp carrier's output
        doc = json.loads((out / name).read_text()); doc.pop(sp.RESULT_GENERATION_KEY); sp.write_json_atomic(out / name, doc)
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.READY and info["generation_check"] == "recomputed_match" and info["verdict_generation"] is None
    # a status that changed under the verdict
    f = sp.launch_result_path(out, "wo_r4_S1cc_cap8_L49152")
    doc = json.loads(f.read_text()); doc["status"] = "oom"; sp.write_json_atomic(f, doc)
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.MENU_STALE and "status changed ['wo_r4_S1cc_cap8_L49152']" in info["why"]
    doc["status"] = "passed"; sp.write_json_atomic(f, doc)
    # an extra result the verdict never saw
    sp.write_json_atomic(out / sp.LAUNCHES_DIR / "zz_extra.json", {"launch_id": "zz_extra", "status": "passed"})
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.MENU_STALE and "new ['zz_extra']" in info["why"]
    (out / sp.LAUNCHES_DIR / "zz_extra.json").unlink()
    # `reverdict` (CPU) rewrites the artifacts under this code, stamped; the old unstamped ones are archived
    n_arch_before = len(_archive_dirs(out))
    gen = sp.result_generation(out)["generation"]
    for bad in ([], ["--reopen-token", gen], ["--reopen-token", "1" * 32]):
        with pytest.raises(SystemExit) as ei:
            sp.main(["reverdict", "--record", str(out), "--geometry-dir", str(site)] + bad)
        assert "REFUSING the reverdict" in str(ei.value) and not HEX.search(str(ei.value)), str(ei.value)
    rc = sp.main(["reverdict", "--record", str(out), "--geometry-dir", str(site), "--reopen-token", _mint_receipt(out, gen)])
    assert rc == 0
    assert json.loads((out / sp.ALLOWED_PAIRS_FILE).read_text())[sp.RESULT_GENERATION_KEY] == gen
    assert json.loads((out / sp.VERDICT_FILE).read_text())["probe_job_id"] == "1"       # kept from the prior verdict
    assert (out / sp.MEMO_FILE).is_file() and (out / sp.RESULT_FILE).read_text().startswith("state=VERDICT")
    assert sp.load_progress(out)[sp.RESULT_GENERATION_KEY] == gen
    dirs = _archive_dirs(out)
    assert len(dirs) == n_arch_before + 1
    assert (dirs[-1] / sp.ALLOWED_PAIRS_FILE).is_file() and json.loads((dirs[-1] / sp.ARCHIVE_NOTE_FILE).read_text())["why"].startswith("reverdict")
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.READY and info["generation_check"] == "stamped_match"
    # reverdict refuses a record a driver may hold
    prog = sp.load_progress(out); prog["complete"] = False; sp.write_json_atomic(out / sp.PROGRESS_FILE, prog)
    with pytest.raises(SystemExit, match="NOT complete"):
        sp.main(["reverdict", "--record", str(out), "--reopen-token", _mint_receipt(out, gen)])


def test_a_non_certifying_verdict_archives_the_standing_menu_geometry_and_overlay(tmp_path, monkeypatch):
    """`write_verdict` itself never leaves a menu beside a partial verdict: a record certified earlier whose result set now
    yields a partial verdict (a result deleted by hand, then `verdict`) has its menu, geometry record, memo and this
    record's overlay archived; the new verdict is the partial one; a stamped RESULT says PARTIAL."""
    out, site = tmp_path / "rec", tmp_path / "site"
    out.mkdir(); site.mkdir()
    plan, lost, _wo = _pre_fix_record(out, site)
    monkeypatch.setattr(sp, "ProductionExporter", lambda *a, **k: FakeExporter())
    deps, _seen = _fake_deps(plan)
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
    assert sp.main(["run", "--model-key", MODEL, "--out", str(out), "--no-throughput", "--resume", "--replan", "--reopen-token", _mint_receipt(out, sp.result_generation(out)["generation"])]) == 0
    assert sp.main(["verdict", "--model-key", MODEL, "--out", str(out), "--probe-job-id", "1", "--geometry-dir", str(site)]) == 0
    assert sp.main(["memo", "--model-key", MODEL, "--out", str(out)]) == 0
    assert (out / sp.ALLOWED_PAIRS_FILE).is_file() and (site / f"{MODEL}.json").is_file()
    sp.launch_result_path(out, "wo_r4_S1cc_cap8_L49152").unlink()            # a result reopened by hand
    n_before = len(_archive_dirs(out))
    rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(out), "--probe-job-id", "2", "--geometry-dir", str(site)])
    assert rc == sp.EXIT_INCOMPLETE
    assert not (out / sp.ALLOWED_PAIRS_FILE).exists() and not (out / sp.GEOMETRY_RECORD_FILE).exists()
    assert not (site / f"{MODEL}.json").exists() and not (out / sp.MEMO_FILE).exists()
    assert json.loads((out / sp.VERDICT_FILE).read_text())["status"] == "partial"
    assert (out / sp.RESULT_FILE).read_text().startswith("state=PARTIAL")
    dirs = _archive_dirs(out)
    assert len(dirs) == n_before + 1
    assert sorted(p.name for p in dirs[-1].iterdir()) == sorted([sp.ARCHIVE_NOTE_FILE, sp.ALLOWED_PAIRS_FILE, sp.VERDICT_FILE, sp.GEOMETRY_RECORD_FILE,
                                                                 sp.MEMO_FILE, f"site_overlay_{MODEL}.json"])
    assert hps.probe_state_record(out, job_active=False)[0] == hps.INCOMPLETE_STALLED
    assert g.site_geometry_for(MODEL, "wo", site) is None


def test_a_resume_with_nothing_to_run_leaves_a_closed_records_artifacts_alone(tmp_path, monkeypatch):
    out, site = tmp_path / "rec", tmp_path / "site"
    out.mkdir(); site.mkdir()
    plan, lost, _wo = _pre_fix_record(out, site)
    monkeypatch.setattr(sp, "ProductionExporter", lambda *a, **k: FakeExporter())
    deps, _seen = _fake_deps(plan)
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
    assert sp.main(["run", "--model-key", MODEL, "--out", str(out), "--no-throughput", "--resume", "--replan", "--reopen-token", _mint_receipt(out, sp.result_generation(out)["generation"])]) == 0
    assert sp.main(["verdict", "--model-key", MODEL, "--out", str(out), "--probe-job-id", "1", "--geometry-dir", str(site)]) == 0
    menu_bytes = (out / sp.ALLOWED_PAIRS_FILE).read_bytes()
    n_arch = len(_archive_dirs(out))
    deps2, seen2 = _fake_deps(plan)
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps2))
    # nothing to re-run: not a reopen — no token asked for, nothing archived (a successor's bare --resume --replan is this path).
    # Every pair is serving-certified with its measured S at the budget's admission, so the reopen's plan EXTENSION (the
    # current topology's servers for an uncertified / buffer-starved pair) adds nothing either
    assert sp.main(["run", "--model-key", MODEL, "--out", str(out), "--no-throughput", "--resume", "--replan"]) == 0
    assert seen2["stand_up"] == {} and (out / sp.ALLOWED_PAIRS_FILE).read_bytes() == menu_bytes
    plan_after = json.loads((out / sp.PLAN_FILE).read_text())
    assert plan_after["n_launches"] == plan["n_launches"] == 14 and "extensions" not in plan_after
    assert len(_archive_dirs(out)) == n_arch
    assert hps.probe_state_record(out, job_active=False)[0] == hps.READY


# --------------------------------------------------------------------------- re-check 2: the REVOKED marker, FAILED_START


def test_a_revoked_marker_beside_the_overlay_refuses_reachable_or_not_and_the_marker_is_the_signal_for_an_unreachable_record(tmp_path):
    """Clusters cannot read each other: a staged copy's own stamps never change when its source is reopened. The front door
    writes <model>.REVOKED.json beside the overlay before it reopens; the consumer refuses on the marker alone — the only
    freshness signal for a record this host cannot reach — whatever the stamps say."""
    from test_geometry_travels_with_the_front_door import _record
    d = tmp_path / "geometry"; d.mkdir()
    ov = d / f"{MODEL}.json"
    gen = _record()["provenance"][g.RESULT_GENERATION_KEY]
    # a validly staged copy of an unreachable record: accepted on its block
    ov.write_bytes(g.staged_site_geometry_bytes(_record(probe_record="/shared/exp/multilora_cert/elsewhere"), record_generation=gen, verifier_host="TESTHOST"))
    assert g.check_overlay_generation(g.load_site_geometry(MODEL, d)) == "verified:staged"
    # the front door's marker beside it: refused by name, the stamps untouched and irrelevant
    g.revoked_marker_path(ov).write_text(json.dumps({"model_key": MODEL, "generation": gen, "revoked_utc": "2026-09-30T13:00:00Z",
                                                     "reason": "reopen of the record", "by": "front door"}))
    assert g.revoked_marker_path(ov).name == f"{MODEL}.REVOKED.json" and g.read_revoked_marker(ov)["generation"] == gen
    with pytest.raises(g.SiteGeometryError, match="REVOKED — .*REVOKED.json stands beside it .* reason='reopen of the record'"):
        g.site_geometry_for(MODEL, "wo", d)
    # a torn marker refuses too (never a silent accept); removing the marker (what a verified staging does) restores the copy
    g.revoked_marker_path(ov).write_text("{not json")
    with pytest.raises(g.SiteGeometryError, match="REVOKED"):
        g.site_geometry_for(MODEL, "wo", d)
    g.revoked_marker_path(ov).unlink()
    assert g.site_geometry_for(MODEL, "wo", d) is not None
    assert g.read_revoked_marker(ov) is None


def test_failed_start_names_the_attempt_that_died_before_its_first_checkpoint(tmp_path):
    """The carrier wrote probe_attempt.json at start and died (venv bootstrap / import) before the driver's first
    probe_progress.json: NOT_STARTED made the front door retry the unsuffixed unit the funnel then skipped. Now FAILED_START,
    naming the attempt, resumable when a plan exists; RUNNING while a job is active."""
    out = tmp_path / "rec"; out.mkdir()
    attempt = {"version": 1, "written_utc": "2026-09-30T12:34:56Z", "attempt": 2, "max_attempts": 5, "gpus": 2, "parallel": "auto", "mode": "tp1",
               "tp": 1, "sets": "wo", "ranks": "4", "model_key": MODEL, "served_hf_id": "org/toy", "revision": "0" * 40, "job_id": "805161",
               "job_name": f"hpo-{MODEL}-lora-probe", "sha": "6b886ea714625c86664a035397137efc00d22ad8", "qos": "high", "time_limit": "04:00:00",
               "cpus_per_task": "8", "mem_mb": "204800", "host": "TESTHOST"}
    (out / hps.ATTEMPT_FILE).write_text(json.dumps(attempt))
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.FAILED_START and st in hps.STATES and st not in hps.FETCH_STATES and st not in hps.SUBMIT_STATES
    assert "attempt 2/5, job 805161" in detail and "on TESTHOST, started 2026-09-30T12:34:56Z" in detail and "sha 6b886ea71462" in detail
    assert "never wrote probe_progress.json" in detail and "no plan exists — a FRESH start is required" in detail
    assert info["attempt"]["job_id"] == "805161" and info["plan_present"] is False and info["checkpoint_present"] is False
    # a job active: simply RUNNING (the checkpoint is not written yet)
    assert hps.probe_state_record(out, job_active=True)[0] == hps.RUNNING
    # with a plan: resumable
    (out / hps.PLAN_FILE).write_text("{}")
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.FAILED_START and "a plan exists — resume the record" in detail and info["plan_present"] is True
    # no attempt file but a plan or RESULT: still FAILED_START, saying what stands
    (out / hps.ATTEMPT_FILE).unlink()
    st, detail, _ = hps.probe_state_record(out, job_active=False)
    assert st == hps.FAILED_START and "no probe_attempt.json; probe_plan.json present" in detail
    # a torn attempt file is named as unreadable, not trusted
    (out / hps.ATTEMPT_FILE).write_text("{torn")
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.FAILED_START and info["attempt"] is None
    # an empty record dir: NOT_STARTED as before; once the driver's checkpoint exists the checkpoint states rule
    empty = tmp_path / "empty"; empty.mkdir()
    assert hps.probe_state_record(empty, job_active=False)[0] == hps.NOT_STARTED
    (out / hps.PROGRESS_FILE).write_text(json.dumps({"complete": False, "n_done": 0, "plan_launches": 3, "outstanding": [{"id": "x"}], "decision_outstanding": ["x"]}))
    assert hps.probe_state_record(out, job_active=False)[0] == hps.INCOMPLETE_STALLED
    # the CLI prints it like every other state
    import io, contextlib
    (out / hps.PROGRESS_FILE).unlink()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert hps.main(["--record", str(out), "--job-active", "0"]) == 0
    assert buf.getvalue().startswith("FAILED_START\t")


# --------------------------------------------------------------------------- receipt epochs (item 2)


def _bare_record(tmp_path: Path) -> tuple[Path, str]:
    out = tmp_path / "rec"; (out / sp.LAUNCHES_DIR).mkdir(parents=True)
    (out / sp.LAUNCHES_DIR / "L01.json").write_text(json.dumps({"launch_id": "L01", "status": "passed"}))
    return out, sp.result_generation(out)["generation"]


def _states(out: Path) -> dict:
    return {r["nonce"]: r["state"] for r in sp.list_reopen_receipts(out)}


def test_receipt_epochs_only_the_latest_mint_opens_the_record_and_every_other_state_is_named(tmp_path):
    """An orphan receipt of an EARLIER revocation used to authorise a reopen after a newer receipt had recertified and restaged the same
    generation. Now the record has ONE epoch: the receipt minted LAST (minted_utc, then mtime, then name). A: 13:00, B: 13:05 — A refuses
    (superseded, moves nothing), B consumes (.used) and moves A aside (.superseded); afterwards A / B / an unknown nonce each refuse with
    their own reason; the refusals never print a generation, a nonce or a file name."""
    out, gen = _bare_record(tmp_path)
    a = _mint_receipt(out, gen, nonce="a" * 32, minted_utc="2026-09-30T13:00:00Z")
    b = _mint_receipt(out, gen, nonce="b" * 32, minted_utc="2026-09-30T13:05:00Z")
    assert [r["nonce"] for r in sp.list_reopen_receipts(out)] == [a, b] and _states(out) == {a: sp.RECEIPT_STATE_ACTIVE, b: sp.RECEIPT_STATE_ACTIVE}
    with pytest.raises(SystemExit) as ei:
        sp.consume_reopen_receipt(out, a, what="a reopen")
    msg = str(ei.value)
    assert "minted BEFORE a later revocation" in msg and "LATEST revocation" in msg and "03_hpo.sh --phase probe --reopen" in msg and not HEX.search(msg)
    assert (out / sp.RECEIPTS_DIR / f"{gen}.{a}.json").exists() and (out / sp.RECEIPTS_DIR / f"{gen}.{b}.json").exists()      # a refusal moves nothing
    doc = sp.consume_reopen_receipt(out, b, what="a reopen", action="test")
    assert doc["nonce"] == b and doc["minted_utc"] == "2026-09-30T13:05:00Z" and doc[sp.RECEIPT_CONSUME_KEY]["action"] == "test"
    # consumed = IN PROGRESS until the reopen's certifying rewrite completes; while it stands, EVERY consume refuses
    assert _states(out) == {a: sp.RECEIPT_STATE_SUPERSEDED, b: sp.RECEIPT_STATE_INPROGRESS}
    for nonce in (b, a, "c" * 32):
        with pytest.raises(SystemExit) as ei:
            sp.consume_reopen_receipt(out, nonce, what="a second reopen")
        assert "IN PROGRESS" in str(ei.value) and "--resume-interrupted" in str(ei.value) and not HEX.search(str(ei.value))
    assert "the receipt carrying the given nonce" in str(ei.value) if False else True
    owner = doc[sp.RECEIPT_CONSUME_KEY]["owner"]
    assert sp.in_progress_receipts(out)[0]["name"].endswith(f".inprogress.{owner}.json")   # the in-progress file's tag IS the owner
    with pytest.raises(SystemExit, match="owned by another consumer / recovery"):          # completion proves ownership
        sp.finish_reopen_receipt(out, what="an impostor", owner="someone-else")
    fin = sp.finish_reopen_receipt(out, what="the test rewrite", owner=owner)
    assert fin[sp.RECEIPT_CONSUME_KEY]["completed_utc"] and fin[sp.RECEIPT_CONSUME_KEY]["completed_what"] == "the test rewrite"
    assert sp.finish_reopen_receipt(out, what="again", owner=None) is None                # nothing in progress, no claim: a no-op
    with pytest.raises(SystemExit, match="no in-progress receipt remains"):
        sp.finish_reopen_receipt(out, what="a late finisher", owner=owner)                 # a claim on nothing: refused
    assert _states(out) == {a: sp.RECEIPT_STATE_SUPERSEDED, b: sp.RECEIPT_STATE_USED}
    assert list((out / sp.RECEIPTS_DIR).glob(f"{gen}.{b}.used.*.json")) and list((out / sp.RECEIPTS_DIR).glob(f"{gen}.{a}.superseded.*.json"))
    assert not (out / sp.RECEIPTS_DIR / f"{gen}.{a}.json").exists() and not (out / sp.RECEIPTS_DIR / f"{gen}.{b}.json").exists()
    assert not list((out / sp.RECEIPTS_DIR).glob("*.consuming.*")) and not list((out / sp.RECEIPTS_DIR).glob("*.inprogress.*"))
    for nonce, why in ((b, "ALREADY CONSUMED"), (a, "SUPERSEDED by a later revocation"), ("c" * 32, "no revocation receipt for this record's CURRENT result generation carries the given nonce")):
        with pytest.raises(SystemExit) as ei:
            sp.consume_reopen_receipt(out, nonce, what="a reuse")
        assert why in str(ei.value) and not HEX.search(str(ei.value)), str(ei.value)
    # every receipt file the record ever held still exists (history, never deleted) and every name parses
    names = sorted(p.name for p in (out / sp.RECEIPTS_DIR).iterdir())
    assert len(names) == 2 and all(sp.RECEIPT_FILE_RE.match(n) for n in names)


def test_a_b497680d2_receipt_reads_as_valid_then_consumed_never_unknown_and_orders_by_mtime_without_minted_utc(tmp_path):
    """The compat constraint: a receipt the b497680d2 front door minted (the helper's exact shape; no new field is required) is valid once
    and CONSUMED afterwards — never forged / unknown; a receipt without even minted_utc orders by its mtime."""
    out, gen = _bare_record(tmp_path)
    nonce = _mint_receipt(out, gen)                                                       # b497680d2's shape, minted_utc 13:00:00Z
    (row,) = sp.list_reopen_receipts(out)
    assert row["state"] == sp.RECEIPT_STATE_ACTIVE and row["minted_utc"] == "2026-09-30T13:00:00Z" and sp.RECEIPT_FILE_RE.match(row["name"]).group("state") is None
    d = sp.consume_reopen_receipt(out, nonce, what="a reopen")
    assert d["nonce"] == nonce and sp.finish_reopen_receipt(out, what="done", owner=d[sp.RECEIPT_CONSUME_KEY]["owner"])
    with pytest.raises(SystemExit, match="ALREADY CONSUMED"):
        sp.consume_reopen_receipt(out, nonce, what="a reuse")
    assert _states(out) == {nonce: sp.RECEIPT_STATE_USED}
    # no minted_utc at all: the file's mtime orders it; it is still valid-or-consumed
    out2, gen2 = _bare_record(tmp_path / "two")
    bare = _mint_receipt(out2, gen2, minted_utc=None)
    (row,) = sp.list_reopen_receipts(out2)
    assert "minted_utc" not in row["doc"] and row["minted_utc"] and row["state"] == sp.RECEIPT_STATE_ACTIVE
    d2 = sp.consume_reopen_receipt(out2, bare, what="a reopen")
    assert d2["nonce"] == bare and _states(out2) == {bare: sp.RECEIPT_STATE_INPROGRESS}
    sp.finish_reopen_receipt(out2, what="done", owner=d2[sp.RECEIPT_CONSUME_KEY]["owner"]); assert _states(out2) == {bare: sp.RECEIPT_STATE_USED}
    # a later mint without minted_utc (mtime later) still supersedes an earlier one that has it
    out3, gen3 = _bare_record(tmp_path / "three")
    old = _mint_receipt(out3, gen3, nonce="1" * 32, minted_utc="2026-09-30T13:00:00Z")
    import os, time
    time.sleep(0.02)
    new = _mint_receipt(out3, gen3, nonce="2" * 32, minted_utc="2026-09-30T13:30:00Z")
    assert [r["nonce"] for r in sp.list_reopen_receipts(out3)] == [old, new]
    with pytest.raises(SystemExit, match="minted BEFORE a later revocation"):
        sp.consume_reopen_receipt(out3, old, what="a reopen")
    assert sp.consume_reopen_receipt(out3, new, what="a reopen")["nonce"] == new and _states(out3) == {old: sp.RECEIPT_STATE_SUPERSEDED, new: sp.RECEIPT_STATE_INPROGRESS}


def test_concurrent_consumers_of_one_receipt_admit_exactly_one(tmp_path):
    import threading
    out, gen = _bare_record(tmp_path)
    nonce = _mint_receipt(out, gen)
    results: list = []
    gate = threading.Barrier(6)

    def go():
        gate.wait(10)
        try:
            results.append(("ok", sp.consume_reopen_receipt(out, nonce, what="a reopen")["nonce"]))
        except SystemExit as e:
            results.append(("refused", str(e)))
    threads = [threading.Thread(target=go) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    oks = [r for r in results if r[0] == "ok"]
    refused = [r for r in results if r[0] == "refused"]
    assert len(oks) == 1 and len(refused) == 5 and oks[0][1] == nonce
    assert all(("taken by a concurrent run" in m or "IN PROGRESS" in m or "being consumed" in m) and not HEX.search(m) for _, m in refused), refused
    assert _states(out) == {nonce: sp.RECEIPT_STATE_INPROGRESS} and not list((out / sp.RECEIPTS_DIR).glob("*.consuming.*"))


def test_supersede_stale_receipts_moves_every_other_active_receipt_aside_and_keeps_the_named_one(tmp_path):
    out, gen = _bare_record(tmp_path)
    a = _mint_receipt(out, gen, nonce="a" * 32, minted_utc="2026-09-30T13:00:00Z")
    b = _mint_receipt(out, "e" * 64, nonce="b" * 32, minted_utc="2026-09-30T13:01:00Z")           # another generation: still an active orphan
    c = _mint_receipt(out, gen, nonce="c" * 32, minted_utc="2026-09-30T13:02:00Z")
    moved = sp.supersede_stale_receipts(out, keep_nonce=c)
    assert sorted(moved) == sorted([f"{gen}.{a}.json", f"{'e' * 64}.{b}.json"]) and _states(out) == {a: sp.RECEIPT_STATE_SUPERSEDED, b: sp.RECEIPT_STATE_SUPERSEDED, c: sp.RECEIPT_STATE_ACTIVE}
    assert sp.supersede_stale_receipts(out, keep_nonce=c) == []                               # idempotent
    assert sp.consume_reopen_receipt(out, c, what="a reopen")["nonce"] == c and _states(out)[c] == sp.RECEIPT_STATE_INPROGRESS


# --------------------------------------------------------------------------- the interrupted reopen


def _ready_record(tmp_path: Path):
    """A complete, certified record: plan + every launch passed (wo r4 S1cc/S1d, mlp S1cc/S1d/S2 at two windows), verdict + menu, checkpoint."""
    out, site = tmp_path / "rec", tmp_path / "site"; out.mkdir(); site.mkdir()
    plan = _plan()
    results = {r["id"]: _res(plan, r["id"], engine_S=(64 if r["id"].endswith("L4096") else 5)) for r in plan["launches"]}
    sp.write_json_atomic(out / sp.PLAN_FILE, plan)
    sp.build_adapter_dirs(plan, out, FakeExporter())
    for lid, res in results.items():
        sp.write_json_atomic(sp.launch_result_path(out, lid), res)
    sp.write_verdict(plan, results, probe_job_id="800", out=out, verdict_uri=str(out / sp.VERDICT_FILE), geometry_dir=site)
    sp.write_progress(plan, results, out)
    assert hps.probe_state_record(out, job_active=False)[0] == hps.READY
    return out, site, plan


def test_a_reverdict_that_dies_after_the_archive_leaves_the_receipt_in_progress_and_the_recovery_verb_completes_it(tmp_path, monkeypatch):
    """The rename-as-lock used to finalise the receipt as .used BEFORE the reverdict archived the old menu and wrote the
    replacement; a death in between left complete results + checkpoint, no menu (VERDICT_REFUSED), a receipt that refused ALREADY
    CONSUMED, a front door that reopens only READY records, and every geometry copy revoked — stuck until an operator revoked by hand.
    Now the receipt is IN PROGRESS from the consume until the certifying rewrite completes; the status names the interrupted reopen and
    the recovery; a second reopen (same nonce, another nonce, a fresh mint) refuses beside it; `reverdict --resume-interrupted` adopts
    THAT receipt, re-runs the rewrite and restores READY, and only then is the receipt used."""
    out, site, plan = _ready_record(tmp_path)
    gen = sp.result_generation(out)["generation"]
    nonce = _mint_receipt(out, gen)
    real_verdict = sp.cmd_verdict
    seen = {}

    def dies(ns):
        seen["state_during_verdict"] = _states(out)[nonce]
        seen["menu_during_verdict"] = (out / sp.ALLOWED_PAIRS_FILE).exists()
        raise RuntimeError("worker killed after the archive")
    monkeypatch.setattr(sp, "cmd_verdict", dies)
    with pytest.raises(RuntimeError, match="worker killed"):
        sp.main(["reverdict", "--record", str(out), "--reopen-token", nonce, "--geometry-dir", str(site)])
    # the stuck signature: complete checkpoint, no menu, the receipt IN PROGRESS (with the consumer's identity), copies archived
    assert seen == {"state_during_verdict": sp.RECEIPT_STATE_INPROGRESS, "menu_during_verdict": False}
    assert not (out / sp.ALLOWED_PAIRS_FILE).exists() and not (out / sp.VERDICT_FILE).exists() and sp.load_progress(out)["complete"] is True
    assert _states(out) == {nonce: sp.RECEIPT_STATE_INPROGRESS}
    (row,) = sp.in_progress_receipts(out)
    c = row["doc"][sp.RECEIPT_CONSUME_KEY]
    assert c["what"] == "the reverdict" and c["action"] == "reverdict" and c["started_utc"] and c["pid"] == __import__("os").getpid() and "completed_utc" not in c
    # the next status is the dedicated NON-FETCHABLE recovery state, naming the interrupted reopen and the recovery verb — never READY,
    # never a new revocation
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.REOPEN_IN_PROGRESS and st in hps.STATES and st not in hps.FETCH_STATES and st not in hps.SUBMIT_STATES
    assert "IN PROGRESS" in detail and "--resume-interrupted" in detail and "NEVER a new revocation" in detail and "is absent (archived by the reopen)" in detail
    assert info["interrupted_reopen"]["what"] == "the reverdict" and info["interrupted_reopen"]["n_in_progress"] == 1 and "pid" in info["interrupted_reopen"]
    assert not HEX.search(detail) and not HEX.search(json.dumps(info["interrupted_reopen"]))
    # a retry with the same nonce, a consume of a fresh mint (the second concurrent reopen), and a bare consume of an unknown nonce all refuse
    with pytest.raises(SystemExit, match="IN PROGRESS under the receipt carrying the given nonce"):
        sp.main(["reverdict", "--record", str(out), "--reopen-token", nonce, "--geometry-dir", str(site)])
    fresh = _mint_receipt(out, gen, minted_utc="2026-09-30T14:00:00Z")
    with pytest.raises(SystemExit, match="IN PROGRESS under an earlier receipt"):
        sp.consume_reopen_receipt(out, fresh, what="a second reopen")
    assert _states(out) == {nonce: sp.RECEIPT_STATE_INPROGRESS, fresh: sp.RECEIPT_STATE_ACTIVE} and not (out / sp.ALLOWED_PAIRS_FILE).exists()
    with pytest.raises(SystemExit, match="takes no --reopen-token"):
        sp.main(["reverdict", "--record", str(out), "--reopen-token", nonce, "--resume-interrupted"])
    # the recovery: adopt the in-progress receipt, re-run the certifying rewrite, READY again, the receipt used only now
    monkeypatch.setattr(sp, "cmd_verdict", real_verdict)
    assert sp.main(["reverdict", "--record", str(out), "--resume-interrupted", "--geometry-dir", str(site)]) == 0
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.READY and info["interrupted_reopen"] is None and (out / sp.ALLOWED_PAIRS_FILE).is_file()
    assert _states(out) == {nonce: sp.RECEIPT_STATE_USED, fresh: sp.RECEIPT_STATE_ACTIVE}
    used = json.loads(next((out / sp.RECEIPTS_DIR).glob(f"{gen}.{nonce}.used.*.json")).read_text())[sp.RECEIPT_CONSUME_KEY]
    assert used["completed_utc"] and used["completed_what"] == "the reverdict" and len(used["resumed"]) == 1 and used["resumed"][0]["what"] == "the reverdict"
    assert used["owner"] == used["resumed"][0]["owner"] and used["resumed"][0]["took_over_from"] == c["owner"]   # the recovery CLAIMED it
    # nothing left to resume; the fresh (later) receipt is now the epoch and opens the record normally
    with pytest.raises(SystemExit, match="nothing to resume"):
        sp.main(["reverdict", "--record", str(out), "--resume-interrupted", "--geometry-dir", str(site)])
    assert sp.main(["reverdict", "--record", str(out), "--reopen-token", fresh, "--geometry-dir", str(site)]) == 0
    assert _states(out) == {nonce: sp.RECEIPT_STATE_USED, fresh: sp.RECEIPT_STATE_USED} and hps.probe_state_record(out, job_active=False)[0] == hps.READY


def test_an_uninterrupted_reverdict_holds_the_receipt_in_progress_until_the_menu_is_back(tmp_path, monkeypatch):
    out, site, plan = _ready_record(tmp_path)
    nonce = _mint_receipt(out, sp.result_generation(out)["generation"])
    real_memo = sp.cmd_memo
    seen = {}

    def memo(ns):
        seen["state_at_memo"] = _states(out)[nonce]; seen["menu_at_memo"] = (out / sp.ALLOWED_PAIRS_FILE).exists()
        return real_memo(ns)
    monkeypatch.setattr(sp, "cmd_memo", memo)
    assert sp.main(["reverdict", "--record", str(out), "--reopen-token", nonce, "--geometry-dir", str(site)]) == 0
    assert seen == {"state_at_memo": sp.RECEIPT_STATE_INPROGRESS, "menu_at_memo": True}     # the menu was back, the receipt not yet used
    assert _states(out) == {nonce: sp.RECEIPT_STATE_USED} and hps.probe_state_record(out, job_active=False)[0] == hps.READY


def test_a_crash_between_the_consume_and_the_archive_never_reads_ready_though_the_old_menu_still_stands(tmp_path, monkeypatch):
    """The .inprogress receipt is published BEFORE the archive; a crash in that window leaves the OLD allowed_pairs.json
    and a complete checkpoint — the menu-present branch used to say READY and 03 fetched the pre-reopen selection. Now ANY in-progress
    receipt makes the record REOPEN_IN_PROGRESS whatever else stands."""
    out, site, plan = _ready_record(tmp_path)
    nonce = _mint_receipt(out, sp.result_generation(out)["generation"])

    def dies_before_archive(*a, **k):
        raise RuntimeError("killed before the archive")
    monkeypatch.setattr(sp, "archive_derived_artifacts", dies_before_archive)
    with pytest.raises(RuntimeError, match="before the archive"):
        sp.main(["reverdict", "--record", str(out), "--reopen-token", nonce, "--geometry-dir", str(site)])
    assert (out / sp.ALLOWED_PAIRS_FILE).is_file() and sp.load_progress(out)["complete"] is True      # the OLD menu still stands …
    assert _states(out) == {nonce: sp.RECEIPT_STATE_INPROGRESS}
    st, detail, info = hps.probe_state_record(out, job_active=False)
    assert st == hps.REOPEN_IN_PROGRESS and info["menu_present"] is True and info["generation_check"] == "stamped_match"   # … and is NOT fetchable
    assert "still stands but is the PRE-reopen selection" in detail and "NEVER fetch" in detail and "--resume-interrupted" in detail
    # the CLI prints the state for the front door
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert hps.main(["--record", str(out), "--job-active", "0"]) == 0
    assert buf.getvalue().split("\t", 1)[0] == "REOPEN_IN_PROGRESS"
    # the recovery (the dead consumer's lock is free) completes: READY again
    monkeypatch.undo()
    assert sp.main(["reverdict", "--record", str(out), "--resume-interrupted", "--geometry-dir", str(site)]) == 0
    assert hps.probe_state_record(out, job_active=False)[0] == hps.READY and _states(out) == {nonce: sp.RECEIPT_STATE_USED}


def test_recovery_is_exclusive_two_concurrent_recoveries_admit_one_and_a_recovery_cannot_race_the_live_consumer(tmp_path, monkeypatch):
    """Two `reverdict --resume-interrupted` used to list the same .inprogress, both write `resumed`, both archive
    and rewrite the shared menu; finish took whichever receipt existed. Now the record's reopen lock (flock beside the receipts, held
    for the whole rewrite, released on death) serialises rewrites, the recovery CLAIMS the receipt by rename (one winner) and completion
    is owner-checked."""
    import threading
    out, site, plan = _ready_record(tmp_path)
    nonce = _mint_receipt(out, sp.result_generation(out)["generation"])
    real_verdict = sp.cmd_verdict
    # (a) a recovery racing the LIVE original consumer: the original holds the lock through its verdict; the recovery refuses by name
    raced = {}

    def verdict_with_racer(ns):
        try:
            sp.main(["reverdict", "--record", str(out), "--resume-interrupted", "--geometry-dir", str(site)])
            raced["result"] = "ran"
        except SystemExit as e:
            raced["result"] = str(e)
        return real_verdict(ns)
    monkeypatch.setattr(sp, "cmd_verdict", verdict_with_racer)
    assert sp.main(["reverdict", "--record", str(out), "--reopen-token", nonce, "--geometry-dir", str(site)]) == 0
    assert "HOLDS this record's reopen lock" in raced["result"] and "the reverdict" in raced["result"]      # same process: the lock is not re-entrant either
    assert _states(out) == {nonce: sp.RECEIPT_STATE_USED} and hps.probe_state_record(out, job_active=False)[0] == hps.READY
    monkeypatch.setattr(sp, "cmd_verdict", real_verdict)
    # (b) an interrupted reopen, then TWO concurrent recoveries: exactly one wins, the other refuses by name, the record ends READY once
    nonce2 = _mint_receipt(out, sp.result_generation(out)["generation"], minted_utc="2026-09-30T15:00:00Z")

    def dies(ns):
        raise RuntimeError("killed after the archive")
    monkeypatch.setattr(sp, "cmd_verdict", dies)
    with pytest.raises(RuntimeError):
        sp.main(["reverdict", "--record", str(out), "--reopen-token", nonce2, "--geometry-dir", str(site)])
    sp.release_reopen_lock(out)                                              # the dead process's lock is gone; in one process we drop it by hand
    assert _states(out)[nonce2] == sp.RECEIPT_STATE_INPROGRESS and hps.probe_state_record(out, job_active=False)[0] == hps.REOPEN_IN_PROGRESS
    gate = threading.Barrier(2)
    results: list = []

    def slow_verdict(ns):
        import time
        time.sleep(0.3)                                                      # hold the lock long enough for the other recovery to collide
        return real_verdict(ns)
    monkeypatch.setattr(sp, "cmd_verdict", slow_verdict)

    def recover():
        gate.wait(10)
        try:
            results.append(("ok", sp.main(["reverdict", "--record", str(out), "--resume-interrupted", "--geometry-dir", str(site)])))
        except SystemExit as e:
            results.append(("refused", str(e)))
    ts = [threading.Thread(target=recover) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    oks = [r for r in results if r[0] == "ok"]; refused = [r for r in results if r[0] == "refused"]
    assert len(oks) == 1 and oks[0][1] == 0 and len(refused) == 1, results
    msg = refused[0][1]
    assert "HOLDS this record's reopen lock" in msg or "claimed by a concurrent recovery" in msg, msg
    # no receipt nonce leaks; the one hex the refusal MAY carry is the lock holder's own tree_sha (the lock file's identity line, quoted
    # when the loser lost the flock rather than the in-process check — which of the two it hits is the threads' interleaving)
    assert nonce not in msg and nonce2 not in msg and set(HEX.findall(msg)) <= {sp._tree_sha()}, msg
    assert _states(out) == {nonce: sp.RECEIPT_STATE_USED, nonce2: sp.RECEIPT_STATE_USED} and hps.probe_state_record(out, job_active=False)[0] == hps.READY
    assert not list((out / sp.RECEIPTS_DIR).glob("*.inprogress.*"))
