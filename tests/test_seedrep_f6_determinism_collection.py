"""qwen-seedrep-native-battery — determinism records never reached the table reader: the producer writes
`<attempt>/audits/determinism/<table leg>.json` (F2), the pull kept the attempt beneath `RS/stores/…` and `battery_tables` reads
`RS/audits/determinism/*.json` non-recursively -> determinism always MISSING. Now the pull PROMOTES each accepted attempt's records to
`RS/audits/determinism/<leg>__<attempt_id>.json` (`seedrep_served.collect_determinism_records`, the pull's collection step) and the reader
(`battery_tables.load_determinism`) COMBINES a leg's records across s1 shards / attempts (counts sum, `reproduces` = all, per-attempt
provenance kept) and REFUSES the same attempt twice. Producer output -> collection -> the REAL reader, end to end."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", SEEDREP, REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import determinism_check as dc  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                      # dataclasses under `from __future__ import annotations` resolve through sys.modules
    spec.loader.exec_module(mod)
    return mod


bt = _load("battery_tables_f6", SEEDREP / "battery_tables.py")


class _Chain:
    def __getattr__(self, name):
        return lambda *a, **k: self


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


def _stub_modal():
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = types.SimpleNamespace(from_name=lambda *a, **k: {})
    m.is_local = lambda: False
    m.current_function_call_id = lambda: "fc-test"
    return m


_ABSENT = object()


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_f6_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


def _faithful_post(raw: bytes, headers):
    body = json.loads(raw)
    item = body["messages"][0]["content"][len("prompt "):]
    return {"id": "chatcmpl-replay", "choices": [{"message": {"content": f"response for {item} under {body['model']}"}, "finish_reason": "stop"}]}


@pytest.fixture
def hooks(monkeypatch):
    monkeypatch.setattr(dc, "POST_FN", _faithful_post)
    monkeypatch.setattr(dc, "METRICS_FN", lambda: {"num_requests_running": 0, "num_requests_waiting": 0, "generation_tokens_total": 10})
    monkeypatch.setattr(dc, "TOKENIZE_FN", None)


def _produced_root(srv, tmp: Path, shard: str):
    """A synthetic pasttense s1 root for one shard, with the PRODUCER's own determinism record written by the app's argv through the
    REAL check (F2's path): <root>/audits/determinism/pasttense.json (+ the layout copy)."""
    root = fx.make_root(tmp / "stores", leg="pasttense", view="s1", arms=("none",), shard=shard)
    fx.populate_root(root, s=1, arms=("none",), n_items=25)
    ar = ac.read_attempt_root(root.path)
    rec_path = root.path / srv.DETERMINISM_COPY_REL / "pasttense.json"
    argv = srv.determinism_check_argv(driver_python=sys.executable, tree=REPO, table_leg="pasttense", root=root.path, proxy_url="http://127.0.0.1:18080/v1",
                                      metrics_url="http://127.0.0.1:8000/metrics", out=rec_path, copy_to=root.path / srv.DETERMINISM_OUT)
    assert dc.main(argv[2:]) == ac.EXIT_OK and rec_path.is_file()
    return root, ar, rec_path


def test_producer_records_are_promoted_per_attempt_and_the_real_reader_combines_the_shards(srv, tmp_path, hooks):
    rs = tmp_path / "rs"
    root0, ar0, rec0 = _produced_root(srv, tmp_path / "a", "shard00")
    root1, ar1, rec1 = _produced_root(srv, tmp_path / "b", "shard01")
    assert ar0.attempt_id != ar1.attempt_id
    got0 = srv.collect_determinism_records(root0.path, rs, attempt_id=ar0.attempt_id)
    got1 = srv.collect_determinism_records(root1.path, rs, attempt_id=ar1.attempt_id)
    assert got0 == [f"audits/determinism/pasttense__{ar0.attempt_id}.json"] and got1 == [f"audits/determinism/pasttense__{ar1.attempt_id}.json"]
    assert (rs / got0[0]).read_bytes() == rec0.read_bytes() and (rs / got1[0]).read_bytes() == rec1.read_bytes()
    # a re-pull of the SAME attempt is byte-identical: a no-op, confirmed
    assert srv.collect_determinism_records(root0.path, rs, attempt_id=ar0.attempt_id) == got0
    # the REAL reader: one combined record for the leg, both attempts' provenance, the counts summed, reproduces = all
    det = bt.load_determinism(rs / "audits" / "determinism")
    assert sorted(det) == ["pasttense"]
    d = det["pasttense"]
    r0, r1 = json.loads(rec0.read_text()), json.loads(rec1.read_text())
    assert d["combined_from"] == 2 and d["n"] == r0["n"] + r1["n"] == 80 and d["identical"] == r0["identical"] + r1["identical"] == 80
    assert d["n_items"] == 40 and d["reproduces"] is True and d["misses"] == [] and sorted(d["attempt_ids"]) == sorted([ar0.attempt_id, ar1.attempt_id])
    assert {x["attempt_id"] for x in d["records"]} == {ar0.attempt_id, ar1.attempt_id} and all(x["shard_leg"] == "pasttense" for x in d["records"])
    # the inputs loader carries it (a bare RS: every other table simply absent) and the leg spelling normalises through the table's aliases
    inp = bt.load_inputs(rs)
    assert inp.determinism["pasttense"]["n"] == 80 and bt._norm_leg("or_bench_hard_1k") == "or_bench_hard" and bt._norm_leg("misalign_leaking") == "misalign_leaking"
    # a single-attempt leg reads as itself plus its provenance (the consumers' n / identical / n_items / reproduces unchanged)
    only = bt.load_determinism(tmp_path / "one")
    assert only == {}
    (tmp_path / "one").mkdir(); shutil.copyfile(rs / got0[0], tmp_path / "one" / f"pasttense__{ar0.attempt_id}.json")
    one = bt.load_determinism(tmp_path / "one")["pasttense"]
    assert one["n"] == r0["n"] and one["identical"] == r0["identical"] and one["attempt_ids"] == [ar0.attempt_id] and len(one["records"]) == 1


def test_duplicates_and_foreign_records_refuse_never_overwrite(srv, tmp_path, hooks):
    rs = tmp_path / "rs"
    root0, ar0, rec0 = _produced_root(srv, tmp_path / "a", "shard00")
    srv.collect_determinism_records(root0.path, rs, attempt_id=ar0.attempt_id)
    # KNOWN-BAD (the old flat `<leg>.json`): the SAME attempt's record under a second name is a duplicate the reader refuses, never silently picks
    ddir = rs / "audits" / "determinism"
    shutil.copyfile(rec0, ddir / "pasttense.json")
    with pytest.raises(bt.BatteryRefusal, match="two determinism records for leg pasttense from attempt"):
        bt.load_determinism(ddir)
    (ddir / "pasttense.json").unlink()
    # a record naming ANOTHER attempt than the pulled accepted one refuses at collection
    with pytest.raises(srv.DeterminismCollectRefused, match="names attempt"):
        srv.collect_determinism_records(root0.path, rs, attempt_id="ta-other")
    # a target already holding DIFFERENT bytes for (leg, attempt) is never overwritten
    tgt = ddir / f"pasttense__{ar0.attempt_id}.json"
    d = json.loads(tgt.read_text()); d["identical"] = 0; tgt.write_text(json.dumps(d))
    with pytest.raises(srv.DeterminismCollectRefused, match="DIFFERENT determinism record"):
        srv.collect_determinism_records(root0.path, rs, attempt_id=ar0.attempt_id)
    # a record whose `leg` disagrees with its file name refuses
    bad = tmp_path / "bad"; (bad / srv.DETERMINISM_COPY_REL).mkdir(parents=True)
    (bad / srv.DETERMINISM_COPY_REL / "bon.json").write_text(json.dumps({**json.loads(rec0.read_text()), "leg": "pasttense"}))
    with pytest.raises(srv.DeterminismCollectRefused, match="is of leg 'pasttense' but the file is named 'bon'"):
        srv.collect_determinism_records(bad, rs, attempt_id=ar0.attempt_id)
    # a root without records collects nothing (a rung / non-s1 container writes none)
    assert srv.collect_determinism_records(tmp_path / "nothing", rs, attempt_id="ta-x") == []


class FakeDict:
    """modal.Dict's primitive the pull reads (`accepted.get(key)`); the spelling of the accepted keys."""
    def __init__(self):
        self.d = {}

    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self.d:
            return False
        self.d[key] = value
        return True

    def get(self, key, default=None):
        return self.d.get(key, default)


@pytest.fixture(scope="module")
def draft():
    """A draft plan on this tree (the review-fixes pull tests' recipe), for `pull_leg`'s accepted keys and attempt roots."""
    import seedrep_launch_plan as lp
    inputs = lp.load_inputs(REPO, config_roots=(REPO.parent / "seedrep-ws1dp",))
    return lp.build_plan(W="2" * 40, wave_uuid="22222222-3333-4444-8444-555555555555", reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"],
                         pins=inputs["pins"], arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], materialisation="draft",
                         generated_at="2026-09-18T00:00:00Z", config_private_runs=inputs.get("config_private_runs"))


def _det_record(attempt_id: str, leg: str = "pasttense", identical: int = 40) -> dict:
    return {"leg": leg, "attempt_id": attempt_id, "n": 40, "n_requests": 40, "identical": identical, "n_items": 20, "reproduces": identical == 40,
            "misses": [], "arms": [], "W": "2" * 40, "S": 45, "L": 11520}


def test_a_shard_whose_determinism_records_refuse_is_not_admitted_and_leaves_no_record_visible(srv, draft, tmp_path):
    """On the REAL `pull_leg()` failure path: the pull used to append the shard's admission rows to RS/pull_manifest.jsonl (prepare's
    admission authority) BEFORE promoting the determinism records, so a collection refusal left a refused shard admitted. Now every record is
    validated first, then promoted, then the manifest rows are written: shard00 (good record) is admitted with its record promoted; shard01
    (one good record + one naming another attempt) is REFUSED with NO manifest row and NO newly visible record -- not even its good one."""
    import re as _re
    import seedrep_launch_plan as lp
    keys = lp.required_accepted_keys(draft, "pasttense", "base")
    by_sid = {r["standup_id"]: r for r in draft["standups"] + draft["rungs"]}
    sids = sorted(keys.values())
    assert len(sids) >= 2, sids
    acc = FakeDict()
    att_of = {}
    for k, sid in keys.items():
        att_of[sid] = "ta-" + sid.replace("__", "-")                               # one DISTINCT accepted attempt per standup (view + shard)
        acc.put(k, att_of[sid])
    good_sid, bad_sid = sids[0], sids[1]

    def run_get(src, out):
        att = Path(src).name
        m = _re.match(r"^(?P<leg>[^/]+)/(?P<engine>[^/]+)/(?:rungs/(?P<stage>[^/]+)/)?shard(?P<shard>\d+)/attempts/[^/]+$", src[len(draft["results_root"]) + 1:])
        assert m, src
        sid = lp.standup_id(m["leg"], m["engine"], m["stage"] or "base", int(m["shard"]))
        root = out / att
        (root / "exp").mkdir(parents=True, exist_ok=True)
        (root / "COMPLETE.json").write_text(json.dumps({"attempt_id": att}))
        (root / "exp" / "x.jsonl").write_text("{}\n")
        d = root / srv.DETERMINISM_COPY_REL; d.mkdir(parents=True, exist_ok=True)
        if sid == good_sid:
            (d / "pasttense.json").write_text(json.dumps(_det_record(att)))
        elif sid == bad_sid:
            (d / "pasttense.json").write_text(json.dumps(_det_record(att)))                       # a GOOD record ...
            (d / "bon.json").write_text(json.dumps(_det_record("ta-someone-else", leg="bon")))     # ... beside one naming ANOTHER attempt
        if by_sid[sid].get("ladder") and by_sid[sid]["ladder"].get("cap_next") is not None:
            (root / "queue_inventory.json").write_text(json.dumps({"n": 2, "entries": [1, 2]}))
            (root / "ladder_enqueued.json").write_text(json.dumps({"n": 2, "complete": True, "entries": [1, 2]}))

    rs = tmp_path / "rs"
    with pytest.raises(srv.PullRefused, match="names attempt 'ta-someone-else'") as ei:
        srv.pull_leg(draft, "pasttense", tmp_path / "stores", rs, stages="base", accepted=acc, run_get=run_get)
    assert bad_sid in str(ei.value) and good_sid not in str(ei.value).split("REFUSED")[1]
    rows = [json.loads(l) for l in (rs / "pull_manifest.jsonl").read_text().splitlines() if l.strip()]
    admitted = {r["attempt_id"] for r in rows}
    assert att_of[good_sid] in admitted and att_of[bad_sid] not in admitted, admitted                # the refused shard has NO admission row
    assert all(r["standup_id"] != bad_sid for r in rows)
    promoted = sorted(p.name for p in (rs / srv.RS_DETERMINISM_REL).glob("*.json"))
    assert promoted == [f"pasttense__{att_of[good_sid]}.json"], promoted                            # the refused shard's GOOD record is not visible either
    det = bt.load_determinism(rs / srv.RS_DETERMINISM_REL)
    assert det["pasttense"]["attempt_ids"] == [att_of[good_sid]] and det["pasttense"]["records"][0]["attempt_id"] == att_of[good_sid]
    # the pulled root itself still holds the shard's files (evidence is never destroyed); it is simply not admitted
    assert (tmp_path / "stores" / "pasttense").exists()
    # the order in the pull: validate + promote, then the manifest rows; a refusal `continue`s before any row is written
    src = DRIVER.read_text()
    i_verify, i_collect, i_manifest = src.index("problems = verify_pulled_root(pulled"), src.index("collected = collect_determinism_records(pulled, rs, attempt_id=att)"), src.index('with (rs / "pull_manifest.jsonl").open("a") as f:')
    assert i_verify < i_collect < i_manifest


def test_validation_precedes_promotion_and_a_copy_failure_rolls_back(srv, tmp_path, monkeypatch):
    """Two records under one root: validation of BOTH precedes any copy (a bad second record leaves the good first one unpromoted); a copy
    failure after validation rolls back what this call created and refuses."""
    root = tmp_path / "root"; d = root / srv.DETERMINISM_COPY_REL; d.mkdir(parents=True)
    (d / "pasttense.json").write_text(json.dumps(_det_record("ta-1")))
    (d / "bon.json").write_text(json.dumps({**_det_record("ta-1", leg="bon"), "leg": "pasttense"}))       # leg / file-name disagreement
    rs = tmp_path / "rs"
    with pytest.raises(srv.DeterminismCollectRefused, match="is of leg 'pasttense' but the file is named 'bon'"):
        srv.collect_determinism_records(root, rs, attempt_id="ta-1")
    assert not (rs / srv.RS_DETERMINISM_REL).exists()                                                # nothing promoted, not even the good record
    assert [str(x[1].name) for x in srv.validate_determinism_records(tmp_path / "none", rs, attempt_id="ta-1")] == []
    (d / "bon.json").write_text(json.dumps(_det_record("ta-1", leg="bon")))
    plan = srv.validate_determinism_records(root, rs, attempt_id="ta-1")
    assert [x[1].name for x in plan] == ["bon__ta-1.json", "pasttense__ta-1.json"]
    real_copy = srv.shutil.copyfile
    calls = []

    def failing_copy(src, dst):
        calls.append(Path(src).name)
        Path(dst).write_bytes(b'{"trunc')                                # the copy CREATES / partially writes its destination ...
        if Path(src).name == "pasttense.json":
            raise OSError(28, "No space left on device")               # ... then raises (ENOSPC)
        return real_copy(src, dst)
    monkeypatch.setattr(srv.shutil, "copyfile", failing_copy)
    with pytest.raises(srv.DeterminismCollectRefused, match="temp file\\(s\\) and 0 published record\\(s\\) removed; nothing promoted"):
        srv.collect_determinism_records(root, rs, attempt_id="ta-1")
    assert calls == ["bon.json", "pasttense.json"]
    ddir = rs / srv.RS_DETERMINISM_REL
    assert sorted(p.name for p in ddir.iterdir()) == []                  # no *.json (the reader's namespace) and no .tmp left behind
    assert bt.load_determinism(ddir) == {}                               # the reader sees nothing partial
    # cleanup itself failing is never reported as a rollback
    monkeypatch.setattr(srv.os, "unlink", lambda p: (_ for _ in ()).throw(OSError(13, "Permission denied")))
    with pytest.raises(srv.DeterminismCollectRefused, match="CLEANUP FAILED for") as ei:
        srv.collect_determinism_records(root, rs, attempt_id="ta-1")
    assert "removed" not in str(ei.value).split("CLEANUP")[1] and ".bon__ta-1.json.tmp" in str(ei.value)
    monkeypatch.undo()
    for t in ddir.glob(".*.tmp"):
        t.unlink()
    # a successful retry promotes both, published by rename, and the reader combines nothing else
    assert srv.collect_determinism_records(root, rs, attempt_id="ta-1") == ["audits/determinism/bon__ta-1.json", "audits/determinism/pasttense__ta-1.json"]
    assert sorted(p.name for p in ddir.iterdir()) == ["bon__ta-1.json", "pasttense__ta-1.json"]
    assert json.loads((ddir / "pasttense__ta-1.json").read_text())["identical"] == 40


# --------------------------------------------------------------------------- F-U: a SMOKE row's record promotes like any other
def test_a_smoke_rows_record_is_written_by_the_apps_argv_and_promotes_under_its_attempt(srv, tmp_path, hooks):
    """The smoke id form (`smoke-S3__…`, `seedrep_launch_plan.smoke_row`) on ATTEMPT.json's `launch_plan_row` used to REFUSE at the parse, so
    no smoke row ever had a record to promote. Now: the app's argv runs the REAL check to rc 0, the record carries `smoke_id`, and the
    collector (keyed on attempt_id + leg, never the standup id) promotes it; `load_determinism` reads it as the leg's record."""
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=25)
    ap = root.path / ac.ATTEMPT_FILE
    att = json.loads(ap.read_text())
    att["launch_plan_row"] = f"smoke-S3__{root.standup_id}"
    ap.write_text(json.dumps(att) + "\n")
    ar = ac.read_attempt_root(root.path)
    assert ar.smoke_id == "S3" and ar.standup_id == "smoke-S3__pasttense__s1__base__shard00"
    rec_path = root.path / srv.DETERMINISM_COPY_REL / "pasttense.json"
    argv = srv.determinism_check_argv(driver_python=sys.executable, tree=REPO, table_leg="pasttense", root=root.path, proxy_url="http://127.0.0.1:18080/v1",
                                      metrics_url="http://127.0.0.1:8000/metrics", out=rec_path, copy_to=root.path / srv.DETERMINISM_OUT)
    assert dc.main(argv[2:]) == ac.EXIT_OK and rec_path.is_file()
    rec = json.loads(rec_path.read_text())
    assert rec["smoke_id"] == "S3" and rec["standup_id"] == "smoke-S3__pasttense__s1__base__shard00" and rec["leg"] == "pasttense"
    rs = tmp_path / "rs"
    got = srv.collect_determinism_records(root.path, rs, attempt_id=ar.attempt_id)
    assert got == [f"audits/determinism/pasttense__{ar.attempt_id}.json"]
    det = bt.load_determinism(rs / "audits" / "determinism")
    assert det["pasttense"]["smoke_id"] == "S3" and det["pasttense"]["attempt_ids"] == [ar.attempt_id] and det["pasttense"]["reproduces"] is True
