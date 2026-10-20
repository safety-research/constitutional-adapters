"""fz2 (the F-AA/F-AE residual): ledger and evidence OWNERSHIP on a multi-child attempt root.

Known-bad at 479c53a13 (from the pinned plan): a GSM8K standup starts ONE usage ledger PER CHILD STORE
(`seedrep_launch_plan._gsm8k_children` declares `ledger.ledger_path_rel`, `seedrep_served.gsm8k_ledger_argv` passes it as `--ledger-path`,
the writer creates the file at start), so with eight s1 children `RootReads.ledger_rows()` refused `8 files called 'usage_ledger.jsonl' …
an ambiguous attempt root is not read` and `request_evidence_audit.load_evidence()` could never produce a verdict. The same holds for
`request_evidence.jsonl` / `response_evidence.jsonl`: a base child's evidence dir IS its store root (`child_store_root`), one per child.

Green: on the declared layout every child store's files are loaded (rows stamped `_source` = root-relative file), a request id in two
files refuses by name, an attempt-level ledger on a GSM8K row refuses by name (no writer exists), a copy outside the root / `exp/` refuses
as unowned, and non-GSM8K roots keep the single-ledger rule. The layout under test is emitted by the REAL plan (`_gsm8k_children`) and the
REAL `gsm8k_ledger_argv` (the app under a stub `modal`), never typed by hand. Synthetic roots only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "scripts" / "modal", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import at_cap_table as act  # noqa: E402
import audit_common as ac  # noqa: E402
import determinism_check as dc  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
from test_seedrep_f2_determinism_argv import _stub_modal  # noqa: E402  (the app under a stub `modal`, the f2 seam's loader)

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
_ABSENT = object()
GEOMETRY_EVAL = "capability:gen_long_gsm8k_common_f22"
PROXY = "http://127.0.0.1:9/v1"


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_fz2_under_test", DRIVER)
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


@pytest.fixture(autouse=True)
def _hooks(monkeypatch):
    def post(raw: bytes, headers):
        body = json.loads(raw)
        return {"id": "chatcmpl-replay", "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    monkeypatch.setattr(dc, "POST_FN", post)
    monkeypatch.setattr(dc, "METRICS_FN", lambda: {"num_requests_running": 0, "num_requests_waiting": 0, "generation_tokens_total": 10})
    monkeypatch.setattr(dc, "TOKENIZE_FN", None)
    yield


def test_fixture_child_store_root_is_the_plans():
    assert fx.GSM8K_OUT_REL == lp.GSM8K_OUT_REL and fx.GSM8K_FIXTURE_CAP == 8192


def real_children():
    """The eight s1 GSM8K children of the REAL plan builder (a pin per fitted cell; the tree path is a string the argv carries)."""
    fitted = [r["cell"] for r in lp.arm_runs("s1") if r["cell"] != "none" and not r["cell"].startswith("prompted__")]
    pins = {c: fx._sha(f"pin|{c}") for c in fitted}
    kids = lp._gsm8k_children("s1", fx.W_FIXTURE, 8192, pins_by_cell=pins, base_cap=8192, geometry_eval=GEOMETRY_EVAL)
    assert len(kids) == 8 and len({k["cell"] for k in kids}) == 8
    return kids


def lay_out_real_gsm8k_root(tmp_path: Path, srv):
    """A gsm8k root populated by the fixture, its pipeline re-laid where the REAL app puts it: each child's evidence dir =
    `child_store_root(child, resolved argv)` (its `--out-root`), its ledger = `gsm8k_ledger_argv(...)`'s `--ledger-path`; the five
    children with no store here still started their ledgers (empty files)."""
    arms = fx.DEFAULT_ARMS  # none + one fitted + one prompted: three of the eight cells
    root = fx.make_root(tmp_path / "stores", leg="gsm8k", view="s1", arms=arms)
    fx.populate_root(root, s=1, arms=arms, n_items=3)
    root.write_gsm8k_cell_copies([k["cell"] for k in real_children()], s=1)
    store_of_arm, ledger_of_arm, empty = {}, {}, []
    for k in real_children():
        argv = [a.replace(lp.ATTEMPT_ROOT_PLACEHOLDER, str(root.path)) for a in k["argv"]]
        store = Path(lp.child_store_root(k, argv))
        led = srv.gsm8k_ledger_argv(k, root=root.path, driver_python="/py", proxy_url=PROXY, port=40000, attempt_id="a1")
        ledger = Path(led[led.index("--ledger-path") + 1])
        assert ledger.parent == store and ledger == root.path / k["ledger"]["ledger_path_rel"], (ledger, store)
        if k["cell"] in arms:
            store_of_arm[k["cell"]], ledger_of_arm[k["cell"]] = store, ledger
        else:
            empty.append(ledger)
    root.split_pipeline_by_store_root(store_of_arm, empty_ledgers=empty)
    return root, ac.read_attempt_root(root.path), store_of_arm, ledger_of_arm, empty


def _audit(ar, tmp_path: Path):
    return rea.audit_root(ar, out_dir=tmp_path / "audits", legs=["gsm8k"], policy_file=None, thinking_expect=dict(ac.THINKING_EXPECT_DEFAULT),
                          petri_behaviours=list(fx.PETRI_BEHAVIOURS), model_key=ac.MODEL_KEY, client_status="absent", siblings=[])


# --------------------------------------------------------------------------- #
# known-bad → green on the real layout                                                                              #
# --------------------------------------------------------------------------- #

def test_known_bad_eight_child_ledgers_refused_the_single_file_walk(tmp_path: Path, srv):
    root, ar, _, _, _ = lay_out_real_gsm8k_root(tmp_path, srv)
    with pytest.raises(ac.AuditRefusal, match=r"8 files called 'usage_ledger.jsonl'.*an ambiguous attempt root is not read"):
        ac.find_one(root.path, ac.LEDGER_FILE)
    with pytest.raises(ac.AuditRefusal, match=r"3 files called 'request_evidence.jsonl'.*ambiguous"):
        ac.find_one(root.path, ac.EVIDENCE_FILE)


def test_real_gsm8k_layout_loads_every_child_with_provenance_and_audits(tmp_path: Path, srv):
    root, ar, store_of_arm, ledger_of_arm, empty = lay_out_real_gsm8k_root(tmp_path, srv)
    rel = lambda p: str(p.relative_to(root.path))  # noqa: E731
    reads = ac.RootReads()
    ledger, records = reads.ledger_rows(ar), reads.records(ar)
    assert len(ledger) == 9 and len(records) == 9  # 3 arms × 3 items, none dropped, none doubled
    assert {r[ac.ROW_SOURCE_FIELD] for r in ledger} == {rel(p) for p in ledger_of_arm.values()}
    assert {r[ac.ROW_SOURCE_FIELD] for r in records} == {rel(d / ac.EVIDENCE_FILE) for d in store_of_arm.values()}
    for r in ledger:  # each row's source is ITS arm's ledger
        arm = r["vmodel"].replace("petri__", "")
        assert r[ac.ROW_SOURCE_FIELD] == rel(ledger_of_arm[arm])
    assert sorted(rel(p) for p in empty) == sorted(rel(p) for p, o in ac.owned_evidence_files(root.path, ac.LEDGER_FILE) if p in empty)
    assert all(o == ac.OWNER_CHILD_STORE for _, o in ac.owned_evidence_files(root.path, ac.LEDGER_FILE))
    # the response side too (owned, stamped)
    assert {r[ac.ROW_SOURCE_FIELD] for r in ac.response_evidence_rows(root.path)} == {rel(d / ac.RESPONSE_EVIDENCE_FILE) for d in store_of_arm.values()}
    assert len(ac.response_text_shas_by_rid(root.path)) == 9
    # load_evidence / audit_root: verdicts for every arm, sources carried
    ev = rea.load_evidence(ar)
    assert len(ev.ledger_rows) == 9 and len(ev.records) == 9 and len(ev.ledger_by_id) == 9
    verdicts = _audit(ar, tmp_path)
    assert {v["arm"] for v in verdicts} == set(fx.DEFAULT_ARMS)
    for v in verdicts:
        assert v["verdict"] == "AGREES", (v["arm"], v["problem"])
        assert v["ledger_sources"] == sorted(rel(p) for p in ledger_of_arm.values())
        assert v["evidence_sources"] == sorted(rel(d / ac.EVIDENCE_FILE) for d in store_of_arm.values())
    # the at-cap table's ledger read and the determinism check's record read, on the same root
    roots = ac.discover_attempt_roots(tmp_path / "stores")
    row = act.row_for("gsm8k", "none", 1, "s1", roots, petri=list(fx.PETRI_BEHAVIOURS), trig_static=act.trigger_share_static(),
                      trig_gsm8k=act.trigger_share_gsm8k())
    assert row["applies"] is True and row["n"] == 3
    det = dc.run_check(ar, leg="gsm8k", arms=["none"], n=3, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                       allow_busy=False, petri_behaviours=[])
    assert det["n_requests"] == 3


def test_request_id_in_two_child_ledgers_refuses_by_name(tmp_path: Path, srv):
    root, ar, _, ledger_of_arm, _ = lay_out_real_gsm8k_root(tmp_path, srv)
    a, b = ledger_of_arm["none"], ledger_of_arm[fx.DEFAULT_ARMS[1]]
    first = a.read_text().splitlines()[0]
    b.write_text(b.read_text() + first + "\n")
    rid = json.loads(first)["request_id"]
    with pytest.raises(ac.AuditRefusal, match=rf"request_id {rid!r} appears in two 'usage_ledger.jsonl' files .* — one request is recorded once"):
        ac.read_usage_ledger(root.path, leg="gsm8k")
    with pytest.raises(ac.AuditRefusal, match="appears in two 'usage_ledger.jsonl' files"):
        rea.load_evidence(ar)


def test_attempt_level_ledger_on_a_gsm8k_row_refuses_by_name(tmp_path: Path, srv):
    root, ar, _, _, _ = lay_out_real_gsm8k_root(tmp_path, srv)
    (root.path / ac.LEDGER_FILE).write_text("")
    with pytest.raises(ac.AuditRefusal, match=r"an attempt-level 'usage_ledger.jsonl' on a GSM8K row .* nothing writes one at the root"):
        ac.read_usage_ledger(root.path, leg="gsm8k")


def test_no_child_ledger_on_a_gsm8k_row_refuses(tmp_path: Path, srv):
    root, ar, _, ledger_of_arm, empty = lay_out_real_gsm8k_root(tmp_path, srv)
    for p in list(ledger_of_arm.values()) + empty:
        p.unlink()
    with pytest.raises(ac.AuditRefusal, match=r"no child 'usage_ledger.jsonl' under exp/ — a GSM8K row starts one per child store"):
        ac.read_usage_ledger(root.path, leg="gsm8k")


def test_unowned_copy_refuses(tmp_path: Path, srv):
    root, ar, store_of_arm, _, _ = lay_out_real_gsm8k_root(tmp_path, srv)
    stray = root.path / "children" / ac.EVIDENCE_FILE
    stray.parent.mkdir(exist_ok=True)
    stray.write_bytes((store_of_arm["none"] / ac.EVIDENCE_FILE).read_bytes())
    with pytest.raises(ac.AuditRefusal, match=r"children/request_evidence.jsonl: a 'request_evidence.jsonl' that is neither the attempt root's nor "
                                              r"under a child store root \(exp/…\) — an unowned evidence file is not read"):
        ac.read_request_evidence(root.path)


# --------------------------------------------------------------------------- #
# non-GSM8K roots: the single-ledger rule, and the attempt-root (rung) shape                                        #
# --------------------------------------------------------------------------- #

def test_non_gsm8k_root_keeps_the_single_ledger_rule(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=3)
    ar = ac.read_attempt_root(root.path)
    rows = ac.RootReads().ledger_rows(ar)
    assert len(rows) == 6 and {r[ac.ROW_SOURCE_FIELD] for r in rows} == {ac.LEDGER_FILE}
    assert [o for _, o in ac.owned_evidence_files(root.path, ac.LEDGER_FILE)] == [ac.OWNER_ATTEMPT_ROOT]
    second = root.path / "exp" / "other" / ac.LEDGER_FILE
    second.parent.mkdir(parents=True)
    second.write_text("")
    with pytest.raises(ac.AuditRefusal, match="2 files called 'usage_ledger.jsonl'.*ambiguous attempt root"):
        ac.read_usage_ledger(root.path, leg="pasttense")
    with pytest.raises(ac.AuditRefusal, match="2 files called 'usage_ledger.jsonl'"):
        ac.RootReads().ledger_rows(ar)


def test_attempt_root_evidence_plus_child_evidence_both_owned(tmp_path: Path):
    """A rung consumer's evidence lands at the literal attempt root (`rung_child_env`) — both owners load, sources say which."""
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none", "prompted__si_constitution"))
    fx.populate_root(root, s=1, arms=("none", "prompted__si_constitution"), n_items=2)
    recs = ac.read_jsonl(root.path / ac.EVIDENCE_FILE)
    child = root.path / "exp" / "some_child_store"
    child.mkdir(parents=True)
    moved = [r for r in recs if r["extra"]["arm"] == "prompted__si_constitution"]
    (child / ac.EVIDENCE_FILE).write_text("".join(json.dumps(r) + "\n" for r in moved))
    (root.path / ac.EVIDENCE_FILE).write_text("".join(json.dumps(r) + "\n" for r in recs if r["extra"]["arm"] == "none"))
    rows = ac.read_request_evidence(root.path)
    assert len(rows) == len(recs)
    assert {r[ac.ROW_SOURCE_FIELD] for r in rows} == {ac.EVIDENCE_FILE, "exp/some_child_store/request_evidence.jsonl"}
    assert sorted(o for _, o in ac.owned_evidence_files(root.path, ac.EVIDENCE_FILE)) == [ac.OWNER_ATTEMPT_ROOT, ac.OWNER_CHILD_STORE]
    # the same id in both files: refused, never shadowed
    (child / ac.EVIDENCE_FILE).write_text((child / ac.EVIDENCE_FILE).read_text() + json.dumps([r for r in recs if r["extra"]["arm"] == "none"][0]) + "\n")
    with pytest.raises(ac.AuditRefusal, match="appears in two 'request_evidence.jsonl' files"):
        ac.read_request_evidence(root.path)
