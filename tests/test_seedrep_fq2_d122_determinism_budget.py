"""The Qwen seed-replicate battery: the determinism spot-check on a BUDGET-LIMITED smoke row selects min(20, budget).

The smokes run under an item budget (`--limit-goals 10` on the S3 bon row, `--limit-items 10` on misalign), so those rows'
stores file 10 items by construction -- and `determinism_check.select_items` REFUSES a store filing fewer than the 20 it selects: on
an earlier S3 bon smoke the check exited 2 with `the store files 10 item(s), fewer than the 20 the spot-check selects`, no record was
written, and S3's PASS (which gates on the determinism spot-check) could never derive at the smoke budget.

The rule: for a budget-limited smoke row the SERVED APP derives `--n min(20, <the row's item budget>)` from the row's own
`smoke.budget` (`seedrep_launch_plan.determinism_spot_check_n`) and names the budget with `--item-budget`; the record gains
`n_planned` (20), `n_selected`, `budget_limited` and `item_budget` (additive). A production row keeps `--n 20`, and the check still
refuses a shorter store -- that refusal is a real defect signal; the check never reads the store's size to shrink its selection.
Every reader of the record (the `S3` smoke derivation `_smoke_check`, `battery_tables.load_determinism` and its summary / markdown)
tolerates the keys absent and surfaces `budget_limited` where they are present.

Known-bad at 2174e5139: `determinism_check_argv` took no `n`, every leg's argv selected the default 20, and the record had none of
the four keys.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import battery_tables as bt  # noqa: E402
import determinism_check as dc  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_review_fixes as RF  # noqa: E402  (the certificate suite's smoke WORLD builder; its Test classes stay unbound here)
from test_seedrep_review_fixes import draft, inputs, srv  # noqa: E402,F401  (module fixtures register by import; `srv` = the app under a modal stub)

_row, _uuid4 = RF._row, RF._uuid4
S3_BON, S3_MISALIGN = "bon__s1__base__shard00", "misalign__s1__base__shard00"
SMOKE_WAVE = "11111111-2222-4333-8444-555555555555"
BUDGET_LINE = "budget_limited (D122): 10 of 20 planned items, the row's smoke item budget 10 (D107)"


def _load_battery_fixture():
    spec = importlib.util.spec_from_file_location("seedrep_battery_fixture_fq2", REPO / "tests" / "fixtures" / "seedrep_battery" / "build_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _faithful_post(raw: bytes, headers):
    body = json.loads(raw)
    item = body["messages"][0]["content"][len("prompt "):]
    return {"id": "chatcmpl-replay", "choices": [{"message": {"content": f"response for {item} under {body['model']}"}, "finish_reason": "stop"}]}


@pytest.fixture(autouse=True)
def _hooks(monkeypatch):
    monkeypatch.setattr(dc, "POST_FN", _faithful_post)
    monkeypatch.setattr(dc, "METRICS_FN", lambda: {"num_requests_running": 0, "num_requests_waiting": 0, "generation_tokens_total": 10})
    monkeypatch.setattr(dc, "TOKENIZE_FN", None)
    yield


def _rank(items):
    return sorted(items, key=lambda i: hashlib.sha256(i.encode()).hexdigest())


def _bon_root(tmp_path: Path, n_goals: int):
    """The synthetic BoN root (the S3 bon row's container leg) with ``n_goals`` filed goals -- 10 is what `--limit-goals 10` leaves."""
    root = fx.make_root(tmp_path / "stores", leg="bon", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=n_goals)
    return root, ac.read_attempt_root(root.path)


def _smoke(plan, standup_id, smoke_id, max_children=5):
    row = lp.plan_row_by_id(plan, standup_id)
    ident = {"smoke_id": smoke_id, "wave_uuid": SMOKE_WAVE, "W7": plan["W"][:7], "W": plan["W"]}
    return lp.smoke_row(row, ident, max_children=max_children, max_timeout_s=4 * 3600)


class _Parsed(Exception):
    """Raised by the injected `read_attempt_root`: argparse ACCEPTED the argv and `main` reached its first read."""


def _real_parser_accepts(argv_after_python_and_script):
    saved = ac.read_attempt_root

    def _first_read(path, **kw):
        raise _Parsed(str(path))
    ac.read_attempt_root = _first_read
    try:
        dc.main(list(argv_after_python_and_script))
    except _Parsed:
        return True
    except SystemExit as e:
        assert e.code == 2, e.code
        return False
    finally:
        ac.read_attempt_root = saved
    raise AssertionError("main returned without reading the attempt root")


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


# --------------------------------------------------------------------------- (1) the derivation and the argv the app builds ---------------- #

def test_planned_count_is_mirrored_and_a_row_derives_min_20_budget(draft):
    """The plan's 20 is ONE number spelled twice (the check imports no plan module); a budgeted S1-S4 BoN / misalign smoke row derives (10, 10),
    every other row (production, S5 full-shape, a smoke leg without a budget flag) the plan's (20, None); a budget above 20 caps at 20."""
    assert dc.N_PLANNED == lp.DETERMINISM_N_PLANNED == 20
    assert lp.DETERMINISM_BUDGET_KEYS == ("n_planned", "n_selected", "budget_limited", "item_budget")
    for sid in (S3_BON, S3_MISALIGN, "pasttense__s1__base__shard00", "controls__s1__base__shard00"):
        assert lp.determinism_spot_check_n(lp.plan_row_by_id(draft, sid)) == (20, None), sid           # production: never budgeted
    for smoke_id in ("S1", "S1b", "S2", "S3", "S4"):
        assert lp.determinism_spot_check_n(_smoke(draft, S3_BON, smoke_id)) == (10, 10) == (lp.SMOKE_BON_GOALS, lp.SMOKE_BON_GOALS)
        assert lp.determinism_spot_check_n(_smoke(draft, S3_MISALIGN, smoke_id)) == (10, 10) == (lp.SMOKE_MISALIGN_ITEMS, lp.SMOKE_MISALIGN_ITEMS)
    assert lp.determinism_spot_check_n(_smoke(draft, S3_BON, "S5")) == (20, None)                     # S5 keeps the FULL shape
    assert lp.determinism_spot_check_n(_smoke(draft, "pasttense__s1__base__shard00", "S3")) == (20, None)   # bounded by children, no budget flag
    assert lp.determinism_spot_check_n(_smoke(draft, "controls__s1__base__shard00", "S3")) == (20, None)
    # a budget at or above the plan's 20 is not a shorter selection -- min(20, budget), the budget still named
    wide = {"leg": "bon", "smoke": {"budget": {"kind": lp.SMOKE_BUDGET_KIND_ITEM, "bon_limit_goals": 30, "misalign_limit_items": 30}}}
    assert lp.determinism_spot_check_n(wide) == (20, 30) and lp.determinism_item_budget(wide) == 30
    assert lp.determinism_item_budget({"leg": "bon"}) is None and lp.determinism_item_budget({"leg": "bon", "smoke": {}}) is None


def test_argv_builder_carries_n_10_and_the_budget_on_a_budgeted_row_and_n_20_on_an_unbudgeted_one(srv, draft, tmp_path):
    """`determinism_check_argv`: `--n` always (the plan's 20 by default), `--item-budget` only when a budget is named; both forms parse on the
    check's REAL parser. Through `determinism_evidence` on REAL plan rows: the S3 bon smoke row's argv carries `--n 10 --item-budget 10`, the
    S3 misalign smoke row's both table legs do, the production bon row's `--n 20` with no budget flag, the S5 bon smoke row's likewise."""
    common = dict(driver_python="/venv/bin/python", tree=REPO, root=tmp_path / "root", proxy_url="http://127.0.0.1:18080/v1",
                  metrics_url="http://127.0.0.1:8000/metrics", out=tmp_path / "audits" / "determinism" / "bon.json")
    budgeted = srv.determinism_check_argv(table_leg="bon", n=10, item_budget=10, **common)
    plain = srv.determinism_check_argv(table_leg="bon", **common)
    assert _flag(budgeted, "--n") == "10" and _flag(budgeted, "--item-budget") == "10"
    assert _flag(plain, "--n") == "20" and "--item-budget" not in plain
    assert budgeted.count("--n") == 1 and plain.count("--n") == 1
    assert _real_parser_accepts(budgeted[2:]) is True and _real_parser_accepts(plain[2:]) is True
    # the same seam from the ROW, through the hook's helper (a recording `run`: nothing is launched)
    calls = {}

    def recording_run(argv, **kw):
        calls.setdefault(argv[argv.index("--root") + 1], []).append(list(argv))
        return types.SimpleNamespace(returncode=2)

    def evidence(row, name):
        root = tmp_path / name
        (root / "children").mkdir(parents=True)
        srv.determinism_evidence(root, tree=REPO, W="f" * 40, row=row, driver_python="py", env={}, proxy_url="u", metrics_url="m", run=recording_run)
        return calls[str(root)]

    bon_s3 = evidence(_smoke(draft, S3_BON, "S3"), "bon_s3")
    assert [(_flag(a, "--leg"), _flag(a, "--n"), _flag(a, "--item-budget")) for a in bon_s3] == [("bon", "10", "10")]
    mis_s3 = evidence(_smoke(draft, S3_MISALIGN, "S3"), "mis_s3")
    assert [(_flag(a, "--leg"), _flag(a, "--n"), _flag(a, "--item-budget")) for a in mis_s3] == [("misalign_leaking", "10", "10"), ("misalign_murder", "10", "10")]
    bon_prod = evidence(lp.plan_row_by_id(draft, S3_BON), "bon_prod")
    assert [(_flag(a, "--leg"), _flag(a, "--n")) for a in bon_prod] == [("bon", "20")] and all("--item-budget" not in a for a in bon_prod)
    bon_s5 = evidence(_smoke(draft, S3_BON, "S5"), "bon_s5")
    assert [(_flag(a, "--leg"), _flag(a, "--n")) for a in bon_s5] == [("bon", "20")] and all("--item-budget" not in a for a in bon_s5)
    ctl_s3 = evidence(_smoke(draft, "controls__s1__base__shard00", "S3"), "ctl_s3")
    assert [_flag(a, "--n") for a in ctl_s3] == ["20"] * 3 and all("--item-budget" not in a for a in ctl_s3)
    for argv in bon_s3 + mis_s3 + bon_prod + bon_s5 + ctl_s3:
        assert _real_parser_accepts(argv[2:]) is True, argv


# --------------------------------------------------------------------------- (2) select_items: all ten in rank order; 20 still refuses ------- #

def test_select_items_n_10_on_a_10_item_store_is_all_ten_in_rank_order_and_n_20_still_refuses_by_name(tmp_path):
    root, ar = _bon_root(tmp_path, n_goals=10)
    store = {s.arm: s for s in ac.discover_stores(ar, "bon")}["none"]
    goals = [f"g{k}" for k in range(10)]
    assert dc.select_items(store, 10) == _rank(goals) and len(set(dc.select_items(store, 10))) == 10
    with pytest.raises(ac.AuditRefusal, match=r"files 10 item\(s\), fewer than the 20 the spot-check selects"):
        dc.select_items(store, 20)
    # the production path end to end: no budget named -> the plan's 20 -> the same refusal (now a real defect signal)
    with pytest.raises(ac.AuditRefusal, match=r"files 10 item\(s\), fewer than the 20"):
        dc.run_check(ar, leg="bon", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None, allow_busy=False,
                     petri_behaviours=[])
    # the check NEVER shrinks its own selection to a named budget: --n above the budget refuses by name instead of selecting fewer
    with pytest.raises(ac.AuditRefusal, match=r"--n 20 exceeds the row's item budget 10 \(D122"):
        dc.run_check(ar, leg="bon", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None, allow_busy=False,
                     petri_behaviours=[], item_budget=10)


# --------------------------------------------------------------------------- (3) the record's additive keys, through the REAL main ---------- #

def test_the_record_carries_n_selected_n_planned_and_budget_limited(srv, tmp_path):
    """The app's budgeted argv on a 10-goal BoN store -> rc 0, the record says `n_planned 20 / n_selected 10 / budget_limited true / item_budget 10`
    (`n_items` stays the selection, for the older readers); the app's production argv on a 25-item store -> `20 / 20 / false / null`.
    A budget below 1, or `--n` above the budget, refuses (exit 2), nothing written."""
    root, ar = _bon_root(tmp_path, n_goals=10)
    rec_path = root.path / srv.DETERMINISM_COPY_REL / "bon.json"
    argv = srv.determinism_check_argv(driver_python=sys.executable, tree=REPO, table_leg="bon", root=root.path, proxy_url="http://127.0.0.1:18080/v1",
                                      metrics_url="http://127.0.0.1:8000/metrics", out=rec_path, copy_to=root.path / srv.DETERMINISM_OUT, n=10, item_budget=10)
    assert dc.main(argv[2:]) == ac.EXIT_OK and rec_path.is_file()
    rec = json.loads(rec_path.read_text())
    assert (rec["n_planned"], rec["n_selected"], rec["budget_limited"], rec["item_budget"]) == (20, 10, True, 10)
    assert rec["n_items"] == 10 and rec["arms"][0]["items"] == _rank([f"g{k}" for k in range(10)])
    assert rec["n_requests"] == rec["n"] == rec["identical"] == 20 and rec["reproduces"] is True      # 10 goals x 2 draws, every one byte-identical
    assert rec["leg"] == "bon" and rec["attempt_id"] == ar.attempt_id and rec["smoke_id"] is None      # the axes are the root's, as before
    assert (root.path / srv.DETERMINISM_OUT).read_bytes() == rec_path.read_bytes()
    for k in lp.DETERMINISM_RECORD_KEYS:                                                                # the earlier record keys are all still there
        assert k in rec, k
    # a production row: the app's default argv, the plan's 20 selected of 25 filed, not budget-limited
    prod = fx.make_root(tmp_path / "prod", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(prod, s=1, arms=("none",), n_items=25)
    prod_rec = prod.path / srv.DETERMINISM_COPY_REL / "pasttense.json"
    argv = srv.determinism_check_argv(driver_python=sys.executable, tree=REPO, table_leg="pasttense", root=prod.path, proxy_url="u", metrics_url="m", out=prod_rec)
    assert dc.main(argv[2:]) == ac.EXIT_OK
    prec = json.loads(prod_rec.read_text())
    assert (prec["n_planned"], prec["n_selected"], prec["budget_limited"], prec["item_budget"], prec["n_items"]) == (20, 20, False, None, 20)
    # refusals at the parser / the seam: nothing written
    out = tmp_path / "never.json"
    base = ["--root", str(root.path), "--leg", "bon", "--out", str(out)]
    assert ac.run_main(dc.main, base + ["--n", "10", "--item-budget", "0"]) == ac.EXIT_REFUSED
    assert ac.run_main(dc.main, base + ["--n", "20", "--item-budget", "10"]) == ac.EXIT_REFUSED
    assert ac.run_main(dc.main, base + ["--n", "20"]) == ac.EXIT_REFUSED                                 # the production form on a short store
    assert not out.exists()


# --------------------------------------------------------------------------- (4) the readers: S3's derivation and battery_tables ------------- #

class TestSmokeReaderAcceptsTheBudgetedRecord:
    """`_smoke_check("determinism", ...)` (S3's PASS derivation) accepts a budgeted record and surfaces its facts; a budget-limited record of a
    PRODUCTION row, or one whose counts disagree, refuses by name; the recorded facts are re-checked at certify."""

    def _world(self, draft, tmp_path):
        T = RF.TestSolR3_2_launch_certificate()
        rs = tmp_path / "RS"
        wave = _uuid4(7)
        wp = lp.smoke_wave_path(rs, "S3")
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_text(json.dumps({"smoke_id": "S3", "wave_uuid": wave}))
        rid = _row(draft, "bon")["standup_id"]
        accepted, texts = T._smoke_world(draft, draft, rs, {"S3": wave}, {"S3": [rid]})
        getter = lp.RecordingGetter(lp.smoke_identity(draft, "S3", wave)["dicts"]["accepted"], accepted["S3"].get)
        return rs, rid, texts, getter, wave

    @staticmethod
    def _det_path(texts):
        return [p for p in texts if p.endswith("/audits/determinism/bon.json")][0]

    @staticmethod
    def _rewrite(texts, path, **changes):
        """The bon record with ``changes`` applied AND re-bound by sha under modal_evidence.json["determinism"]["bon"] (the app's binding)."""
        root = path[: -len("/audits/determinism/bon.json")]
        doc = {**json.loads(texts[path]), **changes}
        texts[path] = json.dumps(doc)
        ev = json.loads(texts[f"{root}/modal_evidence.json"])
        ev[lp.DETERMINISM_EVIDENCE_KEY]["bon"]["sha256"] = lp._sha_text(texts[path])
        texts[f"{root}/modal_evidence.json"] = json.dumps(ev)
        return doc

    def _budgeted(self, texts, **more):
        p = self._det_path(texts)
        doc = json.loads(texts[p])
        arms = [{**a, "items": [f"g{k}" for k in range(10)]} for a in doc["arms"]]
        return self._rewrite(texts, p, **{"arms": arms, "n_items": 10, "n_planned": 20, "n_selected": 10, "budget_limited": True, "item_budget": 10, **more})

    def _derive(self, draft, rs, rid, texts, getter):
        return lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id="S3", standup_ids=[rid], accepted_get=getter,
                                    read_text=lp.volume_only(lambda p_: texts.get(str(p_))), derived_utc="2026-09-18T00:00:00Z")

    def _problems(self, rec, draft, texts, getter, wave):
        return lp.smoke_pass_problems(rec, smoke_id="S3", plan=draft, execution_wave_uuid=wave, read_text=lp.volume_only(lambda p_: texts.get(str(p_))),
                                      accepted_get=getter)

    def test_a_budgeted_record_derives_s3_with_its_facts_surfaced_and_a_w18_record_still_reads(self, draft, tmp_path):
        rs, rid, texts, getter, wave = self._world(draft, tmp_path)
        rec0 = self._derive(draft, rs, rid, texts, getter)                    # the world's own record: the older shape, none of the budget keys
        s0 = rec0["checks"]["determinism"]["sources"][0]
        assert s0["budget_limited"] is False and s0["n_planned"] is None and s0["n_selected"] is None and s0["item_budget"] is None and s0["qualification"] is None
        assert self._problems(rec0, draft, texts, getter, wave) == []
        self._budgeted(texts)
        rec = self._derive(draft, rs, rid, texts, getter)
        det = rec["checks"]["determinism"]
        assert det["ok"] is True and len(det["sources"]) == 1
        s = det["sources"][0]
        assert s["leg"] == "bon" and s["reproduces"] is True and s["identical"] == s["n_requests"] > 0 and s["smoke_id"] == "S3"
        assert (s["budget_limited"], s["n_planned"], s["n_selected"], s["item_budget"], s["n_items"]) == (True, 20, 10, 10, 10)
        assert s["qualification"] == BUDGET_LINE
        assert s["producer"] == {"writer": lp.PRODUCER_WRITERS["determinism"], "tree_sha": draft["W"], "rc": 0}
        assert self._problems(rec, draft, texts, getter, wave) == []

    @pytest.mark.parametrize("changes, why", [
        ({"smoke_id": None}, "a PRODUCTION row's spot-check is never budget-limited"),
        ({"n_selected": 12, "n_items": 12}, r"selected 12 of 20 planned under item budget 10 -- a budget-limited row selects min\(planned, budget\)"),
        ({"n_items": 20}, "records n_items 20 but n_selected 10"),
        ({"budget_limited": "yes"}, "records budget_limited 'yes', not a bool"),
        ({"item_budget": 7}, "records item_budget 7, not the 10 smoke S3 derives for the bon leg"),
        ({"item_budget": None}, "is budget_limited but records item_budget None"),
        ({"budget_limited": False}, "records item_budget 10 on a record that is not budget_limited"),
        ({"n_planned": 0}, "records n_planned 0 / n_selected 10 -- positive ints"),
    ])
    def test_an_inconsistent_or_production_budgeted_record_refuses_s3_by_name(self, draft, tmp_path, changes, why):
        rs, rid, texts, getter, _ = self._world(draft, tmp_path)
        self._budgeted(texts, **changes)
        with pytest.raises(ValueError, match=why):
            self._derive(draft, rs, rid, texts, getter)

    def test_partial_keys_refuse_and_a_budgeted_record_on_a_leg_without_a_budget_flag_refuses(self, draft, tmp_path):
        rs, rid, texts, getter, _ = self._world(draft, tmp_path)
        p = self._det_path(texts)
        self._rewrite(texts, p, n_selected=10)
        with pytest.raises(ValueError, match=r"carries \['n_selected'\] without \['n_planned', 'budget_limited', 'item_budget'\] -- D122's selection facts travel together"):
            self._derive(draft, rs, rid, texts, getter)
        # the smoke id names a smoke whose budget is not an item budget (S5: full shape) -> a budget-limited record cannot be its
        self._budgeted(texts, smoke_id="S5", standup_id=f"smoke-S5__{rid}")
        with pytest.raises(ValueError, match="is budget_limited but smoke S5 on the bon leg carries no item budget"):
            self._derive(draft, rs, rid, texts, getter)
        # `_determinism_budget_facts` on a leg that carries no budget flag (controls) refuses even under an item-budget smoke
        doc = {"leg": "xstest", "smoke_id": "S3", "n_items": 10, "n_planned": 20, "n_selected": 10, "budget_limited": True, "item_budget": 10}
        with pytest.raises(ValueError, match="smoke S3 on the controls leg carries no item budget"):
            lp._determinism_budget_facts("determinism", "x/audits/determinism/xstest.json", doc, row={"leg": "controls"})

    def test_facts_edited_after_derivation_fail_at_certify_by_name(self, draft, tmp_path):
        rs, rid, texts, getter, wave = self._world(draft, tmp_path)
        self._budgeted(texts)
        rec = self._derive(draft, rs, rid, texts, getter)
        assert self._problems(rec, draft, texts, getter, wave) == []

        def edited(**changes):
            det = rec["checks"]["determinism"]
            return {**rec, "checks": {**rec["checks"], "determinism": {**det, "sources": [{**det["sources"][0], **changes}]}}}

        assert any("a budget-limited row selects min(planned, budget)" in p for p in self._problems(edited(n_selected=12), draft, texts, getter, wave))
        assert any("the rendered line must restate the recorded fields" in p for p in self._problems(edited(qualification="budget_limited"), draft, texts, getter, wave))
        assert any("a production row's spot-check is never budget-limited (D122)" in p for p in self._problems(edited(smoke_id=None), draft, texts, getter, wave))


def _budgeted_record(leg: str, *, attempt_id: str, n_selected: int = 10, n_planned: int = 20, item_budget: int = 10, W: str = "2" * 40) -> dict:
    per = 2 * n_selected
    return {"leg": leg, "standup_id": f"smoke-S3__{leg}__s1__base__shard00", "smoke_id": "S3", "arms": [{"arm": "none", "items": [f"g{k}" for k in range(n_selected)],
                                                                                                     "n_requests": per, "identical": per}],
            "n_items": n_selected, "n_planned": n_planned, "n_selected": n_selected, "budget_limited": True, "item_budget": item_budget,
            "n_requests": per, "n": per, "identical": per, "reproduces": True, "misses": [], "W": W, "attempt_id": attempt_id}


def test_battery_tables_load_determinism_tolerates_and_surfaces_budget_limited(tmp_path):
    """`load_determinism`: a budgeted record reads with `budget_limited` true and its counts; an older record reads `budget_limited` false with the
    counts null; two shards of one leg COMBINE (counts sum when every record carries them, budget-limited if any shard was); the note is the
    qualification line beside `identical/n`."""
    d = tmp_path / "audits" / "determinism"
    d.mkdir(parents=True)
    (d / "bon__ta-1.json").write_text(json.dumps(_budgeted_record("bon", attempt_id="ta-1")))
    w18 = {"leg": "gsm8k", "arms": [{"arm": "none", "items": 20, "n_requests": 20, "identical": 17}], "n_items": 20, "n_requests": 20, "n": 20,
           "identical": 17, "reproduces": False, "misses": [], "W": "2" * 40, "attempt_id": "ta-2"}
    (d / "gsm8k__ta-2.json").write_text(json.dumps(w18))
    (d / "pasttense__ta-3.json").write_text(json.dumps(_budgeted_record("pasttense", attempt_id="ta-3")))
    (d / "pasttense__ta-4.json").write_text(json.dumps({**_budgeted_record("pasttense", attempt_id="ta-4", n_selected=20), "budget_limited": False, "item_budget": None,
                                                        "smoke_id": None, "standup_id": "pasttense__s1__base__shard01"}))
    det = bt.load_determinism(d)
    bon = det["bon"]
    assert bon["budget_limited"] is True and (bon["n_planned"], bon["n_selected"], bon["item_budget"], bon["n_items"]) == (20, 10, 10, 10)
    assert bon["records"][0]["budget_limited"] is True and bon["records"][0]["n_selected"] == 10 and bon["records"][0]["n_planned"] == 20
    assert bt.determinism_budget_note(bon) == "budget-limited: 10 of 20 planned items (item budget 10, D122)"
    g = det["gsm8k"]
    assert g["budget_limited"] is False and "n_planned" not in g and g["records"][0]["n_planned"] is None and g["records"][0]["budget_limited"] is False
    assert bt.determinism_budget_note(g) is None and bt.determinism_budget_note(None) is None
    pt = det["pasttense"]
    assert pt["combined_from"] == 2 and pt["budget_limited"] is True and pt["n_selected"] == 30 and pt["n_planned"] == 40 and pt["n_items"] == 30
    assert pt["n"] == 60 and pt["identical"] == 60 and pt["reproduces"] is True and pt["item_budget"] is None       # one shard named no budget
    assert bt.determinism_budget_note(pt) == "budget-limited: 30 of 40 planned items (item budget None, D122)"


def test_battery_tables_summary_and_markdown_surface_a_budgeted_record_and_keep_a_production_blocks_shape(tmp_path):
    """End to end on the battery fixture: the pasttense record rewritten budgeted -> the leg summary's `determinism` block gains the four keys and the
    markdown line carries the note; the gsm8k record (unbudgeted, older shape) keeps EXACTLY the block it had (additive only where it applies)."""
    fxb = _load_battery_fixture()
    root = tmp_path / "battery"
    fxb.build(root)
    p = root / "audits" / "determinism" / "pasttense.json"
    base = json.loads(p.read_text())
    p.write_text(json.dumps({**base, "arms": [{"arm": "none", "items": [f"g{k}" for k in range(10)], "n_requests": 10, "identical": 10}], "n_items": 10, "n_requests": 10,
                             "n": 10, "identical": 10, "reproduces": True, "n_planned": 20, "n_selected": 10, "budget_limited": True, "item_budget": 10, "smoke_id": "S3"}))
    inp = bt.load_inputs(root)
    summary = bt.run(inp, root / "tables_out")
    assert summary["legs"]["pasttense"]["determinism"] == {"n": 10, "identical": 10, "n_items": 10, "reproduces": True, "budget_limited": True,
                                                            "n_planned": 20, "n_selected": 10, "item_budget": 10}
    assert summary["legs"]["gsm8k"]["determinism"] == {"n": 20, "identical": 17, "n_items": 20, "reproduces": False}
    md = bt.render_md(summary)
    assert "determinism: 10/10 identical; budget-limited: 10 of 20 planned items (item budget 10, D122)" in md
    assert "determinism: 17/20 identical\n" in md
