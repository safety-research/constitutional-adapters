"""The request-evidence audit's QUALIFICATION (`reconciliation_contract`, `ledger`,
`ledger_source`, `attempt_source`, `evidence_basis`, `body_digest_check`, `independence_note`) is CONSUMED, VALIDATED, PRESERVED and
RENDERED by both downstream consumers of a verdict:

* the launch certificate (`seedrep_launch_plan._smoke_check` / `_smoke_source_facts_problems` / `smoke_pass_problems`): the facts copied
  into `checks.request_evidence_audit.sources[*]` carry the qualification verbatim plus its rendered line (`contract <version> (...)`),
  a check-level tally (ledger-backed vs ledger-less), and ONE consistency rule applied at derivation and again on the recorded facts at
  certify -- a verdict without a contract version, under an unknown version, or with a contradictory (contract, ledger, attempt_source,
  body digest) tuple is refused BY NAME, never defaulted to the ledger-backed contract.
* the analysis (`battery_tables._contract`): `contract_status` is `ok` ONLY under contract version `/1`, `ok_qualified (...)` under the
  qualified version; a missing / contradictory qualification is `unqualified_verdict (...)` with `contract_verdict = "unqualified"`
  (REFUSED-class: the row is not admitted and never counts as AGREES); new columns `contract_version` / `ledger` / `attempt_source` /
  `independence_lost`; the per-leg summary, the markdown and the figure evidence line state how many arm-runs were ledger-backed vs
  ledger-less.

Known-bad at 0b1783ad1: `_smoke_check` copied only the old fields (a ledger-less qualified AGREES certified exactly like a
ledger-backed one; a pre-fz3 verdict passed), `_contract` folded every AGREES to `ok`.
Synthetic data only (placeholder ids, `TESTHOST`)."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "modal"))
sys.path.insert(0, str(REPO / "tests"))
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_review_fixes as RF  # noqa: E402  (the certificate suite's smoke WORLD builder; its Test class stays unbound here)
from test_seedrep_review_fixes import draft, inputs, srv  # noqa: E402,F401  (module fixtures register by import)
import audit_common as ac  # noqa: E402
import battery_tables as bt  # noqa: E402
import battery_figures as bf  # noqa: E402

_row, _uuid4 = RF._row, RF._uuid4
FIX = REPO / "tests" / "fixtures" / "seedrep_battery"

#: the producer's spellings (audit_common) -- literals here so this module runs unchanged at the known-bad tip
LEDGER_CONTRACT = "seedrep_request_evidence_reconciliation/1"
LEDGERLESS_CONTRACT = "seedrep_request_evidence_reconciliation/2-D106"
BODY_OK = "usage ledger body_sha256 == client record body_sha256"
BODY_NA = "not_available: the ledger is not on this row's path, so the client's body digest has no independent counterpart (D106 -- a revised, qualified contract, not equivalent evidence)"
NOTE_LL = "D106: the engine log is the only server-side record; rejected / pre-engine attempts are corroborated by the client's response hook alone, never independently"
QUAL_KEYS = ("reconciliation_contract", "ledger", "ledger_source", "attempt_source", "evidence_basis", "body_digest_check", "independence_note")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = _load("seedrep_battery_fixture_fg2", FIX / "build_fixture.py")


def ledgerless_q() -> dict:
    """fz3's `audit_root` stamp on a `not_on_path` row (every non-GSM8K leg: `RoutingProxy` writes no ledger)."""
    return {"reconciliation_contract": LEDGERLESS_CONTRACT, "ledger": "not_on_path", "ledger_source": "plan: no child of this row declares a ledger path; none on the volume",
            "attempt_source": "response_evidence", "evidence_basis": ["request_evidence.jsonl", "response_evidence.jsonl", "engine log (Received request)"],
            "body_digest_check": BODY_NA, "independence_note": NOTE_LL}


def ledgered_q() -> dict:
    """fz3's stamp on a `declared` row (the GSM8K children)."""
    return {"reconciliation_contract": LEDGER_CONTRACT, "ledger": "declared", "ledger_source": "plan: children[].ledger.ledger_path_rel",
            "attempt_source": "ledger", "evidence_basis": ["request_evidence.jsonl", "usage_ledger.jsonl", "response_evidence.jsonl", "engine log (Received request)"],
            "body_digest_check": BODY_OK, "independence_note": None}


# --------------------------------------------------------------------------- the certificate ------------------------------------------------ #

class TestCertificate:
    """`_smoke_check("request_evidence_audit", ...)` + `_smoke_source_facts_problems` + `smoke_pass_problems` on an S3 (bon) world."""

    def _world(self, draft, tmp_path, legs=("bon",)):
        T = RF.TestSolR3_2_launch_certificate()
        rs = tmp_path / "RS"
        wave = _uuid4(7)
        wp = lp.smoke_wave_path(rs, "S3")
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_text(json.dumps({"smoke_id": "S3", "wave_uuid": wave}))
        rids = [_row(draft, leg)["standup_id"] for leg in legs]
        accepted, texts = T._smoke_world(draft, draft, rs, {"S3": wave}, {"S3": rids})
        getter = lp.RecordingGetter(lp.smoke_identity(draft, "S3", wave)["dicts"]["accepted"], accepted["S3"].get)
        ident = lp.smoke_identity(draft, "S3", wave)
        roots = {}
        for rid in rids:
            row = lp.plan_row_by_id(draft, rid)
            aid = accepted["S3"][lp.accepted_key(ident["W7"], wave, row["leg"], row["engine"], row["stage"], row["shard"])]
            roots[rid] = lp.attempt_root_for(ident["results_root"], leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=aid, kind=row.get("kind"))
        return {"rs": rs, "rids": rids, "texts": texts, "getter": getter, "wave": wave, "roots": roots, "draft": draft}

    @staticmethod
    def _verdict_paths(w, rid):
        row = lp.plan_row_by_id(w["draft"], rid)
        return [p for p, _k, _leg in lp.smoke_audit_sources(row, w["roots"][rid])]

    @staticmethod
    def _stamp(w, path, q, *, drop=()):
        """Rewrite the producer's verdict at ``path`` with qualification ``q`` (minus ``drop``) and RE-BIND its sha in modal_evidence.json --
        exactly what the producer would have written; the binding check never fires, the qualification rule decides."""
        root = path.split("/audits/", 1)[0]
        rel = path[len(root) + 1:]
        doc = json.loads(w["texts"][path])
        for k in QUAL_KEYS:
            doc.pop(k, None)
        doc.update({k: v for k, v in q.items() if k not in drop})
        text = json.dumps(doc)
        w["texts"][path] = text
        ev = json.loads(w["texts"][f"{root}/modal_evidence.json"])
        ev[lp.PRODUCER_EVIDENCE_KEY][rel]["sha256"] = lp._sha_text(text)
        w["texts"][f"{root}/modal_evidence.json"] = json.dumps(ev)

    def _stamp_all(self, w, q, **kw):
        for rid in w["rids"]:
            for p in self._verdict_paths(w, rid):
                self._stamp(w, p, q, **kw)

    def _derive(self, w):
        return lp.derive_smoke_pass(rs=w["rs"], plan=w["draft"], smoke_id="S3", standup_ids=w["rids"], accepted_get=w["getter"],
                                    read_text=lp.volume_only(lambda p_: w["texts"].get(str(p_))), derived_utc="2026-09-18T00:00:00Z")

    def _revalidate(self, w, rec, *, read=True):
        kw = dict(smoke_id="S3", plan=w["draft"], execution_wave_uuid=w["wave"])
        if read:
            kw.update(read_text=lp.volume_only(lambda p_: w["texts"].get(str(p_))), accepted_get=w["getter"])
        return lp.smoke_pass_problems(rec, **kw)

    def test_constants_mirror_the_producer(self):
        assert lp.AUDIT_CONTRACT_LEDGER == ac.RECONCILIATION_CONTRACT_LEDGER == LEDGER_CONTRACT
        assert lp.AUDIT_CONTRACT_LEDGERLESS == ac.RECONCILIATION_CONTRACT_LEDGERLESS == LEDGERLESS_CONTRACT
        assert lp.AUDIT_LEDGER_NOT_ON_PATH == ac.LEDGER_NOT_ON_PATH and set(lp.AUDIT_LEDGER_BACKED) == {ac.LEDGER_DECLARED, ac.LEDGER_PRESENT_UNDECLARED}
        assert set(lp.AUDIT_LEDGER_BACKED) | {lp.AUDIT_LEDGER_NOT_ON_PATH} == set(ac.LEDGER_STATUSES)
        assert tuple(lp.AUDIT_QUALIFICATION_KEYS) == QUAL_KEYS
        # the analysis and the figure spell the same contract
        assert bt.CONTRACT_LEDGER == LEDGER_CONTRACT and bt.CONTRACT_LEDGERLESS == LEDGERLESS_CONTRACT == bf.LEDGERLESS_CONTRACT
        assert set(bt.CONTRACT_LEDGERS[bt.CONTRACT_LEDGER]) == set(lp.AUDIT_LEDGER_BACKED) and bt.CONTRACT_LEDGERS[bt.CONTRACT_LEDGERLESS] == (lp.AUDIT_LEDGER_NOT_ON_PATH,)

    def test_ledgerless_facts_carry_the_qualification_and_revalidate(self, draft, tmp_path):
        w = self._world(draft, tmp_path)
        self._stamp_all(w, ledgerless_q())
        rec = self._derive(w)
        srcs = rec["checks"]["request_evidence_audit"]["sources"]
        assert len(srcs) == len(self._verdict_paths(w, w["rids"][0])) >= 1
        for s in srcs:
            assert s["verdict"] == "AGREES"
            assert {k: s[k] for k in QUAL_KEYS} == ledgerless_q()                    # copied VERBATIM, evidence basis included
            assert s["independence_lost"] is True
            assert s["qualification"] == (f"contract {LEDGERLESS_CONTRACT} (ledger not on path; attempts from response evidence; body digest not available; "
                                          "rejected / pre-engine attempts not independently corroborated)")
        tally = rec["checks"]["request_evidence_audit"]["reconciliation"]
        assert tally["n_verdicts"] == len(srcs) and tally["n_ledger_less"] == len(srcs) and tally["n_ledger_backed"] == 0
        assert tally["contracts"] == [LEDGERLESS_CONTRACT] and LEDGERLESS_CONTRACT in tally["statement"] and "ledger-less" in tally["statement"]
        assert "bare AGREES" not in json.dumps(rec) and all("contract " in s["qualification"] for s in srcs)
        # the certificate re-validates clean: on the recorded facts alone, on a JSON round trip, and re-run against the volume
        assert self._revalidate(w, rec, read=False) == []
        assert self._revalidate(w, json.loads(json.dumps(rec))) == []

    def test_ledger_backed_facts(self, draft, tmp_path):
        w = self._world(draft, tmp_path)
        self._stamp_all(w, ledgered_q())
        rec = self._derive(w)
        srcs = rec["checks"]["request_evidence_audit"]["sources"]
        for s in srcs:
            assert {k: s[k] for k in QUAL_KEYS} == ledgered_q() and s["independence_lost"] is False
            assert s["qualification"] == f"contract {LEDGER_CONTRACT} (ledger declared; attempts from ledger; body digest checked against the ledger)"
        tally = rec["checks"]["request_evidence_audit"]["reconciliation"]
        assert tally["n_ledger_backed"] == len(srcs) and tally["n_ledger_less"] == 0 and "ledger-backed" in tally["statement"]
        assert self._revalidate(w, rec) == []

    @pytest.mark.parametrize("what, q, drop, match", [
        ("pre-fz3 shape (no qualification at all)", {}, QUAL_KEYS, r"carries no D106 qualification .*reconciliation_contract.*never defaulted"),
        ("contract version missing", ledgerless_q(), ("reconciliation_contract",), r"carries no D106 qualification \['reconciliation_contract'\]"),
        ("unknown contract version", {**ledgerless_q(), "reconciliation_contract": "seedrep_request_evidence_reconciliation/3"}, (),
         r"issued under reconciliation contract 'seedrep_request_evidence_reconciliation/3', not one of"),
        ("/2-D106 with attempts from the ledger", {**ledgerless_q(), "attempt_source": "ledger"}, (), r"/2-D106 with attempt_source 'ledger'.*contradictory"),
        ("/2-D106 with a declared ledger", {**ledgerless_q(), "ledger": "declared"}, (), r"/2-D106 with ledger 'declared'.*contradictory"),
        ("/1 with the ledger not on path", {**ledgered_q(), "ledger": "not_on_path"}, (), r"/1 with ledger 'not_on_path'.*contradictory"),
        ("/1 with attempts from response evidence", {**ledgered_q(), "attempt_source": "response_evidence"}, (), r"/1 with attempt_source 'response_evidence'.*contradictory"),
        ("/2-D106 claiming a body digest check", {**ledgerless_q(), "body_digest_check": BODY_OK}, (), r"/2-D106 with body_digest_check .*no body digest counterpart"),
        ("/1 with the body digest not available", {**ledgered_q(), "body_digest_check": BODY_NA}, (), r"/1 with body_digest_check .*checks the body digest"),
        ("/2-D106 without an independence note", {**ledgerless_q(), "independence_note": None}, (), r"/2-D106 with no independence_note"),
        ("empty evidence basis", {**ledgerless_q(), "evidence_basis": []}, (), r"evidence_basis \[\] is not a non-empty list"),
    ])
    def test_derivation_refuses_a_missing_unknown_or_contradictory_qualification_by_name(self, draft, tmp_path, what, q, drop, match):
        w = self._world(draft, tmp_path)
        self._stamp_all(w, ledgerless_q())
        victim = self._verdict_paths(w, w["rids"][0])[0]
        self._stamp(w, victim, q, drop=drop)
        with pytest.raises(ValueError, match=match):
            self._derive(w)

    def test_revalidation_refuses_tampered_or_stale_recorded_facts_by_name(self, draft, tmp_path):
        """The re-derive-on-certify idiom: the RECORDED facts are held to the same rule (no volume needed), and against the volume the
        recomputed facts must equal them -- so a fact edited after derivation (a `/2-D106` source relabelled ledger-backed, its contract version
        deleted, its `independence_lost` cleared, its rendered line reduced to a bare AGREES, the tally zeroed) fails at certify."""
        w = self._world(draft, tmp_path)
        self._stamp_all(w, ledgerless_q())
        good = self._derive(w)
        assert self._revalidate(w, good) == []

        def edit(fn):
            rec = json.loads(json.dumps(good))
            fn(rec["checks"]["request_evidence_audit"])
            return rec

        def src0(c):
            return c["sources"][0]

        cases = [
            (lambda c: src0(c).update(ledger="declared"), r"/2-D106 with ledger 'declared'.*contradictory"),
            (lambda c: src0(c).update(attempt_source="ledger"), r"/2-D106 with attempt_source 'ledger'"),
            (lambda c: src0(c).pop("reconciliation_contract"), r"carries no D106 qualification \['reconciliation_contract'\]"),
            (lambda c: src0(c).update(reconciliation_contract=LEDGER_CONTRACT, ledger="declared", attempt_source="ledger"), r"/1 with body_digest_check"),
            (lambda c: src0(c).update(independence_lost=False), r"records independence_lost False under contract .*2-D106"),
            (lambda c: src0(c).update(qualification="AGREES"), r"renders its qualification as 'AGREES', not 'contract seedrep_request_evidence_reconciliation/2-D106"),
            (lambda c: c["reconciliation"].update(n_ledger_less=0, n_ledger_backed=c["reconciliation"]["n_verdicts"]), r"records reconciliation .*not the tally of its sources"),
            (lambda c: c.pop("reconciliation"), r"records reconciliation None, not the tally of its sources"),
        ]
        for fn, match in cases:
            bad = edit(fn)
            probs = self._revalidate(w, bad, read=False)
            assert any(__import__("re").search(match, p) for p in probs), (match, probs)
            probs = self._revalidate(w, bad)
            assert any(__import__("re").search(match, p) for p in probs), (match, probs)
        # the recomputed-facts comparison ALSO fires on a value drift (the tampered source disagrees with the fresh derivation)
        bad = edit(lambda c: src0(c).update(ledger_source="edited after derivation"))
        probs = self._revalidate(w, bad)
        assert any("recomputed facts" in p and "differ from the recorded" in p for p in probs), probs

    def test_a_pre_fz3_record_of_facts_is_refused_at_certify(self, draft, tmp_path):
        """A PASS record whose facts predate the qualification, re-validated under the current rule: refused by name, never read as ledger-backed."""
        w = self._world(draft, tmp_path)
        self._stamp_all(w, ledgerless_q())
        good = self._derive(w)
        old = json.loads(json.dumps(good))
        for s in old["checks"]["request_evidence_audit"]["sources"]:
            for k in QUAL_KEYS + ("independence_lost", "qualification"):
                s.pop(k, None)
        old["checks"]["request_evidence_audit"].pop("reconciliation", None)
        probs = self._revalidate(w, old, read=False)
        assert any("carries no D106 qualification" in p for p in probs) and any("not the tally of its sources" in p for p in probs), probs

    def test_the_app_binds_the_verdict_by_sha_only_no_new_key_is_needed(self):
        """`seedrep_served.producer_evidence` binds each verdict file (sha256 / writer / tree sha / rc) and reads none of its fields -- the
        qualification travels INSIDE the producer's file the certificate reads; the app needs no new key (fg2 report, read-only)."""
        src = (REPO / "scripts/modal/seedrep_served.py").read_text()
        body = src.split("def producer_evidence(", 1)[1].split("\ndef ", 1)[0]
        # F-AO (fp3): the per-child binding moved into `bind_audit_verdicts` / `_bind_child_verdict` -- one verdict per (child, table leg) through
        # `lp.audit_verdict_rels`, the FILE resolved through the audit's own index; the binder reads the verdict's IDENTITY stamp (`arm_binding`,
        # `arm`) to hold the file to the plan child, never its qualification or its verdict field
        binder = src.split("def _bind_child_verdict(", 1)[1].split("\ndef producer_evidence(", 1)[0]
        assert "bind_audit_verdicts(root, row=row, W=W, rc_a=rc_a" in body and "lp.audit_verdict_rels(" in binder and "sha256_file(p)" in binder
        for text in (body, binder):
            assert "reconciliation_contract" not in text and '"verdict"' not in text


# --------------------------------------------------------------------------- the analysis --------------------------------------------------- #

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("fg2_battery")
    fx.build(root)
    summary = bt.run(bt.load_inputs(root), root / "tables_out")
    return {"root": root, "summary": summary, "out": root / "tables_out"}


def _rows(summary, leg):
    return {(r["arm"], r["seed"]): r for r in summary["legs"][leg]["rows"]}


def _rebuild(tmp_path, name, rows):
    root = fx.build(tmp_path / name)
    fx._write_csv(root / "audits/request_log_summary.csv", rows)
    return root


def _patched(rows, where, **changes):
    n = len(where)
    return [dict(r, **changes) if (r["leg"], r["arm"], r["s"], r.get("stage"))[:n] == where else r for r in rows]


class TestAnalysis:
    def test_fixture_carries_both_contracts(self):
        rows = fx.request_log_rows()
        by = {r["reconciliation_contract"] for r in rows}
        assert by == {LEDGER_CONTRACT, LEDGERLESS_CONTRACT}
        assert all((r["ledger"], r["attempt_source"]) == ("declared", "ledger") for r in rows if r["leg"] == "gsm8k")
        assert all((r["ledger"], r["attempt_source"]) == ("not_on_path", "response_evidence") for r in rows if r["leg"] != "gsm8k")

    def test_ok_only_under_the_ledger_contract_ok_qualified_under_d106(self, run):
        s = run["summary"]
        g = _rows(s, "gsm8k")[("none", 2)]
        assert g["contract_verdict"] == "AGREES" and g["contract_status"] == "ok" and g["contract_problem"] is None
        assert (g["contract_version"], g["ledger"], g["attempt_source"], g["independence_lost"]) == (LEDGER_CONTRACT, "declared", "ledger", False)
        p = _rows(s, "pasttense")[("none", 1)]
        assert p["contract_verdict"] == "AGREES" and p["admissible"] is True
        assert p["contract_status"].startswith("ok_qualified (D106: ledger not on path; attempts from response evidence; body digest not available")
        assert LEDGERLESS_CONTRACT in p["contract_status"] and p["contract_problem"] is None
        assert (p["contract_version"], p["ledger"], p["attempt_source"], p["independence_lost"]) == (LEDGERLESS_CONTRACT, "not_on_path", "response_evidence", True)
        # the hole rows keep their statuses; the columns are None where no verdict exists
        hole = _rows(s, "bon")[("constv2ctl2-full__lora_r4_ce", 1)]
        assert hole["contract_verdict"] == "unaudited" and hole["contract_version"] is None and hole["independence_lost"] is None
        # per-leg + total tallies
        L = s["legs"]
        assert L["pasttense"]["n_audit_agrees"] == 27 and L["pasttense"]["n_audit_ledgerless"] == 27 and L["pasttense"]["n_audit_ledger_backed"] == 0
        assert L["gsm8k"]["n_audit_agrees"] == 26 and L["gsm8k"]["n_audit_ledger_backed"] == 26           # HOLE_READING: (DIFF, 3) has no request-log row
        assert L["gsm8k"]["n_audit_ledgerless"] == 0 and L["gsm8k"]["n_audit_unqualified"] == 0
        assert L["bon"]["n_audit_agrees"] == 26 and L["bon"]["n_audit_ledgerless"] == 26
        ev = s["evidence"]
        assert ev["n_audit_ledger_backed"] == 26 and ev["n_audit_ledgerless"] == ev["n_audit_agrees"] - 26 and ev["n_audit_unqualified"] == 0

    def test_columns_are_declared_and_written(self, run):
        cols = list(bt.ROW_COLUMNS)
        i = cols.index("contract_problem")
        assert cols[i + 1:i + 5] == ["contract_version", "ledger", "attempt_source", "independence_lost"]
        assert bt.LEG_SCHEMA["independence_lost"] == "bool" and bt.LEG_SCHEMA["contract_version"] == "string"
        with (run["out"] / "pasttense.csv").open(newline="") as fh:
            hdr = csv.DictReader(fh).fieldnames
        assert list(hdr) == cols
        import pandas as pd
        df = pd.read_parquet(run["out"] / "pasttense.parquet")
        assert str(df["independence_lost"].dtype) == "boolean" and df["independence_lost"].all() and (df["contract_version"] == LEDGERLESS_CONTRACT).all()

    @pytest.mark.parametrize("what, changes, match", [
        ("no contract version", {"reconciliation_contract": None}, "no reconciliation_contract"),
        ("unknown version", {"reconciliation_contract": "seedrep_request_evidence_reconciliation/3"}, "unknown reconciliation contract"),
        ("/2-D106 with attempts from the ledger", {"attempt_source": "ledger"}, "requires 'response_evidence'"),
        ("/1 with the ledger not on path", {"reconciliation_contract": LEDGER_CONTRACT, "attempt_source": "ledger"}, "with ledger 'not_on_path'"),
    ])
    def test_an_unqualified_agrees_is_refused_class(self, tmp_path, what, changes, match):
        rows = _patched(fx.request_log_rows(), ("pasttense", "none", 1), **changes)          # both attempt rows of the arm-run
        root = _rebuild(tmp_path, "unq", rows)
        s = bt.run(bt.load_inputs(root), root / "out", legs=(bt.LEG_BY_ID["pasttense"],))
        r = _rows(s, "pasttense")[("none", 1)]
        assert r["contract_verdict"] == "unqualified" and r["contract_status"].startswith("unqualified_verdict (")
        assert match in r["contract_status"] and match in r["contract_problem"] and "AGREES without a valid D106 qualification" in r["contract_problem"]
        assert r["admissible"] is False and r["admissibility"].startswith("audit: unqualified") and r["value"] is None and r["readable_value"] is not None
        leg = s["legs"]["pasttense"]
        assert leg["n_audit_agrees"] == 26 and leg["n_audit_unqualified"] == 1 and leg["n_admissible"] == 25      # 26 in the fixture (HOLE_AT_CAP), minus this one
        assert _rows(s, "pasttense")[("none", 2)]["contract_status"].startswith("ok_qualified")

    def test_mixed_contracts_across_an_arm_runs_attempts_are_qualified_not_clean(self, tmp_path):
        rows = _patched(fx.request_log_rows(), ("pasttense", "none", 1, "base"), **ledgered_q_csv())
        root = _rebuild(tmp_path, "mixed", rows)
        s = bt.run(bt.load_inputs(root), root / "out", legs=(bt.LEG_BY_ID["pasttense"],))
        r = _rows(s, "pasttense")[("none", 1)]
        assert r["contract_verdict"] == "AGREES" and r["contract_status"].startswith("ok_qualified") and "1/2" in r["contract_status"]
        assert r["contract_version"] == f"{LEDGER_CONTRACT} | {LEDGERLESS_CONTRACT}" and r["independence_lost"] is True

    def test_decode_column_absence_composes_with_the_qualified_status(self, tmp_path):
        rows = [{k: v for k, v in r.items() if k not in bt.REQUEST_LOG_DECODE and k != "seed_base"} for r in fx.request_log_rows()]
        root = _rebuild(tmp_path, "nodecode", rows)
        s = bt.run(bt.load_inputs(root), root / "out", legs=(bt.LEG_BY_ID["pasttense"], bt.LEG_BY_ID["gsm8k"]))
        p = _rows(s, "pasttense")[("none", 1)]
        assert p["contract_status"].startswith("ok_qualified (D106") and p["contract_status"].endswith(
            "); decode column(s) absent from the summary: temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, seed_base")
        g = _rows(s, "gsm8k")[("none", 1)]
        assert g["contract_status"] == "ok; decode column(s) absent from the summary: temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, seed_base"

    def test_markdown_and_figure_evidence_state_the_qualification(self, run):
        s = run["summary"]
        md = bt.render_md(s)
        assert LEDGERLESS_CONTRACT in md and "ledger-less" in md and "ledger-backed" in md
        # the per-leg header counts ledger-backed vs ledger-less; a qualified AGREES never renders as a bare AGREES in the row table
        pt = md.split("## Past-tense", 1)[1].split("\n## ", 1)[0] if "## Past-tense" in md else md.split("(`pasttense`)", 1)[1].split("\n## ", 1)[0]
        assert "ledger-less 27" in pt and "| AGREES [D106 ledger-less] |" in pt and "| AGREES |" not in pt
        gs = md.split("(`gsm8k`)", 1)[1].split("\n## ", 1)[0]
        assert "ledger-backed 26" in gs and "| AGREES |" in gs and "[D106 ledger-less]" not in gs
        assert "Criterion 2" in md and "never reported as ledger-backed" in md
        # the figure's evidence line (the assertion in the title) carries the qualification wherever an AGREES was ledger-less
        assert bf.evidence_line(s["legs"]["pasttense"]) == ("26/27 arm-runs admitted (27 readable) · vendor preset, native serving (audit AGREES on every arm-run; "
                                                             f"D106: 27 ledger-less under {LEDGERLESS_CONTRACT}, body digest not available) · engine identity PASS")
        assert bf.evidence_line(s["legs"]["gsm8k"]).split(" · ")[1] == "decode contract UNVERIFIED on 1/27 arm-runs (audit AGREES 26/27; 26 ledger-backed)"
        assert bf.evidence_line(dict(s["legs"]["gsm8k"], n_audit_agrees=27, n_audit_ledger_backed=27)).split(" · ")[1] == (
            "vendor preset, native serving (audit AGREES on every arm-run; 27 ledger-backed)")
        assert f"(audit AGREES 26/27; D106: 26 ledger-less under {LEDGERLESS_CONTRACT})" in bf.evidence_line(s["legs"]["bon"])
        # an unqualified verdict is named in the evidence line, never counted as AGREES
        leg = dict(s["legs"]["pasttense"], n_audit_agrees=26, n_audit_ledgerless=26, n_audit_unqualified=1)
        assert "decode contract UNVERIFIED on 1/27 arm-runs (audit AGREES 26/27; 1 unqualified; D106: 26 ledger-less" in bf.evidence_line(leg)


def ledgered_q_csv() -> dict:
    """The three SUMMARY columns of a ledger-backed row (the CSV carries only these of the qualification)."""
    return {"ledger": "declared", "attempt_source": "ledger", "reconciliation_contract": LEDGER_CONTRACT}
