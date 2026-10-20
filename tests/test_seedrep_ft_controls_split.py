"""F-T + F-U's finding on S3 -- the launch plan composer (`scripts/modal/seedrep_launch_plan.py`).

F-T: previously every controls child -- the BASE row's too -- carried ``--controls xstest,or_bench_hard_1k,mmlu --controls-cap <cap>``, and
`served_capability_run.py` refused each at once: MMLU is a teacher-forced one-token letter readout with NO cap ladder
(`SAMPLED_UNAFFECTED_CONTROLS`), and ``--controls-cap`` is the driver's RUNG flag (a cap above the base, paired with
``--escalate-from-root``), so the base row's argv was doubly refused (mmlu on "a rung", then "both or neither"). Evidence:
the S3 controls s1 x5 and S2 controls-arditi x3 children (`f_l_evidence/w16_*controls*/children/*.log`). The driver's refusal is RIGHT
(mmlu on the base row only; the at-cap table: `not_applicable`; the coverage check expects no rung stage of it), so the
fix is at the source: the composer splits the controls set by leg class -- the base row carries all three with no rung flag; a rung row
carries the ladder controls alone with ``--controls-cap <cap>``, serves no MMLU (no `--max-logprobs`, no `mmlu` eval leg), and the audit /
determinism table legs of a rung are the two ladder legs.

S3: previously ``SMOKE_CHECKS["S3"]`` was the sampler + audit + pool snapshot -- that the determinism spot-check REPRODUCES never
gated S3's PASS. Now a `determinism` check reads the spot-check's record per table leg, bound under
`modal_evidence.json["determinism"]` the way the app writes it, and a missing / non-reproducing / other-row record refuses S3.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "modal"))
sys.path.insert(0, str(REPO / "tests"))
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_review_fixes as RF  # noqa: E402  (the certificate suite's smoke WORLD builder; its Test class stays unbound here so pytest does not re-collect it)
from test_seedrep_review_fixes import draft, inputs, srv  # noqa: E402,F401  (the module fixtures register here by import; `srv` = the app under a modal stub)

_row, _uuid4 = RF._row, RF._uuid4

CONTROLS_ROWS = 4                     # 3 views + Arditi
CONTROLS_RUNGS = 3                    # 512 -> 2,048 -> 8,192 -> 16,384


def _controls_rows(plan, kind):
    return [r for r in plan["standups"] + plan["rungs"] if r["leg"] == "controls" and r["kind"] == kind and not r.get("dose_standup")]


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


# --------------------------------------------------------------------------- the composed shape ------------------------------------------ #

def test_base_controls_children_carry_all_three_controls_and_no_rung_flag(draft):
    rows = _controls_rows(draft, "base")
    assert len(rows) == CONTROLS_ROWS
    for r in rows:
        assert "mmlu" in r["eval_legs"] and _flag(r["argv"], "--max-logprobs") == str(lp.MMLU_TOP_LOGPROBS)
        assert r["children"], r["standup_id"]
        for k in r["children"]:
            a = k["argv"]
            assert _flag(a, "--controls") == "xstest,or_bench_hard_1k,mmlu" == ",".join(lp.CONTROLS_ALL)
            assert "--controls-cap" not in a, "the base row never carries the driver's RUNG flag"
            assert "--escalate-from-root" not in a
            assert _flag(a, "--top-logprobs") == str(lp.MMLU_TOP_LOGPROBS)
            assert k["cap"] == lp.CONTROLS_BASE_CAP == 512 and k["out_rel"] == f"exp/sampled/controls_cap512__s{k['s']}"
            assert set(k["task"]) == set(lp.CONTROLS_ALL) and k["controls"] == list(lp.CONTROLS_ALL)
            assert k["task"]["mmlu"] == "s2parity/mmlu_letter"


def test_rung_controls_children_carry_the_ladder_controls_alone_with_the_rung_flag(draft):
    rows = _controls_rows(draft, "rung")
    assert len(rows) == CONTROLS_ROWS * CONTROLS_RUNGS
    assert sorted({int(r["stage"]) for r in rows}) == [2048, 8192, 16384]
    for r in rows:
        # a rung engine serves no MMLU: no mmlu eval leg, no --max-logprobs on its standup, a readiness contract that asks none
        assert r["eval_legs"] == ["xstest", "or_bench_hard"] and "mmlu" not in r["eval_legs"]
        assert "--max-logprobs" not in r["argv"] and r["readiness"]["max_logprobs"] is None
        assert r["rung_mode"] == lp.RUNG_MODE_CONSUME and r["rung_consumer"]      # the app spawns the CONSUMER on a rung, not these children
        for k in r["children"]:
            a = k["argv"]
            assert _flag(a, "--controls") == "xstest,or_bench_hard_1k" == ",".join(lp.CONTROLS_LADDER)
            assert "mmlu" not in _flag(a, "--controls").split(",")
            assert _flag(a, "--controls-cap") == r["stage"] == str(k["cap"])
            assert "--top-logprobs" not in a
            assert set(k["task"]) == set(lp.CONTROLS_LADDER) and k["controls"] == list(lp.CONTROLS_LADDER)
            assert k["out_rel"] == f"exp/sampled/controls_cap{k['cap']}__s{k['s']}"


def test_dose_controls_base_rows_are_base_rows_too(draft):
    for r in draft["standups"]:
        if r["leg"] == "controls" and r.get("dose_standup"):
            for k in r["children"]:
                assert _flag(k["argv"], "--controls") == "xstest" and "--controls-cap" not in k["argv"] and "--top-logprobs" not in k["argv"]


def test_controls_for_cap_splits_by_leg_class():
    assert lp.controls_for_cap(lp.CONTROLS_ALL, cap=512) == lp.CONTROLS_ALL
    assert lp.controls_for_cap(lp.CONTROLS_ALL, cap=2048) == lp.CONTROLS_LADDER
    assert lp.controls_for_cap(("xstest",), cap=8192) == ("xstest",)
    with pytest.raises(ValueError, match="none of .* rides the cap ladder"):
        lp.controls_for_cap(("mmlu",), cap=2048)
    assert set(lp.CONTROLS_LADDER) | set(lp.CONTROLS_FIXED_CAP) == set(lp.CONTROLS_ALL)


def test_audit_and_determinism_table_legs_follow_the_stage(draft, srv):
    assert lp.audit_table_legs("controls") == lp.CONTROLS_ALL == lp.audit_table_legs("controls", stage="base")
    assert lp.audit_table_legs("controls", stage="2048") == lp.CONTROLS_LADDER == lp.audit_table_legs("controls", stage=16384)
    assert lp.audit_table_legs("bon", stage="2048") == ("bon",)
    base, rung = _controls_rows(draft, "base")[0], _controls_rows(draft, "rung")[0]
    assert [leg for leg, _ in lp.audit_verdict_rels("controls", "none", 1, stage=base["stage"], shard=0)] == list(lp.CONTROLS_ALL)
    assert [leg for leg, _ in lp.audit_verdict_rels("controls", "none", 1, stage=rung["stage"], shard=0)] == list(lp.CONTROLS_LADDER)
    assert lp.determinism_table_legs_of_row(base) == lp.CONTROLS_ALL and lp.determinism_table_legs_of_row(rung) == lp.CONTROLS_LADDER
    assert lp.determinism_rels(rung) == [("xstest", "audits/determinism/xstest.json"), ("or_bench_hard_1k", "audits/determinism/or_bench_hard_1k.json")]
    # the app's mapping is the composer's (the app imported under the suite's modal stub)
    assert srv.determinism_table_legs("controls") == lp.CONTROLS_ALL and srv.determinism_table_legs("controls", stage="8192") == lp.CONTROLS_LADDER
    assert srv.DETERMINISM_COPY_REL == lp.DETERMINISM_OUT_REL and srv.DETERMINISM_REL == lp.PRODUCER_WRITERS["determinism"]
    assert srv.determinism_table_legs("misalign") == lp.EVAL_LEGS_OF["misalign"]


# --------------------------------------------------------------------------- the driver's real validation --------------------------------- #

def _driver_validate(argvs):
    """Every controls child's argv (the tokens after `python script`) through the REAL driver in a fresh interpreter (it binds `sampled_controls`
    at import and refuses a second bind in-process): `build_parser().parse_args`, `parse_controls(ns.controls)`, then the driver's own rung
    refusal condition (served_capability_run.py: `--controls-cap` set AND a `SAMPLED_UNAFFECTED_CONTROLS` control among the legacy controls ->
    `[served-cap] --controls [...] on a rung: a teacher-forced one-token readout has no cap ladder; refusing`), plus the driver's constants."""
    code = (
        "import json, sys, io, contextlib\n"
        "sys.path.insert(0, sys.argv[1] + '/src'); sys.path.insert(0, sys.argv[1] + '/scripts')\n"
        "import served_capability_run as R\n"
        "P = R.build_parser(); out = []\n"
        "for argv in json.load(sys.stdin):\n"
        "    buf = io.StringIO()\n"
        "    try:\n"
        "        with contextlib.redirect_stderr(buf):\n"
        "            ns = P.parse_args(argv)\n"
        "            controls = R.parse_controls(ns.controls)\n"
        "        legacy = [c for c in controls if c in R.SERVED_CONTROLS]\n"
        "        no_ladder = [c for c in legacy if c in R.SAMPLED_UNAFFECTED_CONTROLS]\n"
        "        refused = ns.controls_cap is not None and bool(no_ladder)\n"
        "        out.append({'ok': True, 'controls': controls, 'controls_cap': ns.controls_cap, 'refused_on_rung': refused, 'no_ladder': no_ladder,\n"
        "                    'top_logprobs': ns.top_logprobs, 'escalate_from_root': ns.escalate_from_root})\n"
        "    except SystemExit as e:\n"
        "        out.append({'ok': False, 'error': (buf.getvalue().strip()[-600:] or str(e))})\n"
        "out.append({'SAMPLED_CONTROLS': list(R.SAMPLED_CONTROLS), 'SAMPLED_UNAFFECTED_CONTROLS': list(R.SAMPLED_UNAFFECTED_CONTROLS),\n"
        "            'CONTROLS_CAP_LADDER': list(R.CONTROLS_CAP_LADDER), 'MMLU_TOP_LOGPROBS': R.MMLU_TOP_LOGPROBS})\n"
        "json.dump(out, sys.stdout)\n")
    r = subprocess.run([sys.executable, "-c", code, str(REPO)], input=json.dumps([list(a) for a in argvs]), capture_output=True, text=True, timeout=600, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


def test_every_composed_controls_child_passes_the_drivers_parser_and_rung_refusal_and_w18s_argv_does_not(draft):
    kids = [(r, k) for r in draft["standups"] + draft["rungs"] if r["leg"] == "controls" for k in r["children"]]
    assert kids
    ctx = lp.SpawnContext(attempt_root="/workspace/attempt", attempt_id="ta-1", standup_id="x", plans_root="/vol/p", proxy_url="http://127.0.0.1:9/v1",
                          routing_table="/t.json", driver_python="/py", client_python="/cpy", S=52, ncells=8, bank_view="/b", base_env={})
    resolved = [lp.resolve_spawn(k, ctx)[0][2:] for _, k in kids]
    # the known-bad, verbatim from f_l_evidence/w16_S3_controls_s1/children/*.argv.json -- the BASE row's child with the rung flag
    w18_base = ["generate", "--proxy-url", "http://127.0.0.1:34441/v1", "--routing-table", "/workspace/attempt/exp/serve/routing_table.json",
                "--cell", "none", "--config", "configs/sdfbase_full22_qwen35_4b_s1_sampled_t0.yaml", "--defenses-dir", "/b",
                "--out-root", "/workspace/attempt/exp/sampled/controls_cap512__s1", "--controls", "xstest,or_bench_hard_1k,mmlu", "--concurrency", "8",
                "--top-logprobs", "64", "--trial", "0", "--controls-cap", "512"]
    w18_rung = list(w18_base); w18_rung[-1] = "2048"
    res = _driver_validate(resolved + [w18_base, w18_rung])
    consts, res = res[-1], res[:-1]
    # the composer's leg classes ARE the driver's
    assert consts["SAMPLED_CONTROLS"] == list(lp.CONTROLS_LADDER) and consts["SAMPLED_UNAFFECTED_CONTROLS"] == list(lp.CONTROLS_FIXED_CAP)
    assert consts["CONTROLS_CAP_LADDER"][0] == lp.CONTROLS_BASE_CAP and consts["MMLU_TOP_LOGPROBS"] == lp.MMLU_TOP_LOGPROBS
    for (r, k), v in zip(kids, res):
        assert v["ok"], (r["standup_id"], k["arm"], v)
        assert v["refused_on_rung"] is False, (r["standup_id"], k["arm"], v)
        assert v["controls"] == k["controls"]
        if r["kind"] == "rung":
            assert v["controls_cap"] == int(r["stage"]) and v["controls_cap"] in consts["CONTROLS_CAP_LADDER"][1:] and "mmlu" not in v["controls"]
        else:
            assert v["controls_cap"] is None and v["top_logprobs"] == 64 and "mmlu" in v["controls"]
    for bad in res[len(kids):]:
        assert bad["ok"] and bad["refused_on_rung"] is True and bad["no_ladder"] == ["mmlu"], bad   # parses, then the driver refuses by name


# --------------------------------------------------------------------------- the PLAN gate --------------------------------------------------- #

def test_plan_gate_refuses_a_regressed_controls_split(draft):
    assert lp.validate_plan(draft) == []
    bad = copy.deepcopy(draft)
    r = _row(bad, "controls", "s1", stage="2048")
    k = r["children"][0]
    k["argv"][k["argv"].index("--controls") + 1] = "xstest,or_bench_hard_1k,mmlu"
    k["task"]["mmlu"] = lp.CONTROLS_TASKS["mmlu"]; k["controls"] = list(lp.CONTROLS_ALL)
    probs = lp.validate_plan(bad)
    assert any("controls RUNG child" in p and "refuses ['mmlu'] on a rung" in p for p in probs), probs
    bad = copy.deepcopy(draft)
    r = _row(bad, "controls", "s1")
    r["children"][0]["argv"] += ["--controls-cap", "512"]
    probs = lp.validate_plan(bad)
    assert any("BASE controls child" in p and "--controls-cap" in p for p in probs), probs
    bad = copy.deepcopy(draft)
    r = _row(bad, "controls", "arditi", stage="8192")
    r["eval_legs"] = ["xstest", "or_bench_hard", "mmlu"]
    assert any("a controls RUNG serves no MMLU" in p for p in lp.validate_plan(bad))
    bad = copy.deepcopy(draft)
    k = _row(bad, "controls", "s2")["children"][1]
    k["argv"][k["argv"].index("--controls") + 1] = "xstest,or_bench_hard_1k"
    k["task"].pop("mmlu"); k["controls"] = list(lp.CONTROLS_LADDER)
    assert any("MMLU is measured once per (arm, s), on the base row" in p for p in lp.validate_plan(bad))


def test_ledger_and_control_keys_unchanged_by_the_split(draft):
    led = lp.capacity_ledger(draft)
    assert (led["unstaged_peak"]["containers"], led["unstaged_peak"]["gpus"]) == (134, 134)      # one H200 per pool
    rows = lp._plan_rows_the_app_may_execute(draft)
    keys = {r["control_key"] for r in rows}
    assert len(keys) == len(rows) == 130
    assert sum(1 for r in draft["rungs"] if r["leg"] == "controls") == 12 and sum(1 for r in draft["standups"] if r["leg"] == "controls" and not r.get("dose_standup")) == 4


# --------------------------------------------------------------------------- S3 gates on determinism -------------------------------------- #

class TestS3Determinism:
    """`SMOKE_CHECKS["S3"]` carries `determinism`: the spot-check's record per table leg, bound under `modal_evidence.json["determinism"]`."""

    def _world(self, draft, tmp_path, legs=("bon",)):
        """The S3 smoke world for the named legs' base rows (the certificate suite's builder: producer files at their canonical paths under each
        accepted attempt root, bound in modal_evidence.json). Returns the FIRST row's id as `rid` (the single-row tests) plus every row id."""
        T = RF.TestSolR3_2_launch_certificate()
        rs = tmp_path / "RS"
        waves = {"S3": _uuid4(7)}
        wp = lp.smoke_wave_path(rs, "S3")
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_text(json.dumps({"smoke_id": "S3", "wave_uuid": waves["S3"]}))
        rids = [_row(draft, leg)["standup_id"] for leg in legs]
        accepted, texts = T._smoke_world(draft, draft, rs, waves, {"S3": rids})
        getter = lp.RecordingGetter(lp.smoke_identity(draft, "S3", waves["S3"])["dicts"]["accepted"], accepted["S3"].get)
        self._rids = rids
        return rs, rids[0], texts, getter, waves["S3"]

    def _derive(self, draft, rs, rid, texts, getter):
        rids = getattr(self, "_rids", None) or [rid]
        return lp.derive_smoke_pass(rs=rs, plan=draft, smoke_id="S3", standup_ids=rids, accepted_get=getter, read_text=lp.volume_only(lambda p_: texts.get(str(p_))),
                                    derived_utc="2026-09-18T00:00:00Z")

    def _revalidate(self, rec, draft, texts, getter, wave):
        return lp.smoke_pass_problems(rec, smoke_id="S3", plan=draft, execution_wave_uuid=wave, read_text=lp.volume_only(lambda p_: texts.get(str(p_))), accepted_get=getter)

    @staticmethod
    def _keep(rec, name, pred):
        """The PASS record with check ``name``'s sources filtered to ``pred`` -- an edited / truncated record."""
        return {**rec, "checks": {**rec["checks"], name: {**rec["checks"][name], "sources": [s for s in rec["checks"][name]["sources"] if pred(s)]}}}

    # --- the INVENTORY rule: the record must LIST every accepted row's producer paths, exactly once each ------------------------------------ #

    def test_revalidation_refuses_an_omitted_determinism_or_audit_leg_within_a_controls_row(self, draft, tmp_path):
        rs, rid, texts, getter, wave = self._world(draft, tmp_path, legs=("controls",))
        rec = self._derive(draft, rs, rid, texts, getter)
        det = rec["checks"]["determinism"]["sources"]
        assert [s["leg"] for s in det] == list(lp.CONTROLS_ALL) and len(det) == 3          # the base row: three table legs
        assert self._revalidate(rec, draft, texts, getter, wave) == []
        # reproduction: keep only the xstest source, delete the other two records from the volume -> previously re-validated clean
        cut = self._keep(rec, "determinism", lambda s: s["leg"] == "xstest")
        for s in det[1:]:
            texts.pop(s["path"])
        probs = self._revalidate(cut, draft, texts, getter, wave)
        assert any("check determinism lists 1 source(s) but the accepted rows' producers own 3" in p and "or_bench_hard_1k.json" in p and "mmlu.json" in p for p in probs), probs
        with pytest.raises(ValueError, match="check 'determinism': the producer's .*audits/determinism/or_bench_hard_1k.json is absent"):
            self._derive(draft, rs, rid, texts, getter)                                     # the fresh derivation always refused it
        # the same pattern on the audit: one controls child's or_bench_hard_1k verdict dropped from the record (the file may even stay on the volume)
        aud = rec["checks"]["request_evidence_audit"]["sources"]
        victim = next(s for s in aud if s["leg"] == "or_bench_hard_1k")
        cut = self._keep(rec, "request_evidence_audit", lambda s: s["path"] != victim["path"])
        probs = self._revalidate(cut, draft, texts, getter, wave)
        assert any(f"check request_evidence_audit lists {len(aud) - 1} source(s) but the accepted rows' producers own {len(aud)}" in p for p in probs), probs

    def test_revalidation_refuses_a_whole_rows_set_omitted_across_rows(self, draft, tmp_path):
        rs, rid, texts, getter, wave = self._world(draft, tmp_path, legs=("controls", "bon"))
        rec = self._derive(draft, rs, rid, texts, getter)
        assert len(rec["rows"]) == 2 and len(rec["checks"]["determinism"]["sources"]) == 4 and len(rec["checks"]["resolved_sampler"]["sources"]) == 2
        assert self._revalidate(rec, draft, texts, getter, wave) == []
        bon_root = rec["rows"][self._rids[1]]["attempt_root"]
        # the bon row's whole determinism set dropped (its row and root facts still listed) -> refused by name
        cut = self._keep(rec, "determinism", lambda s: not s["path"].startswith(bon_root + "/"))
        probs = self._revalidate(cut, draft, texts, getter, wave)
        assert any("check determinism lists 3 source(s) but the accepted rows' producers own 4" in p and "bon.json" in p for p in probs), probs
        # the same rule on the sampler: the bon row's resolved_sampler.json dropped from the record
        cut = self._keep(rec, "resolved_sampler", lambda s: not s["path"].startswith(bon_root + "/"))
        probs = self._revalidate(cut, draft, texts, getter, wave)
        assert any("check resolved_sampler lists 1 source(s) but the accepted rows' producers own 2" in p for p in probs), probs

    def test_revalidation_refuses_a_duplicated_or_foreign_path_and_the_valid_record_passes(self, draft, tmp_path):
        rs, rid, texts, getter, wave = self._world(draft, tmp_path, legs=("controls",))
        rec = self._derive(draft, rs, rid, texts, getter)
        assert self._revalidate(rec, draft, texts, getter, wave) == []
        det = rec["checks"]["determinism"]["sources"]
        dup = {**rec, "checks": {**rec["checks"], "determinism": {**rec["checks"]["determinism"], "sources": det + [dict(det[0])]}}}
        probs = self._revalidate(dup, draft, texts, getter, wave)
        assert any("check determinism lists 1 path(s) more often than the accepted rows' producers own them" in p and "xstest.json" in p and "2, 1" in p for p in probs), probs
        # a path no accepted row's producer owns (another root's record) is refused as such, on top of the per-source root check
        foreign = {**det[0], "path": det[0]["path"].replace("/audits/determinism/xstest.json", "/audits/determinism/petri.json")}
        ext = {**rec, "checks": {**rec["checks"], "determinism": {**rec["checks"]["determinism"], "sources": det + [foreign]}}}
        probs = self._revalidate(ext, draft, texts, getter, wave)
        assert any("check determinism lists 1 path(s) no accepted row's producer owns" in p for p in probs), probs

    def _det_path(self, texts):
        return [p for p in texts if p.endswith("/audits/determinism/bon.json")][0]

    def test_s3_requires_the_spot_check_and_a_good_world_passes(self, draft, tmp_path):
        assert "determinism" in lp.SMOKE_CHECKS["S3"] and "determinism" in lp.SMOKE_CHECK_NAMES and lp.PRODUCER_WRITERS["determinism"].endswith("determinism_check.py")
        assert "determinism" not in lp.SMOKE_CHECKS["S1"] and "determinism" not in lp.SMOKE_CHECKS["S4"]
        rs, rid, texts, getter, wave = self._world(draft, tmp_path)
        rec = self._derive(draft, rs, rid, texts, getter)
        det = rec["checks"]["determinism"]
        assert det["ok"] is True and len(det["sources"]) == 1
        s = det["sources"][0]
        assert s["path"] == self._det_path(texts) and s["leg"] == "bon" and s["reproduces"] is True and s["identical"] == s["n_requests"] > 0
        assert s["producer"] == {"writer": lp.PRODUCER_WRITERS["determinism"], "tree_sha": draft["W"], "rc": 0}
        assert s["standup_id"] == f"smoke-S3__{rid}" and s["smoke_id"] == "S3"
        read_text = lp.volume_only(lambda p_: texts.get(str(p_)))
        assert lp.smoke_pass_problems(rec, smoke_id="S3", plan=draft, execution_wave_uuid=wave, read_text=read_text, accepted_get=getter) == []
        # a PASS record that lacks the check (the old derivation) is refused by name
        w18 = {**rec, "checks": {k: v for k, v in rec["checks"].items() if k != "determinism"}}
        assert any("required check 'determinism' is missing" in p for p in lp.smoke_pass_problems(w18, smoke_id="S3", plan=draft, execution_wave_uuid=wave))

    def test_a_missing_record_refuses_s3(self, draft, tmp_path):
        rs, rid, texts, getter, _ = self._world(draft, tmp_path)
        texts.pop(self._det_path(texts))
        with pytest.raises(ValueError, match="check 'determinism': the producer's .*audits/determinism/bon.json is absent"):
            self._derive(draft, rs, rid, texts, getter)

    def test_a_non_reproducing_record_refuses_s3(self, draft, tmp_path):
        rs, rid, texts, getter, _ = self._world(draft, tmp_path)
        p = self._det_path(texts)
        root = p[: -len("/audits/determinism/bon.json")]
        doc = json.loads(texts[p])
        doc.update(identical=doc["n_requests"] - 1, reproduces=False, misses=[{"arm": "none", "item": "i1"}])
        texts[p] = json.dumps(doc)
        ev = json.loads(texts[f"{root}/modal_evidence.json"])
        ev[lp.DETERMINISM_EVIDENCE_KEY]["bon"]["sha256"] = lp._sha_text(texts[p])          # the producer bound THESE bytes -- the content refuses
        texts[f"{root}/modal_evidence.json"] = json.dumps(ev)
        with pytest.raises(ValueError, match="criterion 3 requires every one"):
            self._derive(draft, rs, rid, texts, getter)
        # a record the producer did not bind (rewritten since) refuses on the binding, before any verdict is read
        texts[p] = json.dumps({**doc, "identical": doc["n_requests"], "reproduces": True, "misses": []})
        with pytest.raises(ValueError, match="not the bytes the producer bound"):
            self._derive(draft, rs, rid, texts, getter)

    def test_a_record_of_another_row_or_zero_requests_refuses_s3(self, draft, tmp_path):
        rs, rid, texts, getter, _ = self._world(draft, tmp_path)
        p = self._det_path(texts)
        root = p[: -len("/audits/determinism/bon.json")]

        def rebind(doc):
            texts[p] = json.dumps(doc)
            ev = json.loads(texts[f"{root}/modal_evidence.json"])
            ev[lp.DETERMINISM_EVIDENCE_KEY]["bon"]["sha256"] = lp._sha_text(texts[p])
            texts[f"{root}/modal_evidence.json"] = json.dumps(ev)
        good = json.loads(texts[p])
        rebind({**good, "standup_id": "smoke-S3__bon__s2__base__shard00"})
        with pytest.raises(ValueError, match="is the record of row"):
            self._derive(draft, rs, rid, texts, getter)
        rebind({**good, "leg": "pasttense"})
        with pytest.raises(ValueError, match="not this row's table leg 'bon'"):
            self._derive(draft, rs, rid, texts, getter)
        rebind({**good, "n_requests": 0, "n": 0, "identical": 0, "reproduces": True})
        with pytest.raises(ValueError, match="nothing re-sent proves nothing"):
            self._derive(draft, rs, rid, texts, getter)
        # the app's binding ABSENT by name (the tool never ran / was not on the tree) refuses before the record is read
        ev = json.loads(texts[f"{root}/modal_evidence.json"])
        ev[lp.DETERMINISM_EVIDENCE_KEY]["bon"] = {"absent": True, "why": "determinism_check.py rc 2", "writer": lp.PRODUCER_WRITERS["determinism"], "tree_sha": draft["W"], "rc": 2}
        texts[f"{root}/modal_evidence.json"] = json.dumps(ev)
        rebind_doc = good
        texts[p] = json.dumps(rebind_doc)
        with pytest.raises(ValueError, match="the producer bound no audits/determinism/bon.json"):
            self._derive(draft, rs, rid, texts, getter)
