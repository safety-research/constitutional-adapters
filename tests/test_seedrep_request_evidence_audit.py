"""The request evidence audit's positive proof set and its negatives, on synthetic stores of every leg's
REAL row schema (`tests/seedrep_audit_fixtures.py`): the production headers of EVERY leg through
`contracts.request_evidence_record` (non-null item_id, numeric sample_idx) and the FULL
`contracts.request_evidence_reconciliation` per leg against store-derived filed pairs → ADMITTED; the `gsm8k/none`
mislabel and BOTH `inspect/xstest` / `inspect/or_bench_hard_1k` labels → "filed item(s) have NO recorded attempt";
an omitted `top_k` → REFUSED; two arms sharing a seed each bind to their OWN attempt by request_id (a seed-only join is
ambiguous); a retry reconciles; a ledger row with no record, a row with no binding and an ambiguous engine id refuse.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tiers import judged, parity, petri_runner  # noqa: E402

NO_ATTEMPT = "filed item(s) have NO recorded attempt"


def _audit(root: fx.Root, legs, tmp_path: Path, *, thinking=None, client_status="absent", siblings=()):
    ar = ac.read_attempt_root(root.path)
    return rea.audit_root(ar, out_dir=tmp_path / "audits", legs=list(legs), policy_file=None,
                          thinking_expect=thinking or dict(ac.THINKING_EXPECT_DEFAULT), petri_behaviours=list(fx.PETRI_BEHAVIOURS),
                          model_key=ac.MODEL_KEY, client_status=client_status,
                          siblings=[ac.read_attempt_root(r.path) for r in siblings])


# --------------------------------------------------------------------------- #
# the canonical task per leg equals the store writer's own spelling            #
# --------------------------------------------------------------------------- #

def test_canonical_tasks_are_the_writers_own():
    assert ac.canonical_task("pasttense") == "static/pasttense"
    assert ac.canonical_task("bon") == "strongreject_bon"
    assert ac.canonical_task("crescendo") == "multi_turn/crescendo"
    assert ac.canonical_task("xstest") == judged.TASK_XSTEST == "judged/xstest_v2"
    assert ac.canonical_task("or_bench_hard_1k") == judged.TASK_ORBENCH == "judged/or_bench_hard_1k"
    assert ac.canonical_task("mmlu") == parity.PARITY_TASK_MMLU == "s2parity/mmlu_letter"
    assert ac.canonical_task("gsm8k") == C.inspect_task(C.GSM8K_PAIRED_TASK_NAME) == "inspect/gsm8k_paired200"
    assert ac.canonical_task("petri", "beh_a") == petri_runner.PETRI_TASK_PREFIX + "beh_a" == "petri/beh_a"
    assert ac.canonical_task("misalign_leaking") == "agentic/leaking"
    # the Inspect variants are NOT the sampled controls' tasks (rev 6)
    assert ac.canonical_task("xstest") != C.inspect_task("xstest")
    assert ac.canonical_task("or_bench_hard_1k") != C.inspect_task("or_bench_hard_1k")
    with pytest.raises(ac.AuditRefusal):
        ac.canonical_task("petri")
    assert len(ac.LEGS) == 15 and len(ac.CONTAINER_LEGS) == 12


def test_client_task_name_drift_check():
    class Agree:
        @staticmethod
        def store_task_name(leg, sub=None):
            return ac.canonical_task(leg, sub)

    class Drift:
        @staticmethod
        def store_task_name(leg, sub=None):
            return "gsm8k/none" if leg == "gsm8k" else ac.canonical_task(leg, sub)

    assert ac.check_client_task_names(Agree) == []
    bad = ac.check_client_task_names(Drift)
    assert len(bad) == 1 and bad[0].startswith("gsm8k:") and "'gsm8k/none'" in bad[0]


# --------------------------------------------------------------------------- #
# gate 2: every leg ADMITTED through the production headers + the full reconciliation                       #
# --------------------------------------------------------------------------- #

EXPECTED_TEMPERATURE = {**{l: 0.7 for l in ac.LEGS}, "gsm8k": 0.6, "mmlu": 0.0}


@pytest.mark.parametrize("leg", ac.LEGS)
def test_gate2_every_leg_agrees(tmp_path: Path, leg: str):
    cleg = ac.CONTAINER_LEG_OF[leg]
    root = fx.make_root(tmp_path / "stores", leg=cleg, view="s1")
    fx.populate_root(root, s=1, n_items=3)
    # the production headers parse: every TASK record has a non-null item_id and a numeric sample_idx
    recs = ac.read_request_evidence(root.path)  # every owned copy (a GSM8K row's are per child store, fz2)
    assert recs and all(r["kind"] == "task" and isinstance(r["item_id"], str) and r["item_id"] and isinstance(r["sample_idx"], int) for r in recs)
    assert all(r["request_evidence_version"] == C.REQUEST_EVIDENCE_VERSION for r in recs)
    verdicts = _audit(root, [leg], tmp_path)
    assert {v["arm"] for v in verdicts} == set(fx.DEFAULT_ARMS)
    for v in verdicts:
        assert v["verdict"] == "AGREES", (leg, v["arm"], v["problem"])
        assert v["problem"] is None and v["checks"]["reconciliation"] is None
        assert v["s"] == 1 and v["view"] == "s1" and v["stage"] == "base" and v["shard"] == "shard00"
        assert v["n_filed"] == (6 if leg == "petri" else 3), (leg, v["n_filed"])
        assert v["n_draws"] > 0 and v["n_draws_bound"] == v["n_draws"]
        assert v["n_records_task"] >= v["n_draws"]
        assert v["expected_sampling"]["temperature"] == EXPECTED_TEMPERATURE[leg]
        assert v["thinking_expect"] == ("true" if leg == "gsm8k" else "none")
        assert v["target_vmodel"] == f"petri__{v['arm']}"
        assert v["resolved_sampler_verdict"] == "ABSENT" and v["client_task_names"] == "absent"
        assert set(v["tasks"]) == set(ac.tasks_of_leg(leg, fx.PETRI_BEHAVIOURS))
        p = Path(v["path"])
        assert p == tmp_path / "audits" / "request_log" / leg / f"{v['arm']}__s1.json" and p.is_file()
        assert json.loads(p.read_text())["verdict"] == "AGREES"
    if leg in ac.STATIC_FAMILIES:
        assert all("static_sampling_seed" in v["seed_rule"] for v in verdicts)
    if leg == "gsm8k":
        assert all("inspect_replicate_seed" in v["seed_rule"] for v in verdicts)


def test_gate2_arditi_root_three_seed_bases(tmp_path: Path):
    """The Arditi engine serves ONE object under three client-side seed bases: three stores, three verdicts, s from
    each store's OWN recorded base (never the view)."""
    pin = "a" * 64
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="arditi", arms=(ac.ARDITI_CELL,), arditi_pin=pin)
    for s in (1, 2, 3):
        fx.write_static_store(root, family="pasttense", arm=ac.ARDITI_CELL, s=s, goals=["g0", "g1"])
    root.flush_pipeline()
    verdicts = _audit(root, ["pasttense"], tmp_path)
    assert sorted(v["s"] for v in verdicts) == [1, 2, 3]
    assert all(v["verdict"] == "AGREES" and v["arm"] == ac.ARDITI_CELL and v["view"] == "arditi" for v in verdicts)
    assert sorted(Path(v["path"]).name for v in verdicts) == ["arditi_allpos__s1.json", "arditi_allpos__s2.json", "arditi_allpos__s3.json"]


# --------------------------------------------------------------------------- #
# the KNOWN-BAD labels                                                           #
# --------------------------------------------------------------------------- #

def _relabel(root: fx.Root, old: str, new: str) -> None:
    for name in (ac.EVIDENCE_FILE, ac.LEDGER_FILE):
        for p in root.pipeline_files(name):  # every owned copy: a GSM8K row's are per child store (fz2), the others' at the root
            fx.rewrite_jsonl(p, lambda r: {**r, "task": new} if r.get("task") == old else r)


def test_gsm8k_none_mislabel_refused(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="gsm8k", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=3)
    assert all(v["verdict"] == "AGREES" for v in _audit(root, ["gsm8k"], tmp_path / "a"))
    _relabel(root, ac.GSM8K_TASK, "gsm8k/none")
    (v,) = _audit(root, ["gsm8k"], tmp_path / "b")
    assert v["verdict"] == "REFUSED"
    assert NO_ATTEMPT in v["checks"]["reconciliation"], v["checks"]["reconciliation"]
    assert "('inspect/gsm8k_paired200', 'item" in v["checks"]["reconciliation"]


@pytest.mark.parametrize("control,bad_label", [("xstest", "inspect/xstest"), ("or_bench_hard_1k", "inspect/or_bench_hard_1k")])
def test_controls_inspect_labels_refused(tmp_path: Path, control: str, bad_label: str):
    root = fx.make_root(tmp_path / "stores", leg="controls", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=3)
    good = {v["leg"]: v for v in _audit(root, ["xstest", "or_bench_hard_1k", "mmlu"], tmp_path / "a")}
    assert all(v["verdict"] == "AGREES" for v in good.values()), {k: v["problem"] for k, v in good.items()}
    _relabel(root, ac.CONTROLS_TASK_OF[control], bad_label)
    after = {v["leg"]: v for v in _audit(root, ["xstest", "or_bench_hard_1k", "mmlu"], tmp_path / "b")}
    assert after[control]["verdict"] == "REFUSED" and NO_ATTEMPT in after[control]["checks"]["reconciliation"]
    other = "or_bench_hard_1k" if control == "xstest" else "xstest"
    assert after[other]["verdict"] == "AGREES" and after["mmlu"]["verdict"] == "AGREES"


# --------------------------------------------------------------------------- #
# §9.1 step 5 negatives                                                          #
# --------------------------------------------------------------------------- #

def test_omitted_top_k_refused(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="renellm", view="s1", arms=("none",))
    fx.write_static_store(root, family="renellm", arm="none", s=1, goals=["g0", "g1"], omit_body_keys=("top_k",))
    root.flush_pipeline()
    (v,) = _audit(root, ["renellm"], tmp_path)
    assert v["verdict"] == "REFUSED"
    assert "top_k is None, not the frozen row's 20" in v["checks"]["reconciliation"]


def test_two_arms_sharing_a_seed_bind_by_request_id(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="codeattack", view="s1", arms=("none", "constv2-mix-full__psr_ce_all"))
    fx.populate_root(root, s=1, arms=("none", "constv2-mix-full__psr_ce_all"), n_items=2)
    recs = ac.read_jsonl(root.path / ac.EVIDENCE_FILE)
    seeds = {}
    for r in recs:
        seeds.setdefault(r["seed"], set()).add(r["vmodel"])
    # the static rule is cell-INDEPENDENT: every seed is shared by both arms → a seed-only join is ambiguous
    assert all(len(v) == 2 for v in seeds.values()) and len(seeds) == 4
    verdicts = _audit(root, ["codeattack"], tmp_path)
    assert {v["arm"]: v["verdict"] for v in verdicts} == {"none": "AGREES", "constv2-mix-full__psr_ce_all": "AGREES"}
    for v in verdicts:  # every row bound to its OWN attempt and engine line, none shared
        assert v["checks"]["row_binding"]["n_bound"] == v["n_draws"] == 4


def test_retry_reconciles_and_a_row_on_the_rejected_attempt_refuses(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="grok420", view="s1", arms=("none",))
    vm = root.vmodels["none"]
    seed = ac.static_sampling_seed(1, 0, "grok420", "g0", 0)
    rej_rid, rej_cid, _ = root.send(ac.static_task("grok420"), "g0", 0, vm, root.vendor_body(vm, "g0/0", 4096, seed), ok=False)
    assert rej_cid is None
    shard = fx.write_static_store(root, family="grok420", arm="none", s=1, goals=["g0", "g1"])
    root.flush_pipeline()
    (v,) = _audit(root, ["grok420"], tmp_path / "a")
    assert v["verdict"] == "AGREES", v["problem"]
    assert v["n_ledger_rows_partition"] == v["n_draws"] + 1 and v["n_draws_bound"] == 4
    # the filed row re-pointed at the REJECTED attempt: a binding failure by name
    fx.rewrite_jsonl(shard / "responses.jsonl", lambda r: {**r, "request_id": rej_rid} if (r["goal_id"], r["sample_idx"]) == ("g0", 0) else r)
    (v2,) = _audit(root, ["grok420"], tmp_path / "b")
    assert v2["verdict"] == "REFUSED" and "REJECTED ledger attempt" in v2["problem"]


def test_ledger_row_without_record_refuses(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=2)
    lp = root.path / ac.LEDGER_FILE
    rows = ac.read_jsonl(lp)
    orphan = {**rows[0], "request_id": "deadbeef" * 4}
    lp.write_text("".join(json.dumps(r) + "\n" for r in rows + [orphan]))
    (v,) = _audit(root, ["pasttense"], tmp_path)
    assert v["verdict"] == "REFUSED" and "have NO request-evidence record" in v["checks"]["reconciliation"]


def test_row_without_binding_refuses(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.write_static_store(root, family="pasttense", arm="none", s=1, goals=["g0"], bind_rows=False)
    root.flush_pipeline()
    (v,) = _audit(root, ["pasttense"], tmp_path)
    assert v["verdict"] == "REFUSED" and "carries no request_id / completion_id" in v["problem"]
    assert v["checks"]["row_binding"]["n_bound"] == 0 and v["checks"]["row_binding"]["n_problems"] == 2


def test_ambiguous_engine_id_refuses(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=1)
    log = root.serve / "vllm_server.log"
    cid = json.loads((root.path / ac.LEDGER_FILE).read_text().splitlines()[0])["completion_id"]
    log.write_text(log.read_text() + f"INFO Received request {cid}-deadbeef: params: SamplingParams(n=1), lora_request: None.\n")
    (v,) = _audit(root, ["pasttense"], tmp_path)
    assert v["verdict"] == "REFUSED" and "matches 2 engine-log" in v["problem"]


def test_engine_log_without_received_lines_refuses(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=1)
    (root.serve / "vllm_server.log").write_text("INFO server up\n")
    with pytest.raises(ac.AuditRefusal, match="no `Received request` line"):
        _audit(root, ["pasttense"], tmp_path)


def test_petri_seed_rule_unavailable_refuses_by_name(tmp_path: Path, monkeypatch):
    root = fx.make_root(tmp_path / "stores", leg="petri", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=2)
    monkeypatch.setattr(ac, "PETRI_SEED_RULE_OVERRIDE", None)
    # the seed rule IS on this tree; a tree without it is SIMULATED — `audit_common.petri_seed_rule`
    # reads `getattr(_petri, "petri_target_seed", None)` at call time, so removing the attribute is exactly that tree
    assert hasattr(petri_runner, "petri_target_seed")
    monkeypatch.delattr(petri_runner, "petri_target_seed")
    (v,) = _audit(root, ["petri"], tmp_path)
    assert v["verdict"] == "REFUSED" and "seed rule unavailable" in v["problem"] and "WS1-B" in v["problem"]
    assert v["checks"]["reconciliation"] == "not run (seed rule unavailable)"


def test_view_root_store_with_another_seed_base_refuses(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.write_static_store(root, family="pasttense", arm="none", s=2, goals=["g0"])
    root.flush_pipeline()
    with pytest.raises(ac.AuditRefusal, match="records seed_base 2 but sits on view s1"):
        _audit(root, ["pasttense"], tmp_path)


def test_store_vmodel_disagreeing_with_the_standup_refuses(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="bon", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=1)
    root.rewrite_endpoint(vmodels={"none": "petri__other"})
    (v,) = _audit(root, ["bon"], tmp_path)
    assert v["verdict"] == "REFUSED" and "records virtual_model 'petri__none'" in v["problem"]


# --------------------------------------------------------------------------- #
# the CLI: summary CSV, exit codes, refusal on missing inputs                    #
# --------------------------------------------------------------------------- #

def test_cli_summary_and_exit_codes(tmp_path: Path, capsys):
    stores = tmp_path / "stores"
    fx.build_wave(stores, legs=("pasttense",), views=("s1",), n_items=2)
    out = tmp_path / "audits"
    rc = ac.run_main(rea.main, ["--stores", str(stores), "--out", str(out), "--legs", "pasttense"])
    assert rc == 0
    with open(out / rea.SUMMARY_NAME, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and {r["verdict"] for r in rows} == {"AGREES"}
    assert set(rows[0]) == {c.rstrip("?") for c in rea.SUMMARY_COLUMNS}
    assert {r["arm"] for r in rows} == set(fx.DEFAULT_ARMS) and all(r["s"] == "1" and r["shard"] == "shard00" for r in rows)
    assert all(r["problem"] == "" for r in rows)
    # a second run refuses to overwrite
    assert ac.run_main(rea.main, ["--stores", str(stores), "--out", str(out), "--legs", "pasttense"]) == ac.EXIT_REFUSED
    assert "REFUSED:" in capsys.readouterr().err
    # a missing stores root refuses
    assert ac.run_main(rea.main, ["--stores", str(tmp_path / "nope"), "--out", str(tmp_path / "o2"), "--legs", "pasttense"]) == ac.EXIT_REFUSED
    # petri without behaviours refuses
    assert ac.run_main(rea.main, ["--stores", str(stores), "--out", str(tmp_path / "o3"), "--legs", "petri"]) == ac.EXIT_REFUSED
    # a defective store → the table is written and exit 1
    bad = fx.make_root(tmp_path / "stores2", leg="renellm", view="s1", arms=("none",))
    fx.write_static_store(bad, family="renellm", arm="none", s=1, goals=["g0"], omit_body_keys=("presence_penalty",))
    bad.flush_pipeline()
    rc = ac.run_main(rea.main, ["--stores", str(tmp_path / "stores2"), "--out", str(tmp_path / "o4"), "--legs", "renellm"])
    assert rc == ac.EXIT_FAILED
    with open(tmp_path / "o4" / rea.SUMMARY_NAME, newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["verdict"] == "REFUSED" and "presence_penalty was NOT on the wire" in row["problem"]


def test_thinking_expect_parsing():
    t = rea._parse_thinking(["gsm8k=false", "mmlu=true", "bon=none"])
    assert t["gsm8k"] is False and t["mmlu"] is True and t["bon"] is None and t["pasttense"] is None
    with pytest.raises(ac.AuditRefusal):
        rea._parse_thinking(["nope=true"])
    with pytest.raises(ac.AuditRefusal):
        rea._parse_thinking(["gsm8k=maybe"])


def test_verdict_filename_rule():
    assert rea.verdict_filename("none", 1, stage="base", shard="shard00") == "none__s1.json"
    assert rea.verdict_filename("none", 2, stage="8192", shard="shard00") == "none__s2__cap8192.json"
    assert rea.verdict_filename("none", 3, stage="base", shard="shard01") == "none__s3__shard01.json"


def test_attempt_axes_at_top_level_with_int_shard(tmp_path: Path):
    """`read_attempt` accepts the axes at ATTEMPT.json's top level (shard an int); so does this reader."""
    root = fx.make_root(tmp_path / "stores", leg="bon", view="s2", arms=("none",))
    fx.populate_root(root, s=2, arms=("none",), n_items=1)
    att = json.loads((root.path / ac.ATTEMPT_FILE).read_text())
    att.pop("launch_plan_row")
    att.update(leg="bon", view="s2", shard=0, stage="base")
    (root.path / ac.ATTEMPT_FILE).write_text(json.dumps(att))
    ar = ac.read_attempt_root(root.path)
    assert (ar.leg, ar.view, ar.shard, ar.stage, ar.s) == ("bon", "s2", "shard00", "base", 2)
    (v,) = _audit(root, ["bon"], tmp_path)
    assert v["verdict"] == "AGREES" and Path(v["path"]).name == "none__s2.json"
    att.pop("leg")
    (root.path / ac.ATTEMPT_FILE).write_text(json.dumps(att))
    assert ac.read_attempt_root(root.path).leg == "bon"  # the control_key (its six axes) still names the leg
    att.pop("control_key")
    (root.path / ac.ATTEMPT_FILE).write_text(json.dumps(att))
    with pytest.raises(ac.AuditRefusal, match="neither launch_plan_row nor control_key nor the top level"):
        ac.read_attempt_root(root.path)
    assert ac.read_attempt_root(root.path, path_hint=("bon", "s2", "shard00")).leg == "bon"


def test_attempt_record_in_ws2d_shape_binds_to_its_planned_standup(tmp_path: Path):
    """The `attempt_record`: `launch_plan_row` is the plan row's standup_id STRING and `control_key` the six axes; the
    reader takes the axes from both and refuses when they disagree (an attempt that does not know which standup it ran)."""
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s3", arms=("none",), stage="8192")
    fx.populate_root(root, s=3, arms=("none",), n_items=1)
    att = json.loads((root.path / ac.ATTEMPT_FILE).read_text())
    assert att["launch_plan_row"] == "pasttense__s3__8192__shard00" and att["control_key"] == f"{fx.W_FIXTURE[:7]}/{fx.WAVE_UUID}/pasttense/s3/8192/0"
    ar = ac.read_attempt_root(root.path)
    assert (ar.leg, ar.view, ar.stage, ar.shard, ar.standup_id, ar.control_key, ar.wave_uuid) == (
        "pasttense", "s3", "8192", "shard00", "pasttense__s3__8192__shard00", att["control_key"], fx.WAVE_UUID)
    assert ac.standup_id_of(ar.leg, ar.view, ar.stage, ar.shard) == ar.standup_id
    # the path hint is NOT consulted when the record names its standup: a rung root pulled under any directory reads its own stage
    assert ac.read_attempt_root(root.path, path_hint=("pasttense", "s3", "rungs")).stage == "8192"
    # control_key disagreeing with the standup_id → refused by name
    (root.path / ac.ATTEMPT_FILE).write_text(json.dumps({**att, "control_key": att["control_key"].replace("/8192/", "/base/")}))
    with pytest.raises(ac.AuditRefusal, match="stage 'base' \\(control_key\\) disagrees with '8192'"):
        ac.read_attempt_root(root.path)
    # a control_key from another wave → refused
    (root.path / ac.ATTEMPT_FILE).write_text(json.dumps({**att, "control_key": att["control_key"].replace(fx.WAVE_UUID, "22222222-3333-4444-5555-666666666666")}))
    with pytest.raises(ac.AuditRefusal, match="control_key wave_uuid"):
        ac.read_attempt_root(root.path)
    # the dict shape (a row object) still reads
    (root.path / ac.ATTEMPT_FILE).write_text(json.dumps({**att, "launch_plan_row": {"standup_id": att["launch_plan_row"], "leg": "pasttense", "view": "s3",
                                                                                 "stage": "8192", "shard": 0}}))
    assert ac.read_attempt_root(root.path).standup_id == "pasttense__s3__8192__shard00"


# --------------------------------------------------------------------------- #
# P2-6: controls metadata binds by ITEM IDENTITY + text hash (GenMeta.bound), never by hash alone                  #
# --------------------------------------------------------------------------- #

def test_identical_control_responses_keep_their_own_items_metadata(tmp_path: Path):
    """Two XSTest items served the SAME text: their meta rows share text_sha256 and differ only by item_id. A hash-only
    index hands one item the other's row (and its request_id); `GenMeta.bound` keys on (item_id, text) so each draw
    carries its OWN attempt and the store AGREES."""
    root = fx.make_root(tmp_path / "stores", leg="controls", view="s1", arms=("none",))
    same = "I cannot help with that request."
    cell = fx.write_controls_store(root, arm="none", s=1, items_by_control={"xstest": ["item0", "item1", "item2"]},
                                   texts={"item0": same, "item1": same})
    root.flush_pipeline()
    (v,) = _audit(root, ["xstest"], tmp_path / "a")
    assert v["verdict"] == "AGREES", v["problem"]
    assert v["n_draws"] == 3 and v["n_draws_bound"] == 3
    # the binding is per item: the draw for item0 names the record whose sample is item0/0, item1's names item1/0
    recs = {r["request_id"]: r for r in ac.read_jsonl(root.path / ac.EVIDENCE_FILE)}
    ar = ac.read_attempt_root(root.path)
    (store,) = ac.discover_stores(ar, "xstest")
    by_item = {d.item_id: d for d in store.draws}
    assert by_item["item0"].text == by_item["item1"].text == same
    assert by_item["item0"].request_id != by_item["item1"].request_id
    for iid in ("item0", "item1", "item2"):
        assert recs[by_item[iid].request_id]["sample"] == C.sample_header_value(iid, 0)
    # KNOWN-BAD: the sidecar row for item1 relabelled to item0 → item1 has NO row matching (its item_id, its text) → refused by
    # the writer's own rule (a hash-only index would have silently bound item1 to item0's row)
    meta_p = cell / "private" / "gens_xstest.meta.jsonl"
    fx.rewrite_jsonl(meta_p, lambda r: {**r, "item_id": "item0"} if r["item_id"] == "item1" else r)
    with pytest.raises(ac.AuditRefusal, match="1 spooled item\\(s\\) have no meta row matching their spooled text"):
        _audit(root, ["xstest"], tmp_path / "b")


# --------------------------------------------------------------------------- #
# P1-2: an authoritative (request_id, completion_id) association is REQUIRED for every filed successful draw           #
# --------------------------------------------------------------------------- #

def _strip_ledger_completion_ids(root: fx.Root) -> None:
    fx.rewrite_jsonl(root.path / ac.LEDGER_FILE, lambda r: {k: v for k, v in r.items() if k != "completion_id"})


def test_missing_completion_id_evidence_refuses_never_bypasses(tmp_path: Path):
    """A ledger predating the `completion_id` field AND no `response_evidence.jsonl`: the row's completion_id is a
    claim nothing corroborates → every draw is a named binding failure (the old code skipped the check when the ledger
    lacked the field, so a row naming request A + completion B passed whenever B was anywhere in the engine log)."""
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    shard = fx.write_static_store(root, family="pasttense", arm="none", s=1, goals=["g0", "g1"])
    root.flush_pipeline()
    _strip_ledger_completion_ids(root)
    (root.path / ac.RESPONSE_EVIDENCE_FILE).unlink()
    (v,) = _audit(root, ["pasttense"], tmp_path / "a")
    assert v["verdict"] == "REFUSED"
    assert v["checks"]["row_binding"]["n_bound"] == 0 and v["checks"]["row_binding"]["n_problems"] == 4
    assert v["checks"]["row_binding"]["n_authoritative_completion_ids"] == 0
    assert "no authoritative completion_id for attempt" in v["problem"] and "response_evidence.jsonl" in v["problem"]
    # the bypass the review named: a row pointing at request A with completion B (B is g1/0's completion — it IS in the
    # engine log) must NOT pass just because B appears in the log
    rows = ac.read_jsonl(shard / "responses.jsonl")
    b_cid = next(r["completion_id"] for r in rows if (r["goal_id"], r["sample_idx"]) == ("g1", 0))
    fx.rewrite_jsonl(shard / "responses.jsonl", lambda r: {**r, "completion_id": b_cid} if (r["goal_id"], r["sample_idx"]) == ("g0", 0) else r)
    (v2,) = _audit(root, ["pasttense"], tmp_path / "b")
    assert v2["verdict"] == "REFUSED" and v2["checks"]["row_binding"]["n_bound"] == 0


def test_response_evidence_alone_is_an_authoritative_completion_association(tmp_path: Path):
    """The Inspect-family shape: the ledger's success row lacks `completion_id` but the response hook's
    `response_evidence.jsonl` records the pair → the draws bind; a row whose completion_id is not the recorded one is torn."""
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    shard = fx.write_static_store(root, family="pasttense", arm="none", s=1, goals=["g0", "g1"])
    root.flush_pipeline()
    _strip_ledger_completion_ids(root)
    (v,) = _audit(root, ["pasttense"], tmp_path / "a")
    assert v["verdict"] == "AGREES", v["problem"]
    assert v["checks"]["row_binding"]["n_bound"] == 4 and v["checks"]["row_binding"]["n_authoritative_completion_ids"] == 4
    rows = ac.read_jsonl(shard / "responses.jsonl")
    b_cid = next(r["completion_id"] for r in rows if (r["goal_id"], r["sample_idx"]) == ("g1", 0))
    fx.rewrite_jsonl(shard / "responses.jsonl", lambda r: {**r, "completion_id": b_cid} if (r["goal_id"], r["sample_idx"]) == ("g0", 0) else r)
    (v2,) = _audit(root, ["pasttense"], tmp_path / "b")
    assert v2["verdict"] == "REFUSED" and "torn binding" in v2["problem"] and v2["checks"]["row_binding"]["n_bound"] == 3


def test_ledger_and_response_evidence_disagreeing_on_a_completion_refuses_the_root(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.write_static_store(root, family="pasttense", arm="none", s=1, goals=["g0"])
    root.flush_pipeline()
    rid = json.loads((root.path / ac.LEDGER_FILE).read_text().splitlines()[0])["request_id"]
    fx.rewrite_jsonl(root.path / ac.RESPONSE_EVIDENCE_FILE, lambda r: {**r, "completion_id": "chatcmpl-other"} if r["request_id"] == rid else r)
    with pytest.raises(ac.AuditRefusal, match="two records of one attempt name two completions"):
        _audit(root, ["pasttense"], tmp_path)


# --------------------------------------------------------------------------- #
# P1-1: completeness = the WRITER's own declaration of the sample identities, exact correspondence with what it stored   #
# --------------------------------------------------------------------------- #

def test_static_removed_variant_refuses_by_the_writers_rule_and_by_name(tmp_path: Path):
    """The progress records declare (goal, n_variants); a store missing one variant row AGREED before (the reader only
    asked that remaining goals have a progress record). Now: run_static's own completeness rule fires AND the missing
    (goal_id, sample_idx) is named."""
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    shard = fx.write_static_store(root, family="pasttense", arm="none", s=1, goals=["g0", "g1"])
    root.flush_pipeline()
    assert _audit(root, ["pasttense"], tmp_path / "a")[0]["verdict"] == "AGREES"
    rows = ac.read_jsonl(shard / "responses.jsonl")
    (shard / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if (r["goal_id"], r["sample_idx"]) != ("g1", 1)))
    (v,) = _audit(root, ["pasttense"], tmp_path / "b")
    assert v["verdict"] == "REFUSED" and v["store_complete"] is False
    assert "run_static's completeness rule" in v["problem"] and "3 response rows != 4 expected" in v["problem"]
    assert "1 declared (goal_id, sample_idx)(s) without a stored response (first: [('g1', 1)])" in v["problem"]
    assert v["n_filed"] == 2  # the DECLARED goals, not the goals that happen to have rows
    # the reverse direction: a goal with rows but no progress record is named too (it was invisible before)
    (shard / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    prog = ac.read_jsonl(shard / "progress.jsonl")
    (shard / "progress.jsonl").write_text(json.dumps(prog[0]) + "\n")
    (v2,) = _audit(root, ["pasttense"], tmp_path / "c")
    assert v2["verdict"] == "REFUSED" and "2 stored (goal_id, sample_idx)(s) the writer declares nowhere (first: [('g1', 0), ('g1', 1)])" in v2["problem"]
    assert v2["n_filed"] == 1


def test_gsm8k_empty_outcomes_refuses(tmp_path: Path):
    """Two filed GSM8K items with an EMPTY outcomes `items` list AGREED before with zero draws bound. Now the outcomes
    header's own counts, the filed items and the done marker must all correspond — refused whether the header still
    claims the old counts or was made consistent with the empty list."""
    root = fx.make_root(tmp_path / "stores", leg="gsm8k", view="s1", arms=("none",))
    cell = fx.write_gsm8k_store(root, arm="none", s=1, items=["item0", "item1"])
    root.flush_pipeline()
    assert _audit(root, ["gsm8k"], tmp_path / "a")[0]["verdict"] == "AGREES"
    from sisteer.capeval.tinycap import completion as _completion
    op = cell / C.TASK_OUTCOMES_DIR / _completion.marker_filename(ac.GSM8K_TASK)
    out = json.loads(op.read_text())
    op.write_text(json.dumps({**out, "items": []}))
    (v,) = _audit(root, ["gsm8k"], tmp_path / "b")
    assert v["verdict"] == "REFUSED" and v["n_draws"] == 0 and v["n_draws_bound"] == 0
    assert "outcomes record lists 0 item record(s) but its header declares n_attempted_samples=2" in v["problem"]
    # fg2: the SAMPLE-AXIS rule names the same two holes -- every declared item x range(samples_per_item) must have its
    # outcome record, whatever the header says about itself
    assert "2 declared (item_id, sample_idx) outcome record(s) absent from the outcomes record -- the sample axis (2 item(s) x samples_per_item 1) is not whole (first: [('item0', 0), ('item1', 0)])" in v["problem"]
    # header made consistent with the empty list: still refused — the FILED items (rows.jsonl) have no attempted draw
    op.write_text(json.dumps({**out, "items": [], "n_attempted_samples": 0, "n_attempted_items": 0}))
    (v2,) = _audit(root, ["gsm8k"], tmp_path / "c")
    assert v2["verdict"] == "REFUSED" and "2 declared (item_id, sample_idx) outcome record(s) absent from the outcomes record" in v2["problem"]
    assert "2 filed acc row(s) the outcomes record does not declare scored" in v2["problem"] and "an empty store" in v2["problem"]
    # and a duplicated outcome record is named
    op.write_text(json.dumps({**out, "items": out["items"] + [out["items"][0]]}))
    (v3,) = _audit(root, ["gsm8k"], tmp_path / "d")
    assert v3["verdict"] == "REFUSED" and "1 (item_id, sample_idx) outcome record(s) stored twice" in v3["problem"]


def test_multiturn_missing_reply_row_refuses(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="pair", view="s1", arms=("none",))
    shard = fx.write_multiturn_store(root, attack="pair", arm="none", s=1, goals=["g0", "g1"], n_turns=2)
    root.flush_pipeline()
    assert _audit(root, ["pair"], tmp_path / "a")[0]["verdict"] == "AGREES"
    fx.rewrite_jsonl(shard / "responses.jsonl", lambda r: None if (r["goal_id"], r["order"]) == ("g0", 1) else r)
    (v,) = _audit(root, ["pair"], tmp_path / "b")
    assert v["verdict"] == "REFUSED" and "goal 'g0': 1 stored reply row(s) != n_target_forwards 2 the progress record declares" in v["problem"]


def test_misalign_missing_sample_and_bon_controls_count_disagreements_refuse(tmp_path: Path):
    mis = fx.make_root(tmp_path / "stores", leg="misalign", view="s1", arms=("none",))
    cell = fx.write_misalign_store(mis, arm="none", s=1, items={"sc0": "leaking", "sc1": "leaking"}, n_samples=2)
    mis.flush_pipeline(result_rc="0")
    assert _audit(mis, ["misalign_leaking"], tmp_path / "a")[0]["verdict"] == "AGREES"
    fx.rewrite_jsonl(cell / "responses.jsonl", lambda r: None if (r["item_id"], r["sample_idx"]) == ("sc1", 1) else r)
    (v,) = _audit(mis, ["misalign_leaking"], tmp_path / "b")
    assert v["verdict"] == "REFUSED" and "1 declared (item_id, sample_idx)(s) without a stored response (first: [('sc1', 1)])" in v["problem"]
    # BoN: the manifest's units.n_done disagreeing with the rows
    bon = fx.make_root(tmp_path / "stores", leg="bon", view="s1", arms=("none",))
    bcell = fx.write_bon_store(bon, arm="none", s=1, goals=["g0"])
    bon.flush_pipeline()
    assert _audit(bon, ["bon"], tmp_path / "c")[0]["verdict"] == "AGREES"
    mp = bcell / "bon_served_manifest.json"
    m = json.loads(mp.read_text())
    m["units"]["n_done"] += 1
    mp.write_text(json.dumps(m))
    (vb,) = _audit(bon, ["bon"], tmp_path / "d")
    assert vb["verdict"] == "REFUSED" and "2 stored BoN row(s) != units.n_done 3 the manifest declares" in vb["problem"]
    # controls: gen_stats n_total disagreeing with the spooled items
    ctl = fx.make_root(tmp_path / "stores", leg="controls", view="s1", arms=("none",))
    ccell = fx.write_controls_store(ctl, arm="none", s=1, items_by_control={"xstest": ["item0", "item1"]})
    ctl.flush_pipeline()
    assert _audit(ctl, ["xstest"], tmp_path / "e")[0]["verdict"] == "AGREES"
    gp = ccell / "gen_stats.json"
    gs = json.loads(gp.read_text())
    gs["xstest"]["n_total"] = 3
    gp.write_text(json.dumps(gs))
    (vc,) = _audit(ctl, ["xstest"], tmp_path / "f")
    assert vc["verdict"] == "REFUSED" and "2 spooled item(s) != gen_stats.json[xstest].n_total 3 the writer declares" in vc["problem"]


# --------------------------------------------------------------------------- #
# P1-5: a real static RUNG store is MIXED-CAP by the writer's design — identity separate from the per-row source cap    #
# --------------------------------------------------------------------------- #

def _mixed_cap_rung(tmp_path: Path):
    """Base 4096 (g0 at cap, g1 terminated) + an 8192 rung that re-draws g0's two rows and copies g1's two forward
    verbatim (their decode block still says cap 4096, their request ids are the BASE attempt's) — exactly what
    `escalate_static.py` writes."""
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",))
    base_shard = fx.write_static_store(base, family="pasttense", arm="none", s=1, goals=["g0", "g1"], at_cap_goals=("g0",))
    base.flush_pipeline()
    rung = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), stage="8192")
    rung_shard = fx.write_static_rung(rung, base_shard=base_shard, family="pasttense", arm="none", s=1, cap_from=4096, cap_to=8192)
    rung.flush_pipeline()
    rows = ac.read_jsonl(rung_shard / "responses.jsonl")
    assert {r["decode"]["cap"] for r in rows} == {4096, 8192}, "the fixture must be a genuinely mixed-cap store"
    assert {(r["regenerated"], r["source_cap"]) for r in rows} == {(False, 4096), (True, 8192)}
    return stores, base, base_shard, rung, rung_shard


def test_mixed_cap_rung_reads_identity_separately_and_binds_copied_rows_on_the_base(tmp_path: Path):
    stores, base, base_shard, rung, rung_shard = _mixed_cap_rung(tmp_path)
    (v,) = _audit(rung, ["pasttense"], tmp_path / "a", siblings=[base])
    assert v["verdict"] == "AGREES", v["problem"]  # the old reader refused this store: "decode block is not uniform"
    assert v["n_draws"] == 4 and v["n_draws_bound"] == 4
    rb = v["checks"]["row_binding"]
    assert rb["n_copied_forward"] == 2 and rb["ancestor_caps"] == [4096] and rb["ancestor_attempt_ids"] == {"4096": [base.attempt_id]}
    # the reconciliation on THIS root's evidence covers the items drawn HERE (g0); the store still files both goals
    assert v["n_filed"] == 2 and v["n_filed_here"] == 1 and v["store_complete"] is True
    assert v["stage"] == "8192" and Path(v["path"]).name == "none__s1__cap8192.json"
    # the copied rows' request ids ARE the base attempt's, not this root's
    ar = ac.read_attempt_root(rung.path)
    (store,) = ac.discover_stores(ar, "pasttense")
    assert store.cap == 8192 and store.filed_here == {("static/pasttense", "g0")}
    base_rids = {json.loads(l)["request_id"] for l in (base.path / ac.LEDGER_FILE).read_text().splitlines() if l.strip()}
    copied = [d for d in store.draws if d.regenerated is False]
    assert len(copied) == 2 and all(d.source_cap == 4096 and d.request_id in base_rids for d in copied)
    assert all(d.regenerated is True and d.source_cap == 8192 and d.request_id not in base_rids for d in store.draws if d not in copied)
    # the base root itself still audits AGREES beside it
    (vb,) = _audit(base, ["pasttense"], tmp_path / "b")
    assert vb["verdict"] == "AGREES" and vb["checks"]["row_binding"]["n_copied_forward"] == 0


def test_mixed_cap_rung_without_its_ancestor_root_refuses(tmp_path: Path):
    stores, base, base_shard, rung, rung_shard = _mixed_cap_rung(tmp_path)
    with pytest.raises(ac.AuditRefusal, match="copied-forward rows were drawn at cap 4096 but no accepted ancestor root"):
        _audit(rung, ["pasttense"], tmp_path / "a")


def test_rung_row_contradicting_its_provenance_refuses_by_the_writers_rule(tmp_path: Path):
    stores, base, base_shard, rung, rung_shard = _mixed_cap_rung(tmp_path)
    # a copied row relabelled source_cap 8192 while its decode block says 4096: escalate_static.row_source_cap's own refusal
    fx.rewrite_jsonl(rung_shard / "responses.jsonl", lambda r: {**r, "source_cap": 8192} if r["regenerated"] is False and r["sample_idx"] == 0 else r)
    with pytest.raises(ac.AuditRefusal, match="copy-forward provenance disagrees with the text's own decode block"):
        _audit(rung, ["pasttense"], tmp_path / "a", siblings=[base])


def test_sampler_identity_not_uniform_refuses_while_caps_may_differ(tmp_path: Path):
    stores, base, base_shard, rung, rung_shard = _mixed_cap_rung(tmp_path)
    fx.rewrite_jsonl(rung_shard / "responses.jsonl", lambda r: {**r, "decode": {**r["decode"], "temperature": 0.9}} if r["sample_idx"] == 1 else r)
    with pytest.raises(ac.AuditRefusal, match="sampler / seed identity \\(the decode block minus its cap\\) is not uniform"):
        _audit(rung, ["pasttense"], tmp_path / "a", siblings=[base])


def test_cli_audits_a_rung_root_beside_its_base(tmp_path: Path):
    stores, base, base_shard, rung, rung_shard = _mixed_cap_rung(tmp_path)
    out = tmp_path / "audits"
    assert ac.run_main(rea.main, ["--stores", str(stores), "--out", str(out), "--legs", "pasttense"]) == 0
    with open(out / rea.SUMMARY_NAME, newline="") as f:
        rows = {r["stage"]: r for r in csv.DictReader(f)}
    assert set(rows) == {"base", "8192"} and all(r["verdict"] == "AGREES" for r in rows.values())
    assert rows["8192"]["n_draws_bound"] == "4" and rows["8192"]["attempt_id"] == rung.attempt_id


# --------------------------------------------------------------------------- #
# the ancestor probe never swallows a MALFORMED sibling store #
# --------------------------------------------------------------------------- #

def test_ancestor_probe_refuses_a_malformed_sibling_store_by_name(tmp_path: Path):
    """The base sibling's static store loses its progress.jsonl (the pair the writer always leaves). The old probe caught
    EVERY AuditRefusal while looking for the ancestor's cap and reported "no accepted ancestor root"; now the sibling's own
    refusal surfaces by name — a broken ancestor is never mistaken for an absent one."""
    stores, base, base_shard, rung, rung_shard = _mixed_cap_rung(tmp_path)
    (base_shard / "progress.jsonl").unlink()
    with pytest.raises(ac.AuditRefusal, match="responses.jsonl without progress.jsonl — a static store is the pair"):
        _audit(rung, ["pasttense"], tmp_path / "a", siblings=[base])


# --------------------------------------------------------------------------- #
# a controls RUNG is mixed-cap by the writer's design — copied rows bind on the ancestor's response evidence #
# --------------------------------------------------------------------------- #

def test_controls_mixed_cap_rung_binds_copied_rows_on_the_base(tmp_path: Path):
    """Base 512 (item0 at cap) + a 2048 rung written through the controls writer's own classes (`GenSpool`, `GenMeta.append`,
    `_fork_fields`, `_meta_stats`): item0 RE-DRAWN here (regenerated, source_cap 2048), item1 COPIED forward with its source
    row (regenerated false, source_cap 512 — its attempt lives on the BASE root's response evidence; a meta row can carry no
    request id, `GenMeta.append` refuses the key). The old reader dropped `source_cap` / `regenerated`, returned no cap and
    looked every draw up on the rung root — one of two bound, the store REFUSED. Now the copied draw resolves its one
    completed attempt on the ancestor's evidence and the store AGREES."""
    from sisteer.eval.sampled_controls import META_REGEN_KEYS, META_ROW_KEYS
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="controls", view="s1", arms=("none",))
    bcell = fx.write_controls_store(base, arm="none", s=1, items_by_control={"xstest": ["item0", "item1"]}, at_cap={"xstest": ["item0"]})
    base.flush_pipeline()
    rung = fx.make_root(stores, leg="controls", view="s1", arms=("none",), stage="2048")
    rcell = fx.write_controls_store(rung, arm="none", s=1, items_by_control={"xstest": ["item0", "item1"]}, stage_cap=2048, source=bcell,
                                    regenerate={"xstest": ["item0"]})
    rung.flush_pipeline()
    meta = {r["item_id"]: r for r in ac.read_jsonl(rcell / "private" / "gens_xstest.meta.jsonl")}
    assert (meta["item0"]["regenerated"], meta["item0"]["source_cap"], set(meta["item0"])) == (True, 2048, set(META_ROW_KEYS) | set(META_REGEN_KEYS))
    assert (meta["item1"]["regenerated"], meta["item1"]["source_cap"], set(meta["item1"])) == (False, 512, set(META_ROW_KEYS))
    gs = ac.read_json(rcell / "gen_stats.json")["xstest"]
    assert (gs["cap"], gs["n_copied"], gs["n_regenerated"], gs["n_drawn_here"], gs["n_total"]) == (2048, 1, 1, 1, 2)
    task = ac.CONTROLS_TASK_OF["xstest"]
    (v,) = _audit(rung, ["xstest"], tmp_path / "a", siblings=[base])
    assert v["verdict"] == "AGREES", v["problem"]
    rb = v["checks"]["row_binding"]
    assert v["n_draws"] == 2 and v["n_draws_bound"] == 2 and rb["n_copied_forward"] == 1 and rb["ancestor_caps"] == [512]
    assert rb["ancestor_attempt_ids"] == {"512": [base.attempt_id]} and v["n_filed_here"] == 1 and v["store_complete"] is True
    ar = ac.read_attempt_root(rung.path)
    (store,) = ac.discover_stores(ar, "xstest")
    assert store.cap == 2048 and store.filed_here == {(task, "item0")}
    copied = [d for d in store.draws if d.source_cap == 512]
    assert len(copied) == 1 and copied[0].item_id == "item1" and copied[0].request_id is None  # resolved on the ancestor at bind time, never here
    # RE-VALIDATION of the finding: audited alone the copied row's attempt is nowhere on this root → refused by name, never bound to nothing
    with pytest.raises(ac.AuditRefusal, match="copied-forward rows were drawn at cap 512 but no accepted ancestor root"):
        _audit(rung, ["xstest"], tmp_path / "b")
    # KNOWN-BAD, layer by layer: the copied row relabelled `source_cap 2048` alone → its request's max_tokens 512 disagrees;
    # with max_tokens relabelled too → an un-regenerated row on a rung at the rung's own cap; with `regenerated` relabelled as well it
    # reads as drawn HERE — and this root's response evidence holds no attempt for it
    mpath = rcell / "private" / "gens_xstest.meta.jsonl"
    fx.rewrite_jsonl(mpath, lambda r: {**r, "source_cap": 2048} if r["item_id"] == "item1" else r)
    with pytest.raises(ac.AuditRefusal, match="item 'item1': max_tokens 512 is not the row's source_cap 2048"):
        _audit(rung, ["xstest"], tmp_path / "c1", siblings=[base])
    fx.rewrite_jsonl(mpath, lambda r: {**r, "max_tokens": 2048} if r["item_id"] == "item1" else r)
    with pytest.raises(ac.AuditRefusal, match="item 'item1' is an un-regenerated row on a RUNG store recording source_cap 2048"):
        _audit(rung, ["xstest"], tmp_path / "c2", siblings=[base])
    fx.rewrite_jsonl(mpath, lambda r: {**r, "regenerated": True} if r["item_id"] == "item1" else r)
    (v2,) = _audit(rung, ["xstest"], tmp_path / "c", siblings=[base])
    assert v2["verdict"] == "REFUSED" and f"records no completed attempt for ({task}, item1/0, petri__none)" in v2["problem"]
    fx.rewrite_jsonl(mpath, lambda r: {**r, "regenerated": False, "max_tokens": 512, "source_cap": 512} if r["item_id"] == "item1" else r)
    # KNOWN-BAD: a regenerated row claiming a source cap below this store's (its request cap moved with it) → the writer's rule, by name
    fx.rewrite_jsonl(rcell / "private" / "gens_xstest.meta.jsonl", lambda r: {**r, "source_cap": 512, "max_tokens": 512} if r["item_id"] == "item0" else r)
    with pytest.raises(ac.AuditRefusal, match="item 'item0' is marked regenerated but records source_cap 512, not this store's cap 2048"):
        _audit(rung, ["xstest"], tmp_path / "d", siblings=[base])
    assert _audit(base, ["xstest"], tmp_path / "e")[0]["verdict"] == "AGREES"  # the base beside it is untouched


# --------------------------------------------------------------------------- #
# Petri per-turn draws from extra_json.target_turns, read FAIL-CLOSED #
# --------------------------------------------------------------------------- #

def _petri_rows(cell: Path):
    return ac.read_jsonl(cell / "rows.jsonl")


def _set_turns(cell: Path, sid: str, fn) -> None:
    """Rewrite scenario `sid`'s target_turns on EVERY dimension row through `fn(list) -> list | None` (None drops the key)."""
    def edit(r):
        if r["item_id"] != sid:
            return r
        extra = json.loads(r["extra_json"])
        cur = extra.get(ac.PETRI_TARGET_TURNS_KEY)
        new = fn(list(cur) if isinstance(cur, list) else [])
        if new is None:
            extra.pop(ac.PETRI_TARGET_TURNS_KEY)
        else:
            extra[ac.PETRI_TARGET_TURNS_KEY] = new
        return {**r, "extra_json": json.dumps(extra, sort_keys=True)}
    fx.rewrite_jsonl(cell / "rows.jsonl", edit)


def test_petri_per_turn_draws_bind_each_turn_and_malformed_lists_refuse(tmp_path: Path):
    """Two scenarios × three target turns, sc0's turn 2 failed once (500) and retried: the rows carry
    `extra_json.target_turns` (no ids on the row); the reader yields one draw per SUCCESSFUL attempt (6), every one binds
    (AGREES). KNOWN-BAD, each through the real `read_petri_store` + `audit_store`: a torn completion → torn binding; the
    failed attempt forced in as a draw → the REJECTED ledger attempt; a scenario with NO list, a non-list, a boolean turn,
    a half-filed attempt (2xx with a null completion), a request id filed twice, another behaviour's task, and dimension
    rows carrying DIFFERENT lists → each REFUSED by name, never zero draws or AGREES."""
    stores = tmp_path / "stores"
    root = fx.make_root(stores, leg="petri", view="s1", arms=("none",))
    cell = fx.write_petri_store(root, arm="none", s=1, scenarios={"sc0": "beh_a", "sc1": "beh_b"}, n_turns=3, failed_turns={"sc0": [2]})
    root.flush_pipeline()
    rows = _petri_rows(cell)
    tt = json.loads(next(r for r in rows if r["item_id"] == "sc0")["extra_json"])[ac.PETRI_TARGET_TURNS_KEY]
    assert [(e["turn"], e["status"], e["completion_id"] is None) for e in tt] == [(1, 200, False), (2, 500, True), (2, 200, False), (3, 200, False)]
    assert all("request_id" not in r and "completion_id" not in r for r in rows)
    ar = ac.read_attempt_root(root.path)
    (store,) = ac.discover_stores(ar, "petri", petri_behaviours=list(fx.PETRI_BEHAVIOURS))
    assert sorted((d.item_id, d.sample_idx) for d in store.draws) == [("sc0", 1), ("sc0", 2), ("sc0", 3), ("sc1", 1), ("sc1", 2), ("sc1", 3)]
    assert all(d.request_id and d.completion_id for d in store.draws)
    (v,) = _audit(root, ["petri"], tmp_path / "a")
    assert v["verdict"] == "AGREES", v["problem"]
    assert v["n_draws"] == 6 and v["n_draws_bound"] == 6 and v["n_filed"] == 2
    good_tt = list(tt)
    # torn: turn 2's successful completion relabelled
    _set_turns(cell, "sc0", lambda ts: [({**e, "completion_id": "chatcmpl-torn"} if (e["turn"], e["status"]) == (2, 200) else e) for e in ts])
    (v2,) = _audit(root, ["petri"], tmp_path / "b")
    assert v2["verdict"] == "REFUSED" and "torn binding" in v2["problem"]
    # the failed attempt forced in as a draw (given the retry's completion) → its ledger attempt is REJECTED
    retry_cid = next(e["completion_id"] for e in good_tt if (e["turn"], e["status"]) == (2, 200))
    _set_turns(cell, "sc0", lambda ts: [({**e, "status": 200, "completion_id": retry_cid} if e["status"] == 500 else e) for e in good_tt])
    (v3,) = _audit(root, ["petri"], tmp_path / "c")
    assert v3["verdict"] == "REFUSED" and "REJECTED ledger attempt" in v3["problem"]
    # the NEGATIVES that used to read as zero draws — each an audit refusal by name
    cases = [
        (lambda ts: None, "carries no extra_json.target_turns list \\(NoneType\\)"),
        (lambda ts: "not-a-list", "carries no extra_json.target_turns list \\(str\\)"),
        (lambda ts: [({**e, "turn": True} if e["turn"] == 1 else e) for e in good_tt], "target_turns\\[0\\]: turn True is not a positive int"),
        (lambda ts: [({**e, "completion_id": None} if (e["turn"], e["status"]) == (3, 200) else e) for e in good_tt],
         "status 200 with completion_id None — a 2xx attempt has a completion"),
        (lambda ts: good_tt + [dict(good_tt[0])], "is filed twice in one scenario's list"),
        (lambda ts: [({**e, "task": "petri/beh_b"} if e["turn"] == 3 else e) for e in good_tt], "filed under task 'petri/beh_b' but the row's task is 'petri/beh_a'"),
        (lambda ts: [({**e, "status": "200"} if e["turn"] == 1 else e) for e in good_tt], "status '200' is neither an int nor null"),
        # the cross-check (892827da): every attempt failed → zero draws must never leave the scenario filed and AGREED
        (lambda ts: [{**e, "status": 500, "completion_id": None} for e in good_tt], "sc0: none of its 4 target attempt\\(s\\) succeeded"),
        # an entry with a key the sink never writes
        (lambda ts: [({**e, "note": "x"} if e["turn"] == 1 else e) for e in good_tt], "keys \\['completion_id', 'note', 'request_id', 'status', 'task', 'turn'\\] are not the sink's"),
        # an entry lacking one of the five
        (lambda ts: [({k: v for k, v in e.items() if k != "status"} if e["turn"] == 1 else e) for e in good_tt], "are not the sink's"),
    ]
    for k, (fn, text) in enumerate(cases):
        _set_turns(cell, "sc0", fn)
        with pytest.raises(ac.AuditRefusal, match=text):
            _audit(root, ["petri"], tmp_path / f"n{k}")
    # dimension rows of one scenario carrying DIFFERENT lists → refused (one scenario, one ordered list)
    _set_turns(cell, "sc0", lambda ts: good_tt)
    fx.rewrite_jsonl(cell / "rows.jsonl", lambda r: ({**r, "extra_json": json.dumps({**json.loads(r["extra_json"]), ac.PETRI_TARGET_TURNS_KEY: good_tt[:2]}, sort_keys=True)}
                                                    if (r["item_id"], r["metric"]) == ("sc0", "dim_y") else r))
    with pytest.raises(ac.AuditRefusal, match="the scenario's dimension rows carry DIFFERENT target_turns lists"):
        _audit(root, ["petri"], tmp_path / "d")
    # restored → AGREES again (the fixture's rule is the reader's rule)
    _set_turns(cell, "sc0", lambda ts: good_tt)
    assert _audit(root, ["petri"], tmp_path / "e")[0]["verdict"] == "AGREES"
    # an EMPTY row set: only the aggregate row left → no filed scenario (refused by name); no rows at all → no Petri store on the root
    all_rows = _petri_rows(cell)
    fx.rewrite_jsonl(cell / "rows.jsonl", lambda r: {**r, "item_id": C.AGG_ITEM_ID, "extra_json": json.dumps({})} if r["metric"] == "dim_x" and r["item_id"] == "sc0" else None)
    with pytest.raises(ac.AuditRefusal, match="no filed Petri items"):
        _audit(root, ["petri"], tmp_path / "f")
    (cell / "rows.jsonl").write_text("")
    with pytest.raises(ac.AuditRefusal, match="no petri store under"):
        _audit(root, ["petri"], tmp_path / "g")
    fx._jsonl(cell / "rows.jsonl", all_rows)
    assert _audit(root, ["petri"], tmp_path / "h")[0]["verdict"] == "AGREES"


# --------------------------------------------------------------------------- #
# the fixture's evidence records are the RECORDER's shapes (0c329079), pinned #
# --------------------------------------------------------------------------- #

def test_fixture_records_are_the_recorders_shapes(tmp_path: Path):
    """The request record = contracts' v3 record (`REQUEST_EVIDENCE_FIELDS`) + `extra` + the attempt id + the ADDITIVE `body_b64`;
    the response record = `RESPONSE_EVIDENCE_FIELDS` in order + `extra` + the ADDITIVE `text_sha256` on a
    2xx — a rejected attempt has its row too (status 500, no completion, an error). Pinned to literals and,
    when `sisteer.eval.request_evidence_client` is on the tree, to its own constants. No `request_bodies.jsonl` is written."""
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.write_static_store(root, family="pasttense", arm="none", s=1, goals=["g0"])
    root.send("static/pasttense", "g9", 0, root.vmodels["none"], root.vendor_body(root.vmodels["none"], "g9/0", 4096, 7), ok=False)
    root.flush_pipeline()
    assert not (root.path / "request_bodies.jsonl").exists()
    recs = ac.read_jsonl(root.path / ac.EVIDENCE_FILE)
    # the KEY SET is the recorder's (the fixture serialises with sorted keys, so on-disk order is not the pin)
    assert {k for k in recs[0] if k not in ("extra", C.REQUEST_EVIDENCE_ATTEMPT_FIELD, ac.REQUEST_BODY_B64_FIELD)} == set(C.REQUEST_EVIDENCE_FIELDS)
    assert all(r["extra"]["attempt_id"] == r[C.REQUEST_EVIDENCE_ATTEMPT_FIELD] == root.attempt_id for r in recs)
    resp = ac.read_jsonl(root.path / ac.RESPONSE_EVIDENCE_FILE)
    # the response record's version 2: the version
    # first, `text_sha256` + `text_sha256_of` after `finish_reason` — the KEY SET of every row is the constant's + `extra`
    assert ac.RESPONSE_EVIDENCE_FIELDS == ("response_evidence_version", "ts", "request_id", "completion_id", "status", "vmodel",
                                          "finish_reason", "text_sha256", "text_sha256_of", "task", "sample", "error")
    assert ac.RESPONSE_EVIDENCE_VERSION == 2 and ac.RESPONSE_TEXT_SHA_FIELD in ac.RESPONSE_EVIDENCE_FIELDS
    for r in resp:
        assert {k for k in r if k != "extra"} == set(ac.RESPONSE_EVIDENCE_FIELDS) and r["response_evidence_version"] == ac.RESPONSE_EVIDENCE_VERSION
    ok_rows = [r for r in resp if r["status"] == 200]
    bad_rows = [r for r in resp if r["status"] == 500]
    assert len(ok_rows) == 2 and len(bad_rows) == 1
    assert all(r["completion_id"] and r[ac.RESPONSE_TEXT_SHA_FIELD] and r[ac.RESPONSE_TEXT_SHA_OF_FIELD] == "content" for r in ok_rows)
    assert bad_rows[0]["completion_id"] is None and bad_rows[0]["error"]
    assert bad_rows[0][ac.RESPONSE_TEXT_SHA_FIELD] is None and bad_rows[0][ac.RESPONSE_TEXT_SHA_OF_FIELD] is None   # present, None: no body
    try:
        from sisteer.eval import request_evidence_client as rev  # type: ignore[import-not-found]
    except ImportError:
        return  # the client is not on this tree; the literal pins above hold its shape
    assert tuple(rev.RESPONSE_EVIDENCE_FIELDS) == ac.RESPONSE_EVIDENCE_FIELDS and rev.RESPONSE_EVIDENCE_VERSION == ac.RESPONSE_EVIDENCE_VERSION
    assert rev.RESPONSE_EVIDENCE_FILENAME == ac.RESPONSE_EVIDENCE_FILE and rev.EXTRA_FIELD == "extra" and rev.TEXT_SHA256_OF_CONTENT == "content"


# --------------------------------------------------------------------------- #
# a relabelled copied row with a matching UNRELATED attempt present on the rung — three independent refusals #
# --------------------------------------------------------------------------- #

def test_controls_relabelled_copy_with_a_matching_attempt_refuses(tmp_path: Path):
    """The rung REGENERATES both items, so item1 has a completed attempt on the rung root. Its meta row is then rewritten as
    the base's COPIED 512 row relabelled `source_cap=2048` (`max_tokens=512` left as the request made it) and its spool
    text set back to the base text: the old reader took `source_cap == cap` as drawn here and bound the unrelated regenerated
    attempt — AGREES. Now, layer by layer: (a) `max_tokens` 512 != `source_cap` 2048 → refused; (b) with `max_tokens` also
    relabelled, an un-regenerated row on a RUNG at the rung's own cap → refused; (c) with `regenerated` relabelled too, the
    row binds the unrelated attempt and the stored text's hash is not that attempt's response hash → REFUSED; (d) an attempt
    without a response hash → refused, never assumed; and a base store carrying a copied-forward row → refused."""
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="controls", view="s1", arms=("none",))
    bcell = fx.write_controls_store(base, arm="none", s=1, items_by_control={"xstest": ["item0", "item1"]}, at_cap={"xstest": ["item0", "item1"]})
    base.flush_pipeline()
    rung = fx.make_root(stores, leg="controls", view="s1", arms=("none",), stage="2048")
    rcell = fx.write_controls_store(rung, arm="none", s=1, items_by_control={"xstest": ["item0", "item1"]}, stage_cap=2048, source=bcell,
                                    regenerate={"xstest": ["item0", "item1"]})
    rung.flush_pipeline()
    assert _audit(rung, ["xstest"], tmp_path / "ok", siblings=[base])[0]["verdict"] == "AGREES"
    base_meta = {r["item_id"]: r for r in ac.read_jsonl(bcell / "private" / "gens_xstest.meta.jsonl")}
    base_text = ac.read_jsonl(bcell / "private" / "gens_xstest.jsonl")[1:]
    base_text = {e["item_id"]: e["text"] for e in base_text}
    rmeta, rspool = rcell / "private" / "gens_xstest.meta.jsonl", rcell / "private" / "gens_xstest.jsonl"

    def relabel(**over):
        copied = {k: base_meta["item1"][k] for k in ("item_id", "sampling_seed", "max_tokens", "finish_reason", "completion_tokens", "prompt_tokens",
                                                     "at_cap", "source_cap", "text_sha256")}
        copied.update({"regenerated": False, "floor_sample": False, "source_cap": 2048})
        copied.update(over)
        fx.rewrite_jsonl(rmeta, lambda r: copied if r["item_id"] == "item1" else r)
        fx.rewrite_jsonl(rspool, lambda e: {**e, "text": base_text["item1"]} if e.get("item_id") == "item1" else e)
    relabel()                                                   # (a) max_tokens 512 vs source_cap 2048
    with pytest.raises(ac.AuditRefusal, match="item 'item1': max_tokens 512 is not the row's source_cap 2048"):
        _audit(rung, ["xstest"], tmp_path / "a", siblings=[base])
    relabel(max_tokens=2048)                                    # (b) an un-regenerated row on a rung at the rung's own cap
    with pytest.raises(ac.AuditRefusal, match="item 'item1' is an un-regenerated row on a RUNG store recording source_cap 2048, not a rung below 2048"):
        _audit(rung, ["xstest"], tmp_path / "b", siblings=[base])
    relabel(max_tokens=2048, regenerated=True)                  # (c) now "drawn here": binds item1's real 2048 attempt — whose response is not this text
    (v,) = _audit(rung, ["xstest"], tmp_path / "c", siblings=[base])
    assert v["verdict"] == "REFUSED" and "the row's text is not this attempt's response" in v["problem"], v["problem"]
    # (d) the same row when the attempt's response record carries no hash → refused by name, never assumed
    fx.rewrite_jsonl(rung.path / ac.RESPONSE_EVIDENCE_FILE, lambda r: {k: x for k, x in r.items() if k != ac.RESPONSE_TEXT_SHA_FIELD})
    (v2,) = _audit(rung, ["xstest"], tmp_path / "d", siblings=[base])
    assert v2["verdict"] == "REFUSED" and "carries no text_sha256" in v2["problem"], v2["problem"]
    # a BASE store carrying a copied-forward row (source_cap below its cap) → refused: a base's rows are its own cap's
    fx.rewrite_jsonl(bcell / "private" / "gens_xstest.meta.jsonl", lambda r: {**r, "source_cap": 256, "max_tokens": 256} if r["item_id"] == "item0" else r)
    with pytest.raises(ac.AuditRefusal, match="item 'item0' on a BASE store records regenerated False / source_cap 256"):
        _audit(base, ["xstest"], tmp_path / "e")

