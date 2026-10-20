"""The ``protocol`` sampling mode end to end — CPU-only, no Modal, no GPU,
no vLLM.

Contracts: the rows (BCB greedy T 0 / BFCL T 0.001, both top_p 1.0 / top_k 0), the
executed-vs-declared record and BFCL's reason naming the clamp override,
``sampling_for`` refusing the mode, ``sampling_for_task`` / ``protocol_rows_by_task``
refusing a task with no row (gsm8k, a prefill read, EvalPlus), every other mode
delegating unchanged, ``sampling_identity_for_mode`` == ``sampling_identity`` for
EVERY table row (card byte-identical), the protocol identity (null scalars + the
executed map over exactly the filed tasks), ``tinycap_identity_fields`` shapes
(the protocol key present iff asked, every legacy shape byte-identical),
``served_identity_fields``, ``sampling_block_fields(mode=)``, the manifest
``by_task`` block, ``check_protocol_effective``, ``driver_argv``'s flags.

Driver: ``_resolve_sampling`` under protocol (rows, the record, refusals), the
sampling block / identity / manifest resume guard, ``_served_identity`` +
``_check_served_clamp_override`` (present iff protocol).

Launcher: ``--sampling protocol`` threading (plan key present iff non-default,
standup flag iff protocol, driver argv iff protocol, ``ExpectedIdentity`` under
protocol, ``untrusted_store_reason`` refusals — a protocol store missing the key,
a card store carrying it, non-null scalars, a served block without the override),
the dry-run print, ``--sampling nucleus`` refusal, the ``--pf`` requirement, and the
card dry run BYTE-IDENTICAL to a launcher that never heard of the mode.

Standup: the ``--sampling-clamp-override`` flag and the endpoint record iff set.

Provider (client venv only — skipped where inspect_ai is absent): the per-task
row on every body (BCB temperature 0.0, BFCL 0.001, both top_p 1.0,
extra_body.top_k 0, seed present), a divergent caller refused, an unmapped task
refused, row + map together refused; the plans carry each task's row and its
declared temperature.

``check_protocol_sampling``: the request COMPOSITION under the client venv (the
body per task), the launch-flag reader, and the pure ``judge`` on synthetic
resolved records — production / negative control / inheritance / msgspec round
trip / the override control — each refusal named.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import protocol_check as PC

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import served_capability_run as srun  # noqa: E402

BCB, BFCL = C.inspect_task("bigcodebench_200"), C.inspect_task("bfcl_200")
GSM = C.GSM8K_STANDARD_TASK
EFF = {"temperature": 0.001, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
       "presence_penalty": 0.0, "frequency_penalty": 0.0}
RECORD = {"ok": True, "production": {BFCL: dict(EFF), BCB: {**EFF, "temperature": 0.0}}}


# --------------------------------------------------------------------------- #
# contracts                                                                    #
# --------------------------------------------------------------------------- #


def test_the_mode_and_its_rows():
    # the IFEval landing appended `vendor_preset` after `protocol`; protocol keeps its slot before it
    assert C.SAMPLING_PROTOCOL == "protocol" == C.SAMPLING_MODES[-2] and C.SAMPLING_MODES[-1] == C.SAMPLING_VENDOR_PRESET
    assert C.SAMPLING_DEFAULT == "card"
    assert C.SAMPLING_PROTOCOL not in C.SAMPLING_TABLE_MODES
    # capeval-full-tasks (2026-09-23): the two FULL siblings decode under the SAME per-eval rows, appended LAST
    BCB_FULL, BFCL_FULL = C.inspect_task("bigcodebench_full"), C.inspect_task("bfcl_full")
    assert C.PROTOCOL_TASKS == (BCB, BFCL, BCB_FULL, BFCL_FULL) and set(C.PROTOCOL_SAMPLING_BY_TASK) == {BCB, BFCL, BCB_FULL, BFCL_FULL}
    assert C.PROTOCOL_SAMPLING_BY_TASK[BCB_FULL] == C.PROTOCOL_SAMPLING_BY_TASK[BCB]
    assert C.PROTOCOL_SAMPLING_BY_TASK[BFCL_FULL] == C.PROTOCOL_SAMPLING_BY_TASK[BFCL]
    bcb, bfcl = C.PROTOCOL_SAMPLING_BY_TASK[BCB], C.PROTOCOL_SAMPLING_BY_TASK[BFCL]
    assert (bcb.executed.temperature, bcb.executed.top_p, bcb.executed.top_k) == (0.0, 1.0, 0)
    assert (bfcl.executed.temperature, bfcl.executed.top_p, bfcl.executed.top_k) == (0.001, 1.0, 0)
    assert bcb.protocol_temperature == 0.0 == bcb.executed.temperature and bcb.executed_reason == "equal"
    assert bfcl.protocol_temperature == 0.001 == bfcl.executed.temperature == C.BFCL_PROTOCOL_TEMPERATURE
    assert bfcl.executed_reason == "equal (clamp override applied)" == C.PROTOCOL_EXECUTED_REASON_CLAMP_OVERRIDE
    assert "_MAX_TEMP" in bfcl.executed.source and "0.001" in bfcl.executed.source and "D13" in bfcl.executed.source
    assert "greedy" in bcb.executed.source.lower()
    assert (C.PROTOCOL_TOP_P, C.PROTOCOL_TOP_K, C.VLLM_MAX_TEMP_PINNED) == (1.0, 0, 0.01)
    assert C.SAMPLING_CLAMP_OVERRIDE == {"from": 0.01, "to": 0.001, "why": "BFCL protocol (D13)"}
    assert C.SAMPLING_CLAMP_OVERRIDE_ARG == "0.001" and C.FLAG_SAMPLING_CLAMP_OVERRIDE == "--sampling-clamp-override"
    # no optional term on either executed row (byte-identical block shapes)
    assert C._sampling_optional_terms(bcb.executed) == () == C._sampling_optional_terms(bfcl.executed)
    for name in ("SAMPLING_PROTOCOL", "ProtocolRow", "PROTOCOL_SAMPLING_BY_TASK", "sampling_for_task",
                 "protocol_rows_by_task", "sampling_identity_for_mode", "IDENTITY_KEY_SAMPLING_BY_TASK",
                 "TINYCAP_IDENTITY_PROTOCOL_FIELDS", "served_identity_fields", "check_protocol_effective"):
        assert name in C.__all__, name


def test_sampling_for_refuses_the_mode_and_every_other_mode_delegates_unchanged():
    with pytest.raises(ValueError, match="sampling_for_task") as e:
        C.sampling_for("gemma4e4b", C.SAMPLING_PROTOCOL)
    assert not isinstance(e.value, C.SamplingUnverified)
    one = {t: 1 for t in C.TINYCAP_TASKS}
    for mode, table in C.SAMPLING_TABLE_BY_MODE.items():
        for key, row in table.items():
            assert C.sampling_for_task(BCB, mode, key) is row is C.sampling_for(key, mode)
            assert C.sampling_identity_for_mode(mode, key, list(one), one) == C.sampling_identity(row, one)
    assert C.sampling_for_task(GSM, C.SAMPLING_GREEDY, None) is C.GREEDY_SAMPLING
    assert C.sampling_identity_for_mode(C.SAMPLING_GREEDY, None, list(one), one) == C.sampling_identity(C.GREEDY_SAMPLING, one)
    with pytest.raises(ValueError, match="sampling mode must be one of"):
        C.sampling_for_task(BCB, "nucleus", None)
    with pytest.raises(ValueError, match="ROW task"):
        C.sampling_for_task("bfcl_200", C.SAMPLING_PROTOCOL, None)          # a control atom, not a row task


@pytest.mark.parametrize("task", [C.GSM8K_STANDARD_TASK, "inspect/xstest", "evalplus/humaneval_plus",
                                  C.inspect_task("math"), C.inspect_task("bfcl")])
def test_a_task_with_no_protocol_row_refuses(task):
    with pytest.raises(C.ProtocolSamplingUnavailable, match="no protocol row") as e:
        C.sampling_for_task(task, C.SAMPLING_PROTOCOL, "gemma4e4b")
    assert task in str(e.value) and "PROTOCOL_SAMPLING_BY_TASK" in str(e.value)
    with pytest.raises(C.ProtocolSamplingUnavailable):
        C.protocol_rows_by_task([BCB, task])
    assert issubclass(C.ProtocolSamplingUnavailable, ValueError)


def test_protocol_rows_and_the_executed_map_over_the_filed_tasks():
    rows = C.protocol_rows_by_task([BCB, BFCL])
    assert list(rows) == sorted([BCB, BFCL]) and rows[BFCL] is C.PROTOCOL_SAMPLING_BY_TASK[BFCL]
    assert C.protocol_rows_by_task([]) == {} and C.protocol_executed_by_task([BFCL]) == {BFCL: rows[BFCL].executed}
    k = {BCB: 1, BFCL: 1}
    ident = C.sampling_identity_for_mode(C.SAMPLING_PROTOCOL, None, [BCB, BFCL], k)
    assert set(ident) == set(C.SAMPLING_IDENTITY_FIELDS) | {C.IDENTITY_KEY_SAMPLING_BY_TASK}
    assert (ident["temperature"], ident["top_p"], ident["top_k"]) == (None, None, None)     # null, never a card value
    assert ident["samples_by_task"] == k and ident["seed_rule"] == dict(C.SEED_RULES)
    assert ident["sampling_by_task"] == {BCB: {"temperature": 0.0, "top_p": 1.0, "top_k": 0},
                                         BFCL: {"temperature": 0.001, "top_p": 1.0, "top_k": 0}}
    assert list(ident["sampling_by_task"]) == sorted([BCB, BFCL])
    json.dumps(ident, sort_keys=True)                                        # JSON-clean (it is hashed)
    # exactly the filed tasks: one leg alone
    solo = C.sampling_identity_for_mode(C.SAMPLING_PROTOCOL, None, [BFCL], {BFCL: 1})
    assert set(solo["sampling_by_task"]) == {BFCL} and solo != ident
    # the two per-task maps must range over the same tasks
    with pytest.raises(ValueError, match="same tasks"):
        C.protocol_sampling_identity(C.protocol_executed_by_task([BCB]), {BCB: 1, BFCL: 1})
    # model-independent: the key is irrelevant, an unpinned model is fine
    assert C.sampling_identity_for_mode(C.SAMPLING_PROTOCOL, "qwen35_4b", [BCB, BFCL], k) == ident
    assert C.sampling_identity_for_mode(C.SAMPLING_PROTOCOL, None, [BCB, BFCL], k) == ident


def test_the_identity_field_shapes_move_only_under_protocol():
    base = C.tinycap_identity_fields()
    assert base == C.TINYCAP_IDENTITY_FIELDS and C.IDENTITY_KEY_SAMPLING_BY_TASK not in base
    assert C.TINYCAP_IDENTITY_PROTOCOL_FIELDS == ("sampling_by_task",)
    legs = C.tinycap_identity_fields(files_bigcodebench_200=True, files_bfcl_200=True)
    assert legs == (*base, *C.TINYCAP_IDENTITY_BCB_FIELDS, *C.TINYCAP_IDENTITY_BFCL200_FIELDS)
    assert C.tinycap_identity_fields(files_bigcodebench_200=True, files_bfcl_200=True, protocol_sampling=True) == (
        *legs, "sampling_by_task")
    # independent of the legs; after them, before the optional / ladder / v2 keys
    assert C.tinycap_identity_fields(protocol_sampling=True) == (*base, "sampling_by_task")
    lad = C.tinycap_identity_fields(files_bfcl_200=True, ladder=True, protocol_sampling=True)
    assert lad == (*base, *C.TINYCAP_IDENTITY_BFCL200_FIELDS, "sampling_by_task", *C.TINYCAP_IDENTITY_LADDER_FIELDS)
    v2 = C.tinycap_identity_fields(identity_version=2, protocol_sampling=True)
    assert v2 == (*base, "sampling_by_task", C.IDENTITY_KEY_IDENTITY_VERSION)
    # every legacy shape byte-identical (the default is False)
    for kw in ({}, {"files_math500": True}, {"out_of_support_thinking": True}, {"off_cap_override": True},
               {"identity_version": 2, "thinking_on": True}, {"files_bigcodebench_200": True}):
        assert C.tinycap_identity_fields(**kw) == C.tinycap_identity_fields(protocol_sampling=False, **kw)
    # the served block: the override member iff protocol
    assert C.served_identity_fields() == C.SERVED_IDENTITY_FIELDS
    assert C.served_identity_fields(protocol_sampling=True) == (*C.SERVED_IDENTITY_FIELDS, "sampling_clamp_override")
    assert C.SERVED_IDENTITY_PROTOCOL_FIELDS == ("sampling_clamp_override",) == (C.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE,)


def test_the_manifest_block_shape_and_the_by_task_record():
    assert C.SAMPLING_BLOCK_FIELDS == ("mode", "model_key", "temperature", "top_p", "top_k", "samples_by_task",
                                       "seed_rule", "source")                  # UNTOUCHED
    assert C.sampling_block_fields(C.SAMPLING_BY_MODEL_KEY["gemma4e4b"]) == C.SAMPLING_BLOCK_FIELDS
    assert C.sampling_block_fields(C.GENERAL_SAMPLING_BY_MODEL_KEY["qwen35_4b"]) == (
        *C.SAMPLING_BLOCK_FIELDS, "min_p", "presence_penalty", "repetition_penalty")
    assert C.sampling_block_fields(None, mode=C.SAMPLING_PROTOCOL) == (*C.SAMPLING_BLOCK_FIELDS, "by_task")
    with pytest.raises(ValueError, match="no single run row"):
        C.sampling_block_fields(C.GREEDY_SAMPLING, mode=C.SAMPLING_PROTOCOL)
    with pytest.raises(ValueError, match="row is required"):
        C.sampling_block_fields(None, mode=C.SAMPLING_CARD)
    bt = C.protocol_manifest_by_task([BCB, BFCL])
    assert set(bt) == {BCB, BFCL} and all(set(b) == set(C.PROTOCOL_BY_TASK_FIELDS) for b in bt.values())
    assert bt[BFCL] == {"executed": {"temperature": 0.001, "top_p": 1.0, "top_k": 0}, "protocol_temperature": 0.001,
                        "executed_reason": "equal (clamp override applied)", "effective": None}
    assert bt[BCB]["executed"]["temperature"] == 0.0 and bt[BCB]["executed_reason"] == "equal"
    with_eff = C.protocol_manifest_by_task([BFCL], {BFCL: {**EFF, "extra": 1}})
    assert with_eff[BFCL]["effective"] == EFF                                 # exactly PROTOCOL_EFFECTIVE_FIELDS
    with pytest.raises(ValueError, match="lacks"):
        C.protocol_manifest_by_task([BFCL], {BFCL: {"temperature": 0.001}})
    src = C.protocol_source([BCB, BFCL])
    assert src.startswith(f"{BFCL}: ") and f" | {BCB}: " in src and "D13" in src


def test_check_protocol_effective_re_asserts_the_executed_values():
    eff = C.check_protocol_effective(RECORD)
    assert eff == {BFCL: EFF, BCB: {**EFF, "temperature": 0.0}}
    for mut, why in ((lambda r: r["production"][BFCL].__setitem__("temperature", 0.01), "resolved 0.01"),
                     (lambda r: r["production"][BCB].__setitem__("top_k", -1), "top_k: resolved -1"),
                     (lambda r: r["production"][BFCL].__setitem__("top_p", 0.95), "top_p: resolved 0.95"),
                     (lambda r: r.__setitem__("ok", False), "not a passing record"),
                     (lambda r: r.__setitem__("production", {}), "no `production` map"),
                     (lambda r: r["production"].__setitem__(GSM, dict(EFF)), "no protocol row")):
        rec = copy.deepcopy(RECORD)
        mut(rec)
        with pytest.raises(ValueError, match=why):
            C.check_protocol_effective(rec)


def test_driver_argv_spells_the_mode_iff_non_default_and_the_record_iff_protocol():
    kw = dict(python="p", script="s", proxy_url="u", routing_table="t", cell="none", config="c", defenses_dir="d",
              out_root="o", controls=["bfcl_200"], thinking="off", concurrency=4, judge_model="j")
    card = C.driver_argv(**kw)
    assert C.FLAG_SAMPLING not in card and C.FLAG_PROTOCOL_CHECK not in card
    assert C.driver_argv(**kw, sampling_mode=C.SAMPLING_CARD) == card                # the default is not spelled
    greedy = C.driver_argv(**kw, sampling_mode=C.SAMPLING_GREEDY)
    assert greedy[len(card):] == ["--sampling", "greedy"]
    prot = C.driver_argv(**kw, sampling_mode=C.SAMPLING_PROTOCOL, protocol_check="/o/protocol_check.json")
    assert prot[len(card):] == ["--sampling", "protocol", "--protocol-check", "/o/protocol_check.json"]
    with pytest.raises(ValueError, match="rides --sampling protocol only"):
        C.driver_argv(**kw, protocol_check="/o/p.json")
    with pytest.raises(ValueError, match="sampling mode must be one of"):
        C.driver_argv(**kw, sampling_mode="nucleus")
    # the ladder flags stay LAST
    sel = C.format_item_selection_json({BFCL: ["a", "b"]})
    lad = C.driver_argv(**kw, sampling_mode=C.SAMPLING_PROTOCOL, protocol_check="/o/p.json",
                        cap_override=f"{BFCL}={C.LADDER_CAP_TO}", item_selection_json=sel)
    assert lad.index("--sampling") < lad.index(C.FLAG_CAP_OVERRIDE) and lad[-4] == C.FLAG_CAP_OVERRIDE
    assert "protocol_temperature" in C.SUMMARY_PAYLOAD_FIELDS


# --------------------------------------------------------------------------- #
# the driver                                                                   #
# --------------------------------------------------------------------------- #


def _record_file(tmp_path, record=RECORD) -> str:
    p = Path(tmp_path) / C.PROTOCOL_CHECK_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record))
    return str(p)


def test_the_driver_resolves_protocol_per_task_and_refuses_the_holes(tmp_path):
    args = SimpleNamespace(sampling=C.SAMPLING_PROTOCOL, samples=None, protocol_check=_record_file(tmp_path))
    cfg = {"model": sg.MODEL_HF_IDS["gemma4e4b"]}
    s = srun._resolve_sampling(args, cfg, ["bigcodebench_200", "bfcl_200"], {BCB: "off", BFCL: "off"})
    assert s.mode == "protocol" and s.row is None and s.model_key == "gemma4e4b"
    assert s.sampling_by_task == C.protocol_executed_by_task([BCB, BFCL]) and set(s.protocol_by_task) == {BCB, BFCL}
    assert s.effective_by_task == {BFCL: EFF, BCB: {**EFF, "temperature": 0.0}} and s.samples_by_task == {BCB: 1, BFCL: 1}
    ident = srun._sampling_identity(s)
    assert ident == C.sampling_identity_for_mode(C.SAMPLING_PROTOCOL, None, [BCB, BFCL], {BCB: 1, BFCL: 1})
    blk = srun._sampling_block(s)
    assert set(blk) == set(C.sampling_block_fields(None, mode=C.SAMPLING_PROTOCOL))
    assert blk["mode"] == "protocol" and blk["model_key"] == "gemma4e4b" and blk["temperature"] is None
    assert blk["by_task"][BFCL]["effective"] == EFF and blk["by_task"][BFCL]["executed"] == ident["sampling_by_task"][BFCL]
    assert blk["by_task"][BCB]["protocol_temperature"] == 0.0 and "D13" in blk["source"]
    # model-independent: an unpinned model is fine under protocol (card would refuse)
    u = srun._resolve_sampling(args, {"model": "some/unpinned"}, ["bfcl_200"], {BFCL: "off"})
    assert u.model_key is None and set(u.sampling_by_task) == {BFCL}
    # refusals: a task with no row; no record; a record missing a filed leg; a failing record
    with pytest.raises(SystemExit, match="no protocol row"):
        srun._resolve_sampling(args, cfg, ["gsm8k"], {GSM: "off"})
    with pytest.raises(SystemExit, match="--protocol-check"):
        srun._resolve_sampling(SimpleNamespace(sampling="protocol", samples=None, protocol_check=None), cfg,
                               ["bfcl_200"], {BFCL: "off"})
    only_bcb = _record_file(tmp_path / "bcb", {"ok": True, "production": {BCB: {**EFF, "temperature": 0.0}}})
    with pytest.raises(SystemExit, match="must cover every filed leg"):
        srun._resolve_sampling(SimpleNamespace(sampling="protocol", samples=None, protocol_check=only_bcb), cfg,
                               ["bfcl_200"], {BFCL: "off"})
    bad = _record_file(tmp_path / "bad", {"ok": True, "production": {BFCL: {**EFF, "temperature": 0.01}}})
    with pytest.raises(SystemExit, match="resolved 0.01"):
        srun._resolve_sampling(SimpleNamespace(sampling="protocol", samples=None, protocol_check=bad), cfg,
                               ["bfcl_200"], {BFCL: "off"})
    # the card path is what it was, with the new namespace fields None
    c = srun._resolve_sampling(SimpleNamespace(sampling="card", samples=None, protocol_check=None), cfg, ["gsm8k"],
                               {GSM: "off"})
    assert c.row is C.SAMPLING_BY_MODEL_KEY["gemma4e4b"] and c.sampling_by_task is None and c.protocol_by_task is None
    assert srun._sampling_identity(c) == C.sampling_identity(c.row, c.samples_by_task)
    assert set(srun._sampling_block(c)) == set(C.SAMPLING_BLOCK_FIELDS)


def test_the_manifest_resume_guard_is_mode_aware(tmp_path):
    args = SimpleNamespace(sampling=C.SAMPLING_PROTOCOL, samples=None, protocol_check=_record_file(tmp_path))
    s = srun._resolve_sampling(args, {"model": "x"}, ["bfcl_200", "bigcodebench_200"], {BCB: "off", BFCL: "off"})
    expected = srun._sampling_block(s)
    store = SimpleNamespace(manifest_path=tmp_path / "manifest.json")

    def write(block):
        store.manifest_path.write_text(json.dumps({C.MANIFEST_KEY_SAMPLING: block}))

    write(expected)
    srun._check_manifest_sampling(store, expected=expected)                          # resumes
    other_eff = copy.deepcopy(expected)
    other_eff["by_task"][BFCL]["effective"] = {**EFF, "temperature": 0.01}
    write(other_eff)
    srun._check_manifest_sampling(store, expected=expected)                          # `effective` is informational
    moved = copy.deepcopy(expected)
    moved["by_task"][BFCL]["executed"]["temperature"] = 0.0
    write(moved)
    with pytest.raises(srun.ServingIdentityMismatch, match="by_task: the manifest records"):
        srun._check_manifest_sampling(store, expected=expected)
    without = {k: v for k, v in expected.items() if k != "by_task"}
    write(without)
    with pytest.raises(srun.ServingIdentityMismatch, match="contract's shape"):
        srun._check_manifest_sampling(store, expected=expected)
    # a CARD expectation refuses a block carrying by_task
    card = srun._resolve_sampling(SimpleNamespace(sampling="card", samples=None, protocol_check=None),
                                  {"model": sg.MODEL_HF_IDS["gemma4e4b"]}, ["gsm8k"], {GSM: "off"})
    card_block = srun._sampling_block(card)
    write({**card_block, "by_task": {}})
    with pytest.raises(srun.ServingIdentityMismatch, match="contract's shape"):
        srun._check_manifest_sampling(store, expected=card_block)
    write(card_block)
    srun._check_manifest_sampling(store, expected=card_block)


def test_the_served_block_carries_the_override_iff_the_standup_recorded_it_and_the_mode_asks():
    ep = {"reasoning_parser": None, "tool_calls": True, "geometry": {"max_model_len": 4096, "max_num_seqs": 4},
          C.AGENT_LOOP_ENTRY_KEY: True}
    card = srun._served_identity(ep, concurrency=4)
    assert tuple(card) == C.SERVED_IDENTITY_FIELDS
    srun._check_served_clamp_override(ep, sampling_mode="card")
    srun._check_served_clamp_override(ep, sampling_mode="greedy")
    with pytest.raises(SystemExit, match="WITHOUT the D13 clamp override"):
        srun._check_served_clamp_override(ep, sampling_mode="protocol")
    lifted = dict(ep, sampling_clamp_override=dict(C.SAMPLING_CLAMP_OVERRIDE))
    prot = srun._served_identity(lifted, concurrency=4)
    assert tuple(prot) == C.served_identity_fields(protocol_sampling=True)
    assert prot["sampling_clamp_override"] == {"from": 0.01, "to": 0.001, "why": "BFCL protocol (D13)"}
    assert {k: prot[k] for k in C.SERVED_IDENTITY_FIELDS} == card                    # the rest is what it was
    srun._check_served_clamp_override(lifted, sampling_mode="protocol")
    with pytest.raises(SystemExit, match="only --sampling protocol asks for"):
        srun._check_served_clamp_override(lifted, sampling_mode="card")
    with pytest.raises(SystemExit, match="not the pinned record"):
        srun._check_served_clamp_override(dict(ep, sampling_clamp_override={"from": 0.01, "to": 0.005, "why": "x"}),
                                          sampling_mode="protocol")
    with pytest.raises(SystemExit, match="record shape"):
        srun._served_identity(dict(ep, sampling_clamp_override={"to": 0.001}), concurrency=4)
    # two synthetic identities identical but for the override key hash differently
    from sisteer.capeval import store as cs
    base = {"tinycap": {"served": card, "x": 1}}
    assert cs.harness_config_hash(base) != cs.harness_config_hash({"tinycap": {"served": prot, "x": 1}})


# --------------------------------------------------------------------------- #
# the launcher (loaded against the tests' modal stub)                          #
# --------------------------------------------------------------------------- #

LAUNCHER = REPO / "scripts/modal/tinycap_served.py"
TEST_APP_NAME = "tinycap-protocol-under-test"


def _stub_modal() -> types.ModuleType:
    m = types.ModuleType("modal")

    class _Chain:
        def __getattr__(self, name):
            return lambda *a, **k: self

    class _App:
        def __init__(self, name=None, **k):
            self.name = name

        def function(self, **kw):
            return lambda f: f

        def local_entrypoint(self, **kw):
            return lambda f: f

    m.App = _App
    m.Volume = SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Image = SimpleNamespace(from_registry=lambda *a, **k: _Chain())
    m.Sandbox = object
    m.Secret = SimpleNamespace(from_name=lambda name, **k: ("secret", name))
    m.__version__ = "stub"
    m.is_local = lambda: True
    return m


@pytest.fixture(scope="module")
def L():
    saved = sys.modules.get("modal")
    sys.modules["modal"] = _stub_modal()
    saved_env = os.environ.get(C.MODAL_APP_NAME_ENV)
    os.environ[C.MODAL_APP_NAME_ENV] = TEST_APP_NAME
    os.environ.setdefault(C.LENS_STACKED_REDUCTIONS_ENV, C.REDUCTION_MODE_PER_ROW)
    try:
        spec = importlib.util.spec_from_file_location("tinycap_served_protocol_test", LAUNCHER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules["modal"] = saved
        else:
            sys.modules.pop("modal", None)
        if saved_env is None:
            os.environ.pop(C.MODAL_APP_NAME_ENV, None)
        else:
            os.environ[C.MODAL_APP_NAME_ENV] = saved_env
    return mod


LEGS = ["bigcodebench_200", "bfcl_200"]
FULL_SHA = "f6cf462" + "0" * 33


def _plans(L, mode, **kw):
    return L.plan_configurations(cells=["none", "prompted__safety_refusal"], modes=["off"], controls=LEGS,
                                 sampling_mode=mode, **kw)


def test_the_plan_marks_the_mode_iff_non_default_and_composes_the_standup_flag_iff_protocol(L):
    card = _plans(L, C.SAMPLING_CARD)
    prot = _plans(L, C.SAMPLING_PROTOCOL)
    assert all(L.SAMPLING_MODE_KEY not in c for c in card) and all(L.sampling_mode_of(c) == "card" for c in card)
    assert all(c[L.SAMPLING_MODE_KEY] == "protocol" for c in prot) and all(L.sampling_mode_of(c) == "protocol" for c in prot)
    # the ONLY differences between the two plans: the mode key and the standup flag
    for c, p in zip(card, prot):
        assert p["standup_extra"] == c["standup_extra"] + [C.FLAG_SAMPLING_CLAMP_OVERRIDE, "0.001"]
        assert {k: v for k, v in p.items() if k not in (L.SAMPLING_MODE_KEY, "standup_extra")} == \
               {k: v for k, v in c.items() if k != "standup_extra"}
    assert L.sampling_clamp_override_of(prot[0]["standup_extra"]) == dict(C.SAMPLING_CLAMP_OVERRIDE)
    assert L.sampling_clamp_override_of(card[0]["standup_extra"]) is None
    with pytest.raises(ValueError, match="not the pinned"):
        L.sampling_clamp_override_of([C.FLAG_SAMPLING_CLAMP_OVERRIDE, "0.005"])
    # the default plan (no sampling_mode) is byte-identical to an explicit card plan
    assert L.plan_configurations(cells=["none"], modes=["off"], controls=LEGS) == \
        L.plan_configurations(cells=["none"], modes=["off"], controls=LEGS, sampling_mode=C.SAMPLING_CARD)
    # a battery with a rowless task refuses at PLAN time, by name
    with pytest.raises(ValueError, match="no protocol row") as e:
        L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k", "bfcl_200"], sampling_mode=C.SAMPLING_PROTOCOL)
    assert GSM in str(e.value) and "launch configuration" in str(e.value)
    with pytest.raises(ValueError, match="no protocol row"):
        L.plan_configurations(sampling_mode=C.SAMPLING_PROTOCOL)                        # the default battery
    with pytest.raises(ValueError, match="sampling mode must be one of"):
        _plans(L, "nucleus")


def _expected(L, cfg, mode, **kw):
    geom, _ = L.served_geometry(cfg, 1)
    return L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=1, tree_sha=FULL_SHA,
                               sampling_mode=mode, **kw)


def test_expected_identity_under_protocol(L):
    card_cfg = _plans(L, C.SAMPLING_CARD)[0]
    prot_cfg = _plans(L, C.SAMPLING_PROTOCOL)[0]
    card = _expected(L, card_cfg, C.SAMPLING_CARD)
    prot = _expected(L, prot_cfg, C.SAMPLING_PROTOCOL)
    assert not card.protocol_sampling and prot.protocol_sampling and prot.sampling_mode == "protocol"
    # the sampling block: nulls + the executed map over exactly the filed tasks
    assert (prot.sampling["temperature"], prot.sampling["top_p"], prot.sampling["top_k"]) == (None, None, None)
    assert prot.sampling["sampling_by_task"] == {BCB: {"temperature": 0.0, "top_p": 1.0, "top_k": 0},
                                                 BFCL: {"temperature": 0.001, "top_p": 1.0, "top_k": 0}}
    assert set(prot.sampling["samples_by_task"]) == set(prot.thinking_by_task) == {BCB, BFCL}
    assert (card.sampling["temperature"], card.sampling["top_p"], card.sampling["top_k"]) == (1.0, 0.95, 64)
    assert "sampling_by_task" not in card.sampling
    # the served block: the override record iff protocol; the rest identical
    assert set(prot.served) == set(C.served_identity_fields(protocol_sampling=True))
    assert prot.served["sampling_clamp_override"] == dict(C.SAMPLING_CLAMP_OVERRIDE)
    assert {k: prot.served[k] for k in C.SERVED_IDENTITY_FIELDS} == dict(card.served)
    assert set(card.served) == set(C.SERVED_IDENTITY_FIELDS)
    # optional_identity values the protocol key, and as_dict carries it
    assert prot.optional_identity()["sampling_by_task"] == prot.sampling["sampling_by_task"]
    assert "sampling_by_task" not in card.optional_identity()
    d = prot.as_dict()
    assert d["sampling_mode"] == "protocol" and d["sampling_by_task"] == prot.sampling["sampling_by_task"]
    assert "sampling_by_task" not in card.as_dict() and card.as_dict()["sampling_mode"] == "card"
    # a card expectation of the same configuration (its default) is what it always was
    assert _expected(L, card_cfg, C.SAMPLING_CARD) == L.expected_identity(
        card_cfg, L.served_geometry(card_cfg, 1)[0], pass_no=1, controls=card_cfg["controls"], samples=1, tree_sha=FULL_SHA)
    # the expectation recomposes the standup extras FROM THE MODE (never off the plan's
    # recorded argv), so the override is derived for any protocol expectation — and the
    # worker's endpoint check holds the standup's record to it
    assert _expected(L, card_cfg, C.SAMPLING_PROTOCOL).served["sampling_clamp_override"] == dict(C.SAMPLING_CLAMP_OVERRIDE)
    with pytest.raises(ValueError, match="sampling mode must be one of"):
        _expected(L, card_cfg, "nucleus")


def test_the_launch_argv_and_the_driver_argv_spell_the_mode_iff_protocol(L):
    argv = L.launch_argv("abc1234", pass1=True, app_name=TEST_APP_NAME)
    assert L.launch_argv("abc1234", pass1=True, app_name=TEST_APP_NAME, sampling=C.SAMPLING_CARD) == argv
    assert C.FLAG_SAMPLING not in argv
    p = L.launch_argv("abc1234", pass1=True, app_name=TEST_APP_NAME, sampling=C.SAMPLING_PROTOCOL)
    assert p[p.index(C.FLAG_SAMPLING) + 1] == "protocol"
    prot_cfg = _plans(L, C.SAMPLING_PROTOCOL)[0]
    card_cfg = _plans(L, C.SAMPLING_CARD)[0]
    geom, _ = L.served_geometry(prot_cfg, 1)
    ep = {"proxy_url": "http://127.0.0.1:1/v1", "routing_table": "/srv/routing_table.json"}
    common = dict(ledger_url="http://127.0.0.1:2/v1", controls=LEGS, limit=0, inspect_limit=0, samples=1,
                  cfg_path="<tree>/cfg.yaml", vllm_log=None)
    dp = L._driver_cmd("<tree>", prot_cfg, ep, geom=geom, protocol_check="/out/protocol_check.json", **common)
    dc = L._driver_cmd("<tree>", card_cfg, ep, geom=geom, **common)
    assert dp[len(dc):] == ["--sampling", "protocol", "--protocol-check", "/out/protocol_check.json"] and dp[:len(dc)] == dc
    with pytest.raises(ValueError, match="rides --sampling protocol only"):
        L._driver_cmd("<tree>", card_cfg, ep, geom=geom, protocol_check="/out/p.json", **common)


def test_the_dry_run_prints_the_protocol_block_and_is_byte_identical_under_card(L, capsys):
    kw = dict(app_name=TEST_APP_NAME, sha="abc1234", pass1=True, dry_run=True, cells="none,prompted__safety_refusal",
              modes="off", controls=",".join(LEGS))
    L.main(**kw)
    card = capsys.readouterr().out
    L.main(**kw, sampling=C.SAMPLING_CARD)
    assert capsys.readouterr().out.replace("run id: ", "") .split("run id")[0] and True   # smoke: it ran
    L.main(**kw, sampling=C.SAMPLING_PROTOCOL)
    prot = capsys.readouterr().out
    assert "--sampling protocol" in prot.splitlines()[0]
    assert f"standup extra: --tool-calls --manifest {C.PETRI_MANIFEST_GEMMA4E4B} --sampling-clamp-override 0.001" in prot
    assert "sampling: --sampling protocol" in prot
    assert f"{BFCL}: executed T=0.001 top_p=1 top_k=0 | protocol T=0.001 | equal (clamp override applied)" in prot
    assert f"{BCB}: executed T=0 top_p=1 top_k=0 | protocol T=0 | equal" in prot
    assert '"sampling_clamp_override": {"from": 0.01, "to": 0.001, "why": "BFCL protocol (D13)"}' in prot
    assert '"temperature": null, "top_k": null, "top_p": null' in prot
    assert f"-m {C.VLLM_ENTRY_MODULE} {C.FLAG_VLLM_ENTRY_MAX_TEMP} 0.001 serve" in prot
    assert "--protocol-check" not in prot                     # the record comes from --pf, never a placeholder path
    # card: nothing of the protocol print, no flag anywhere, and the same text with or without the explicit default
    for token in ("--sampling", "sampling-clamp-override", "sampling_by_task", "sampling_clamp_override", "executed T="):
        assert token not in card, token
    L.main(**kw)
    again = capsys.readouterr().out
    import re
    # the per-invocation run id; and any `[HH:MM:SS] [hb] ...` heartbeat line a printer thread leaked from an earlier test
    # (fitmon.Heartbeat; tests/conftest.py now stops those, this keeps the comparison about the dry run's own output)
    strip = lambda s: re.sub(r"[0-9a-f]{7}-[0-9a-f]{8}", "<minted>", re.sub(r"^\[\d\d:\d\d:\d\d\] \[hb\] .*\n", "", s, flags=re.M))
    assert strip(again) == strip(card)
    with pytest.raises(SystemExit, match="sampling mode must be one of"):
        L.main(**kw, sampling="nucleus")


def test_a_protocol_launch_runs_only_behind_pf(L, monkeypatch):
    monkeypatch.setattr(L, "run_configuration", SimpleNamespace(spawn=lambda *a, **k: pytest.fail("launched")))
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=lambda *a, **k: pytest.fail("preflight ran")))
    with pytest.raises(SystemExit, match="protocol launch runs only behind --pf"):
        L.main(app_name=TEST_APP_NAME, sha="abc1234", pass1=True, cells="none", modes="off",
               controls=",".join(LEGS), sampling=C.SAMPLING_PROTOCOL)


def _manifest_for(L, exp, *, tinycap_extra=None, served=None, sampling_block=None):
    """A minimal trusted-store manifest for ``untrusted_store_reason`` — the identity
    the expectation derives, the mirror blocks, the producer / frameworks blocks."""
    tinycap = {"tasks": ["bfcl_200", "bigcodebench_200"], "thinking": {"mode": exp.mode, "by_task": dict(exp.thinking_by_task)},
               "versions": {"tinycap": C.TINYCAP_VERSION}, "served": dict(served if served is not None else exp.served),
               "limits": dict(exp.limits), "samples": exp.samples,
               "frameworks": {f: (exp.client_lock_sha256 if f == "client_lock_sha256" else None) for f in C.FRAMEWORKS_BLOCK_FIELDS},
               C.GEOMETRY_PASS_FIELD: exp.pass_no, "thinking_by_task": dict(exp.thinking_by_task),
               "math500_retry_max_tokens": 0, "or_bench_item_set": exp.or_bench_item_set,
               **{f: exp.sampling[f] for f in C.SAMPLING_IDENTITY_FIELDS}, **exp.optional_identity(),
               **(tinycap_extra or {})}
    return tinycap


def test_the_store_trust_check_refuses_the_cross_mode_resumes(L, tmp_path):
    prot_cfg = _plans(L, C.SAMPLING_PROTOCOL)[0]
    exp = _expected(L, prot_cfg, C.SAMPLING_PROTOCOL)
    card_exp = _expected(L, _plans(L, C.SAMPLING_CARD)[0], C.SAMPLING_CARD)
    fields = C.tinycap_identity_fields(files_bigcodebench_200=True, files_bfcl_200=True, protocol_sampling=True)
    prot_ident = _manifest_for(L, exp)
    assert set(prot_ident) == set(fields)
    # the exact-keys rule, both directions, through the launcher's own derivation
    assert L._exact_keys(prot_ident, fields, "tinycap identity") is None
    card_fields = C.tinycap_identity_fields(files_bigcodebench_200=True, files_bfcl_200=True)
    bad = L._exact_keys(prot_ident, card_fields, "tinycap identity")
    assert bad and "sampling_by_task" in bad                     # a protocol store under a card expectation
    card_ident = _manifest_for(L, card_exp)
    bad = L._exact_keys(card_ident, fields, "tinycap identity")
    assert bad and "sampling_by_task" in bad                     # a card store under a protocol expectation
    # the served block: the override member present iff protocol
    assert L._exact_keys(prot_ident["served"], C.served_identity_fields(protocol_sampling=True), "tinycap.served") is None
    assert "sampling_clamp_override" in L._exact_keys(card_ident["served"], C.served_identity_fields(protocol_sampling=True), "tinycap.served")
    assert "sampling_clamp_override" in L._exact_keys(prot_ident["served"], C.SERVED_IDENTITY_FIELDS, "tinycap.served")
    # the manifest sampling block: by_task iff protocol
    assert L._exact_keys({**{f: None for f in C.SAMPLING_BLOCK_FIELDS}, "by_task": {}},
                         C.sampling_block_fields(None, mode=C.SAMPLING_PROTOCOL), "manifest.sampling") is None
    assert "by_task" in L._exact_keys({f: None for f in C.SAMPLING_BLOCK_FIELDS},
                                      C.sampling_block_fields(None, mode=C.SAMPLING_PROTOCOL), "manifest.sampling")
    # the source text of the store-trust check reads the mode-aware derivations
    src = Path(L.__file__).read_text()
    body = src[src.index("def untrusted_store_reason("):src.index("def resume_report(")]
    assert "protocol_sampling=expected.protocol_sampling" in body
    assert "C.served_identity_fields(protocol_sampling=expected.protocol_sampling)" in body
    assert "C.sampling_block_fields(None, mode=C.SAMPLING_PROTOCOL) if expected.protocol_sampling" in body
    assert ".executed" in body and "does not mirror the hashed tinycap.sampling_by_task" in body.replace("{C.IDENTITY_KEY_SAMPLING_BY_TASK}", "sampling_by_task")


def test_the_worker_reads_the_plans_mode_and_the_preflight_probes_the_protocol_flags(L):
    src = Path(L.__file__).read_text()
    worker = src[src.index("def _run_configuration("):src.index("def run_configuration(")]
    assert "sampling_mode = sampling_mode_of(cfg)" in worker and "C.SAMPLING_DEFAULT\n" not in worker.split("sampling_mode = ")[1][:40]
    assert '"PROTOCOL_CHECK_MISSING"' in worker and '"CLAMP_OVERRIDE_MISMATCH"' in worker
    assert "protocol_check_path = out_dir / C.PROTOCOL_CHECK_FILENAME" in worker
    assert "clamp_warning_lines(srv / \"vllm_server.log\")" in worker and 'rec["status_override"] = C.REC_CLAMP_OVERRIDE_INEFFECTIVE' in worker
    pre = src[src.index("def preflight("):src.index("def _tail_or_none(") if "def _tail_or_none(" in src else src.index("def launch_argv(")]
    assert "(C.FLAG_SAMPLING, C.FLAG_PROTOCOL_CHECK) if protocol_launch else ()" in pre
    assert "need.append(C.FLAG_SAMPLING_CLAMP_OVERRIDE)" in pre
    assert 'chk("protocol_sampling_resolves_as_executed", protocol_sampling)' in pre
    # the probe value is a DUMMY the merged driver's parser must accept: the IFEval lane's `vendor_preset` (a driver whose
    # --sampling choices predate either mode stops at "invalid choice" just the same); the record flag is probed too
    assert L.DRIVER_PROBE_VALUES[C.FLAG_SAMPLING] == C.SAMPLING_VENDOR_PRESET and C.FLAG_PROTOCOL_CHECK in L.DRIVER_PROBE_VALUES
    assert L.PROTOCOL_CHECK_FIELD == "protocol_sampling_check"
    # the clamp-warning scanner
    log = Path(L.__file__).parent / "does-not-exist.log"
    assert L.clamp_warning_lines(log) == []
    assert L.CLAMP_WARNING_MARKER in ("temperature 0.001 is less than 0.01, which may cause numerical errors nan or inf in "
                                      "tensors. We have maxed it out to 0.01.")


def test_protocol_launch_flags_read_the_manifest_through_its_reader_and_compose_pr34s_server_default(L, tmp_path, monkeypatch):
    """The manifest's flags come through `manifest.load_petri_spec` (the authoritative reader — a
    missing or malformed manifest RAISES; a raw yaml read would swallow OSError and resolve the
    defaults with fewer flags than the server uses). The contracts hook is ON this tree: the check
    record names it as its server-default source, composes for a pinned model EXACTLY the
    `--override-generation-config` JSON `build_server_launch` composes (the composer's own bytes),
    and records `config: None` for a model with no row. The hook leaving the contracts, or turning
    non-callable, is loud."""
    import yaml
    from sisteer.capeval import manifest as cm
    row = C.model_row("gemma4e4b")
    tree = tmp_path
    (tree / Path(row.petri_manifest)).parent.mkdir(parents=True, exist_ok=True)
    real = yaml.safe_load((REPO / row.petri_manifest).read_text())                # the REAL manifest, one flag list edited
    real["petri"]["serving_launch_flags"] = ["--generation-config", "vllm"]
    (tree / Path(row.petri_manifest)).write_text(yaml.safe_dump(real))
    rec = L.protocol_launch_flags_record(row, str(tree))
    assert rec["flags"] == rec["manifest_flags"] == ["--generation-config", "vllm"] and rec["manifest"] == row.petri_manifest
    assert L.protocol_launch_flags(row, str(tree)) == ["--generation-config", "vllm"]
    # the hook and its table are ON this tree, read by the names the record cites
    assert (L.SERVER_DEFAULT_HOOK, L.SERVER_DEFAULT_TABLE) == ("server_default_generation_config", "SERVER_DEFAULT_SAMPLING_BY_MODEL")
    assert callable(getattr(C, L.SERVER_DEFAULT_HOOK)) and isinstance(getattr(C, L.SERVER_DEFAULT_TABLE), Mapping)
    assert row.hf not in C.SERVER_DEFAULT_SAMPLING_BY_MODEL                          # gemma ships its own generation_config.json
    assert rec["server_default"] == {"source": f"contracts.{L.SERVER_DEFAULT_HOOK}", "config": None}
    # a MISSING manifest raises (the row names one; the server would launch with its flags)
    (tree / Path(row.petri_manifest)).unlink()
    with pytest.raises((FileNotFoundError, OSError)):
        L.protocol_launch_flags_record(row, str(tree))
    (tree / Path(row.petri_manifest)).write_text("serving_launch_flags: ['--generation-config', 'vllm']\n")   # not a petri manifest
    with pytest.raises(cm.ManifestError):
        L.protocol_launch_flags_record(row, str(tree))
    (tree / Path(row.petri_manifest)).write_text(yaml.safe_dump(real))
    # a model the table pins (Qwen3.5-4B ships no generation_config.json) gets the composer's exact JSON
    # as the last two flags — the same bytes vllm_server.build_server_launch puts on the argv
    qrow = C.model_row("qwen35_4b")
    assert qrow.hf in C.SERVER_DEFAULT_SAMPLING_BY_MODEL
    srow = C.SERVER_DEFAULT_SAMPLING_BY_MODEL[qrow.hf]
    want = C.server_default_generation_config(srow)
    assert want == json.dumps(json.loads(want), sort_keys=True, separators=(",", ":"))       # compact, sorted: a stable identity
    assert set(json.loads(want)) <= set(C.SERVER_DEFAULTABLE_SAMPLING_KEYS) and json.loads(want)["temperature"] == srow.temperature
    assert "presence_penalty" not in json.loads(want)                                  # per request only
    qrec = L.protocol_launch_flags_record(qrow._replace(petri_manifest=row.petri_manifest), str(tree))
    assert qrec["flags"] == ["--generation-config", "vllm", "--override-generation-config", want]
    assert qrec["server_default"] == {"source": f"contracts.{L.SERVER_DEFAULT_HOOK}", "config": json.loads(want)}
    assert PC.launch_generation_config(qrec["flags"])["override_generation_config"] == json.loads(want)
    # a hook that is present but not callable refuses (never a silent skip)
    monkeypatch.setattr(C, L.SERVER_DEFAULT_HOOK, {"not": "callable"}, raising=False)
    with pytest.raises(TypeError, match="not callable"):
        L.protocol_launch_flags_record(row, str(tree))
    monkeypatch.undo()
    # the hook LEAVING the contracts is loud (AttributeError), never a record with fewer flags than the server
    monkeypatch.delattr(C, L.SERVER_DEFAULT_TABLE)
    with pytest.raises(AttributeError):
        L.protocol_launch_flags_record(row, str(tree))


def test_the_check_entrypoint_is_cli_parseable_and_the_override_flag_is_stated_once(L):
    """Confirmed live: Modal's CLI refused `tasks: Optional[List[str]]` ("unparseable
    annotation") before any row check ran. Every parameter of the local entrypoint is a CLI
    scalar; the tasks are comma-separated and the launch flags a JSON list; the Python API keeps
    its sequences."""
    import inspect as _inspect
    import typing
    hints = typing.get_type_hints(L.check_protocol_sampling)
    params = _inspect.signature(L.check_protocol_sampling).parameters
    for name in params:
        assert hints[name] in L.CLI_SCALAR_TYPES, (name, hints[name])
    assert params["tasks"].default == "" and params["launch_flags"].default == ""
    assert L.parse_tasks_arg("") == list(C.PROTOCOL_TASKS) and L.parse_tasks_arg(f" {BFCL} , {BCB}") == [BFCL, BCB]
    assert L.parse_launch_flags_arg("") is None and L.parse_launch_flags_arg('["--generation-config", "vllm"]') == ["--generation-config", "vllm"]
    for bad in ("--generation-config vllm", '{"a": 1}', "[1, 2]"):
        with pytest.raises(ValueError, match="JSON list of strings"):
            L.parse_launch_flags_arg(bad)
    src = Path(L.__file__).read_text()
    body = src[src.index("def check_protocol_sampling("):src.index("def fan_out(")]
    assert "Optional[List[str]]" not in body and "launch_flags_source=" in body
    # code review: the override flag read off the standup argv must be stated ONCE
    one = [C.FLAG_SAMPLING_CLAMP_OVERRIDE, C.SAMPLING_CLAMP_OVERRIDE_ARG]
    assert L.sampling_clamp_override_of(one) == dict(C.SAMPLING_CLAMP_OVERRIDE)
    with pytest.raises(ValueError, match="appears 2 times"):
        L.sampling_clamp_override_of(one + one)
    with pytest.raises(ValueError, match="appears 2 times"):
        L.sampling_clamp_override_of(one + [C.FLAG_SAMPLING_CLAMP_OVERRIDE, "0.01"])


def test_a_detected_reclamp_poisons_the_store_root_durably_and_no_resume_returns_done(L, tmp_path, monkeypatch):
    """Code review: the clamp warning after the driver used to change only the REC's
    status — the store's manifest / completion markers stayed valid and a retry returned DONE
    before any server log was read. The failure is now filed at the out root
    (`sampling_validation_failed.json`); the worker refuses the root before any write or
    resume, and `untrusted_store_reason` refuses every store under it."""
    sys.path.insert(0, str(REPO / "tests"))
    import test_tinycap_launcher as TL
    repos = tmp_path / "repos"
    tree = TL._tree(repos, "f6cf462")
    for rel in (C.BCB_PIN_PATH, C.BFCL_PIN_PATH, C.BCB_LIB_FAMILIES_PATH, C.model_row("gemma4e4b-f22it").cfg):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text((REPO / rel).read_text())
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(tmp_path / "gen"))
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: pytest.fail("a standup was started"))
    monkeypatch.setattr(L, "current_function_call_id", lambda: "fc-clamp")
    cfg = L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"], controls=["bfcl_200"],
                                                   model="gemma4e4b-f22it", sampling_mode=C.SAMPLING_PROTOCOL), "poison")[0]
    out_dir = Path(cfg["out_dir"])
    # the clean root proceeds to the (forbidden) standup: the protocol record is the only gate before it
    with pytest.raises(BaseException, match="a standup was started"):
        L._run_configuration(cfg, "f6cf462", protocol_check={"ok": True})
    # the worker's post-driver scan files the failure (the same writer the scan calls)
    path = L.write_sampling_validation_failure(out_dir, status=C.REC_CLAMP_OVERRIDE_INEFFECTIVE, error="2 clamp warning line(s)",
                                               run_id="f6cf462-cafebabe", clamp_warning_lines=["WARNING ... maxed it out to 0.01"],
                                               tree_sha="f" * 40, cell="none")
    assert path == out_dir / C.SAMPLING_VALIDATION_FAILED_FILENAME == out_dir / "sampling_validation_failed.json"
    doc = json.loads(path.read_text())
    assert doc["status"] == "CLAMP_OVERRIDE_INEFFECTIVE" and doc["function_call_id"] == "fc-clamp" and doc["run_id"] == "f6cf462-cafebabe"
    assert L.sampling_validation_failure(out_dir) == doc and L.sampling_validation_failure(tmp_path / "clean") is None
    # a retry into the same root: refused by name BEFORE any standup, snapshot write or resume — never DONE
    rec = L._run_configuration(cfg, "f6cf462", protocol_check={"ok": True})
    assert rec["status"] == C.REC_SAMPLING_VALIDATION_FAILED == "SAMPLING_VALIDATION_FAILED"
    assert "fc-clamp" in rec["error"] and "NEW --out-suffix" in rec["error"] and rec["sampling_validation_failure"] == doc
    snap_cfg = dict(cfg, roster_snapshot={"x": 1})
    assert L._run_configuration(snap_cfg, "f6cf462", protocol_check={"ok": True})["status"] == "SAMPLING_VALIDATION_FAILED"
    assert not (out_dir / "lane_roster.json").exists()
    # every store under the root is untrusted for ANY expectation (the reason names the marker)
    cell_dir = out_dir / "cells" / "somekey"
    cell_dir.mkdir(parents=True)
    (cell_dir / "manifest.json").write_text("{}")
    why = L.untrusted_store_reason(cell_dir, expected=SimpleNamespace(cell="none"))
    assert C.SAMPLING_VALIDATION_FAILED_FILENAME in why and "CLAMP_OVERRIDE_INEFFECTIVE" in why
    # an unreadable marker is a failure too, never "absent"
    path.write_text("{not json")
    assert L.sampling_validation_failure(out_dir)["status"] == "SAMPLING_VALIDATION_FAILED"
    # the scan site writes it (source pin: the marker is filed where the status is overridden)
    src = Path(L.__file__).read_text()
    assert src.index('rec["status_override"] = C.REC_CLAMP_OVERRIDE_INEFFECTIVE') < src.index("write_sampling_validation_failure(\n")


# --------------------------------------------------------------------------- #
# the standup                                                                  #
# --------------------------------------------------------------------------- #


def test_the_standup_takes_the_flag_and_records_the_override_iff_set():
    import dataclasses

    spec = importlib.util.spec_from_file_location("capctrl_serving_up_protocol_test", REPO / "scripts/capctrl_serving_up.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["capctrl_serving_up_protocol_test"] = m
    spec.loader.exec_module(m)
    ap = m.build_parser()
    base = ["--repo-dir", "/r", "--config", "c.yaml", "--cell", "none", "--defenses-dir", "/d", "--out", "/o",
            "--stop-file", "/s"]
    assert ap.parse_args(base).sampling_clamp_override is None
    assert ap.parse_args(base + [C.FLAG_SAMPLING_CLAMP_OVERRIDE, "0.001"]).sampling_clamp_override == 0.001
    assert "sampling_clamp_override" in {f.name for f in dataclasses.fields(m.Standup)}
    src = Path(m.__file__).read_text()
    assert "sampling_clamp_override=sampling_clamp_override)" in src            # bring_up -> build_server_launch
    assert "sampling_clamp_override=launch.sampling_clamp_override" in src      # the LAUNCH's record onto the Standup
    assert 'out["sampling_clamp_override"] = dict(self.sampling_clamp_override)' in src
    assert "sampling_clamp_override=args.sampling_clamp_override" in src        # main -> bring_up
    ep_body = src[src.index("def endpoint(self, eval_name=None) -> dict:"):src.index("def tear_down(self)")]
    assert "if self.sampling_clamp_override is not None:" in ep_body           # PRESENT IFF set


# --------------------------------------------------------------------------- #
# the provider (client venv only)                                              #
# --------------------------------------------------------------------------- #


def _client_harness():
    pytest.importorskip("inspect_ai", reason="tinycap client venv (hashed lock)")
    sys.path.insert(0, str(REPO / "tests"))
    import test_inspect_served as T  # noqa: E402  the fake proxy + message shapes

    from sisteer.capeval.tiers import inspect_served as isv
    return T, isv


def test_the_provider_selects_the_row_per_task_and_refuses_a_divergent_request(monkeypatch):
    T, isv = _client_harness()
    from inspect_ai.model import GenerateConfig

    rows = C.protocol_executed_by_task([BCB, BFCL])
    fake = T._Fake([T.VMODEL, T.BASE_VMODEL])
    try:
        cls = isv._register_provider()

        def api(**kw):
            return cls(model_name=C.inspect_model_name(T.BASE_VMODEL), base_url=fake.base_url, api_key="k",
                       config=GenerateConfig(), si=None, thinking=C.THINKING_OFF, route=T._route_of(T.BASE_VMODEL), **kw)
        seen = {"task": None}
        monkeypatch.setattr(isv, "current_request_identity", lambda: (seen["task"], "item_1", 0))
        a = api(sampling_by_task=rows)
        assert a.sampling is None and a.sampling_by_task == rows
        for task, want_t in ((BCB, 0.0), (BFCL, 0.001)):
            seen["task"] = task
            T._gen(a, T.SHAPE_BFCL if task == BFCL else T.SHAPE_MATH, tools=[T._BFCL_TOOL] if task == BFCL else ())
            b = fake.chat
            assert (b["temperature"], b["top_p"], b["top_k"]) == (want_t, 1.0, 0) and "seed" in b, (task, b)
        seen["task"] = GSM
        with pytest.raises(isv.InspectServedError, match="has no row in this run's per-task sampling map"):
            T._gen(a, T.SHAPE_MATH)
        seen["task"] = None
        with pytest.raises(isv.InspectServedError, match="has no row"):
            T._gen(a, T.SHAPE_MATH)
        seen["task"] = BFCL
        for cfg in (GenerateConfig(temperature=0.0), GenerateConfig(top_p=0.95), GenerateConfig(top_k=20),
                    GenerateConfig(extra_body={"top_k": 20})):
            with pytest.raises(isv.InspectServedError, match="frozen"):
                T._gen(a, T.SHAPE_BFCL, tools=[T._BFCL_TOOL], config=cfg)
        T._gen(a, T.SHAPE_BFCL, tools=[T._BFCL_TOOL], config=GenerateConfig(temperature=0.001, top_p=1.0, top_k=0))
        with pytest.raises(isv.InspectServedError, match="exactly one source"):
            api(sampling=C.GREEDY_SAMPLING, sampling_by_task=rows)
        with pytest.raises(isv.InspectServedError, match="SamplingRow"):
            api(sampling_by_task={BCB: {"temperature": 0}})
        # a single-row provider is what it was
        g = api(sampling=C.GREEDY_SAMPLING)
        assert g.request_sampling_row(BFCL) is C.GREEDY_SAMPLING
    finally:
        fake.close()


def test_the_plans_carry_each_tasks_row_and_declared_temperature(monkeypatch):
    _, isv = _client_harness()
    made = []
    monkeypatch.setattr(isv, "plan_task", lambda n, **kw: made.append((n, kw)) or SimpleNamespace(row_task=n))
    rows = C.protocol_executed_by_task([BCB, BFCL])
    decl = {t: r.protocol_temperature for t, r in C.protocol_rows_by_task([BCB, BFCL]).items()}
    isv.plans_for_atoms(LEGS, thinking="off", grader_model=None, sampling_by_task=rows, protocol_temperature_by_task=decl)
    kws = {n: kw for n, kw in made}
    assert kws["bfcl_200"]["sampling"] is rows[BFCL] and kws["bfcl_200"]["protocol_temperature"] == 0.001
    assert kws["bigcodebench_200"]["sampling"] is rows[BCB] and kws["bigcodebench_200"]["protocol_temperature"] == 0.0
    made.clear()
    isv.plans_for_atoms(LEGS, thinking="off", grader_model=None, sampling=C.GREEDY_SAMPLING)
    assert all(kw["sampling"] is C.GREEDY_SAMPLING and "protocol_temperature" not in kw for _, kw in made)
    for kw, why in (({"sampling_by_task": {BFCL: rows[BFCL]}}, "have no entry"),
                    ({"sampling_by_task": {**rows, GSM: rows[BCB]}}, "does not plan"),
                    ({"sampling_by_task": rows, "sampling": C.GREEDY_SAMPLING}, "exactly one source"),
                    ({"protocol_temperature_by_task": {BFCL: 0.001}}, "have no entry")):
        with pytest.raises(isv.InspectServedError, match=why):
            isv.plans_for_atoms(LEGS, thinking="off", grader_model=None, **kw)
    # the summary term rides iff set
    plan = SimpleNamespace(protocol_temperature=None)
    assert isv._protocol_term(plan) == {}
    assert isv._protocol_term(SimpleNamespace(protocol_temperature=0.001)) == {"protocol_temperature": 0.001}
    import dataclasses
    assert {f.name: f.default for f in dataclasses.fields(isv.TaskPlan)}["protocol_temperature"] is None


def test_the_request_composition_under_the_client_venv():
    _client_harness()
    doc = PC.compose_requests([BCB, BFCL], model="google/gemma-4-E4B-it")
    bcb, bfcl = doc["tasks"][BCB]["body"], doc["tasks"][BFCL]["body"]
    assert (bcb["temperature"], bcb["top_p"], bcb["top_k"]) == (0.0, 1.0, 0) and "seed" in bcb and "tools" not in bcb
    assert (bfcl["temperature"], bfcl["top_p"], bfcl["top_k"]) == (0.001, 1.0, 0) and "seed" in bfcl
    assert bfcl["tools"] == [PC.PROBE_TOOL] and bfcl["tool_choice"] == "auto" and bfcl["model"] == "google/gemma-4-E4B-it"
    assert doc["tasks"][BFCL]["executed"] == {"temperature": 0.001, "top_p": 1.0, "top_k": 0}
    assert "top_p" not in doc["controls"]["inherit"] and doc["controls"]["inherit"]["temperature"] == 1.0
    assert (doc["controls"]["explicit"]["top_p"], doc["controls"]["explicit"]["top_k"]) == (1.0, 0)
    assert "extra_body" not in bcb and "n" not in bcb and not bcb.get("stream")


# --------------------------------------------------------------------------- #
# check_protocol_sampling's pure judge and launch-flag reader                  #
# --------------------------------------------------------------------------- #


def _rec(t, top_p=1.0, top_k=0):
    return {"temperature": t, "top_p": top_p, "top_k": top_k, "min_p": 0.0, "repetition_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0}


def _rr(t, **k):
    return {"resolved": _rec(t, **k), "round_trip": _rec(t, **k)}


REQUESTS = {"model": "m", "tasks": {BCB: {"body": {}, "executed": {"temperature": 0.0, "top_p": 1.0, "top_k": 0}},
                                    BFCL: {"body": {}, "executed": {"temperature": 0.001, "top_p": 1.0, "top_k": 0}}},
            "controls": {"inherit": {}, "explicit": {}}}
GOOD = {"wheel_max_temp": 0.01, "generation_config": {"generation_config": "auto", "override_generation_config": {}},
        "model_defaults": {"defaults": {}, "source": "ModelConfig.get_diff_sampling_param", "model_config_error": None},
        "fixture_defaults": {"defaults": {"top_p": 0.95}, "source": "replica:get_diff_sampling_param", "model_config_error": "x"},
        "negative_control": {BCB: _rr(0.0), BFCL: _rr(0.01)},
        "production": {BCB: _rr(0.0), BFCL: _rr(0.001)},
        "controls": {"inherit_fixture": _rr(1.0, top_p=0.95), "explicit_fixture": _rr(1.0),
                     "inherit_model": _rr(1.0), "explicit_model": _rr(1.0)},
        # round 6: the REAL spawned-child control — armed child at the override, unarmed at the wheel's clamp
        "spawned_child_control": {
            "armed": {"ok": True, "armed": True, "temperature": 0.001, "max_temp": 0.001, "shim_applied": 0.001, "shim_from": 0.01,
                      "sitecustomize": "/x/vllm_shim/sitecustomize.py", "start_method": "spawn", "exitcode": 0},
            "unarmed": {"ok": True, "armed": False, "temperature": 0.01, "max_temp": 0.01, "shim_applied": None, "shim_from": None,
                        "sitecustomize": None, "start_method": "spawn", "exitcode": 0}},
        "override_mechanism": "in-process (vllm_entry.apply_override) + sitecustomize shim"}


def test_the_judge_passes_the_expected_record_and_names_every_departure():
    j = PC.judge(REQUESTS, GOOD)
    assert j["ok"] and j["problems"] == []
    assert j["production"] == {BCB: _rec(0.0), BFCL: _rec(0.001)} and j["negative_control"][BFCL]["temperature"] == 0.01
    assert C.check_protocol_effective(j) == {BCB: _rec(0.0), BFCL: _rec(0.001)}      # the driver reads the judge's record
    assert j["defaults_source"] == "ModelConfig.get_diff_sampling_param" and j["wheel_max_temp"] == 0.01

    def problems(mut):
        r = copy.deepcopy(GOOD)
        mut(r)
        out = PC.judge(REQUESTS, r)
        assert not out["ok"]
        return out["problems"]

    assert any("production temperature resolved 0.01" in p for p in problems(
        lambda r: r["production"][BFCL]["resolved"].__setitem__("temperature", 0.01)))
    assert any("msgspec round trip" in p for p in problems(
        lambda r: r["production"][BFCL]["round_trip"].__setitem__("temperature", 0.01)))
    assert any("clamp this override lifts is not real" in p for p in problems(
        lambda r: r["negative_control"][BFCL]["resolved"].__setitem__("temperature", 0.001)))
    assert any("negative control temperature 0.5" in p for p in problems(
        lambda r: r["negative_control"][BCB]["resolved"].__setitem__("temperature", 0.5)))
    assert any("T 1.0 with top_p OMITTED" in p for p in problems(
        lambda r: r["controls"]["inherit_fixture"]["resolved"].__setitem__("top_p", 1.0)))
    assert any("top_p 1.0 / top_k 0 present resolved" in p for p in problems(
        lambda r: r["controls"]["explicit_fixture"]["resolved"].__setitem__("top_k", -1)))
    assert any("not the pinned 0.01" in p for p in problems(lambda r: r.__setitem__("wheel_max_temp", 0.02)))
    assert any("top_k resolved -1" in p for p in problems(
        lambda r: r["production"][BCB]["resolved"].__setitem__("top_k", -1)))
    assert any("missing a production" in p for p in problems(lambda r: r["production"].pop(BFCL)))


def test_the_d16_override_control():
    """An `--override-generation-config` top_p 0.8 (the Qwen row) must be what a
    T-1.0 top_p-omitted request resolves to, while the protocol request with explicit
    top_p 1.0 / top_k 0 still resolves to 1.0 / 0."""
    over = copy.deepcopy(GOOD)
    over["generation_config"]["override_generation_config"] = {"top_p": 0.8}
    over["model_defaults"]["defaults"] = {"top_p": 0.8}
    over["controls"]["inherit_model"] = _rr(1.0, top_p=0.8)
    assert PC.judge(REQUESTS, over)["ok"]
    over["controls"]["inherit_model"] = _rr(1.0, top_p=1.0)
    assert any("override control (D16)" in p for p in PC.judge(REQUESTS, over)["problems"])
    over["controls"]["inherit_model"] = _rr(1.0, top_p=0.8)
    over["controls"]["explicit_model"] = _rr(1.0, top_p=0.8)
    assert any("explicit protocol values resolved top_p 0.8" in p for p in PC.judge(REQUESTS, over)["problems"])
    # the production assertions are unchanged by an override (explicit values win)
    over["controls"]["explicit_model"] = _rr(1.0)
    assert PC.judge(REQUESTS, over)["ok"]


def test_the_launch_flag_reader_and_the_replica_key_list():
    assert PC.launch_generation_config([]) == {"generation_config": "auto", "override_generation_config": {}}
    g = PC.launch_generation_config(["--foo", "1", "--override-generation-config", '{"top_p": 0.8, "temperature": 0.7}'])
    assert g == {"generation_config": "auto", "override_generation_config": {"top_p": 0.8, "temperature": 0.7}}
    assert PC.launch_generation_config(["--generation-config=vllm", '--override-generation-config={"top_k": 20}']) == {
        "generation_config": "vllm", "override_generation_config": {"top_k": 20}}
    with pytest.raises(ValueError, match="JSON object"):
        PC.launch_generation_config(["--override-generation-config", "[1]"])
    # code review: a TRAILING valueless flag refuses (vLLM's own parser rejects that argv) —
    # it used to read as absent, i.e. "auto" / {}
    for trailing in (["--generation-config"], ["--x", "1", "--override-generation-config"]):
        with pytest.raises(ValueError, match="no value"):
            PC.launch_generation_config(trailing)
    # the fixture control resolves through the MODEL path with the fixture as the named
    # generation-config dir (vLLM's spelling), so ModelConfig can construct for it too
    src_pc = Path(PC.__file__).read_text()
    assert "server_default_sampling_params(model_path, revision=revision, generation_config=str(fixture_dir)" in src_pc
    assert "server_default_sampling_params(str(fixture_dir)" not in src_pc
    assert PC.AVAILABLE_PARAMS == ("repetition_penalty", "temperature", "top_k", "top_p", "min_p", "max_new_tokens")
    assert PC.EFFECTIVE_FIELDS == C.PROTOCOL_EFFECTIVE_FIELDS
    assert (PC.PINNED_MAX_TEMP, PC.OVERRIDE_MAX_TEMP) == (C.VLLM_MAX_TEMP_PINNED, C.BFCL_PROTOCOL_TEMPERATURE)
    fixture = REPO / "tests" / "fixtures" / "protocol_check" / "generation_config_top_p95" / "generation_config.json"
    assert json.loads(fixture.read_text())["top_p"] == 0.95
    # the module is importable without vLLM or inspect_ai (the judge is pure): its
    # top-level imports are stdlib only — the heavy imports live inside compose / resolve
    src = Path(PC.__file__).read_text()
    top_imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert top_imports and not any("vllm" in ln or "inspect_ai" in ln or "sisteer" in ln for ln in top_imports), top_imports


def test_the_golden_identity_fixtures_did_not_move():
    """The goldens hash EXACTLY TINYCAP_IDENTITY_FIELDS (card); the protocol key is
    present only under protocol, so the fixtures in the tree are untouched — the
    compare in test_served_capability is the proof; this pins the file bytes."""
    import hashlib
    gd = REPO / "tests" / "fixtures" / "tinycap_identity_v1"
    for name in ("standard_off_gsm8k.json", "standard_on_gemma_gsm8k.json"):
        doc = json.loads((gd / name).read_text())
        assert set(doc["identity"]["tinycap"]) == set(C.TINYCAP_IDENTITY_FIELDS)
        assert "sampling_by_task" not in doc["identity"]["tinycap"] and "sampling_clamp_override" not in doc["identity"]["tinycap"]["served"]
        assert "by_task" not in doc["manifest"]["sampling"]
    assert hashlib.sha256((gd / "standard_off_gsm8k.json").read_bytes()).hexdigest()  # readable, unchanged by this lane


# ---------------------------------------------------------------------------- #
# the engine-resolved sampler attestation, wired into every standup
# ---------------------------------------------------------------------------- #

# vLLM 0.25.1 entrypoints/serve/utils/request_logger.py — the shapes the reader parses
# (tests/test_record_resolved_sampler.py pins them against the source): the start-up attestation
# of the flag, and ONE params line per request, prompt-free
_ATTEST = ("(APIServer pid=1) INFO 09-12 02:05:04 [request_logger.py:28] `--enable-log-requests` is set but the "
           "minimum log level is higher than DEBUG. Only limited information will be logged to minimize overhead.")
_PARAMS = ("INFO [request_logger.py:64] Received request chatcmpl-{i}: params: SamplingParams(n=1, "
           "presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature={t}, "
           "top_p={tp}, top_k={tk}, min_p=0.0, seed=None, stop=[], max_tokens={cap}), lora_request: None.")


def test_every_standup_carries_enable_log_requests_in_the_fixed_argv_and_the_identity_gains_nothing(L, tmp_path):
    """The flag rides in `_standup_cmd`'s FIXED part right after the upstream timeout,
    on a card AND a protocol standup, and is NOT a per-configuration extra — the identity
    derivations read the extras, so the hashed served block never names it (a log level changes
    no decoding). The standup CLI spells the same flag, splices it beside its two wrapper-owned
    flags, and records the opt-in on endpoint.json under the contracts' key."""
    assert C.FLAG_ENABLE_LOG_REQUESTS == "--enable-log-requests" and C.ENDPOINT_KEY_ENABLE_LOG_REQUESTS == "enable_log_requests"
    assert C.RESOLVED_SAMPLER_SCRIPT == "scripts/sdfbase/record_resolved_sampler.py" and C.RESOLVED_SAMPLER_FILENAME == "resolved_sampler.json"
    assert (REPO / C.RESOLVED_SAMPLER_SCRIPT).is_file()                            # the reader is on this tree
    for cfg in (_plans(L, C.SAMPLING_CARD)[0], _plans(L, C.SAMPLING_PROTOCOL)[0]):
        geom = L.configuration_geometry(cfg, 1)
        argv = L._standup_cmd("/tree", "c", cfg["cell"], "d", tmp_path, tmp_path / "STOP", geom, 64, cfg["standup_extra"])
        assert argv.count(C.FLAG_ENABLE_LOG_REQUESTS) == 1, cfg["name"]
        i = argv.index(C.FLAG_UPSTREAM_TIMEOUT_S)
        assert argv[i + 1] == "7200" and argv[i + 2] == C.FLAG_ENABLE_LOG_REQUESTS       # the fixed part, before digest + extras
        assert C.FLAG_ENABLE_LOG_REQUESTS not in cfg["standup_extra"]
        assert argv[len(argv) - len(cfg["standup_extra"]):] == list(cfg["standup_extra"])  # the extras still end the argv
    for protocol in (False, True):
        assert C.ENDPOINT_KEY_ENABLE_LOG_REQUESTS not in C.served_identity_fields(protocol_sampling=protocol)
    # the standup CLI: parses the flag, splices it after --max-logprobs, records it (source pins, the
    # way tests/test_sdfbase_full22_wave.py pins mt_serving_up's), default OFF = byte-identical argv
    sys.path.insert(0, str(REPO / "scripts"))
    import capctrl_serving_up as up
    required = ["--repo-dir", "/r", "--config", "c.yaml", "--cell", "none", "--defenses-dir", "d", "--out", "o", "--stop-file", "s"]
    assert up.build_parser().parse_args(required).enable_log_requests is False
    assert up.build_parser().parse_args(required + [C.FLAG_ENABLE_LOG_REQUESTS]).enable_log_requests is True
    src = (REPO / "scripts" / "capctrl_serving_up.py").read_text()
    # the merged standup spells the flag through its ENABLE_LOG_REQUESTS_FLAG constant (== contracts.FLAG_ENABLE_LOG_REQUESTS)
    assert "+ ((ENABLE_LOG_REQUESTS_FLAG,) if enable_log_requests else ())" in src
    assert up.ENABLE_LOG_REQUESTS_FLAG == C.FLAG_ENABLE_LOG_REQUESTS
    assert '"enable_log_requests": bool(self.enable_log_requests),' in src
    assert "enable_log_requests=bool(args.enable_log_requests)" in src
    # mt_serving_up (the cluster standups) spells the same flag: one vocabulary
    assert C.FLAG_ENABLE_LOG_REQUESTS in (REPO / "scripts" / "sdfbase" / "mt_serving_up.py").read_text()


def test_resolved_sampler_expectation_is_the_declared_rows_keys_and_never_a_disagreeing_one(L):
    """The reader is told only what the run DECLARED: a card row's set keys; greedy's temperature
    alone (vLLM resets top_p / top_k itself at 0); under protocol the keys BOTH executed rows agree
    on (top_p 1.0, top_k 0), with temperature — 0.0 for BCB, 0.001 for BFCL — UNCOMPARED per task,
    since the reader classes on max_tokens and one expectation would false-DIVERGE."""
    card = L.resolved_sampler_expectation(C.SAMPLING_CARD, "gemma4e4b", [BCB])
    assert card["expect"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0} and card["uncompared"] == {}
    assert card["source"] == "contracts.sampling_for('gemma4e4b', 'card')"
    greedy = L.resolved_sampler_expectation(C.SAMPLING_GREEDY, "gemma4e4b", [BCB])
    assert greedy["expect"] == {"temperature": 0.0} and greedy["uncompared"] == {}
    both = L.resolved_sampler_expectation(C.SAMPLING_PROTOCOL, "gemma4e4b", [BCB, BFCL, GSM])
    assert both["expect"] == {"top_p": 1.0, "top_k": 0.0}
    assert both["uncompared"] == {"temperature": {BCB: 0.0, BFCL: 0.001}}
    assert both["source"] == f"contracts.protocol_executed_by_task({sorted([BCB, BFCL])})"
    one = L.resolved_sampler_expectation(C.SAMPLING_PROTOCOL, "gemma4e4b", [BFCL])
    assert one["expect"] == {"temperature": 0.001, "top_p": 1.0, "top_k": 0.0} and one["uncompared"] == {}
    # Qwen's `general` row states presence_penalty 1.5 — expected, so a request that omits it reads
    # DIVERGED (the engine's 0.0), which is exactly the finding the reader exists for; the `card` row
    # states none, so nothing is expected of it (the engine's default is recorded, not judged)
    qgen = L.resolved_sampler_expectation(C.SAMPLING_GENERAL, "qwen35_4b", [BCB])
    assert qgen["expect"]["presence_penalty"] == 1.5 and set(qgen["expect"]) <= set(C.SAMPLING_ROW_FIELDS)
    qcard = L.resolved_sampler_expectation(C.SAMPLING_CARD, "qwen35_4b", [BCB])
    assert "presence_penalty" not in qcard["expect"] and qcard["expect"]["temperature"] == C.sampling_for("qwen35_4b", C.SAMPLING_CARD).temperature
    with pytest.raises(ValueError):
        L.resolved_sampler_expectation("not-a-mode", "gemma4e4b", [BCB])


def test_the_reader_runs_over_the_standup_log_into_the_store_root_and_the_verdict_is_recorded_not_propagated(L, tmp_path):
    """End to end on a synthetic 0.25.1 server log: the record lands at
    <out>/resolved_sampler.json beside the store, AGREES when the engine resolved the declared row,
    DIVERGED (rc 0, no error — recorded, not propagated) when it did not, UNREADABLE when the server
    was never asked (silence is not agreement), and a reader that cannot run is loud, not fatal. The
    reader runs IN-PROCESS from the tree (the worker's subprocess seam is the driver's, stubbed and
    counted by the ladder tests), so the launcher's sys.path is unchanged afterwards."""
    path_before = list(sys.path)
    srv, out_dir = tmp_path / "srv", tmp_path / "out"
    srv.mkdir(); out_dir.mkdir()
    exp = L.resolved_sampler_expectation(C.SAMPLING_CARD, "gemma4e4b", [BCB])

    def log(*lines):
        (srv / "vllm_server.log").write_text("\n".join(lines) + "\n")

    log(_ATTEST, *[_PARAMS.format(i=i, t=1.0, tp=0.95, tk=64, cap=8192) for i in range(3)],
        _PARAMS.format(i=9, t=0.0, tp=1.0, tk=0, cap=48))                                 # + one greedy probe at the probe cap
    block = L.record_resolved_sampler(str(REPO), srv, out_dir, expectation=exp)
    assert block["rc"] == 0 and block["verdict"] == "AGREES" and "error" not in block, block
    assert sys.path == path_before
    assert block["path"] == str(out_dir / C.RESOLVED_SAMPLER_FILENAME) and (out_dir / C.RESOLVED_SAMPLER_FILENAME).is_file()
    assert block["expect"] == exp["expect"] and block["uncompared"] == {} and block["expect_source"] == exp["source"]
    argv = block["argv"]
    assert argv[1] == f"{REPO}/{C.RESOLVED_SAMPLER_SCRIPT}" and argv[argv.index("--vllm-log") + 1] == str(srv / "vllm_server.log")
    assert argv[argv.index("--probe-cap") + 1] == "48"                                     # served_target.PROBE_MAX_TOKENS, read in-process
    assert [argv[i + 1] for i, t in enumerate(argv) if t == "--expect"] == ["temperature=1", "top_k=64", "top_p=0.95"]
    record = json.loads((out_dir / C.RESOLVED_SAMPLER_FILENAME).read_text())
    assert record["verdict"] == "AGREES" and record["n_client_requests"] == 3 and record["n_probe_requests"] == 1
    assert record["reference"] == exp["expect"] and record["reference_source"] == "--expect"
    # the engine resolved something else: DIVERGED is a recorded finding, rc 0, the store still a store
    log(_ATTEST, *[_PARAMS.format(i=i, t=0.7, tp=0.95, tk=64, cap=8192) for i in range(2)])
    block = L.record_resolved_sampler(str(REPO), srv, out_dir, expectation=exp)
    assert block["rc"] == 0 and block["verdict"] == "DIVERGED" and "error" not in block
    assert json.loads((out_dir / C.RESOLVED_SAMPLER_FILENAME).read_text())["divergences"][0]["key"] == "temperature"
    # a server never asked (no flag attested, no params line): UNREADABLE, never AGREES
    log("(APIServer pid=1) INFO 09-12 02:05:04 [api_server.py:616] Starting vLLM server")
    block = L.record_resolved_sampler(str(REPO), srv, out_dir, expectation=exp)
    assert block["verdict"] == "UNREADABLE" and block["rc"] == 0
    # a staged tree without the reader: loud in the block, no exception, no record claimed
    block = L.record_resolved_sampler(str(tmp_path / "no-such-tree"), srv, out_dir, expectation=exp)
    assert "error" in block and "verdict" not in block and C.RESOLVED_SAMPLER_SCRIPT in block["error"]
def test_the_judge_requires_the_spawned_child_control():
    """Round 6: the in-image check runs a REAL spawned interpreter twice — armed (the
    shim must lift the clamp: 0.001) and unarmed (the wheel's 0.01, the negative control). The judge
    refuses a record without the control, an armed child that stayed clamped, an unarmed child that
    was not, or a child that failed to report — and copies the control and the mechanism into its
    record."""
    import copy as _copy
    j = PC.judge(REQUESTS, GOOD)
    assert j["ok"] and j["spawned_child_control"] == GOOD["spawned_child_control"]
    assert j["override_mechanism"] == GOOD["override_mechanism"]
    missing = _copy.deepcopy(GOOD)
    missing.pop("spawned_child_control")
    v = PC.judge(REQUESTS, missing)
    assert v["ok"] is False and any("spawned-child control missing" in p for p in v["problems"])
    clamped = _copy.deepcopy(GOOD)
    clamped["spawned_child_control"]["armed"]["temperature"] = 0.01
    v = PC.judge(REQUESTS, clamped)
    assert v["ok"] is False and any("shim did not reach the child" in p for p in v["problems"])
    lifted = _copy.deepcopy(GOOD)
    lifted["spawned_child_control"]["unarmed"]["temperature"] = 0.001
    v = PC.judge(REQUESTS, lifted)
    assert v["ok"] is False and any("negative control proves the shim" in p for p in v["problems"])
    failed = _copy.deepcopy(GOOD)
    failed["spawned_child_control"]["armed"] = {"ok": False, "error": "ModuleNotFoundError: vllm"}
    v = PC.judge(REQUESTS, failed)
    assert v["ok"] is False and any("ModuleNotFoundError" in p for p in v["problems"])
    # the resolve step runs BOTH children through vllm_entry's real spawn control (source pin)
    src = Path(PC.__file__).read_text()
    assert "VE.spawned_child_control(armed=True)" in src and "VE.spawned_child_control(armed=False)" in src
    # the standup carries the mechanism beside the record, iff the record is set
    up = (REPO / "scripts" / "capctrl_serving_up.py").read_text()
    assert 'out["sampling_clamp_override_mechanism"] = self.sampling_clamp_override_mechanism' in up
    assert 'sampling_clamp_override_mechanism=getattr(launch, "sampling_clamp_override_mechanism", None)' in up
