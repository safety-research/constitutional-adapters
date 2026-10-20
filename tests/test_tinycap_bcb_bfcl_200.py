"""bcb-bfcl-200 — the two pinned legs in the LAUNCHER and the DRIVER,
CPU-only: the legs are launchable by name alone (every table indexed by control
or task has both rows), the expected identity carries the item-set / sandbox
keys when and only when a leg is filed and the store-trust check holds a store
to them, the standup argv and the dry run name the sandbox block / owner tuple
/ watcher, the ownership helpers (owner, pointer, record, sweep) work against a
fake Modal, and the driver's route block carries the sandbox facts only on a
BigCodeBench run.

Loads the launcher the way ``test_tinycap_launcher`` does (a modal stub, the
app name in the environment)."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import sandbox_scope as scope

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from test_tinycap_launcher import _expected, _load_launcher, _spawn_only, _store, _stub_modal  # noqa: E402

TEST_APP_NAME = "tinycap-bcb-bfcl-200-under-test"
DERIVED = "constv2-mix-full__lora_r4_ce_minus_ctl"
BCB, BFCL200 = "bigcodebench_200", "bfcl_200"


@pytest.fixture(scope="module")
def L():
    return _load_launcher(_stub_modal(), app_name=TEST_APP_NAME, name="tinycap_served_bcb_bfcl_200")


# --------------------------------------------------------------------------- #
# launchable by name: tables, tool calls, geometry, frozen counts               #
# --------------------------------------------------------------------------- #


def test_the_legs_plan_by_name_with_the_right_geometry_class_and_tool_calls(L):
    cfgs = L.plan_configurations(cells=["none", DERIVED], modes=["off"], controls=[BCB, BFCL200],
                                 model="gemma4e4b-sdfbase")
    assert [c["cell"] for c in cfgs] == ["none", DERIVED]
    for cfg in cfgs:
        assert cfg["controls"] == [BCB, BFCL200] and cfg["tool_calls"] is True       # bfcl_200 carries tools
        assert cfg["thinking_by_task"] == {"inspect/bigcodebench_200": "off", "inspect/bfcl_200": "off"}
        assert set(cfg["geometry_keys"]) == {"capability:gen_long", "capability:agent"}
        assert "--tool-calls" in cfg["standup_extra"] and "--reasoning-parser" not in cfg["standup_extra"]
        assert cfg["out_dir"].startswith(f"{C.VOL_GEN_ROOT}/gemma4e4b-sdfbase/")
        assert cfg["name"].startswith("tinycap-e4b_sdfbase-")
    assert cfgs[1]["steered"] is True and cfgs[0]["base"] is True
    # needs_tool_calls keys on TOOL_CALL_CONTROLS, so bfcl_200 alone composes it and bigcodebench_200 alone does not
    assert L.needs_tool_calls([BFCL200]) is True and L.needs_tool_calls([BCB]) is False
    assert L.needs_tool_calls(["bfcl_200"]) is True
    # the tool-parser refusal keeps the non-tool legs and drops both tool-calling ones
    row = C.model_row("qwen35_9b")                                       # no verified tool parser
    msg = L.no_tool_parser_message(row, [BCB, BFCL200])
    assert f"--controls {BCB}" in msg and "bfcl" not in msg.split("--controls ")[1].split(",")[0]
    with pytest.raises(ValueError, match="no petri manifest"):
        L.plan_configurations(cells=["none"], modes=["off"], controls=[BFCL200], model="qwen35_9b")
    # frozen counts: the pins' n, so a done marker is trusted only at exactly 200
    assert L.FULL_ITEM_COUNT_OF_TASK["inspect/bigcodebench_200"] == 200
    assert L.FULL_ITEM_COUNT_OF_TASK["inspect/bfcl_200"] == 200
    assert L.sandboxed_controls([BCB], "off") is True and L.sandboxed_controls([BFCL200, "gsm8k"], "off") is False


def test_the_geometry_gate_serves_both_legs_at_the_pass1_footprint(L):
    for controls in ([BCB], [BFCL200], [BCB, BFCL200]):
        (cfg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=controls)
        g, budget = L.served_geometry(cfg, 1)
        assert g.max_model_len == 16384 and g.max_num_seqs == 36 and g.seqs_source == L.MODEL_BUDGET_SOURCE
        assert budget["identity"] is True and budget["over_budget"] is False


# --------------------------------------------------------------------------- #
# identity: present only when filed; the store-trust check                     #
# --------------------------------------------------------------------------- #


def test_expected_identity_carries_the_item_sets_and_sandbox_only_when_filed(L):
    both = _expected(L, cell="none", controls=[BCB, BFCL200])
    d = both.as_dict()
    pin_b, pin_f = C.load_bcb_pin(REPO), C.load_bfcl_pin(REPO)
    assert d[C.IDENTITY_KEY_BCB_ITEM_SET] == C.item_set_identity(pin_b) == both.bcb_item_set
    assert d[C.IDENTITY_KEY_BFCL_ITEM_SET] == C.item_set_identity(pin_f) == both.bfcl_item_set
    assert d[C.IDENTITY_KEY_SANDBOX] == C.bcb_sandbox_identity() == {"image": C.BCB_SANDBOX_IMAGE,
                                                                     "verify_timeout_s": 30}
    assert both.files_bigcodebench_200 and both.files_bfcl_200 and not both.files_math500
    assert set(both.optional_identity()) == {"bcb_item_set", "bcb_sandbox", "bfcl_item_set"}
    # served: tool calls composed (bfcl_200), the pass-1 footprint
    assert both.served["tool_calls"] is True and both.served["max_model_len"] == 16384
    # one leg alone: exactly its keys
    bfcl_only = _expected(L, cell="none", controls=[BFCL200])
    assert set(bfcl_only.optional_identity()) == {"bfcl_item_set"} and bfcl_only.bcb_sandbox is None
    bcb_only = _expected(L, cell="none", controls=[BCB])
    assert set(bcb_only.optional_identity()) == {"bcb_item_set", "bcb_sandbox"}
    assert bcb_only.served["tool_calls"] is False
    # a standard configuration: byte-identical to before (no optional key)
    std = _expected(L)
    assert std.optional_identity() == {} and std.bcb_item_set is None and std.bfcl_item_set is None
    for key in (C.IDENTITY_KEY_BCB_ITEM_SET, C.IDENTITY_KEY_BFCL_ITEM_SET, C.IDENTITY_KEY_SANDBOX):
        assert key not in std.as_dict()


def test_a_pin_that_does_not_regenerate_refuses_the_expectation_before_any_gpu(L, tmp_path, monkeypatch):
    """The launcher reads the pin from the tree it serves; a tampered pin is a
    SubsetPinError at plan time, never an identity valued from a moved item set."""
    tree = tmp_path / "tree"
    (tree / "configs/capeval/tasks").mkdir(parents=True)
    (tree / "src" / "sisteer").mkdir(parents=True)
    (tree / C.TREE_SHA_FILENAME).write_text("a" * 40 + "\n")
    good = json.loads((REPO / C.BCB_PIN_PATH).read_text())
    bad = {**good, "ids": [*good["ids"][:-1], good["population"]["ids"][0]]}
    (tree / C.BCB_PIN_PATH).write_text(json.dumps(bad))
    (tree / C.BFCL_PIN_PATH).write_text((REPO / C.BFCL_PIN_PATH).read_text())
    (tree / C.BCB_LIB_FAMILIES_PATH).write_text((REPO / C.BCB_LIB_FAMILIES_PATH).read_text())
    (cfg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=[BCB])
    geom, _ = L.served_geometry(cfg, 1)
    with pytest.raises(C.SubsetPinError, match="do not regenerate"):
        L.expected_identity(cfg, geom, pass_no=1, controls=[BCB], samples=1, tree=str(tree), tree_sha="a" * 40)


def _bcb_manifest_kwargs(exp):
    """The identity block a BigCodeBench store's driver writes: the pinned
    legs' keys through the same ``optional`` selection the helper applies."""
    return dict(tasks=[BCB, BFCL200])


def test_untrusted_store_reason_holds_a_pinned_leg_store_to_its_item_set_and_sandbox(L, tmp_path):
    exp = _expected(L, cell="none", controls=[BCB, BFCL200])
    out = tmp_path / "gen"

    def mutate_add(m):
        # the driver stamps the three keys beside the base identity (both blocks)
        for blk in (m["resolved_config"]["tinycap"], m["tinycap"]):
            blk.update(exp.optional_identity())
        from sisteer.capeval import store as cs
        m["harness_config_hash"] = cs.harness_config_hash(m["resolved_config"])

    # the helper builds a store without the keys; a manifest lacking them for a
    # store that FILES the legs is refused by the exact-keys rule, by name
    store = _store(out, exp, tasks=[BCB, BFCL200])
    why = L.untrusted_store_reason(store, expected=exp)
    assert why and "bcb_item_set" in why and "pinned legs' item-set / sandbox keys" in why
    # a store whose driver stamped them, with the same values: trusted
    store2 = _store(out / "b", exp, tasks=[BCB, BFCL200], mutate=mutate_add)
    key = store2.name
    # rehash the directory name to the mutated manifest's hash prefix
    manifest = json.loads((store2 / "manifest.json").read_text())
    good = out / "b" / "cells" / (key[:key.rfind("__") + 2] + manifest["harness_config_hash"][:8])
    store2.rename(good)
    assert L.untrusted_store_reason(good, expected=exp) is None
    # another pin (a different ids digest) is another instrument: refused, naming the key
    other = exp._replace(bcb_item_set={**exp.bcb_item_set, "ids_sha256": "0" * 64})
    why = L.untrusted_store_reason(good, expected=other)
    assert why and "tinycap.bcb_item_set" in why and "another pin" in why
    # another verifier image likewise
    other = exp._replace(bcb_sandbox={**exp.bcb_sandbox, "verify_timeout_s": 60})
    why = L.untrusted_store_reason(good, expected=other)
    assert why and "tinycap.bcb_sandbox" in why and "another verifier" in why
    # a standard store lends nothing to (and is not refused for) the legs' keys: it
    # simply never requested their atoms
    std = _store(out / "c", _expected(L, cell="none"), tasks=list(C.expand_control("gsm8k")))
    assert L.untrusted_store_reason(std, expected=_expected(L, cell="none")) is None
    report = L.resume_report(out / "c", [BCB], exp)
    assert report["controls"][BCB]["done_in"] is None


# --------------------------------------------------------------------------- #
# the standup argv, the dry run and the watcher command                         #
# --------------------------------------------------------------------------- #


def test_the_dry_run_names_the_legs_caps_sandbox_block_owner_and_watcher(L, capsys):
    L.main(app_name=L.APP_NAME, sha="abc1234", pass1=True, model="gemma4e4b", cells="none", modes="off",
           controls=f"{BCB},{BFCL200}", out_suffix="smoketest", dry_run=True)
    out = capsys.readouterr().out
    assert "[tinycap] run id: abc1234-" in out
    assert f"bigcodebench_200: the pinned 200 of 992 non-Hard BigCodeBench ids (seed 0), thinking OFF, cap {C.BCB200_MAX_TOKENS_OFF}" in out
    assert f"bfcl_200: the pinned 200 of 3981 BFCL ids (seed 0, Hamilton by category, 18 categories), thinking OFF, cap {C.BFCL200_MAX_TOKENS_OFF}" in out
    assert f"sandbox: provider {C.BCB_SANDBOX_PROVIDER} (app {C.BCB_SANDBOX_APP_NAME}), image {C.BCB_SANDBOX_IMAGE}" in out
    assert "sandbox timeout 10800 s, verify timeout 30 s" in out and "max_sandboxes = S = 36" in out
    assert "sandbox owner: (run_id=abc1234-" in out and "cell=none, function_call_id=<this container's FunctionCall>" in out
    assert "watch/attempt_<fcid>.json" in out and "watch/sweep_<fcid>.json" in out
    assert ("sandbox watcher (arm in the same turn as the launch): ~/venvs/modal/bin/python "
            "scripts/tinycap_sandbox_watch.py --run-id abc1234-") in out
    assert "--suffix smoketest --model gemma4e4b --cells none --poll 120 --max-hours 6" in out
    # the identity the store is held to names both item sets and the sandbox
    assert '"bcb_item_set": {"ids_sha256"' in out and '"bfcl_item_set": {"ids_sha256"' in out
    assert '"bcb_sandbox": {"image"' in out
    # every standup argv still carries the upstream timeout and, on this leg, --tool-calls
    assert f"{C.FLAG_UPSTREAM_TIMEOUT_S} 7200" in out and "--tool-calls" in out
    # a run WITHOUT a sandboxed leg prints no watcher line
    L.main(app_name=L.APP_NAME, sha="abc1234", pass1=True, cells="none", modes="off", controls="gsm8k", dry_run=True)
    out = capsys.readouterr().out
    assert "sandbox watcher" not in out and "sandbox owner" not in out


def test_the_ownership_helpers_mint_owner_pointer_record_and_sweep(L, tmp_path, monkeypatch):
    rid = L.mint_run_id("abc1234" + "0" * 33)
    assert rid.startswith("abc1234-") and len(rid) == 7 + 1 + 8
    owner = L.sandbox_owner(rid, "none")
    assert owner.run_id == rid and owner.cell == "none" and owner.function_call_id.startswith("local-")
    # outside a container the attempt id is still unique per process; an unlaunched
    # direct .remote still gets a (marked) run id, never an ownerless sandbox
    assert L.sandbox_owner("", "none").run_id.startswith("unlaunched-")
    out = tmp_path / "out"
    record = L.sandbox_record_path(out, owner)
    assert record == out / "watch" / f"sandboxes_{owner.function_call_id}.jsonl"
    cfg = {"name": "tinycap-e4b-base-think_off", "model": "gemma4e4b", "mode": "off", "cell": "none"}
    p = L.write_attempt_pointer(out, owner, cfg=cfg, controls=[BCB], tree_sha="a" * 40, record=record)
    doc = json.loads(p.read_text())
    assert p == out / "watch" / f"attempt_{owner.function_call_id}.json"
    assert set(doc) == set(L.POINTER_FIELDS) and doc["run_id"] == rid and doc["cell"] == "none"
    assert doc["sandbox_record"] == str(record) and doc["sandbox_app_name"] == "inspect_modal_sandbox"
    # the driver environment: the tuple + the record, nothing else
    env = scope.owner_env(owner, record)
    assert set(env) == {*scope.OWNER_ENV_VARS, scope.SANDBOX_RECORD_ENV}
    assert scope.owner_from_env(env) == owner
    # the per-attempt COMPLETION record (stage 07's receipt, round 4): the owner tuple, what the attempt drove, what was left to do when it
    # started and what it finished (control -> the store key), the REC's status, sandboxed or not, the tree — beside the pointer, named for the call
    cp = L.write_completion_record(out, owner, cfg=cfg, controls=[BCB, BFCL200], remaining=[BFCL200], completed={BFCL200: "none__controls__k1"},
                                   status="OK", sandboxed=False, tree_sha="a" * 40)
    cdoc = json.loads(cp.read_text())
    assert cp == L.completion_record_path(out, owner.function_call_id) == out / "watch" / f"completion_{owner.function_call_id}.json"
    assert set(cdoc) == set(L.COMPLETION_FIELDS) and cdoc["run_id"] == rid and cdoc["cell"] == "none" and cdoc["function_call_id"] == owner.function_call_id
    assert cdoc["controls"] == [BCB, BFCL200] and cdoc["remaining"] == [BFCL200] and cdoc["completed"] == {BFCL200: "none__controls__k1"}
    assert cdoc["status"] == "OK" and cdoc["sandboxed"] is False and cdoc["tree_sha"] == "a" * 40 and cdoc["model"] == "gemma4e4b" and cdoc["mode"] == "off"
    # the record is NOT an attempt pointer: the watcher and the analysis read `attempt_*.json` by prefix and never see it; watch/ holds exactly
    # the pointer and the record at this point (the sweep certificate is written below)
    assert not cp.name.startswith("attempt_") and sorted(f.name for f in (out / "watch").iterdir()) == sorted([p.name, cp.name])
    # where the solo worker writes it: at the end of `_drive_cell`, after the status is final and before the RESULT stamp, on the solo path only
    worker = LAUNCHER.read_text(encoding="utf-8")
    drive = worker[worker.index("def _drive_cell("):]
    assert "if not shared_standup:\n        # the per-attempt COMPLETION record" in drive
    assert drive.index('rec["status"] = rec.pop("status_override")') < drive.index("write_completion_record(") < drive.index("result_stamp_line(rec")
    assert 'return Path(out_dir) / WATCH_DIR / f"completion_{function_call_id}.json"' in worker
    # the finally sweep under the stub modal (no App.lookup): RECORDED as unverified with the
    # error — never a silent verified_empty — and the certificate written per attempt
    cert = L.sweep_owned_sandboxes(owner, out, record)
    assert cert["verified_empty"] is False and "error" in cert and cert["owner"] == owner.as_dict()
    written = json.loads(L.sweep_certificate_path(out, owner.function_call_id).read_text())
    assert written == json.loads(json.dumps(cert, default=str)) and written["certificate"] == "sweep"
    # a fake modal that lists two owned boxes and a sibling's: only the owned are terminated

    class _SB:
        def __init__(self, sid, tags):
            self.object_id, self._tags, self.terminated = sid, tags, False

        def get_tags(self):
            return dict(self._tags)

        def terminate(self):
            self.terminated = True

    mine = [_SB("sb-1", scope.owner_tags(owner, BCB, {"__sample_id__": 4})),
            _SB("sb-2", scope.owner_tags(owner, BCB, {"__sample_id__": 8}))]
    sib = _SB("sb-sib", scope.owner_tags(scope.Owner(rid, DERIVED, "fc-9"), BCB, {"__sample_id__": 4}))
    calls = {}

    def _list(app_id=None, tags=None):
        calls["app_id"], calls["tags"] = app_id, dict(tags)
        # Modal's server-side filter: only sandboxes carrying ALL the tags
        return [b for b in [*mine, sib] if not b.terminated and all(b._tags.get(k) == v for k, v in tags.items())]

    monkeypatch.setattr(L.modal, "App", SimpleNamespace(lookup=lambda name, create_if_missing: SimpleNamespace(app_id=f"ap-{name}")))
    monkeypatch.setattr(L.modal, "Sandbox", SimpleNamespace(list=_list))
    cert = L.sweep_owned_sandboxes(owner, out, record)
    assert cert["verified_empty"] is True and cert["ids_terminated"] == ["sb-1", "sb-2"]
    assert not sib.terminated and cert["left"] == [] and cert["sandbox_app_id"] == "ap-inspect_modal_sandbox"
    assert calls["tags"] == scope.owner_filter(owner) and calls["app_id"] == "ap-inspect_modal_sandbox"
    events = scope.read_events(record)
    assert sorted(e["sandbox_id"] for e in events if e["event"] == "terminated") == ["sb-1", "sb-2"]
    assert all(e["verified_by"] == "relist" for e in events if e["event"] == "terminated")


def test_run_configuration_exports_the_owner_and_writes_the_pointer_for_a_sandboxed_leg(L, tmp_path, monkeypatch):
    """Up to the (faked) standup: the driver environment carries the owner
    tuple + record path, the per-attempt pointer is written under watch/ ONLY
    when a sandboxed leg is among the remaining controls, and the REC names the
    owner."""
    from test_tinycap_launcher import _FakeStandupProc, _tree

    repos = tmp_path / "repos"
    tree = _tree(repos, "f6cf462")
    # the launcher reads the two pins (and the BCB family map the verified loader
    # recomputes per_family through) off the STAGED tree — a tree without them refuses
    for rel in (C.BCB_PIN_PATH, C.BFCL_PIN_PATH, C.BCB_LIB_FAMILIES_PATH):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text((REPO / rel).read_text())
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "0")
    monkeypatch.setattr(L.subprocess, "Popen", _FakeStandupProc)
    monkeypatch.setattr(L, "LEDGER_WAIT_S", 0.01)
    monkeypatch.setattr(L, "client_lock_sha256", lambda tree: None)
    out = tmp_path / "gen"
    (cfg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=[BCB])
    cfg = dict(cfg, out_dir=str(out))
    import shutil as _shutil
    _shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)      # a stale record would read as a standup
    rec = L._run_configuration(cfg, "f6cf462", run_id="f6cf462-cafebabe")
    assert rec["status"] == "LEDGER_FAILED"                                 # the fake ledger dies at once
    assert rec["sandboxed"] is True and rec["sandbox_owner"]["run_id"] == "f6cf462-cafebabe"
    fcid = rec["sandbox_owner"]["function_call_id"]
    env = _FakeStandupProc.last_env
    assert env[scope.OWNER_ENV_RUN_ID] == "f6cf462-cafebabe" and env[scope.OWNER_ENV_CELL] == "none"
    assert env[scope.OWNER_ENV_FUNCTION_CALL_ID] == fcid
    assert env[scope.SANDBOX_RECORD_ENV] == str(out / "watch" / f"sandboxes_{fcid}.jsonl") == rec["sandbox_record"]
    pointer = json.loads(Path(rec["sandbox_pointer"]).read_text())
    assert pointer["function_call_id"] == fcid and pointer["controls"] == [BCB]
    assert Path(rec["sandbox_pointer"]) == out / "watch" / f"attempt_{fcid}.json"
    # the standup argv carries the upstream timeout in its fixed part
    argv = rec["standup_argv"]
    assert argv[argv.index(C.FLAG_UPSTREAM_TIMEOUT_S) + 1] == "7200"
    # a non-sandboxed leg: the owner still rides the env (the provider never runs), no pointer
    (cfg2,) = L.plan_configurations(cells=["none"], modes=["off"], controls=[BFCL200])
    rec2 = L._run_configuration(dict(cfg2, out_dir=str(tmp_path / "gen2")), "f6cf462", run_id="f6cf462-cafebabe")
    assert rec2["sandboxed"] is False and "sandbox_pointer" not in rec2
    assert not (tmp_path / "gen2" / "watch").exists()


def test_run_configuration_and_main_thread_the_run_id(L, monkeypatch, capsys):
    import inspect

    assert "run_id" in inspect.signature(L.run_configuration).parameters
    assert "run_id" in inspect.signature(L._run_configuration).parameters
    src = (REPO / "scripts/modal/tinycap_served.py").read_text()
    # the one keyword dictionary every container receives is built by configuration_kwargs (code review B-P1)
    assert "oos_prompt_budget=container_budget," in src and "run_id=run_id, ladder=ladder_kw, protocol_check=container_protocol_check" in src
    assert "run_id = mint_run_id(sha)" in src
    # ONE FunctionCall per configuration — the launcher spawns each
    # (never .map, whose one id made every cell's attempt read as input 0's)
    # the function is selected per ROW (the Nemotron family runs on
    # image_nemotron); every other row resolves to run_configuration, spawned one per configuration
    # and per CONTROLS as well (a full leg picks the `*_full` variant)
    assert "recs += fan_out(configuration_function(row, sel_controls), cfgs," in src and "run_configuration.map(" not in src
    assert L.configuration_function(C.model_row(C.MODEL_SLUG)) is L.run_configuration
    assert L.configuration_function(C.model_row(C.MODEL_SLUG), ["bigcodebench_200", "bfcl_200"]) is L.run_configuration
    assert L.configuration_function(C.model_row(C.MODEL_SLUG), ["bigcodebench_full"]) is L.run_configuration_full
    assert L.configuration_function(C.model_row("nemotron3nano4b"), ["bfcl_full"]) is L.run_configuration_nemotron_full
    assert L.controls_name_full_task(["bigcodebench_200", "bfcl_full"]) and not L.controls_name_full_task(["bfcl_200"])
    # the launch print names the run id and, for a sandboxed leg, the watcher command
    calls = []
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=lambda *a: {"ok": True, "checks": {}}))
    monkeypatch.setattr(L, "run_configuration",
                        _spawn_only(lambda c: {"name": c["name"], "status": "OK"}, lambda c, k: calls.append((c, k))))
    L.main(app_name=L.APP_NAME, sha="abc1234", pf=True, pass1=True, cells="none,prompted__safety_refusal",
           modes="off", controls=BCB, out_suffix="s7a")
    out = capsys.readouterr().out
    assert len(calls) == 2 and all(k["run_id"].startswith("abc1234-") for _, k in calls)
    assert [c["cell"] for c, _ in calls] == ["none", "prompted__safety_refusal"]
    assert out.count("[tinycap] spawned ") == 2 and "-> FunctionCall fc-" in out
    assert "[tinycap] run id: abc1234-" in out
    assert "[tinycap] sandboxed leg planned — arm the watcher now: ~/venvs/modal/bin/python" in out
    assert "--suffix s7a --model gemma4e4b --cells none" in out


def test_preflight_subset_pin_and_derived_provenance_checks_are_wired(L):
    src = (REPO / "scripts/modal/tinycap_served.py").read_text()
    assert 'chk("subset_pins_regenerate", subset_pins)' in src
    assert 'chk("derived_object_provenance", derived_provenance)' in src
    assert "preflight_derived_object(obj, derived_id=cell, parents_root=DERIVED_PARENTS_ROOT)" in src
    assert L.DERIVED_PARENTS_ROOT == "/vol/out"
    assert 'raise AssertionError(f"derived object(s) refused by the sha-bound parent check: {bad}")' in src
    assert 'rep["derived_preflight"] = derived' in src


# --------------------------------------------------------------------------- #
# bcb-bfcl-200: the truncation ladder — the geometry gate, the                  #
# identity, the strict PARTIAL validator, the launch args, the admission and    #
# the two invocations, the probe walk, the watcher fixes                        #
# --------------------------------------------------------------------------- #

import copy  # noqa: E402
import dataclasses  # noqa: E402
import importlib.util  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402

from sisteer.capeval.serving import geometry as sg  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402
from sisteer.capeval.tinycap import completion  # noqa: E402
from sisteer.capeval.tinycap import rung_probe as RP  # noqa: E402
from test_tinycap_launcher import FULL_SHA, LAUNCHER, _Call, _tree  # noqa: E402

BCB_TASK, BFCL_TASK = C.LADDER_ROW_TASKS
PIN_B, PIN_F = C.load_bcb_pin(REPO), C.load_bfcl_pin(REPO)
SEL_B, SEL_F = list(PIN_B["ids"][:3]), list(PIN_F["ids"][:2])
SEL_IDS = {BCB_TASK: SEL_B, BFCL_TASK: SEL_F}
CAPS = {BCB_TASK: 16384, BFCL_TASK: 16384}
KV_REPORT = {"kv_num_cache_groups": 1, "kv_num_gpu_blocks": 33_000, "kv_block_size": 16}
REFUSAL_18 = ("VllmServerError: resolved scheduler state does not match the composed no-chunking invariant:\n"
              "- worker 0: " + str(vs.kv_coverage_problem(KV_REPORT, max_num_seqs=18, max_model_len=32_768)) + "\n")


def _block(task, ids, cap_from, prompt):
    return {"ids": list(ids), "n": len(ids), "n_selected": len(ids), "ids_sha256": C.ids_sha256(ids),
            "cap_from": cap_from, "cap_to": 16384, "max_prompt_tokens_rung1": prompt}


def _selection_doc(ids_by_task):
    sel = {}
    if BCB_TASK in ids_by_task:
        sel[BCB_TASK] = _block(BCB_TASK, ids_by_task[BCB_TASK], 4096, 9_000)
    if BFCL_TASK in ids_by_task:
        sel[BFCL_TASK] = _block(BFCL_TASK, ids_by_task[BFCL_TASK], 1024, 2_100)
    return {"schema": 1, "generated_by": "scripts/tinycap_ladder.py select", "generated_at": "2026-09-10T00:00:00Z",
            "from_suffix": "bcbbfcl200", "threshold": 0.01, "full_n": 200, "cells": ["none"], "detector": {},
            "fired_any": True, "sources": {}, "selection": sel, "selection_sha256": RP.selection_digest(sel)}


def _write_selection(path, ids_by_task):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_selection_doc(ids_by_task), indent=1, sort_keys=True))
    return path


def _ladder_cfg(L, controls=(BCB, BFCL200), S=16, cell="none"):
    (cfg,) = L.plan_configurations(cells=[cell], modes=["off"], controls=list(controls))
    tasks = {t for c in controls for t in L.expected_tasks(c, "off")}
    return dict(cfg, ladder_rung=RP.accepted_block(S), cap_overrides_by_task={t: CAPS[t] for t in tasks},
                item_selection={t: SEL_IDS[t] for t in tasks})


def _ladder_tree(tmp_path, sha="f6cf462"):
    """A fake staged tree with the pins and the row's checked-in config (the
    certificate binds its digest; the expectation reads the pins off it)."""
    repos = tmp_path / "repos"
    tree = _tree(repos, sha)
    for rel in (C.BCB_PIN_PATH, C.BFCL_PIN_PATH, C.BCB_LIB_FAMILIES_PATH, C.model_row(C.MODEL_SLUG).cfg):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text((REPO / rel).read_text())
    return repos, tree


def _ladder_finish(cell_dir, task, log_ids, *, cap=16384, limits=None, n_expected_full=200, n_items=None):
    """A rung-2 task the way ``completion.finish_task`` files it: the summary
    (every field the outcome record copies), the outcome record over ``log_ids``
    and the PARTIAL marker (``limits`` non-empty)."""
    n = len(log_ids) if n_items is None else n_items
    summary = {"task": task, C.METRIC_ACC: 0.5, C.METRIC_N: n, "n_requests": n, "item_set": "pinned200",
               "thinking": "off", "think_split_mode": C.THINK_SPLIT_RECORD_ONLY, "unterminated_policy": "score",
               "max_tokens": cap, "samples_per_item": 1, "n_scored": n, "n_unscored": 0, "n_scored_samples": n,
               "n_unscored_samples": 0, "n_stop_reason": {"stop": n}, "n_at_cap": 0, "n_think_block_present": 0,
               "n_think_block_unterminated": 0, "n_unterminated_not_at_cap": 0, "n_completion_tokens": 20 * n}
    items = [{"item_id": i, "sample_idx": 0, "attempt": 1, "scored": True, "value": 1.0, "unparseable": False,
              "stop_reason": "stop", "at_cap": False, "think_block_present": False, "think_block_unterminated": False,
              "stray_open_tag": False, "content_empty_with_reasoning": False, "prompt_tokens": 80,
              "completion_tokens": 20, "seed": 0} for i in log_ids]
    outcomes = {**{f: summary[f] for f in C.TASK_OUTCOMES_SUMMARY_FIELDS}, "n_attempted_items": n,
                "n_attempted_samples": n, "producer_tree_sha": FULL_SHA, "items": items}
    return completion.finish_task(cell_dir, task=task, n_items=n, n_expected_full=n_expected_full,
                                  limits=dict(C.LADDER_LIMITS) if limits is None else limits, n_rows=n,
                                  n_requests=n, n_rejected=0, wall_clock_s=1.0, summary=summary, outcomes=outcomes)


def test_the_geometry_gate_serves_a_ladder_rung_through_served_geometry_and_refuses_the_rest(L, tmp_path, monkeypatch):
    cfg = _ladder_cfg(L)
    g, rec = L.served_geometry(cfg, 1)
    assert (g.max_model_len, g.max_num_seqs, g.seqs_source, g.window_source) == (32_768, 16, "override", "ladder_window")
    assert g.overridden and not g.is_identity and not g.over_budget and g.identity_S == 18
    assert g.budget.measured and g.model_key == "gemma4e4b" and g.batched_tokens == 16 * 32_768 == 524_288
    assert rec["rung_source"] == "probe" and rec["ladder_reduced_s"] is True
    assert rec["identity"] is False and rec["over_budget"] is False and rec["max_num_seqs"] == 16
    assert rec["ladder_window"] == 32_768 and rec["ladder_seqs"] == [18, 16, 14, 12] and rec["identity_max_num_seqs"] == 18
    assert rec["batched_token_budget"] == 589_824 and rec["budget_source"] == "measured"
    # the record's shared fields ARE model_budget_record's (one field builder)
    ordinary = L.model_budget_record(L.configuration_geometry(cfg, 1))
    # bcb-bfcl-200-widen S6: the record names the MODEL whose ladder sized it (ladder_model_key)
    assert set(ordinary) < set(rec) and set(rec) - set(ordinary) == {
        "rung_source", "ladder_window", "ladder_seqs", "identity_max_num_seqs", "ladder_reduced_s", "ladder_model_key"}
    assert rec["ladder_model_key"] == "gemma4e4b"
    # S = 18: the identity rung
    g18, rec18 = L.served_geometry(dict(cfg, ladder_rung=RP.accepted_block(18)), 1)
    assert (g18.max_num_seqs, g18.seqs_source, g18.is_identity) == (18, "model_budget", True)
    assert rec18["rung_source"] == "identity" and rec18["ladder_reduced_s"] is False and rec18["identity"] is True
    # ONE entry point: served_geometry IS ladder_geometry for a rung configuration
    assert L.ladder_geometry(cfg)[0] == g
    # the standup argv carries the rung (no --eval, the model key rides)
    argv = L._standup_cmd("/tree", "c", cfg["cell"], "d", tmp_path, tmp_path / "STOP", g, 64, cfg["standup_extra"])
    pairs = dict(zip(argv[2::2], argv[3::2]))
    assert (pairs["--max-model-len"], pairs["--max-num-seqs"], pairs["--model-key"]) == ("32768", "16", "gemma4e4b")
    assert "--eval" not in argv and "--allow-over-budget" not in argv
    # refusals of the gate, each by name
    base = L.configuration_geometry(cfg, 1)

    def geom(L_, S, seqs):
        return dataclasses.replace(base, max_model_len=L_, max_num_seqs=S, window_source="ladder_window", seqs_source=seqs)
    with pytest.raises(L.LadderGeometryRefused, match="LADDER_WINDOW"):
        L.ladder_budget_record(geom(16_384, 16, "override"), RP.accepted_block(16))
    with pytest.raises(L.LadderGeometryRefused, match="not on the ladder"):
        L.ladder_budget_record(geom(32_768, 17, "override"), {**RP.accepted_block(16), "max_num_seqs": 17})
    with pytest.raises(L.LadderGeometryRefused, match="seqs_source"):
        L.ladder_budget_record(geom(32_768, 16, "model_budget"), RP.accepted_block(16))
    with pytest.raises(L.LadderGeometryRefused, match="seqs_source"):
        L.ladder_budget_record(geom(32_768, 16, "override"), {**RP.accepted_block(16), "seqs_source": "model_budget"})
    with pytest.raises(L.LadderGeometryRefused, match="not below identity"):
        L.ladder_budget_record(geom(32_768, 18, "model_budget"), {**RP.accepted_block(18), "rung_source": "probe"})
    with pytest.raises(L.LadderGeometryRefused, match="rung_source"):
        L.ladder_budget_record(geom(32_768, 16, "override"), {**RP.accepted_block(16), "rung_source": "guess"})
    with pytest.raises(L.LadderGeometryRefused, match="window_source"):
        L.ladder_budget_record(dataclasses.replace(base, max_model_len=32_768, max_num_seqs=16, seqs_source="override"),
                               RP.accepted_block(16))
    # S x L over the budget: a (fake) measured row of 16 x 32,768 refuses S = 18
    small = sg.ModelBudget(model_key="gemma4e4b", batched_token_budget=524_288,
                           basis=sg.BudgetBasis(method="test", max_model_len=32_768, max_num_seqs=16,
                                                gpu_memory_utilization=0.9, kv_tokens_reported=None, probe_job_id=None,
                                                measured_on="test", source="test"))
    rung18 = RP.accepted_block(18)
    monkeypatch.setitem(sg.BUDGET_BY_MODEL, "gemma4e4b", small)
    # bcb-bfcl-200-widen S6: the MODEL's ladder is derived from its budget row, so with a 16 x 32,768
    # budget the rungs are (16, 14, 12) and S = 18 is refused as off the ladder (the "> the measured
    # budget" belt behind it can no longer be reached from a spec rung — the same fact, named first)
    with pytest.raises(L.LadderGeometryRefused, match="not on the ladder|> the measured budget"):
        L.served_geometry(dict(cfg, ladder_rung=rung18), 1)
    assert RP.ladder_spec("gemma4e4b").seqs == (16, 14, 12)
    # a provisional / default budget refuses: the ladder serves nothing the model did not measure
    # (the block is built BEFORE the table is emptied: accepted_block itself reads the model's ladder — S6)
    rung16 = RP.accepted_block(16, identity=18)
    monkeypatch.setitem(sg.BUDGET_BY_MODEL, "gemma4e4b", None)
    with pytest.raises(L.LadderGeometryRefused, match="MEASURED"):
        L.served_geometry(dict(cfg, ladder_rung=rung16), 1)
    with pytest.raises(RP.LadderError, match="MEASURED"):
        RP.accepted_block(16, identity=18)                     # and the pure half refuses by the same name
    monkeypatch.undo()
    # an "override" geometry WITHOUT ladder_rung is still refused by the ordinary record
    with pytest.raises(L.UnmeasuredBudget, match="not MEASURED"):
        L.model_budget_record(dataclasses.replace(base, seqs_source="override"))
    # the source-grep pins the gate was built not to move
    src = LAUNCHER.read_text()
    assert "served_geometry(cfg, pass_no, tree)" in src and "served_geometry(cfg, 1, tree)" in src
    # ifeval-canonical-battery: the fourth site is configuration_geometry's IFEval branch
    # (resolve_geometry(<capability:ifeval_cap>, strict_identity=True, model_key=model_key))
    assert "served_geometry(cfg, pass_no)" in src and src.count("model_key=model_key)") == 4
    assert not re.search(r"\b36\b", src)
    assert 'if cfg.get("ladder_rung"):\n        return ladder_geometry(cfg, tree)' in src


def test_the_ladder_expectation_hashes_the_tuple_and_the_store_trust_holds_a_rung_store_to_it(L, tmp_path):
    cfg = _ladder_cfg(L)
    geom, _ = L.served_geometry(cfg, 1)
    exp = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=1, tree_sha=FULL_SHA)
    assert exp.ladder and dict(exp.limits) == {"item_selection": 1} == dict(C.LADDER_LIMITS)
    assert exp.served["max_model_len"] == 32_768 and exp.served["max_num_seqs"] == exp.served["concurrency"] == 16
    assert set(exp.optional_identity()) == {"bcb_item_set", "bcb_sandbox", "bfcl_item_set",
                                            "cap_overrides_by_task", "item_selection"}
    assert exp.item_selection == C.item_selection_identity(SEL_IDS) == exp.as_dict()["item_selection"]
    assert exp.as_dict()["cap_overrides_by_task"] == CAPS and exp.item_selection_ids == SEL_IDS
    # a smoke cap beside the rung refuses; a tuple naming tasks the run does not file refuses
    with pytest.raises(ValueError, match="takes no --limit"):
        L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=1, inspect_limit=5)
    with pytest.raises(ValueError, match="fired legs alone"):
        L.expected_identity(dict(cfg, controls=[BFCL200]), geom, pass_no=1, controls=[BFCL200], samples=1)
    # a battery expectation carries none of it, byte for byte
    std = _expected(L, cell="none", controls=[BCB, BFCL200])
    assert not std.ladder and std.item_selection is None and "item_selection" not in std.as_dict()
    # store trust: a rung store with the tuple is trusted under the rung expectation ...
    out = tmp_path / "gen"
    store = _store(out, exp, tasks=[BCB, BFCL200], extra_identity=exp.optional_identity())
    assert L.untrusted_store_reason(store, expected=exp) is None
    # ... never under the battery one (and a battery store never under the rung one)
    why = L.untrusted_store_reason(store, expected=std)
    assert why and "cap_overrides_by_task" in why and "truncation ladder" in why
    battery = _store(tmp_path / "b", std, tasks=[BCB, BFCL200], extra_identity=std.optional_identity())
    why = L.untrusted_store_reason(battery, expected=exp)
    assert why and "truncation ladder's tuple" in why
    # another selection or another cap is another rung, named
    other_sel = exp._replace(item_selection_ids={**SEL_IDS, BFCL_TASK: list(PIN_F["ids"][:3])})
    why = L.untrusted_store_reason(store, expected=other_sel)
    assert why and "tinycap.item_selection" in why and "another item selection" in why
    other_cap = exp._replace(cap_overrides_by_task={**CAPS, BFCL_TASK: 8192})
    why = L.untrusted_store_reason(store, expected=other_cap)
    assert why and "tinycap.cap_overrides_by_task" in why and "another rung-2 cap" in why
    # a smoke PARTIAL store (limits inspect_limit) is not a ladder store
    smoke = _store(tmp_path / "s", exp, tasks=[BCB, BFCL200], extra_identity=exp.optional_identity(),
                   limits={"inspect_limit": 5})
    why = L.untrusted_store_reason(smoke, expected=exp)
    assert why and "tinycap.limits {'inspect_limit': 5}" in why
    # the driver argv carries the tuple LAST, exactly the contracts' spelling
    argv = L._driver_cmd("/tree", cfg, {"proxy_url": "u", "routing_table": "r"}, "l", geom, controls=cfg["controls"],
                         limit=0, inspect_limit=0, samples=1, cfg_path="c", vllm_log=None)
    assert argv[-4:] == [C.FLAG_CAP_OVERRIDE, C.format_cap_overrides(CAPS), C.FLAG_ITEM_SELECTION_JSON,
                         C.format_item_selection_json(SEL_IDS)]
    assert argv[argv.index("--concurrency") + 1] == "16"
    plain = L._driver_cmd("/tree", dict(cfg, ladder_rung=None), {"proxy_url": "u", "routing_table": "r"}, "l", geom,
                          controls=cfg["controls"], limit=0, inspect_limit=0, samples=1, cfg_path="c", vllm_log=None)
    assert C.FLAG_CAP_OVERRIDE not in plain and C.FLAG_ITEM_SELECTION_JSON not in plain


def test_ladder_partial_reason_is_the_one_strict_validator(L, tmp_path):
    cfg = _ladder_cfg(L, controls=(BFCL200,))
    geom, _ = L.served_geometry(cfg, 1)
    exp = L.expected_identity(cfg, geom, pass_no=1, controls=[BFCL200], samples=1, tree_sha=FULL_SHA)
    out = tmp_path / "gen"
    store = _store(out, exp, tasks=[BFCL200], extra_identity=exp.optional_identity())
    assert L.ladder_partial_reason(store, BFCL_TASK, exp) == "no partial marker"
    fin = _ladder_finish(store, BFCL_TASK, SEL_F)
    assert fin["complete"] is False
    assert L.ladder_partial_reason(store, BFCL_TASK, exp) is None
    report = L.ladder_resume_report(out, [BFCL200], exp)
    assert report["controls"][BFCL200]["done_in"] == store.name and report["stores"][store.name] is None
    # the ordinary resume report never accepts a PARTIAL as done
    assert L.resume_report(out, [BFCL200], exp)["controls"][BFCL200]["done_in"] is None
    # a battery expectation cannot ask for the ladder report
    with pytest.raises(ValueError, match="ladder expectation"):
        L.ladder_resume_report(out, [BFCL200], _expected(L, cell="none", controls=[BFCL200]))

    def fresh(name, **kw):
        s = _store(tmp_path / name, exp, tasks=[BFCL200], extra_identity=exp.optional_identity())
        _ladder_finish(s, BFCL_TASK, kw.pop("log_ids", SEL_F), **kw)
        return s
    # a smoke PARTIAL is not a ladder store
    why = L.ladder_partial_reason(fresh("smoke", limits={"inspect_limit": 5}), BFCL_TASK, exp)
    assert why and "{'inspect_limit': 5}" in why and "smoke PARTIAL" in why
    # a selected id missing from the outcomes (and the count with it)
    why = L.ladder_partial_reason(fresh("short", log_ids=SEL_F[:-1]), BFCL_TASK, exp)
    assert why and "n_items 1" in why
    # the right count over the WRONG ids
    why = L.ladder_partial_reason(fresh("wrong", log_ids=[SEL_F[0], "not_selected"]), BFCL_TASK, exp)
    assert why and "outcomes items != the selection" in why and "not_selected" in why
    # another cap
    why = L.ladder_partial_reason(fresh("cap", cap=4096), BFCL_TASK, exp)
    assert why and "max_tokens 4096" in why
    # the frozen full count
    why = L.ladder_partial_reason(fresh("full", n_expected_full=199), BFCL_TASK, exp)
    assert why and "n_expected_full 199" in why
    # a malformed marker (an extra field) and a done marker
    mal = fresh("mal")
    mp = completion.partial_marker_path(mal, BFCL_TASK)
    mp.write_text(json.dumps({**json.loads(mp.read_text()), "extra": 1}))
    why = L.ladder_partial_reason(mal, BFCL_TASK, exp)
    assert why and "partial marker fields" in why
    done = fresh("done")
    completion.partial_marker_path(done, BFCL_TASK).unlink()
    (done / C.DONE_DIR).mkdir()
    (done / C.DONE_DIR / C.marker_filename(BFCL_TASK)).write_text("{}")
    assert "done marker" in L.ladder_partial_reason(done, BFCL_TASK, exp)
    # the outcomes record gone / refused
    gone = fresh("gone")
    completion.task_outcomes_path(gone, BFCL_TASK).unlink()
    assert "no outcomes record" in L.ladder_partial_reason(gone, BFCL_TASK, exp)
    completion.task_outcomes_path(gone, BFCL_TASK).write_text("{}")
    assert "outcomes record refused" in L.ladder_partial_reason(gone, BFCL_TASK, exp)


def test_ladder_launch_args_and_configurations_are_validated_on_the_cpu(L, tmp_path):
    sel_path = _write_selection(tmp_path / "ladder_selection.json", SEL_IDS)
    caps = C.format_cap_overrides(CAPS)
    assert L.ladder_launch_args() is None
    ladder = L.ladder_launch_args(cap_override=caps, item_selection=str(sel_path), out_suffix="ladder-1")
    assert ladder.item_selection == SEL_IDS and ladder.cap_overrides_by_task == CAPS and ladder.operator_rung is None
    assert ladder.probe_cell == "" and ladder.cells == ()                 # bound to a plan by ladder_launch_bound
    assert ladder.selection_sha256 == RP.selection_digest(_selection_doc(SEL_IDS)["selection"])
    assert L.LadderLaunch.from_dict(json.loads(json.dumps(ladder.as_dict()))) == ladder
    op = L.ladder_launch_args(cap_override=caps, item_selection=str(sel_path), out_suffix="l", ladder_rung="32768,14")
    assert op.operator_rung == (32_768, 14)
    for kw, needle in ((dict(cap_override=caps), "travel together"),
                       (dict(item_selection=str(sel_path)), "travel together"),
                       (dict(cap_override=caps, item_selection=str(sel_path)), "--out-suffix"),
                       (dict(cap_override=f"{BFCL_TASK}=16384", item_selection=str(sel_path), out_suffix="l"), "same legs"),
                       (dict(cap_override=caps.replace("16384", "8192", 1), item_selection=str(sel_path), out_suffix="l"),
                        "cap_to"),
                       (dict(cap_override=caps, item_selection=str(sel_path), out_suffix="l", ladder_rung="32768,18"),
                        "not below the identity"),
                       (dict(cap_override=caps, item_selection=str(sel_path), out_suffix="l", ladder_rung="32768,17"),
                        "not on the ladder"),
                       (dict(cap_override=caps, item_selection=str(sel_path), out_suffix="l", ladder_rung="16384,16"),
                        "LADDER_WINDOW")):
        with pytest.raises((ValueError, RP.LadderError), match=needle):
            L.ladder_launch_args(**kw)
    # the configurations: the fired legs alone, thinking OFF
    cfgs = L.plan_configurations(cells=["none", DERIVED], modes=["off"], controls=[BCB, BFCL200],
                                 model="gemma4e4b-sdfbase")
    lcfgs = L.ladder_configurations(cfgs, ladder)
    assert all(c["cap_overrides_by_task"] == CAPS and c["item_selection"] == SEL_IDS for c in lcfgs)
    assert all("ladder_rung" not in c for c in lcfgs)                 # the rung is the certificate's, later
    with pytest.raises(ValueError, match="fired legs alone"):
        L.ladder_configurations(L.plan_configurations(cells=["none"], modes=["off"], controls=[BFCL200]), ladder)
    with pytest.raises(ValueError, match="thinking-OFF"):
        L.ladder_configurations([dict(cfgs[0], mode="on")], ladder)
    # the launch argv spells the flags
    argv = " ".join(L.launch_argv("abc1234", pass1=True, cap_override=caps, item_selection=str(sel_path),
                                  ladder_rung="32768,14", ladder_reprobe=True, out_suffix="l"))
    assert f"{C.FLAG_CAP_OVERRIDE} {caps}" in argv and f"{C.FLAG_ITEM_SELECTION} {sel_path}" in argv
    assert f"{C.FLAG_LADDER_RUNG} 32768,14" in argv and C.FLAG_LADDER_REPROBE in argv
    assert C.FLAG_CAP_OVERRIDE not in " ".join(L.launch_argv("abc1234", pass1=True))
    # the driver probe values name the two ladder flags
    assert set(C.TINYCAP_LADDER_DRIVER_FLAGS) <= set(L.DRIVER_PROBE_VALUES)


class _LadderProcs:
    """A Popen double for ``_run_configuration``: the standup writes the
    ``endpoint.json`` the launcher reads (the rung it was asked for), the ledger
    writes its endpoint record; both stay alive until stopped."""

    def __init__(self, argv, *a, **k):
        self.argv, self.rc = list(argv), None
        joined = " ".join(self.argv[:3])
        if "capctrl_serving_up.py" in joined:
            out = Path(self.argv[self.argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            L_, S = int(self.argv[self.argv.index("--max-model-len") + 1]), int(self.argv[self.argv.index("--max-num-seqs") + 1])
            (out / "endpoint.json").write_text(json.dumps({
                "proxy_url": "http://127.0.0.1:1/v1", "routing_table": str(out / "routing_table.json"),
                "tool_calls": "--tool-calls" in self.argv, "reasoning_parser": None, "agent_loop_entry": True,
                "geometry": {"max_model_len": L_, "max_num_seqs": S}}))
        elif "usage_ledger" in joined:
            ep = Path(self.argv[self.argv.index("--endpoint-json") + 1])
            ep.write_text(json.dumps({"ledger_url": "http://127.0.0.1:1/v1"}))

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        # lifecycle-faithful: the REAL standup unlinks its ``endpoint.json`` as it
        # tears down (capctrl_serving_up.py, after ``su.tear_down()``), so a launcher that reads the record after
        # ``_tear_down_standup`` finds nothing — the double deletes it exactly where the producer does
        out = Path(self.argv[self.argv.index("--out") + 1])
        (out / "endpoint.json").unlink(missing_ok=True)
        return self.rc if self.rc is not None else 0

    def terminate(self):
        self.rc = 0

    def kill(self):
        self.rc = 0


def _ladder_world(L, tmp_path, monkeypatch, *, ids_by_task, controls, cells="none,prompted__safety_refusal"):
    """The two-invocation world: a fake staged tree, a fake volume root, a
    selection file, a preflight double computing the REAL admission, a probe
    double writing a REAL certificate (through the real prelude, so a re-probe
    supersedes), containers spawned ONE per configuration into the REAL
    ``_run_configuration`` under a scripted standup / ledger and a driver double
    that files the store through the real ``finish_task``."""
    repos, tree = _ladder_tree(tmp_path)
    gen = tmp_path / "gen"
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(gen))
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "0")
    monkeypatch.setattr(L, "client_lock_sha256", lambda tree: None)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L.subprocess, "Popen", _LadderProcs)
    # the launcher's standup dir is the REAL /tmp/srv_<name>; a record this world's
    # scripted standup / ledger leaves there would satisfy another test's
    # `_wait_for` (a stale endpoint reads as a standup that came up) — cleared
    # before every spawn and again at teardown
    names = [c["name"] for c in L.plan_configurations(cells=cells.split(","), modes=["off"], controls=controls.split(","))]
    for n in names:
        shutil.rmtree(f"/tmp/srv_{n}", ignore_errors=True)
    monkeypatch.setattr(L, "_ladder_world_srv_names", names, raising=False)
    sel_path = _write_selection(tmp_path / "ladder_selection.json", ids_by_task)
    world = {"probe": [], "spawned": [], "driver": [], "mode": {}, "tree": str(tree), "sel_path": sel_path}

    def _remote(sha, plan_args, expect, ladder=None):
        assert ladder is not None
        ll = L.LadderLaunch.from_dict(ladder)
        cfgs = L.ladder_configurations(L.isolate_out_dirs(L.plan_for(**plan_args), ll.out_suffix), ll)
        adm = L.ladder_admission_report(cfgs, ll, tree=str(tree), tree_sha=FULL_SHA, object_digests={})
        return {"ok": True, "checks": {}, L.LADDER_ADMISSION_FIELD: adm}
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=_remote))

    def _probe(cfg, **kw):
        world["probe"].append(cfg)
        ll = L.LadderLaunch.from_dict(kw["ladder"])
        rec = {}
        pre = L._ladder_certificate_prelude(cfg, kw["sha"], object_digests=kw["object_digests"], ladder=ll,
                                            reprobe=kw["reprobe"], rec=rec, probed_cell=cfg["cell"])
        assert "status" not in pre, pre
        assert cfg["cell"] == ll.probe_cell == "none" and ll.probe_cells == ("none",)   # no steered cell here: the first
        cert = RP.build_certificate(accepted=RP.accepted_block(16, identity=18),
                                    walked=[RP.walked_entry(18, RP.WALK_REFUSED, REFUSAL_18),
                                            RP.walked_entry(16, RP.WALK_ACCEPTED)],
                                    provenance=pre["provenance"], probe_run_id=kw["run_id"], standup_s=100.0)
        assert RP.certificate_is_valid(cert, pre["provenance"]) is None
        RP.write_certificate(pre["path"], cert)
        return {"name": L.LADDER_PROBE_NAME, "status": "OK", "certificate": cert, **rec}
    # the launcher spawns every absent certificate's probe FIRST, then harvests (spawn / get, like fan_out)
    monkeypatch.setattr(L, "probe_rung", SimpleNamespace(remote=_probe, spawn=lambda cfg, **kw: _Call(lambda: _probe(cfg, **kw))))
    # the wave record writer runs its REAL body against the fake volume root
    monkeypatch.setattr(L, "write_ladder_admission", SimpleNamespace(remote=lambda *a, **k: L._write_ladder_admission(*a, **k)))

    def _spawn(cfg, **kw):
        world["spawned"].append(cfg)
        shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)
        return _Call(lambda: L._run_configuration(cfg, **kw))
    monkeypatch.setattr(L, "run_configuration", SimpleNamespace(spawn=_spawn))

    def _driver(argv, **kw):
        world["driver"].append(list(argv))
        cell = argv[argv.index("--cell") + 1]
        cfg = next(c for c in world["spawned"] if c["cell"] == cell)
        out_root = Path(argv[argv.index(C.FLAG_OUT_ROOT) + 1])
        caps = C.parse_cap_overrides(argv[argv.index(C.FLAG_CAP_OVERRIDE) + 1])
        selection = C.parse_item_selection_json(argv[argv.index(C.FLAG_ITEM_SELECTION_JSON) + 1])
        assert caps == cfg["cap_overrides_by_task"] and {t: b["ids"] for t, b in selection.items()} == cfg["item_selection"]
        assert "--limit 0" in " ".join(argv) and f"{C.FLAG_INSPECT_LIMIT} 0" in " ".join(argv)
        geom, _ = L.served_geometry(cfg, 1, world["tree"])
        exp = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=kw_samples[0], tree=world["tree"],
                                  tree_sha=FULL_SHA, reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
        mode = world["mode"]
        store = _store(out_root, exp, tasks=list(cfg["controls"]), extra_identity=exp.optional_identity(),
                       limits=mode.get("limits"))
        for task, ids in cfg["item_selection"].items():
            log_ids = C.ladder_log_ids(task, ids)
            if mode.get("drop_one"):
                log_ids = log_ids[:-1]
            _ladder_finish(store, task, log_ids, limits=mode.get("limits"))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    kw_samples = [C.MATH500_SAMPLES_PER_ITEM]
    monkeypatch.setattr(L.subprocess, "run", _driver)
    world["main"] = dict(app_name=L.APP_NAME, sha="f6cf462", pf=True, pass1=True, cells=cells, modes="off",
                         # native-serving-default: the ladder world's fake standup attests the LENS engine (no
                         # `engine` key) and its rung geometry is the lens instrument's; the launcher's default
                         # is native now, so the world states the lens launch explicitly
                         serving_path="lens",
                         controls=controls, cap_override=C.format_cap_overrides(
                             {t: 16384 for t in ids_by_task}), item_selection=str(sel_path))
    return world


@pytest.fixture
def _clean_srv_dirs(L):
    yield
    for n in getattr(L, "_ladder_world_srv_names", ()):
        shutil.rmtree(f"/tmp/srv_{n}", ignore_errors=True)


def test_the_ladder_two_invocations_first_partial_complete_then_zero_gpu_calls(L, tmp_path, monkeypatch, capsys,
                                                                             _clean_srv_dirs):
    """The FIRST scoring invocation probes
    once, spawns one container per cell, and exits 0 with every REC
    PARTIAL_COMPLETE (the real `_run_configuration`'s post-driver step through
    the real `finish_task` store); the SECOND finds every cell complete under the
    valid certificate and exits 0 with ZERO probe / scoring calls; a stale
    certificate refuses before any GPU (and `--ladder-reprobe` supersedes it,
    recorded); a smoke PARTIAL and a missing selected id end INCOMPLETE."""
    w = _ladder_world(L, tmp_path, monkeypatch, ids_by_task={BFCL_TASK: SEL_F}, controls=BFCL200)
    # the dry run says what it will do — and launches nothing
    L.main(**w["main"], out_suffix="ladder-1", dry_run=True)
    out = capsys.readouterr().out
    assert "[tinycap] ladder: rung 2 of the truncation ladder" in out and "ladder rung: 32768 x 18" in out
    assert f"{C.FLAG_CAP_OVERRIDE} {BFCL_TASK}=16384" in out and "identity keys ['cap_overrides_by_task', 'item_selection']" in out
    assert "limits {'item_selection': 1}" in out and not w["probe"] and not w["spawned"]
    # FIRST invocation
    L.main(**w["main"], out_suffix="ladder-1")
    out = capsys.readouterr().out
    assert len(w["probe"]) == 1 and w["probe"][0]["cell"] == "none"            # no steered cell: the first
    assert [c["cell"] for c in w["spawned"]] == ["none", "prompted__safety_refusal"]
    assert all(c["ladder_rung"] == RP.accepted_block(16) for c in w["spawned"])
    assert "[tinycap] ladder admission: certificate absent" in out
    # the probe REC, the wave-record REC and the two scoring RECs (bcb-bfcl-200-widen S6)
    assert out.count('"status": "PARTIAL_COMPLETE"') == 2 and "[tinycap] 4/4 container(s) OK" in out
    assert out.count("[tinycap] spawned ") == 2 and len(w["driver"]) == 2
    cert_path = RP.certificate_path(str(tmp_path / "gen"), C.MODEL_SLUG, "ladder-1", "none")
    assert cert_path.is_file() and RP.admission_path(str(tmp_path / "gen"), C.MODEL_SLUG, "ladder-1").is_file()
    for cfg in w["spawned"]:
        stores = list((Path(cfg["out_dir"]) / "cells").iterdir())
        assert len(stores) == 1 and completion.partial_marker_path(stores[0], BFCL_TASK).is_file()
        assert (Path(cfg["out_dir"]) / L.SERVED_GEOMETRY_FILENAME).is_file()
    # SECOND invocation: complete under the valid certificate -> zero probe / scoring calls
    w["probe"].clear(); w["spawned"].clear(); w["driver"].clear()
    L.main(**w["main"], out_suffix="ladder-1")
    out = capsys.readouterr().out
    assert not w["probe"] and not w["spawned"] and not w["driver"]
    assert "[tinycap] ladder admission: certificate valid" in out and "zero probe / scoring calls" in out
    assert out.count('"status": "PARTIAL_COMPLETE"') == 2 and "[tinycap] 2/2 container(s) OK" in out
    # a STALE certificate (another selection under the same suffix): refused before any GPU
    other = _write_selection(tmp_path / "other_selection.json", {BFCL_TASK: list(PIN_F["ids"][:3])})
    with pytest.raises(SystemExit, match="never silently re-probed") as e:
        L.main(**{**w["main"], "item_selection": str(other)}, out_suffix="ladder-1")
    assert "selection_sha256" in str(e.value) and not w["probe"] and not w["spawned"]
    capsys.readouterr()
    # ... --ladder-reprobe supersedes it (recorded) and probes anew
    L.main(**{**w["main"], "item_selection": str(other)}, out_suffix="ladder-1", ladder_reprobe=True)
    out = capsys.readouterr().out
    assert len(w["probe"]) == 1 and len(w["spawned"]) == 2 and out.count('"status": "PARTIAL_COMPLETE"') == 2
    assert any(p.name.startswith("rung_probe.") and p.name.endswith(".superseded.json") for p in cert_path.parent.iterdir())
    assert '"superseded"' in out
    # a valid certificate with --ladder-reprobe: nothing to re-probe
    with pytest.raises(SystemExit, match="nothing to re-probe"):
        L.main(**{**w["main"], "item_selection": str(other)}, out_suffix="ladder-1", ladder_reprobe=True)
    capsys.readouterr()
    # NEGATIVE: a store whose PARTIAL carries a smoke limit -> INCOMPLETE, non-zero
    w["probe"].clear(); w["spawned"].clear()
    w["mode"]["limits"] = {"inspect_limit": 5}
    with pytest.raises(SystemExit, match="did NOT finish") as e:
        L.main(**w["main"], out_suffix="ladder-2")
    assert "INCOMPLETE" in str(e.value) and len(w["probe"]) == 1 and len(w["spawned"]) == 2
    out = capsys.readouterr().out
    assert "{'inspect_limit': 5}" in out
    # NEGATIVE: a ladder PARTIAL missing one selected id -> INCOMPLETE
    w["mode"].clear()
    w["mode"]["drop_one"] = True
    w["probe"].clear(); w["spawned"].clear()
    with pytest.raises(SystemExit, match="did NOT finish") as e:
        L.main(**w["main"], out_suffix="ladder-3")
    assert "INCOMPLETE" in str(e.value)
    assert "n_items 1" in capsys.readouterr().out
    # the ladder launch is refused without --pf, with a smoke cap, without a suffix
    w["mode"].clear()
    for kw, needle in ((dict(pf=False), "only behind --pf"), (dict(inspect_limit=5), "rung-2 launch is --pass1"),
                       (dict(smoke=True), "rung-2 launch is --pass1")):
        with pytest.raises(SystemExit, match=needle):
            L.main(**{**w["main"], **kw}, out_suffix="ladder-4")
    with pytest.raises(SystemExit, match="--out-suffix"):
        L.main(**w["main"])
    capsys.readouterr()
    # the REC statuses: PARTIAL_COMPLETE is OK; INCOMPLETE is not
    assert L.failed_recs([{"name": "a", "status": "PARTIAL_COMPLETE"}]) == []
    assert L.failed_recs([{"name": "a", "status": "INCOMPLETE"}]) == [("a", "INCOMPLETE")]


def lens_ladder_endpoint_geometry(S: int, *, L: int = 32_768) -> dict:
    """The ``geometry`` block the LENS standup writes for a gemma standup at ``S`` x the ladder window
    (``capctrl_serving_up``: ``ResolvedGeometry.as_dict()`` at the served S, no ``adjustments`` key, no engine
    key anywhere on the record — lens by era) — what the probe's readiness check reads on the lens engine
    too."""
    return sg.resolve_geometry(None, max_model_len=L, max_num_seqs=int(S), model_key="gemma4e4b").as_dict()


class _ProbeProcs:
    """A Popen double for ``_probe_rung``: at S = 18 the standup dies with the
    REAL KV refusal in its log (no endpoint); at S = 16 it comes up with the lens
    standup's ``endpoint.json`` — the ``geometry`` block at the served S
    (:func:`lens_ladder_endpoint_geometry`), which the probe now holds the rung to
    on the lens engine as well (a bare ``proxy_url`` record no producer writes is
    refused as "no geometry block")."""

    script = {18: "refuse", 16: "accept"}

    def __init__(self, argv, *a, **k):
        self.argv = list(argv)
        out = Path(self.argv[self.argv.index("--out") + 1])
        S = int(self.argv[self.argv.index("--max-num-seqs") + 1])
        what = type(self).script.get(S, "die")
        self.rc = None
        if what == "accept":
            (out / "endpoint.json").write_text(json.dumps({"proxy_url": "http://127.0.0.1:1/v1",
                                                           "geometry": lens_ladder_endpoint_geometry(S)}))
        elif what == "refuse":
            k["stdout"].write(REFUSAL_18)
            k["stdout"].flush()
            self.rc = 1
        else:
            k["stdout"].write("[standup] engine died\n")
            k["stdout"].flush()
            self.rc = 1

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        # lifecycle-faithful: the REAL standup unlinks its ``endpoint.json`` as it
        # tears down (capctrl_serving_up.py, after ``su.tear_down()``), so a launcher that reads the record after
        # ``_tear_down_standup`` finds nothing — the double deletes it exactly where the producer does
        out = Path(self.argv[self.argv.index("--out") + 1])
        (out / "endpoint.json").unlink(missing_ok=True)
        return self.rc if self.rc is not None else 0

    def terminate(self):
        self.rc = 0

    def kill(self):
        self.rc = 0


def test_probe_rung_walks_down_on_the_kv_refusal_and_writes_a_valid_certificate(L, tmp_path, monkeypatch):
    repos, tree = _ladder_tree(tmp_path)
    gen = tmp_path / "gen"
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(gen))
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L.subprocess, "Popen", _ProbeProcs)
    sel_path = _write_selection(tmp_path / "ladder_selection.json", SEL_IDS)
    ladder = L.ladder_launch_args(cap_override=C.format_cap_overrides(CAPS), item_selection=str(sel_path),
                                  out_suffix="ladder-1")
    cfgs = L.ladder_configurations(L.plan_configurations(cells=["none", DERIVED], modes=["off"], controls=[BCB, BFCL200],
                                                         model="gemma4e4b-sdfbase"), ladder)
    # the probe target is the launch's FITTED cell (the authoritative classifier, never cfg["steered"]);
    # a direct probe of the base cell of a launch that has a fitted cell writes nothing
    assert L.ladder_probe_cell(cfgs) == DERIVED and L.ladder_probe_cell(cfgs[:1]) == "none"
    assert L.ladder_probe_cell([dict(cfgs[1], steered=False)]) == DERIVED            # the kind, not the flag
    two = L.ladder_launch_bound(ladder, cfgs)
    assert two.probe_cell == DERIVED and two.cells == ("none", DERIVED)
    (cfg,) = L.ladder_configurations(L.plan_configurations(cells=["none"], modes=["off"], controls=[BCB, BFCL200]), ladder)
    denied = L._probe_rung(cfg, "f6cf462", object_digests={}, ladder=two, run_id="x")
    assert denied["status"] == "LADDER_PROBE_CELL_MISMATCH" and DERIVED in denied["error"]
    # a FORGED record — the two-cell launch with probe_cell="none" — is refused too (recomputed from its cells)
    forged = two._replace(probe_cell="none")
    denied = L._probe_rung(cfg, "f6cf462", object_digests={}, ladder=forged, run_id="x")
    assert denied["status"] == "LADDER_PROBE_CELL_MISMATCH" and "probe_cell 'none' !=" in denied["error"]
    with pytest.raises(ValueError, match="names probe cell 'none'"):
        L.ladder_launch_bound(forged, cfgs)
    with pytest.raises(RP.LadderError, match="names no probe cell"):
        L.ladder_provenance(L.row_of(cfg), tree, FULL_SHA, {}, ladder)
    ladder = L.ladder_launch_bound(ladder, [cfg])
    rec = L._probe_rung(cfg, "f6cf462", object_digests={}, ladder=ladder, run_id="f6cf462-cafebabe")
    assert rec["status"] == "OK", rec
    cert = rec["certificate"]
    assert [(w["S"], w["outcome"], w["kv_tokens"]) for w in cert["walked"]] == [(18, "refused", 528_000), (16, "accepted", None)]
    assert cert["accepted"] == RP.accepted_block(16) and rec["accepted"]["rung_source"] == "probe"
    assert cert["probe_run_id"] == "f6cf462-cafebabe" and cert["probed_cell"] == "none" and cert["suffix"] == "ladder-1"
    path = RP.certificate_path(str(gen), C.MODEL_SLUG, "ladder-1", "none")          # per-cell (S6)
    assert path.name == "rung_probe.none.json" and path.is_file() and RP.read_certificate(path)[0] == cert
    prov = L.ladder_provenance(L.row_of(cfg), tree, FULL_SHA, {}, ladder)
    assert RP.certificate_is_valid(cert, prov) is None and cert["cfg_sha256"] == L.sha256_file(tree / L.row_of(cfg).cfg)
    assert cert["ladder"] == RP.ladder_spec("gemma4e4b").block() and cert["reprobed_from"] is None
    assert [s["S"] for s in rec["walked"]] == [18, 16] and (path.parent / "probe_none_S18_standup.log").is_file()
    # the probe stood each rung up through the gate, at the ladder window
    assert all(s["geometry"]["ladder_window"] == 32_768 for s in rec["walked"])
    # an existing certificate is never silently re-probed; --ladder-reprobe supersedes it
    again = L._probe_rung(cfg, "f6cf462", object_digests={}, ladder=ladder, run_id="r2")
    assert again["status"] == "LADDER_CERTIFICATE_EXISTS"
    redo = L._probe_rung(cfg, "f6cf462", object_digests={}, ladder=ladder, run_id="r3", reprobe=True)
    assert redo["status"] == "OK" and Path(redo["superseded"]).is_file() and "superseded" in Path(redo["superseded"]).name
    # a standup that dies WITHOUT a KV marker is STANDUP_FAILED (no ladder)
    monkeypatch.setattr(_ProbeProcs, "script", {18: "die"})
    dead = L._probe_rung(cfg, "f6cf462", object_digests={}, ladder=ladder, run_id="r4", reprobe=True)
    assert dead["status"] == "STANDUP_FAILED" and "neither came up nor named a KV-coverage refusal" in dead["error"]
    # every rung refused with a cache too small for any: NO_LADDER_RUNG
    monkeypatch.setattr(_ProbeProcs, "script", {18: "refuse_small"})
    _ProbeProcs.__init__.__globals__["REFUSAL_SMALL"] = REFUSAL_18.replace("528000", "300000")
    orig_init = _ProbeProcs.__init__

    def _init(self, argv, *a, **k):
        if int(argv[argv.index("--max-num-seqs") + 1]) == 18:
            out = Path(argv[argv.index("--out") + 1])
            self.argv, self.rc = list(argv), 1
            k["stdout"].write(REFUSAL_18.replace("528000", "300000"))
            k["stdout"].flush()
            out.mkdir(exist_ok=True)
            return
        orig_init(self, argv, *a, **k)
    monkeypatch.setattr(_ProbeProcs, "__init__", _init)
    none = L._probe_rung(cfg, "f6cf462", object_digests={}, ladder=ladder, run_id="r5", reprobe=True)
    assert none["status"] == "NO_LADDER_RUNG" and "300000" in none["error"]
    # the OPERATOR certificate: no standup, rung_source operator, walked []
    op_ladder = L.ladder_launch_bound(L.ladder_launch_args(cap_override=C.format_cap_overrides(CAPS),
                                                           item_selection=str(sel_path), out_suffix="ladder-op",
                                                           ladder_rung="32768,14"), [cfg])
    oprec = L._operator_certificate(cfg, "f6cf462", object_digests={}, ladder=op_ladder, run_id="op")
    assert oprec["status"] == "OK" and oprec["certificate"]["accepted"] == RP.accepted_block(14, identity=18, operator=True)
    assert oprec["certificate"]["walked"] == [] and oprec["certificate"]["standup_s"] is None
    op_prov = L.ladder_provenance(L.row_of(cfg), tree, FULL_SHA, {}, op_ladder)
    assert RP.certificate_is_valid(oprec["certificate"], op_prov, operator_rung=(32_768, 14)) is None
    assert "operator certificate never admits" in RP.certificate_is_valid(oprec["certificate"], op_prov)
    # the admission reads that operator certificate ONLY for an operator launch at that rung; the wave
    # record (S6) is what carries `accepted` — written by the record writer over the final certificate
    lcfgs = L.ladder_configurations(L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"],
                                                                             controls=[BCB, BFCL200]), "ladder-op"), op_ladder)
    adm = L.ladder_admission_report(lcfgs, op_ladder, tree=str(tree), tree_sha=FULL_SHA, object_digests={})
    assert adm["certificate"] == "valid" and adm["admission"] == "absent" and adm["accepted"] is None
    ar = L._write_ladder_admission("f6cf462", C.MODEL_SLUG, "ladder-op", oprec["certificate"]["accepted"],
                                   {"none": oprec["certificate"]})
    assert ar["status"] == "OK", ar
    adm = L.ladder_admission_report(lcfgs, op_ladder, tree=str(tree), tree_sha=FULL_SHA, object_digests={})
    assert adm["admission"] == "valid" and adm["accepted"]["max_num_seqs"] == 14 and adm["all_complete"] is False
    plain = op_ladder._replace(operator_rung=None)
    adm2 = L.ladder_admission_report(lcfgs, plain, tree=str(tree), tree_sha=FULL_SHA, object_digests={})
    assert adm2["certificate"].startswith("invalid(") and "operator" in adm2["certificate"] and adm2["accepted"] is None
    # a preflight report without the admission refuses the ladder launch in main
    assert L.LADDER_ADMISSION_FIELD == "ladder_admission"
    # a VALID certificate copied into another suffix's path is refused there — before probe or scoring
    other = L.ladder_launch_bound(L.ladder_launch_args(cap_override=C.format_cap_overrides(CAPS),
                                                       item_selection=str(sel_path), out_suffix="ladder-2"), [cfg])
    good = _ladder_kw(L, tree, cfg, ladder, gen_root=str(gen))       # a fresh VALID ladder-1 certificate + record, filed
    src_path = RP.certificate_path(str(gen), C.MODEL_SLUG, "ladder-1", "none")
    dst_path = RP.certificate_path(str(gen), C.MODEL_SLUG, "ladder-2", "none")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, dst_path)
    shutil.copy(RP.admission_path(str(gen), C.MODEL_SLUG, "ladder-1"), RP.admission_path(str(gen), C.MODEL_SLUG, "ladder-2"))
    lcfgs2 = L.ladder_configurations(L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"],
                                                                              controls=[BCB, BFCL200]), "ladder-2"), other)
    adm3 = L.ladder_admission_report(lcfgs2, other, tree=str(tree), tree_sha=FULL_SHA, object_digests={})
    assert adm3["certificate"] == "invalid(suffix 'ladder-1' != this launch's 'ladder-2')" and adm3["accepted"] is None
    assert adm3["admission"].startswith("invalid(")
    # ... and the worker refuses the transplant too
    rec = L._run_configuration(dict(lcfgs2[0], ladder_rung=RP.accepted_block(16)), "f6cf462",
                               ladder={"launch": other.as_dict(), "certificates": {"none": json.loads(dst_path.read_text())},
                                       "admission": good["admission"]})
    assert rec["status"] == "LADDER_CERTIFICATE_INVALID" and "suffix" in rec["error"]


# --------------------------------------------------------------------------- #
# the external watcher: remote reads only, one attempt id per cell            #
# --------------------------------------------------------------------------- #


def _load_watcher():
    name = "tinycap_sandbox_watch_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/tinycap_sandbox_watch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _RemoteVolume:
    """A Volume double that REFUSES ``reload()`` (a container-only RPC from the
    grading host) and serves committed state through the remote ``listdir`` /
    ``read_file`` APIs."""

    def __init__(self, files):
        self.files, self.reloads = dict(files), 0

    def reload(self):
        self.reloads += 1
        raise RuntimeError("VolumeReload: reload() is only available inside a container that mounts the volume")

    def listdir(self, path):
        prefix = path.rstrip("/") + "/"
        return [SimpleNamespace(path=p) for p in self.files if p.startswith(prefix)]

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path].encode("utf-8")


def _pointer(W, run_id, cell, fcid, suffix="s7b"):
    out_dir = W.cell_out_dir("gemma4e4b", cell, "off", suffix)
    rel = out_dir[len("/vol/"):]
    doc = {"run_id": run_id, "cell": cell, "function_call_id": fcid, "name": f"n-{cell}", "model": "gemma4e4b",
           "mode": "off", "out_dir": out_dir, "controls": [BCB], "tree_sha": "a" * 40, "sandbox_record": "r",
           "sandbox_app_name": C.BCB_SANDBOX_APP_NAME, "started_at": 1.0}
    return f"{rel}/watch/attempt_{fcid}.json", json.dumps(doc)


def _observed(state):
    return SimpleNamespace(unavailable=state == "unavailable", pending=state == "pending",
                           terminal=state == "terminal", describe=lambda: state)


def test_the_watcher_reads_the_volume_remotely_and_sweeps_only_the_terminal_attempt(monkeypatch, tmp_path, capsys):
    W = _load_watcher()
    src = (REPO / "scripts/tinycap_sandbox_watch.py").read_text()
    assert not re.search(r"^\s*\w+\.reload\(\)", src, re.M)                  # no reload CALL anywhere
    files = dict([_pointer(W, "r1", "none", "fc-a"), _pointer(W, "r1", DERIVED, "fc-b")])
    vol = _RemoteVolume(files)
    monkeypatch.setattr(W, "_volume", lambda: vol)
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    states = {"fc-a": "terminal", "fc-b": "pending"}
    monkeypatch.setattr(W, "classify", lambda fcid: _observed(states[fcid]))
    swept = []
    monkeypatch.setattr(W, "sweep_attempt", lambda owner, **kw: (swept.append(owner) or
                                                                 {"verified_empty": True, "ids_terminated": [], "still_live": []}))
    rc = W.watch(run_id="r1", suffix="s7b", model="gemma4e4b", cells=["none", DERIVED], mode="off", poll_s=0,
                 max_hours=1, state_dir=str(tmp_path), once=True)
    out = capsys.readouterr().out
    # the reload-refusing volume never stopped the tick; input 0 (none) terminal -> swept; input 1 still running -> not
    # — and a --once tick with an attempt still pending is NOT done
    assert vol.reloads == 0 and rc == W.EXIT_UNVERIFIED
    assert [o.cell for o in swept] == ["none"] and swept[0].function_call_id == "fc-a"
    assert f"{DERIVED}: attempt fc-b PENDING" in out
    # both terminal and certified -> DONE; no pointer yet / an unobservable attempt -> UNVERIFIED
    states["fc-b"] = "terminal"
    assert W.watch(run_id="r1", suffix="s7b", model="gemma4e4b", cells=["none", DERIVED], mode="off", poll_s=0,
                   max_hours=1, state_dir=str(tmp_path), once=True) == W.EXIT_DONE
    monkeypatch.setattr(W, "_volume", lambda: _RemoteVolume(dict([_pointer(W, "r1", "none", "fc-a")])))
    assert W.watch(run_id="r1", suffix="s7b", model="gemma4e4b", cells=["none", DERIVED], mode="off", poll_s=0,
                   max_hours=1, state_dir=str(tmp_path), once=True) == W.EXIT_UNVERIFIED      # DERIVED: no pointer
    assert "no attempt pointer yet" in capsys.readouterr().out
    states["fc-a"] = "unavailable"
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    assert W.watch(run_id="r1", suffix="s7b", model="gemma4e4b", cells=["none"], mode="off", poll_s=0,
                   max_hours=1, state_dir=str(tmp_path), once=True) == W.EXIT_UNVERIFIED      # unobservable
    assert "observation unavailable" in capsys.readouterr().out
    # --sweep-now: the same remote reads (both pointers back in view)
    monkeypatch.setattr(W, "_volume", lambda: vol)
    swept.clear()
    assert W.sweep_now(run_id="r1", suffix="s7b", model="gemma4e4b", cells=["none", DERIVED], mode="off",
                       state_dir=str(tmp_path)) == W.EXIT_DONE
    assert [o.cell for o in swept] == ["none", DERIVED] and vol.reloads == 0


def test_the_watcher_never_acts_on_an_attempt_id_two_cells_share(monkeypatch, tmp_path, capsys):
    """One FunctionCall for a whole `.map` read input 0's state for every
    cell — with `none` finished first, the still-running steered cell read
    TERMINAL and its live sandboxes were swept. The launcher now spawns one call
    per configuration; the watcher refuses to classify or sweep a shared id."""
    W = _load_watcher()
    files = dict([_pointer(W, "r1", "none", "fc-map"), _pointer(W, "r1", DERIVED, "fc-map")])
    vol = _RemoteVolume(files)
    monkeypatch.setattr(W, "_volume", lambda: vol)
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    classified, swept = [], []
    monkeypatch.setattr(W, "classify", lambda fcid: (classified.append(fcid) or _observed("terminal")))
    monkeypatch.setattr(W, "sweep_attempt", lambda owner, **kw: pytest.fail(f"swept {owner}"))
    rc = W.watch(run_id="r1", suffix="s7b", model="gemma4e4b", cells=["none", DERIVED], mode="off", poll_s=0,
                 max_hours=1, state_dir=str(tmp_path), once=True)
    out = capsys.readouterr().out
    assert rc == W.EXIT_UNVERIFIED and not classified and not swept       # a shared id is never DONE
    assert "attempt fc-map is shared with" in out and "NOT swept" in out
    assert W.shared_attempt_ids({"none": {"function_call_id": "x"}, DERIVED: {"function_call_id": "x"}, "c": None}) == {
        "x": {"none", DERIVED}}
    assert W.shared_attempt_ids({"none": {"function_call_id": "x"}, DERIVED: {"function_call_id": "y"}}) == {}
    # the launcher's side of the fix: one FunctionCall per configuration
    src = LAUNCHER.read_text()
    assert "def fan_out(" in src and "fn.spawn(cfg, **kwargs)" in src and "run_configuration.map(" not in src


# --------------------------------------------------------------------------- #
# a resumed two-leg rung, and the worker's certificate                          #
# --------------------------------------------------------------------------- #


def _ladder_kw(L, tree, cfg, ladder, *, S=16, gen_root=None, write=True, tamper=None):
    """The worker's ``ladder`` kwarg (bcb-bfcl-200-widen S6: ``{launch, certificates: {cell: cert},
    admission}``) — a REAL probe-style certificate for the launch's probe cell built from its
    provenance on ``tree`` plus the wave record over it, both (by default) filed where the worker
    re-reads them; ``tamper(cert)`` edits the carried certificate copy."""
    row = L.row_of(cfg)
    cell = ladder.probe_cell
    prov = L.ladder_provenance(row, tree, FULL_SHA, {}, ladder, probed_cell=cell)
    # a LEGAL walk to S: the identity rung refused by a cache that holds S x L (+ a little), so next_rung -> S
    tail = REFUSAL_18.replace("528000", str(S * RP.LADDER_WINDOW + 1000))
    cert = RP.build_certificate(accepted=RP.accepted_block(S, identity=18),
                                walked=[RP.walked_entry(18, RP.WALK_REFUSED, tail), RP.walked_entry(S, RP.WALK_ACCEPTED)]
                                if S != 18 else [RP.walked_entry(18, RP.WALK_ACCEPTED)],
                                provenance=prov, probe_run_id="r", standup_s=1.0)
    assert RP.certificate_is_valid(cert, prov) is None
    admission = RP.build_admission(suffix=ladder.out_suffix, common=cert["accepted"], certificates={cell: cert})
    if write:
        root = gen_root or C.VOL_GEN_ROOT
        RP.write_certificate(RP.certificate_path(root, row.slug, ladder.out_suffix, cell), cert)
        RP.write_admission(RP.admission_path(root, row.slug, ladder.out_suffix), admission)
    carried = copy.deepcopy(cert)
    if tamper is not None:
        tamper(carried)
    return {"launch": ladder.as_dict(), "certificates": {cell: carried}, "admission": admission}


def test_the_scoring_worker_admits_a_rung_only_against_its_certificate(L, tmp_path, monkeypatch):
    """`run_configuration` is a separately callable entry point
    — a bare, edited or cached `cfg["ladder_rung"]` must never generate. The
    worker re-validates the carried certificate on ITS tree and requires the
    rung == accepted, the tuple == the launch's and the filed certificate ==
    the carried one, all BEFORE any directory or standup."""
    repos, tree = _ladder_tree(tmp_path)
    gen = tmp_path / "gen"
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(gen))
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: pytest.fail("a standup was started"))
    sel_path = _write_selection(tmp_path / "sel.json", {BFCL_TASK: SEL_F})
    ladder = L.ladder_launch_args(cap_override=f"{BFCL_TASK}=16384", item_selection=str(sel_path), out_suffix="w")
    (cfg,) = L.ladder_configurations(L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"],
                                                                              controls=[BFCL200]), "w"), ladder)
    ladder = L.ladder_launch_bound(ladder, [cfg])
    cfg16 = dict(cfg, ladder_rung=RP.accepted_block(16))
    good = _ladder_kw(L, tree, cfg, ladder, gen_root=str(gen))
    # a bare rung (no certificate carried): refused, no out dir made
    rec = L._run_configuration(cfg16, "f6cf462")
    assert rec["status"] == "LADDER_UNCERTIFIED" and not Path(cfg16["out_dir"]).exists()
    # the certificate admitted S = 16; a cached configuration at S = 12 is refused by name
    rec = L._run_configuration(dict(cfg, ladder_rung=RP.accepted_block(12)), "f6cf462", ladder=good)
    assert rec["status"] == "LADDER_RUNG_MISMATCH" and "max_num_seqs': 12" in rec["error"]
    # an edited certificate (another selection digest) is invalid against this launch
    # write=False: the edit lives in the CARRIED copy only. A refile here stamps a fresh second-resolution `at` into the
    # filed certificate and wave record, and every later `ladder=good` step then reads LADDER_CERTIFICATE_MISMATCH whenever
    # the two writes straddle a second boundary (a loaded ten-shard container, 2026-09-17) — the filed records stay `good`'s
    bad = _ladder_kw(L, tree, cfg, ladder, gen_root=str(gen), write=False,
                     tamper=lambda c: c.__setitem__("selection_sha256", "0" * 64))
    rec = L._run_configuration(cfg16, "f6cf462", ladder=bad)
    assert rec["status"] == "LADDER_CERTIFICATE_INVALID" and "selection_sha256" in rec["error"]
    # a carried certificate that is not the one FILED for this suffix
    other = _ladder_kw(L, tree, cfg, ladder, S=14, gen_root=str(gen), write=False)
    rec = L._run_configuration(dict(cfg, ladder_rung=RP.accepted_block(14)), "f6cf462", ladder=other)
    assert rec["status"] == "LADDER_CERTIFICATE_MISMATCH" and "not the one filed" in rec["error"]
    # a configuration whose tuple is not the certified launch's
    rec = L._run_configuration(dict(cfg16, item_selection={BFCL_TASK: list(PIN_F["ids"][:3])}), "f6cf462", ladder=good)
    assert rec["status"] == "LADDER_TUPLE_MISMATCH"
    # a malformed carrier (the earlier single-certificate shape included)
    rec = L._run_configuration(cfg16, "f6cf462", ladder={"certificate": {}, "launch": {}})
    assert rec["status"] in ("LADDER_UNCERTIFIED", "LADDER_CERTIFICATE_INVALID")
    rec = L._run_configuration(cfg16, "f6cf462", ladder={"certificate": good["certificates"]["none"], "launch": good["launch"]})
    assert rec["status"] == "LADDER_UNCERTIFIED"
    # the carried WAVE RECORD must be the filed one, and its common rung the configuration's
    stale = {**good, "admission": {**good["admission"], "at": "2026-01-01T00:00:00+00:00"}}
    rec = L._run_configuration(cfg16, "f6cf462", ladder=stale)
    assert rec["status"] == "LADDER_CERTIFICATE_MISMATCH" and "wave record" in rec["error"]
    # the admitted worker proceeds past the gate (and stops at this test's forbidden standup)
    with pytest.raises(BaseException, match="a standup was started"):
        L._run_configuration(cfg16, "f6cf462", ladder=good)
    # the converse of `stale`, and the straddle the flake was made of, pinned deterministically: a wave record REFILED
    # with a later stamp makes the still-carried `good` the stale one — refused by name (the carried record must be the
    # filed one, byte for byte); the contract is right, the old test merely tripped it by accident
    refiled = {**good["admission"], "at": "2099-01-01T00:00:00+00:00"}
    RP.write_admission(RP.admission_path(str(gen), L.row_of(cfg).slug, ladder.out_suffix), refiled)
    rec = L._run_configuration(cfg16, "f6cf462", ladder=good)
    assert rec["status"] == "LADDER_CERTIFICATE_MISMATCH" and "wave record" in rec["error"]
    # main hands every scoring container the certificate it validated
    src = LAUNCHER.read_text()
    assert "ladder=ladder_kw" in src and "ladder_worker_admission(cfg, row=row" in src
    assert 'return recs, scored, {"launch": ladder.as_dict(), "certificates": cert_docs, "admission": admission_doc}' in src


def test_an_interrupted_two_leg_rung_resumes_its_unfinished_leg_through_the_real_driver_validation(
        L, tmp_path, monkeypatch, capsys):
    """BCB complete, BFCL not — the relaunch must hand the
    driver the FULL two-task plan (the hashed tuple IS the store's identity),
    which the REAL driver argument validation (`served_capability_run` parser +
    `_ladder_args`) accepts, and skip the finished leg at dispatch (the shared
    validator) instead of trimming the maps and dying after a standup."""
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as srun

    repos, tree = _ladder_tree(tmp_path)
    gen = tmp_path / "gen"
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(gen))
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "0")
    monkeypatch.setattr(L, "client_lock_sha256", lambda tree: None)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L.subprocess, "Popen", _LadderProcs)
    sel_path = _write_selection(tmp_path / "sel.json", SEL_IDS)
    ladder = L.ladder_launch_args(cap_override=C.format_cap_overrides(CAPS), item_selection=str(sel_path),
                                  out_suffix="resume")
    (cfg,) = L.ladder_configurations(L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"],
                                                                              controls=[BCB, BFCL200]), "resume"), ladder)
    ladder = L.ladder_launch_bound(ladder, [cfg])
    cfg = dict(cfg, ladder_rung=RP.accepted_block(16))
    shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)
    kw = _ladder_kw(L, tree, cfg, ladder, gen_root=str(gen))
    geom, _ = L.served_geometry(cfg, 1, str(tree))
    exp = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=1, tree=str(tree),
                              tree_sha=FULL_SHA, reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
    # the interrupted first attempt: BCB validly complete, BFCL never filed
    store = _store(Path(cfg["out_dir"]), exp, tasks=[BCB, BFCL200], extra_identity=exp.optional_identity())
    _ladder_finish(store, BCB_TASK, C.ladder_log_ids(BCB_TASK, SEL_B))
    seen = {}

    def _driver(argv, **_kw):
        seen["argv"] = list(argv)
        # the REAL driver argument validation: the parser, then _ladder_args over the run's row tasks
        monkeypatch.setattr(srun, "cmd_generate", lambda args: seen.__setitem__("args", args))
        srun.main(list(argv[2:]))
        args = seen["args"]
        by_task = srun._resolve_thinking_by_task(srun.tinycap_atoms(srun.parse_controls(args.controls)), args.thinking)
        caps, selection = srun._ladder_args(args, by_task)
        assert caps == CAPS and {t: b["ids"] for t, b in selection.items()} == SEL_IDS
        # dispatch: the finished leg is skipped by the shared validator, the other leg is generated
        skipped = [t for t in by_task if RP.ladder_task_partial_reason(
            store, t, log_ids=C.ladder_log_ids(t, selection[t]["ids"]), cap=caps[t], n_expected_full=200) is None]
        assert skipped == [BCB_TASK]
        _ladder_finish(store, BFCL_TASK, C.ladder_log_ids(BFCL_TASK, SEL_F))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(L.subprocess, "run", _driver)
    try:
        rec = L._run_configuration(cfg, "f6cf462", ladder=kw, run_id="f6cf462-resume")
    finally:
        shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)
    assert rec["controls_remaining"] == [BFCL200] and rec["controls_driven"] == [BCB, BFCL200]
    assert rec["status"] == "PARTIAL_COMPLETE" and rec["controls_still_remaining"] == []
    # the per-attempt COMPLETION record this (solo, sandboxed) resumption wrote at the end of its drive (stage 07's receipt, round 4): it drove
    # the FULL list, had the BFCL leg left to do, and finished THAT leg in the first attempt's store — never the BCB leg the first attempt did
    fcid = rec["sandbox_owner"]["function_call_id"]
    assert rec["completion_record"] == str(L.completion_record_path(cfg["out_dir"], fcid)) and Path(rec["completion_record"]).is_file()
    cdoc = json.loads(Path(rec["completion_record"]).read_text())
    assert set(cdoc) == set(L.COMPLETION_FIELDS) and cdoc["run_id"] == "f6cf462-resume" and cdoc["cell"] == "none" and cdoc["function_call_id"] == fcid
    assert cdoc["controls"] == [BCB, BFCL200] and cdoc["remaining"] == [BFCL200] and cdoc["completed"] == {BFCL200: store.name}
    assert cdoc["status"] == "PARTIAL_COMPLETE" and cdoc["sandboxed"] is True and cdoc["tree_sha"] == FULL_SHA and cdoc["out_dir"] == cfg["out_dir"]
    argv = seen["argv"]
    assert argv[argv.index("--controls") + 1] == f"{BCB},{BFCL200}"
    assert C.parse_cap_overrides(argv[argv.index(C.FLAG_CAP_OVERRIDE) + 1]) == CAPS
    assert {t: b["ids"] for t, b in C.parse_item_selection_json(argv[argv.index(C.FLAG_ITEM_SELECTION_JSON) + 1]).items()} == SEL_IDS
    # ONE store, both legs PARTIAL, the second attempt's rows on the first's store
    assert [p.name for p in (Path(cfg["out_dir"]) / "cells").iterdir()] == [store.name]
    for t in (BCB_TASK, BFCL_TASK):
        assert completion.partial_marker_path(store, t).is_file()
    # the trimmed argv the launcher used to build is exactly what the real driver refuses
    trimmed = list(argv)
    trimmed[trimmed.index("--controls") + 1] = BFCL200
    monkeypatch.setattr(srun, "cmd_generate", lambda args: seen.__setitem__("args", args))
    srun.main(trimmed[2:])
    by_task = srun._resolve_thinking_by_task(srun.tinycap_atoms(srun.parse_controls(seen["args"].controls)),
                                             seen["args"].thinking)
    with pytest.raises(SystemExit, match="fired legs alone"):
        srun._ladder_args(seen["args"], by_task)


def test_a_transient_listing_failure_is_recorded_backed_off_and_retried_never_fatal(monkeypatch, tmp_path, capsys):
    """A `Sandbox.list()` RPC that raises used to propagate out of
    `watch()` and kill the external backstop. Now the sweep writes an
    owner-bound local certificate with `verified_empty: false` + the error,
    the cell stays pending, the backoff applies, and the next tick retries —
    a listing that fails once then succeeds ends DONE with a clean certificate."""
    W = _load_watcher()
    files = dict([_pointer(W, "r1", "none", "fc-a")])
    monkeypatch.setattr(W, "_volume", lambda: _RemoteVolume(files))
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    monkeypatch.setattr(W, "classify", lambda fcid: _observed("terminal"))
    calls = {"n": 0}

    def _lister(owner, app_id):
        def _list():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("grpc UNAVAILABLE: Sandbox.list")
            return []
        return _list
    monkeypatch.setattr(W, "owned_lister", _lister)
    slept = []
    monkeypatch.setattr(W.time, "sleep", lambda s: slept.append(s))
    rc = W.watch(run_id="r1", suffix="s7b", model="gemma4e4b", cells=["none"], mode="off", poll_s=0,
                 max_hours=1, state_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert rc == W.EXIT_DONE and calls["n"] >= 2                # the first listing raised; the retry's sweep listed (+ re-listed to verify)
    assert "none: sweep FAILED (RuntimeError: grpc UNAVAILABLE: Sandbox.list); retry in 60s" in out and 60.0 in slept
    cert = json.loads(W.local_certificate_path("r1", "fc-a", str(tmp_path)).read_text())
    # the final certificate is the clean retry's; the failed one was written first (owner-bound, error named)
    assert cert["verified_empty"] is True and cert["owner"] == {"run_id": "r1", "cell": "none", "function_call_id": "fc-a"}
    assert "sweep FAILED: RuntimeError: grpc UNAVAILABLE" in out and "DONE: every attempt terminal and certified" in out
    # --sweep-now on a failing listing: the failed certificate is on disk, exit UNVERIFIED, no exception
    calls["n"] = -10                                            # every call raises for a while
    monkeypatch.setattr(W, "owned_lister", lambda owner, app_id: (lambda: (_ for _ in ()).throw(RuntimeError("down"))))
    rc = W.sweep_now(run_id="r1", suffix="s7b", model="gemma4e4b", cells=["none"], mode="off", state_dir=str(tmp_path))
    cert = json.loads(W.local_certificate_path("r1", "fc-a", str(tmp_path)).read_text())
    assert rc == W.EXIT_UNVERIFIED and cert["verified_empty"] is False and cert["error"] == "RuntimeError: down"
    assert cert["owner"]["function_call_id"] == "fc-a" and cert["certificate"] == "sweep"
    # a sandbox-app lookup failure backs off too (once: UNVERIFIED, never an exception)
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("lookup down")))
    assert W.watch(run_id="r1", suffix="s7b", model="gemma4e4b", cells=["none"], mode="off", poll_s=0, max_hours=1,
                   state_dir=str(tmp_path), once=True) == W.EXIT_UNVERIFIED


def test_fan_out_harvests_every_sibling_and_turns_a_failed_get_into_a_rec(L, capsys):
    """The first `.get()` that raises used to abort the harvest —
    the siblings' RECs were lost and the exit came from a raw exception. Now
    every call is collected; a failure is that configuration's explicit REC,
    `failed_recs` names it and `main`'s exit is the RECs' verdict."""
    cfgs = L.plan_configurations(cells=["none", "prompted__safety_refusal"], modes=["off"], controls=["gsm8k"])

    class _Failing(_Call):
        def get(self, timeout=None):
            raise RuntimeError("container OOM (decoded by modal)")
    spawned = []

    def _spawn(cfg, **kw):
        fc = _Failing(None) if cfg["cell"] == "none" else _Call({"name": cfg["name"], "status": "OK"})
        spawned.append(fc)
        return fc
    recs = L.fan_out(SimpleNamespace(spawn=_spawn), cfgs, {"sha": "x"})
    assert [r["status"] for r in recs] == ["FUNCTIONCALL_FAILED", "OK"]
    assert recs[0]["error"] == "RuntimeError: container OOM (decoded by modal)" and recs[0]["cell"] == "none"
    assert recs[0]["function_call_id"] == spawned[0].object_id
    assert L.failed_recs(recs) == [(cfgs[0]["name"], "FUNCTIONCALL_FAILED")]
    msg = L.rec_failure_message(recs)
    assert "1/2 container(s) did NOT finish" in msg and "container OOM" in msg
    # a remote SystemExit (decoded raw) is a REC too; a local Ctrl-C is not swallowed

    class _Exiting(_Call):
        def get(self, timeout=None):
            raise SystemExit("[served-cap] refused")
    recs = L.fan_out(SimpleNamespace(spawn=lambda cfg, **kw: _Exiting(None)), cfgs[:1], {})
    assert recs[0]["status"] == "FUNCTIONCALL_FAILED" and "SystemExit" in recs[0]["error"]

    class _Interrupted(_Call):
        def get(self, timeout=None):
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        L.fan_out(SimpleNamespace(spawn=lambda cfg, **kw: _Interrupted(None)), cfgs[:1], {})


def test_the_ladder_dry_run_prints_the_operators_meter_step(L, tmp_path, monkeypatch, capsys):
    """The meter (operator-run BEFORE launch) is named at
    the launch boundary — the exact pre-probe invocation over the ACTUAL
    selection with the selected-item counts it forecasts from, and the
    certified re-forecast after the probe."""
    import importlib.util as _ilu
    w = _ladder_world(L, tmp_path, monkeypatch, ids_by_task={BFCL_TASK: SEL_F, BCB_TASK: SEL_B}, controls=f"{BCB},{BFCL200}")
    L.main(**w["main"], out_suffix="ladder-m", dry_run=True)
    out = capsys.readouterr().out
    sel = str(w["sel_path"])
    cert = RP.certificate_path(C.VOL_GEN_ROOT, C.MODEL_SLUG, "ladder-m", "none")
    # bcb-bfcl-200-widen S6 / S10: the widen meter's vocabulary — a reservation keyed (wave, suffix, unit) with
    # the stage KIND separate, the MODEL named (rung_probe.ladder_spec), one probe reservation per probe cell
    pre = (f"meter (operator, BEFORE launch): python scripts/bcb_bfcl_200_meter.py --stage-kind ladder-probe --wave <wave> "
           f"--suffix ladder-m --unit probe:none --model gemma4e4b --selection {sel} --fitted-cell none   # pre-probe "
           f"reservation over the ACTUAL selection: n_selected "
           f'{{"{BFCL_TASK}": 2, "{BCB_TASK}": 3}} = 5 item(s), worst-case S 12 at 32768, cap 16384')   # sorted by task
    assert pre in out
    post = (f"meter (operator, AFTER the probes): python scripts/bcb_bfcl_200_meter.py --stage-kind ladder-score --wave <wave> "
            f"--suffix ladder-m --unit score --model gemma4e4b --selection {sel} --certificate {cert}")
    assert post in out
    assert "probe cells: none (the launch's fitted cells" in out
    # the flags the dry run names are the meter's own (slice C's CLI, read by argparse)
    spec = _ilu.spec_from_file_location("bcb_bfcl_200_meter_under_test", REPO / "scripts/bcb_bfcl_200_meter.py")
    meter = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = meter
    spec.loader.exec_module(meter)
    line = next(ln for ln in out.splitlines() if "meter (operator, BEFORE launch)" in ln)
    argv = line.split("python scripts/bcb_bfcl_200_meter.py ", 1)[1].split("   #", 1)[0].split()
    args = meter.build_parser().parse_args(argv)
    assert (args.stage_kind, args.selection, args.model, args.unit, args.fitted_cell) == ("ladder-probe", sel, "gemma4e4b", "probe:none", ["none"])
    line = next(ln for ln in out.splitlines() if "meter (operator, AFTER the probes)" in ln)
    argv = line.split("python scripts/bcb_bfcl_200_meter.py ", 1)[1].split("   #", 1)[0].split()
    args = meter.build_parser().parse_args(argv)
    assert args.stage_kind == "ladder-score" and args.certificate == [str(cert)] and args.unit == "score"
    assert not w["probe"] and not w["spawned"]


class _CanaryModal:
    """A `modal` double for the four --canary modes: App.lookup, Sandbox.create /
    from_id / list (server-side tag filtering: every filter tag must match),
    Image.from_registry; sandboxes poll None while live."""

    class _SB:
        def __init__(self, sid, tags):
            self.object_id, self._tags, self.terminated = sid, dict(tags), False

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self, **kw):
            self.terminated = True

        def get_tags(self):
            return dict(self._tags)

    def __init__(self):
        self.registry, self.created = {}, []
        m = self

        class Sandbox:
            @staticmethod
            def create(*argv, app=None, tags=None, **kw):
                sb = _CanaryModal._SB(f"sb-{len(m.registry) + 1}", tags or {})
                m.registry[sb.object_id] = sb
                m.created.append({"argv": argv, "tags": dict(tags or {}), **{k: v for k, v in kw.items() if k != "image"}})
                return sb

            @staticmethod
            def from_id(sid):
                return m.registry[sid]

            @staticmethod
            def list(app_id=None, tags=None):
                return [sb for sb in m.registry.values() if not sb.terminated
                        and all(sb._tags.get(k) == v for k, v in (tags or {}).items())]
        self.Sandbox = Sandbox
        self.App = SimpleNamespace(lookup=lambda name, create_if_missing=False: SimpleNamespace(app_id=f"ap-{name}"))
        self.Image = SimpleNamespace(from_registry=lambda ref: ("image", ref))
        self.__version__ = "double"


def test_the_four_canary_modes_create_check_sweep_and_terminate_an_owner_tagged_sandbox(monkeypatch, tmp_path, capsys):
    """S7b / S7d's sibling-survival instrument, end to end against a modal
    double: create (the attempt's EXACT owner tags, the pinned image, held
    open), check (listed by its own owner filter, NOT by a sibling's), sweep
    (the very same `sandbox_scope.sweep_owned` the container and the watcher
    run terminates it), terminate; unknown attempt / dead canary refuse."""
    W = _load_watcher()
    fake = _CanaryModal()
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setattr(scope.time, "sleep", lambda s: None)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    rel_a, doc_a = _pointer(W, "r1", "none", "fc-a")
    rel_b, doc_b = _pointer(W, "r1", DERIVED, "fc-b")
    pa, pb = tmp_path / "attempt_fc-a.json", tmp_path / "attempt_fc-b.json"
    pa.write_text(doc_a)
    pb.write_text(doc_b)
    state = str(tmp_path / "state")
    owner = scope.Owner("r1", "none", "fc-a")
    # check / sweep / terminate before create: no canary recorded
    assert W.canary("check", owner_from=str(pa), state_dir=state) == W.EXIT_UNVERIFIED
    assert W.canary("create", owner_from=str(pa), state_dir=state) == W.EXIT_DONE
    created = fake.created[0]
    assert created["argv"] == ("tail", "-f", "/dev/null") and created["tags"] == scope.owner_tags(
        owner, W.CANARY_TASK, {"__sample_id__": W.CANARY_TASK})
    assert (created["cpu"], created["memory"], created["timeout"], created["block_network"]) == (
        W.CANARY_CPU, W.CANARY_MEMORY_MIB, W.CANARY_TIMEOUT_S, True)
    doc = json.loads(W.canary_state_path("r1", state).read_text())
    assert doc["fc-a"]["sandbox_id"] == "sb-1" and doc["fc-a"]["owner"] == owner.as_dict()
    # check: live, listed by its own owner filter, not by the sibling's
    assert W.canary("check", owner_from=str(pa), state_dir=state, sibling_from=str(pb)) == W.EXIT_DONE
    rep = json.loads(capsys.readouterr().out.split("[canary] check ")[-1].splitlines()[0])
    assert rep["live"] is True and rep["listed_by_own_owner_filter"] is True and rep["listed_by_sibling_owner_filter"] is False
    # a sibling's sweep (the SAME sweep_owned, by ITS owner) leaves the canary alone
    W.sweep_attempt(scope.Owner("r1", DERIVED, "fc-b"), app_id="ap-x", state_dir=state, by="canary:sibling")
    assert not fake.registry["sb-1"].terminated
    # sweep by the canary's own owner terminates it, verified
    assert W.canary("sweep", owner_from=str(pa), state_dir=state) == W.EXIT_DONE
    assert fake.registry["sb-1"].terminated
    cert = json.loads(W.local_certificate_path("r1", "fc-a", state).read_text())
    assert cert["verified_empty"] is True and cert["ids_terminated"] == ["sb-1"] and cert["by"] == "canary:sweep"
    # a dead canary fails check; terminate records the time
    assert W.canary("check", owner_from=str(pa), state_dir=state) == W.EXIT_UNVERIFIED
    assert W.canary("terminate", owner_from=str(pa), state_dir=state) == W.EXIT_DONE
    doc = json.loads(W.canary_state_path("r1", state).read_text())
    assert doc["fc-a"]["terminated_at"] is not None
    with pytest.raises(SystemExit, match="unknown canary action"):
        W.canary("dance", owner_from=str(pa), state_dir=state)
    # the CLI wiring: --canary needs --owner-from
    with pytest.raises(SystemExit, match="--owner-from"):
        W.main(["--canary", "check"])


def test_a_forged_launch_record_cannot_self_certify_the_derived_cell(L, tmp_path, monkeypatch):
    """A direct Modal caller sets `launch.probe_cell="none"`,
    probes `none`, files the self-consistent certificate and admits the derived
    cell with it. Every door recomputes the probe cell from the launch's cell
    list through the authoritative classifier: the probe refuses, the preflight
    refuses the serialized mismatch, and the worker refuses the derived cell
    (and `none`) BEFORE any directory or standup."""
    repos, tree = _ladder_tree(tmp_path)
    for rel in (C.model_row("gemma4e4b-sdfbase").cfg,):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text((REPO / rel).read_text())
    gen = tmp_path / "gen"
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(gen))
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: pytest.fail("a standup was started"))
    sel_path = _write_selection(tmp_path / "sel.json", {BFCL_TASK: SEL_F})
    launch = L.ladder_launch_args(cap_override=f"{BFCL_TASK}=16384", item_selection=str(sel_path), out_suffix="forge")
    cfgs = L.ladder_configurations(L.isolate_out_dirs(L.plan_configurations(
        cells=["none", DERIVED], modes=["off"], controls=[BFCL200], model="gemma4e4b-sdfbase"), "forge"), launch)
    bound = L.ladder_launch_bound(launch, cfgs)
    assert bound.probe_cell == DERIVED and bound.cells == ("none", DERIVED)
    forged = bound._replace(probe_cell="none")
    # (1) the preflight re-derives the binding from its re-plan and refuses the serialized mismatch
    with pytest.raises(ValueError, match="probe cell 'none'"):
        L.ladder_launch_bound(forged, cfgs)
    # (3) the probe refuses the forged record for `none`; the derived cell's probe needs the derived config
    base_cfg, derived_cfg = cfgs
    assert L._probe_rung(base_cfg, "f6cf462", object_digests={}, ladder=forged, run_id="x")["status"] == "LADDER_PROBE_CELL_MISMATCH"
    assert L._probe_rung(base_cfg, "f6cf462", object_digests={}, ladder=bound, run_id="x")["status"] == "LADDER_PROBE_CELL_MISMATCH"
    # a certificate FILED anyway (a hand-written one, self-consistent with the forged record)
    row = L.row_of(base_cfg)
    prov = RP.Provenance(tree_sha=FULL_SHA, model_slug=row.slug, hf=row.hf, rev=row.rev, cfg_path=row.cfg,
                         cfg_sha256=L.sha256_file(tree / row.cfg), object_digests={},
                         selection=forged.selection, cap_overrides_by_task=forged.cap_overrides_by_task,
                         batched_token_budget=589_824, budget_source="measured", suffix="forge", probed_cell="none")
    cert = RP.build_certificate(accepted=RP.accepted_block(16, identity=18),
                                walked=[RP.walked_entry(18, RP.WALK_REFUSED, REFUSAL_18), RP.walked_entry(16, RP.WALK_ACCEPTED)],
                                provenance=prov, probe_run_id="forged", standup_s=1.0)
    assert RP.certificate_is_valid(cert, prov) is None                       # self-consistent with its own claim
    RP.write_certificate(RP.certificate_path(str(gen), row.slug, "forge", "none"), cert)
    record = RP.build_admission(suffix="forge", common=cert["accepted"], certificates={"none": cert})
    RP.write_admission(RP.admission_path(str(gen), row.slug, "forge"), record)
    carrier = {"launch": forged.as_dict(), "certificates": {"none": cert}, "admission": record}
    # (2) the worker: the derived cell is refused (a fitted scoring cell must BE the probed cell) ...
    rec = L._run_configuration(dict(derived_cfg, ladder_rung=RP.accepted_block(16)), "f6cf462", ladder=carrier,
                               object_digests={DERIVED: {"artifact_sha256": "ab" * 32}})
    assert rec["status"] == "LADDER_PROBE_CELL_MISMATCH" and "probe_cell 'none' !=" in rec["error"]
    assert not Path(derived_cfg["out_dir"]).exists()
    # ... and so is `none` under the forged record (its fitted sibling was not probed)
    rec = L._run_configuration(dict(base_cfg, ladder_rung=RP.accepted_block(16)), "f6cf462", ladder=carrier)
    assert rec["status"] == "LADDER_PROBE_CELL_MISMATCH" and not Path(base_cfg["out_dir"]).exists()
    # the honest record with the forged certificate: refused on the probed cell too
    rec = L._run_configuration(dict(base_cfg, ladder_rung=RP.accepted_block(16)), "f6cf462",
                               ladder={"launch": bound.as_dict(), "certificates": {"none": cert}, "admission": record})
    assert rec["status"] == "LADDER_PROBE_CELL_MISMATCH" and "probed_cell 'none' != the launch's fitted cell" in rec["error"]
    # the pure classifiers: every fitted cell is a probe cell; an object-free launch probes its first
    assert L.ladder_probe_cells_of(["none", DERIVED]) == (DERIVED,) and L.ladder_probe_cells_of(["none", "prompted__safety_refusal"]) == ("none",)
    assert L.fitted_cells_of(["none", DERIVED, "prompted__safety_refusal"]) == (DERIVED,)
    # the pure classifier reads the KIND, never a caller's flag
    assert L.ladder_probe_cell_of(["none", DERIVED]) == DERIVED == L.ladder_probe_cell_of([DERIVED, "none"])
    assert L.ladder_probe_cell_of(["none", "prompted__safety_refusal"]) == "none"
    assert L.ladder_probe_cell_reason(bound, DERIVED, scoring_cell="none") is None
    assert L.ladder_probe_cell_reason(bound, DERIVED, scoring_cell=DERIVED) is None
    assert "not one of the launch's cells" in L.ladder_probe_cell_reason(bound, DERIVED, scoring_cell="prompted__safety_refusal")
    assert "names no cells" in L.ladder_probe_cell_reason(launch, DERIVED)


def test_a_two_fitted_cell_wave_certifies_once_and_scores_every_cell_against_the_whole_waves_record(L, tmp_path, monkeypatch):
    """Code review: gemma-it has TWO fitted
    cells. Certification is whole-wave (`--cells` = all, every fitted cell probed, its certificate
    bound to BOTH fitted objects' digests, the wave record over both); scoring is per cell with
    `--cells X` NARROWED and `--ladder-wave-cells` = the whole wave. Before: the second fitted
    cell's scoring container was refused `LADDER_PROBE_CELL_MISMATCH` ("must be the probed cell"
    — the FIRST one), and a narrowed launch shrank the provenance to one digest, so no
    certificate validated and the record's cell set never matched. Now every wave cell — the
    first fitted, the second fitted, the base, the prompted — is admitted on the whole wave's
    certificates and record, re-validated on the worker's own tree."""
    repos, tree = _ladder_tree(tmp_path)
    (tree / C.model_row("gemma4e4b-f22it").cfg).parent.mkdir(parents=True, exist_ok=True)
    (tree / C.model_row("gemma4e4b-f22it").cfg).write_text((REPO / C.model_row("gemma4e4b-f22it").cfg).read_text())
    gen = tmp_path / "gen"
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(gen))
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: pytest.fail("a standup was started"))
    A, B = "constv2-mix-full__lora_r4_ce", "constv2-mix-full__psr_ce_all"
    WAVE = ["none", "prompted__safety_refusal", A, B]
    digests = {A: {"artifact_sha256": "aa" * 32}, B: {"artifact_sha256": "bb" * 32}}
    sel_path = _write_selection(tmp_path / "sel.json", {BFCL_TASK: SEL_F})
    launch = L.ladder_launch_args(cap_override=f"{BFCL_TASK}=16384", item_selection=str(sel_path), out_suffix="wave2",
                                  model="gemma4e4b-f22it")
    # --- step 1: certify over the WHOLE wave --------------------------------------------------------
    all_cfgs = L.ladder_configurations(L.isolate_out_dirs(L.plan_configurations(
        cells=WAVE, modes=["off"], controls=[BFCL200], model="gemma4e4b-f22it"), "wave2"), launch)
    bound = L.ladder_launch_bound(launch, all_cfgs, wave_cells=WAVE)
    assert bound.cells == tuple(WAVE) and bound.probe_cells == (A, B) and bound.probe_cell == A
    row = L.row_of(all_cfgs[0])
    certs = {}
    for cell in (A, B):
        prov = L.ladder_provenance(row, tree, FULL_SHA, digests, bound, probed_cell=cell)
        assert set(prov.object_digests) == {A, B}                                # BOTH fitted objects, on every certificate
        tail = REFUSAL_18.replace("528000", str(16 * RP.LADDER_WINDOW + 1000))
        cert = RP.build_certificate(accepted=RP.accepted_block(16, identity=18),
                                    walked=[RP.walked_entry(18, RP.WALK_REFUSED, tail), RP.walked_entry(16, RP.WALK_ACCEPTED)],
                                    provenance=prov, probe_run_id="r", standup_s=1.0)
        assert RP.certificate_is_valid(cert, prov) is None
        RP.write_certificate(RP.certificate_path(str(gen), row.slug, "wave2", cell), cert)
        certs[cell] = cert
    admission = RP.build_admission(suffix="wave2", common=certs[A]["accepted"], certificates=certs)
    RP.write_admission(RP.admission_path(str(gen), row.slug, "wave2"), admission)
    # the CPU admission over the whole wave: both certificates valid, the record valid, docs carried
    adm = L.ladder_admission_report(all_cfgs, bound, tree=str(tree), tree_sha=FULL_SHA, object_digests=digests, gen_root=str(gen))
    assert adm["admission"] == RP.CERTIFICATE_VALID and set(adm["certificate_docs"]) == {A, B}
    assert {c: r["state"] for c, r in adm["certificates"].items()} == {A: "valid", B: "valid"}
    # --- step 2: score each cell with --cells NARROWED to it and the wave stated ----------------------
    for scored in (B, A, "none", "prompted__safety_refusal"):
        (cfg,) = L.ladder_configurations(L.isolate_out_dirs(L.plan_configurations(
            cells=[scored], modes=["off"], controls=[BFCL200], model="gemma4e4b-f22it"), "wave2"), launch)
        narrow = L.ladder_launch_bound(launch, [cfg], wave_cells=WAVE)
        assert narrow.cells == tuple(WAVE) and narrow.probe_cells == (A, B)         # the WAVE's membership, not the plan's
        adm1 = L.ladder_admission_report([cfg], narrow, tree=str(tree), tree_sha=FULL_SHA, object_digests=digests, gen_root=str(gen))
        assert adm1["admission"] == RP.CERTIFICATE_VALID, (scored, adm1["admission"], adm1["certificates"])
        recs, scored_cfgs, kw = L.ladder_admit(adm1, [cfg], narrow, sha="f6cf462", object_digests=digests, run_id="r",
                                                reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT)
        assert recs == [] and [c["cell"] for c in scored_cfgs] == [scored] and set(kw["certificates"]) == {A, B}
        # the worker admits the cell on the whole wave's certificates and record (and stops at the forbidden standup)
        with pytest.raises(BaseException, match="a standup was started"):
            L._run_configuration(scored_cfgs[0], "f6cf462", object_digests=digests, ladder=kw)
        assert L.ladder_worker_admission(scored_cfgs[0], row=row, tree=str(tree), tree_sha=FULL_SHA,
                                         object_digests=digests, ladder=kw) is None, scored
    # the plan must be a SUBSET of the stated wave; a cell outside it refuses by name
    (stray,) = L.ladder_configurations(L.isolate_out_dirs(L.plan_configurations(
        cells=[B], modes=["off"], controls=[BFCL200], model="gemma4e4b-f22it"), "wave2"), launch)
    with pytest.raises(ValueError, match="not all members of the certified wave"):
        L.ladder_launch_bound(launch, [stray], wave_cells=["none", A])
    with pytest.raises(ValueError, match="repeat"):
        L.ladder_launch_bound(launch, [stray], wave_cells=[A, A, B])
    # the OLD shape (no wave stated): the plan IS the wave
    assert L.ladder_launch_bound(launch, all_cfgs).cells == tuple(WAVE)
    # a fitted scoring cell is still held to ITS OWN certificate: B's container carrying only A's certificate refuses
    (cfg_b,) = L.ladder_configurations(L.isolate_out_dirs(L.plan_configurations(
        cells=[B], modes=["off"], controls=[BFCL200], model="gemma4e4b-f22it"), "wave2"), launch)
    narrow = L.ladder_launch_bound(launch, [cfg_b], wave_cells=WAVE)
    only_a = {"launch": narrow.as_dict(), "certificates": {A: certs[A]}, "admission": admission}
    status, why = L.ladder_worker_admission(dict(cfg_b, ladder_rung=RP.accepted_block(16)), row=row, tree=str(tree),
                                            tree_sha=FULL_SHA, object_digests=digests, ladder=only_a)
    assert status == "LADDER_UNCERTIFIED" and "its OWN" in why
    swapped = {"launch": narrow.as_dict(), "certificates": {A: certs[A], B: certs[A]}, "admission": admission}
    status, why = L.ladder_worker_admission(dict(cfg_b, ladder_rung=RP.accepted_block(16)), row=row, tree=str(tree),
                                            tree_sha=FULL_SHA, object_digests=digests, ladder=swapped)
    assert status == "LADDER_PROBE_CELL_MISMATCH" and "must be the probed cell" in why
    # `certificate_docs` is carried even when the wave RECORD is absent (an interrupted --ladder-certify
    # keeps its valid per-cell certificates instead of re-probing them and refusing LADDER_CERTIFICATE_EXISTS)
    RP.admission_path(str(gen), row.slug, "wave2").unlink()
    adm0 = L.ladder_admission_report(all_cfgs, bound, tree=str(tree), tree_sha=FULL_SHA, object_digests=digests, gen_root=str(gen))
    assert adm0["admission"] == RP.CERTIFICATE_ABSENT and set(adm0["certificate_docs"]) == {A, B}


def test_the_dispatchers_return_modal_functions_with_spawn_for_every_row(tmp_path):
    """Round 4 (closing pass of 0257c073): the shared-body refactor left `@app.function` on the
    PRIVATE bodies, so `configuration_function()` / `probe_function()` returned plain Python functions for
    every non-Nemotron row and `fan_out`'s `.spawn` failed at once, while the Nemotron wrappers called a
    decorated Modal Function directly. Signature binding cannot see this; a faithful Function double can:
    the stub's `App.function` here returns an object that carries `.spawn` / `.remote` and is NOT callable
    (a real `modal.Function` is not the function it wraps), so a dispatcher returning the undecorated body,
    or a wrapper calling its body as a Function, fails here."""
    import types

    class _Fn:
        """A modal.Function double: spawn / remote, not callable, and it knows its options."""
        def __init__(self, f, opts):
            self._f, self.options, self.__name__ = f, dict(opts), f.__name__

        def spawn(self, *a, **k):
            return _Call(lambda: self._f(*a, **k))

        def remote(self, *a, **k):
            return self._f(*a, **k)

        def __call__(self, *a, **k):
            raise TypeError(f"modal.Function {self.__name__!r} is not directly callable — use .remote() / .spawn()")

    stub = _stub_modal()

    class _App(stub.App):
        def function(self, **kw):
            return lambda f: _Fn(f, kw)
    stub.App = _App
    L = _load_launcher(stub, app_name=TEST_APP_NAME, name="tinycap_served_function_double")
    nemo = C.model_row("nemotron3nano4b")
    for slug in ("gemma4e4b-f22it", "gemma4e4b-f22", "qwen35_4b-f22", "gemma4e4b-sdfbase", C.MODEL_SLUG, "nemotron3nano4b"):
        row = C.model_row(slug)
        for chooser, base, nemotron in ((L.configuration_function, "run_configuration", "run_configuration_nemotron"),
                                        (L.probe_function, "probe_rung", "probe_rung_nemotron")):
            fn = chooser(row)
            assert isinstance(fn, _Fn) and hasattr(fn, "spawn") and hasattr(fn, "remote"), (slug, chooser.__name__, type(fn))
            assert fn.__name__ == (nemotron if row.family == nemo.family else base), (slug, fn.__name__)
            with pytest.raises(TypeError, match="not directly callable"):
                fn({"name": "cfg"}, "sha")
    # the base and Nemotron pairs carry the SAME options but for the image (+ the Nemotron cpu); the bodies are plain
    for base, nemotron, timeout in (("run_configuration", "run_configuration_nemotron", 8 * 3600), ("probe_rung", "probe_rung_nemotron", 4 * 3600),
                                    # capeval-full-tasks: the full legs' variants under their own function timeout
                                    ("run_configuration_full", "run_configuration_nemotron_full", C.FULL_FUNCTION_TIMEOUT_S)):
        b, n = getattr(L, base), getattr(L, nemotron)
        assert {k: v for k, v in b.options.items() if k != "image"} == {k: v for k, v in n.options.items() if k not in ("image", "cpu")}
        assert b.options["gpu"] == "H200" and b.options["timeout"] == timeout and b.options["max_containers"] == C.MAX_GPU_CONTAINERS
        assert n.options["cpu"] == L.NEMOTRON_FUNCTION_CPU and n.options["image"] is L.image_nemotron and b.options["image"] is L.image
    assert isinstance(L._configuration_worker, types.FunctionType) and isinstance(L._probe_worker, types.FunctionType)
    # the wrappers call the PLAIN body (never a Function object), and fan_out spawns through the chosen Function
    seen = []
    L._run_configuration = lambda cfg, sha, **kw: seen.append((cfg["name"], kw.get("protocol_check"))) or {"name": cfg["name"], "status": "DONE"}
    kwargs = L.configuration_kwargs(sha="s", limit=0, inspect_limit=0, samples=1, pass_no=1, controls=None,
                                    reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT, object_digests={}, oos_prompt_budget=None,
                                    run_id="r", ladder=None, protocol_check={"ok": True})
    recs = L.fan_out(L.configuration_function(C.model_row("gemma4e4b-f22it")), [{"name": "a"}, {"name": "b"}], kwargs)
    assert [r["status"] for r in recs] == ["DONE", "DONE"] and seen == [("a", {"ok": True}), ("b", {"ok": True})]
    assert L.run_configuration_nemotron.remote({"name": "n"}, **kwargs)["status"] == "DONE" and seen[-1] == ("n", {"ok": True})
    L._probe_rung = lambda cfg, sha, **kw: {"name": cfg["name"], "status": "OK", "ladder": kw["ladder"].as_dict()}
    ladder = L.LadderLaunch(selection_path="sel.json", selection={}, cap_overrides_by_task={}, operator_rung=None, reprobe=False,
                            reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT, samples=1, out_suffix="w").as_dict()
    pr = L.probe_function(nemo).remote({"name": "p"}, "s", object_digests={}, ladder=ladder, run_id="r")
    assert pr["status"] == "OK" and pr["ladder"] == ladder
    # source pin: every @app.function decorates a PUBLIC name — never a `_`-prefixed shared body
    decorated, pending = [], False
    for ln in LAUNCHER.read_text().splitlines():
        if ln.startswith("@app.function("):
            pending = True
        elif pending and ln.startswith("def "):
            decorated.append(ln[4:].split("(")[0])
            pending = False
    assert decorated and not [d for d in decorated if d.startswith("_")], decorated
    assert {"run_configuration", "run_configuration_nemotron", "probe_rung", "probe_rung_nemotron"} <= set(decorated)


def test_a_failed_standup_leaves_a_pointer_with_an_empty_event_file(L, tmp_path, monkeypatch):
    """Round 4: the attempt pointer was published BEFORE the standup but the per-attempt
    sandbox event file (`watch/sandboxes_<fcid>.jsonl`, the path the driver appends to) was not
    initialized — a STANDUP_FAILED attempt left a BCB pointer with NO event file, which the analysis's
    provenance check refused permanently, even after a successful retry under the same suffix. The
    empty event file is now created before the pointer is published."""
    from test_tinycap_launcher import _tree

    repos = tmp_path / "repos"
    tree = _tree(repos, "f6cf462")
    for rel in (C.BCB_PIN_PATH, C.BFCL_PIN_PATH, C.BCB_LIB_FAMILIES_PATH):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text((REPO / rel).read_text())
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "0")
    monkeypatch.setattr(L, "client_lock_sha256", lambda tree: None)
    monkeypatch.setattr(L, "STANDUP_WAIT_S", 0)
    monkeypatch.setattr(L, "current_function_call_id", lambda: "fc-standup-dies")

    class _DeadStandup:
        """A standup that never writes endpoint.json and exits at once."""
        def __init__(self, argv, env=None, stdout=None, stderr=None, **kw):
            if stdout is not None and hasattr(stdout, "write"):
                stdout.write("standup: boom\n")
                stdout.close()

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

        def kill(self):
            pass
    monkeypatch.setattr(L.subprocess, "Popen", _DeadStandup)
    out = tmp_path / "gen"
    (cfg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=[BCB])
    cfg = dict(cfg, out_dir=str(out))
    shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)
    rec = L._run_configuration(cfg, "f6cf462", run_id="f6cf462-cafebabe")
    assert rec["status"] == "STANDUP_FAILED" and rec["sandboxed"] is True
    pointer = Path(rec["sandbox_pointer"])
    record = Path(rec["sandbox_record"])
    assert pointer.is_file() and json.loads(pointer.read_text())["controls"] == [BCB]
    assert record == out / "watch" / "sandboxes_fc-standup-dies.jsonl"
    assert record.is_file() and record.read_bytes() == b""                    # published pointer -> EMPTY event file, never absent
    # an event file already carrying events (a resumed attempt id) is never truncated
    record.write_text('{"event": "created"}\n')
    rec2 = L._run_configuration(cfg, "f6cf462", run_id="f6cf462-cafebabe")
    assert rec2["status"] == "STANDUP_FAILED" and record.read_text() == '{"event": "created"}\n'
    # source pin: the event file is created BEFORE the pointer is written
    src = LAUNCHER.read_text()
    assert src.index("record.touch()") < src.index('rec["sandbox_pointer"] = str(write_attempt_pointer(')


def test_every_name_main_calls_locally_is_a_plain_callable_and_every_remote_a_function():
    """Round 8 (found live: the wave wrappers' `modal run` printed "Created function is_lane_row" and
    `main` raised `TypeError: 'Function' object is not callable`): round 7 inserted helpers between
    `preflight`'s `@app.function` decorator and its `def`, so the decorator bound the FIRST helper
    and `preflight` was left undecorated. Under a faithful non-callable Function double, every
    module-level name that `main` / the preflight body / the worker call as a plain function must
    stay callable, and every name dispatched with `.remote` / `.spawn` must be a Function — and the
    set of decorated names is pinned exactly."""
    import ast
    import types

    class _Fn:
        def __init__(self, f, opts):
            self._f, self.options, self.__name__ = f, dict(opts), f.__name__

        def spawn(self, *a, **k):
            return _Call(lambda: self._f(*a, **k))

        def remote(self, *a, **k):
            return self._f(*a, **k)

        def __call__(self, *a, **k):
            raise TypeError(f"modal.Function {self.__name__!r} is not directly callable — use .remote() / .spawn()")

    stub = _stub_modal()

    class _App(stub.App):
        def function(self, **kw):
            return lambda f: _Fn(f, kw)
    stub.App = _App
    L = _load_launcher(stub, app_name=TEST_APP_NAME, name="tinycap_served_function_double_r8")
    src = LAUNCHER.read_text()
    tree = ast.parse(src)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    decorated = {name for name, n in functions.items()
                 if any(isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "function" for d in n.decorator_list)}
    assert decorated == {"run_configuration", "run_configuration_nemotron", "probe_rung", "probe_rung_nemotron",
                         # capeval-full-tasks: the full legs' function variants (FULL_FUNCTION_TIMEOUT_S)
                         "run_configuration_full", "run_configuration_nemotron_full",
                         "ladder_operator_certificate", "check_protocol_sampling",
                         # tinycap-shared-standup §1.9.4 (module E): the wave-close prune, a CPU function
                         "close_wave", "write_ladder_admission",
                         # tinycap-shared-standup §1.1.3 (module A): ONE container per model row, both images
                         "run_row_full", "run_row_nemotron_full",
                         # tinycap-shared-standup landing 2 (§2.2.4): the CPU BigCodeBench verifier main spawns per GENERATED cell
                         "verify_cell_cpu",
                         # landing 2 code review H3 (b): the CPU function main's --verify-pending calls (.remote) to read a
                         # GENERATED row's durable state off the volume before spawning the still-pending verifiers
                         "row_verify_pending_report",
                         "merge_summaries", "stage", "preflight",
                         # the IFEval canonical battery: the Nemotron backend + same-GPU peer probe
                         "nemotron_backend_probe"}, decorated
    for name in decorated:
        assert isinstance(getattr(L, name), _Fn), name
    # every module-level function that main / the preflight body / the worker CALL directly (`name(`) is plain
    def direct_calls(fn_name):
        node = functions[fn_name]
        return {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    for caller in ("main", "preflight", "_run_configuration", "ladder_admit", "fan_out"):
        for callee in direct_calls(caller) & set(functions):
            obj = getattr(L, callee)
            assert not isinstance(obj, _Fn), f"{caller} calls {callee}() directly but it is a Modal Function"
            assert callable(obj), (caller, callee)
    # and every `.remote` / `.spawn` receiver named in main / ladder_admit IS a Function
    for caller in ("main", "ladder_admit"):
        for c in ast.walk(functions[caller]):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in ("remote", "spawn") \
                    and isinstance(c.func.value, ast.Name) and c.func.value.id in functions:
                assert isinstance(getattr(L, c.func.value.id), _Fn), (caller, c.func.value.id)
    # the round-7 predicate in particular
    assert isinstance(L.is_lane_row, types.FunctionType) and L.is_lane_row(C.model_row("gemma4e4b-f22it")) is True
    assert L.is_lane_row(C.model_row(C.MODEL_SLUG)) is False


# --------------------------------------------------------------------------- #
# the truncation ladder runs on the NATIVE engine over the ENGINE-SERVED rungs  #
# — coalesced, certificates and identities bound to the served geometry; the    #
# lens ladder byte-identical. The sandbox watcher derives a native run's out    #
# root with the launcher                                                        #
# --------------------------------------------------------------------------- #

#: a KV cache of 480,000 tokens: 15 x 32,768 = 491,520 is refused, 14 x 32,768 = 458,752 fits — the native
#: engine's own verdict names the SERVED S (the core composes vLLM at the kernel-lowered S before readiness)
KV_REPORT_NATIVE = {"kv_num_cache_groups": 1, "kv_num_gpu_blocks": 30_000, "kv_block_size": 16}
REFUSAL_15 = ("sisteer.capeval.serving.vllm_server.KvCoverageShortfall: resolved scheduler state does not match the "
              "composed no-chunking invariant:\n"
              "- worker 0: " + str(vs.kv_coverage_problem(KV_REPORT_NATIVE, max_num_seqs=15, max_model_len=32_768)) + "\n")
NATIVE_LADDER_EP = {"engine": "lora_native", "serving_engine": "lora_native",
                    "adapter_bank": "/vol/tinycap/multilora_adapters/gemma4e4b"}


def native_ladder_endpoint_geometry(asked: int, *, L: int = 32_768) -> dict:
    """The ``geometry`` block the NATIVE core writes for a gemma standup ASKED ``asked`` at the ladder window
    (``native_standup``: ``ResolvedGeometry.as_dict()`` at the asked S — the model's budget, ``seqs_source``
    ``override`` below the budget's 18, ``identity`` iff 18 — LOWERED under the LoRA kernel ceiling when the
    asked S exceeds it, restated at the lowered S with ``seqs_source`` ``override`` (the no-``--eval`` branch of
    ``lora_kernel_ceiling_geometry``) and the §1.8 kernel step recorded; else ``adjustments: []``). The record
    the review-r3 regressions push through the scoring driver's ``_geometry_block``."""
    block = sg.resolve_geometry(None, max_model_len=L, max_num_seqs=int(asked), model_key="gemma4e4b").as_dict()
    step = sg.kernel_ceiling_adjustment("gemma4e4b", L, int(asked))
    if step is not None:
        block.update(max_num_seqs=int(step["to_S"]), batched_tokens=int(step["to_S"]) * L, identity=False, seqs_source="override")
    block[sg.ADJUSTMENTS_KEY] = [dict(step)] if step is not None else []
    return block


class _NativeProbeProcs:
    """A Popen double for the NATIVE rung probe that emulates the ENGINE (the probe asks the
    REQUESTED rung, so asked 18 the core composes vLLM at the kernel-lowered 15 and records the step 18 -> 15; asked
    14 it serves 14 with an EMPTY adjustments record): at the SERVED S 15 (the engine-served identity) the standup
    dies with the KV refusal naming 15 x 32,768 (no endpoint); at the served 14 it comes up with a native
    ``endpoint.json`` — the engine attested, the geometry block the native core writes
    (:func:`native_ladder_endpoint_geometry`) — ``endpoint_extra`` merged over it for the refusal cases."""

    script = {15: "refuse", 14: "accept"}          # keyed by the SERVED S (what the engine's verdict names)
    endpoint_extra: dict = {}
    argvs: list = []

    def __init__(self, argv, *a, **k):
        self.argv = list(argv)
        type(self).argvs.append(self.argv)
        out = Path(self.argv[self.argv.index("--out") + 1])
        asked = int(self.argv[self.argv.index("--max-num-seqs") + 1])
        S = sg.adjusted_seqs("gemma4e4b", 32_768, asked)              # the S the engine SERVES for what it was asked
        what = type(self).script.get(S, "die")
        self.rc = None
        if what == "accept":
            rec = {"proxy_url": "http://127.0.0.1:1/v1", **NATIVE_LADDER_EP,
                   "geometry": native_ladder_endpoint_geometry(asked), **type(self).endpoint_extra}
            (out / "endpoint.json").write_text(json.dumps(rec))
        elif what == "refuse":
            k["stdout"].write(REFUSAL_15)
            k["stdout"].flush()
            self.rc = 1
        else:
            k["stdout"].write("[standup] engine died\n")
            k["stdout"].flush()
            self.rc = 1

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        # lifecycle-faithful: the REAL standup unlinks its ``endpoint.json`` as it
        # tears down (capctrl_serving_up.py, after ``su.tear_down()``), so a launcher that reads the record after
        # ``_tear_down_standup`` finds nothing — the double deletes it exactly where the producer does
        out = Path(self.argv[self.argv.index("--out") + 1])
        (out / "endpoint.json").unlink(missing_ok=True)
        return self.rc if self.rc is not None else 0

    def terminate(self):
        self.rc = 0

    def kill(self):
        self.rc = 0


def test_the_native_ladder_is_the_models_ladder_through_the_kernel_ceiling_coalesced(L, monkeypatch):
    """The pure half: the native engine lowers S under the LoRA kernel's int32 index ceiling
    before it launches, so the ladder it serves is the model's ladder mapped through the SAME pre-computed step every
    native configuration gets (``geometry.adjusted_seqs`` / ``expected_geometry_for``), rungs collapsing onto one served
    S coalesced — gemma at 32,768: 18 and 16 -> 15, served (15, 14, 12), identity 15; nemotron (D = 5120): every rung
    -> the floor 12, ONE rung; llama8b: 14 and 12 stand, and the block STILL names the engine. The lens block is
    byte-identical. The ONE narrow refusal: a native ladder that cannot be sized (no pinned width; a ceiling that puts
    every rung below the floor), by name."""
    NATIVE, LENS = L.SERVING_PATH_LORA_NATIVE, L.SERVING_PATH_LENS
    gemma = RP.ladder_spec("gemma4e4b")
    native = RP.native_ladder_spec(gemma)
    assert (native.seqs, native.identity_S) == ((15, 14, 12), 15) and native.native and native.serving_engine == NATIVE
    assert native.served_rungs == ((15, (18, 16)), (14, (14,)), (12, (12,)))
    assert native.kernel_ceiling == {"D": 4096, "ceiling": (2 ** 31 - 1) // 4096, "max_S": 15}
    assert native.requested_seqs == gemma.seqs == (18, 16, 14, 12) and native.requested_identity_S == 18
    assert native.budget is gemma.budget and native.window == gemma.window == 32_768
    for requested, served in ((18, 15), (16, 15), (14, 14), (12, 12)):        # each served rung IS the §1.8 step's S
        assert sg.adjusted_seqs("gemma4e4b", 32_768, requested) == served
        step = sg.kernel_ceiling_adjustment("gemma4e4b", 32_768, requested)
        assert (step["to_S"] if step else requested) == served
    block = native.block()
    assert tuple(block) == RP.LADDER_BLOCK_FIELDS_NATIVE == RP.LADDER_BLOCK_FIELDS + (
        "serving_engine", "requested_seqs", "requested_identity_S", "served_rungs", "kernel_ceiling")
    assert block["served_rungs"] == [{"served": 15, "requested": [18, 16]}, {"served": 14, "requested": [14]},
                                     {"served": 12, "requested": [12]}]
    assert block["seqs"] == [15, 14, 12] and block["identity_S"] == 15 and block["requested_seqs"] == [18, 16, 14, 12]
    # the LENS block: the four keys, byte-identical (the goldens in this module stand on it)
    assert gemma.block() == {"model_key": "gemma4e4b", "window": 32_768, "seqs": [18, 16, 14, 12], "identity_S": 18}
    assert tuple(gemma.block()) == RP.LADDER_BLOCK_FIELDS and not gemma.native and gemma.serving_engine is None
    # the launcher's ONE derivation: the lens ladder by default, the served one under the path
    row = C.model_row("gemma4e4b-sdfbase")
    assert L.ladder_spec_for(row, LENS) == gemma == L.ladder_spec_of(row) and L.ladder_spec_for(row, NATIVE) == native
    # nemotron: every requested rung collapses onto the floor — one rung, no operator rung below it
    nemo = RP.native_ladder_spec(RP.ladder_spec("nemotron3nano4b"))
    assert (nemo.seqs, nemo.identity_S, nemo.served_rungs) == ((12,), 12, ((12, (24, 22, 20, 18)),))
    assert nemo.kernel_ceiling == {"D": 5120, "ceiling": (2 ** 31 - 1) // 5120, "max_S": 12}
    # llama8b: the ceiling's 15 lowers neither 14 nor 12 — the rungs stand and the block still binds the engine
    llama = RP.native_ladder_spec(RP.ladder_spec("llama8b"))
    assert (llama.seqs, llama.identity_S) == ((14, 12), 14) == (RP.ladder_spec("llama8b").seqs, RP.ladder_spec("llama8b").identity_S)
    assert llama.block()["serving_engine"] == NATIVE and llama.block() != RP.ladder_spec("llama8b").block()
    assert llama.served_rungs == ((14, (14,)), (12, (12,)))
    # a recorded block reads back as the spec it was written from (the certificate readers' fallback)
    assert RP.spec_from_block(native.block(), gemma.budget) == native
    assert RP.spec_from_block(gemma.block(), gemma.budget) == gemma
    with pytest.raises(RP.LadderError, match="neither the lens ladder's"):
        RP.spec_from_block({**gemma.block(), "serving_engine": NATIVE}, gemma.budget)
    with pytest.raises(RP.LadderError, match="already the native engine's"):
        RP.native_ladder_spec(native)
    # the narrow refusals, by name: no pinned width; a ceiling below the floor (a D of 8192: at most 7 at 32,768)
    monkeypatch.delitem(sg.LORA_MAX_MODULE_DIM_BY_MODEL, "gemma4e4b")
    with pytest.raises(RP.LadderError, match="no pinned LoRA max module dim"):
        RP.native_ladder_spec(gemma)
    with pytest.raises(L.LadderGeometryRefused, match=r"--serving-path lora_native: .*no pinned LoRA max module dim"):
        L.ladder_spec_for(row, NATIVE)
    assert L.ladder_spec_for(row, LENS) == gemma                                   # the lens ladder needs no width
    monkeypatch.setitem(sg.LORA_MAX_MODULE_DIM_BY_MODEL, "gemma4e4b", 8192)
    with pytest.raises(RP.LadderError, match=r"at most S = 7 at window 32768 .* below the ladder floor 12"):
        RP.native_ladder_spec(gemma)


def test_a_native_ladder_probes_the_served_rungs_and_binds_its_certificates_to_the_engine(L, tmp_path, monkeypatch, capsys):
    """The gemma case end to end on the CPU: a native ladder launch plans and gates over the
    SERVED rungs (15, 14, 12), every rung's store identity is at the served S on the engine, the fitted cell's probe
    stands up natively (serving path, adapter bank, KV-derived S off) at 15 then 14 and writes a certificate whose
    ``ladder`` block names the engine and the 18/16 -> 15 mapping, filed under the NATIVE namespace; the certificate
    admits the native provenance and refuses the lens one (and a lens certificate refuses the native one); the wave
    record and the scoring worker admit a native carrier and refuse a lens certificate or a record naming the other
    engine; the operator certificate names a SERVED rung; readiness is held to the engine and the served S; main's dry
    run describes the served ladder. The lens ladder's goldens elsewhere in this module stand untouched."""
    NATIVE, LENS = L.SERVING_PATH_LORA_NATIVE, L.SERVING_PATH_LENS
    repos, tree = _ladder_tree(tmp_path)
    sdf_cfg = C.model_row("gemma4e4b-sdfbase").cfg                  # the LoRA row's checked-in config (the certificate binds its digest)
    (tree / sdf_cfg).parent.mkdir(parents=True, exist_ok=True)
    (tree / sdf_cfg).write_text((REPO / sdf_cfg).read_text())
    gen = tmp_path / "gen"
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(gen))
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L.subprocess, "Popen", _NativeProbeProcs)
    monkeypatch.setattr(_NativeProbeProcs, "argvs", [])
    monkeypatch.setattr(_NativeProbeProcs, "script", {15: "refuse", 14: "accept"})
    monkeypatch.setattr(_NativeProbeProcs, "endpoint_extra", {})
    sel_path = _write_selection(tmp_path / "ladder_selection.json", SEL_IDS)
    caps = C.format_cap_overrides(CAPS)
    digests = {DERIVED: {"artifact_sha256": "ab" * 32}}
    # the launch record carries the path (serialized iff native: the lens record's dict is byte-identical)
    ladder = L.ladder_launch_args(cap_override=caps, item_selection=str(sel_path), out_suffix="ladder-n", serving_path=NATIVE)
    assert ladder.serving_path == NATIVE and ladder.native and ladder.serving_engine == NATIVE
    assert ladder.as_dict()[L.SERVING_PATH_KEY] == NATIVE
    assert L.LadderLaunch.from_dict(json.loads(json.dumps(ladder.as_dict()))) == ladder
    lens_ladder = L.ladder_launch_args(cap_override=caps, item_selection=str(sel_path), out_suffix="ladder-n")
    assert lens_ladder.serving_path == LENS and not lens_ladder.native and L.SERVING_PATH_KEY not in lens_ladder.as_dict()
    cfgs = L.ladder_configurations(L.plan_configurations(cells=["none", DERIVED], modes=["off"], controls=[BCB, BFCL200],
                                                         model="gemma4e4b-sdfbase", serving_path=NATIVE), ladder)
    assert all(L.serving_path_of(c) == NATIVE for c in cfgs)
    lens_plan = L.plan_configurations(cells=["none", DERIVED], modes=["off"], controls=[BCB, BFCL200], model="gemma4e4b-sdfbase")
    lens_cfgs = L.ladder_configurations(lens_plan, lens_ladder)
    # a record and a plan that disagree about the engine never meet: refused where they first do, and at the bind
    with pytest.raises(ValueError, match="the ladder launch record says lens"):
        L.ladder_configurations(cfgs, lens_ladder)
    with pytest.raises(ValueError, match="the ladder launch record says lora_native"):
        L.ladder_launch_bound(ladder, lens_plan)
    bound, lens_bound = L.ladder_launch_bound(ladder, cfgs), L.ladder_launch_bound(lens_ladder, lens_cfgs)
    row = L.row_of(cfgs[0])
    spec, lens_spec = L.ladder_spec_for(row, NATIVE), L.ladder_spec_for(row, LENS)
    assert (spec.seqs, spec.identity_S) == ((15, 14, 12), 15) and (lens_spec.seqs, lens_spec.identity_S) == ((18, 16, 14, 12), 18)
    # the CPU gate walks every SERVED rung; the dry-run rung is the served identity 15 — `model_budget` / `identity`
    # against the served identity, the record naming the engine and the requested rungs it stands for
    L.ladder_cpu_gate(cfgs, bound)
    dry = L.ladder_dry_run_cfgs(cfgs, bound)
    assert all(c["ladder_rung"] == RP.accepted_block(15, spec=spec) == {
        "max_model_len": 32_768, "max_num_seqs": 15, "seqs_source": "model_budget", "rung_source": "identity"} for c in dry)
    g, record = L.served_geometry(dry[0], 1)
    # the served identity rung 15 is ASKED as 18 — the budget's identity S, so the engine's own
    # kernel step lowers it and records the step — and the record is consistent: model_budget AND identity True
    assert (g.max_model_len, g.max_num_seqs, g.seqs_source, g.window_source) == (32_768, 18, "model_budget", RP.LADDER_WINDOW_SOURCE)
    assert (record["max_num_seqs"], record["identity"], record["ladder_requested_max_num_seqs"], record["ladder_served_max_num_seqs"]) == (18, True, 18, 15)
    assert L.GEOMETRY_LADDER_REDUCTION_KEY not in record                      # the identity rung reduces nothing
    assert (record["rung_source"], record["identity_max_num_seqs"], record["ladder_seqs"], record["ladder_reduced_s"]) == (
        "identity", 15, [15, 14, 12], False)
    assert record[C.SERVED_ENGINE_FIELD] == NATIVE and record["ladder_requested_for_S"] == [18, 16]
    assert record["ladder_requested_seqs"] == [18, 16, 14, 12] and record["ladder_requested_identity_S"] == 18
    assert record["ladder_served_rungs"] == spec.block()["served_rungs"]
    # the store identity every rung hashes: the SERVED S, on the engine (the step lowers nothing further)
    exp = L.expected_identity(dry[0], g, pass_no=1, controls=dry[0]["controls"], samples=1, tree=str(tree), tree_sha=FULL_SHA,
                              serving_path=NATIVE)
    assert exp.served["max_num_seqs"] == exp.served["concurrency"] == 15 and exp.served[C.SERVED_ENGINE_FIELD] == NATIVE
    g14, rec14 = L.served_geometry(dict(cfgs[0], ladder_rung=RP.accepted_block(14, spec=spec)), 1)
    assert (g14.max_num_seqs, g14.seqs_source, rec14["rung_source"], rec14["ladder_requested_for_S"]) == (14, "override", "probe", [14])
    # a rung BELOW the budget's identity carries the ladder's reduction record (geometry.ladder_reduction) for the driver
    assert rec14["identity"] is False and (rec14["ladder_requested_max_num_seqs"], rec14["ladder_served_max_num_seqs"]) == (14, 14)
    assert rec14[L.GEOMETRY_LADDER_REDUCTION_KEY] == sg.ladder_reduction_record(
        identity_S=18, requested_S=14, max_model_len=32_768, ladder_seqs=(18, 16, 14, 12), rung_source="probe")
    # a REQUESTED rung the engine serves elsewhere is not a native rung: the gate refuses it by name
    with pytest.raises(L.LadderGeometryRefused, match=r"ladder S 16 is not on the ladder \[15, 14, 12\]"):
        L.served_geometry(dict(cfgs[0], ladder_rung=RP.accepted_block(16)), 1)
    # the lens plan's gate is byte-identical: its identity rung is 18 on the model's ladder
    g_l, rec_l = L.served_geometry(dict(lens_cfgs[0], ladder_rung=RP.accepted_block(18)), 1)
    assert (g_l.max_num_seqs, rec_l["identity_max_num_seqs"], rec_l["ladder_seqs"]) == (18, 18, [18, 16, 14, 12])
    assert C.SERVED_ENGINE_FIELD not in rec_l and "ladder_requested_seqs" not in rec_l
    # the PROBE of the fitted cell: 15 refused by the engine's own verdict (naming 15 x 32,768), 14 accepted natively
    probe_cfg = next(c for c in cfgs if c["cell"] == DERIVED)
    rec = L._probe_rung(probe_cfg, "f6cf462", object_digests=digests, ladder=bound, run_id="f6cf462-native")
    assert rec["status"] == "OK", rec
    assert rec["serving_path"] == NATIVE and rec[C.SERVED_ENGINE_FIELD] == NATIVE and rec["ladder"] == spec.block()
    cert = rec["certificate"]
    assert [(w["S"], w["outcome"], w["kv_tokens"]) for w in cert["walked"]] == [(15, "refused", 480_000), (14, "accepted", None)]
    assert cert["accepted"] == RP.accepted_block(14, spec=spec) == {
        "max_model_len": 32_768, "max_num_seqs": 14, "seqs_source": "override", "rung_source": "probe"}
    assert cert["ladder"] == spec.block() and cert["ladder"]["serving_engine"] == NATIVE
    assert cert["object_digests"] == {DERIVED: "ab" * 32} and cert["probed_cell"] == DERIVED
    assert tuple(cert) == RP.CERTIFICATE_FIELDS                       # the top-level schema is unchanged
    assert rec["walked"][1]["served"] == {"max_num_seqs": 14, "max_model_len": 32_768, "configured_max_num_seqs": 14, "kv_shrink": None}
    assert all(s["geometry"][C.SERVED_ENGINE_FIELD] == NATIVE for s in rec["walked"])
    # filed under the NATIVE namespace — never at the lens address of the same suffix
    path = RP.certificate_path(L.native_gen_root(str(gen)), row.slug, "ladder-n", DERIVED)
    assert Path(rec["certificate_path"]) == path and path.is_file() and str(path).startswith(f"{gen}/native/")
    assert RP.read_certificate(path)[0] == cert and not RP.certificate_path(str(gen), row.slug, "ladder-n", DERIVED).exists()
    # the two standups: on the serving path with the bank, KV-derived S OFF, asked the REQUESTED rung each served
    # rung stands for — 18 for the served 15 (the engine's kernel step lowers it), then 14
    argvs = _NativeProbeProcs.argvs
    assert [int(a[a.index("--max-num-seqs") + 1]) for a in argvs] == [18, 14]
    for a in argvs:
        i = a.index(L.SERVING_PATH_FLAG)
        assert a[i + 1:i + 4] == [NATIVE, L.ADAPTER_BANK_FLAG, f"{L.ADAPTER_BANK_ROOT}/gemma4e4b"]
        assert a[a.index(L.STANDUP_KV_DERIVED_SEQS_FLAG) + 1] == "off" and a[a.index("--max-model-len") + 1] == "32768"
    # BOUND to the engine: valid against the native provenance, refused against the lens one, and a lens
    # certificate for the same launch refuses the native provenance — whatever the numbers
    prov_n = L.ladder_provenance(row, tree, FULL_SHA, digests, bound, probed_cell=DERIVED)
    prov_l = L.ladder_provenance(row, tree, FULL_SHA, digests, lens_bound, probed_cell=DERIVED)
    assert prov_n.spec == spec and prov_l.spec == lens_spec and prov_n.as_dict()["ladder"] == spec.block()
    assert RP.certificate_is_valid(cert, prov_n) is None
    why = RP.certificate_is_valid(cert, prov_l)
    assert why and "ladder (the lens: this launch's) keys" in why and "serving_engine" in why
    lens_cert = RP.build_certificate(accepted=RP.accepted_block(16), provenance=prov_l, probe_run_id="lens", standup_s=1.0,
                                     walked=[RP.walked_entry(18, RP.WALK_REFUSED, REFUSAL_18), RP.walked_entry(16, RP.WALK_ACCEPTED)])
    assert RP.certificate_is_valid(lens_cert, prov_l) is None and tuple(lens_cert["ladder"]) == RP.LADDER_BLOCK_FIELDS
    why = RP.certificate_is_valid(lens_cert, prov_n)
    assert why and "ladder (the native engine: this launch's) keys" in why and "missing" in why
    forged = json.loads(json.dumps(cert))
    forged["ladder"]["served_rungs"][0]["requested"] = [18]                    # another mapping: the recompute refuses
    assert "!= this launch's" in RP.certificate_is_valid(forged, prov_n)
    tampered = json.loads(json.dumps(cert))
    tampered["ladder"]["serving_engine"] = LENS
    assert "'lens' engine's; this launch serves on 'lora_native'" in RP.certificate_is_valid(tampered, prov_n)
    # the WAVE record (under the native namespace, naming the engine) and the scoring WORKER's admission
    ar = L._write_ladder_admission("f6cf462", row.slug, "ladder-n", cert["accepted"], {DERIVED: cert}, NATIVE)
    assert ar["status"] == "OK", ar
    assert Path(ar["admission_path"]) == RP.admission_path(L.native_gen_root(str(gen)), row.slug, "ladder-n")
    assert ar["serving_path"] == NATIVE and ar[C.SERVED_ENGINE_FIELD] == NATIVE and ar["admission"]["common"] == cert["accepted"]
    carrier = {"launch": bound.as_dict(), "certificates": {DERIVED: cert}, "admission": ar["admission"]}
    admit = dict(row=row, tree=str(tree), tree_sha=FULL_SHA, object_digests=digests)
    scored = dict(probe_cfg, ladder_rung=dict(cert["accepted"]))
    assert L.ladder_worker_admission(scored, ladder=carrier, **admit) is None                        # the fitted cell: its own certificate
    base_scored = dict(next(c for c in cfgs if c["cell"] == "none"), ladder_rung=dict(cert["accepted"]))
    assert L.ladder_worker_admission(base_scored, ladder=carrier, **admit) is None                   # the base cell: the wave record
    lens_scored = dict(next(c for c in lens_cfgs if c["cell"] == DERIVED), ladder_rung=dict(cert["accepted"]))
    status, why = L.ladder_worker_admission(lens_scored, ladder=carrier, **admit)
    assert status == L.LADDER_ENGINE_MISMATCH == "LADDER_ENGINE_MISMATCH" and f"planned under {L.SERVING_PATH_FLAG} lens" in why
    status, why = L.ladder_worker_admission(scored, ladder={**carrier, "certificates": {DERIVED: lens_cert}}, **admit)
    assert status == "LADDER_CERTIFICATE_INVALID" and "ladder (the native engine: this launch's) keys" in why
    status, _ = L.ladder_worker_admission(scored, ladder={**carrier, "launch": lens_bound.as_dict()}, **admit)
    assert status == L.LADDER_ENGINE_MISMATCH
    assert L.ladder_worker_admission(dict(scored, ladder_rung=RP.accepted_block(15, spec=spec)), ladder=carrier, **admit)[0] == "LADDER_RUNG_MISMATCH"
    # the admission REPORT (the preflight's reading): the certificate at its native path, the record valid, the served rungs
    adm = L.ladder_admission_report(cfgs, bound, tree=str(tree), tree_sha=FULL_SHA, object_digests=digests)
    assert (adm["certificate"], adm["admission"], adm["accepted"], adm["path"]) == ("valid", "valid", cert["accepted"], str(path))
    assert adm[C.SERVED_ENGINE_FIELD] == NATIVE and adm["serving_path"] == NATIVE and adm["ladder"] == spec.block()
    assert adm["served_rungs"] == spec.block()["served_rungs"] and adm["requested_seqs"] == [18, 16, 14, 12]
    assert set(adm["cells"]) == {c["name"] for c in cfgs} and not adm["all_complete"]
    assert all(c["geometry"]["identity_max_num_seqs"] == 15 and c["geometry"][C.SERVED_ENGINE_FIELD] == NATIVE for c in adm["cells"].values())
    lens_adm = L.ladder_admission_report(lens_cfgs, lens_bound, tree=str(tree), tree_sha=FULL_SHA, object_digests=digests)
    assert lens_adm["certificate"] == "absent" and C.SERVED_ENGINE_FIELD not in lens_adm and "served_rungs" not in lens_adm
    assert lens_adm["path"] == str(RP.certificate_path(str(gen), row.slug, "ladder-n", DERIVED))       # the lens address, untouched
    # the OPERATOR certificate names a SERVED rung below the served identity; anything else refuses by name
    op = L.ladder_launch_args(cap_override=caps, item_selection=str(sel_path), out_suffix="ladder-op", ladder_rung="32768,14",
                              serving_path=NATIVE)
    oprec = L._operator_certificate(probe_cfg, "f6cf462", object_digests=digests, ladder=L.ladder_launch_bound(op, cfgs), run_id="op")
    assert oprec["status"] == "OK", oprec
    assert oprec["accepted"] == {"max_model_len": 32_768, "max_num_seqs": 14, "seqs_source": "override", "rung_source": "operator"}
    assert oprec["certificate"]["ladder"] == spec.block() and oprec[C.SERVED_ENGINE_FIELD] == NATIVE
    assert str(oprec["certificate_path"]).startswith(f"{gen}/native/")
    for rung, needle in (("32768,16", r"S 16 is not below the identity S 15 \(the S the lora_native engine SERVES for the budget's 18"),
                         ("32768,15", "S 15 is not below the identity S 15"),
                         ("32768,13", r"S 13 is not on the lora_native engine's served ladder \[15, 14, 12\] .*D = 4096: at most S = 15")):
        with pytest.raises(ValueError, match=needle):
            L.ladder_launch_args(cap_override=caps, item_selection=str(sel_path), out_suffix="x", ladder_rung=rung, serving_path=NATIVE)
    assert L.ladder_launch_args(cap_override=caps, item_selection=str(sel_path), out_suffix="x", ladder_rung="32768,16").operator_rung == (32768, 16)
    # main: a native ladder launch is planned, gated and described — no refusal; the dry run names the served ladder
    monkeypatch.setattr(L, "preflight", SimpleNamespace(remote=lambda *a, **k: pytest.fail("a dry run reached the preflight")))
    monkeypatch.setattr(L, "client_lock_sha256", lambda tree_: None)
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "0")
    capsys.readouterr()
    L.main(app_name=L.APP_NAME, sha="f6cf462", pass1=True, cells=f"none,{DERIVED}", modes="off", controls=f"{BCB},{BFCL200}",
           cap_override=caps, item_selection=str(sel_path), out_suffix="ladder-n", dry_run=True, serving_path=NATIVE)
    out = capsys.readouterr().out
    assert "S walked [15, 14, 12] (identity S 15" in out and "[18, 16] -> served 15" in out
    assert "ladder rung: 32768 x 15 served by the lora_native engine (the standup is asked S 18, the budget's rung it stands for; the identity rung: no reduction record" in out
    assert f"serving path: {NATIVE} ({L.SERVING_PATH_FLAG}; engine 'lora_native'): the ladder walks the SERVED rungs" in out
    assert f"{gen}/native/" in out and "nothing launched" in out
    # readiness is held to the ENGINE and the SERVED S: a lens record, or another S, refuses the walk by name
    for extra, status, needle in (({"engine": None, "serving_engine": None}, "SERVING_ENGINE_MISMATCH", "attests engine='lens'"),
                                  # a full native block serving 13 where the ladder's reduction names 14 — refused
                                  # with the scoring driver's words (the production reading), never as "a smaller S"
                                  ({"geometry": {**native_ladder_endpoint_geometry(14), "max_num_seqs": 13}},
                                   "SERVED_GEOMETRY_MISMATCH", "the served S 13 is not the ladder_reduction's to_S=14")):
        monkeypatch.setattr(_NativeProbeProcs, "endpoint_extra", extra)
        bad = L._probe_rung(probe_cfg, "f6cf462", object_digests=digests, ladder=bound, run_id="bad", reprobe=True)
        assert bad["status"] == status and needle in bad["error"] and bad["error"].startswith("rung S=14: "), bad
        assert bad["walked"][-1]["outcome"] == "failed" and "certificate" not in bad


def test_native_ladder_rung_records_pass_the_scoring_drivers_geometry_block(L, tmp_path):
    """ONE geometry interpretation for certification and production. An earlier probe asked the
    standup the SERVED S (15) directly, so the native core recorded ``adjustments: []`` and the probe certified it, while
    the scoring driver re-derived the budget's identity S = 18 from the SAME record and refused it for lacking the kernel
    step 18 -> 15 — a certificate production could not consume. Now, for every served rung of gemma's native ladder: the
    launcher asks the REQUESTED rung the served S stands for (18 for 15 — the engine's own kernel step lowers and records
    it, production's identity footprint; 14 and 12 as themselves, the launcher's ``ladder_reduction`` record beside), the
    (emulated) native core records what it was asked and served, the launcher files ``served_geometry.json`` beside the
    cells as the worker does, and ``served_capability_run._geometry_block`` — the scoring driver's OWN boundary, not a
    mocked readiness — accepts the record and copies the whole chain into the manifest; the launcher's readiness reading
    (``native_ladder_geometry_problem``, run by the probe before it certifies and by the worker before its driver) is the
    same reading. The earlier shape (asked the served 15 directly, ``[]``) is refused by the driver AND by the probe's
    readiness check with the driver's own words; a reduced rung with no launcher record beside it is a bare smaller S and
    refuses; a reduction beside a LENS record refuses; a lens record without one is copied as before."""
    sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as srun
    NATIVE, KEY = L.SERVING_PATH_LORA_NATIVE, L.GEOMETRY_LADDER_REDUCTION_KEY
    assert KEY == sg.LADDER_REDUCTION_KEY == "ladder_reduction"
    cfgs = L.plan_configurations(cells=["none", DERIVED], modes=["off"], controls=[BCB, BFCL200], model="gemma4e4b-sdfbase",
                                 serving_path=NATIVE)
    fitted = next(c for c in cfgs if c["cell"] == DERIVED)
    spec = L.ladder_spec_for(L.row_of(fitted), NATIVE)
    assert spec.seqs == (15, 14, 12)
    k18 = sg.kernel_ceiling_adjustment("gemma4e4b", 32_768, 18)
    ep_of = lambda asked: {"proxy_url": "http://127.0.0.1:1/v1", **NATIVE_LADDER_EP, "geometry": native_ladder_endpoint_geometry(asked)}

    def filed(out_root, geom, record, ep):
        """What the scoring WORKER files beside the cells (`_run_configuration`): its REC geometry block — the resolver's
        dict + the pass + the ladder record, the reduction lifted to the top — restated at the served S with the standup's
        adjustments copied whole (served_geometry_record), under served_geometry.json."""
        g = {**geom.as_dict(), C.GEOMETRY_PASS_FIELD: 1, "model_budget": record}
        if KEY in record:
            g[KEY] = dict(record[KEY])
        served, shrink, problem = L.served_geometry_of_endpoint(ep, geom, ladder_record=g)   # the worker's call, the record given
        assert problem is None, problem
        g = L.served_geometry_record(g, served, shrink, ep["geometry"][sg.ADJUSTMENTS_KEY])
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / L.SERVED_GEOMETRY_FILENAME).write_text(json.dumps(
            {"geometry": g, "endpoint": ep, "serving_path": NATIVE, C.SERVED_ENGINE_FIELD: NATIVE}, default=str))
        return g

    geoms, records = {}, {}
    for S in spec.seqs:
        geom, record = L.served_geometry(dict(fitted, ladder_rung=RP.accepted_block(S, spec=spec)), 1)
        geoms[S], records[S] = geom, record
        asked = int(geom.max_num_seqs)
        assert asked == RP.requested_rung_for(spec, S) == {15: 18, 14: 14, 12: 12}[S]
        ep = ep_of(asked)
        assert (ep["geometry"]["max_num_seqs"], ep["geometry"]["budget"], ep["geometry"]["model_key"]) == (S, 589_824, "gemma4e4b")
        g = filed(tmp_path / f"rung{S}", geom, record, ep)
        assert (g["max_num_seqs"], g["configured_max_num_seqs"], g["identity"]) == (S, asked, False)
        # the launcher's readiness reading and the driver's are ONE reading: both accept, the driver copies the chain
        assert L.native_ladder_geometry_problem(ep, geom, record) is None
        block = srun._geometry_block(ep, tmp_path / f"rung{S}")
        assert (block["max_num_seqs"], block["max_model_len"], block[C.GEOMETRY_PASS_FIELD], block["source"]) == (
            S, 32_768, 1, f"endpoint.json+{L.SERVED_GEOMETRY_FILENAME}")
        assert block[sg.ADJUSTMENTS_KEY] == ep["geometry"][sg.ADJUSTMENTS_KEY]
        assert srun._native_identity_seqs(ep["geometry"]) == sg.recorded_identity_seqs(ep["geometry"]) == 18   # the budget's, whatever was asked
        if S == 15:
            assert block[sg.ADJUSTMENTS_KEY] == [k18] and KEY not in block and KEY not in record and record["identity"] is True
        else:
            assert block[sg.ADJUSTMENTS_KEY] == [] and block[KEY] == record[KEY] == sg.ladder_reduction_record(
                identity_S=18, requested_S=S, max_model_len=32_768, ladder_seqs=(18, 16, 14, 12), rung_source="probe")
    # the shape where the standup asked the SERVED 15 directly — the core lowers nothing and records [] — is
    # refused by the driver, and the probe's readiness check now says the same words (a reader without the
    # record — a non-ladder standup's — reads the block against the S it asked, the OTHER interpretation; a ladder
    # rung's reader is handed the record and makes the driver's reading)
    ep15 = ep_of(15)
    assert ep15["geometry"][sg.ADJUSTMENTS_KEY] == [] and ep15["geometry"]["max_num_seqs"] == 15
    geom15 = dataclasses.replace(geoms[15], max_num_seqs=15, seqs_source="override")
    with pytest.raises(SystemExit, match=r"must record the 'lora_kernel_index_ceiling' step \(18 -> 15; .*and this record names none"):
        srun._geometry_block(ep15, tmp_path / "rung15")
    _, _, without_record = L.served_geometry_of_endpoint(ep15, geom15)
    assert without_record and "identity=False with 0 adjustment(s)" in without_record and "scoring driver" not in without_record
    _, _, with_record = L.served_geometry_of_endpoint(ep15, geom15, ladder_record=records[15])
    assert with_record and with_record.startswith("the scoring driver would refuse this standup record") and "(18 -> 15" in with_record
    why = L.native_ladder_geometry_problem(ep15, geom15, records[15])
    assert why and "must record the 'lora_kernel_index_ceiling' step (18 -> 15" in why and "names none" in why
    (tmp_path / "ep15").mkdir()
    (tmp_path / "ep15" / "endpoint.json").write_text(json.dumps(ep15))
    _, problem = L.native_rung_problem(tmp_path / "ep15" / "endpoint.json", geom15, 15, record=records[15])
    assert problem is not None and problem[0] == "SERVED_GEOMETRY_MISMATCH"
    assert "the scoring driver would refuse this standup record" in problem[1] and "(18 -> 15" in problem[1]
    # the honest identity rung passes the same readiness check (asked 18, served 15, the step recorded)
    (tmp_path / "ep18").mkdir()
    (tmp_path / "ep18" / "endpoint.json").write_text(json.dumps(ep_of(18)))
    assert L.native_rung_problem(tmp_path / "ep18" / "endpoint.json", geoms[15], 15, record=records[15]) == (json.loads(json.dumps(ep_of(18))), None)
    # a reduced rung with NO launcher record beside it is a bare smaller S: refused by name, never read as an identity
    (tmp_path / "bare").mkdir()
    with pytest.raises(SystemExit, match=r"must record the 'lora_kernel_index_ceiling' step \(18 -> 15"):
        srun._geometry_block(ep_of(14), tmp_path / "bare")
    assert "must record" in L.native_ladder_geometry_problem(ep_of(14), geoms[14], {})
    # a reduction record whose rung is not the one served refuses through the same rule
    with pytest.raises(SystemExit, match=r"the served S 12 is not the ladder_reduction's to_S=14"):
        srun._geometry_block(ep_of(12), tmp_path / "rung14")
    # a reduction beside a LENS standup record is a wiring error, refused by name; a lens record without one is copied as before
    lens_ep = {"proxy_url": "http://127.0.0.1:1/v1", "geometry": {"max_model_len": 32_768, "max_num_seqs": 14, "seqs_source": "override", "identity": False}}
    (tmp_path / "lens").mkdir()
    (tmp_path / "lens" / L.SERVED_GEOMETRY_FILENAME).write_text(json.dumps(
        {"geometry": {"max_model_len": 32_768, "max_num_seqs": 14, C.GEOMETRY_PASS_FIELD: 1, KEY: records[14][KEY]}}))
    with pytest.raises(SystemExit, match="beside a LENS standup record"):
        srun._geometry_block(lens_ep, tmp_path / "lens")
    (tmp_path / "lens-plain").mkdir()
    (tmp_path / "lens-plain" / L.SERVED_GEOMETRY_FILENAME).write_text(json.dumps(
        {"geometry": {"max_model_len": 32_768, "max_num_seqs": 14, C.GEOMETRY_PASS_FIELD: 1}}))
    plain = srun._geometry_block(lens_ep, tmp_path / "lens-plain")
    assert plain["max_num_seqs"] == 14 and sg.ADJUSTMENTS_KEY not in plain and KEY not in plain and plain[C.GEOMETRY_PASS_FIELD] == 1


def _pointer_under(W, run_id, cell, fcid, *, suffix="s7b", serving_engine):
    out_dir = W.cell_out_dir("gemma4e4b", cell, "off", suffix, serving_engine=serving_engine)
    rel = out_dir[len("/vol/"):]
    doc = {"run_id": run_id, "cell": cell, "function_call_id": fcid, "name": f"n-{cell}", "model": "gemma4e4b",
           "mode": "off", "out_dir": out_dir, "controls": [BCB], "tree_sha": "a" * 40, "sandbox_record": "r",
           "sandbox_app_name": C.BCB_SANDBOX_APP_NAME, "started_at": 1.0}
    return f"{rel}/watch/attempt_{fcid}.json", json.dumps(doc)


def test_the_watcher_derives_a_native_runs_out_root_with_the_launcher_and_sweeps_its_terminal_attempt(L, monkeypatch, tmp_path, capsys):
    """A native launch writes its attempt pointers under ``gen/native/<model>/…/watch`` but the printed
    watcher command named no engine, so the watcher derived the lens root, never found a pointer, never observed the
    FunctionCall's terminal state and left the orphan BCB sandboxes to their 10,800 s timeout. ONE root derivation is now
    shared (``contracts.gen_root_for_engine``), the printed argv names the path under native (lens byte-identical), and
    the watcher finds, classifies and sweeps the native attempt."""
    W = _load_watcher()
    NATIVE, LENS = C.SERVING_ENGINE_LORA_NATIVE, C.SERVING_ENGINE_LENS
    # ONE derivation: the launcher's native namespace IS the watcher's, and the planned out dir IS what the watcher reads
    assert L.native_gen_root() == C.gen_root_for_engine(C.VOL_GEN_ROOT, NATIVE) == f"{C.VOL_GEN_ROOT}/native"
    assert C.gen_root_for_engine(C.VOL_GEN_ROOT, LENS) == C.gen_root_for_engine(C.VOL_GEN_ROOT, None) == C.VOL_GEN_ROOT
    assert L.OUT_SUBDIR_NATIVE == C.OUT_SUBDIR_NATIVE == "native"
    with pytest.raises(ValueError, match="serving_engine must be one of"):
        C.gen_root_for_engine(C.VOL_GEN_ROOT, "hf")
    (cfg,) = L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"], controls=[BCB], serving_path=NATIVE), "s7b")
    assert cfg["out_dir"] == W.cell_out_dir("gemma4e4b", "none", "off", "s7b", serving_engine=NATIVE) \
        == f"{C.VOL_GEN_ROOT}/native/gemma4e4b/none/thinking_off/s7b"
    (lens_cfg,) = L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"], controls=[BCB]), "s7b")
    assert lens_cfg["out_dir"] == W.cell_out_dir("gemma4e4b", "none", "off", "s7b") == f"{C.VOL_GEN_ROOT}/gemma4e4b/none/thinking_off/s7b"
    # the printed command: lens byte-identical (no engine token); native names the path, and the watcher's parser reads it
    lens_cmd = L.watcher_command("r1", out_suffix="s7b", model="gemma4e4b", cells=["none"])
    assert lens_cmd == L.watcher_command("r1", out_suffix="s7b", model="gemma4e4b", cells=["none"], serving_path=LENS)
    assert lens_cmd.endswith("--cells none --poll 120 --max-hours 6") and L.SERVING_PATH_FLAG not in lens_cmd
    native_cmd = L.watcher_command("r1", out_suffix="s7b", model="gemma4e4b", cells=["none"], serving_path=NATIVE)
    assert native_cmd == f"{lens_cmd} {L.SERVING_PATH_FLAG} {NATIVE}"
    assert W.build_parser().parse_args(native_cmd.split()[2:]).serving_path == NATIVE
    assert W.build_parser().parse_args(lens_cmd.split()[2:]).serving_path == LENS
    # THE finding: the pointer lives under the native root; the lens-derived watcher never sees it
    files = dict([_pointer_under(W, "r1", "none", "fc-n", serving_engine=NATIVE)])
    assert next(iter(files)).startswith("tinycap/gen/native/gemma4e4b/none/thinking_off/s7b/watch/")
    vol = _RemoteVolume(files)
    monkeypatch.setattr(W, "_volume", lambda: vol)
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    monkeypatch.setattr(W, "classify", lambda fcid: _observed("terminal"))
    swept = []
    monkeypatch.setattr(W, "sweep_attempt", lambda owner, **kw: (swept.append(owner) or
                                                                 {"verified_empty": True, "ids_terminated": ["sb-1"], "still_live": []}))
    common = dict(run_id="r1", suffix="s7b", model="gemma4e4b", cells=["none"], mode="off", poll_s=0, max_hours=1,
                  state_dir=str(tmp_path), once=True)
    assert W.watch(**common) == W.EXIT_UNVERIFIED                        # the lens root: no pointer, nothing swept
    assert f"no attempt pointer yet under {C.VOL_GEN_ROOT}/gemma4e4b/none/thinking_off/s7b/watch" in capsys.readouterr().out
    assert swept == []
    assert W.watch(**common, serving_engine=NATIVE) == W.EXIT_DONE       # the native root: found, terminal, swept
    out = capsys.readouterr().out
    assert f"serving engine {NATIVE} (out root {C.VOL_GEN_ROOT}/native)" in out and "attempt fc-n TERMINAL" in out
    assert [o.cell for o in swept] == ["none"] and swept[0].function_call_id == "fc-n"
    swept.clear()
    assert W.sweep_now(run_id="r1", suffix="s7b", model="gemma4e4b", cells=["none"], mode="off", state_dir=str(tmp_path),
                       serving_engine=NATIVE) == W.EXIT_DONE and len(swept) == 1
    swept.clear()
    # the printed argv drives the same code path end to end through the watcher's own main
    assert W.main(native_cmd.split()[2:] + ["--once"]) == W.EXIT_DONE and len(swept) == 1
    # the dry run prints the native watcher line with the path (the lens line is the historical one, byte for byte)
    out = L.describe_plan(L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"], controls=[BCB, BFCL200],
                                                                   serving_path=NATIVE), "s7b"),
                          limit=0, inspect_limit=0, samples=1, pass_no=1, run_id="r1", out_suffix="s7b", serving_path=NATIVE)
    assert f"sandbox watcher (arm in the same turn as the launch): {native_cmd}" in out


# --------------------------------------------------------------------------- #
# the ladder scoring standup is a FIXED rung on both engines; the native         #
# launch's meter commands name the engine                                        #
# --------------------------------------------------------------------------- #


class _KvHonestStandup:
    """A Popen double for the scoring WORKER's ladder standup that behaves as ``capctrl_serving_up`` does under
    each value of ``--kv-derived-seqs`` when the container's KV cache backs ONE full-length sequence FEWER than the
    configured S (a scoring container slightly smaller than the probe's): ``on`` comes up at S - 1 with a ``kv_shrink``
    record that HOLDS the shrink rule (the record the launcher's reader re-derives the served identity from); ``off``
    dies with the KV verdict naming S x L and no endpoint (the fixed rung's refusal). A native argv's endpoint carries
    the engine's stamps. The ledger writes its endpoint record (never reached once the standup refuses)."""

    flag = "--kv-derived-seqs"
    argvs: list = []

    def __init__(self, argv, *a, **k):
        self.argv, self.rc = list(argv), None
        joined = " ".join(self.argv[:3])
        if "capctrl_serving_up.py" in joined:
            type(self).argvs.append(self.argv)
            out = Path(self.argv[self.argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            L_, S = int(self.argv[self.argv.index("--max-model-len") + 1]), int(self.argv[self.argv.index("--max-num-seqs") + 1])
            blocks = ((S - 1) * L_ + 1000) // 16                       # a cache that holds S - 1 sequences at L, not S
            report = {"kv_num_cache_groups": 1, "kv_num_gpu_blocks": blocks, "kv_block_size": 16}
            if self.argv[self.argv.index(type(self).flag) + 1] == "on":
                cap = blocks * 16
                shrink = sg.kv_shrink_record(configured_max_num_seqs=S, max_model_len=L_, kv_backed_concurrency=cap / L_,
                                             kv_capacity_tokens=cap, basis="the double's scheduler report")
                native = "lora_native" in self.argv
                (out / "endpoint.json").write_text(json.dumps({
                    "proxy_url": "http://127.0.0.1:1/v1", "routing_table": str(out / "routing_table.json"),
                    "tool_calls": "--tool-calls" in self.argv, "reasoning_parser": None, "agent_loop_entry": True,
                    **(NATIVE_LADDER_EP if native else {}),
                    "geometry": {"max_model_len": L_, "max_num_seqs": S - 1, sg.KV_SHRINK_KEY: shrink}}))
            else:
                k["stdout"].write("VllmServerError: resolved scheduler state does not match the composed no-chunking invariant:\n"
                                  "- worker 0: " + str(vs.kv_coverage_problem(report, max_num_seqs=S, max_model_len=L_)) + "\n")
                k["stdout"].flush()
                self.rc = 1
        elif "usage_ledger" in joined:
            ep = Path(self.argv[self.argv.index("--endpoint-json") + 1])
            ep.write_text(json.dumps({"ledger_url": "http://127.0.0.1:1/v1"}))

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        # lifecycle-faithful: the REAL standup unlinks its ``endpoint.json`` as it
        # tears down (capctrl_serving_up.py, after ``su.tear_down()``), so a launcher that reads the record after
        # ``_tear_down_standup`` finds nothing — the double deletes it exactly where the producer does
        out = Path(self.argv[self.argv.index("--out") + 1])
        (out / "endpoint.json").unlink(missing_ok=True)
        return self.rc if self.rc is not None else 0

    def terminate(self):
        self.rc = 0

    def kill(self):
        self.rc = 0


def test_a_ladder_scoring_standup_is_a_fixed_rung_on_both_engines_and_a_kv_shortfall_writes_no_store(L, tmp_path, monkeypatch):
    """`_standup_cmd` defaults ``kv_derived_seqs=True``; the native rung PROBE disabled
    it but the scoring WORKER did not — a probe certifies gemma at the served S=15, a slightly smaller scoring-container KV
    cache appends a `kv_shrink` step to 14, the worker rewrites its expected identity at 14 and runs the driver there, and
    `tinycap_ladder merge` refuses that store against the S=15 certificate — fail-safe, the whole scoring run spent for
    nothing. Now EVERY ladder scoring standup, lens and native, carries ``--kv-derived-seqs off``: against a standup double
    that honours the flag the way capctrl does (on: up at S - 1 with a holding kv_shrink record; off: the KV verdict, no
    endpoint), the worker ends ``STANDUP_FAILED`` with the shortfall named in `kv_refusal` BEFORE any driver — no rung-2
    store, no resume, no proxy — on the lens ladder at 16 and on the native ladder at the served 14 (serving path and bank
    beside the flag). The pre-fix mechanism is pinned too: with the flag on, the double's endpoint is one the launcher's own
    shrink reader ACCEPTS as a shrink to S - 1 — the identity the worker then drove the run at. Every NON-ladder standup keeps
    the default ``on`` (the lens sidecar golden and `test_tinycap_launcher_port`'s shrink pin stand)."""
    repos, tree = _ladder_tree(tmp_path)
    sdf_cfg = C.model_row("gemma4e4b-sdfbase").cfg                  # the LoRA row's checked-in config (the certificate binds its digest)
    (tree / sdf_cfg).parent.mkdir(parents=True, exist_ok=True)
    (tree / sdf_cfg).write_text((REPO / sdf_cfg).read_text())
    gen = tmp_path / "gen"
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(gen))
    monkeypatch.delenv(C.LENS_STACKED_REDUCTIONS_ENV, raising=False)
    monkeypatch.setattr(L, "client_lock_sha256", lambda tree_: None)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L.subprocess, "Popen", _KvHonestStandup)
    monkeypatch.setattr(_KvHonestStandup, "argvs", [])
    monkeypatch.setattr(_KvHonestStandup, "flag", L.STANDUP_KV_DERIVED_SEQS_FLAG)
    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: pytest.fail(f"the driver ran at a shrunk S: {list(a[0])[:8]}"))
    sel_path = _write_selection(tmp_path / "sel.json", {BFCL_TASK: SEL_F})
    cap = f"{BFCL_TASK}=16384"

    def scored(cfg, carrier, **kw):
        shutil.rmtree(f"/tmp/srv_{cfg['name']}", ignore_errors=True)
        rec = L._run_configuration(cfg, "f6cf462", ladder=carrier, **kw)
        argv = rec["standup_argv"]
        assert argv[argv.index(L.STANDUP_KV_DERIVED_SEQS_FLAG) + 1] == "off", argv          # the fixed rung: KV-derived S OFF
        assert rec["status"] == "STANDUP_FAILED", rec
        assert rec["kv_refusal"] and "KV cache holds" in rec["kv_refusal"] and f"(= {argv[argv.index('--max-num-seqs') + 1]} x 32768)" in rec["kv_refusal"]
        assert "proxy_url" not in rec and "standup_s" not in rec and "kv_shrink" not in rec            # nothing past readiness
        assert not list(Path(cfg["out_dir"]).rglob("manifest.json")) and not (Path(cfg["out_dir"]) / "cells").exists()   # no rung-2 store
        return rec, argv

    # --- the LENS ladder: the certificate admitted 16 (a legal walk 18 -> 16); the worker for `none` at 16 -------------
    ladder = L.ladder_launch_args(cap_override=cap, item_selection=str(sel_path), out_suffix="kvfix")
    (cfg,) = L.ladder_configurations(L.isolate_out_dirs(L.plan_configurations(cells=["none"], modes=["off"], controls=[BFCL200]), "kvfix"), ladder)
    ladder = L.ladder_launch_bound(ladder, [cfg])
    good = _ladder_kw(L, tree, cfg, ladder, gen_root=str(gen))
    cfg16 = dict(cfg, ladder_rung=RP.accepted_block(16))
    rec, argv = scored(cfg16, good)
    assert argv[argv.index("--max-num-seqs") + 1] == "16" and argv[argv.index(L.SERVING_PATH_FLAG) + 1] == L.SERVING_PATH_LENS
    assert L.ADAPTER_BANK_FLAG not in argv and rec["serving_path"] == L.SERVING_PATH_LENS
    # the ONE token that moved on a ladder scoring argv: the same call with the kv-derived-seqs default is byte-identical
    i = argv.index(L.STANDUP_KV_DERIVED_SEQS_FLAG)
    assert argv[i:i + 3] == [L.STANDUP_KV_DERIVED_SEQS_FLAG, "off", "--ready-timeout-s"]
    # the PRE-FIX mechanism, pinned: under `on` the double comes up at 15 with a kv_shrink record the launcher's reader ACCEPTS —
    # the served identity the worker re-derived and drove the run at, which the merge then refused against the S=16 certificate
    on_argv = list(argv)
    on_argv[i + 1] = "on"
    on_argv[on_argv.index("--out") + 1] = str(tmp_path / "srv_on")
    _KvHonestStandup(on_argv, stdout=(tmp_path / "on.log").open("w"))
    ep_on = json.loads((tmp_path / "srv_on" / "endpoint.json").read_text())
    geom16, _ = L.served_geometry(cfg16, 1, str(tree))
    served, shrink, problem = L.served_geometry_of_endpoint(ep_on, geom16)
    assert problem is None and int(served.max_num_seqs) == 15 and shrink["served_max_num_seqs"] == 15 and shrink["configured_max_num_seqs"] == 16
    assert int(L.expected_at_served_seqs(L.expected_identity(cfg16, geom16, pass_no=1, controls=cfg16["controls"], samples=1, tree=str(tree),
                                                             tree_sha=FULL_SHA, reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT),
                                         served.max_num_seqs).served["max_num_seqs"]) == 15
    # --- the NATIVE ladder: a certificate at the served 14 (15 refused by the engine), filed in the native namespace with the
    # wave record; the base cell's scoring worker stands up on the serving path with the bank — the same fixed rung -----
    NATIVE = L.SERVING_PATH_LORA_NATIVE
    n_ladder = L.ladder_launch_args(cap_override=cap, item_selection=str(sel_path), out_suffix="kvfix-n", serving_path=NATIVE)
    n_cfgs = L.ladder_configurations(L.isolate_out_dirs(L.plan_configurations(
        cells=["none", DERIVED], modes=["off"], controls=[BFCL200], model="gemma4e4b-sdfbase", serving_path=NATIVE), "kvfix-n"), n_ladder)
    n_bound = L.ladder_launch_bound(n_ladder, n_cfgs)
    n_row, n_spec = L.row_of(n_cfgs[0]), L.ladder_spec_for(L.row_of(n_cfgs[0]), NATIVE)
    assert n_spec.seqs == (15, 14, 12)
    digests = {DERIVED: {"artifact_sha256": "ab" * 32}}
    prov = L.ladder_provenance(n_row, tree, FULL_SHA, digests, n_bound, probed_cell=DERIVED)
    cert = RP.build_certificate(accepted=RP.accepted_block(14, spec=n_spec),
                                walked=[RP.walked_entry(15, RP.WALK_REFUSED, REFUSAL_15), RP.walked_entry(14, RP.WALK_ACCEPTED)],
                                provenance=prov, probe_run_id="r", standup_s=1.0)
    assert RP.certificate_is_valid(cert, prov) is None and cert["ladder"]["serving_engine"] == NATIVE
    RP.write_certificate(RP.certificate_path(L.native_gen_root(str(gen)), n_row.slug, "kvfix-n", DERIVED), cert)
    ar = L._write_ladder_admission("f6cf462", n_row.slug, "kvfix-n", cert["accepted"], {DERIVED: cert}, NATIVE)
    assert ar["status"] == "OK", ar
    carrier = {"launch": n_bound.as_dict(), "certificates": {DERIVED: cert}, "admission": ar["admission"]}
    base_scored = dict(next(c for c in n_cfgs if c["cell"] == "none"), ladder_rung=dict(cert["accepted"]))
    assert base_scored["out_dir"].startswith(f"{gen}/native/")
    rec_n, argv_n = scored(base_scored, carrier, object_digests=digests)
    j = argv_n.index(L.SERVING_PATH_FLAG)
    assert argv_n[j + 1:j + 4] == [NATIVE, L.ADAPTER_BANK_FLAG, f"{L.ADAPTER_BANK_ROOT}/gemma4e4b"]
    assert argv_n[argv_n.index("--max-num-seqs") + 1] == "14" and argv_n[argv_n.index("--max-model-len") + 1] == "32768"
    assert rec_n["serving_path"] == NATIVE and rec_n[C.SERVED_ENGINE_FIELD] == NATIVE and rec_n["ladder_rung"] == cert["accepted"]
    # the double saw three standups: the worker's two ladder scoring standups, each `off` and each refused once, and the hand-built
    # `on` argv of the pre-fix pin above; the worker's source says why in one place
    flag = L.STANDUP_KV_DERIVED_SEQS_FLAG
    assert [(a[a.index("--max-num-seqs") + 1], a[a.index(flag) + 1]) for a in _KvHonestStandup.argvs] == [("16", "off"), ("16", "on"), ("14", "off")]
    body = LAUNCHER.read_text()
    body = body[body.index("def _run_configuration("):body.index("def run_configuration(")]
    assert "kv_derived_seqs=not rung)" in body and body.count("kv_derived_seqs=") == 1
    # the dry run's described standup follows the worker's rule (a ladder rung off, a plain configuration on)
    dry = LAUNCHER.read_text()
    dry = dry[dry.index("def describe_plan("):]
    assert 'kv_derived_seqs=not bool(cfg.get("ladder_rung"))' in dry


def test_a_native_ladders_meter_commands_name_the_engine_and_the_lens_commands_are_byte_identical(L, tmp_path):
    """The meter takes ``--serving-engine``, but the launcher's generated pre-probe command omitted it, so a NATIVE
    launch's operator step defaulted to the LENS forecast — gemma reserved four walked rungs instead of the three
    served ones, nemotron four instead of its single coalesced rung — which can refuse a valid launch at the GPU-hour
    ceiling and records the wrong ladder in the reservation. Now both native commands end in ``--serving-engine
    lora_native`` (the post-probe one as the explicit cross-check the meter holds the certificates' engine to): parsed
    by the meter's own parser, and RUN through the meter they reserve the SERVED ladder (three rungs per walk, the
    native ladder block recorded); the lens commands carry no flag and are byte-identical to the pinned text."""
    import importlib.util as _ilu
    NATIVE, LENS = L.SERVING_PATH_LORA_NATIVE, L.SERVING_PATH_LENS
    sel_doc = _selection_doc(SEL_IDS)
    sel_doc["cells"] = ["none", DERIVED]           # the meter reads it through `tinycap_ladder.load_selection`: the paired design's cells
    sel_path = tmp_path / "ladder_selection.json"
    sel_path.write_text(json.dumps(sel_doc, indent=1, sort_keys=True))
    caps = C.format_cap_overrides(CAPS)

    def bound_for(serving_path):
        kw = {"serving_path": serving_path} if serving_path == NATIVE else {}
        ladder = L.ladder_launch_args(cap_override=caps, item_selection=str(sel_path), out_suffix="ladder-m", **kw)
        cfgs = L.ladder_configurations(L.plan_configurations(cells=["none", DERIVED], modes=["off"], controls=[BCB, BFCL200],
                                                             model="gemma4e4b-sdfbase", **kw), ladder)
        return L.ladder_launch_bound(ladder, cfgs), L.row_of(cfgs[0])
    n_bound, row = bound_for(NATIVE)
    l_bound, _ = bound_for(LENS)
    n_spec, l_spec = L.ladder_spec_for(row, NATIVE), L.ladder_spec_for(row, LENS)
    assert (n_spec.seqs, l_spec.seqs) == ((15, 14, 12), (18, 16, 14, 12)) and n_bound.probe_cells == l_bound.probe_cells == (DERIVED,)
    native_lines, lens_lines = L.describe_ladder_meter(n_bound, row, n_spec), L.describe_ladder_meter(l_bound, row, l_spec)
    assert len(native_lines) == len(lens_lines) == 2
    spec = _ilu.spec_from_file_location("bcb_bfcl_200_meter_under_test_p2", REPO / "scripts/bcb_bfcl_200_meter.py")
    meter = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = meter
    spec.loader.exec_module(meter)

    def argv_of(line):
        return line.split("python scripts/bcb_bfcl_200_meter.py ", 1)[1].split("   #", 1)[0].split()
    # the NATIVE commands: the flag last, parsed by the meter's parser, the certificate at its native address
    pre_n, post_n = (meter.build_parser().parse_args(argv_of(ln)) for ln in native_lines)
    assert argv_of(native_lines[0])[-2:] == argv_of(native_lines[1])[-2:] == ["--serving-engine", NATIVE]
    assert (pre_n.stage_kind, pre_n.serving_engine, pre_n.unit, pre_n.fitted_cell, pre_n.model) == ("ladder-probe", NATIVE, f"probe:{DERIVED}", [DERIVED], "gemma4e4b")
    assert (post_n.stage_kind, post_n.serving_engine, post_n.unit) == ("ladder-score", NATIVE, "score")
    assert post_n.certificate == [str(RP.certificate_path(L.native_gen_root(C.VOL_GEN_ROOT), row.slug, "ladder-m", DERIVED))]
    assert "worst-case S 12 at 32768" in native_lines[0] and f"(the {NATIVE} engine's SERVED rungs [15, 14, 12])" in native_lines[0]
    assert native_lines[1].endswith("(the flag is held to the engine the certificates' ladder blocks name)")
    # RUN through the meter, the native pre-probe command reserves the SERVED ladder — three standups per walk, the engine and the
    # mapping in the reservation's ladder block; the same command without the flag (the pre-fix text) reserved the LENS walk of four
    ledger = tmp_path / "gpu_ledger.jsonl"
    pre_argv = [("w" if t == "<wave>" else t) for t in argv_of(native_lines[0])]
    assert meter.main([*pre_argv, "--ledger", str(ledger)]) == 0
    entry = [e for e in meter.read_ledger(ledger) if e["unit"] == f"probe:{DERIVED}"][-1]
    fc = entry["inputs"]["forecast"]
    assert fc["probes"][DERIVED]["rungs"] == 3 and fc["ladder"]["seqs"] == [15, 14, 12] and fc["ladder"]["serving_engine"] == NATIVE
    assert fc["ladder"]["served_rungs"] == [{"served": 15, "requested": [18, 16]}, {"served": 14, "requested": [14]}, {"served": 12, "requested": [12]}]
    assert meter.main(["--ledger", str(ledger), "--release", "never ran", "--wave", "w", "--suffix", "ladder-m", "--unit", f"probe:{DERIVED}"]) == 0
    assert meter.main([*pre_argv[:-2], "--ledger", str(ledger)]) == 0                    # the pre-fix command: the lens forecast
    entry = [e for e in meter.read_ledger(ledger) if e["unit"] == f"probe:{DERIVED}"][-1]
    fc = entry["inputs"]["forecast"]
    assert fc["probes"][DERIVED]["rungs"] == 4 and fc["ladder"]["seqs"] == [18, 16, 14, 12] and "serving_engine" not in fc["ladder"]
    # the LENS commands: no flag, and byte-identical to the pinned text (the dry-run test above pins the same shape for `none`)
    pre_l, post_l = (meter.build_parser().parse_args(argv_of(ln)) for ln in lens_lines)
    assert pre_l.serving_engine is None and post_l.serving_engine is None and not any("--serving-engine" in ln for ln in lens_lines)
    sel = str(sel_path)
    cert = RP.certificate_path(C.VOL_GEN_ROOT, row.slug, "ladder-m", DERIVED)
    assert lens_lines == [
        (f"   meter (operator, BEFORE launch): python scripts/bcb_bfcl_200_meter.py --stage-kind ladder-probe --wave <wave> "
         f"--suffix ladder-m --unit probe:{DERIVED} --model gemma4e4b --selection {sel} --fitted-cell {DERIVED}   # pre-probe "
         f"reservation over the ACTUAL selection: n_selected "
         f'{{"{BFCL_TASK}": 2, "{BCB_TASK}": 3}} = 5 item(s), worst-case S 12 at 32768, cap 16384; refuses at the ceiling'),
        (f"   meter (operator, AFTER the probes): python scripts/bcb_bfcl_200_meter.py --stage-kind ladder-score --wave <wave> "
         f"--suffix ladder-m --unit score --model gemma4e4b --selection {sel} --certificate {cert}   # the certified common-S "
         "re-forecast over every final per-cell certificate")]
    assert not str(cert).startswith(f"{C.VOL_GEN_ROOT}/native/") and str(post_n.certificate[0]).startswith(f"{C.VOL_GEN_ROOT}/native/")


# --------------------------------------------------------------------------- #
# the LENS rung probe is a fixed rung too, and a ready endpoint is held to the  #
# engine and the served geometry on BOTH engines                                #
# --------------------------------------------------------------------------- #


class _KvHonestProbe:
    """A Popen double for the LENS rung probe that behaves as ``capctrl_serving_up`` does under each value of
    ``--kv-derived-seqs`` when the container's KV cache backs ``backed`` (17) full-length sequences at the ladder
    window — one FEWER than the identity rung 18: ``on`` at 18 comes up at 17 with a ``kv_shrink`` record that HOLDS the
    shrink rule (the endpoint the pre-fix probe certified the requested 18 over); ``off`` at 18 dies with the KV verdict
    naming 18 x 32,768 and no endpoint (the fixed rung's refusal, the verdict the walk steps down on); a rung the cache
    backs (16) comes up serving exactly it with the lens standup's geometry block (no engine key: lens by era).
    ``endpoint_extra`` is merged over a ready record (the engine-mismatch case); ``serve_at`` makes a ready standup serve
    that S regardless of the flag and record NO shrink (a shrink the standup made anyway)."""

    flag = "--kv-derived-seqs"
    backed = 17
    argvs: list = []
    endpoint_extra: dict = {}
    serve_at: Optional[int] = None

    def __init__(self, argv, *a, **k):
        self.argv, self.rc = list(argv), None
        type(self).argvs.append(self.argv)
        out = Path(self.argv[self.argv.index("--out") + 1])
        L_, S = int(self.argv[self.argv.index("--max-model-len") + 1]), int(self.argv[self.argv.index("--max-num-seqs") + 1])
        blocks = (type(self).backed * L_ + 1000) // 16                  # a cache that holds `backed` sequences at L, not more
        on = self.argv[self.argv.index(type(self).flag) + 1] == "on"
        if type(self).serve_at is not None:
            self._up(out, L_, int(type(self).serve_at), shrink=None)
        elif S <= type(self).backed:
            self._up(out, L_, S, shrink=None)
        elif on:
            cap = blocks * 16
            self._up(out, L_, type(self).backed, shrink=sg.kv_shrink_record(
                configured_max_num_seqs=S, max_model_len=L_, kv_backed_concurrency=cap / L_, kv_capacity_tokens=cap,
                basis="the double's scheduler report"))
        else:
            report = {"kv_num_cache_groups": 1, "kv_num_gpu_blocks": blocks, "kv_block_size": 16}
            k["stdout"].write("VllmServerError: resolved scheduler state does not match the composed no-chunking invariant:\n"
                              "- worker 0: " + str(vs.kv_coverage_problem(report, max_num_seqs=S, max_model_len=L_)) + "\n")
            k["stdout"].flush()
            self.rc = 1

    def _up(self, out, L_, S_served, *, shrink):
        geometry = lens_ladder_endpoint_geometry(S_served, L=L_)
        if shrink is not None:
            geometry[sg.KV_SHRINK_KEY] = shrink
        (out / "endpoint.json").write_text(json.dumps({
            "proxy_url": "http://127.0.0.1:1/v1", "routing_table": str(out / "routing_table.json"), "tool_calls": "--tool-calls" in self.argv,
            "reasoning_parser": None, "agent_loop_entry": True, "geometry": geometry, **type(self).endpoint_extra}))

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        # lifecycle-faithful: the REAL standup unlinks its ``endpoint.json`` as it
        # tears down (capctrl_serving_up.py, after ``su.tear_down()``), so a launcher that reads the record after
        # ``_tear_down_standup`` finds nothing — the double deletes it exactly where the producer does
        out = Path(self.argv[self.argv.index("--out") + 1])
        (out / "endpoint.json").unlink(missing_ok=True)
        return self.rc if self.rc is not None else 0

    def terminate(self):
        self.rc = 0

    def kill(self):
        self.rc = 0


def test_the_lens_rung_probe_is_a_fixed_rung_and_a_ready_endpoint_is_held_to_the_engine_and_the_served_geometry(L, tmp_path, monkeypatch):
    """Under an ordinary KV-capacity shortfall the LENS rung probe still launched
    with ``--kv-derived-seqs on`` (the native probe alone ran ``off``) and checked NOTHING about a ready
    endpoint (geometry and engine were held for native only), so capctrl came up at S - 1 with a holding `kv_shrink`
    record and the walk certified the REQUESTED S off it (reproduced: asked 18, served 17, a certificate accepting 18).
    Scoring is fixed-rung on both engines since bec49631, so that certificate is one production can never
    consume — scoring refuses where the probe never tried the lower rung. Now the fixed-rung policy applies to PROBING
    on both engines and the ready endpoint is held to the engine and the served geometry on both (`rung_problem`, of
    which `native_rung_problem` is the native reading): against a standup double honouring the flag as capctrl does,
    the lens probe's every standup carries ``off``, 18 REFUSES with the KV verdict and the walk steps down to 16, which
    is certified serving exactly 16 (the lens certificate byte-identical: the four-key ladder block); the pre-fix
    mechanism is pinned (under ``on`` the double's endpoint is one the launcher's shrink reader ACCEPTS as a shrink to
    17, and `rung_problem` refuses it as SERVED_GEOMETRY_MISMATCH); a lens ladder whose standup attests `lora_native`
    and a standup that shrank anyway refuse by name and file no certificate. DELIBERATE LENS CHANGE: the lens ladder
    PROBE argv moves by ONE token (`on` -> `off`) so the certificate is consumable by the fixed-rung scoring standup;
    every non-ladder lens argv keeps the default and is byte-identical (the sidecar golden and the shrink pin)."""
    repos, tree = _ladder_tree(tmp_path)
    gen = tmp_path / "gen"
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(gen))
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L.subprocess, "Popen", _KvHonestProbe)
    monkeypatch.setattr(_KvHonestProbe, "argvs", [])
    monkeypatch.setattr(_KvHonestProbe, "flag", L.STANDUP_KV_DERIVED_SEQS_FLAG)
    monkeypatch.setattr(_KvHonestProbe, "endpoint_extra", {})
    monkeypatch.setattr(_KvHonestProbe, "serve_at", None)
    LENS = L.SERVING_PATH_LENS
    sel_path = _write_selection(tmp_path / "ladder_selection.json", SEL_IDS)
    ladder = L.ladder_launch_args(cap_override=C.format_cap_overrides(CAPS), item_selection=str(sel_path), out_suffix="ladder-fixed")
    (cfg,) = L.ladder_configurations(L.plan_configurations(cells=["none"], modes=["off"], controls=[BCB, BFCL200]), ladder)
    ladder = L.ladder_launch_bound(ladder, [cfg])
    assert not ladder.native and ladder.serving_path == LENS and ladder.serving_engine == L.ENGINE_LENS
    rec = L._probe_rung(cfg, "f6cf462", object_digests={}, ladder=ladder, run_id="lens-fixed")
    assert rec["status"] == "OK", rec
    cert = rec["certificate"]
    # the walk: 18 REFUSED by the engine's own verdict (the cache backs 17), 16 accepted — never a certificate at 18
    assert [(w["S"], w["outcome"]) for w in cert["walked"]] == [(18, "refused"), (16, "accepted")]
    assert 17 * 32_768 <= cert["walked"][0]["kv_tokens"] < 18 * 32_768
    assert cert["accepted"] == RP.accepted_block(16) and cert["ladder"] == RP.ladder_spec("gemma4e4b").block()   # the lens block, byte-identical
    assert tuple(cert["ladder"]) == RP.LADDER_BLOCK_FIELDS and "serving_engine" not in cert["ladder"]
    assert rec["walked"][1]["served"] == {"max_num_seqs": 16, "max_model_len": 32_768, "configured_max_num_seqs": 16, "kv_shrink": None}
    assert "served" not in rec["walked"][0]                                          # a refused rung served nothing
    path = RP.certificate_path(str(gen), C.MODEL_SLUG, "ladder-fixed", "none")
    assert path.is_file() and RP.read_certificate(path)[0] == cert
    # EVERY lens probe standup: --kv-derived-seqs off (THE one moved token), on the lens path, no adapter bank
    argvs = list(_KvHonestProbe.argvs)
    assert [int(a[a.index("--max-num-seqs") + 1]) for a in argvs] == [18, 16]
    for a in argvs:
        i = a.index(L.STANDUP_KV_DERIVED_SEQS_FLAG)
        assert a[i:i + 3] == [L.STANDUP_KV_DERIVED_SEQS_FLAG, "off", "--ready-timeout-s"], a
        assert a[a.index(L.SERVING_PATH_FLAG) + 1] == LENS and L.ADAPTER_BANK_FLAG not in a
        assert a.count(L.STANDUP_KV_DERIVED_SEQS_FLAG) == 1 and a[a.index("--max-model-len") + 1] == "32768"
    # the PRE-FIX mechanism, pinned: under `on` the double comes up at 17 with a kv_shrink record the launcher's own
    # shrink reader ACCEPTS as a shrink to 17 — the endpoint the old probe (which read nothing) certified 18 over.
    # The probe's reader now refuses it by name: the certificate would name an S the engine did not serve
    geom18, _ = L.served_geometry(dict(cfg, ladder_rung=RP.accepted_block(18)), 1, str(tree))
    assert int(geom18.max_num_seqs) == 18
    on_argv = list(argvs[0])
    on_argv[on_argv.index(L.STANDUP_KV_DERIVED_SEQS_FLAG) + 1] = "on"
    on_argv[on_argv.index("--out") + 1] = str(tmp_path / "srv_on")
    (tmp_path / "srv_on").mkdir()
    _KvHonestProbe(on_argv, stdout=(tmp_path / "on.log").open("w"))
    ep_on = json.loads((tmp_path / "srv_on" / "endpoint.json").read_text())
    served, shrink, problem = L.served_geometry_of_endpoint(ep_on, geom18)
    assert problem is None and int(served.max_num_seqs) == 17 and shrink["configured_max_num_seqs"] == 18 and shrink["served_max_num_seqs"] == 17
    assert L.endpoint_engine(ep_on) == L.ENGINE_LENS
    endpoint, problem = L.rung_problem(tmp_path / "srv_on" / "endpoint.json", geom18, 18, serving_path=LENS)
    assert endpoint == ep_on and problem is not None and problem[0] == C.REC_SERVED_GEOMETRY_MISMATCH
    assert "the standup served 32768 x 17 but this rung is 32768 x 18 (asked as S 18 with KV-derived S off)" in problem[1]
    assert "the certificate would name an S the engine did not serve" in problem[1]
    # a lens ladder whose ready standup attests the OTHER engine: SERVING_ENGINE_MISMATCH, no certificate
    (tmp_path / "srv_native").mkdir()
    (tmp_path / "srv_native" / "endpoint.json").write_text(json.dumps({**ep_on, **NATIVE_LADDER_EP}))
    _, problem = L.rung_problem(tmp_path / "srv_native" / "endpoint.json", geom18, 18, serving_path=LENS)
    assert problem == (L.REC_SERVING_ENGINE_MISMATCH, "the standup attests engine='lora_native' but this ladder runs under "
                       f"{L.SERVING_PATH_FLAG} lens ('lens') — a certificate is evidence about ONE engine's footprint")
    monkeypatch.setattr(_KvHonestProbe, "endpoint_extra", dict(NATIVE_LADDER_EP))
    bad = L._probe_rung(cfg, "f6cf462", object_digests={}, ladder=ladder, run_id="lens-native-ep", reprobe=True)
    assert bad["status"] == L.REC_SERVING_ENGINE_MISMATCH == "SERVING_ENGINE_MISMATCH" and "certificate" not in bad
    assert "rung S=16: the standup attests engine='lora_native'" in bad["error"] and bad["walked"][-1]["outcome"] == "failed"
    assert not path.is_file()                                                        # the OK certificate was superseded; none replaced it
    # a standup that shrank ANYWAY (up at 17 with no record, asked 18 with KV-derived S off): SERVED_GEOMETRY_MISMATCH, by name
    monkeypatch.setattr(_KvHonestProbe, "endpoint_extra", {})
    monkeypatch.setattr(_KvHonestProbe, "serve_at", 17)
    shrunk = L._probe_rung(cfg, "f6cf462", object_digests={}, ladder=ladder, run_id="lens-shrunk", reprobe=True)
    assert shrunk["status"] == C.REC_SERVED_GEOMETRY_MISMATCH and "certificate" not in shrunk
    assert "rung S=18: the standup served max_num_seqs=17, not the resolved 18, and records no kv_shrink" in shrunk["error"]
    assert [w["outcome"] for w in shrunk["walked"]] == ["failed"] and not path.is_file()
    # the native reading is the same function (its callers and messages unchanged); the pure builder's default stands
    assert L.native_rung_problem.__wrapped__ if hasattr(L.native_rung_problem, "__wrapped__") else True
    src = inspect.getsource(L.native_rung_problem)
    assert "return rung_problem(endpoint_path, geom, S, serving_path=SERVING_PATH_LORA_NATIVE, record=record)" in src
    body = inspect.getsource(L._probe_rung)
    assert body.count("kv_derived_seqs=False)") == 1 and "kv_derived_seqs=not native" not in body
    assert "if ready and native:" not in body and 'rung_problem(srv / "endpoint.json", geom, S, serving_path=ladder.serving_path,' in body
    assert inspect.signature(L._standup_cmd).parameters["kv_derived_seqs"].default is True     # every non-ladder argv: byte-identical



# --------------------------------------------------------------------------- #
# tinycap-shared-standup §1.10 (module E): the FULL leg is sandboxed, and the    #
# watcher's --row-dir mode for a row engine's shared FunctionCall id            #
# --------------------------------------------------------------------------- #


def test_the_full_bcb_leg_is_a_sandboxed_row_task_so_the_solo_full_path_writes_its_pointer(L):
    """The full BigCodeBench leg executes in owned sandboxes like the 200 leg, but was never in
    `SANDBOXED_ROW_TASKS` — no full store carried a pointer or a sweep certificate, so no watcher could certify a
    full wave. Now it is; the solo full path's worker writes the pointer through the SAME `if sandboxed:` block."""
    assert L.SANDBOXED_ROW_TASKS == (C.inspect_task("bigcodebench_200"), C.inspect_task("bigcodebench_full"))
    assert L.sandboxed_controls(["bigcodebench_full"], "off") is True
    assert L.sandboxed_controls(["bigcodebench_full", "bfcl_full"], "off") is True
    assert L.sandboxed_controls(["bfcl_full"], "off") is False and L.sandboxed_controls(["bfcl_full", "gsm8k"], "off") is False
    assert L.sandboxed_controls([BCB], "off") is True                                   # the 200 leg, unchanged
    src = LAUNCHER.read_text()
    worker = src[src.index("def _run_configuration("):src.index("def _producer_conflict_error(")]
    assert 'sandboxed = sandboxed_controls(driven, cfg["mode"])' in worker and "if sandboxed:" in worker
    assert "write_attempt_pointer(" in worker[worker.index("if sandboxed:"):]


def _row_pointer(W, run_id, fcid, cells, *, row_dir, phase="coupled", standup_id="sa-1", row="gemma4e4b-f22s1", started_at=1.0):
    rel = row_dir[len("/vol/"):]
    doc = {"run_id": run_id, "cell": None, "function_call_id": fcid, "name": f"row-{row}", "model": row, "mode": "off",
           "out_dir": row_dir, "controls": ["bigcodebench_full", "bfcl_full"], "tree_sha": "a" * 40, "sandbox_record": None,
           "sandbox_app_name": C.BCB_SANDBOX_APP_NAME, "started_at": started_at,
           "phase": phase, "cells": list(cells), "row": row, "row_attempt": {"standup_id": standup_id},
           "records": {c: f"/vol/tinycap/gen/native/{row}/{c}/thinking_off/s/watch/sandboxes_{fcid}.jsonl" for c in cells}}
    return f"{rel}/watch/attempt_{fcid}.json", json.dumps(doc)


def _verify_pointer(W, run_id, fcid, cell, *, row_dir, generation_fcid="fc-row", started_at=2.0):
    rel = row_dir[len("/vol/"):]
    doc = {"run_id": run_id, "cell": cell, "function_call_id": fcid, "name": f"verify-{cell}", "model": "gemma4e4b-f22s1",
           "mode": "off", "out_dir": row_dir, "controls": ["bigcodebench_full"], "tree_sha": "a" * 40,
           "sandbox_record": f"{row_dir}/watch/sandboxes_{fcid}__{cell}.jsonl", "sandbox_app_name": C.BCB_SANDBOX_APP_NAME,
           "started_at": started_at, "phase": "verify", "standup_id": "sa-1", "generation_fcid": generation_fcid}
    return f"{rel}/watch/attempt_{fcid}.json", json.dumps(doc)


def _row_cert(W, row_dir, fcid, cell, owner_cell=None, verified=True):
    rel = row_dir[len("/vol/"):]
    cert = {"owner": {"run_id": "r1", "cell": owner_cell or cell, "function_call_id": fcid}, "verified_empty": verified,
            "ids_terminated": ["sb-1"] if verified else [], "still_live": [] if verified else ["sb-9"], "could_not_terminate": [],
            "certificate": "sweep"}
    return f"{rel}/watch/{W.row_sweep_certificate_name(fcid, cell)}", json.dumps(cert)


ROW_DIR = "/vol/tinycap/gen/rows/gemma4e4b-f22s1/bcbw-gemma4e4b_s1-abcdef0-shared"
ROW_CELLS = ("none", DERIVED, "prompted__si_constitution")


def test_the_watcher_row_dir_mode_classifies_a_declared_row_once_and_sweeps_every_cell_owner(monkeypatch, tmp_path, capsys):
    """The synthetic FULL row: three cells share `fc-row` under a DECLARED row pointer (cells listed, a
    row_attempt block) — classified ONCE per tick, on TERMINAL every cell's owner swept by its own exact tuple (one
    certificate each; the container's own per-cell certificate accepted first); two verifier pointers, `fc-v1`
    terminated with one live owned sandbox (swept and certified) and `fc-v2` still running (pending) — so the tick is
    NOT done until `fc-v2` terminates."""
    W = _load_watcher()
    assert W.row_sweep_certificate_name("fc-row", DERIVED) == f"sweep_fc-row__{DERIVED}.json"
    files = dict([_row_pointer(W, "r1", "fc-row", ROW_CELLS, row_dir=ROW_DIR),
                  _verify_pointer(W, "r1", "fc-v1", "none", row_dir=ROW_DIR),
                  _verify_pointer(W, "r1", "fc-v2", DERIVED, row_dir=ROW_DIR),
                  # the container's own certificate for ONE row cell (exact owner) — accepted without a sweep
                  _row_cert(W, ROW_DIR, "fc-row", "none"),
                  # a certificate for another cell that names the WRONG owner cell: not admissible, that cell is swept
                  _row_cert(W, ROW_DIR, "fc-row", DERIVED, owner_cell="none"),
                  # a pointer of ANOTHER run in the same row dir: ignored
                  _row_pointer(W, "r0", "fc-old", ("none",), row_dir=ROW_DIR)])
    vol = _RemoteVolume(files)
    monkeypatch.setattr(W, "_volume", lambda: vol)
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    states = {"fc-row": "terminal", "fc-v1": "terminal", "fc-v2": "pending"}
    classified = []
    monkeypatch.setattr(W, "classify", lambda fcid: (classified.append(fcid) or _observed(states[fcid])))
    swept = []

    def _sweep(owner, **kw):
        swept.append(owner)
        # fc-v1's verifier left one live owned sandbox: terminated by the sweep, verified empty on relist
        return {"verified_empty": True, "ids_terminated": ["sb-live"] if owner.function_call_id == "fc-v1" else [], "still_live": []}
    monkeypatch.setattr(W, "sweep_attempt", _sweep)
    rc = W.watch_row_dir(run_id="r1", row_dir=ROW_DIR, poll_s=0, max_hours=1, state_dir=str(tmp_path), once=True)
    out = capsys.readouterr().out
    assert rc == W.EXIT_UNVERIFIED and vol.reloads == 0
    # the row's call classified ONCE for its three cells; each verifier once; the other run's pointer never
    assert sorted(classified) == ["fc-row", "fc-v1", "fc-v2"]
    # every owner of the terminal calls swept SEPARATELY by its own tuple — except the one the container certified
    assert sorted((o.cell, o.function_call_id) for o in swept) == sorted([(DERIVED, "fc-row"), ("prompted__si_constitution", "fc-row"),
                                                                          ("none", "fc-v1")])
    assert all(o.run_id == "r1" for o in swept)
    assert "none: the container's own sweep certified clean" in out
    assert f"{DERIVED}: no admissible container certificate (certificate owner" in out         # the wrong-owner cert refused
    assert f"verify {DERIVED}: attempt fc-v2 PENDING" in out and "standup sa-1" in out
    # fc-v2 terminates: the tick certifies its owner and the run is DONE (nothing re-swept: the row's owners are remembered
    # within a watch; a fresh --once tick re-reads the certificates it can and sweeps the rest again — idempotent)
    states["fc-v2"] = "terminal"
    swept.clear()
    classified.clear()
    assert W.watch_row_dir(run_id="r1", row_dir=ROW_DIR, poll_s=0, max_hours=1, state_dir=str(tmp_path), once=True) == W.EXIT_DONE
    assert (DERIVED, "fc-v2") in [(o.cell, o.function_call_id) for o in swept] and sorted(classified) == ["fc-row", "fc-v1", "fc-v2"]
    assert "DONE: every attempt terminal and every owner certified" in capsys.readouterr().out
    # an UNDECLARED row pointer (no cells yet / no row_attempt block) is named as pending, never classified or swept
    undeclared = dict(files)
    p, doc = _row_pointer(W, "r1", "fc-row", ROW_CELLS, row_dir=ROW_DIR)
    undeclared[p] = json.dumps({**json.loads(doc), "row_attempt": None})
    monkeypatch.setattr(W, "_volume", lambda: _RemoteVolume(undeclared))
    classified.clear()
    swept.clear()
    states["fc-v1"] = states["fc-v2"] = "pending"
    assert W.watch_row_dir(run_id="r1", row_dir=ROW_DIR, poll_s=0, max_hours=1, state_dir=str(tmp_path), once=True) == W.EXIT_UNVERIFIED
    assert "fc-row" not in classified and not swept and "row attempt fc-row carries no row_attempt block" in capsys.readouterr().out
    # --sweep-now over the row dir: every owner of every declared pointer once
    monkeypatch.setattr(W, "_volume", lambda: vol)
    swept.clear()
    assert W.sweep_now_row_dir(run_id="r1", row_dir=ROW_DIR, state_dir=str(tmp_path)) == W.EXIT_DONE
    assert sorted((o.cell, o.function_call_id) for o in swept) == sorted([(c, "fc-row") for c in ROW_CELLS] + [("none", "fc-v1"), (DERIVED, "fc-v2")])
    # a standup that never declared readiness: the terminal row is swept all the same (nothing was served), and said
    never_ready = dict(files)
    never_ready[p] = json.dumps({**json.loads(doc), "row_attempt": {"standup_id": None}})
    monkeypatch.setattr(W, "_volume", lambda: _RemoteVolume(never_ready))
    states.update({"fc-row": "terminal", "fc-v1": "terminal", "fc-v2": "terminal"})
    swept.clear()
    assert W.watch_row_dir(run_id="r1", row_dir=ROW_DIR, poll_s=0, max_hours=1, state_dir=str(tmp_path), once=True) == W.EXIT_DONE
    assert "the standup never declared readiness" in capsys.readouterr().out and len(swept) == 4


def test_the_watcher_row_dir_mode_still_refuses_an_undeclared_shared_id(monkeypatch, tmp_path, capsys):
    """The rule stands in the row dir: two PER-CELL pointers sharing one FunctionCall id (no `cells` declaration) are
    never classified or swept — and the per-cell mode is untouched (its own test above)."""
    W = _load_watcher()
    rel = ROW_DIR[len("/vol/"):]

    def per_cell(cell, fcid, name):
        return f"{rel}/watch/attempt_{name}.json", json.dumps(
            {"run_id": "r1", "cell": cell, "function_call_id": fcid, "started_at": 1.0, "phase": "verify"})
    a = per_cell("none", "fc-map", "fc-map-a")
    b = per_cell(DERIVED, "fc-map", "fc-map-b")
    files = dict([a, b])                             # two pointer FILES whose function_call_id is one call
    vol = _RemoteVolume(files)
    monkeypatch.setattr(W, "_volume", lambda: vol)
    monkeypatch.setattr(W, "sandbox_app_id", lambda *a, **k: "ap-1")
    monkeypatch.setattr(W, "classify", lambda fcid: pytest.fail(f"classified {fcid}"))
    monkeypatch.setattr(W, "sweep_attempt", lambda owner, **kw: pytest.fail(f"swept {owner}"))
    rc = W.watch_row_dir(run_id="r1", row_dir=ROW_DIR, poll_s=0, max_hours=1, state_dir=str(tmp_path), once=True)
    out = capsys.readouterr().out
    assert rc == W.EXIT_UNVERIFIED and "attempt fc-map is shared with" in out and "NOT swept" in out
    assert W.undeclared_shared_cells([json.loads(a[1]), json.loads(b[1])]) == sorted(["none", DERIVED])
    row_p = json.loads(_row_pointer(W, "r1", "fc-row", ROW_CELLS, row_dir=ROW_DIR)[1])
    assert W.undeclared_shared_cells([row_p]) is None                                  # declared: not a refusal
    assert W.undeclared_shared_cells([row_p, json.loads(a[1])]) is not None             # a row pointer beside a per-cell one: refused
    assert W.pointer_owners(row_p) == [scope.Owner("r1", c, "fc-row") for c in ROW_CELLS]
    assert W.row_pointer_undeclared({**row_p, "cells": []}) == "lists no cells"
    assert W.row_pointer_undeclared({**row_p, "cells": ["none", "none"]}).startswith("lists a cell twice")
    assert W.sweep_now_row_dir(run_id="r1", row_dir=ROW_DIR, state_dir=str(tmp_path)) == W.EXIT_UNVERIFIED
    # the CLI: --row-dir needs --run-id and reads no --cells; the per-cell spelling is byte-identical
    ap = W.build_parser()
    a_ = ap.parse_args(["--run-id", "r1", "--row-dir", ROW_DIR])
    assert a_.row_dir == ROW_DIR and a_.cells == ""
    called = {}
    monkeypatch.setattr(W, "watch_row_dir", lambda **kw: called.update(kw) or W.EXIT_DONE)
    assert W.main(["--run-id", "r1", "--row-dir", ROW_DIR, "--once"]) == W.EXIT_DONE
    assert called["row_dir"] == ROW_DIR and called["run_id"] == "r1" and called["once"] is True
    with pytest.raises(SystemExit, match="--run-id is required with --row-dir"):
        W.main(["--row-dir", ROW_DIR])
