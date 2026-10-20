"""The decode-policy redirect: every production sampling entry point now builds its numbers from
the committed export `configs/registry/decode_policy.json`, and NOTHING re-keyed.

* `sampling_literals_before.json` is the ORACLE: the literal tables as they stood at commit 275b7073, BEFORE any
  entry point read the export (generated once by `gen_sampling_literals_before.py`, never regenerated). Every
  (mode, model_key) row, its `sampling_identity()` / `sampling_identity_for_mode()` block, the four Qwen vendor
  rows, both server defaults and their `--override-generation-config` JSON, both protocol rows, the misalignment
  runner's samplers, BoN's harness default and the target-decode boundary must equal the oracle — values AND
  None-ness, byte-identical identity blocks.
* Changing the export (a modified copy under $SISTEER_DECODE_POLICY) changes the ACTUAL request parameters the
  drivers send — `contracts.sampling_for`, a `ProxyGenerateEngine` chat body, BoN's request body, the runner's vendor
  row, the composed vLLM server default, the target-decode boundary — without editing any other preset.
* No production module imports the oracle fixture.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from sisteer.capeval.misalign import runner as MR
from sisteer.capeval.tiers import inspect_served
from sisteer.capeval.tinycap import contracts as C
from sisteer.eval.attacks import bon_served as BON
from sisteer.eval.attacks import target_decode as TD
from sisteer.capeval.serving import geometry as sg
from sisteer.registry import decode_policy as DP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = os.path.join(ROOT, "tests", "fixtures", "registry", "sampling_literals_before.json")
EXPORT = os.path.join(ROOT, "configs", "registry", "decode_policy.json")
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}


@pytest.fixture(scope="module")
def oracle():
    with open(ORACLE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _row(r: C.SamplingRow) -> dict:
    return {k: getattr(r, k) for k in C.SAMPLING_ROW_FIELDS}


# --------------------------------------------------------------------------------------------
# equality with the oracle, per entry point
# --------------------------------------------------------------------------------------------


def test_every_mode_table_reproduces_the_legacy_literals_and_identities(oracle):
    tasks = oracle["identity_tasks"]
    n = 0
    for mode, tab in oracle["modes"].items():
        table = C.SAMPLING_TABLE_BY_MODE[mode]
        # every legacy model is still served with its literal row, none lost; the ONLY keys the oracle does not know are the
        # onboarded pipeline-test models (geometry.ONBOARDED_MODEL_KEYS -- their GSM8K front door's thinking_card rows;
        # pinned in tests/test_gsm8k_thinking_card_row_test_models.py), which post-date 275b7073
        added = set(table) - set(tab["rows"])
        assert set(tab["rows"]) <= set(table) and added <= set(sg.ONBOARDED_MODEL_KEYS), (mode, sorted(added))
        for key, ent in tab["rows"].items():
            r = table[key]
            assert _row(r) == ent["row"], (mode, key)                             # values, None-ness AND the source prose
            for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
                want = ent["row"][k]
                assert type(getattr(r, k)) is type(want), (mode, key, k)       # float stays float, int stays int
            assert C.sampling_identity(r, tasks) == ent["identity"], (mode, key)
            assert C.sampling_identity_for_mode(mode, key, list(tasks), tasks) == ent["identity_for_mode"], (mode, key)
            n += 1
        # the refusals are unchanged: a model key with no mapping row still refuses under that mode
        for key in tab["refuses"]:
            with pytest.raises((C.SamplingUnverified, ValueError)):
                C.sampling_for(key, mode)
    assert n == 19


def test_greedy_row_and_temperature(oracle):
    g = oracle["greedy"]
    assert _row(C.GREEDY_SAMPLING) == g["row"]
    assert C.sampling_identity(C.GREEDY_SAMPLING, oracle["identity_tasks"]) == g["identity"]
    assert C.sampling_identity_for_mode("greedy", "gemma4e4b", list(oracle["identity_tasks"]), oracle["identity_tasks"]) == g["identity_for_mode"]
    assert C.GREEDY_TEMPERATURE == g["GREEDY_TEMPERATURE"] == inspect_served.GREEDY_TEMPERATURE == TD.GREEDY_TEMPERATURE
    assert C.sampling_for("nemotron3nano4b", "greedy") is C.GREEDY_SAMPLING


def test_qwen_vendor_rows_and_server_default(oracle):
    assert {k: _row(v) for k, v in C.QWEN35_VENDOR_ROWS.items()} == oracle["qwen35_vendor_rows"]
    assert sorted(C.SERVER_DEFAULT_SAMPLING_BY_MODEL) == sorted(oracle["server_default"])
    for hf, ent in oracle["server_default"].items():
        row = C.SERVER_DEFAULT_SAMPLING_BY_MODEL[hf]
        assert _row(row) == ent["row"]
        assert C.server_default_generation_config(row) == ent["override_generation_config"]
        assert row is C.QWEN35_VENDOR_ROWS["instruct_general"]                   # the identity pin the launcher tests rely on
    assert list(C.SERVER_DEFAULTABLE_SAMPLING_KEYS) == oracle["server_defaultable_keys"]


def test_protocol_rows(oracle):
    P = oracle["protocol"]
    assert C.BFCL_PROTOCOL_TEMPERATURE == P["BFCL_PROTOCOL_TEMPERATURE"] and C.PROTOCOL_TOP_P == P["PROTOCOL_TOP_P"] and C.PROTOCOL_TOP_K == P["PROTOCOL_TOP_K"]
    assert dict(C.SAMPLING_CLAMP_OVERRIDE) == P["SAMPLING_CLAMP_OVERRIDE"] and C.SAMPLING_CLAMP_OVERRIDE_ARG == P["SAMPLING_CLAMP_OVERRIDE_ARG"]
    # the oracle's rows are unchanged; capeval-full-tasks (2026-09-23) ADDED the two FULL siblings, each equal
    # to its 200 sibling's row (the protocol is the eval's, not the item set's) — nothing re-keyed
    assert set(P["tasks"]) <= set(C.PROTOCOL_SAMPLING_BY_TASK)
    added = sorted(set(C.PROTOCOL_SAMPLING_BY_TASK) - set(P["tasks"]))
    assert added == ["inspect/bfcl_full", "inspect/bigcodebench_full"]
    assert C.PROTOCOL_SAMPLING_BY_TASK["inspect/bigcodebench_full"] == C.PROTOCOL_SAMPLING_BY_TASK["inspect/bigcodebench_200"]
    assert C.PROTOCOL_SAMPLING_BY_TASK["inspect/bfcl_full"] == C.PROTOCOL_SAMPLING_BY_TASK["inspect/bfcl_200"]
    for task, ent in P["tasks"].items():
        pr = C.PROTOCOL_SAMPLING_BY_TASK[task]
        assert _row(pr.executed) == ent["executed"] and pr.protocol_temperature == ent["protocol_temperature"] and pr.executed_reason == ent["executed_reason"]
        assert _row(C.sampling_for_task(task, "protocol", None)) == ent["executed"]
    tasks = {t: 1 for t in P["tasks"]}          # the oracle's identity is over ITS tasks (the 200 legs)
    assert C.sampling_identity_for_mode("protocol", None, list(tasks), tasks) == P["identity_for_mode"]


def test_card_dicts_modes_and_unverified_keys(oracle):
    assert {"_GEMMA_CARD": dict(C._GEMMA_CARD), "_LLAMA_CARD": dict(C._LLAMA_CARD), "_QWEN35_CARD": dict(C._QWEN35_CARD)} == oracle["card_dicts"]
    assert list(C.SAMPLING_MODES) == oracle["sampling_modes"]
    # the unverified-sampling keys are the oracle's (generated once at 275b7073): an onboarded model that decoded only from the registry's
    # decode-policy rows was pinned here by name while it was in the tree (ORACLEPIN); the public tree carries none
    assert sorted(C.SAMPLING_UNVERIFIED_MODEL_KEYS) == sorted(oracle["sampling_unverified_model_keys"])


def test_misalign_runner_samplers(oracle):
    M = oracle["misalign_runner"]
    vendor = {f"{m}|{t}": dict(v) for (m, t), v in MR.VENDOR_SAMPLING.items()}
    # the oracle's two Qwen rows survive verbatim. The ruling of 2026-09-21 (decode-vendor-default-all-evals-2026-09-21: the vendor preset
    # is the default for EVERY family, the agentic T = 1.0 default was a misinterpretation) ADDED a `<model>|agentic|<thinking>|vendor` row for
    # every other served model (+ Llama's not_available twin), each restating the model's static vendor preset -- an upstream addition, not a
    # redirect effect, so the oracle is not regenerated for it: the additions are pinned by name and by value here
    assert {k: vendor[k] for k in M["VENDOR_SAMPLING"]} == M["VENDOR_SAMPLING"]
    assert {k: v for k, v in vendor.items() if k not in M["VENDOR_SAMPLING"]} == {
        "google/gemma-4-E4B-it|off": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16|off": {"temperature": 1.0, "top_p": 0.95},
        "NousResearch/Meta-Llama-3.1-8B-Instruct|off": {"temperature": 0.6, "top_p": 0.9},
        "NousResearch/Meta-Llama-3.1-8B-Instruct|not_available": {"temperature": 0.6, "top_p": 0.9},
        "zai-org/GLM-4.6V-Flash|off": {"temperature": 0.8, "top_p": 0.6, "top_k": 2, "repetition_penalty": 1.1},
        # the paper-companion-repo PIPELINE-TEST models (2026-09-29; configs/registry/decode_policy_seed.yaml ruling
        # decode-qwen35-35b-a3b-shipped-generation-config-2026-09-29): each `<model>|agentic|off|vendor` row states ITS shipped
        # generation_config.json (the 35B-A3B ships one, unlike the 4B; gemma-4-31B-it's file is byte-identical to the E4B's) --
        # pinned here by name and value like the 2026-09-21 additions; tests/test_registry_seeds_test_models.py owns their shape
        "Qwen/Qwen3.5-35B-A3B|off": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "google/gemma-4-31B-it|off": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    }
    assert MR.resolve_sampling("t1", 1.0, None) == M["t1_default"]
    assert MR.resolve_sampling("vendor", 1.0, {"base_model": "Qwen/Qwen3.5-4B", "thinking_mode": "off"}) == M["vendor_qwen4b"]
    assert MR.build_payload("m", [{"role": "user", "content": "x"}], max_tokens=1)["temperature"] == M["build_payload_default_temperature"]
    assert MR.T1_TEMPERATURE == M["build_payload_default_temperature"]
    # gemma-4-E4B has a vendor row since 2026-09-21 (its static vendor preset); a model with no `<model>|agentic|<thinking>|vendor` row
    # (the E2B reference model) is still refused by name -- a row is never guessed
    assert MR.vendor_sampling_for({"base_model": "google/gemma-4-E4B-it", "thinking_mode": "off"}) == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    with pytest.raises(MR.MisalignRunnerError, match="no frozen vendor sampling row"):
        MR.vendor_sampling_for({"base_model": "google/gemma-4-E2B-it", "thinking_mode": "off"})


def test_bon_harness_default_and_target_decode(oracle):
    # THE ORACLE'S BoN ROW IS THE GRANDFATHERED T=1 (the pre-redirect literal). Since the rule of 2026-09-17
    # it is no longer the DEFAULT — best-of-N draws at each served model's vendor preset, and a caller that names no
    # model is refused — but it is still exactly what the NON-DEFAULT `<model>|bon|off|grandfathered_T1` rows state,
    # resolvable BY NAME (the stores drawn under it are read that way): the oracle's numbers survive, re-keyed. Superseded by
    # the vendor rows from 2026-09-17 to 2026-10-14; LIVE (not superseded, still non-default) since the rule of 2026-10-14
    # (decode-bon-canonical-T1-live-non-default-2026-10-14: canonical best-of-N stays the T = 1 instrument).
    for m in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b", "glm46vflash"):
        assert BON.bon_sampler(m, BON.BON_HARNESS_DEFAULT) == oracle["bon_served"]["sent_sampler_default"]
        t1 = BON.bon_policy(m, BON.BON_HARNESS_DEFAULT)
        assert t1.superseded_by_policy_id is None and t1.is_default == 0 and t1.mode == "grandfathered_T1"
        assert BON.bon_policy(m).policy_id != t1.policy_id and BON.bon_policy(m).mode == BON.BON_VENDOR_PRESET     # the default is unchanged
    assert BON._sent_sampler(None, BON.BON_HARNESS_DEFAULT) == oracle["bon_served"]["sent_sampler_default"]
    assert BON._preset_label(None, BON.BON_HARNESS_DEFAULT) == oracle["bon_served"]["preset_label_default"] == BON.BON_HARNESS_DEFAULT
    assert BON.bon_sampler("gemma4e4b", BON.BON_HARNESS_DEFAULT)["temperature"] == oracle["bon_served"]["generate_one_detailed_default_temperature"]
    # ... and the model-independent default is GONE: no constant, and a nameless call refuses
    assert not hasattr(BON, "HARNESS_DEFAULT_SAMPLER") and not hasattr(BON, "HARNESS_DEFAULT_TEMPERATURE")
    with pytest.raises(BON.BonServedError):
        BON._sent_sampler(None)
    assert BON._preset_label(None, None, "gemma4e4b") == BON.BON_VENDOR_PRESET
    T = oracle["target_decode"]
    # the oracle's three WIRE-level modes are unchanged and still lead MODES; main appended the config-level
    # VENDOR_PRESET on 2026-09-16 (the multi-turn campaign default), after the oracle was generated -- an upstream mode
    # addition, not a redirect effect, so the oracle is not regenerated for it
    assert list(TD.MODES)[:3] == T["modes"] and list(TD.MODES) == T["modes"] + [TD.VENDOR_PRESET]
    assert TD.decode_mode(None) == T["decode_mode_None"] and TD.decode_mode(0.0) == T["decode_mode_0"] and TD.decode_mode(0.6) == T["decode_mode_0p6"]
    # the registry's default multi-turn row sends nothing (GENERATION_CONFIG) on every served model but Qwen, whose default
    # is the vendor-preset TARGET row since 2026-09-17
    for m in ("gemma4e4b", "nemotron3nano4b", "llama8b", "glm46vflash"):
        assert TD.campaign_default_temperature(m) is None, m
    assert TD.campaign_default_temperature("qwen35_4b") == 0.7


def test_resolution_api_and_refusals():
    pf = DP.load(EXPORT)
    # aliases: legacy key, HF repo, model_id
    for name in ("gemma4e4b", "google/gemma-4-E4B-it", "gemma4e4b_it"):
        assert pf.resolve(name, mode="card").policy_id == "gemma4e4b_it|capability|off|card@v1"
    # eval row beats category row; the category is inferred from the eval
    assert pf.resolve("qwen35_4b_it", eval_id="gsm8k", thinking_mode="on").policy_id == "qwen35_4b_it|gsm8k|on|thinking_card@v1"
    assert pf.resolve("qwen35_4b_it", eval_id="math500", thinking_mode="on").request_params() == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
    assert pf.resolve("llama8b_it", eval_id="gsm8k", thinking_mode="not_available").policy_ref == "llama8b_it|gsm8k|not_available|card"
    # a task spelling resolves to its eval; the protocol row is model-independent
    assert pf.protocol_row("inspect/bfcl_200").numbers()["temperature"] == 0.001
    # the campaign default for multi-turn sends nothing; the not-sent map says what the server fills
    mt = pf.resolve("gemma4e4b_it", eval_category="multi_turn")
    assert mt.request_params() == {} and mt.params_not_sent["temperature"].startswith("1.0")
    # refusals: no row (a reference model's agentic key -- no row at all), a mode with no mapping, an unknown model / eval
    with pytest.raises(DP.NoPolicy):
        pf.resolve("gemma4e2b_it", eval_category="agentic")
    with pytest.raises(DP.NoPolicy):
        pf.resolve("nemotron3nano4b", mode="card")
    with pytest.raises(DP.NoPolicy):
        pf.resolve("gemma4e4b", mode="battery")
    with pytest.raises(DP.NoPolicy):
        pf.resolve("unknown_model", eval_id="gsm8k")
    with pytest.raises(DP.NoPolicy):
        pf.resolve("gemma4e4b_it", eval_id="no_such_eval")
    # the 9B agentic key: its vendor row is THE default since the rule of 2026-09-21 (decode-vendor-default-all-evals-2026-09-21);
    # before it the key had no default (every row explicitly non-default) and refused
    assert pf.resolve("qwen35_9b_it", eval_category="agentic").policy_id == "qwen35_9b_it|agentic|off|vendor@v1"
    assert pf.find("qwen35_9b_it", eval_category="agentic", mode="vendor")[0].numbers()["presence_penalty"] == 1.5


# --------------------------------------------------------------------------------------------
# a changed export changes the ACTUAL request parameters (subprocess: fresh import under $SISTEER_DECODE_POLICY)
# --------------------------------------------------------------------------------------------

_PROBE = r'''
import json, sys
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.serving import proxy_engine as PE
from sisteer.capeval.serving.proxy import RouteEntry
from sisteer.capeval.misalign import runner as MR
from sisteer.eval.attacks import bon_served as BON
from sisteer.eval.attacks import target_decode as TD
from sisteer.capeval.serving import vllm_server as VS
import huggingface_hub
huggingface_hub.try_to_load_from_cache = lambda *a, **k: None   # no local file to mask (the launcher tests' premise)
out = {}
out["card_gemma"] = C.sampling_for("gemma4e4b", "card").temperature
route = RouteEntry(virtual_model="petri__vm", cell_id="none", cell_class="none", serving_identity="proxy_vllm+lens+interleaved", intervention_hash="0"*16)
eng = PE.ProxyGenerateEngine("http://127.0.0.1:1/v1", route=route, system=None, thinking="off", concurrency=1, task="inspect/x",
                             sampling=C.sampling_for("gemma4e4b", "card"))
body = eng.build_request(PE.ChatItem(item_id="i", user="hi"), max_tokens=4)
out["proxy_body"] = {k: body.get(k) for k in ("temperature", "top_p", "top_k")}
greedy_eng = PE.ProxyGenerateEngine("http://127.0.0.1:1/v1", route=route, system=None, thinking="off", concurrency=1, task="inspect/x")
out["proxy_greedy_body_temperature"] = greedy_eng.build_request(PE.ChatItem(item_id="i", user="hi"), max_tokens=4)["temperature"]
out["bon_sent"] = BON.bon_sampler("gemma4e4b")                      # the DEFAULT: the served model's vendor preset
out["bon_sent_t1"] = BON.bon_sampler("gemma4e4b", BON.BON_HARNESS_DEFAULT)   # the superseded row, by name
out["runner_vendor"] = MR.vendor_sampling_for({"base_model": "Qwen/Qwen3.5-4B", "thinking_mode": "off"})
out["runner_t1"] = MR.build_payload("m", [{"role": "user", "content": "x"}], max_tokens=1)["temperature"]
out["server_default"] = VS._server_default_generation_config("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
out["bfcl"] = C.PROTOCOL_SAMPLING_BY_TASK[C.inspect_task("bfcl_200")].executed.temperature
out["td_mode_0p3"] = TD.decode_mode(0.3)
out["identity_card_gemma"] = C.sampling_identity(C.sampling_for("gemma4e4b", "card"), {"inspect/gsm8k_paired200": 1})
print(json.dumps(out))
'''


def _probe(export_path: str) -> dict:
    env = {**ENV, DP.ENV_VAR: export_path}
    res = subprocess.run([PY, "-c", _PROBE], capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert res.returncode == 0, res.stderr[-4000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


def _mutated_export(tmp_path) -> str:
    with open(EXPORT, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    def edit(pid, **changes):
        p = next(x for x in data["policies"] if x["policy_id"] == pid)
        p.update(changes)

    edit("gemma4e4b_it|capability|off|card@v1", temperature=0.42, top_k=7)             # contracts card row -> proxy body
    edit("gemma4e4b_it|static|off|greedy@v1", temperature=0.05)                       # the greedy row (every model must agree)
    for m in ("qwen35_4b_it", "nemotron3nano4b", "llama8b_it", "glm46vflash"):
        edit(f"{m}|static|off|greedy@v1", temperature=0.05)
    for m in ("gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_it", "glm46vflash"):
        edit(f"{m}|bon|off|grandfathered_T1@v1", temperature=0.9)                     # BoN's superseded T=1 row (by name)
        edit(f"{m}|agentic|off|runner_forced_T1@v1", temperature=0.8)                # runner t1
        edit(f"{m}|bfcl|off|protocol@v1", temperature=0.002)                          # protocol row
    for m in ("qwen35_35b_a3b", "gemma4_31b"):
        edit(f"{m}|bfcl|off|protocol@v1", temperature=0.002)                          # ...and the pipeline-test models' EXPLICIT protocol rows
        # (2026-09-29; paper_status reference, so no `*` expansion reaches them): model_independent() refuses a disagreement, so every row moves
    edit("gemma4e4b_it|bon|off|vendor_preset@v1", top_k=7)                             # BoN's DEFAULT: gemma's vendor row
    edit("qwen35_4b_it|agentic|off|vendor@v1", top_p=0.55)                             # runner vendor row
    edit("qwen35_4b_it|omit_params|off|server_default@v1", top_p=0.66, min_p=0.01)     # composed server default (4B route row)
    edit("qwen35_9b_it|omit_params|off|server_default@v1", top_p=0.66, min_p=0.01)     # ...and the 9B route row: both compose ONE vendor row
    edit("qwen35_4b_it|ifeval|off|vendor_preset@v1", top_p=0.66, min_p=0.01)          # the vendor row it composes (agreement asserted at import)
    # ...and the worked example's twins of every 4B row moved above (2026-10-09, ruling decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09:
    # qwen35_4b_demo re-pins the same checkpoint and its rows ARE the 4B's, same_values_as-checked by the loader); the route row among them
    # composes the same vendor row, so contracts' import-time agreement check reads it beside the 4B's and the 9B's
    edit("qwen35_4b_demo|omit_params|off|server_default@v1", top_p=0.66, min_p=0.01)
    edit("qwen35_4b_demo|ifeval|off|vendor_preset@v1", top_p=0.66, min_p=0.01)
    edit("qwen35_4b_demo|agentic|off|vendor@v1", top_p=0.55)
    edit("qwen35_4b_demo|agentic|off|runner_forced_T1@v1", temperature=0.8)
    edit("qwen35_4b_demo|bon|off|grandfathered_T1@v1", temperature=0.9)
    for m in ("gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_it", "glm46vflash"):
        edit(f"{m}|multi_turn|off|greedy@v1", temperature=0.5)                         # target-decode boundary
    data["header"]["policy_set_sha256"] = DP.policy_set_sha256(data)
    path = tmp_path / "decode_policy.mutated.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return str(path)


def test_policy_fixture_change_changes_the_actual_request_parameters(tmp_path):
    base = _probe(EXPORT)
    changed = _probe(_mutated_export(tmp_path))
    # baseline = today's numbers
    assert base["card_gemma"] == 1.0 and base["proxy_body"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    assert base["proxy_greedy_body_temperature"] == 0.0 and base["runner_t1"] == 1.0
    assert base["bon_sent"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 64} and base["bon_sent_t1"] == {"temperature": 1.0}
    assert base["runner_vendor"]["top_p"] == 0.8 and base["bfcl"] == 0.001 and base["td_mode_0p3"] == "explicit"
    assert json.loads(base["server_default"]) == {"min_p": 0.0, "repetition_penalty": 1.0, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
    # the mutated export moves EVERY entry point's actual request parameters
    assert changed["card_gemma"] == 0.42 and changed["proxy_body"] == {"temperature": 0.42, "top_p": 0.95, "top_k": 7}
    assert changed["proxy_greedy_body_temperature"] == 0.05
    assert changed["bon_sent"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 7} and changed["bon_sent_t1"] == {"temperature": 0.9}
    assert changed["runner_vendor"]["top_p"] == 0.55 and changed["runner_t1"] == 0.8
    assert json.loads(changed["server_default"]) == {"min_p": 0.01, "repetition_penalty": 1.0, "temperature": 0.7, "top_k": 20, "top_p": 0.66}
    assert changed["bfcl"] == 0.002
    assert changed["td_mode_0p3"] == "greedy"                                        # 0.3 <= the new 0.5 boundary
    # and the hashed identity moves with the numbers (a changed number IS a different instrument)
    assert changed["identity_card_gemma"]["temperature"] == 0.42 and base["identity_card_gemma"]["temperature"] == 1.0


def test_a_disagreeing_server_default_route_row_refuses_at_import(tmp_path):
    with open(EXPORT, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    p = next(x for x in data["policies"] if x["policy_id"] == "qwen35_4b_it|omit_params|off|server_default@v1")
    p["top_p"] = 0.11                                                                   # the route row now contradicts the vendor row
    data["header"]["policy_set_sha256"] = DP.policy_set_sha256(data)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    res = subprocess.run([PY, "-c", "import sisteer.capeval.tinycap.contracts"], capture_output=True, text=True,
                         env={**ENV, DP.ENV_VAR: str(path)}, cwd=ROOT, timeout=300)
    assert res.returncode != 0 and "disagrees with the registry's route row" in res.stderr


def test_missing_export_refuses_by_path(tmp_path):
    res = subprocess.run([PY, "-c", "import sisteer.capeval.tinycap.contracts"], capture_output=True, text=True,
                         env={**ENV, DP.ENV_VAR: str(tmp_path / "nope.json")}, cwd=ROOT, timeout=300)
    assert res.returncode != 0 and "nope.json" in res.stderr and "NoPolicy" in res.stderr


def test_reader_is_stdlib_only():
    probe = ("import sys, sisteer.registry.decode_policy as d; p = d.load(%r).resolve('gemma4e4b', mode='card'); "
             "print(p.policy_id, sorted(m for m in sys.modules if m.split('.')[0] in ('pandas','numpy','torch','yaml','sqlite3')))" % EXPORT)
    res = subprocess.run([PY, "-S", "-c", probe], capture_output=True, text=True, cwd="/", env={"PYTHONPATH": ENV["PYTHONPATH"], "PATH": os.environ["PATH"]})
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "gemma4e4b_it|capability|off|card@v1 []"


def test_a_src_only_staged_tree_imports_contracts_from_the_package_copy(tmp_path):
    """The Modal mount and the launcher's staged trees carry `sisteer` with no `configs/`: contracts must import there,
    reading the package-data copy beside decode_policy.py (byte-identical to configs/registry/decode_policy.json)."""
    tree = tmp_path / "tree"
    (tree / "src").mkdir(parents=True)
    (tree / "src" / "sisteer").symlink_to(os.path.join(ROOT, "src", "sisteer"), target_is_directory=True)
    probe = ("import sisteer.registry.decode_policy as d, sisteer.capeval.tinycap.contracts as C; "
             "print(d.default_export_path()); print(C.sampling_for('gemma4e4b', 'card').temperature)")
    env = {k: v for k, v in ENV.items() if k != DP.ENV_VAR}
    env["PYTHONPATH"] = str(tree / "src")
    res = subprocess.run([PY, "-c", probe], capture_output=True, text=True, env=env, cwd=str(tree), timeout=300)
    assert res.returncode == 0, res.stderr[-3000:]
    path, temp = res.stdout.strip().splitlines()[-2:]
    assert path == str(tree / "src" / "sisteer" / "registry" / "decode_policy.json") and temp == "1.0"


def test_the_two_committed_export_copies_are_byte_identical():
    with open(EXPORT, "rb") as a, open(DP.PACKAGE_EXPORT_PATH, "rb") as b:
        assert a.read() == b.read()
    assert sorted(DP.export_paths()) == sorted([EXPORT, DP.PACKAGE_EXPORT_PATH])


def test_no_production_module_imports_the_literal_fixture():
    hits = []
    for base in ("src", "scripts"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, base)):
            for fn in files:
                if fn.endswith(".py"):
                    with open(os.path.join(dirpath, fn), "r", encoding="utf-8", errors="replace") as fh:
                        if "sampling_literals_before" in fh.read():
                            hits.append(os.path.relpath(os.path.join(dirpath, fn), ROOT))
    assert not hits, hits


def test_no_sampling_number_literal_survives_in_the_redirected_tables():
    """The tables are built by lookup: the module text no longer spells the numbers as table literals."""
    with open(os.path.join(ROOT, "src", "sisteer", "capeval", "tinycap", "contracts.py"), "r", encoding="utf-8") as fh:
        src = fh.read()
    for needle in ("_GEMMA_CARD = dict(temperature=1.0", "_LLAMA_CARD = dict(temperature=0.6", "_QWEN35_CARD = dict(temperature=0.7",
                   "GREEDY_TEMPERATURE = 0.0", "BFCL_PROTOCOL_TEMPERATURE = 0.001", "PROTOCOL_TOP_P = 1.0",
                   '"thinking_general": SamplingRow(', '"Qwen/Qwen3.5-4B": QWEN35_VENDOR_ROWS["instruct_general"]'):
        assert needle not in src, needle
    with open(os.path.join(ROOT, "src", "sisteer", "capeval", "misalign", "runner.py"), "r", encoding="utf-8") as fh:
        assert '{"temperature": 0.7, "top_p": 0.8, "top_k": 20' not in fh.read()
    with open(os.path.join(ROOT, "src", "sisteer", "eval", "attacks", "bon_served.py"), "r", encoding="utf-8") as fh:
        assert 'return {"temperature": 1.0} if decode is None' not in fh.read()
