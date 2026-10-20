"""THE DECLARED-RENDER CASES of scripts/lens_vectorize_gate.py — the CPU half
(the ``test_lens_resumed_prefill_gate.py`` pattern).

Pins: the case registry (three names, their families, opt-in ordering, the
startup refusal of a case named for another family, the results sub-dir);
the proxy literals the checks assert against (codes, ledger keys, policy
labels, regimes, the attestation keys) against ``proxy.py``; the request
shapes byte for byte in their keys, with every vmodel DERIVED
through ``harness_cell_id`` / ``NOOP_LENS_CELL_ID``; the expected masked-row
counts DERIVED from renders (never literals) — the synthetic all-turns ids of
the proxy test, and the REAL pinned tokenizers offline (Qwen3.5-4B: 7 vs
5; gemma-4-E4B-it: the 3-turn conversation's spans, the quoted delimiter's
lexical refusal, the complete fake boundary's ``ChatMarkerError``, exercised
through the REAL proxy code); the three checkers' truth tables over recorded
case dicts (a green record passes every check; each perturbation fails
exactly the named check(s); an empty record fails everything); the result
schema riding into ``overall_exit_code`` and the report; the standup
helper's marker derivation, digest binding, attestation reader and
cross-tree kwarg guard, including an OFFLINE end-to-end registry export +
load for a ``none`` + base + ``@variant`` standup over a synthetic psr
object; the run_gate / start_standup source pins; and that the Modal wrapper
needs no change beyond passing ``--cases`` through.

All synthetic: invented ids, placeholder digests, no real recorded data — the
one exception is ``REAL_ENGINE_ID_PAIRS``, the three vLLM-minted request ids
(no content) of the 2026-09-07 GPU run whose ``one_entry`` checks refused the
engine's ``-<8 hex>`` internal suffix. The real-tokenizer tests skip on a clone
without the pinned HF cache.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# scripts/ goes to the FRONT unconditionally: scripts/modal/lens_vectorize_gate.py shares the bare module name with the
# reviewed gate in scripts/, and 52 test modules put scripts/modal on sys.path — whichever directory is in front at the
# FIRST bare import wins for the whole process. A conditional insert left scripts/ wherever an earlier module had put
# it, and a sharded gate whose shard collected such a module first imported the Modal launcher here (AttributeError:
# module 'modal' has no attribute 'App' under a modal stub; cpu-suite-8968828 shard 1, 2026-09-16).
_SCRIPTS_DIR = str(REPO / "scripts")
if _SCRIPTS_DIR in sys.path:
    sys.path.remove(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lens_vectorize_gate as g  # noqa: E402
assert Path(g.__file__).resolve() == (REPO / "scripts" / "lens_vectorize_gate.py").resolve(), \
    f"the bare name resolved to {g.__file__}, not the reviewed gate in scripts/ (sys.path order)"
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.petri.registry_adapter import harness_cell_id  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import serve_declaration as sd  # noqa: E402
from sisteer.capeval.serving import span_locator as sl  # noqa: E402

from test_lens_vectorize_gate import HEADER, TestObjectResolution, _cfg, _helper, _sig  # noqa: E402

DIMS = TestObjectResolution.DIMS
#: a synthetic Qwen-shaped header pair (the OFF header 7 ids ending in a closed
#: think block, the ON header 5 ids ending in an open one — the shape the tests
#: use), never the real ids
OFF = (901, 902, 903, 904, 905, 906, 907)
ON = (901, 902, 903, 904, 908)
GEMMA_HDR = (900, 901, 902)          # a 3-id header like gemma's
#: gemma's thinking-header fact shape: a switch that leaves the header in place (moves False)
GEMMA_TH = {"switch": True, "moves": False, "off_ids": list(GEMMA_HDR), "on_ids": list(GEMMA_HDR)}
#: a PARTIAL fact (code review X1 / S-G3): the switch is there, the ON header could not be certified
PARTIAL_TH = {"switch": True, "moves": None, "off_ids": list(OFF), "on_ids": None}
PARTIAL_REASON = "ChatMarkerError: probe: the add_generation_prompt render is not a prefix of the probe render"
NEMOTRON = ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f")
SHA = "ab" * 32
QWEN = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
GEMMA = ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2")
CELL = g.DECLARED_RENDER_CELL
HB = harness_cell_id(CELL)
HV_THINKING = harness_cell_id(g.variant_cell(CELL, "thinking"))
HV_ALL_POS = harness_cell_id(g.variant_cell(CELL, "all_positions"))


def _cached_tok(model, revision):
    transformers = pytest.importorskip("transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
    except Exception as e:  # noqa: BLE001 — not cached here: skip, never fetch
        pytest.skip(f"pinned {model} tokenizer @ {revision[:8]} not in the local HF cache ({type(e).__name__})")
    from sisteer.models import pin_reasoning_template_off
    pin_reasoning_template_off(tok)
    return tok


# --------------------------------------------------------------------------- #
# literals and the case registry                                              #
# --------------------------------------------------------------------------- #


class TestLiteralMirrors:
    PAIRS = ("THINKING_ON_KEY", "CHAT_TEMPLATE_KWARGS_KEY", "ENABLE_THINKING_KEY", "THINKING_KWARGS_INVALID",
             "THINKING_ON_UNDECLARED", "THINKING_ON_MISMATCH", "THINKING_ON_NOT_LENS",
             "THINKING_ON_CELL_NOT_DECLARED", "ALL_TURNS_RENDER_REFUSED", "ALL_TURNS_EXPECTED_UNAVAILABLE",
             "LEDGER_KEY_RENDER_REGIME", "LEDGER_KEY_ALL_TURNS_CHECK", "ALL_TURNS_CHECK_OK",
             "RENDER_REGIME_THINKING_ON", "RENDER_REGIME_THINKING_OFF", "LEDGER_KEY_SALT_REGIME",
             "LEDGER_KEY_RESUMED_POLICY", "SALT_REGIME_CELL_STABLE", "SALT_REGIME_REQUEST_UNIQUE",
             "POLICY_DECLARED_SHARES_NOTHING", "POLICY_FORCED_REQUEST_UNIQUE", "POLICY_NOT_CONSULTED",
             "LENS_SALT_POLICY_ATTESTED", "LENS_SALT_POLICY_REQUEST_UNIQUE_ALL", "LENS_SALT_POLICY_KEY",
             "SERVE_DECLARATIONS_KEY", "THINKING_HEADER_KEY", "THINKING_HEADER_ERROR_KEY",
             "OBJECT_SHA256_BY_CELL_KEY")

    @pytest.mark.parametrize("name", PAIRS)
    def test_gate_literal_equals_the_proxys(self, name):
        assert getattr(g, name) == getattr(px, name), name

    def test_the_plan_literals(self):
        # §2.5.6 / §2.5.11: the codes and stamp literals the harness reads too
        assert g.THINKING_ON_KEY == "petri_thinking_on" and g.CHAT_TEMPLATE_KWARGS_KEY == "chat_template_kwargs"
        assert (g.THINKING_ON_UNDECLARED, g.THINKING_ON_MISMATCH, g.THINKING_KWARGS_INVALID,
                g.THINKING_ON_CELL_NOT_DECLARED, g.THINKING_ON_NOT_LENS) == (
            "thinking_on_undeclared", "thinking_on_mismatch", "thinking_kwargs_invalid",
            "thinking_on_cell_not_declared", "thinking_on_not_lens")
        assert (g.SALT_REGIME_REQUEST_UNIQUE, g.SALT_REGIME_CELL_STABLE) == ("request-unique", "cell-stable")
        assert (g.POLICY_DECLARED_SHARES_NOTHING, g.POLICY_FORCED_REQUEST_UNIQUE, g.POLICY_NOT_CONSULTED) == (
            "declared-request-unique", "forced-request-unique", "not-consulted")
        assert (g.RENDER_REGIME_THINKING_ON, g.RENDER_REGIME_THINKING_OFF) == ("thinking_on", "thinking_off")
        assert g.RESUMED_CONDITIONAL == px.RESUMED_PREFILL_CONDITIONAL == "conditional"
        assert (g.ALL_TURNS_RENDER_REFUSED, g.ALL_TURNS_EXPECTED_UNAVAILABLE) == (
            "all_turns_render_refused", "all_turns_expected_unavailable")
        assert g.SERVE_DECLARATIONS_KEY == "lens_serve_declarations" and g.LENS_SALT_POLICY_KEY == "lens_salt_policy"
        assert g.OBJECT_SHA256_BY_CELL_KEY == "object_artifact_sha256_by_cell"

    def test_expected_declarations_are_derived_from_the_closed_vocabulary(self):
        for variant in g.DECLARED_RENDER_VARIANT.values():
            assert variant in sd.SERVE_VARIANTS
            assert g.expected_declarations(variant) == sd.variant_declarations(variant)
        # the §2.5.7 literals the harness asserts
        assert g.expected_declarations("thinking") == {"serve_out_of_support": ["thinking"]}
        assert g.expected_declarations("all_positions") == {"serve_experiment": "fit_response_serve_all_positions"}
        assert g.expected_declarations("all_turns") == {"serve_experiment": "fit_response_serve_all_turns"}
        with pytest.raises(sd.ServeDeclarationError):
            g.expected_declarations("sideways")

    def test_attest_fields_are_the_selection_records(self):
        import torch
        from sisteer.capeval.serving import lens_adapters as la
        from sisteer.capeval.serving import lens_worker as lw
        from test_petri_lens_worker import HIDDEN, _obj
        obj = _obj("c", [la.HookOp(op=la.OP_ADD_VECTOR, layer=1, vector=torch.ones(HIDDEN),
                                   positions=la.POSITIONS_RESPONSE, header_ids=OFF)], scale=1.0)
        rec = lw.SelectionRecord()
        ids = (5, 6, *OFF)
        ctx = la.HookContext(la.PHASE_PREFILL, tuple(range(len(ids))), ids)
        rec.observe(req_id="r", layer=1, requested="c", requested_hash="h", requested_scale=None,
                    applied=obj, modified=True, path=lw.SELECTION_PATH_PER_SLOT, ctx=ctx)
        entry = rec.as_dict()["requests"]["r"]
        assert set(g.DECLARED_ATTEST_FIELDS) <= set(entry)
        assert entry["facts"]["thinking_on"] is False and entry["prefill_masked_rows"] == len(OFF)
        assert g.one_declared_attestation(rec.as_dict()["requests"])["prefill_masked_rows"] == len(OFF)
        assert g.one_declared_attestation({})["n_entries"] == 0
        assert g.one_declared_attestation({"a": {}, "b": {}})["n_entries"] == 2

    def test_helper_salt_policies_mirror_the_proxy(self):
        helper = _helper()
        assert tuple(helper.LENS_SALT_POLICIES) == tuple(px.LENS_SALT_POLICIES)
        assert set(helper.PROXY_DECLARED_KWARGS) <= set(inspect.signature(px.RoutingProxy.__init__).parameters)
        assert "thinking_header_error" in helper.PROXY_DECLARED_KWARGS            # S-G3: the reason reaches the proxy
        from sisteer.capeval.serving.lens_steer import export_worker_registry
        assert set(helper.EXPORT_DECLARED_KWARGS) <= set(inspect.signature(export_worker_registry).parameters)
        # the routing-table kwargs: the fact (this tip) and the reason beside it (the proxy lane's
        # contract — a table key the helper hands over ONLY when export_routing_table takes it)
        assert helper.TABLE_DECLARED_KWARGS == ("thinking_header", "thinking_header_error")
        assert "thinking_header" in inspect.signature(px.export_routing_table).parameters
        kw, dropped = helper.accepted_kwargs(px.export_routing_table, {"thinking_header": {"switch": False},
                                                                        "thinking_header_error": "why"})
        assert kw["thinking_header"] == {"switch": False}
        assert ("thinking_header_error" in kw) == (
            "thinking_header_error" in inspect.signature(px.export_routing_table).parameters)
        assert helper.VARIANT_SEP == sd.VARIANT_SEP == "@"


class TestCaseRegistry:
    def test_names_families_variants(self):
        assert g.DECLARED_RENDER_CASES == ("declared_thinking", "all_positions_carveout", "all_turns_render_check")
        assert g.DECLARED_RENDER_FAMILIES == {"declared_thinking": ("qwen35_4b",),
                                              "all_positions_carveout": ("gemma4e4b",),
                                              "all_turns_render_check": ("gemma4e4b",)}
        assert g.DECLARED_RENDER_VARIANT == {"declared_thinking": "thinking",
                                             "all_positions_carveout": "all_positions",
                                             "all_turns_render_check": "all_turns"}
        assert g.DECLARED_RENDER_CELL == "safety_refusal__psr_kl_all"
        assert set(g.DECLARED_RENDER_CHECKERS) == set(g.DECLARED_RENDER_CASES)
        # not inventory ids: the derived inventory's grammar has six ':' fields
        for cid in g.DECLARED_RENDER_CASES:
            assert ":" not in cid and cid != g.MULTIGROUP
            with pytest.raises(Exception):
                g.parse_case_id(cid)

    def test_order_is_opt_in_and_family_scoped(self):
        qwen, gemma = g.MODELS["qwen35_4b"], g.MODELS["gemma4e4b"]
        assert g.declared_render_order(qwen, None) == [] and g.declared_render_order(gemma, []) == []
        assert g.declared_render_order(qwen, ["declared_thinking"]) == ["declared_thinking"]
        assert g.declared_render_order(gemma, ["all_turns_render_check", "all_positions_carveout"]) == [
            "all_positions_carveout", "all_turns_render_check"]
        # inventory ids and multigroup beside them are ignored here (gate_case_order's business)
        assert g.declared_render_order(gemma, ["multigroup", "psr:relu_gated_vector:gemv:all:skipbos0:both",
                                               "all_turns_render_check"]) == ["all_turns_render_check"]
        with pytest.raises(g.GateError, match="runs on \\['gemma4e4b'\\] only, not on 'qwen35_4b'"):
            g.declared_render_order(qwen, ["all_turns_render_check"])
        with pytest.raises(g.GateError, match="runs on \\['qwen35_4b'\\] only"):
            g.declared_render_order(gemma, ["declared_thinking"])
        assert g.declared_render_only(["declared_thinking"]) is True
        assert g.declared_render_only(["all_positions_carveout", "all_turns_render_check"]) is True
        assert g.declared_render_only(["declared_thinking", "multigroup"]) is False
        assert g.declared_render_only(None) is False and g.declared_render_only([]) is False

    def test_subset_slugs_and_results_dirs(self):
        assert g.subset_slug(["declared_thinking"]) == "declared_thinking"
        assert g.subset_slug(["all_positions_carveout", "all_turns_render_check"]) == \
            "all_positions_carveout+all_turns_render_check"

    def test_main_refuses_a_case_named_for_another_family_and_accepts_its_own(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run_gate(ctx, **kw):
            captured["ctx"], captured["kw"] = ctx, kw
            return {"exit_code": 3}
        monkeypatch.setattr(g, "run_gate", fake_run_gate)
        monkeypatch.setattr(g, "plan_inventory", lambda *a, **k: {})
        monkeypatch.setattr(g, "header_len_for", lambda *a, **k: 4)
        common = ["--reference-repo", str(tmp_path), "--results-root", str(tmp_path / "r"),
                  "--objects-root", str(tmp_path / "o"), "--candsi-root", str(tmp_path / "c")]
        with pytest.raises(SystemExit, match="runs on \\['gemma4e4b'\\] only"):
            g.main(["--model", "qwen35_4b", *common, "--cases", "all_turns_render_check"])
        assert "ctx" not in captured                       # refused BEFORE anything ran
        assert g.main(["--model", "qwen35_4b", *common, "--cases", "declared_thinking"]) == 3
        assert captured["kw"]["only_cases"] == ["declared_thinking"]
        assert captured["ctx"].results_dir == tmp_path / "r" / "qwen35_4b" / "subset" / "declared_thinking"
        assert g.main(["--model", "gemma4e4b", *common, "--cases",
                       "all_positions_carveout,all_turns_render_check"]) == 3
        assert captured["ctx"].results_dir == (tmp_path / "r" / "gemma4e4b" / "subset"
                                               / "all_positions_carveout+all_turns_render_check")

    def test_docstring_names_the_section_and_every_check_key(self):
        doc = g.__doc__
        sec = doc[doc.index("THE DECLARED-RENDER CASES (fit-support"):doc.index("EXIT CODES (``overall_exit_code``")]
        for cid in g.DECLARED_RENDER_CASES:
            assert cid in sec
        for cid, fn in g.DECLARED_RENDER_CHECKERS.items():
            for key in fn({}):
                assert re.search(rf"(?<![A-Za-z_]){re.escape(key)}(?![A-Za-z_])", sec), (cid, key)
        assert "opt-in" in sec.lower() and "candidate_declared" in sec


# --------------------------------------------------------------------------- #
# request shapes (§2.5.3) and derived expectations                            #
# --------------------------------------------------------------------------- #


class TestRequestShapes:
    MSGS = [{"role": "user", "content": "Passage 1. hello"}]
    SAMPLING = {"max_tokens": 4, "temperature": 0.0, "seed": 0}

    def test_vmodels_are_derived_never_spelled(self):
        variant = g.variant_cell(CELL, "thinking")
        assert variant == "safety_refusal__psr_kl_all@thinking"
        assert harness_cell_id(variant) == "si_safety__psr_kl_all@thinking"
        assert f"{dc.VMODEL_PREFIX}{harness_cell_id(variant)}" == "petri__si_safety__psr_kl_all@thinking"
        assert f"{dc.VMODEL_PREFIX}{dc.NOOP_LENS_CELL_ID}" == "petri__none__lens0"
        assert HB == "si_safety__psr_kl_all"
        with pytest.raises(sd.ServeDeclarationError):
            g.variant_cell(CELL, "sideways")
        with pytest.raises(sd.ServeDeclarationError):
            g.variant_cell(variant, "thinking")   # a variant is not a base

    def test_declared_on_shape_byte_for_byte(self):
        body = g.thinking_request("petri__x", self.MSGS, self.SAMPLING, enable_thinking=True, declare=True)
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        assert body["petri_thinking_on"] is True
        assert set(body) == {"model", "messages", "chat_template_kwargs", "petri_thinking_on",
                             "max_tokens", "temperature", "seed"}
        assert body["messages"] == self.MSGS and body["messages"] is not self.MSGS
        # the ON unsteered arm on petri__base: kwargs, no declaration
        body = g.thinking_request("petri__base", self.MSGS, self.SAMPLING, enable_thinking=True, declare=False)
        assert body["chat_template_kwargs"] == {"enable_thinking": True} and "petri_thinking_on" not in body
        # thinking-OFF: kwargs false, no declaration
        body = g.thinking_request("petri__y", self.MSGS, self.SAMPLING, enable_thinking=False, declare=False)
        assert body["chat_template_kwargs"] == {"enable_thinking": False} and "petri_thinking_on" not in body
        # the mismatch probe: a declaration with NO kwargs at all
        body = g.thinking_request("petri__x", self.MSGS, self.SAMPLING, enable_thinking=None, declare=True)
        assert "chat_template_kwargs" not in body and body["petri_thinking_on"] is True
        # the shape probe: a foreign kwarg beside enable_thinking
        body = g.thinking_request("petri__x", self.MSGS, self.SAMPLING, enable_thinking=True, declare=True,
                                  extra_kwargs={"foreign_kwarg": 1})
        assert body["chat_template_kwargs"] == {"enable_thinking": True, "foreign_kwarg": 1}
        # the gemma cases send NO render keys
        plain = g.plain_request("petri__z", self.MSGS, self.SAMPLING)
        assert set(plain) == {"model", "messages", "max_tokens", "temperature", "seed"}

    def test_the_declared_shape_is_what_the_real_proxy_serves_and_refuses(self, tmp_path):
        """The gate's bodies through the REAL proxy (the thinking matrix
        fixtures): the declared ON body is served on the @thinking cell and
        on the declared twin, and each refusal probe draws exactly the code
        the check expects — so the drive's expectations are the proxy's
        behaviour, not the gate's guess."""
        import asyncio
        from test_petri_proxy import CANONICAL_22, StubBackend, _cells, _source, _spec
        spec = _spec(serving_mode="interleaved", cells=list(CANONICAL_22) + [f"{HB}@thinking"])
        src = _source(tmp_path)
        entries = dict(src.entries)
        base = entries[HB]
        entries[f"{HB}@thinking"] = dc.RegistryEntry(
            cell_id=f"{HB}@thinking", kind=base.kind, si_id=base.si_id, object_path=base.object_path,
            serving_path="lens", resolved_spec=sd.apply_variant(base.resolved_spec, "thinking"))
        source = dc.MappingCellSource(entries)
        cells = _cells(spec, source)
        backend = StubBackend()
        decl = g.expected_declarations("thinking")
        header = {"switch": True, "moves": True, "off_ids": list(OFF), "on_ids": list(ON)}
        proxy = px.RoutingProxy(px.routes_from_cells(cells), backend, base_model=spec.target_model,
                                serving_mode="interleaved", prompted_prefixes=px.prompted_prefix_specs(cells, source),
                                exact_replay_strategy=None, petri_agent_loop_entry=False, thinking_header=header,
                                lens_serve_declarations={f"{HB}@thinking": decl, dc.NOOP_LENS_CELL_ID: decl})
        vm_variant, vm_twin, vm_base = f"petri__{HB}@thinking", f"petri__{dc.NOOP_LENS_CELL_ID}", f"petri__{HB}"

        def ask(body):
            return asyncio.run(proxy.handle_chat_completions(body))

        def refuses(body):
            with pytest.raises(px.ProxyRefusal) as ei:
                ask(body)
            return ei.value.code
        s = self.SAMPLING
        out = ask(g.thinking_request(vm_variant, self.MSGS, s, enable_thinking=True, declare=True))
        assert out["petri"][g.LEDGER_KEY_RENDER_REGIME] == g.RENDER_REGIME_THINKING_ON
        assert out["petri"][g.LEDGER_KEY_RESUMED_POLICY] == g.POLICY_DECLARED_SHARES_NOTHING
        assert out["petri"][g.LEDGER_KEY_SALT_REGIME] == g.SALT_REGIME_REQUEST_UNIQUE
        assert backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]["thinking_on"] == 1
        out = ask(g.thinking_request(vm_twin, self.MSGS, s, enable_thinking=True, declare=True))
        assert out["petri"][g.LEDGER_KEY_RESUMED_POLICY] == g.POLICY_DECLARED_SHARES_NOTHING
        out = ask(g.thinking_request(vm_base, self.MSGS, s, enable_thinking=False, declare=False))
        assert out["petri"][g.LEDGER_KEY_RENDER_REGIME] == g.RENDER_REGIME_THINKING_OFF
        out = ask(g.thinking_request("petri__base", self.MSGS, s, enable_thinking=True, declare=False))
        assert (out["petri"][g.LEDGER_KEY_SALT_REGIME], out["petri"][g.LEDGER_KEY_RESUMED_POLICY],
                out["petri"][g.LEDGER_KEY_RENDER_REGIME]) == (
            g.SALT_REGIME_CELL_STABLE, g.POLICY_NOT_CONSULTED, g.RENDER_REGIME_THINKING_ON)
        assert refuses(g.thinking_request(vm_variant, self.MSGS, s, enable_thinking=True, declare=False)) == \
            g.THINKING_ON_UNDECLARED
        assert refuses(g.thinking_request(vm_variant, self.MSGS, s, enable_thinking=None, declare=True)) == \
            g.THINKING_ON_MISMATCH
        assert refuses(g.thinking_request(vm_variant, self.MSGS, s, enable_thinking=True, declare=True,
                                          extra_kwargs={"foreign_kwarg": 1})) == g.THINKING_KWARGS_INVALID
        assert refuses(g.thinking_request(vm_base, self.MSGS, s, enable_thinking=True, declare=True)) == \
            g.THINKING_ON_CELL_NOT_DECLARED
        assert refuses(g.thinking_request("petri__base", self.MSGS, s, enable_thinking=True, declare=True)) == \
            g.THINKING_ON_NOT_LENS

    def test_evidence_sampling_is_the_gates_greedy_evidence(self):
        import types
        ctx = types.SimpleNamespace(max_tokens=64, top_logprobs=5, ignore_eos=True)
        s = g.evidence_sampling(ctx)
        up = g.upstream_payload(ctx, "m", [], cache_salt="x")
        assert s == {k: up[k] for k in s} and set(s) == set(up) - {"model", "messages", "cache_salt"}
        assert s["temperature"] == 0.0 and s["seed"] == 0 and s["logprobs"] is True


class TestDerivedExpectations:
    def test_response_rows_are_the_header_width(self):
        assert g.response_rows_expected(list(ON)) == 5 and g.response_rows_expected(list(OFF)) == 7
        assert g.response_rows_expected(list(GEMMA_HDR)) == 3
        assert g.response_rows_expected(None) is None and g.response_rows_expected([]) is None
        assert g.response_rows_expected("abc") is None

    def test_all_positions_rows_are_the_prompt_minus_bos(self):
        assert g.all_positions_rows_expected(200, skip_bos=False) == 200
        assert g.all_positions_rows_expected(200, skip_bos=True) == 199
        assert g.all_positions_rows_expected(None, skip_bos=False) is None
        assert g.all_positions_rows_expected(0, skip_bos=False) is None

    def test_all_turns_spans_on_the_synthetic_render(self):
        """The synthetic ids: a user turn, a completed assistant turn, a user
        turn, the generating header — the gate's own arithmetic is the
        worker's (span_locator)."""
        from test_all_turns_render_check import END, HDR, IDS
        spans = g.all_turns_spans_of(IDS, HDR, END)
        final = sl.locate_final_header(list(IDS), HDR, declared=False)
        assert spans == sl.assistant_turn_spans(list(IDS), HDR, END, final)
        assert spans == ((2, 9), (13, 16))
        assert g.all_turns_rows_expected(spans) == 7 + 3
        # an unterminated earlier turn: the lexical locator refuses, so does the gate
        with pytest.raises(sl.SpanLocatorError):
            g.all_turns_spans_of((1, *HDR, 50, 51, 900, 903, 902, 3, *HDR), HDR, END)
        with pytest.raises(sl.SpanLocatorError):
            g.all_turns_spans_of((1, *HDR, 50), HDR, END)      # not ending at the header

    def test_probe_conversations(self):
        convo = list(g.ALL_TURNS_CONVERSATION)
        assert [m["role"] for m in convo] == ["user", "assistant", "user", "assistant", "user"]
        quoted = g.all_turns_probe_conversation("quoted")
        fake = g.all_turns_probe_conversation("fake")
        assert quoted[3]["content"] == g.ALL_TURNS_QUOTED_DELIMITER and "<turn|>\n" in quoted[3]["content"]
        assert fake[3]["content"] == g.ALL_TURNS_FAKE_BOUNDARY and "<turn|>\n<|turn>model\n" in fake[3]["content"]
        assert quoted[:3] == convo[:3] and quoted[4] == convo[4]
        with pytest.raises(KeyError):
            g.all_turns_probe_conversation("other")
        assert g.declared_render_prompt()[0]["role"] == "user" and g.FILLER in g.declared_render_prompt()[0]["content"]

    def test_degeneracy_instrument(self):
        healthy = g.degeneracy_instrument("the quick brown fox jumps over the lazy dog and runs away home")
        assert healthy["ran"] is True and healthy["ratio"] == 1.0 and healthy["degenerate"] is False
        loop = g.degeneracy_instrument("e e e e e e e e e e e e e e e e e e e e")
        assert loop["ran"] and loop["ratio"] < g.DEGENERACY_THRESHOLD and loop["degenerate"] is True
        short = g.degeneracy_instrument("ok")
        assert short["ran"] and short["degenerate"] is None      # < 3 words: not judged
        assert g.degeneracy_instrument(None)["ran"] is False
        from sisteer.eval.capability import distinct_ngram_ratio
        assert healthy["ratio"] == distinct_ngram_ratio(["the quick brown fox jumps over the lazy dog and runs away home"], n=3)
        assert (g.DEGENERACY_NGRAM, g.DEGENERACY_THRESHOLD) == (3, 0.25)


# --------------------------------------------------------------------------- #
# the REAL tokenizers, offline                                                #
# --------------------------------------------------------------------------- #


class TestRealQwen:
    def test_thinking_header_for_and_the_7_vs_5(self):
        tok = _cached_tok(*QWEN)
        from sisteer import chat_markers as cmk
        facts = cmk.thinking_header_facts(tok)
        assert facts["switch"] is True and facts["moves"] is True
        assert g.response_rows_expected(facts["off_ids"]) == 7 and g.response_rows_expected(facts["on_ids"]) == 5
        assert facts["off_ids"][:4] == facts["on_ids"][:4]      # the shared prefix; the tails differ
        # the gate's own derivation is the same fact, with no reason beside it (S-G3: the
        # non-fatal helper, so a template whose ON header cannot certify does not kill the run)
        assert g.thinking_header_for(*QWEN) == (facts, None)

    def test_helper_derives_the_on_header_and_refuses_all_turns_on_qwen(self):
        tok = _cached_tok(*QWEN)
        helper = _helper()
        from sisteer import chat_markers as cmk
        m = helper.registry_markers(tok, {"thinking": True, "all_turns": False}, cmk)
        facts = cmk.thinking_header_facts(tok)
        assert m["assistant_header_ids"] == facts["off_ids"] and m["assistant_header_ids_thinking_on"] == facts["on_ids"]
        assert m["assistant_turn_end_ids"] is None and m["thinking_header"] == facts
        with pytest.raises(RuntimeError, match="turn end cannot be located"):
            helper.registry_markers(tok, {"thinking": False, "all_turns": True}, cmk)


class TestRealGemma:
    def test_helper_derives_the_scaffold_and_records_the_equal_thinking_header(self):
        tok = _cached_tok(*GEMMA)
        helper = _helper()
        from sisteer import chat_markers as cmk
        m = helper.registry_markers(tok, {"thinking": False, "all_turns": True}, cmk)
        assert m["assistant_turn_end_ids"] == [tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("\n")]
        assert m["assistant_header_ids_thinking_on"] is None and len(m["assistant_header_ids"]) == 3
        assert m["thinking_header"]["moves"] is False
        # a @thinking cell on gemma-4: the ON header EQUALS the OFF header and is recorded as the
        # second header — admitted (plan P9's refusal withdrawn)
        m2 = helper.registry_markers(tok, {"thinking": True, "all_turns": False}, cmk)
        assert m2["assistant_header_ids_thinking_on"] == m2["assistant_header_ids"] == m["assistant_header_ids"]
        assert m2["thinking_header"]["moves"] is False and m2["thinking_header_error"] is None
        assert m2["thinking_header"]["on_ids"] == m2["thinking_header"]["off_ids"]
        assert g.turn_end_ids_for(*GEMMA) == m["assistant_turn_end_ids"]

    def test_the_gates_conversation_spans_and_the_two_probes(self):
        """The gate's own expectation for the 3-turn conversation equals the
        message-derived spans; the base cell's rows are the header's width;
        the quoted delimiter refuses lexically; the complete fake boundary
        passes the lexical locator (two historical spans — the blind spot)
        and is refused by the strict helper's control-token invariant."""
        tok = _cached_tok(*GEMMA)
        from sisteer import chat_markers as cmk
        header = tuple(cmk.generation_header_ids(tok))
        turn_end = tuple(cmk.assistant_turn_end_ids(tok))
        convo = list(g.ALL_TURNS_CONVERSATION)
        ids = cmk.render_ids(tok, convo, add_generation_prompt=True)
        spans = g.all_turns_spans_of(ids, header, turn_end)
        full, content = cmk.assistant_content_spans(tok, convo)
        final = sl.locate_final_header(ids, header, declared=False)
        assert spans == sl.expected_all_turns_spans(content, header_len=len(header), turn_end_len=len(turn_end),
                                                    final_start=final, prompt_len=len(ids))
        assert len(spans) == 3 and spans[-1] == (final, len(ids))
        rows = g.all_turns_rows_expected(spans)
        assert rows == sum(hi - lo for lo, hi in spans) > len(header) == g.response_rows_expected(list(header))
        # quoted delimiter: the lexical side refuses first
        quoted_ids = cmk.render_ids(tok, g.all_turns_probe_conversation("quoted"), add_generation_prompt=True)
        with pytest.raises(sl.SpanLocatorError) as e:
            g.all_turns_spans_of(quoted_ids, header, turn_end)
        assert e.value.reason == sl.REASON_SUCCESSOR_UNCERTIFIED
        # complete fake boundary: the lexical side returns an EXTRA span, the strict helper refuses
        fake = g.all_turns_probe_conversation("fake")
        fake_ids = cmk.render_ids(tok, fake, add_generation_prompt=True)
        fake_spans = g.all_turns_spans_of(fake_ids, header, turn_end)
        assert len(fake_spans) == len(spans) + 1          # the fake boundary splits one turn into two spans
        with pytest.raises(cmk.ChatMarkerError, match="control tokens survived inside the content span"):
            cmk.assistant_content_spans(tok, fake)

    def test_the_gates_probes_through_the_real_proxy(self, tmp_path):
        """The exact conversations the GPU case sends, through the real
        proxy's all-turns check on the real template: ok / render_refused /
        expected_unavailable naming the ChatMarkerError — and the base route
        never checked."""
        from test_all_turns_render_check import ALL_TURNS_VM, LENS_VM, _ask, _gemma_probe, _proxy, _refuses
        probe, _tok = _gemma_probe()
        proxy, backend = _proxy(tmp_path, probe)
        out = _ask(proxy, ALL_TURNS_VM, list(g.ALL_TURNS_CONVERSATION))
        assert out["petri"][g.LEDGER_KEY_ALL_TURNS_CHECK] == g.ALL_TURNS_CHECK_OK
        assert out["petri"][g.LEDGER_KEY_RESUMED_POLICY] == g.POLICY_DECLARED_SHARES_NOTHING
        msg = _refuses(proxy, ALL_TURNS_VM, g.all_turns_probe_conversation("quoted"), g.ALL_TURNS_RENDER_REFUSED)
        assert "complete turn-end scaffold" in msg
        msg = _refuses(proxy, ALL_TURNS_VM, g.all_turns_probe_conversation("fake"), g.ALL_TURNS_EXPECTED_UNAVAILABLE)
        assert "ChatMarkerError" in msg and "control tokens survived" in msg
        for probe_kind in ("quoted", "fake"):
            out = _ask(proxy, LENS_VM, g.all_turns_probe_conversation(probe_kind))
            assert g.LEDGER_KEY_ALL_TURNS_CHECK not in out["petri"]
        assert proxy.n_all_turns_check == {"ok": 1, "mismatch": 0, "refused": 1, "unavailable": 1, "failed": 0}


# --------------------------------------------------------------------------- #
# the truth tables                                                            #
# --------------------------------------------------------------------------- #


def _engine_id(rid):
    """The pinned vLLM's (0.25.1) INTERNAL request id for completion id ``rid``:
    ``InputProcessor.assign_request_id`` appends ``-{random_uuid():.8}`` — ``-``
    plus 8 lowercase hex — once per ``add_request``; the runner's
    ``input_batch.req_ids`` (the worker's SelectionRecord key) carries THAT,
    while the chat completion's ``id`` stays the external one. Deterministic
    here, from the id itself."""
    return f"{rid}-{hashlib.sha256(rid.encode()).hexdigest()[:8]}"


def _served(name, vmodel, sig, *, regime, policy, render, facts_on=None, rows=None, n_mod=None,
            lens=True, rid=None, extra_petri=None, requested=None, scale=None):
    rid = rid or f"chatcmpl-{name}"
    if lens and requested is None:
        assert vmodel.startswith(dc.VMODEL_PREFIX)
        requested = vmodel[len(dc.VMODEL_PREFIX):]          # the worker files the harness cell it was asked for
    entry = {"status": 200, "vmodel": vmodel, "lens_route": lens, "id": rid, "signature": sig,
             "text": "a healthy sentence of prose that repeats no three words in a row at all",
             "petri": {g.LEDGER_KEY_SALT_REGIME: regime, g.LEDGER_KEY_RESUMED_POLICY: policy,
                       g.LEDGER_KEY_RENDER_REGIME: render, **(extra_petri or {})}}
    if lens:
        # the worker keys its record by the ENGINE's id (``_engine_id``): a green
        # record carries the suffixed key the runtime produces, never the bare id
        entry["attest"] = {"n_entries": 1, "req_id": _engine_id(rid),
                           "facts": {"response_prefill": False, "agent_loop": False, "thinking_on": bool(facts_on)},
                           "prefill_masked_rows": rows, "n_applications_modifying": n_mod,
                           "n_applications": (n_mod or 0) + 2, "requested": requested,
                           # the worker files float(applied.effective_scale): 0.0 on the scale-0 twin
                           "applied_effective_scale": (0.0 if n_mod == 0 else 1.0) if scale is None else scale,
                           "matched": True, "stable": True}
        entry["claimed_failure"] = None
    return entry


def _refusal(vmodel, code, *, lens=True):
    return {"status": 400, "vmodel": vmodel, "lens_route": lens, "id": None, "code": code,
            "error": f"refused: {code}", "attest": {"n_entries": 0}}


def _cells(variant):
    hv = harness_cell_id(g.variant_cell(CELL, variant))
    return {"base_tt": CELL, "variant_tt": g.variant_cell(CELL, variant), "base": HB, "variant": hv,
            "twin": dc.NOOP_LENS_CELL_ID, "vm_base": f"petri__{HB}", "vm_variant": f"petri__{hv}",
            "vm_twin": f"petri__{dc.NOOP_LENS_CELL_ID}", "vm_passthrough": "petri__base"}


def _green_thinking():
    c = _cells("thinking")
    decl = g.expected_declarations("thinking")
    th = {"switch": True, "moves": True, "off_ids": list(OFF), "on_ids": list(ON)}
    amap = {c["variant"]: dict(decl), c["twin"]: dict(decl)}
    ep = {g.SERVE_DECLARATIONS_KEY: copy.deepcopy(amap), g.SERVE_DECLARATIONS_KEY + "_error": None,
          g.THINKING_HEADER_KEY: dict(th), g.THINKING_HEADER_ERROR_KEY: None, g.LENS_SALT_POLICY_KEY: "attested",
          g.OBJECT_SHA256_BY_CELL_KEY: {c["base"]: SHA, c["variant"]: SHA},
          "object_artifact_sha256": SHA, "object_artifact_sha256_bound": True, "object_artifact_sha256_error": None,
          "registry_markers": {"assistant_header_ids": list(OFF), "assistant_turn_end_ids": None,
                               "assistant_header_ids_thinking_on": list(ON)}}
    models = {g.SERVE_DECLARATIONS_KEY: copy.deepcopy(amap), g.THINKING_HEADER_KEY: dict(th),
              g.THINKING_HEADER_ERROR_KEY: None, g.LENS_SALT_POLICY_KEY: "attested",
              g.OBJECT_SHA256_BY_CELL_KEY: {c["base"]: SHA, c["variant"]: SHA}}
    ru, dcl, on, off = g.SALT_REGIME_REQUEST_UNIQUE, g.POLICY_DECLARED_SHARES_NOTHING, \
        g.RENDER_REGIME_THINKING_ON, g.RENDER_REGIME_THINKING_OFF
    reqs = {
        "on_steered": _served("on", c["vm_variant"], _sig("abc", ids=[1, 2, 3]), regime=ru, policy=dcl, render=on,
                              facts_on=True, rows=len(ON), n_mod=3),
        "on_twin": _served("twin", c["vm_twin"], _sig("xyz", ids=[7, 8, 9]), regime=ru, policy=dcl, render=on,
                           facts_on=True, rows=len(ON), n_mod=0),
        "off_base": _served("off", c["vm_base"], _sig("def", ids=[4, 5, 6]), regime=ru, policy=g.RESUMED_CONDITIONAL, render=off,
                            facts_on=False, rows=len(OFF), n_mod=2),
        "passthrough_on": _served("pt", "petri__base", _sig("xyz", ids=[7, 8, 9]), regime=g.SALT_REGIME_CELL_STABLE,
                                  policy=g.POLICY_NOT_CONSULTED, render=on, lens=False),
        "refuse_undeclared": _refusal(c["vm_variant"], g.THINKING_ON_UNDECLARED),
        "refuse_mismatch": _refusal(c["vm_variant"], g.THINKING_ON_MISMATCH),
        "refuse_kwargs_invalid": _refusal(c["vm_variant"], g.THINKING_KWARGS_INVALID),
        "refuse_cell_not_declared": _refusal(c["vm_base"], g.THINKING_ON_CELL_NOT_DECLARED),
        "refuse_not_lens": _refusal("petri__base", g.THINKING_ON_NOT_LENS, lens=False),
    }
    return {"case_id": g.DECLARED_THINKING, "variant": "thinking", "cells": c, "expected_declaration": decl,
            "object": {"cell": CELL, "sha256": SHA, "skip_bos": False}, "endpoint": ep, "models": models,
            "twin_donor": c["variant"], "tokenizer_thinking_header": dict(th), "tokenizer_thinking_header_error": None,
            "requests": reqs, "signature_errors": []}


def _green_carveout():
    c = _cells("all_positions")
    decl = g.expected_declarations("all_positions")
    amap = {c["variant"]: dict(decl)}
    P = 143
    markers = {"assistant_header_ids": list(GEMMA_HDR), "assistant_turn_end_ids": None,
               "assistant_header_ids_thinking_on": None}
    th = GEMMA_TH

    def ep(policy):
        return {g.SERVE_DECLARATIONS_KEY: copy.deepcopy(amap), g.SERVE_DECLARATIONS_KEY + "_error": None,
                g.THINKING_HEADER_KEY: dict(th), g.THINKING_HEADER_ERROR_KEY: None,
                g.LENS_SALT_POLICY_KEY: policy, g.OBJECT_SHA256_BY_CELL_KEY: {c["base"]: SHA, c["variant"]: SHA},
                "object_artifact_sha256": SHA, "object_artifact_sha256_bound": True,
                "object_artifact_sha256_error": None, "registry_markers": markers}

    def models(policy):
        return {g.SERVE_DECLARATIONS_KEY: copy.deepcopy(amap), g.THINKING_HEADER_KEY: dict(th),
                g.THINKING_HEADER_ERROR_KEY: None, g.LENS_SALT_POLICY_KEY: policy,
                g.OBJECT_SHA256_BY_CELL_KEY: {c["base"]: SHA, c["variant"]: SHA}}
    ru = g.SALT_REGIME_REQUEST_UNIQUE

    def legs(policy_variant, policy_other):
        return {"variant_single": _served("v", c["vm_variant"], _sig("abc", ids=[1, 2, 3]), regime=ru, policy=policy_variant,
                                          render="unspecified", rows=P, n_mod=5),
                "base_single": _served("b", c["vm_base"], _sig("def", ids=[4, 5, 6]), regime=ru, policy=policy_other,
                                       render="unspecified", rows=len(GEMMA_HDR), n_mod=2),
                "twin_single": _served("t", c["vm_twin"], _sig("xyz", ids=[7, 8, 9]), regime=ru, policy=policy_other,
                                       render="unspecified", rows=len(GEMMA_HDR), n_mod=0)}
    rec = {"case_id": g.ALL_POSITIONS_CARVEOUT, "variant": "all_positions", "cells": c,
           "expected_declaration": decl, "object": {"cell": CELL, "sha256": SHA, "skip_bos": False},
           "endpoint": ep("attested"), "models": models("attested"), "twin_donor": c["base"],
           "guard": {"base_positions": "response",
                     "declared": {"adapted": True, "served_positions": ["all"],
                                  "experiment": "fit_response_serve_all_positions"},
                     "undeclared": {"refused": True,
                                    "error": "AdapterError: ... declare `serve_experiment: "
                                             "fit_response_serve_all_positions` on the CELL ..."}},
           "prompt_len": P, "expected_all_rows": P,
           "requests": legs(g.POLICY_DECLARED_SHARES_NOTHING, g.RESUMED_CONDITIONAL),
           "degeneracy": {"variant_single": g.degeneracy_instrument("a healthy sentence of prose that repeats nothing"),
                          "base_single": g.degeneracy_instrument("another healthy sentence of prose"),
                          "twin_single": g.degeneracy_instrument("a third healthy sentence of prose")},
           "signature_errors": [],
           "forced": {"cells": c, "object": {"cell": CELL, "sha256": SHA, "skip_bos": False},
                      "endpoint": ep("request_unique_all"), "models": models("request_unique_all"),
                      "prompt_len": P, "expected_all_rows": P,
                      "requests": legs(g.POLICY_FORCED_REQUEST_UNIQUE, g.POLICY_FORCED_REQUEST_UNIQUE),
                      "signature_errors": []}}
    return rec


def _green_all_turns():
    c = _cells("all_turns")
    decl = g.expected_declarations("all_turns")
    amap = {c["variant"]: dict(decl)}
    turn_end = [990, 77]
    markers = {"assistant_header_ids": list(GEMMA_HDR), "assistant_turn_end_ids": turn_end,
               "assistant_header_ids_thinking_on": None}
    spans = [[8, 15], [22, 29], [36, 39]]
    ru = g.SALT_REGIME_REQUEST_UNIQUE
    reqs = {
        "convo_variant": _served("cv", c["vm_variant"], _sig("abc", ids=[1, 2, 3]), regime=ru, policy=g.POLICY_DECLARED_SHARES_NOTHING,
                                 render="unspecified", rows=17, n_mod=6,
                                 extra_petri={g.LEDGER_KEY_ALL_TURNS_CHECK: g.ALL_TURNS_CHECK_OK}),
        "convo_base": _served("cb", c["vm_base"], _sig("def", ids=[4, 5, 6]), regime=ru, policy=g.RESUMED_CONDITIONAL,
                              render="unspecified", rows=len(GEMMA_HDR), n_mod=2),
        "convo_twin": _served("ct", c["vm_twin"], _sig("xyz", ids=[7, 8, 9]), regime=ru, policy=g.RESUMED_CONDITIONAL,
                              render="unspecified", rows=len(GEMMA_HDR), n_mod=0),
        "quoted_variant": _refusal(c["vm_variant"], g.ALL_TURNS_RENDER_REFUSED),
        "quoted_base": _served("qb", c["vm_base"], _sig("ghi", ids=[10, 11, 12]), regime=ru, policy=g.RESUMED_CONDITIONAL,
                               render="unspecified", rows=len(GEMMA_HDR), n_mod=2),
        "fake_variant": dict(_refusal(c["vm_variant"], g.ALL_TURNS_EXPECTED_UNAVAILABLE),
                             error="the message-derived spans are unavailable: ChatMarkerError: assistant message 1: "
                                   "control tokens survived inside the content span: [(13, 106), (15, 105)]"),
        "fake_base": _served("fb", c["vm_base"], _sig("jkl", ids=[13, 14, 15]), regime=ru, policy=g.RESUMED_CONDITIONAL,
                             render="unspecified", rows=len(GEMMA_HDR), n_mod=2),
    }
    return {"case_id": g.ALL_TURNS_RENDER_CHECK, "variant": "all_turns", "cells": c, "expected_declaration": decl,
            "object": {"cell": CELL, "sha256": SHA, "skip_bos": False},
            "endpoint": {g.SERVE_DECLARATIONS_KEY: copy.deepcopy(amap), g.SERVE_DECLARATIONS_KEY + "_error": None,
                         g.THINKING_HEADER_KEY: dict(GEMMA_TH), g.THINKING_HEADER_ERROR_KEY: None,
                         g.LENS_SALT_POLICY_KEY: "attested", "registry_markers": markers,
                         g.OBJECT_SHA256_BY_CELL_KEY: {c["base"]: SHA, c["variant"]: SHA},
                         "object_artifact_sha256": SHA, "object_artifact_sha256_bound": True,
                         "object_artifact_sha256_error": None},
            "models": {g.SERVE_DECLARATIONS_KEY: copy.deepcopy(amap), g.THINKING_HEADER_KEY: dict(GEMMA_TH),
                       g.THINKING_HEADER_ERROR_KEY: None, g.LENS_SALT_POLICY_KEY: "attested",
                       g.OBJECT_SHA256_BY_CELL_KEY: {c["base"]: SHA, c["variant"]: SHA}},
            "twin_donor": c["base"], "tokenizer_turn_end_ids": list(turn_end),
            "prompt_len": 39, "located_spans": spans, "expected_all_turns_rows": 17,
            "requests": reqs, "signature_errors": []}


GREEN = {g.DECLARED_THINKING: _green_thinking, g.ALL_POSITIONS_CARVEOUT: _green_carveout,
         g.ALL_TURNS_RENDER_CHECK: _green_all_turns}


def _failing(checks):
    return sorted(k for k, c in checks.items() if not c["ok"])


class TestTruthTables:
    @pytest.mark.parametrize("cid", g.DECLARED_RENDER_CASES)
    def test_green_records_pass_every_check(self, cid):
        checks = g.declared_render_checks(cid, GREEN[cid]())
        assert _failing(checks) == [], _failing(checks)
        assert all(isinstance(c["detail"], str) and c["detail"] for c in checks.values())
        assert all(c["ok"] is True for c in checks.values())      # never None: nothing here is inconclusive

    @pytest.mark.parametrize("cid", g.DECLARED_RENDER_CASES)
    def test_an_empty_record_fails_every_check(self, cid):
        checks = g.declared_render_checks(cid, {})
        assert all(c["ok"] is False for c in checks.values())
        assert len(checks) == len(g.DECLARED_RENDER_CHECKERS[cid](GREEN[cid]()))   # the key set is fixed

    def test_unknown_case_refuses(self):
        with pytest.raises(g.GateError, match="not a declared-render case"):
            g.declared_render_checks("psr:relu_gated_vector:gemv:all:skipbos0:both", {})

    THINKING_PERTURBATIONS = [
        # (path to mutate, new value, the EXACT set of checks that must fail)
        (("requests", "on_steered", "claimed_failure"), {"error_type": "AdapterError"},
         {"thinking_on_served", "thinking_claims_empty"}),
        (("requests", "on_steered", "attest", "n_entries"), 2, {"thinking_on_one_entry"}),
        (("requests", "on_steered", "attest", "req_id"), "chatcmpl-other", {"thinking_on_one_entry"}),
        (("requests", "on_steered", "attest", "facts", "thinking_on"), False, {"thinking_on_facts"}),
        (("requests", "on_steered", "attest", "prefill_masked_rows"), len(OFF), {"thinking_on_masked_rows"}),
        (("requests", "on_steered", "attest", "n_applications_modifying"), 0, {"thinking_on_modifying"}),
        (("requests", "on_steered", "petri", g.LEDGER_KEY_RESUMED_POLICY), g.RESUMED_CONDITIONAL, {"thinking_on_stamp"}),
        (("requests", "on_steered", "petri", g.LEDGER_KEY_RENDER_REGIME), g.RENDER_REGIME_THINKING_OFF,
         {"thinking_on_stamp"}),
        (("requests", "on_steered", "petri", g.LEDGER_KEY_SALT_REGIME), g.SALT_REGIME_CELL_STABLE, {"thinking_on_stamp"}),
        (("requests", "on_twin", "attest", "n_applications_modifying"), 1, {"thinking_twin_modifying"}),
        # AG-1: the twin's selection and scale, the ON steered cell's and the OFF base's selection
        (("requests", "on_twin", "attest", "applied_effective_scale"), 1.0, {"thinking_twin_scale"}),
        (("requests", "on_twin", "attest", "requested"), HB, {"thinking_twin_selection"}),
        (("requests", "on_steered", "attest", "matched"), False, {"thinking_on_selection"}),
        (("requests", "off_base", "attest", "requested"), HV_THINKING, {"thinking_off_selection"}),
        (("requests", "off_base", "attest", "n_applications"), 0, {"thinking_off_selection"}),
        (("requests", "on_twin", "attest", "prefill_masked_rows"), len(OFF), {"thinking_twin_masked_rows"}),
        (("requests", "on_twin", "attest", "facts", "thinking_on"), False, {"thinking_twin_facts"}),
        (("requests", "on_twin", "signature"), _sig("abc", ids=[1, 2, 3]), {"thinking_bites_on"}),
        (("requests", "off_base", "attest", "facts", "thinking_on"), True, {"thinking_off_facts"}),
        (("requests", "off_base", "attest", "prefill_masked_rows"), len(ON), {"thinking_off_masked_rows"}),
        (("requests", "off_base", "attest", "n_applications_modifying"), 0, {"thinking_off_modifying"}),
        (("requests", "off_base", "petri", g.LEDGER_KEY_RENDER_REGIME), g.RENDER_REGIME_THINKING_ON,
         {"thinking_off_stamp"}),
        (("requests", "off_base", "petri", g.LEDGER_KEY_RESUMED_POLICY), g.POLICY_DECLARED_SHARES_NOTHING,
         {"thinking_off_stamp"}),
        (("requests", "refuse_undeclared", "code"), g.THINKING_ON_MISMATCH, {"thinking_refuse_undeclared"}),
        # a 200 where a refusal was expected is ALSO an unclaimed served lens response
        (("requests", "refuse_undeclared", "status"), 200, {"thinking_refuse_undeclared", "thinking_claims_empty"}),
        (("requests", "refuse_undeclared", "attest", "n_entries"), 1, {"thinking_refuse_undeclared"}),
        (("requests", "refuse_mismatch", "code"), g.THINKING_ON_UNDECLARED, {"thinking_refuse_mismatch"}),
        (("requests", "refuse_kwargs_invalid", "code"), None, {"thinking_refuse_kwargs_invalid"}),
        (("requests", "refuse_cell_not_declared", "code"), g.THINKING_ON_UNDECLARED,
         {"thinking_refuse_cell_not_declared"}),
        (("requests", "refuse_not_lens", "status"), 500, {"thinking_refuse_not_lens"}),
        (("requests", "passthrough_on", "petri", g.LEDGER_KEY_RESUMED_POLICY), g.RESUMED_CONDITIONAL,
         {"thinking_passthrough_on_stamp"}),
        (("requests", "passthrough_on", "status"), 400, {"thinking_passthrough_on_served"}),
        (("endpoint", "object_artifact_sha256"), "cd" * 32, {"thinking_artifact_sha256"}),
        (("endpoint", "object_artifact_sha256_bound"), False, {"thinking_artifact_sha256"}),
        # an endpoint fact that /v1/models does not echo fails the cross-surface check too (S-G3)
        (("endpoint", g.THINKING_HEADER_KEY, "moves"), False,
         {"thinking_header_matches_tokenizer", "thinking_header_attested"}),
        (("endpoint", g.THINKING_HEADER_KEY, "on_ids"), list(OFF),
         {"thinking_header_matches_tokenizer", "thinking_on_masked_rows", "thinking_twin_masked_rows",
          "thinking_header_attested"}),
        (("models", g.THINKING_HEADER_KEY, "on_ids"), [1, 2],
         {"thinking_header_matches_tokenizer", "thinking_header_attested"}),
        # S-G2: the proxy must publish the EXACT digest map, not just the endpoint
        (("models", g.OBJECT_SHA256_BY_CELL_KEY), None, {"thinking_artifact_sha256"}),
        (("models", g.OBJECT_SHA256_BY_CELL_KEY), {HB: SHA}, {"thinking_artifact_sha256"}),
        (("models", g.OBJECT_SHA256_BY_CELL_KEY, HV_THINKING), "cd" * 32, {"thinking_artifact_sha256"}),
        (("endpoint", g.OBJECT_SHA256_BY_CELL_KEY), None, {"thinking_artifact_sha256"}),
        # S-G3: the reason rides beside the fact in BOTH surfaces, and pairs with it
        (("endpoint", g.THINKING_HEADER_ERROR_KEY), PARTIAL_REASON, {"thinking_header_attested"}),
        (("models", g.THINKING_HEADER_ERROR_KEY), PARTIAL_REASON, {"thinking_header_attested"}),
        (("tokenizer_thinking_header_error",), PARTIAL_REASON, {"thinking_header_matches_tokenizer"}),
        (("tokenizer_thinking_header",), dict(PARTIAL_TH), {"thinking_header_matches_tokenizer"}),
        (("twin_donor",), HB, {"thinking_twin_donor"}),
        (("signature_errors",), ["on_steered: no logprobs"], {"thinking_signatures"}),
    ]

    @pytest.mark.parametrize("path, value, expect", THINKING_PERTURBATIONS)
    def test_declared_thinking_perturbations(self, path, value, expect):
        rec = _green_thinking()
        _set(rec, path, value)
        assert set(_failing(g.declared_thinking_checks(rec))) == expect, path

    def test_declared_thinking_attestation_map_exactness(self):
        rec = _green_thinking()
        c = rec["cells"]
        rec["endpoint"][g.SERVE_DECLARATIONS_KEY][c["base"]] = g.expected_declarations("thinking")   # the base present
        assert _failing(g.declared_thinking_checks(rec)) == ["thinking_attestation_map"]
        rec = _green_thinking()
        del rec["models"][g.SERVE_DECLARATIONS_KEY][c["twin"]]                                        # models lacks the twin
        assert _failing(g.declared_thinking_checks(rec)) == ["thinking_attestation_map"]
        rec = _green_thinking()
        del rec["endpoint"][g.SERVE_DECLARATIONS_KEY]                                                 # unattested
        rec["endpoint"][g.SERVE_DECLARATIONS_KEY + "_error"] = "worker report 0 carries no lens_serve_declarations"
        checks = g.declared_thinking_checks(rec)
        assert _failing(checks) == ["thinking_attestation_map"]
        assert "carries no lens_serve_declarations" in checks["thinking_attestation_map"]["detail"]
        rec = _green_thinking()
        rec["endpoint"][g.OBJECT_SHA256_BY_CELL_KEY] = {c["base"]: SHA}                              # variant missing
        assert _failing(g.declared_thinking_checks(rec)) == ["thinking_artifact_sha256"]

    CARVEOUT_PERTURBATIONS = [
        (("guard", "declared", "adapted"), False, {"carveout_guard_admits_declared"}),
        (("guard", "declared", "served_positions"), ["response"], {"carveout_guard_admits_declared"}),
        (("guard", "undeclared", "refused"), False, {"carveout_guard_refuses_undeclared"}),
        (("guard", "undeclared", "error"), "AdapterError: some other refusal", {"carveout_guard_refuses_undeclared"}),
        (("requests", "variant_single", "attest", "prefill_masked_rows"), 3, {"carveout_variant_masked_rows"}),
        (("expected_all_rows",), None, {"carveout_variant_masked_rows"}),
        (("requests", "variant_single", "attest", "n_applications_modifying"), 0, {"carveout_variant_modifying"}),
        (("requests", "variant_single", "petri", g.LEDGER_KEY_RESUMED_POLICY), "always", {"carveout_variant_stamp"}),
        (("requests", "variant_single", "petri", g.LEDGER_KEY_SALT_REGIME), g.SALT_REGIME_CELL_STABLE,
         {"carveout_variant_stamp"}),
        (("requests", "base_single", "attest", "prefill_masked_rows"), 143, {"carveout_base_masked_rows"}),
        (("requests", "base_single", "petri", g.LEDGER_KEY_RESUMED_POLICY), g.POLICY_DECLARED_SHARES_NOTHING,
         {"carveout_base_stamp"}),
        (("requests", "twin_single", "attest", "n_applications_modifying"), 2, {"carveout_twin_modifying"}),
        # AG-1: the default-policy legs carry the validator's evidence too
        (("requests", "twin_single", "attest", "applied_effective_scale"), 12.0, {"carveout_twin_scale"}),
        (("requests", "twin_single", "attest", "requested"), HB, {"carveout_twin_selection"}),
        (("requests", "twin_single", "attest", "prefill_masked_rows"), 143, {"carveout_twin_masked_rows"}),
        (("requests", "twin_single", "attest", "n_entries"), 2, {"carveout_twin_one_entry"}),
        (("requests", "base_single", "attest", "n_entries"), 0, {"carveout_base_one_entry"}),
        (("requests", "base_single", "attest", "stable"), False, {"carveout_base_selection"}),
        (("requests", "variant_single", "attest", "requested"), HB, {"carveout_variant_selection"}),
        (("requests", "twin_single", "signature"), _sig("abc", ids=[1, 2, 3]), {"carveout_bites"}),
        (("requests", "variant_single", "claimed_failure"), {"error_type": "X"},
         {"carveout_variant_served", "carveout_claims_empty"}),
        (("endpoint", g.LENS_SALT_POLICY_KEY), "request_unique_all", {"carveout_attested_policy_endpoint"}),
        (("degeneracy", "variant_single"), {"ran": False, "error": "no text"}, {"carveout_degeneracy_instrument"}),
        (("forced", "endpoint", g.LENS_SALT_POLICY_KEY), "attested", {"carveout_forced_policy_endpoint"}),
        (("forced", "models", g.LENS_SALT_POLICY_KEY), None, {"carveout_forced_policy_endpoint"}),
        (("forced", "requests", "base_single", "petri", g.LEDGER_KEY_RESUMED_POLICY), g.RESUMED_CONDITIONAL,
         {"carveout_forced_base_stamp"}),
        (("forced", "requests", "twin_single", "petri", g.LEDGER_KEY_SALT_REGIME), g.SALT_REGIME_CELL_STABLE,
         {"carveout_forced_twin_stamp"}),
        (("forced", "requests", "variant_single", "status"), 500,
         {"carveout_forced_variant_served"}),
        (("forced", "requests", "twin_single", "attest", "applied_effective_scale"), 1.0,
         {"carveout_forced_twin_scale"}),
        (("twin_donor",), "si_safety__psr_kl_all@all_positions", {"carveout_twin_donor"}),
        (("forced", "signature_errors"), ["x"], {"carveout_signatures"}),
        # S-G1: every forced leg carries the worker's positive evidence — the fields left in
        # place, the one perturbed fact fails exactly its check (the realistic no-entry shape
        # is test_forced_legs_that_never_reached_the_worker_fail_the_case)
        (("forced", "requests", "variant_single", "attest", "n_entries"), 0, {"carveout_forced_variant_one_entry"}),
        (("forced", "requests", "variant_single", "attest", "n_entries"), 2, {"carveout_forced_variant_one_entry"}),
        (("forced", "requests", "base_single", "attest", "n_entries"), 0, {"carveout_forced_base_one_entry"}),
        (("forced", "requests", "base_single", "attest", "n_entries"), 2, {"carveout_forced_base_one_entry"}),
        (("forced", "requests", "twin_single", "attest", "n_entries"), 0, {"carveout_forced_twin_one_entry"}),
        (("forced", "requests", "twin_single", "attest", "n_entries"), 2, {"carveout_forced_twin_one_entry"}),
        (("forced", "requests", "variant_single", "attest", "req_id"), "chatcmpl-other",
         {"carveout_forced_variant_one_entry"}),
        (("forced", "requests", "variant_single", "attest", "requested"), HB, {"carveout_forced_variant_selection"}),
        (("forced", "requests", "base_single", "attest", "requested"), HV_ALL_POS, {"carveout_forced_base_selection"}),
        (("forced", "requests", "twin_single", "attest", "requested"), HB, {"carveout_forced_twin_selection"}),
        (("forced", "requests", "variant_single", "attest", "matched"), False, {"carveout_forced_variant_selection"}),
        (("forced", "requests", "base_single", "attest", "stable"), False, {"carveout_forced_base_selection"}),
        (("forced", "requests", "twin_single", "attest", "matched"), None, {"carveout_forced_twin_selection"}),
        (("forced", "requests", "variant_single", "attest", "n_applications"), 0, {"carveout_forced_variant_selection"}),
        (("forced", "requests", "variant_single", "attest", "prefill_masked_rows"), 3,
         {"carveout_forced_variant_masked_rows"}),
        (("forced", "expected_all_rows"), None, {"carveout_forced_variant_masked_rows"}),
        (("forced", "requests", "base_single", "attest", "prefill_masked_rows"), 143,
         {"carveout_forced_base_masked_rows"}),
        (("forced", "requests", "twin_single", "attest", "prefill_masked_rows"), 143,
         {"carveout_forced_twin_masked_rows"}),
        (("forced", "requests", "variant_single", "attest", "n_applications_modifying"), 0,
         {"carveout_forced_variant_modifying"}),
        (("forced", "requests", "base_single", "attest", "n_applications_modifying"), 0,
         {"carveout_forced_base_modifying"}),
        (("forced", "requests", "twin_single", "attest", "n_applications_modifying"), 2,
         {"carveout_forced_twin_modifying"}),
        # S-G2: the artifact binding is checked on BOTH standups, in endpoint.json AND /v1/models
        (("endpoint", "object_artifact_sha256_bound"), False, {"carveout_artifact_sha256"}),
        (("endpoint", g.OBJECT_SHA256_BY_CELL_KEY), None, {"carveout_artifact_sha256"}),
        (("models", g.OBJECT_SHA256_BY_CELL_KEY), {}, {"carveout_artifact_sha256"}),
        (("forced", "endpoint", "object_artifact_sha256_bound"), False, {"carveout_forced_artifact_sha256"}),
        (("forced", "endpoint", "object_artifact_sha256"), "cd" * 32, {"carveout_forced_artifact_sha256"}),
        (("forced", "models", g.OBJECT_SHA256_BY_CELL_KEY), None, {"carveout_forced_artifact_sha256"}),
        # S-G3: the fact/reason pair agrees across endpoint.json and /v1/models, on both standups
        (("endpoint", g.THINKING_HEADER_ERROR_KEY), PARTIAL_REASON, {"carveout_thinking_header_attested"}),
        (("models", g.THINKING_HEADER_KEY), None, {"carveout_thinking_header_attested"}),
        (("forced", "models", g.THINKING_HEADER_ERROR_KEY), PARTIAL_REASON,
         {"carveout_forced_thinking_header_attested"}),
        (("forced", "endpoint", g.THINKING_HEADER_KEY), None, {"carveout_forced_thinking_header_attested"}),
    ]

    @pytest.mark.parametrize("leg, cell_role", [("variant_single", "variant"), ("base_single", "base"),
                                                ("twin_single", "twin")])
    @pytest.mark.parametrize("n_entries", [0, 2])
    def test_forced_legs_that_never_reached_the_worker_fail_the_case(self, leg, cell_role, n_entries):
        """S-G1's scenario, in the shape the drive really records: a forced leg
        answered 200 with the right proxy stamp but the reset SelectionRecord
        held zero (or two) entries — ``one_declared_attestation`` files the
        count and NO fields. Every worker-evidence check of that leg fails and
        the case exits 2; the stamp check alone would have passed it."""
        rec = _green_carveout()
        requests = {} if n_entries == 0 else {"a": {}, "b": {}}
        rec["forced"]["requests"][leg]["attest"] = g.one_declared_attestation(requests)
        assert rec["forced"]["requests"][leg]["attest"]["n_entries"] == n_entries
        checks = g.all_positions_carveout_checks(rec)
        role = leg.split("_")[0]
        want = {f"carveout_forced_{role}_one_entry", f"carveout_forced_{role}_selection",
                f"carveout_forced_{role}_masked_rows", f"carveout_forced_{role}_modifying"}
        if role == "twin":
            want.add("carveout_forced_twin_scale")                              # AG-1: the twin's scale too
        assert set(_failing(checks)) == want
        assert checks[f"carveout_forced_{role}_stamp"]["ok"] is True          # what the case used to rest on
        assert f"held {n_entries!r} entries" in checks[f"carveout_forced_{role}_one_entry"]["detail"]
        kr = dict(rec, checks=checks, kind="declared_render", arms=[g.DECLARED_RENDER_ARM])
        assert g.overall_exit_code({"cases": [kr], "complete": True}) == 2

    def test_forced_legs_carry_the_default_legs_evidence(self):
        """Same evidence on both standups: the forced variant's rows are P (minus BOS
        iff skipped) from the FORCED standup's own render, the base's and the
        twin's the header's width (the twin's donor is the base, P12); modifying
        >= 1 on base and variant, 0 on the twin; requested is the harness cell."""
        rec = _green_carveout()
        checks = g.all_positions_carveout_checks(rec)
        want = {f"carveout_forced_{r}_{k}" for r in ("variant", "base", "twin")
                for k in ("stamp", "served", "one_entry", "selection", "masked_rows", "modifying")} | {
            "carveout_forced_twin_scale"}
        assert want <= set(checks) and all(checks[k]["ok"] for k in want)
        assert "requested='si_safety__psr_kl_all@all_positions'" in checks["carveout_forced_variant_selection"]["detail"]
        assert "requested='none__lens0'" in checks["carveout_forced_twin_selection"]["detail"]
        # the forced standup's geometry is read from ITS record, not the default standup's
        rec["forced"]["expected_all_rows"], rec["expected_all_rows"] = 200, rec["expected_all_rows"]
        assert _failing(g.all_positions_carveout_checks(rec)) == ["carveout_forced_variant_masked_rows"]
        rec = _green_carveout()
        rec["forced"]["object"]["skip_bos"] = True
        rec["forced"]["expected_all_rows"] = g.all_positions_rows_expected(rec["forced"]["prompt_len"], skip_bos=True)
        assert _failing(g.all_positions_carveout_checks(rec)) == ["carveout_forced_variant_masked_rows"]
        rec["forced"]["requests"]["variant_single"]["attest"]["prefill_masked_rows"] = rec["forced"]["prompt_len"] - 1
        assert _failing(g.all_positions_carveout_checks(rec)) == []

    @pytest.mark.parametrize("path, value, expect", CARVEOUT_PERTURBATIONS)
    def test_carveout_perturbations(self, path, value, expect):
        rec = _green_carveout()
        _set(rec, path, value)
        assert set(_failing(g.all_positions_carveout_checks(rec))) == expect, path

    def test_carveout_base_must_stay_absent_from_both_maps(self):
        rec = _green_carveout()
        rec["forced"]["endpoint"][g.SERVE_DECLARATIONS_KEY][rec["cells"]["base"]] = {"serve_experiment":
                                                                                    "fit_response_serve_all_positions"}
        assert _failing(g.all_positions_carveout_checks(rec)) == ["carveout_forced_attestation_map"]
        rec = _green_carveout()
        rec["endpoint"][g.SERVE_DECLARATIONS_KEY][rec["cells"]["twin"]] = g.expected_declarations("all_positions")
        assert _failing(g.all_positions_carveout_checks(rec)) == ["carveout_attestation_map"]
        # the skip_bos rule: the expectation is P-1 when the object skips the BOS row
        rec = _green_carveout()
        rec["object"]["skip_bos"] = True
        rec["expected_all_rows"] = g.all_positions_rows_expected(rec["prompt_len"], skip_bos=True)
        assert _failing(g.all_positions_carveout_checks(rec)) == ["carveout_variant_masked_rows"]
        rec["requests"]["variant_single"]["attest"]["prefill_masked_rows"] = rec["prompt_len"] - 1
        assert _failing(g.all_positions_carveout_checks(rec)) == []

    ALL_TURNS_PERTURBATIONS = [
        (("requests", "convo_variant", "petri", g.LEDGER_KEY_ALL_TURNS_CHECK), "mismatch", {"all_turns_check_label_ok"}),
        (("requests", "convo_variant", "attest", "prefill_masked_rows"), 3, {"all_turns_convo_masked_rows"}),
        (("expected_all_turns_rows",), None, {"all_turns_convo_masked_rows"}),
        (("requests", "convo_variant", "attest", "n_applications_modifying"), 0, {"all_turns_convo_modifying"}),
        (("requests", "convo_variant", "petri", g.LEDGER_KEY_RESUMED_POLICY), g.RESUMED_CONDITIONAL,
         {"all_turns_stamp"}),
        (("requests", "convo_base", "attest", "prefill_masked_rows"), 17, {"all_turns_base_masked_rows"}),
        (("requests", "convo_base", "petri", g.LEDGER_KEY_ALL_TURNS_CHECK), "ok", {"all_turns_base_unlabelled"}),
        (("requests", "quoted_variant", "code"), g.ALL_TURNS_EXPECTED_UNAVAILABLE, {"all_turns_refuse_quoted"}),
        (("requests", "quoted_variant", "attest", "n_entries"), 1, {"all_turns_refuse_quoted"}),
        # AG-1: the refusal-probe CONTROLS are positive lens legs — a 400 there is a missing control
        (("requests", "quoted_base", "status"), 400, {"all_turns_quoted_base_served"}),
        (("requests", "fake_variant", "code"), g.ALL_TURNS_RENDER_REFUSED, {"all_turns_refuse_fake"}),
        (("requests", "fake_variant", "error"), "the message-derived spans are unavailable: something else",
         {"all_turns_refuse_fake"}),
        (("requests", "fake_base", "status"), 500, {"all_turns_fake_base_served"}),
        (("requests", "quoted_base", "attest", "requested"), "none__lens0", {"all_turns_quoted_base_selection"}),
        (("requests", "fake_base", "attest", "prefill_masked_rows"), 17, {"all_turns_fake_base_masked_rows"}),
        (("requests", "fake_base", "petri", g.LEDGER_KEY_RESUMED_POLICY), g.POLICY_DECLARED_SHARES_NOTHING,
         {"all_turns_fake_base_stamp"}),
        # AG-1: the twin the bites comparison rests on
        (("requests", "convo_twin", "attest", "n_applications_modifying"), 20, {"all_turns_twin_modifying"}),
        (("requests", "convo_twin", "attest", "applied_effective_scale"), 1.0, {"all_turns_twin_scale"}),
        (("requests", "convo_twin", "attest", "requested"), HB, {"all_turns_twin_selection"}),
        (("requests", "convo_twin", "attest", "prefill_masked_rows"), 17, {"all_turns_twin_masked_rows"}),
        (("requests", "convo_twin", "attest", "n_entries"), 0, {"all_turns_twin_one_entry"}),
        (("requests", "convo_twin", "petri", g.LEDGER_KEY_SALT_REGIME), g.SALT_REGIME_CELL_STABLE,
         {"all_turns_twin_stamp"}),
        (("requests", "convo_twin", "status"), 500, {"all_turns_twin_served"}),   # the fixture keeps its signature
        (("endpoint", "registry_markers", "assistant_turn_end_ids"), [990], {"all_turns_scaffold"}),
        (("endpoint", "registry_markers", "assistant_turn_end_ids"), None, {"all_turns_scaffold"}),
        (("tokenizer_turn_end_ids",), None, {"all_turns_scaffold"}),
        (("requests", "convo_twin", "signature"), _sig("abc", ids=[1, 2, 3]), {"all_turns_bites"}),
        (("twin_donor",), "si_safety__psr_kl_all@all_turns", {"all_turns_twin_donor"}),
        (("requests", "convo_variant", "claimed_failure"), {"error_type": "X"},
         {"all_turns_convo_served", "all_turns_claims_empty"}),
        # S-G2: the gemma cases check the artifact binding too, in both surfaces
        (("endpoint", "object_artifact_sha256_bound"), False, {"all_turns_artifact_sha256"}),
        (("endpoint", "object_artifact_sha256"), None, {"all_turns_artifact_sha256"}),
        (("endpoint", g.OBJECT_SHA256_BY_CELL_KEY, HB), "cd" * 32, {"all_turns_artifact_sha256"}),
        (("models", g.OBJECT_SHA256_BY_CELL_KEY), None, {"all_turns_artifact_sha256"}),
        # S-G3
        (("models", g.THINKING_HEADER_ERROR_KEY), PARTIAL_REASON, {"all_turns_thinking_header_attested"}),
        (("models", g.THINKING_HEADER_KEY, "moves"), True, {"all_turns_thinking_header_attested"}),
    ]

    @pytest.mark.parametrize("cid", g.DECLARED_RENDER_CASES)
    def test_artifact_identity_absent_from_the_proxy_fails_every_case(self, cid):
        """S-G2's consequence closed: a proxy that never received the digest map
        publishes none in ``/v1/models`` — the case fails naming the surface,
        even though endpoint.json still carries the map."""
        prefix = {g.DECLARED_THINKING: "thinking", g.ALL_POSITIONS_CARVEOUT: "carveout",
                  g.ALL_TURNS_RENDER_CHECK: "all_turns"}[cid]
        rec = GREEN[cid]()
        del rec["models"][g.OBJECT_SHA256_BY_CELL_KEY]
        checks = g.declared_render_checks(cid, rec)
        assert _failing(checks) == [f"{prefix}_artifact_sha256"]
        assert "/v1/models" in checks[f"{prefix}_artifact_sha256"]["detail"]
        # ...and an endpoint that says null + unbound (the helper's shape when the served tree
        # dropped the kwarg) fails naming the reason it recorded
        rec = GREEN[cid]()
        rec["endpoint"][g.OBJECT_SHA256_BY_CELL_KEY] = None
        rec["endpoint"]["object_artifact_sha256"] = None
        rec["endpoint"]["object_artifact_sha256_bound"] = False
        rec["endpoint"]["object_artifact_sha256_error"] = "the served sisteer's RoutingProxy takes no object_artifact_sha256_by_cell"
        checks = g.declared_render_checks(cid, rec)
        assert _failing(checks) == [f"{prefix}_artifact_sha256"]
        assert "takes no object_artifact_sha256_by_cell" in checks[f"{prefix}_artifact_sha256"]["detail"]

    @pytest.mark.parametrize("cid", g.DECLARED_RENDER_CASES)
    def test_thinking_header_pairing_is_verified_in_both_surfaces(self, cid):
        """S-G3's contract at the gate: the fact and the reason agree between
        endpoint.json and /v1/models, and pair — a PARTIAL fact (switch, no
        on_ids) carries a reason, a complete one carries None. A partial fact
        honestly paired is CONSISTENT (the gemma cases never need the ON
        header); only the thinking case additionally needs certification."""
        key = {g.DECLARED_THINKING: "thinking_header_attested",
               g.ALL_POSITIONS_CARVEOUT: "carveout_thinking_header_attested",
               g.ALL_TURNS_RENDER_CHECK: "all_turns_thinking_header_attested"}[cid]
        rec = GREEN[cid]()
        for surface in ("endpoint", "models"):
            rec[surface][g.THINKING_HEADER_KEY] = dict(PARTIAL_TH)
        checks = g.declared_render_checks(cid, rec)
        assert key in _failing(checks) and "no reason" in checks[key]["detail"]           # partial, unexplained
        for surface in ("endpoint", "models"):
            rec[surface][g.THINKING_HEADER_ERROR_KEY] = PARTIAL_REASON
        checks = g.declared_render_checks(cid, rec)
        assert checks[key]["ok"] is True                                                    # partial, explained
        if cid == g.DECLARED_THINKING:
            # ...but the thinking case's certification and its derived row counts are gone
            assert {"thinking_header_matches_tokenizer", "thinking_on_masked_rows",
                    "thinking_twin_masked_rows"} <= set(_failing(checks))
        else:
            assert _failing(checks) == []
        # the two surfaces disagreeing on the reason is its own failure
        rec["models"][g.THINKING_HEADER_ERROR_KEY] = "some other reason"
        checks = g.declared_render_checks(cid, rec)
        assert key in _failing(checks) and "/v1/models" in checks[key]["detail"]
        # a missing key on either surface is a missing fact, never a pass
        rec = GREEN[cid]()
        del rec["models"][g.THINKING_HEADER_ERROR_KEY]
        assert key in _failing(g.declared_render_checks(cid, rec))
        rec = GREEN[cid]()
        del rec["endpoint"][g.THINKING_HEADER_ERROR_KEY]
        assert key in _failing(g.declared_render_checks(cid, rec))

    @pytest.mark.parametrize("path, value, expect", ALL_TURNS_PERTURBATIONS)
    def test_all_turns_perturbations(self, path, value, expect):
        rec = _green_all_turns()
        _set(rec, path, value)
        assert set(_failing(g.all_turns_render_checks(rec))) == expect, path

    @pytest.mark.parametrize("cid", g.DECLARED_RENDER_CASES)
    def test_every_positive_lens_leg_carries_the_validators_evidence(self, cid):
        """AG-1: ONE successful-lens-leg validator (``lens_leg_checks``) holds
        EVERY leg whose success is evidence, in every case — the map
        ``POSITIVE_LENS_LEGS`` (plus the carve-out's forced legs) is complete
        against the record: every served lens request the drive files is in
        it, and every leg in it carries the validator's whole key set."""
        rec = GREEN[cid]()
        checks = g.declared_render_checks(cid, rec)
        legs = dict(g.POSITIVE_LENS_LEGS[cid])
        served = {n for n, e in rec["requests"].items() if e.get("lens_route") and e.get("status") == 200}
        assert served == set(legs), (served ^ set(legs))                          # complete: no leg escapes
        scopes = [(legs, checks)]
        if cid == g.ALL_POSITIONS_CARVEOUT:
            forced_served = {n for n, e in rec["forced"]["requests"].items()
                             if e.get("lens_route") and e.get("status") == 200}
            assert forced_served == set(g.FORCED_LENS_LEGS)
            scopes.append((dict(g.FORCED_LENS_LEGS), checks))
        for leg_map, chk in scopes:
            for name, (prefix, role) in leg_map.items():
                names = g.LENS_TWIN_CHECK_NAMES if role == "twin" else g.LENS_LEG_CHECK_NAMES
                for k in names:
                    assert f"{prefix}_{k}" in chk, (name, prefix, k)
                    assert chk[f"{prefix}_{k}"]["ok"] is True, (name, k, chk[f"{prefix}_{k}"]["detail"])
        assert g.LENS_LEG_CHECK_NAMES == ("served", "one_entry", "selection", "masked_rows", "modifying")
        assert g.LENS_TWIN_CHECK_NAMES == g.LENS_LEG_CHECK_NAMES + ("scale",)

    @pytest.mark.parametrize("cid", g.DECLARED_RENDER_CASES)
    @pytest.mark.parametrize("scenario", ["missing_evidence", "misrouted_twin", "modifying_twin",
                                          "control_without_evidence"])
    def test_ag1_scenarios_fail_exactly_the_evidence_checks(self, cid, scenario):
        """The reviewer's four in-memory perturbations (AG-1), on EVERY case and
        every applicable leg: the twin's evidence emptied (the drive's real
        no-entry shape), the twin routed to the base object, the twin modifying
        20 applications at scale 1.0, a non-twin control's evidence removed —
        each fails exactly that leg's evidence checks (plus the thinking case's
        attest-derived ``facts``) and exits 2; nothing else moves."""
        rec = GREEN[cid]()
        legs = dict(g.POSITIVE_LENS_LEGS[cid])
        twins = [n for n, (_p, role) in legs.items() if role == "twin"]
        controls = [n for n, (_p, role) in legs.items() if role != "twin"]
        assert len(twins) == 1 and controls
        facts_key = {"facts"} if cid == g.DECLARED_THINKING else set()
        evidence = ("one_entry", "selection", "masked_rows", "modifying")
        if scenario == "control_without_evidence":
            targets = [(n, legs[n][0], evidence + tuple(facts_key)) for n in controls]
        elif scenario == "missing_evidence":
            targets = [(twins[0], legs[twins[0]][0], evidence + ("scale",) + tuple(facts_key))]
        elif scenario == "misrouted_twin":
            targets = [(twins[0], legs[twins[0]][0], ("selection",))]
        else:
            targets = [(twins[0], legs[twins[0]][0], ("modifying", "scale"))]
        for leg, prefix, expect in targets:
            r = copy.deepcopy(rec)
            a = r["requests"][leg]["attest"]
            if scenario in ("missing_evidence", "control_without_evidence"):
                r["requests"][leg]["attest"] = g.one_declared_attestation({})        # n_entries 0, no fields
            elif scenario == "misrouted_twin":
                a["requested"] = r["cells"]["base"]
            else:
                a["n_applications_modifying"], a["applied_effective_scale"] = 20, 1.0
            checks = g.declared_render_checks(cid, r)
            assert set(_failing(checks)) == {f"{prefix}_{k}" for k in expect}, (leg, scenario, _failing(checks))
            kr = dict(r, checks=checks, kind="declared_render", arms=[g.DECLARED_RENDER_ARM])
            assert g.overall_exit_code({"cases": [kr], "complete": True}) == 2

    def test_all_turns_synthetic_span_mismatch_names_both_numbers(self):
        """The synthetic mismatch probe: the worker filed a row count the
        gate's own span sum does not reproduce — the check fails naming the
        filed count, the expected sum and the spans it came from."""
        rec = _green_all_turns()
        rec["requests"]["convo_variant"]["attest"]["prefill_masked_rows"] = 16
        checks = g.all_turns_render_checks(rec)
        d = checks["all_turns_convo_masked_rows"]["detail"]
        assert checks["all_turns_convo_masked_rows"]["ok"] is False
        assert "prefill_masked_rows=16" in d and "expected 17" in d and "[8, 15]" in d


#: the three (completion ``id``, SelectionRecord key) pairs the qwen35_4b
#: ``declared_thinking`` GPU run recorded (Modal app lens-gate-stacked-7a531bf,
#: 2026-09-07; vLLM-minted ids, no content) — every other check of the case
#: passed, and today's exact-equality ``one_entry`` refused all three
REAL_ENGINE_ID_PAIRS = [
    ("chatcmpl-84f8a0249961c15f", "chatcmpl-84f8a0249961c15f-b4a6dee9"),   # on_steered
    ("chatcmpl-82e93127dfdd1374", "chatcmpl-82e93127dfdd1374-bbabadf8"),   # on_twin
    ("chatcmpl-95a9a6131b15c21e", "chatcmpl-95a9a6131b15c21e-8bdfd586"),   # off_base
]
RID, ENGINE_RID = REAL_ENGINE_ID_PAIRS[0]


class TestOneEntryEngineId:
    """``one_entry``'s id match holds the one SelectionRecord entry to THIS
    request's ENGINE id: the completion ``id`` itself, or that id plus exactly
    the pinned vLLM's internal suffix — ``-`` and 8 lowercase hex
    (``InputProcessor.assign_request_id``: ``f"{external_req_id}-{random_uuid():.8}"``,
    ``random_uuid()`` being ``:016x``), minted once per request before any n>1
    fan-out. STRICT where it matters: a different request's id (even one
    sharing a prefix), a second entry, another delimiter or width, a
    parallel-sampling child (``<i>_<id>-…``), or a second suffix all fail."""

    @staticmethod
    def _entry(rid, req_id, n=1):
        return {"status": 200, "id": rid, "lens_route": True, "vmodel": "petri__x",
                "attest": {"n_entries": n, "req_id": req_id}}

    @pytest.mark.parametrize("rid, req_id", REAL_ENGINE_ID_PAIRS)
    def test_the_real_suffixed_engine_id_is_this_request(self, rid, req_id):
        ok, detail = g._one_entry(self._entry(rid, req_id))
        assert ok is True, detail
        assert req_id in detail
        assert g.is_engine_id_of(req_id, rid) is True

    def test_the_bare_completion_id_still_passes(self):
        # VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 (or an offline record) keys by the external id
        ok, detail = g._one_entry(self._entry(RID, RID))
        assert ok is True, detail
        assert g.is_engine_id_of(RID, RID) is True

    @pytest.mark.parametrize("req_id", [
        "chatcmpl-other",
        "chatcmpl-other-b4a6dee9",
        "chatcmpl-95a9a6131b15c21e-b4a6dee9",          # another request's id wearing THIS one's suffix
        "chatcmpl-84f8a0249961c15e",                   # one character off
        "chatcmpl-84f8a0249961c15e-b4a6dee9",
        "chatcmpl-84f8a0249961c15",                    # a PREFIX of this id
        "chatcmpl-84f8a0249961c15-fb4a6dee9",          # a prefix of this id + a well-formed suffix
    ])
    def test_a_different_requests_id_fails(self, req_id):
        ok, detail = g._one_entry(self._entry(RID, req_id))
        assert ok is False
        assert f"not this request's id {RID!r}" in detail and "-<8 hex>" in detail
        assert g.is_engine_id_of(req_id, RID) is False

    @pytest.mark.parametrize("n", [0, 2, 3, None])
    @pytest.mark.parametrize("req_id", [RID, ENGINE_RID])
    def test_anything_but_one_entry_fails_even_under_this_requests_id(self, n, req_id):
        ok, detail = g._one_entry(self._entry(RID, req_id, n=n))
        assert ok is False
        assert f"held {n!r} entries" in detail

    @pytest.mark.parametrize("req_id", [
        "chatcmpl-84f8a0249961c15fb4a6dee9",           # the shared prefix, no delimiter
        "chatcmpl-84f8a0249961c15f_b4a6dee9",          # the wrong delimiter (vLLM's multi-prompt `_` form)
        "chatcmpl-84f8a0249961c15f-b4a6dee",           # 7 hex
        "chatcmpl-84f8a0249961c15f-b4a6dee9f",         # 9 hex
        "chatcmpl-84f8a0249961c15f-B4A6DEE9",          # not random_uuid()'s lowercase alphabet
        "chatcmpl-84f8a0249961c15f-b4a6deeg",          # not hex
        "chatcmpl-84f8a0249961c15f-b4a6dee9-8bdfd586", # two suffixes
        "0_chatcmpl-84f8a0249961c15f-b4a6dee9",        # a parallel-sampling child: never ONE entry for ONE request
        "chatcmpl-84f8a0249961c15f-",
        "chatcmpl-84f8a0249961c15f-b4a6dee9 ",
        " chatcmpl-84f8a0249961c15f-b4a6dee9",
        "chatcmpl-84f8a0249961c15f-b4a6dee9\n",
    ])
    def test_a_shared_prefix_without_the_exact_suffix_fails(self, req_id):
        ok, detail = g._one_entry(self._entry(RID, req_id))
        assert ok is False, detail
        assert g.is_engine_id_of(req_id, RID) is False

    def test_the_matcher_refuses_an_empty_or_missing_id(self):
        assert g.is_engine_id_of("-b4a6dee9", "") is False       # an empty response id matches nothing
        assert g.is_engine_id_of("", "") is False
        assert g.is_engine_id_of(None, RID) is False
        assert g.is_engine_id_of(ENGINE_RID, None) is False
        # ...while one_entry, as before, does not judge an id it was never given
        ok, _ = g._one_entry({"status": 200, "id": None, "attest": {"n_entries": 1, "req_id": ENGINE_RID}})
        assert ok is True

    def test_the_green_records_carry_the_engine_shape(self):
        """The fixtures model the runtime: every served lens leg's key is its
        completion id + the suffix, never the bare id — so the truth tables
        exercise the shape the GPU produces."""
        for cid in g.DECLARED_RENDER_CASES:
            rec = GREEN[cid]()
            scopes = [rec["requests"]] + ([rec["forced"]["requests"]] if cid == g.ALL_POSITIONS_CARVEOUT else [])
            for reqs in scopes:
                legs = [(n, e) for n, e in reqs.items() if e.get("lens_route") and e.get("status") == 200]
                assert legs
                for name, e in legs:
                    assert e["attest"]["req_id"] != e["id"], name
                    assert e["attest"]["req_id"] == _engine_id(e["id"]), name
                    assert g.is_engine_id_of(e["attest"]["req_id"], e["id"]) is True, name


def _set(rec, path, value):
    cur = rec
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value


# --------------------------------------------------------------------------- #
# the result schema: exit code + report                                       #
# --------------------------------------------------------------------------- #


def _case_doc(kr, **over):
    ms = g.MODELS["gemma4e4b"]
    doc = {"model": dict(key=ms.key, hf=ms.hf, rev=ms.rev, cfg=ms.cfg, bank=ms.bank, branch=ms.branch),
           "geometry": {"max_model_len": 4096, "max_num_seqs": 64}, "trees": {}, "batches": [1],
           "timing_batches": [8], "batch_invariant": 0, "inventory": {}, "uncovered_cases": [],
           "complete": True, "only_cases": [kr["case_id"]], "cases": [kr]}
    doc.update(over)
    doc["exit_code"] = g.overall_exit_code(doc)
    return doc


class TestResultSchema:
    @pytest.mark.parametrize("cid", g.DECLARED_RENDER_CASES)
    def test_checks_ride_into_the_exit_code(self, cid):
        rec = GREEN[cid]()
        kr = dict(rec, checks=g.declared_render_checks(cid, rec), kind="declared_render",
                  arms=[g.DECLARED_RENDER_ARM], cells_tt=[CELL, rec["cells"]["variant_tt"]],
                  objects="real", source="/vol/objects/scale-12/x/object.pt")
        assert g.overall_exit_code({"cases": [kr], "complete": True}) == 0
        assert g.overall_exit_code({"cases": [kr], "complete": False}) == 3          # a subset run: never a certification
        bad = copy.deepcopy(kr)
        name = next(iter(bad["checks"]))
        bad["checks"][name]["ok"] = False
        assert g.overall_exit_code({"cases": [bad], "complete": True}) == 2
        assert g.overall_exit_code({"cases": [{"case_id": cid, "error": "GateError: no bank"}], "complete": True}) == 3
        assert g.overall_exit_code({"cases": [kr], "complete": True, "uncovered_cases": [cid]}) == 3

    def test_report_renders_the_positive_artifact(self):
        rec = _green_thinking()
        kr = dict(rec, checks=g.declared_thinking_checks(rec), kind="declared_render", arms=[g.DECLARED_RENDER_ARM],
                  cells_tt=[CELL, rec["cells"]["variant_tt"]], objects="real", source="/vol/x/object.pt",
                  arm=g.DECLARED_RENDER_ARM)
        text = g.render_report(_case_doc(kr, only_cases=["declared_thinking"]))
        assert "DECLARED-RENDER case declared_thinking" in text
        assert f"attested {g.SERVE_DECLARATIONS_KEY}: " in text and "'serve_out_of_support': ['thinking']" in text
        for name in kr["checks"]:
            assert f"check {name}: ok" in text, name
        assert "on_steered [petri__si_safety__psr_kl_all@thinking]: status 200" in text
        assert "refuse_undeclared [petri__si_safety__psr_kl_all@thinking]: status 400 code 'thinking_on_undeclared'" in text
        assert "SUBSET RUN" in text
        # an errored case renders its error; a carve-out case renders its guard probe and instrument
        text = g.render_report(_case_doc({"case_id": "declared_thinking", "error": "GateError: no bank"}))
        assert "ERROR: GateError: no bank" in text
        rec = _green_carveout()
        kr = dict(rec, checks=g.all_positions_carveout_checks(rec), kind="declared_render", arms=[g.DECLARED_RENDER_ARM],
                  cells_tt=[CELL, rec["cells"]["variant_tt"]], objects="real", source="/vol/x/object.pt")
        text = g.render_report(_case_doc(kr))
        assert "guard probe:" in text and "degeneracy instrument (evidence)" in text
        assert "forced variant_single" in text and "'forced-request-unique'" in text
        rec = _green_all_turns()
        kr = dict(rec, checks=g.all_turns_render_checks(rec), kind="declared_render", arms=[g.DECLARED_RENDER_ARM],
                  cells_tt=[CELL, rec["cells"]["variant_tt"]], objects="real", source="/vol/x/object.pt")
        text = g.render_report(_case_doc(kr))
        assert "all-turns spans (the gate's own" in text and "[[8, 15], [22, 29], [36, 39]] -> 17 rows" in text
        assert "all_turns_check='ok'" in text

    def test_report_lines_skip_inventory_cases(self):
        assert g.declared_render_report_lines({"cases": [{"case_id": "psr:relu_gated_vector:gemv:all:skipbos0:both"}]}) == []
        assert g.declared_render_report_lines({}) == []


# --------------------------------------------------------------------------- #
# the standup helper                                                          #
# --------------------------------------------------------------------------- #


class TestStandupHelper:
    def test_accepted_kwargs_is_the_cross_tree_guard(self):
        helper = _helper()

        def old(a, *, b=None):
            return a, b

        def new(a, *, b=None, c=None, d=None):
            return a
        kw, dropped = helper.accepted_kwargs(old, {"b": 1, "c": 2, "d": None})
        assert kw == {"b": 1} and set(dropped) == {"c"} and "takes no c" in dropped["c"]
        kw, dropped = helper.accepted_kwargs(new, {"b": 1, "c": 2, "d": None})
        assert kw == {"b": 1, "c": 2} and dropped == {}                      # None never passed, never reported

        class Ctor:
            def __init__(self, *, x=None):
                self.x = x
        kw, dropped = helper.accepted_kwargs(Ctor, {"x": 1, "y": 2})
        assert kw == {"x": 1} and set(dropped) == {"y"}
        # the real constructor takes every declared kwarg; the base commit's took none of them
        kw, dropped = helper.accepted_kwargs(px.RoutingProxy, {k: "v" for k in helper.PROXY_DECLARED_KWARGS})
        assert set(kw) == set(helper.PROXY_DECLARED_KWARGS) and dropped == {}

    def test_declared_render_needs(self):
        helper = _helper()
        base = {"kind": "psr", "positions": "response"}
        assert helper.declared_render_needs([base, sd.apply_variant(base, "thinking")], None) == \
            {"thinking": False, "all_turns": False}                                    # no vocabulary, no need
        assert helper.declared_render_needs([base], sd) == {"thinking": False, "all_turns": False}
        assert helper.declared_render_needs([base, sd.apply_variant(base, "thinking")], sd) == \
            {"thinking": True, "all_turns": False}
        assert helper.declared_render_needs([base, sd.apply_variant(base, "all_turns")], sd) == \
            {"thinking": False, "all_turns": True}
        assert helper.declared_render_needs([base, sd.apply_variant(base, "all_positions")], sd) == \
            {"thinking": False, "all_turns": False}
        with pytest.raises(sd.ServeDeclarationError):
            helper.declared_render_needs([dict(base, serve_out_of_support="thinking")], sd)   # malformed: never undeclared

    def test_serve_declarations_reader_guard(self):
        import types
        helper = _helper()
        assert helper.serve_declarations_attested({}, types.SimpleNamespace())[0] is None
        assert "predates" in helper.serve_declarations_attested({}, types.SimpleNamespace())[1]
        from sisteer.capeval.serving import lens_worker as lw
        att = {"worker_reports": [{lw.LENS_SERVE_DECLARATIONS_KEY: {f"{HB}@thinking": {"serve_out_of_support": ["thinking"]}}}]}
        assert helper.serve_declarations_attested(att, lw) == ({f"{HB}@thinking": {"serve_out_of_support": ["thinking"]}}, None)
        assert helper.serve_declarations_attested({"worker_reports": [{lw.LENS_SERVE_DECLARATIONS_KEY: {}}]}, lw) == ({}, None)
        assert helper.legacy_object_sha256({"a": SHA, "b": SHA}) == SHA
        assert helper.legacy_object_sha256({"a": SHA, "b": "cd" * 32}) is None and helper.legacy_object_sha256({}) is None

    def test_registry_markers_thinking_header_discipline(self):
        """S-G3: capctrl's derivation discipline, cross-tree. A declaring cell
        derives the whole fact HARD; every other standup keeps the partial
        fact + reason through the served tree's ``thinking_header_facts_or_
        error``; a tree without that helper catches ``ChatMarkerError`` ONLY
        around ``thinking_header_facts``; any other exception is a defect and
        propagates; a tree with no fact function at all records why."""
        import types
        helper = _helper()

        class CME(ValueError):
            pass

        def _cmk(*, facts=None, soft=None, on_header=ON):
            mod = types.SimpleNamespace(ChatMarkerError=CME,
                                        generation_header_ids=lambda tok: OFF,
                                        generation_header_ids_under=lambda tok, *, enable_thinking: (
                                            on_header if enable_thinking else OFF))
            if facts is not None:
                mod.thinking_header_facts = facts
            if soft is not None:
                mod.thinking_header_facts_or_error = soft
            return mod
        full = {"switch": True, "moves": True, "off_ids": list(OFF), "on_ids": list(ON)}
        partial = dict(PARTIAL_TH)

        def raising_cme(tok):
            raise CME("probe: the ON render is not a prefix")

        def raising_defect(tok):
            raise TypeError("a programming defect")
        off = {"thinking": False, "all_turns": False}
        on = {"thinking": True, "all_turns": False}
        # (a) not declaring, the non-fatal helper present: the partial fact + the reason are KEPT
        m = helper.registry_markers("tok", off, _cmk(facts=raising_cme, soft=lambda tok: (partial, PARTIAL_REASON)))
        assert m["thinking_header"] == partial and m["thinking_header_error"] == PARTIAL_REASON
        # (a') ...and a fully derived fact carries None
        m = helper.registry_markers("tok", off, _cmk(facts=lambda tok: full, soft=lambda tok: (full, None)))
        assert m["thinking_header"] == full and m["thinking_header_error"] is None
        # (b) DECLARING: the whole fact is derived hard — the soft helper is never consulted
        with pytest.raises(RuntimeError, match="thinking-header fact cannot be derived") as ei:
            helper.registry_markers("tok", on, _cmk(facts=raising_cme, soft=lambda tok: (partial, PARTIAL_REASON)))
        assert isinstance(ei.value.__cause__, CME)
        m = helper.registry_markers("tok", on, _cmk(facts=lambda tok: full, soft=lambda tok: (partial, "never")))
        assert m["thinking_header"] == full and m["thinking_header_error"] is None
        assert m["assistant_header_ids_thinking_on"] == list(ON)
        # (c) a tree with the fact but not the helper: ChatMarkerError ONLY, no partial fact to keep
        m = helper.registry_markers("tok", off, _cmk(facts=raising_cme))
        assert m["thinking_header"] is None
        assert m["thinking_header_error"].startswith("CME: probe") and "thinking_header_facts_or_error" in m["thinking_header_error"]
        # (d) any other exception is a DEFECT, never "unavailable" — on either path
        with pytest.raises(TypeError, match="programming defect"):
            helper.registry_markers("tok", off, _cmk(facts=raising_defect))
        with pytest.raises(TypeError, match="programming defect"):
            helper.registry_markers("tok", off, _cmk(facts=lambda tok: full, soft=raising_defect))
        # (e) the base commit: no fact function at all
        m = helper.registry_markers("tok", off, _cmk())
        assert m["thinking_header"] is None and "no thinking_header_facts" in m["thinking_header_error"]

    def test_real_nemotron_off_standup_keeps_the_partial_fact_and_a_declaring_one_refuses(self):
        """The pinned Nemotron-3 template (X1's reproduction): the gate helper's
        OFF derivation keeps ``{switch, off_ids, on_ids: None, moves: None}`` +
        the ChatMarkerError reason instead of dropping the fact; a cell
        declaring thinking still refuses; and the gate's own derivation no
        longer raises on this family."""
        tok = _cached_tok(*NEMOTRON)
        from sisteer import chat_markers as cmk
        helper = _helper()
        m = helper.registry_markers(tok, {"thinking": False, "all_turns": False}, cmk)
        facts, why = cmk.thinking_header_facts_or_error(tok)
        assert facts["switch"] is True and facts["on_ids"] is None and facts["moves"] is None
        assert m["thinking_header"] == facts and m["thinking_header_error"] == why
        assert why.startswith("ChatMarkerError: ")
        assert m["assistant_header_ids"] == facts["off_ids"]
        with pytest.raises((RuntimeError, cmk.ChatMarkerError)):
            helper.registry_markers(tok, {"thinking": True, "all_turns": False}, cmk)
        # the gate's own derivation (thinking_header_matches_tokenizer's reference)
        assert g.thinking_header_for(*NEMOTRON) == (facts, why)

    def test_declared_render_refusal(self):
        """S-G2: on a declared-render standup a dropped digest-binding or
        digest-publication kwarg is FATAL, naming the kwarg; a non-declared
        standup (the base-commit reference) keeps standing up."""
        helper = _helper()
        digests = {HB: SHA, f"{HB}@thinking": SHA}
        exp_drop = {"expected_artifact_sha256": "the served sisteer's export_worker_registry takes no expected_artifact_sha256"}
        px_drop = {"object_artifact_sha256_by_cell": "the served sisteer's RoutingProxy takes no object_artifact_sha256_by_cell"}
        ok = helper.declared_render_refusal(declared_render=True, digests=digests, export_dropped={}, proxy_dropped={})
        assert ok is None
        why = helper.declared_render_refusal(declared_render=True, digests=digests, export_dropped=exp_drop, proxy_dropped={})
        assert "expected_artifact_sha256" in why and "takes no expected_artifact_sha256" in why
        why = helper.declared_render_refusal(declared_render=True, digests=digests, export_dropped={}, proxy_dropped=px_drop)
        assert "object_artifact_sha256_by_cell" in why and "takes no object_artifact_sha256_by_cell" in why
        why = helper.declared_render_refusal(declared_render=True, digests={}, export_dropped={}, proxy_dropped={})
        assert "no fitted lens cell" in why
        for dropped in ({}, exp_drop, px_drop):
            assert helper.declared_render_refusal(declared_render=False, digests=digests,
                                                  export_dropped=dropped, proxy_dropped=dropped) is None
        # the flag OR a served variant cell makes a standup declared-render
        assert helper.is_declared_render_standup(["safety_refusal__psr_kl_all", f"{CELL}@thinking"], flag=False) is True
        assert helper.is_declared_render_standup(["none", CELL], flag=False) is False
        assert helper.is_declared_render_standup(["none", CELL], flag=True) is True

    def test_offline_standup_main_publishes_what_it_handed(self, tmp_path, monkeypatch):
        """END TO END, offline: the helper's ``main`` on the REAL pinned tokenizers
        and configs with only the GPU launch, the upstream client and the
        aiohttp thread replaced. (1) qwen ``none`` + base + ``@thinking`` under
        ``--declared-render``: the digest map is bound at export, handed to the
        REAL RoutingProxy and published identically in endpoint.json, the
        routing table and ``/v1/models``, with the header fact and a null
        reason beside it in both. (2) a served tree whose RoutingProxy takes no
        ``object_artifact_sha256_by_cell`` REFUSES a declared-render standup
        before any launch, naming the kwarg. (3) the same tree on a plain
        ``none`` standup (the reference's shape) stands up and publishes null +
        unbound + the reason — never the helper's local digests. (4) a Nemotron
        OFF standup keeps the partial fact and its reason in endpoint.json AND
        ``/v1/models``, agreeing (S-G3)."""
        import types
        from test_serve_declarations_attestation import _FakeServer, _RecordingBackend, _att
        from sisteer.capeval.serving import vllm_server as vs
        helper = _helper()
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        # main() drops its own directory from sys.path IN PLACE (_drop_own_dir); give it a copy so
        # the rest of the session keeps importing from scripts/
        monkeypatch.setattr(sys, "path", list(sys.path))
        built = {}

        def fake_launch(launch, *, log_path, **_):
            built["launched"] = built.get("launched", 0) + 1
            # two workers attesting the same declaration map (the proven stub shape)
            (Path(log_path).parent / vs.STEERING_ATTESTATION_FILE).write_text(
                json.dumps(_att(built["decls"], built["decls"])))
            return _FakeServer()
        monkeypatch.setattr(vs, "launch_server", fake_launch)
        monkeypatch.setattr(vs, "_shutdown_child", lambda server: None)
        monkeypatch.setattr(px, "make_app", lambda proxy: built.setdefault("proxy", proxy))
        monkeypatch.setattr(px, "HTTPBackend", _RecordingBackend)
        monkeypatch.setattr(helper, "_start_proxy_thread",
                            lambda app, host, port: types.SimpleNamespace(call_soon_threadsafe=lambda f: None,
                                                                          stop=lambda: None))
        monkeypatch.setattr(helper, "verify_tree", lambda repo, sha: sha)     # the working tree under test

        def run(family, cells, *extra, decls):
            ms = g.MODELS[family]
            _cached_tok(ms.hf, ms.rev)
            out = tmp_path / family / "_".join(c.replace("@", "-") for c in cells) / str(len(built))
            out.mkdir(parents=True)
            (out / "STOP").write_text("")                                     # leave the wait loop at once
            ddir = out / "bank"
            if any(c != "none" for c in cells):
                cell, _ = g.write_synthetic_object("psr_kl_all", DIMS, ddir, len(HEADER), si=g.DECLARED_RENDER_SI)
                assert cell == CELL
            else:
                ddir.mkdir()
            built.pop("proxy", None)
            built["decls"] = decls
            argv = ["--repo-dir", str(REPO), "--expect-sha", "0" * 40, "--config", ms.cfg,
                    "--defenses-dir", str(ddir), "--out", str(out), "--stop-file", str(out / "STOP"),
                    "--max-model-len", "4096", "--max-num-seqs", "8", *extra]
            for c in cells:
                argv += ["--cell", c]
            monkeypatch.setattr(sys, "argv", ["lens_gate_standup.py", *argv])
            rc = helper.main()
            ep = json.loads((out / "endpoint.json").read_text())
            return rc, ep, out
        decl = g.expected_declarations("thinking")
        hv = harness_cell_id(f"{CELL}@thinking")
        # (1) the full-API tree, declared-render
        rc, ep, out = run("qwen35_4b", [CELL, f"{CELL}@thinking"], "--declared-render",
                          decls={hv: decl, dc.NOOP_LENS_CELL_ID: decl})
        assert rc == 0 and built["launched"] == 1
        want = hashlib.sha256((out / "bank" / CELL / "object.pt").read_bytes()).hexdigest()
        assert ep[g.OBJECT_SHA256_BY_CELL_KEY] == {HB: want, hv: want}
        assert ep["object_artifact_sha256"] == want and ep["object_artifact_sha256_bound"] is True
        assert ep["object_artifact_sha256_error"] is None
        proxy = built["proxy"]
        block = proxy.models_payload()["petri_proxy"]
        assert block[g.OBJECT_SHA256_BY_CELL_KEY] == ep[g.OBJECT_SHA256_BY_CELL_KEY]
        table = px.load_routing_table(out / "routing_table.json")
        assert table[g.THINKING_HEADER_KEY] == ep[g.THINKING_HEADER_KEY] == block[g.THINKING_HEADER_KEY]
        assert ep[g.THINKING_HEADER_KEY]["moves"] is True
        assert ep[g.THINKING_HEADER_ERROR_KEY] is None and block[g.THINKING_HEADER_ERROR_KEY] is None
        assert ep[g.SERVE_DECLARATIONS_KEY] == block[g.SERVE_DECLARATIONS_KEY] == {hv: decl, dc.NOOP_LENS_CELL_ID: decl}
        assert ep["declared_render"] is True
        # (2) a tree whose RoutingProxy takes no digest map: a declared-render standup REFUSES before launch
        real_proxy = px.RoutingProxy

        class OldProxy(real_proxy):
            def __init__(self, routes, backend, *, base_model, serving_mode, prompted_prefixes=(),
                         exact_replay_strategy=None, launch_attestation=None, agent_loop_render=None,
                         petri_agent_loop_entry=True, resumed_prefill_policy_cells=None,
                         lens_serve_declarations=None, lens_serve_declarations_error=None,
                         thinking_header=None, thinking_header_error=None, lens_salt_policy="attested"):
                super().__init__(routes, backend, base_model=base_model, serving_mode=serving_mode,
                                 prompted_prefixes=prompted_prefixes, exact_replay_strategy=exact_replay_strategy,
                                 launch_attestation=launch_attestation, agent_loop_render=agent_loop_render,
                                 petri_agent_loop_entry=petri_agent_loop_entry,
                                 resumed_prefill_policy_cells=resumed_prefill_policy_cells,
                                 lens_serve_declarations=lens_serve_declarations,
                                 lens_serve_declarations_error=lens_serve_declarations_error,
                                 thinking_header=thinking_header, thinking_header_error=thinking_header_error,
                                 lens_salt_policy=lens_salt_policy)
        monkeypatch.setattr(px, "RoutingProxy", OldProxy)
        with pytest.raises(SystemExit, match="object_artifact_sha256_by_cell"):
            run("qwen35_4b", [CELL, f"{CELL}@thinking"], "--declared-render",
                decls={hv: decl, dc.NOOP_LENS_CELL_ID: decl})
        assert built["launched"] == 1                                          # refused BEFORE the launch
        # (2') ...and without the flag too: a served variant cell IS a declared-render standup
        with pytest.raises(SystemExit, match="object_artifact_sha256_by_cell"):
            run("qwen35_4b", [CELL, f"{CELL}@thinking"], decls={hv: decl, dc.NOOP_LENS_CELL_ID: decl})
        assert built["launched"] == 1
        # (3) the same tree on a plain standup: stands up; the endpoint publishes what was HANDED (nothing)
        rc, ep, out = run("qwen35_4b", [CELL], decls={})
        assert rc == 0 and built["launched"] == 2
        assert ep[g.OBJECT_SHA256_BY_CELL_KEY] is None and ep["object_artifact_sha256"] is None
        assert ep["object_artifact_sha256_bound"] is False
        assert "takes no object_artifact_sha256_by_cell" in ep["object_artifact_sha256_error"]
        assert g.OBJECT_SHA256_BY_CELL_KEY not in built["proxy"].models_payload()["petri_proxy"]
        assert ep["declared_render"] is False
        monkeypatch.setattr(px, "RoutingProxy", real_proxy)
        # (4) Nemotron OFF: the partial fact + reason, identical in endpoint.json and /v1/models
        _cached_tok(*NEMOTRON)
        nemo = types.SimpleNamespace(hf=NEMOTRON[0], rev=NEMOTRON[1], cfg="configs/two_si_nemotron3nano4b.yaml")
        monkeypatch.setitem(g.MODELS, "nemotron_test", nemo)
        rc, ep, out = run("nemotron_test", ["none"], decls={})
        assert rc == 0
        from sisteer import chat_markers as cmk
        facts, why = cmk.thinking_header_facts_or_error(_cached_tok(*NEMOTRON))
        assert ep[g.THINKING_HEADER_KEY] == facts and ep[g.THINKING_HEADER_KEY]["on_ids"] is None
        assert ep[g.THINKING_HEADER_ERROR_KEY] == why and why.startswith("ChatMarkerError: ")
        block = built["proxy"].models_payload()["petri_proxy"]
        assert block[g.THINKING_HEADER_KEY] == facts and block[g.THINKING_HEADER_ERROR_KEY] == why
        assert built["proxy"].thinking_header_error == why
        table = px.load_routing_table(out / "routing_table.json")
        assert table[g.THINKING_HEADER_KEY] == facts

    def test_cli_flag_and_the_gate_threads_it(self):
        helper_src = (REPO / "scripts" / "lens_gate_standup.py").read_text()
        assert '"--lens-salt-policy", choices=LENS_SALT_POLICIES, default="attested"' in helper_src
        assert "proxy cannot honour is refused, never silently dropped" in helper_src
        src = inspect.getsource(g.start_standup)
        assert inspect.signature(g.start_standup).parameters["lens_salt_policy"].default is None
        assert 'cmd += ["--lens-salt-policy", lens_salt_policy]' in src and "if lens_salt_policy is not None:" in src
        assert 'endpoint.get(LENS_SALT_POLICY_KEY) != lens_salt_policy' in src      # the endpoint must say what was asked
        # the standups other cases run are byte-identical: the flag rides only when asked
        assert '"--manifest", ctx.manifest, "--prefix-caching", prefix_caching]' in src
        # S-G2: the declared-render standups say so, and the helper's endpoint publishes what it HANDED
        assert '"--declared-render", action="store_true"' in helper_src
        assert 'digests_handed = "object_artifact_sha256_by_cell" in declared_kwargs' in helper_src
        assert '"object_artifact_sha256_by_cell": (dict(digests) if digests_handed else None)' in helper_src
        assert '"object_artifact_sha256_bound": bool(digests_handed)' in helper_src
        assert "raise SystemExit(f\"[standup] {refusal}\")" in helper_src
        assert inspect.signature(g.start_standup).parameters["declared_render"].default is False
        assert 'if declared_render:' in src and 'cmd += ["--declared-render"]' in src
        assert src.index('"--lens-salt-policy"') < src.index('"--declared-render"')
        csrc = inspect.getsource(g.run_declared_render_case)
        assert csrc.count("declared_render=True") == 2                                 # both standups
        # S-G3: the reason rides with the fact into the table AND the proxy, through the guard
        assert '"thinking_header_error": thinking_header_error' in helper_src
        assert "accepted_kwargs(export_routing_table, {" in helper_src
        assert "thinking_header_facts_or_error" in helper_src
        msrc = inspect.getsource(_helper().registry_markers)
        header_block = msrc[msrc.index('facts = getattr(cmk, "thinking_header_facts", None)'):]
        assert "except Exception" not in header_block                                   # ChatMarkerError ONLY
        assert header_block.count("except cmk.ChatMarkerError as e:") == 2             # the hard and the no-helper paths

    def test_run_gate_and_driver_source_pins(self):
        rsrc = inspect.getsource(g.run_gate)
        assert "declared_render_order(ms, only_cases)" in rsrc and "declared_render_only(only_cases)" in rsrc
        assert rsrc.count("_declared_render_case(ctx, case_id, cand_sha=cand_sha") == 2   # the oracle-less path and the mixed one
        csrc = inspect.getsource(g.run_declared_render_case)
        assert "side=DECLARED, tree=ctx.candidate" in csrc
        assert "side=DECLARED_FORCED, tree=ctx.candidate" in csrc
        assert "lens_salt_policy=LENS_SALT_POLICY_REQUEST_UNIQUE_ALL" in csrc
        assert "REFERENCE" not in csrc                                              # no reference standup for these cases
        fsrc = inspect.getsource(g.drive_forced_policy)
        assert "proxy_tokenize(su, cells[\"vm_variant\"], msgs)" in fsrc                   # S-G1: this standup's P
        assert 'rec["expected_all_rows"] = all_positions_rows_expected(' in fsrc
        dsrc = inspect.getsource(g._DeclaredDriver.send)
        assert 'rpc(su, "sisteer_selection_record", [True])' in dsrc                 # reset before every lens leg
        assert 'entry["attest"] = one_declared_attestation(selection_requests(su))' in dsrc
        assert 'entry["claimed_failure"] = claim_request_failure(su, entry.get("id"))' in dsrc
        assert "proxy_chat(su, payload)" in dsrc                                     # THROUGH the proxy
        osrc = inspect.getsource(g.declared_render_object)
        assert "no synthetic" in osrc.lower() and "DECLARED_RENDER_CELL" in osrc
        gsrc = inspect.getsource(g.guard_probe)
        assert 'sd.apply_variant(base_spec, "all_positions")' in gsrc and 'undeclared_spec["positions"] = "all"' in gsrc
        assert "la.get_adapter(obj[\"kind\"])" in gsrc                                # the REAL guarded adapter

    def test_offline_registry_export_and_load_for_each_variant_standup(self, tmp_path):
        """END TO END, offline, what the GPU standup does before any launch: the
        helper's enumeration of ``none`` + base + ``@variant`` over a synthetic
        psr object under the REAL config, the marker needs, the digests, the
        export with the optional markers + the digest binding, and the worker
        registry LOAD — the twin declared under its own id on a thinking
        standup, the guard admitting the named all-positions cell, the
        all-turns cell loading with its scaffold."""
        import torch  # noqa: F401 — the load adapts ops
        from sisteer.capeval import manifest as cm
        from sisteer.capeval.serving import lens_steer as ls
        helper = _helper()
        manifest = "configs/capeval/manifests/petri_twosi.yaml"
        for family, variant, needs_want in (("qwen35_4b", "thinking", {"thinking": True, "all_turns": False}),
                                            ("gemma4e4b", "all_positions", {"thinking": False, "all_turns": False}),
                                            ("gemma4e4b", "all_turns", {"thinking": False, "all_turns": True})):
            ms = g.MODELS[family]
            cfg = _cfg(ms)
            ddir = tmp_path / family / variant / "bank"
            cell, _ = g.write_synthetic_object("psr_kl_all", DIMS, ddir, len(HEADER), si=g.DECLARED_RENDER_SI)
            assert cell == CELL
            variant_tt = g.variant_cell(cell, variant)
            out = tmp_path / family / variant / "standup"
            out.mkdir(parents=True)
            spec, src, cset, _cfg_path, _added = helper.enumerate_gate_cells(
                repo=REPO, cfg=cfg, cells_tt=[cell, variant_tt], defenses_dir=ddir, manifest=manifest,
                chat_template_hash="ab" * 8, out=out)
            served = {c.cell_id for c in cset.all_served()}
            hv = harness_cell_id(variant_tt)
            assert {"none", HB, hv, dc.NOOP_LENS_CELL_ID} <= served and cset.absent == ()
            specs = helper.lens_resolved_specs(cset, src, cm.parse_serving_identity)
            assert len(specs) == 2 and helper.declared_render_needs(specs, sd) == needs_want
            digests = helper.artifact_digests(cset, src, cm.parse_serving_identity)
            want = hashlib.sha256((ddir / cell / "object.pt").read_bytes()).hexdigest()
            assert digests == {HB: want, hv: want} and helper.legacy_object_sha256(digests) == want
            markers = {"assistant_header_ids": list(HEADER), "assistant_turn_end_ids": None,
                       "assistant_header_ids_thinking_on": None}
            if needs_want["thinking"]:
                markers["assistant_header_ids_thinking_on"] = [7, 8, 9, 11]      # a distinct synthetic ON header
            if needs_want["all_turns"]:
                markers["assistant_turn_end_ids"] = [990, 77]
            export_kwargs, dropped = helper.accepted_kwargs(ls.export_worker_registry, {
                "assistant_turn_end_ids": (tuple(markers["assistant_turn_end_ids"])
                                           if markers["assistant_turn_end_ids"] else None),
                "assistant_header_ids_thinking_on": (tuple(markers["assistant_header_ids_thinking_on"])
                                                     if markers["assistant_header_ids_thinking_on"] else None),
                "expected_artifact_sha256": dict(digests)})
            assert dropped == {} and "expected_artifact_sha256" in export_kwargs
            reg_dir = out / "lens_registry"
            ls.export_worker_registry(cset, src, reg_dir, assistant_header_ids=tuple(HEADER),
                                      zero_op_hidden_size=DIMS["hidden_size"], **export_kwargs)
            index = json.loads((reg_dir / ls.REGISTRY_INDEX).read_text())
            twin = next(e for e in index["entries"] if e["cell_id"] == dc.NOOP_LENS_CELL_ID)
            assert index["assistant_header_ids_thinking_on"] == markers["assistant_header_ids_thinking_on"]
            assert index["assistant_turn_end_ids"] == markers["assistant_turn_end_ids"]
            reg = ls.WorkerObjectRegistry.load(reg_dir)
            decl = g.expected_declarations(variant)
            if variant == "thinking":
                assert twin["donor_cell"] == hv                                   # P12: the render-declaring cell
                assert reg.serve_declarations_by_cell() == {hv: decl, dc.NOOP_LENS_CELL_ID: decl}
                assert reg.serves_thinking_on() is True
            else:
                assert twin["donor_cell"] == HB                                   # a position variant is never a donor
                assert reg.serve_declarations_by_cell() == {hv: decl}
                assert reg.serves_thinking_on() is False
            # a digest that does not match REFUSES the export (the binding is real)
            with pytest.raises(ls.LensSteerError, match="was replaced"):
                ls.export_worker_registry(cset, src, out / "reg2", assistant_header_ids=tuple(HEADER),
                                          zero_op_hidden_size=DIMS["hidden_size"],
                                          **dict(export_kwargs, expected_artifact_sha256={HB: "cd" * 32, hv: want}))

    def test_the_guard_probe_on_a_synthetic_response_fit(self, tmp_path):
        """The in-process guard probe on a synthetic response-fit psr object:
        the ``@all_positions`` overlay adapts to ["all"]; ``positions="all"``
        without the name is refused naming the carve-out."""
        ddir = tmp_path / "bank"
        cell, _ = g.write_synthetic_object("psr_kl_all", DIMS, ddir, len(HEADER), si=g.DECLARED_RENDER_SI)
        obj = {"source": str(ddir / cell / "object.pt"), "kind": "psr", "arm": "psr_kl_all"}
        probe = g.guard_probe(obj, header_len=len(HEADER))
        assert probe["base_positions"] == "response"
        assert probe["declared"] == {"adapted": True, "served_positions": ["all"],
                                     "experiment": "fit_response_serve_all_positions"}
        assert probe["undeclared"]["refused"] is True and "fit_response_serve_all_positions" in probe["undeclared"]["error"]
        # and the resolver reads the same object as a response fit (no bank -> refuses; no cell -> refuses)
        ms = g.MODELS["gemma4e4b"]
        with pytest.raises(g.GateError, match="no scale-12 bank"):
            g.declared_render_object(ms, bank_dir=None, header_len=len(HEADER))
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(g.GateError, match=f"holds no {CELL} artifact"):
            g.declared_render_object(ms, bank_dir=empty, header_len=len(HEADER))
        got = g.declared_render_object(ms, bank_dir=ddir, header_len=len(HEADER))
        assert got["cell"] == CELL and got["served_positions"] == ["response"] and got["objects"] == "real"
        assert got["sha256"] == hashlib.sha256((ddir / cell / "object.pt").read_bytes()).hexdigest()
        assert got["skip_bos"] is False and got["kind"] == "psr"
        # an all-position fit is not what these cases serve
        cell2, ddir2 = g.write_synthetic_object("psr_kl_all", DIMS, tmp_path / "bank2", len(HEADER),
                                                si=g.DECLARED_RENDER_SI, positions="all")
        with pytest.raises(g.GateError, match="not \\['response'\\]"):
            g.declared_render_object(ms, bank_dir=ddir2, header_len=len(HEADER))


class TestModalWrapperNeedsNoChange:
    def test_cases_pass_through_and_the_app_name_rule_stands(self):
        src = (REPO / "scripts/modal/lens_vectorize_gate.py").read_text()
        for cid in g.DECLARED_RENDER_CASES:
            assert cid not in src                                    # nothing case-specific in the wrapper
        assert "--lens-salt-policy" not in src                       # the gate's start_standup passes it, never the wrapper
        assert 'argv += ["--cases", ",".join(cases)]' in src          # the selection rides through unchanged
        assert 'expected_name = f"lens-gate-stacked-{sha[:7]}"' in src   # the ONE app-name rule
        assert "scripts/lens_gate_standup.py" in src
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "results_rel")
        assert fn is not None
