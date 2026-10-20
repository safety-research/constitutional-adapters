"""``sisteer.capeval.serving.standup_thinking`` — the standup steps factored out
of ``scripts/capctrl_serving_up.py`` so the cluster standup runs the same code.
Two things are pinned: the module's own behaviour on synthetic inputs, and that
capctrl's names are STILL the module's (the factor moved code, it did not fork
it)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.serving import lens_worker as lw
from sisteer.capeval.serving import serve_declaration as sd
from sisteer.capeval.serving import standup_thinking as st
from sisteer.capeval.serving import vllm_server as vs

KEY = lw.LENS_SERVE_DECLARATIONS_KEY


def _load_capctrl():
    spec = importlib.util.spec_from_file_location("_st_capctrl", REPO / "scripts/capctrl_serving_up.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_st_capctrl"] = mod
    spec.loader.exec_module(mod)
    return mod


def _report(decls, policy="conditional"):
    return {"registration_count": 1, "layers": [3], "oproj_layers": [], "plan_layer": 3,
            "cells": ["none__lens0", "si_constv2_mix_full__lora_r4_ce_minus_ctl"], "ledger_cells": [],
            "lens_reduction_mode": "stacked",
            "resumed_prefill_policy_cells": {"none__lens0": policy,
                                             "si_constv2_mix_full__lora_r4_ce_minus_ctl": policy},
            KEY: decls}


def _att(*maps):
    reps = [_report(m) for m in maps]
    return {"registry_dir": "/tmp/synthetic/lens_registry", "rpc_method": "sisteer_install_steering",
            "worker_reports": reps,
            "scheduler_geometry": {"worker_reports": [{"lens_reduction_mode": "stacked"} for _ in reps]}}


def _write(tmp_path, doc):
    p = tmp_path / vs.STEERING_ATTESTATION_FILE
    p.write_text(json.dumps(doc))
    return p


# --------------------------------------------------------------------------- #
# the readers: the module's, and capctrl's names are the module's             #
# --------------------------------------------------------------------------- #


def test_the_three_readers_are_lens_workers_own_and_capctrl_delegates_to_them(tmp_path):
    capctrl = _load_capctrl()
    doc = _att({}, {})
    p = _write(tmp_path, doc)
    assert st.attested_serve_declarations_at(p) == lw.serve_declarations_attested(doc) == ({}, None)
    assert st.attested_reduction_mode_at(p) == lw.attested_reduction_mode(doc)
    assert st.attested_resumed_prefill_policy_cells_at(p) == lw.resumed_prefill_policy_cells_attested(doc)
    for name in ("attested_serve_declarations_at", "attested_reduction_mode_at",
                 "attested_resumed_prefill_policy_cells_at"):
        assert getattr(capctrl, name)(p) == getattr(st, name)(p), name
    # a missing file is (None, why) on every reader — recorded, never guessed
    for name in ("attested_serve_declarations_at", "attested_reduction_mode_at",
                 "attested_resumed_prefill_policy_cells_at"):
        value, why = getattr(st, name)(tmp_path / "nope.json")
        assert value is None and why, name
        assert getattr(capctrl, name)(tmp_path / "nope.json") == (value, why)
    # disagreeing workers: None + the reason
    decls, why = st.attested_serve_declarations_at(_write(tmp_path, _att({"a@thinking": {"serve_out_of_support": ["thinking"]}}, {})))
    assert decls is None and "disagree" in why


def test_capctrl_keeps_no_second_copy_of_the_factored_code():
    """The factor MOVED the code: capctrl's bring_up reads the derivation off the
    module, and its module-level names are thin wrappers, not a fork that can
    drift (the plan's 'behaviour-preserving, capctrl's own call sites switch')."""
    src = (REPO / "scripts/capctrl_serving_up.py").read_text()
    assert "from sisteer.capeval.serving import standup_thinking as st" in src
    assert "st.lens_resolved_specs(cells, src)" in src
    assert "st.declares_thinking(lens_specs)" in src
    assert "st.thinking_on_header_ids(tok, header_ids, declares_thinking=declares_thinking)" in src
    assert "st.backend_timeout_kwargs(upstream_timeout_s)" in src
    for moved in ("reader = getattr(lw, \"serve_declarations_attested\", None)",
                  "return lw.attested_reduction_mode(att)",
                  "generation_header_ids_under(tok, enable_thinking=True)",
                  "return thinking_header_facts_or_error(tok)"):
        assert moved not in src, moved        # the body lives in the module now
        assert moved in (REPO / "src/sisteer/capeval/serving/standup_thinking.py").read_text(), moved
    # the strings the earlier landings' tests pin on bring_up are untouched
    for kept in ("assistant_header_ids_thinking_on=on_header_ids", "attested_serve_declarations_at(",
                 "lens_serve_declarations=decls", "thinking_header=table.get(\"thinking_header\")",
                 "object_artifact_sha256_by_cell=dict(expected_digests)"):
        assert kept in src, kept


def test_the_module_is_import_light():
    """Loads without torch / transformers / aiohttp (the client venv, the sbatch's
    python3, the grading host test venv all import it)."""
    code = ("import sys; sys.path.insert(0, %r); import sisteer.capeval.serving.standup_thinking as m; "
            "print(sorted(k for k in ('torch', 'transformers', 'aiohttp', 'vllm') if k in sys.modules))"
            % str(REPO / "src"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", r.stdout


# --------------------------------------------------------------------------- #
# the declaration and the two headers                                          #
# --------------------------------------------------------------------------- #


class _Tok:
    """``generation_header_ids_under`` is monkeypatched around this; the object
    only needs to exist."""


def _declared_spec():
    return {"kind": "lora_subtraction", "positions": "all", sd.SERVE_OUT_OF_SUPPORT_KEY: [sd.OUT_OF_SUPPORT_THINKING]}


def test_declares_thinking_reads_the_hashed_declaration_only():
    assert st.declares_thinking([]) is False
    assert st.declares_thinking([{"kind": "lora_subtraction", "positions": "all"}]) is False
    assert st.declares_thinking([{"kind": "psr", "positions": "response"}, _declared_spec()]) is True


def test_the_second_header_is_derived_only_for_a_declaring_cell_and_an_equal_one_is_recorded(monkeypatch):
    import sisteer.chat_markers as cmk
    calls = []

    def fake_under(tok, *, enable_thinking):
        calls.append(enable_thinking)
        return (1, 2, 3, 40)
    monkeypatch.setattr(cmk, "generation_header_ids_under", fake_under)
    # no declaring cell: NO derivation runs (rule 5 — never a second header for an undeclared cell;
    # and the Nemotron ON derivation that fails must not run on a base-only standup)
    assert st.thinking_on_header_ids(_Tok(), (1, 2, 3, 4), declares_thinking=False) is None
    assert calls == []
    assert st.thinking_on_header_ids(_Tok(), (1, 2, 3, 4), declares_thinking=True) == (1, 2, 3, 40)
    assert calls == [True]
    # a header-INVARIANT template (gemma-4): the ON header EQUALS the OFF header and is
    # returned as the second header all the same — admitted, never refused as vacuous
    # (the rule 2026-09-10 withdrew plan P9)
    assert st.thinking_on_header_ids(_Tok(), (1, 2, 3, 40), declares_thinking=True) == (1, 2, 3, 40)
    assert calls == [True, True]


def test_standup_thinking_header_is_hard_for_a_declaring_cell_and_soft_otherwise(monkeypatch):
    import sisteer.chat_markers as cmk
    partial = {"switch": True, "moves": None, "off_ids": [10, 12, 13], "on_ids": None}

    def hard(tok):
        raise cmk.ChatMarkerError("boundary retokenization at token 16")

    monkeypatch.setattr(cmk, "thinking_header_facts", hard)
    monkeypatch.setattr(cmk, "thinking_header_facts_or_error",
                        lambda tok: (dict(partial), "ChatMarkerError: boundary retokenization at token 16"))
    fact, why = st.standup_thinking_header(_Tok(), declares_thinking=False)
    assert fact == partial and why.startswith("ChatMarkerError: ")      # the partial fact, VERBATIM
    with pytest.raises(cmk.ChatMarkerError):
        st.standup_thinking_header(_Tok(), declares_thinking=True)
    monkeypatch.setattr(cmk, "thinking_header_facts",
                        lambda tok: {"switch": True, "moves": True, "off_ids": [1, 2], "on_ids": [1]})
    assert st.standup_thinking_header(_Tok(), declares_thinking=True) == (
        {"switch": True, "moves": True, "off_ids": [1, 2], "on_ids": [1]}, None)


# --------------------------------------------------------------------------- #
# the proxy's construction                                                     #
# --------------------------------------------------------------------------- #


def test_backend_timeout_kwargs_is_empty_without_the_flag_and_the_float_with_it():
    assert st.backend_timeout_kwargs(None) == {}
    assert st.backend_timeout_kwargs(7200) == {"timeout_s": 7200.0}
    assert st.backend_timeout_kwargs("7200") == {"timeout_s": 7200.0}
    for bad in (0, -1, True, "abc"):
        with pytest.raises(ValueError):
            st.backend_timeout_kwargs(bad)


class _ProxyWithStamp:
    def __init__(self, routes, backend, *, base_model, lens_served_positions=None, **kw):
        pass


class _ProxyWithoutStamp:
    def __init__(self, routes, backend, *, base_model, **kw):
        pass


def test_served_positions_passthrough_hands_the_stamp_over_only_when_the_constructor_takes_it():
    table = {"routes": (), st.LENS_SERVED_POSITIONS_KEY: {"si_constv2_mix_full__lora_r4_ce_minus_ctl": ["all"]}}
    kwargs, rec = st.served_positions_passthrough(table, _ProxyWithStamp)
    assert kwargs == {st.LENS_SERVED_POSITIONS_KEY: table[st.LENS_SERVED_POSITIONS_KEY]}
    assert rec == {st.LENS_SERVED_POSITIONS_KEY: table[st.LENS_SERVED_POSITIONS_KEY],
                   st.LENS_SERVED_POSITIONS_PASSTHROUGH_KEY: st.PASSTHROUGH_PASSED}
    kwargs, rec = st.served_positions_passthrough(table, _ProxyWithoutStamp)
    assert kwargs == {}                                    # never a TypeError at the constructor
    assert rec[st.LENS_SERVED_POSITIONS_KEY] == table[st.LENS_SERVED_POSITIONS_KEY]
    assert rec[st.LENS_SERVED_POSITIONS_PASSTHROUGH_KEY].startswith("not passed")
    assert "thinking_on_undeclared" in rec[st.LENS_SERVED_POSITIONS_PASSTHROUGH_KEY]
    # a table without the stamp (a pre-refinement exporter): None, recorded — never manufactured
    kwargs, rec = st.served_positions_passthrough({"routes": ()}, _ProxyWithStamp)
    assert kwargs == {st.LENS_SERVED_POSITIONS_KEY: None} and rec[st.LENS_SERVED_POSITIONS_KEY] is None


def test_the_real_routing_proxy_is_read_for_the_stamp_kwarg_not_assumed():
    """Whichever side of Evals' refinement this tree is on, the passthrough
    agrees with the real constructor's signature — so mt_serving_up can never
    raise a TypeError on it nor silently skip a kwarg the proxy accepts."""
    import inspect

    from sisteer.capeval.serving.proxy import RoutingProxy
    accepts = st.LENS_SERVED_POSITIONS_KEY in inspect.signature(RoutingProxy.__init__).parameters
    kwargs, rec = st.served_positions_passthrough({"routes": (), st.LENS_SERVED_POSITIONS_KEY: {"x": ["all"]}}, RoutingProxy)
    assert (st.LENS_SERVED_POSITIONS_KEY in kwargs) is accepts
    assert (rec[st.LENS_SERVED_POSITIONS_PASSTHROUGH_KEY] == st.PASSTHROUGH_PASSED) is accepts


def test_routing_proxy_attestation_kwargs_hands_the_empty_map_over_and_records_the_facts(tmp_path):
    p = _write(tmp_path, _att({}, {}))
    table = {"routes": (), "thinking_header": {"switch": True, "moves": None, "off_ids": [10], "on_ids": None}}
    digests = {"si_constv2_mix_full__lora_r4_ce_minus_ctl": "ab" * 32}
    kwargs, facts = st.routing_proxy_attestation_kwargs(
        attestation_path=p, table=table, thinking_header_error="ChatMarkerError: x",
        lens_salt_policy="attested", expected_digests=digests, routing_proxy_cls=_ProxyWithStamp)
    # the attested-EMPTY map is handed over with a null error
    assert kwargs["lens_serve_declarations"] == {} and kwargs["lens_serve_declarations_error"] is None
    assert kwargs["resumed_prefill_policy_cells"] == {"none__lens0": "conditional",
                                                      "si_constv2_mix_full__lora_r4_ce_minus_ctl": "conditional"}
    assert kwargs["thinking_header"] == table["thinking_header"]
    assert kwargs["thinking_header_error"] == "ChatMarkerError: x"
    assert kwargs["lens_salt_policy"] == "attested"
    assert kwargs["object_artifact_sha256_by_cell"] == digests and kwargs["object_artifact_sha256_by_cell"] is not digests
    assert kwargs[st.LENS_SERVED_POSITIONS_KEY] is None       # no stamp in this table, passed as None
    # the facts endpoint.json carries: exactly what check_serve_attestation reads
    assert facts["lens_serve_declarations"] == {} and facts["lens_serve_declarations_error"] is None
    assert facts["resumed_prefill_policy_error"] is None
    assert facts["thinking_header"] == table["thinking_header"] and facts["thinking_header_error"] == "ChatMarkerError: x"
    assert facts[st.LENS_SERVED_POSITIONS_PASSTHROUGH_KEY] == st.PASSTHROUGH_PASSED
    # an unattested map: ABSENT from the facts, the error present; the proxy gets None + the reason
    kwargs2, facts2 = st.routing_proxy_attestation_kwargs(
        attestation_path=tmp_path / "missing.json", table=table, thinking_header_error=None,
        lens_salt_policy="attested", expected_digests={}, routing_proxy_cls=_ProxyWithoutStamp)
    assert kwargs2["lens_serve_declarations"] is None and kwargs2["lens_serve_declarations_error"]
    assert "lens_serve_declarations" not in facts2 and facts2["lens_serve_declarations_error"]
    assert st.LENS_SERVED_POSITIONS_KEY not in kwargs2
    assert facts2[st.LENS_SERVED_POSITIONS_PASSTHROUGH_KEY].startswith("not passed")


def test_lens_resolved_specs_reads_lens_cells_only():
    lens_id = "proxy_vllm+lens+interleaved"
    base_id = "proxy_vllm+passthrough+interleaved"       # the base / prompted route: no lens object
    cells = types.SimpleNamespace(canonical=(
        types.SimpleNamespace(cell_id="none", serving_identity=base_id),
        types.SimpleNamespace(cell_id="si_constv2_mix_full__lora_r4_ce_minus_ctl", serving_identity=lens_id),
        types.SimpleNamespace(cell_id="si_x__psr_kl_all", serving_identity=lens_id)))
    specs = {"si_constv2_mix_full__lora_r4_ce_minus_ctl": types.SimpleNamespace(resolved_spec={"positions": "all"}),
             "si_x__psr_kl_all": types.SimpleNamespace(resolved_spec=None),
             "none": types.SimpleNamespace(resolved_spec={"positions": "response"})}
    source = types.SimpleNamespace(get=lambda cid: specs.get(cid))
    assert st.lens_resolved_specs(cells, source) == [{"positions": "all"}]


def test_vacuous_declarations_apply_rule_3_at_the_standup_on_served_positions():
    """A `@thinking` declaration certifies a header-located mask's boundary;
    on a cell with no header-located served op it is VACUOUS (lens rule 3,
    F11) — named at the standup, before any launch. Undeclared cells are never
    listed; a stacked cell is read through its OBJECT leg (the one unwrapper)."""
    decl = {sd.SERVE_OUT_OF_SUPPORT_KEY: [sd.OUT_OF_SUPPORT_THINKING]}
    specs = {
        "si_a__lora_r4_ce_minus_ctl@thinking": {"kind": "lora_subtraction", "positions": "all", **decl},
        "si_b__psr_kl_all@thinking": {"kind": "psr", "positions": "response", **decl},
        "si_c__psr_kl_all@thinking": {"kind": "psr", "positions": "response_all_turns", **decl},
        "si_d__psr_kl_all@thinking": {"kind": "psr", "positions": "last", **decl},
        "si_e__psr_kl_all": {"kind": "psr", "positions": "all"},                       # undeclared: never listed
        "si_f__psr_kl_all@thinking": {"kind": "psr", **decl},                          # no served mode stated
        "stacked_ok@thinking": {"kind": "stacked", "object": {"positions": "response"}, **decl},
        "stacked_vacuous@thinking": {"kind": "stacked", "object": {"positions": "all"}, **decl},
    }
    from sisteer.capeval import manifest as cm
    for k in ("stacked_ok@thinking", "stacked_vacuous@thinking"):
        specs[k]["kind"] = cm.PETRI_STACKED_KIND
    assert st.vacuous_declarations(specs) == ["si_a__lora_r4_ce_minus_ctl@thinking", "si_d__psr_kl_all@thinking",
                                              "si_f__psr_kl_all@thinking", "stacked_vacuous@thinking"]
    assert st.vacuous_declarations({}) == []
