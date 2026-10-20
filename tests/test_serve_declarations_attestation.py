"""``lens_serve_declarations`` reaches every attestation surface under ONE
key with ONE vocabulary, mirroring
``test_lens_reduction_mode_attestation.py``: worker install report ->
``steering_attestation.json`` -> the ONE strict reader
(``lens_worker.serve_declarations_attested``) -> ``capctrl_serving_up``'s
``endpoint.json`` -> ``RoutingProxy`` -> ``/v1/models.petri_proxy``. Plus:

- the one-key-one-vocabulary sweep over BOTH value shapes
  (``serve_out_of_support`` lists, ``serve_experiment`` names; a value
  carrying both keys, an unknown name, a non-list refuse);
- workers disagreeing refuse; a worker from an older tree refuses;
- the peer's shapes (Q10 / Q11): an UNDECLARED standup carries ``{}`` +
  ``null`` error (PRESENT, never absent); a declared standup the per-cell
  map + ``null``; an attestation failure the map ABSENT + a non-null error;
  the policy map keys the declared cell under its variant spelling with the
  op's ``conditional`` while the per-request stamp wears
  ``declared-request-unique``;
- the harness-side reader pattern (look up ONE cell, tolerate sibling
  entries of the other shape), pinned so the peer can copy it.

Synthetic data only: invented ids, ``TESTHOST``, tmp paths.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from sisteer.capeval.serving import lens_worker as lw
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving import serve_declaration as sd
from sisteer.capeval.serving import vllm_server as vs

REPO = Path(__file__).resolve().parents[1]
KEY = "lens_serve_declarations"
THINKING = "si_safety__psr_kl_all@thinking"
ALL_TURNS = "si_safety__psr_kl_all@all_turns"
ALL_POS = "si_safety__psr_kl_all@all_positions"
TWIN = "none__lens0"
DECL_THINKING = {"serve_out_of_support": ["thinking"]}
DECL_TURNS = {"serve_experiment": "fit_response_serve_all_turns"}
DECL_POS = {"serve_experiment": "fit_response_serve_all_positions"}
EXP1_MAP = {THINKING: DECL_THINKING, TWIN: DECL_THINKING}
EXP2_MAP = {ALL_TURNS: DECL_TURNS, ALL_POS: DECL_POS}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_report(decls=..., **over):
    rep = {"registration_count": 1, "layers": [3], "oproj_layers": [], "plan_layer": 3,
           "cells": ["none__lens0", "si_safety__psr_kl_all"], "ledger_cells": [],
           "lens_reduction_mode": "stacked",
           "resumed_prefill_policy_cells": {"none__lens0": "conditional",
                                            "si_safety__psr_kl_all": "conditional"}}
    if decls is not ...:
        rep[KEY] = decls
    rep.update(over)
    return rep


def _att(*maps, reports=None):
    reps = [_install_report(m) for m in maps] if reports is None else reports
    return {"registry_dir": "/tmp/synthetic/lens_registry", "rpc_method": "sisteer_install_steering",
            "worker_reports": reps,
            "scheduler_geometry": {"worker_reports": [{"lens_reduction_mode": "stacked"}
                                                      for _ in reps]}}


# --------------------------------------------------------------------------- #
# the key and the vocabulary are ONE                                          #
# --------------------------------------------------------------------------- #


def test_one_key_across_the_three_surfaces():
    assert lw.LENS_SERVE_DECLARATIONS_KEY == px.SERVE_DECLARATIONS_KEY == KEY
    src = (REPO / "scripts/capctrl_serving_up.py").read_text()
    assert '"lens_serve_declarations"' in src and '"lens_serve_declarations_error"' in src
    # the values are serve_declaration's two closed shapes
    assert sd.variant_declarations("thinking") == DECL_THINKING
    assert sd.variant_declarations("all_turns") == DECL_TURNS
    assert sd.variant_declarations("all_positions") == DECL_POS


# --------------------------------------------------------------------------- #
# the ONE strict reader                                                       #
# --------------------------------------------------------------------------- #


class TestStrictReader:
    def test_agreeing_workers_yield_the_map(self):
        assert lw.serve_declarations_attested(_att(EXP1_MAP)) == (EXP1_MAP, None)
        assert lw.serve_declarations_attested(_att(EXP1_MAP, EXP1_MAP)) == (EXP1_MAP, None)
        assert lw.serve_declarations_attested(_att(EXP2_MAP, EXP2_MAP, EXP2_MAP)) == (EXP2_MAP, None)

    def test_an_attested_empty_map_is_a_stated_fact(self):
        """Q10: an undeclared standup attests {} — distinct from None."""
        m, err = lw.serve_declarations_attested(_att({}, {}))
        assert m == {} and err is None

    def test_missing_key_names_the_worker(self):
        m, err = lw.serve_declarations_attested(_att(...))
        assert m is None and KEY in err and "worker report 0" in err
        m, err = lw.serve_declarations_attested(_att(EXP1_MAP, ...))
        assert m is None and "worker report 1" in err

    def test_disagreement_is_none(self):
        m, err = lw.serve_declarations_attested(_att(EXP1_MAP, {}))
        assert m is None and "disagree" in err

    @pytest.mark.parametrize("bad", [
        {THINKING: {"serve_out_of_support": "thinking"}},             # non-list
        {THINKING: {"serve_out_of_support": []}},                     # empty
        {THINKING: {"serve_out_of_support": ["dreaming"]}},           # unknown token
        {THINKING: {"serve_out_of_support": ["thinking", "thinking"]}},
        {ALL_TURNS: {"serve_experiment": "fit_all_serve_none"}},      # unknown name
        {ALL_TURNS: {"serve_experiment": 3}},
        {ALL_TURNS: {**DECL_TURNS, **DECL_THINKING}},                 # both keys
        {ALL_TURNS: {**DECL_TURNS, "positions": "response_all_turns"}},  # a stray key
        {ALL_TURNS: {}},                                              # an empty declaration
        {ALL_TURNS: "fit_response_serve_all_turns"},                  # not a mapping
        {"": DECL_TURNS},                                             # a bad cell id
    ])
    def test_one_key_one_vocabulary_sweep(self, bad):
        m, err = lw.serve_declarations_attested(_att(bad))
        assert m is None and err and KEY in err, (bad, err)

    def test_malformed_shapes_are_none(self):
        for bad in ("x", None, [], {}, {"worker_reports": []}, {"worker_reports": "nope"},
                    {"worker_reports": [{KEY: "not a map"}]}):
            m, err = lw.serve_declarations_attested(bad)
            assert m is None and err, bad

    def test_the_install_payload_round_trips(self):
        report = lw.InstallReport(registration_count=1, layers=(3,), plan_layer=3,
                                  cells=("none__lens0", THINKING), ledger_cells=(),
                                  lens_serve_declarations=EXP1_MAP)
        payload = lw.install_report_payload(report)
        assert payload[KEY] == EXP1_MAP
        expected = {"registration_count": 1, "layers": [3], "oproj_layers": [], "plan_layer": 3,
                    "cells": [TWIN, THINKING], "ledger_cells": []}
        # the frozen validator copies the extra key through untouched
        reports = vs.validate_install_reports([payload, dict(payload)], expected)
        assert lw.serve_declarations_attested({"worker_reports": reports}) == (EXP1_MAP, None)
        # a report built outside install (None) stamps no key
        bare = lw.install_report_payload(lw.InstallReport(registration_count=1, layers=(3,),
                                                          plan_layer=3, cells=("c",), ledger_cells=()))
        assert KEY not in bare


# --------------------------------------------------------------------------- #
# capctrl_serving_up: endpoint.json + bring_up wiring                          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def capctrl():
    return _load("_serve_decl_capctrl", "scripts/capctrl_serving_up.py")


def _standup(capctrl, tmp_path, **over):
    geom = types.SimpleNamespace(max_model_len=4096, max_num_seqs=8,
                                 as_dict=lambda: {"max_model_len": 4096, "max_num_seqs": 8})
    kw = dict(out=tmp_path, model="synthetic-org/synthetic-model", revision="0" * 40,
              cell="safety_refusal__psr_kl_all@thinking", harness_cell=THINKING, geometry=geom,
              launch=None, server=None, proxy_loop=None, proxy_url="http://127.0.0.1:1/v1",
              table_path=tmp_path / "rt.json", tokenizer=None)
    kw.update(over)
    return capctrl.Standup(**kw)


def _write_att(tmp_path, doc):
    (tmp_path / vs.STEERING_ATTESTATION_FILE).write_text(json.dumps(doc))


class TestCapctrlEndpoint:
    def test_declared_standup_writes_the_map_and_a_null_error(self, capctrl, tmp_path, monkeypatch):
        import socket
        monkeypatch.setattr(socket, "gethostname", lambda: "TESTHOST")
        _write_att(tmp_path, _att(EXP1_MAP, EXP1_MAP))
        ep = _standup(capctrl, tmp_path).endpoint()
        assert ep[KEY] == EXP1_MAP and ep["lens_serve_declarations_error"] is None

    def test_undeclared_standup_writes_an_empty_map_and_a_null_error(self, capctrl, tmp_path):
        """Q10: PRESENT and empty, never absent; the error PRESENT and null."""
        _write_att(tmp_path, _att({}, {}))
        ep = _standup(capctrl, tmp_path, cell="safety_refusal__psr_kl_all",
                      harness_cell="si_safety__psr_kl_all").endpoint()
        assert KEY in ep and ep[KEY] == {} and ep["lens_serve_declarations_error"] is None

    def test_attestation_failure_omits_the_map_and_states_why(self, capctrl, tmp_path):
        _write_att(tmp_path, _att(EXP1_MAP, {}))            # workers disagree
        ep = _standup(capctrl, tmp_path).endpoint()
        assert KEY not in ep and "disagree" in ep["lens_serve_declarations_error"]
        (tmp_path / vs.STEERING_ATTESTATION_FILE).unlink()  # no attestation at all
        ep = _standup(capctrl, tmp_path).endpoint()
        assert KEY not in ep and ep["lens_serve_declarations_error"]
        m, err = capctrl.attested_serve_declarations_at(tmp_path / "nope.json")
        assert m is None and err

    def test_the_reader_is_the_workers_own(self, capctrl, tmp_path):
        _write_att(tmp_path, _att(EXP2_MAP))
        assert capctrl.attested_serve_declarations_at(tmp_path / vs.STEERING_ATTESTATION_FILE) == \
            lw.serve_declarations_attested(_att(EXP2_MAP))

    def test_bring_up_hands_the_map_to_the_proxy(self, capctrl):
        import inspect
        src = inspect.getsource(capctrl.bring_up)
        assert "attested_serve_declarations_at(" in src
        assert "lens_serve_declarations=decls" in src
        assert "lens_serve_declarations_error=(decls_error if decls is None else None)" in src
        assert "thinking_header=table.get(\"thinking_header\")" in src
        # X1: the reason the ON certification is unavailable rides beside it
        assert "thinking_header_error=thinking_header_error," in src
        # the served-positions stamp (§2.5.4 narrowing) reaches the proxy from
        # the table under the SAME name, the published passthrough shape a
        # peer standup builds against: lens_served_positions=table.get(...)
        assert 'lens_served_positions=table.get("lens_served_positions")' in src
        assert "lens_salt_policy=lens_salt_policy" in src
        assert "assistant_header_ids_thinking_on=on_header_ids" in src
        assert "assistant_turn_end_ids=turn_end_ids" in src
        assert 'thinking_header=thinking_header)' in src
        assert "object_artifact_sha256_by_cell=dict(expected_digests)" in src
        # the reduction-mode attestation line is the earlier landing's, verbatim
        assert "launch_attestation=({lw.LENS_REDUCTION_MODE_KEY: mode}" in src
        assert "if mode is not None else None)" in src

    def test_endpoint_carries_the_new_launch_facts(self, capctrl, tmp_path):
        _write_att(tmp_path, _att(EXP1_MAP))
        ep = _standup(capctrl, tmp_path, object_sha256="ab" * 32,
                      object_sha256_by_cell={THINKING: "ab" * 32},
                      cells=("safety_refusal__psr_kl_all@thinking",), harness_cells=(THINKING,),
                      lens_salt_policy="request_unique_all",
                      thinking_header={"switch": True, "moves": True, "off_ids": [1], "on_ids": [2]},
                      lens_served_positions={"si_safety__psr_kl_all": ["response"]},
                      ).endpoint("capability")
        assert ep["object_artifact_sha256"] == "ab" * 32
        assert ep["object_artifact_sha256_by_cell"] == {THINKING: "ab" * 32}
        assert ep["cells"] == ["safety_refusal__psr_kl_all@thinking"] and ep["harness_cells"] == [THINKING]
        assert ep["lens_salt_policy"] == "request_unique_all"
        assert ep["thinking_header"]["moves"] is True
        # the served-positions map the proxy was handed, recorded as passed
        # (null when the table carried none — the default Standup)
        assert ep["lens_served_positions"] == {"si_safety__psr_kl_all": ["response"]}
        assert _standup(capctrl, tmp_path).endpoint()["lens_served_positions"] is None
        assert ep["resumed_prefill_policy_cells"] == {"none__lens0": "conditional",
                                                      "si_safety__psr_kl_all": "conditional"}


# --------------------------------------------------------------------------- #
# the proxy echo, and the harness-side reader pattern                          #
# --------------------------------------------------------------------------- #


def _proxy(declarations, error=None):
    route = px.RouteEntry(virtual_model=f"petri__{THINKING}", cell_id=THINKING, cell_class="class_a",
                          serving_identity="proxy_vllm+lens+interleaved", intervention_hash="ab" * 8)

    class _B:
        async def chat_completions(self, payload):
            return {"id": "x", "choices": []}

    return px.RoutingProxy([route], _B(), base_model="synthetic-org/synthetic-model",
                           serving_mode="interleaved", lens_serve_declarations=declarations,
                           lens_serve_declarations_error=error)


def test_the_identity_block_carries_exactly_what_was_attested():
    block = _proxy(EXP1_MAP).models_payload()["petri_proxy"]
    assert block[KEY] == EXP1_MAP and block["lens_serve_declarations_error"] is None
    block = _proxy({}).models_payload()["petri_proxy"]
    assert block[KEY] == {} and block["lens_serve_declarations_error"] is None
    block = _proxy(None, "synthetic: worker report 0 carries no lens_serve_declarations").models_payload()["petri_proxy"]
    assert KEY not in block and block["lens_serve_declarations_error"].startswith("synthetic")
    block = _proxy(None).models_payload()["petri_proxy"]
    assert KEY not in block and "lens_serve_declarations_error" not in block
    # the ProxyClientEngine's read surfaces it on proxy_identity untouched
    from sisteer.capeval.serving.proxy import ProxyClientEngine
    eng = ProxyClientEngine("http://127.0.0.1:1", identity="proxy_vllm+lens+interleaved")
    listing = _proxy(EXP1_MAP).models_payload()
    eng._http_json = lambda url, payload: listing  # type: ignore[method-assign]
    eng.load()
    assert eng.proxy_identity[KEY] == EXP1_MAP


def serve_out_of_support_attested(value, cell):
    """THE HARNESS-SIDE READER PATTERN: look up YOUR cell, never
    validate the whole map; a non-list value or a token outside {"thinking"}
    refuses."""
    entry = value.get(cell, {})
    tokens = entry.get("serve_out_of_support", [])
    if not isinstance(tokens, list) or any(t not in ("thinking",) for t in tokens):
        raise ValueError(f"malformed serve_out_of_support for {cell!r}: {tokens!r}")
    return tokens


def test_harness_side_reader_tolerates_sibling_entries_of_the_other_shape():
    mixed = {**EXP1_MAP, **EXP2_MAP}
    assert serve_out_of_support_attested(mixed, THINKING) == ["thinking"]
    assert serve_out_of_support_attested(mixed, TWIN) == ["thinking"]
    assert serve_out_of_support_attested(mixed, "si_safety__psr_kl_all") == []   # absent -> in support
    assert serve_out_of_support_attested(mixed, ALL_TURNS) == []                  # the other shape
    with pytest.raises(ValueError):
        serve_out_of_support_attested({THINKING: {"serve_out_of_support": "thinking"}}, THINKING)
    with pytest.raises(ValueError):
        serve_out_of_support_attested({THINKING: {"serve_out_of_support": ["dreaming"]}}, THINKING)
    # the <DECL_EXPECTED> literal is exactly ["thinking"] — stable, no volatile member
    assert sd.SERVE_VARIANTS["thinking"] == {"serve_out_of_support": ["thinking"]}


# --------------------------------------------------------------------------- #
# capctrl_serving_up: repeatable --cell and per-cell digests (the peer passes
# a digest for EVERY fitted spelling)                                         #
# --------------------------------------------------------------------------- #

TT_BASE = "safety_refusal__psr_kl_all"
TT_TURNS = f"{TT_BASE}@all_turns"
TT_POS = f"{TT_BASE}@all_positions"
TT_CONST_TURNS = "si_constitution__psr_kl_all@all_turns"
H_BASE, H_TURNS, H_POS = "si_safety__psr_kl_all", ALL_TURNS, ALL_POS
H_CONST_TURNS = "si_constitution__psr_kl_all@all_turns"
SHA_A, SHA_B = "a" * 64, "b" * 64


class TestCapctrlDigests:
    def test_cell_list_normalisation(self, capctrl):
        assert capctrl._as_cell_list("none") == ["none"]
        assert capctrl._as_cell_list(TT_BASE) == [TT_BASE]
        assert capctrl._as_cell_list(["none", TT_BASE, TT_TURNS, TT_BASE]) == [TT_BASE, TT_TURNS]
        assert capctrl._as_cell_list(["none", "none"]) == ["none"]
        with pytest.raises(ValueError):
            capctrl._as_cell_list([""])

    def test_three_spelling_digest_map_on_a_three_cell_standup(self, capctrl):
        """A three-cell standup: a digest for the base AND both variants, all the
        same artifact sha, keyed by the exact --cell spelling."""
        cells = ["none", TT_BASE, TT_TURNS, TT_POS]
        tt = capctrl._as_cell_list(cells)
        digests = capctrl.parse_object_digest_args(
            [f"{TT_BASE}={SHA_A}", f"{TT_TURNS}={SHA_A}", f"{TT_POS}={SHA_A}"], tt)
        assert digests == {TT_BASE: SHA_A, TT_TURNS: SHA_A, TT_POS: SHA_A}
        by_harness = capctrl.expected_artifact_digests_for_cells(
            tt, [H_BASE, H_TURNS, H_POS], digests)
        assert by_harness == {H_BASE: SHA_A, H_TURNS: SHA_A, H_POS: SHA_A}
        # one artifact -> the legacy scalar is that sha (experiment 1's contract)
        assert capctrl.legacy_object_sha256(by_harness) == SHA_A

    def test_one_entry_missing_refuses_naming_the_missing_cell(self, capctrl):
        tt = capctrl._as_cell_list([TT_BASE, TT_TURNS, TT_POS])
        digests = capctrl.parse_object_digest_args([f"{TT_BASE}={SHA_A}", f"{TT_POS}={SHA_A}"], tt)
        with pytest.raises(ValueError) as ei:
            capctrl.expected_artifact_digests_for_cells(tt, [H_BASE, H_TURNS, H_POS], digests)
        assert TT_TURNS in str(ei.value) and "no --expect-object-sha256" in str(ei.value)

    def test_two_artifact_standup_has_two_entries_and_a_null_scalar(self, capctrl):
        tt = capctrl._as_cell_list([TT_TURNS, TT_CONST_TURNS])
        digests = capctrl.parse_object_digest_args(
            [f"{TT_TURNS}={SHA_A}", f"{TT_CONST_TURNS}={SHA_B}"], tt)
        by_harness = capctrl.expected_artifact_digests_for_cells(tt, [H_TURNS, H_CONST_TURNS], digests)
        assert by_harness == {H_TURNS: SHA_A, H_CONST_TURNS: SHA_B}
        assert capctrl.legacy_object_sha256(by_harness) is None
        assert capctrl.legacy_object_sha256({}) is None

    def test_bare_digest_means_the_single_fitted_cell_only(self, capctrl):
        assert capctrl.parse_object_digest_args(SHA_A, [TT_TURNS]) == {TT_TURNS: SHA_A}
        assert capctrl.parse_object_digest_args([SHA_A], ["none", TT_TURNS]) == {TT_TURNS: SHA_A}
        with pytest.raises(ValueError, match="bare digest"):
            capctrl.parse_object_digest_args([SHA_A], [TT_BASE, TT_TURNS])
        assert capctrl.parse_object_digest_args(None, ["none"]) == {}

    def test_mis_keyed_duplicate_or_malformed_digests_refuse(self, capctrl):
        tt = [TT_BASE, TT_TURNS]
        with pytest.raises(ValueError, match="not one of this standup"):
            capctrl.parse_object_digest_args([f"{TT_POS}={SHA_A}"], tt)
        with pytest.raises(ValueError, match="twice"):
            capctrl.parse_object_digest_args([f"{TT_BASE}={SHA_A}", f"{TT_BASE}={SHA_B}"], tt)
        with pytest.raises(ValueError, match="not a 64-hex"):
            capctrl.parse_object_digest_args([f"{TT_BASE}=xyz"], tt)
        # a digest on a cell that serves no object still refuses
        with pytest.raises(ValueError, match="serves no object"):
            capctrl.expected_artifact_digests_for_cells(["none"], ["none"], {"none": SHA_A})

    def test_the_one_cell_function_is_unchanged(self, capctrl):
        assert capctrl.expected_artifact_digests(TT_BASE, H_BASE, SHA_A) == {H_BASE: SHA_A}
        assert capctrl.expected_artifact_digests("none", "none", None) == {}
        with pytest.raises(ValueError):
            capctrl.expected_artifact_digests(TT_BASE, H_BASE, None)

    def test_the_cli_takes_repeated_cells_and_digests(self, capctrl):
        ap = capctrl.build_parser()
        ns = ap.parse_args(["--repo-dir", "r", "--config", "c", "--defenses-dir", "d", "--out", "o",
                            "--stop-file", "s", "--cell", "none", "--cell", TT_BASE, "--cell", TT_TURNS,
                            "--expect-object-sha256", f"{TT_BASE}={SHA_A}",
                            "--expect-object-sha256", f"{TT_TURNS}={SHA_A}",
                            "--lens-salt-policy", "request_unique_all"])
        assert ns.cell == ["none", TT_BASE, TT_TURNS]
        assert ns.expect_object_sha256 == [f"{TT_BASE}={SHA_A}", f"{TT_TURNS}={SHA_A}"]
        assert ns.lens_salt_policy == "request_unique_all"
        # the one-value form is today's behaviour: SCALARS on the Namespace (the
        # tinycap launcher's pin), the readers accept either shape
        ns1 = ap.parse_args(["--repo-dir", "r", "--config", "c", "--defenses-dir", "d", "--out", "o",
                             "--stop-file", "s", "--cell", TT_BASE, "--expect-object-sha256", SHA_A])
        assert ns1.cell == TT_BASE and ns1.expect_object_sha256 == SHA_A
        tt = capctrl._as_cell_list(ns1.cell)
        assert capctrl.parse_object_digest_args(ns1.expect_object_sha256, tt) == {TT_BASE: SHA_A}
        # the multi-cell shape: two --cell values -> a list
        ns2 = ap.parse_args(["--repo-dir", "r", "--config", "c", "--defenses-dir", "d", "--out", "o",
                             "--stop-file", "s", "--cell", "none", "--cell", TT_TURNS,
                             "--expect-object-sha256", f"{TT_TURNS}={SHA_A}"])
        assert list(ns2.cell) == ["none", TT_TURNS] and ns2.expect_object_sha256 == f"{TT_TURNS}={SHA_A}"
        assert capctrl.parse_object_digest_args(ns2.expect_object_sha256,
                                                capctrl._as_cell_list(ns2.cell)) == {TT_TURNS: SHA_A}


# --------------------------------------------------------------------------- #
# X1: a legacy thinking-OFF standup must not depend on ON certifiability #
# --------------------------------------------------------------------------- #
#
# The pinned Nemotron-3 template has a thinking switch whose ON generation
# prompt ends in an OPEN ``<think>\n`` where the completed turn carries
# ``<think></think>``, so chat_markers' prefix rule refuses the ON header —
# while the OFF header derives exactly as before. An ordinary ``--cell none``
# standup used to die there, before the server launched. Now: a standup with
# no cell declaring the thinking render records the ON certification as
# UNAVAILABLE (switch + off_ids kept, on_ids / moves null, the reason beside
# it) and serves thinking-OFF; a DECLARING cell on that template still refuses
# hard. These run the REAL bring_up with only the GPU launch, the upstream
# HTTP client and the aiohttp thread replaced; the tokenizer is the cached
# pinned one (skipped, never fetched, when it is not in the local HF cache).

import asyncio  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402

from sisteer import chat_markers as cmk  # noqa: E402

NEMOTRON = ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f")
NEMOTRON_CONFIG = "configs/two_si_nemotron3nano4b.yaml"
#: the OFF generation header the reviewer reproduced (generation_header_ids)
NEMOTRON_OFF_HEADER = [10, 1503, 19464, 1010, 12, 13]
NEMOTRON_UNAVAILABLE = {"switch": True, "moves": None, "off_ids": NEMOTRON_OFF_HEADER, "on_ids": None}
TT_PSR = "safety_refusal__psr_kl_all"


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


class _FakeServer:
    """What launch_server returns, as far as the standup reads it."""

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


class _RecordingBackend:
    """The upstream client the proxy is built over, recording every payload
    (constructed by bring_up exactly like HTTPBackend: (base_url, api_key=,
    and — only when the standup was given ``--upstream-timeout-s`` — timeout_s=);
    ``timeout_s`` stays None when bring_up left the proxy's default)."""
    instances: list = []

    def __init__(self, base_url, api_key=None, timeout_s=None):
        self.base_url, self.payloads, self.timeout_s = base_url, [], timeout_s
        type(self).instances.append(self)

    async def take_request_failure(self, request_id):
        return None

    async def chat_completions(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        return {"id": "cmpl-stub", "object": "chat.completion", "created": 0,
                "model": payload["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _stubbed_bring_up(capctrl, tmp_path, monkeypatch, *, cell="none", attest=None, **kw):
    """The REAL bring_up on the pinned Nemotron config: every derivation, the
    registry and routing-table exports, the launch composition and the proxy
    are the real ones; only the GPU launch (returns a fake child and writes
    ``attest`` as the workers' attestation when given), the upstream client
    and the aiohttp thread are replaced. Returns ``(Standup, proxy)``."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def fake_launch(launch, *, log_path, **_):
        if attest is not None:
            (Path(log_path).parent / vs.STEERING_ATTESTATION_FILE).write_text(json.dumps(attest))
        return _FakeServer()
    monkeypatch.setattr(vs, "launch_server", fake_launch)
    monkeypatch.setattr(capctrl, "_start_proxy_thread",
                        lambda app, host, port: types.SimpleNamespace(
                            call_soon_threadsafe=lambda f: None, stop=lambda: None))
    built = {}
    monkeypatch.setattr(px, "make_app", lambda proxy: built.setdefault("proxy", proxy))
    monkeypatch.setattr(px, "HTTPBackend", _RecordingBackend)
    geom = types.SimpleNamespace(max_model_len=4096, max_num_seqs=8,
                                 as_dict=lambda: {"max_model_len": 4096, "max_num_seqs": 8})
    # the LENS standup this harness stubs (native-serving-default flipped the library default to
    # lora_native; every lens byte-identity test here names the path it exercises)
    kw.setdefault("serving_path", "lens")
    st = capctrl.bring_up(repo=REPO, config_path=REPO / NEMOTRON_CONFIG, cell=cell,
                          defenses_dir=tmp_path / "cells", out=tmp_path / "out", geom=geom, **kw)
    return st, built["proxy"]


def _chat(vm, *, enable=..., declare=False):
    body = {"model": vm, "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]}
    if enable is not ...:
        body[px.CHAT_TEMPLATE_KWARGS_KEY] = {px.ENABLE_THINKING_KEY: enable}
    if declare:
        body[px.THINKING_ON_KEY] = True
    return body


class TestX1LegacyOffStandupOnNemotron:
    def test_the_pinned_template_derives_off_and_refuses_on(self):
        """The reproduction the fix is bound to (reviewer, both shas): OFF
        derives to the pinned six ids; the whole fact refuses; the new
        non-raising twin states the partial fact and the reason."""
        tok = _cached_tok(*NEMOTRON)
        assert list(cmk.generation_header_ids(tok)) == NEMOTRON_OFF_HEADER
        with pytest.raises(cmk.ChatMarkerError):
            cmk.thinking_header_facts(tok)
        facts, why = cmk.thinking_header_facts_or_error(tok)
        assert facts == NEMOTRON_UNAVAILABLE
        assert why.startswith("ChatMarkerError: ") and "<think>" in why
        # the existing function's behaviour is unchanged: it still raises here

    def test_an_off_only_standup_builds_and_attests_the_unavailable_certification(
            self, capctrl, tmp_path, monkeypatch):
        _cached_tok(*NEMOTRON)  # skip, never fetch, when the pin is not cached
        # the workers attest an EMPTY declaration map (a stated "no declared
        # cell"), so a declared ON request gets past the map check and the
        # header fact is what refuses it
        st, proxy = _stubbed_bring_up(capctrl, tmp_path, monkeypatch, attest=_att({}, {}))
        assert st.thinking_header == NEMOTRON_UNAVAILABLE
        assert st.thinking_header_error.startswith("ChatMarkerError: ")
        # endpoint.json: the partial fact with the reason BESIDE it
        ep = st.endpoint()
        assert ep["thinking_header"] == NEMOTRON_UNAVAILABLE
        assert ep["thinking_header_error"] == st.thinking_header_error
        assert ep[KEY] == {} and ep["lens_serve_declarations_error"] is None
        # the routing table on disk carries the partial fact and loads
        assert px.load_routing_table(st.table_path)[px.THINKING_HEADER_KEY] == NEMOTRON_UNAVAILABLE
        # /v1/models.petri_proxy: the same two keys
        block = proxy.models_payload()["petri_proxy"]
        assert block[px.THINKING_HEADER_KEY] == NEMOTRON_UNAVAILABLE
        assert block[px.THINKING_HEADER_ERROR_KEY] == st.thinking_header_error
        assert block[px.SERVE_DECLARATIONS_KEY] == {}
        # the standup's routes: the base and the none__lens0 twin (a lens route)
        assert set(proxy.served_virtual_models()) == {"petri__base", "petri__none__lens0"}
        backend = _RecordingBackend.instances[-1]
        # thinking-OFF serving proceeds on both routes
        for vm in ("petri__base", "petri__none__lens0"):
            out = asyncio.run(proxy.handle_chat_completions(_chat(vm, enable=False)))
            assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_off"
            out = asyncio.run(proxy.handle_chat_completions(_chat(vm)))
            assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "unspecified"
        # the base route's ON render is accepted (§2.5.5: no mask there)
        out = asyncio.run(proxy.handle_chat_completions(_chat("petri__base", enable=True)))
        assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
        served = len(backend.payloads)
        assert served == 5
        # every lens-route ON request refuses: declared -> thinking_on_unattested
        # (naming the reason), undeclared -> thinking_on_undeclared; none served
        with pytest.raises(px.ProxyRefusal) as ei:
            asyncio.run(proxy.handle_chat_completions(_chat("petri__none__lens0", enable=True, declare=True)))
        assert ei.value.code == "thinking_on_unattested" and ei.value.http_status == 400
        assert st.thinking_header_error in str(ei.value)
        with pytest.raises(px.ProxyRefusal) as ei:
            asyncio.run(proxy.handle_chat_completions(_chat("petri__none__lens0", enable=True)))
        assert ei.value.code == "thinking_on_undeclared"
        assert len(backend.payloads) == served

    def test_a_declaring_cell_on_that_template_still_refuses_hard(self, capctrl, tmp_path, monkeypatch):
        """Do not weaken the declaring path: a cell declaring the thinking-ON
        render (``@thinking``) needs the ON header, and on this template it
        cannot be derived — so the standup refuses BEFORE any registry export
        or launch, exactly as before. Through the real bring_up on a
        synthetic response-fit PSR blob at the store path the registry
        resolves, digest-bound as a fitted cell must be."""
        tok = _cached_tok(*NEMOTRON)
        torch = pytest.importorskip("torch")
        cell_dir = tmp_path / "cells" / TT_PSR
        cell_dir.mkdir(parents=True)
        layers = [0, 2]
        torch.save({"kind": "psr", "layers": layers, "train_direction": False,
                    "positions": "response", "response_header_len": len(NEMOTRON_OFF_HEADER),
                    "directions": {str(l): torch.zeros(8) for l in layers},
                    "gate_w": {str(l): torch.zeros(8) for l in layers},
                    "gate_b": {str(l): torch.zeros(1) for l in layers}},
                   cell_dir / "object.pt")
        sha = hashlib.sha256((cell_dir / "object.pt").read_bytes()).hexdigest()
        exported = []
        from sisteer.capeval.serving import lens_steer
        monkeypatch.setattr(lens_steer, "export_worker_registry",
                            lambda *a, **k: exported.append(1))
        with pytest.raises(cmk.ChatMarkerError):
            _stubbed_bring_up(capctrl, tmp_path, monkeypatch, cell=f"{TT_PSR}@thinking",
                              expect_object_sha256=sha)
        assert exported == [] and not (tmp_path / "out" / "routing_table.json").exists()
        # the helper seam says the same thing on its own: declaring -> hard
        with pytest.raises(cmk.ChatMarkerError):
            capctrl.standup_thinking_header(tok, declares_thinking=True)
        facts, why = capctrl.standup_thinking_header(tok, declares_thinking=False)
        assert facts == NEMOTRON_UNAVAILABLE and why.startswith("ChatMarkerError: ")


# --------------------------------------------------------------------------- #
# the multi-cell standup hands build_server_launch the PetriSpec               #
# --------------------------------------------------------------------------- #


class TestMultiCellStandupHandsThePetriSpecToTheLaunch:
    """Experiment 2's target standup (repeated ``--cell``) died ~100 s in,
    right after ``registry + routing table exported``, composing the launch:
    ``validate_petri_spec: 'dict' object has no attribute 'audit'``. In
    ``bring_up`` the optional-markers loop over the lens cells reused the
    name ``spec`` for each cell's ``resolved_spec``, so once ANY canonical
    cell walked the lens branch the PetriSpec bound at the top of the
    function was gone and ``build_server_launch`` received the last lens
    cell's spec dict instead. Nothing caught it: the GPU gate stands up
    through ``lens_gate_standup.py``, not this path; the real-``bring_up``
    tests above run ``--cell none`` (the ``none__lens0`` twin is a DERIVED
    control leg, outside ``cells.canonical``, so that loop never rebinds
    anything) or refuse before the launch is composed; and every other
    ``bring_up`` consumer stubs the standup wholesale. This drives the real
    ``bring_up`` on two fitted lens cells (each carrying a resolved_spec)
    plus the implicit ``none``, lets the REAL ``build_server_launch`` (and so
    the real ``validate_petri_spec``) run on whatever the standup hands it,
    and asserts that argument is the PetriSpec, never a Mapping."""

    def test_the_launch_is_composed_from_the_petri_spec_not_a_lens_cells_dict(
            self, capctrl, tmp_path, monkeypatch):
        from collections.abc import Mapping

        from sisteer.capeval import manifest as cm
        _cached_tok(*NEMOTRON)  # skip, never fetch, when the pin is not cached
        torch = pytest.importorskip("torch")
        tt_cells = [TT_PSR, "si_constitution__psr_kl_all"]
        digests = []
        for tt in tt_cells:
            cell_dir = tmp_path / "cells" / tt
            cell_dir.mkdir(parents=True)
            layers = [0, 2]
            torch.save({"kind": "psr", "layers": layers, "train_direction": False,
                        "positions": "response", "response_header_len": len(NEMOTRON_OFF_HEADER),
                        "directions": {str(l): torch.zeros(8) for l in layers},
                        "gate_w": {str(l): torch.zeros(8) for l in layers},
                        "gate_b": {str(l): torch.zeros(1) for l in layers}},
                       cell_dir / "object.pt")
            digests.append(f"{tt}={hashlib.sha256((cell_dir / 'object.pt').read_bytes()).hexdigest()}")
        # a SPY, not a stub: record what the standup hands the composer, then
        # run the real composer on it — the argument under test is never faked
        handed = []
        real_build = vs.build_server_launch

        def spy(spec, **kw):
            handed.append(spec)
            return real_build(spec, **kw)
        monkeypatch.setattr(vs, "build_server_launch", spy)

        st, proxy = _stubbed_bring_up(capctrl, tmp_path, monkeypatch, cell=tt_cells,
                                      expect_object_sha256=digests, attest=_att({}, {}))
        assert len(handed) == 1
        spec = handed[0]
        assert isinstance(spec, cm.PetriSpec), type(spec)
        assert not isinstance(spec, Mapping)
        # the PetriSpec built at the top of bring_up, not the loaded manifest:
        # the standup's target, its cell list, its template hash
        assert (spec.target_model, spec.target_revision) == NEMOTRON
        assert spec.cells == ["none", "si_safety__psr_kl_all", "si_constitution__psr_kl_all"]
        assert spec.chat_template_hash == st.chat_template_hash
        assert spec.serving_mode == "interleaved"
        # the lens cells really were in the loop: both are canonical (resolved,
        # not absent) and both are served, beside the base and the twin
        assert set(proxy.served_virtual_models()) == {
            "petri__base", "petri__none__lens0",
            "petri__si_safety__psr_kl_all", "petri__si_constitution__psr_kl_all"}
        assert st.cells == tuple(tt_cells)
        assert st.harness_cells == ("si_safety__psr_kl_all", "si_constitution__psr_kl_all")
        # and the composition the spy passed through is the launch the standup
        # kept: the real composer's argv, with bring_up's own splice at its tail
        assert tuple(st.launch.cmd[-4:]) == ("--gpu-memory-utilization", "0.9",
                                             "--max-logprobs", "64")


# ---------------------------------------------------------------------------- #
# On the Modal standup: the --enable-log-requests opt-in                       #
# ---------------------------------------------------------------------------- #
class TestEnableLogRequestsOptIn:
    def test_the_flag_is_opt_in_recorded_on_the_endpoint_and_off_by_default(self, capctrl, tmp_path):
        """The resolved-sampler readback needs vLLM's `--enable-log-requests` on the LAUNCH; the standup takes
        it as an opt-in (default off, the launch command byte-identical without it), threads it to `bring_up`, and
        records it in endpoint.json as `enable_log_requests` so a reader tells "no request lines" from "never set"."""
        assert capctrl.ENABLE_LOG_REQUESTS_FLAG == "--enable-log-requests"
        ap = capctrl.build_parser()
        base = ["--repo-dir", "r", "--config", "c", "--cell", "none", "--defenses-dir", "d", "--out", "o", "--stop-file", "s"]
        assert ap.parse_args(base).enable_log_requests is False
        assert ap.parse_args(base + [capctrl.ENABLE_LOG_REQUESTS_FLAG]).enable_log_requests is True
        _write_att(tmp_path, _att(EXP1_MAP))
        assert _standup(capctrl, tmp_path).endpoint()["enable_log_requests"] is False
        assert _standup(capctrl, tmp_path, enable_log_requests=True).endpoint()["enable_log_requests"] is True
        src = (REPO / "scripts/capctrl_serving_up.py").read_text()
        # RE-PINNED (native-serving-default, 2026-09-13): `bring_up` gained keyword parameters AFTER the opt-in
        # (`serving_path`, `adapter_bank`, `probe_max_tokens`) and lost its `-> Standup` annotation, because it now
        # returns a lens `Standup` OR the shared core's `native_standup.LoraStandup` (one shape, two classes); the
        # opt-in itself is unchanged — still a keyword `bool = False`, still threaded to the lens body's `Standup`
        # exactly once and to the native front once, and `main` still forwards `bool(args.enable_log_requests)`
        assert "enable_log_requests: bool = False,\n" in src
        assert "serving_path=None, adapter_bank=None, probe_max_tokens: int = 48):" in src
        assert "+ ((ENABLE_LOG_REQUESTS_FLAG,) if enable_log_requests else ()))" in src
        assert "enable_log_requests=bool(args.enable_log_requests)," in src
        assert src.count("enable_log_requests=bool(enable_log_requests))") == 1      # the lens body's Standup
        assert src.count("enable_log_requests=bool(enable_log_requests), probe_max_tokens") == 1   # the native front
