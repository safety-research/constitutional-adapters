"""The truncation ladder's rung probe, pure
(``sisteer.capeval.tinycap.rung_probe``): the ladder constants against the
geometry module, the KV-refusal reader against the REAL
``vllm_server.kv_coverage_problem`` verdict, ``next_rung``, the certificate
(build / write / read / ``certificate_is_valid``'s nine checks, every negative
case named), the selection file reader against slice C's shape, and the
one-implementation cross-checks against ``scripts/tinycap_ladder.py``."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import rung_probe as RP

REPO = Path(__file__).resolve().parents[1]
BCB, BFCL = C.LADDER_ROW_TASKS
DERIVED = "constv2-mix-full__lora_r4_ce_minus_ctl"
PIN_B, PIN_F = C.load_bcb_pin(REPO), C.load_bfcl_pin(REPO)

#: the production-shaped uniform-cache report: 33,000 blocks x 16 = 528,000 tokens
KV_REPORT = {"kv_num_cache_groups": 1, "kv_num_gpu_blocks": 33_000, "kv_block_size": 16}
REFUSAL_18 = ("[standup] launching\nVllmServerError: resolved scheduler state does not match the composed "
              "no-chunking invariant — a silent clamp:\n- worker 0: "
              + str(vs.kv_coverage_problem(KV_REPORT, max_num_seqs=18, max_model_len=32_768)) + "\n")


def _ladder_script():
    """Slice C's ``scripts/tinycap_ladder.py`` by path (registered in
    ``sys.modules`` before exec: its dataclasses resolve annotations through
    ``sys.modules[cls.__module__]``)."""
    name = "tinycap_ladder_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/tinycap_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _selection(n_b=3, n_f=2, prompt_b=9_000, prompt_f=2_100):
    ids_b, ids_f = list(PIN_B["ids"][:n_b]), list(PIN_F["ids"][:n_f])
    return {BCB: {"ids": ids_b, "n": n_b, "n_selected": n_b, "ids_sha256": C.ids_sha256(ids_b),
                  "cap_from": 4096, "cap_to": 16384, "max_prompt_tokens_rung1": prompt_b},
            BFCL: {"ids": ids_f, "n": n_f, "n_selected": n_f, "ids_sha256": C.ids_sha256(ids_f),
                   "cap_from": 1024, "cap_to": 16384, "max_prompt_tokens_rung1": prompt_f}}


def _prov(selection=None, **over):
    kw = dict(tree_sha="f" * 40, model_slug="gemma4e4b-sdfbase", hf="google/gemma-4-E4B-it", rev="e" * 40,
              cfg_path="configs/two_si_gemma4e4b_sdfbase.yaml", cfg_sha256="1" * 64,
              object_digests={DERIVED: "ab" * 32}, selection=selection or _selection(),
              cap_overrides_by_task={BCB: 16384, BFCL: 16384},
              batched_token_budget=589_824, budget_source=sg.MEASURED_BUDGET_SOURCE,
              suffix="ladder-1", probed_cell=DERIVED)
    kw.update(over)
    return RP.Provenance(**kw)


def _walk_18_then_16():
    return [RP.walked_entry(18, RP.WALK_REFUSED, REFUSAL_18), RP.walked_entry(16, RP.WALK_ACCEPTED)]


def _cert(prov, S=16, walked=None, operator=False, **over):
    cert = RP.build_certificate(
        accepted=RP.accepted_block(S, identity=18, operator=operator),
        walked=_walk_18_then_16() if walked is None else walked, provenance=prov,
        probe_run_id="abc1234-deadbeef", standup_s=None if operator else 1200.5, at="2026-09-10T12:00:00+00:00")
    cert.update(over)
    return cert


# --------------------------------------------------------------------------- #
# the ladder                                                                   #
# --------------------------------------------------------------------------- #


def test_the_ladder_is_sized_from_the_measured_gemma_row_and_mirrors_slice_c():
    assert RP.LADDER_WINDOW == 32_768 and RP.LADDER_SEQS == (18, 16, 14, 12)
    row = sg.BUDGET_BY_MODEL["gemma4e4b"]
    assert row.source == sg.MEASURED_BUDGET_SOURCE and row.batched_token_budget == 589_824 == 18 * 32_768
    assert RP.identity_S() == 18 == sg.identity_seqs(32_768, model_key="gemma4e4b") == row.seqs_at(32_768)
    assert RP.ladder_budget() is row
    assert RP.RUNG_SOURCES == ("identity", "probe", "operator") and RP.SEQS_SOURCES == ("model_budget", "override")
    assert RP.LADDER_CAP_TO == C.LADDER_CAP_TO == 16384 and RP.LADDER_WINDOW_SOURCE == "ladder_window"
    assert RP.seqs_source_for(18, 18) == "model_budget" and RP.seqs_source_for(16, 18) == "override"
    assert RP.rung_source_for(18, 18) == "identity" and RP.rung_source_for(16, 18) == "probe"
    assert RP.rung_source_for(16, 18, operator=True) == "operator"
    with pytest.raises(RP.LadderError, match="identity S"):
        RP.rung_source_for(18, 18, operator=True)
    assert RP.accepted_block(16) == {"max_model_len": 32_768, "max_num_seqs": 16, "seqs_source": "override",
                                     "rung_source": "probe"}
    assert RP.accepted_block(18)["rung_source"] == "identity" and RP.accepted_block(18)["seqs_source"] == "model_budget"
    assert RP.accepted_block(14, operator=True)["rung_source"] == "operator"
    for bad in ((17,), (16, {"max_model_len": 16_384}), (36,)):
        with pytest.raises(RP.LadderError):
            RP.accepted_block(bad[0], **(bad[1] if len(bad) > 1 else {}))
    assert RP.parse_ladder_rung("32768,16") == (32_768, 16) and RP.format_ladder_rung((32_768, 16)) == "32768,16"
    with pytest.raises(RP.LadderError):
        RP.parse_ladder_rung("32768")
    # ONE implementation: slice C's script names the same vocabulary and digests
    lad = _ladder_script()
    assert tuple(lad.CERTIFICATE_KEYS) == RP.CERTIFICATE_FIELDS
    assert tuple(lad.ACCEPTED_KEYS) == RP.ACCEPTED_FIELDS
    assert tuple(lad.RUNG_SOURCES) == RP.RUNG_SOURCES and tuple(lad.SEQS_SOURCES) == RP.SEQS_SOURCES
    assert lad.CAP_TO == RP.LADDER_CAP_TO and tuple(lad.LADDER_TASKS) == C.LADDER_ROW_TASKS
    assert dict(lad.LIMITS_LADDER) == dict(C.LADDER_LIMITS) == {"item_selection": 1}
    sel = _selection()
    assert lad.selection_digest(sel) == RP.selection_digest(sel)
    assert lad.to_log_id(BCB, "BigCodeBench/17") == "17" == C.ladder_log_ids(BCB, ["BigCodeBench/17"])[0]
    assert lad.to_log_id(BFCL, "simple_5") == "simple_5" == C.ladder_log_ids(BFCL, ["simple_5"])[0]


def test_the_production_uniform_cache_report_refuses_18_admits_16_and_next_rung_reads_it():
    refused = vs.kv_coverage_problem(KV_REPORT, max_num_seqs=18, max_model_len=32_768)
    assert refused and "KV cache holds 528000 tokens < the 589824-token budget (= 18 x 32768)" in refused
    assert vs.kv_coverage_problem(KV_REPORT, max_num_seqs=16, max_model_len=32_768) is None
    tail = RP.kv_coverage_refusal(REFUSAL_18)
    # the VERDICT line alone (never the generic header, which any scheduler mismatch raises too)
    assert tail and "KV cache holds 528000 tokens" in tail and "no-chunking invariant" not in tail
    assert RP.parse_kv_tokens(tail) == 528_000
    assert RP.next_rung(tail, [16, 14, 12]) == 16                     # 16 x 32,768 = 524,288 <= 528,000 < 589,824

    def verdict(kv, S=18):
        return "- worker 0: " + vs.kv_coverage_problem(
            {"kv_num_cache_groups": 1, "kv_num_gpu_blocks": kv // 16, "kv_block_size": 16},
            max_num_seqs=S, max_model_len=32_768)
    assert RP.next_rung(verdict(400_000), [16, 14, 12]) == 12
    assert RP.next_rung(verdict(300_000), [16, 14, 12]) is None       # no rung fits: no ladder
    assert RP.next_rung("no-chunking invariant — the engine died", [16, 14, 12]) == 16   # no number: the next
    assert RP.next_rung(tail, []) is None
    assert RP.kv_coverage_refusal("[standup] proxy up\n") is None and RP.parse_kv_tokens(None) is None
    # several workers: the SMALLEST cache sizes the rung (thousands separators read too)
    two = verdict(528_000) + "\n" + verdict(500_000).replace("500000", "500,000")
    assert RP.parse_kv_tokens(two) == 500_000 and RP.kv_coverage_refusal(two).count("\n") == 1
    assert RP.KV_REFUSAL_MARKERS == ("KV cache holds", "hybrid KV cache (")
    # a refused walk entry needs a recognised tail; an accepted one carries none
    e = RP.walked_entry(18, RP.WALK_REFUSED, REFUSAL_18)
    assert (e["S"], e["outcome"], e["kv_tokens"]) == (18, "refused", 528_000) and e["refusal_tail"] == REFUSAL_18
    assert RP.walked_entry(16, RP.WALK_ACCEPTED) == {"S": 16, "outcome": "accepted", "kv_tokens": None,
                                                     "refusal_tail": None}
    with pytest.raises(RP.LadderError, match="KV-coverage verdict"):
        RP.walked_entry(18, RP.WALK_REFUSED, "engine died")


# --------------------------------------------------------------------------- #
# the certificate                                                              #
# --------------------------------------------------------------------------- #


def test_a_valid_certificate_passes_writes_reads_back_and_is_accepted_by_slice_c(tmp_path):
    prov = _prov()
    cert = _cert(prov)
    assert RP.certificate_is_valid(cert, prov) is None
    assert tuple(cert) == RP.CERTIFICATE_FIELDS and cert["schema"] == RP.CERTIFICATE_SCHEMA == 2 and cert["suffix"] == "ladder-1"
    # bcb-bfcl-200-widen S6: schema 2 carries the ladder it was walked on and (null here) the re-probe origin
    assert cert["ladder"] == RP.ladder_spec("gemma4e4b").block() == {"model_key": "gemma4e4b", "window": 32_768,
                                                                    "seqs": [18, 16, 14, 12], "identity_S": 18}
    assert cert["reprobed_from"] is None
    assert cert["probed_cell"] == DERIVED == prov.probed_cell and prov.as_dict()["suffix"] == "ladder-1"
    # the writer files only the provenance's suffix / probe cell
    assert RP.build_certificate(accepted=cert["accepted"], walked=cert["walked"], provenance=prov, probe_run_id="r",
                                standup_s=1.0, suffix="ladder-1", probed_cell=DERIVED)["suffix"] == "ladder-1"
    for kw in (dict(suffix="ladder-2"), dict(probed_cell="none")):
        with pytest.raises(RP.LadderError, match="!= the launch"):
            RP.build_certificate(accepted=cert["accepted"], walked=cert["walked"], provenance=prov, probe_run_id="r",
                                 standup_s=1.0, **kw)
    assert cert["accepted"] == {"max_model_len": 32_768, "max_num_seqs": 16, "seqs_source": "override",
                                "rung_source": "probe"}
    assert cert["selection_sha256"] == RP.selection_digest(_selection()) and cert["budget"] == {
        "batched_token_budget": 589_824, "budget_source": "measured"}
    assert cert["object_digests"] == {DERIVED: "ab" * 32} and cert["cap_overrides_by_task"] == {BCB: 16384, BFCL: 16384}
    p = RP.write_certificate(RP.certificate_path(str(tmp_path), "gemma4e4b-sdfbase", "ladder-1"), cert)
    assert p == tmp_path / "gemma4e4b-sdfbase" / "_ladder" / "ladder-1" / "rung_probe.json"
    back, err = RP.read_certificate(p)
    assert back == cert and err is None and RP.certificate_state(back, None, prov) == "valid"
    assert RP.certificate_state(None, None, prov) == "absent"
    (tmp_path / "bad.json").write_text("{")
    c, e = RP.read_certificate(tmp_path / "bad.json")
    assert c is None and "unreadable" in e and RP.certificate_state(c, e, prov).startswith("invalid(")
    assert RP.superseded_certificate_path(p, at=0.0).name == "rung_probe.19700101T000000Z.superseded.json"
    # the identity rung: model_budget / identity, one accepted entry
    c18 = _cert(prov, S=18, walked=[RP.walked_entry(18, RP.WALK_ACCEPTED)])
    assert RP.certificate_is_valid(c18, prov) is None and c18["accepted"]["seqs_source"] == "model_budget"
    # a two-step skip that agrees with the numbers: 18 refused at 400,000 tokens -> 12
    tail12 = REFUSAL_18.replace("528000", "400000")
    c12 = _cert(prov, S=12, walked=[RP.walked_entry(18, RP.WALK_REFUSED, tail12), RP.walked_entry(12, RP.WALK_ACCEPTED)])
    assert RP.certificate_is_valid(c12, prov) is None
    # slice C's reader accepts what this writer files
    lad = _ladder_script()
    assert lad.load_certificate(p)["accepted"] == cert["accepted"]
    with pytest.raises(RP.LadderError):
        RP.ladder_dir(str(tmp_path), "m", "")                        # a certificate is filed per suffix


def _mutate(cert, path, value):
    c = copy.deepcopy(cert)
    d = c
    for k in path[:-1]:
        d = d[k]
    if value is ...:
        del d[path[-1]]
    else:
        d[path[-1]] = value
    return c


def test_every_invalid_certificate_returns_its_reason():
    prov = _prov()
    good = _cert(prov)
    cases = [
        (_mutate(good, ("hf",), ...), "missing"),
        ({**good, "extra": 1}, "extra"),
        (_mutate(good, ("accepted", "max_num_seqs"), "16"), "not a positive int"),
        (_mutate(good, ("schema",), 1), "schema"),
        (_mutate(good, ("schema",), 3), "schema"),
        # bcb-bfcl-200-widen S6: the ladder block is this launch's model ladder; a re-probe origin must be well-formed
        (_mutate(good, ("ladder", "seqs"), [24, 22, 20, 18]), "ladder"),
        (_mutate(good, ("ladder", "model_key"), "nemotron3nano4b"), "ladder"),
        (_mutate(good, ("reprobed_from",), {"certificate_sha256": "zz", "accepted_S": 18}), "reprobed_from.certificate_sha256"),
        (_mutate(good, ("reprobed_from",), {"certificate_sha256": "a" * 64}), "reprobed_from"),
        # a re-probe must walk exactly its one accepted rung, below the first certificate's S
        (_mutate(good, ("reprobed_from",), {"certificate_sha256": "a" * 64, "accepted_S": 18}), "walks exactly its one accepted rung"),
        (_mutate(good, ("at",), "yesterday"), "ISO-8601"),
        (_mutate(good, ("walked", 0, "kv_tokens"), 999), "kv_tokens"),
        (_mutate(good, ("accepted", "max_model_len"), 16_384), "LADDER_WINDOW"),
        # off-ladder S (17), the walk's last entry moved with it
        (_mutate(_mutate(good, ("accepted", "max_num_seqs"), 17), ("walked", 1, "S"), 17), "not on the ladder"),
        # source mismatch: S 16 with model_budget; S 18 with override; operator without --ladder-rung
        (_mutate(good, ("accepted", "seqs_source"), "model_budget"), "seqs_source"),
        (_mutate(_mutate(_cert(prov, S=18, walked=[RP.walked_entry(18, RP.WALK_ACCEPTED)]),
                         ("accepted", "seqs_source"), "override"), ("accepted", "rung_source"), "identity"), "seqs_source"),
        (_mutate(good, ("accepted", "rung_source"), "identity"), "identity"),
        (_mutate(good, ("accepted", "rung_source"), "operator"), "operator certificate never admits"),
        # the budget
        (_mutate(good, ("budget", "batched_token_budget"), 500_000), "budget"),
        # illegal walks
        # an ascending walk (hand-built: the writer itself refuses a refusal whose verdict is another rung's)
        (_cert(prov, S=18, walked=[{"S": 16, "outcome": "refused", "kv_tokens": 528_000, "refusal_tail": REFUSAL_18},
                                   RP.walked_entry(18, RP.WALK_ACCEPTED)]), "walked[0].S"),
        (_cert(prov, S=14, walked=[RP.walked_entry(18, RP.WALK_REFUSED, REFUSAL_18), RP.walked_entry(16, RP.WALK_ACCEPTED),
                                   RP.walked_entry(14, RP.WALK_ACCEPTED)]), "exactly one"),
        # a skip 18 -> 14 whose tail says 528,000 tokens (next_rung -> 16)
        (_cert(prov, S=14, walked=[RP.walked_entry(18, RP.WALK_REFUSED, REFUSAL_18), RP.walked_entry(14, RP.WALK_ACCEPTED)]),
         "does not agree"),
        (_mutate(good, ("walked", 0, "refusal_tail"), "the engine died"), "no KV-coverage refusal"),
        (_mutate(good, ("walked",), []), "walked is empty"),
        (_mutate(good, ("walked", 1, "kv_tokens"), 1), "carries a refusal"),
        # provenance
        (_mutate(good, ("tree_sha",), "a" * 40), "tree_sha"),
        (_mutate(good, ("cfg_sha256",), "2" * 64), "cfg_sha256"),
        (_mutate(good, ("object_digests",), {}), "object_digests"),
        (_mutate(good, ("cap_overrides_by_task", BFCL), 8192), "cap_overrides_by_task"),
        (_mutate(good, ("selection_sha256",), "3" * 64), "selection_sha256"),
        # a certificate filed under another suffix, or written by probing another cell
        (_mutate(good, ("suffix",), "ladder-2"), "suffix 'ladder-2' != this launch's 'ladder-1'"),
        (_mutate(good, ("probed_cell",), "none"), "probed_cell 'none' != this launch's"),
    ]
    for cert, needle in cases:
        why = RP.certificate_is_valid(cert, prov)
        assert why is not None and needle in why, (needle, why)
    # prompt-bound tampering: the selection the launch reads was edited -> the digest moves
    tampered = _selection(prompt_b=9_500)
    why = RP.certificate_is_valid(good, _prov(selection=tampered))
    assert why and "selection_sha256" in why
    # a bound that violates the fit guard (with a matching digest) refuses on the guard
    too_long = _prov(selection=_selection(prompt_b=20_000))
    why = RP.certificate_is_valid(_cert(too_long), too_long)
    assert why and "20000" in why and "32768" in why
    assert RP.fit_guard_reason(_selection(prompt_b=16_384), 32_768) is None
    assert "16385" in RP.fit_guard_reason(_selection(prompt_b=16_385), 32_768)
    # types: an accepted walk entry's S must be an int; walked must be a list
    assert "walked is not a list" in RP.certificate_is_valid(_mutate(good, ("walked",), {}), prov)
    assert "not an object" in RP.certificate_is_valid([], prov)


def test_an_operator_certificate_admits_only_the_launch_that_asked_for_it():
    prov = _prov()
    op = _cert(prov, S=14, walked=[], operator=True)
    assert op["accepted"]["rung_source"] == "operator" and op["walked"] == [] and op["standup_s"] is None
    assert RP.certificate_is_valid(op, prov, operator_rung=(32_768, 14)) is None
    why = RP.certificate_is_valid(op, prov)
    assert why and "operator certificate never admits" in why
    why = RP.certificate_is_valid(op, prov, operator_rung=(32_768, 12))
    assert why and "--ladder-rung 32768,12" in why
    # an operator certificate that walked something is not an operator certificate
    why = RP.certificate_is_valid(_cert(prov, S=14, walked=_walk_18_then_16(), operator=True), prov, operator_rung=(32_768, 14))
    assert why and "walks nothing" in why
    # a probe certificate under an operator launch refuses too (the operator asked for another rung's certificate)
    why = RP.certificate_is_valid(_cert(prov), prov, operator_rung=(32_768, 16))
    assert why and "operator rung needs an operator certificate" in why


# --------------------------------------------------------------------------- #
# the selection file (slice C's shape)                                         #
# --------------------------------------------------------------------------- #


def _selection_doc(sel):
    return {"schema": 1, "generated_by": "scripts/tinycap_ladder.py select", "generated_at": "2026-09-10T00:00:00Z",
            "from_suffix": "bcbbfcl200", "threshold": 0.01, "full_n": 200, "cells": ["none", DERIVED], "detector": {},
            "fired_any": True, "sources": {}, "selection": sel, "selection_sha256": RP.selection_digest(sel)}


def test_read_ladder_selection_reads_slice_cs_file_and_refuses_every_tampering(tmp_path):
    sel = _selection()
    p = tmp_path / "ladder_selection.json"
    p.write_text(json.dumps(_selection_doc(sel), indent=1, sort_keys=True))
    doc = RP.read_ladder_selection(p)
    assert doc["selection"] == sel and doc["selection_sha256"] == RP.selection_digest(sel)
    lad = _ladder_script()
    assert lad.load_selection(p)["selection_sha256"] == doc["selection_sha256"]     # C's loader agrees
    assert RP.SELECTION_BLOCK_FIELDS == ("ids", "n", "n_selected", "ids_sha256", "cap_from", "cap_to",
                                         "max_prompt_tokens_rung1")

    def refuses(edit, needle):
        s = copy.deepcopy(sel)
        edit(s)
        d = _selection_doc(s)
        q = tmp_path / "x.json"
        q.write_text(json.dumps(d))
        with pytest.raises(RP.LadderError, match=needle):
            RP.read_ladder_selection(q)
        return d

    refuses(lambda s: s[BCB].__setitem__("n", 2), "disagree")
    refuses(lambda s: s[BCB].__setitem__("ids_sha256", "0" * 64), "does not recompute")
    refuses(lambda s: s[BCB].__setitem__("cap_to", 8192), "cap_to")
    refuses(lambda s: s[BCB].__setitem__("cap_from", 16384), "cap_from")
    refuses(lambda s: s[BCB].__setitem__("max_prompt_tokens_rung1", 0), "max_prompt_tokens_rung1")
    refuses(lambda s: s[BCB].pop("cap_from"), "keys")
    refuses(lambda s: s.__setitem__("inspect/math", s.pop(BFCL)), "not a ladder row task")
    refuses(lambda s: s[BCB].__setitem__("ids", ["17", "3", "4"]) or s[BCB].__setitem__("ids_sha256", C.ids_sha256(["17", "3", "4"])),
            "BigCodeBench task id")                                   # the log vocabulary is refused
    # a tampered prompt bound with a stale top-level digest refuses on the digest
    d = _selection_doc(sel)
    d["selection"][BCB]["max_prompt_tokens_rung1"] = 9_500
    (tmp_path / "y.json").write_text(json.dumps(d))
    with pytest.raises(RP.LadderError, match="selection_sha256 does not recompute"):
        RP.read_ladder_selection(tmp_path / "y.json")
    (tmp_path / "z.json").write_text(json.dumps({**_selection_doc(sel), "selection": {}}))
    with pytest.raises(RP.LadderError, match="nothing fired"):
        RP.read_ladder_selection(tmp_path / "z.json")
    with pytest.raises(RP.LadderError):
        RP.read_ladder_selection(tmp_path / "absent.json")


def _expected_state(S=18, L=32_768, prefix_caching=True):
    """``vllm_server.expected_scheduler_state``'s dict for a gemma ladder rung
    (the launch's composed no-chunking geometry), built from the module's own
    pins — what the readiness validator holds every worker's report to."""
    return {"max_num_seqs": S, "max_num_batched_tokens": S * L, "max_model_len": L,
            "effective_max_num_scheduled_tokens": S * L, "speculative": None,
            "scheduler_cls": vs.PINNED_DEFAULT_SCHEDULER_CLS, "enable_prefix_caching": prefix_caching,
            **vs._NO_CHUNK_SCHEDULER_PINS}


def _report(expected, **over):
    rep = {k: v for k, v in expected.items()}
    rep.update({"kv_num_cache_groups": 1, "kv_num_gpu_blocks": 40_000, "kv_block_size": 16})   # 640,000 tokens
    rep.update(over)
    return rep


def test_only_an_insufficient_capacity_verdict_is_a_refusal_the_ladder_steps_down_on():
    """The readiness validator raises the SAME no-chunking
    header for a prefix-caching / scheduler mismatch with a HEALTHY cache; that
    is STANDUP_FAILED, never a step-down. The two producer verdicts (uniform,
    hybrid) are recognised; the header alone, an informational KV-size line and
    a crash are not."""
    exp = _expected_state()
    # a real mismatch diagnostic: 640,000 KV tokens (enough for S = 18) but prefix caching resolved off
    with pytest.raises(vs.VllmServerError) as ei:
        vs.validate_scheduler_reports([_report(exp, enable_prefix_caching=False)], exp)
    diag = str(ei.value)
    assert "no-chunking invariant" in diag and "enable_prefix_caching resolved to False" in diag
    assert RP.kv_coverage_refusal("[standup] launching\nVllmServerError: " + diag + "\n") is None
    # the SAME header carrying the real capacity verdict IS a refusal, and reads its number
    with pytest.raises(vs.VllmServerError) as ei:
        vs.validate_scheduler_reports([_report(exp, kv_num_gpu_blocks=33_000)], exp)
    diag = str(ei.value)
    hit = RP.kv_coverage_refusal("VllmServerError: " + diag)
    assert hit and "KV cache holds 528000 tokens < the 589824-token budget (= 18 x 32768)" in hit
    assert "resolved scheduler state" not in hit and RP.parse_kv_tokens(hit) == 528_000
    # a mismatch AND a shortfall in one diagnostic: the verdict is still read (the shortfall is real)
    with pytest.raises(vs.VllmServerError) as ei:
        vs.validate_scheduler_reports([_report(exp, kv_num_gpu_blocks=33_000, enable_prefix_caching=False)], exp)
    assert RP.parse_kv_tokens(RP.kv_coverage_refusal(str(ei.value))) == 528_000
    # a healthy report raises nothing at all
    assert len(vs.validate_scheduler_reports([_report(exp)], exp)) == 1
    # informational KV lines and crashes are not refusals
    for text in ("INFO: KV cache size: 640,000 tokens (40000 blocks x 16)\n",
                 "GPU KV cache usage: 12.5%\nMaximum concurrency for 32768 tokens per request: 19.53x\n",
                 "kv_coverage_problem passed; no-chunking invariant holds\n",
                 "[standup] engine died: CUDA out of memory\n", ""):
        assert RP.kv_coverage_refusal(text) is None, text
        assert RP.parse_kv_tokens(text) is None
    # the HYBRID verdict (the allocator's own max-concurrency), through the real producer
    hybrid = {"kv_num_cache_groups": 2,
              "kv_cache_groups": [{"num_layers": 8, "spec": "FullAttentionSpec", "block_size": 16},
                                  {"num_layers": 24, "spec": "MambaSpec", "block_size": 544}],
              "kv_max_concurrency": 12.5}
    verdict = vs.kv_coverage_problem(hybrid, max_num_seqs=18, max_model_len=32_768)
    assert verdict and verdict.startswith("hybrid KV cache (")
    hit = RP.kv_coverage_refusal("- worker 0: " + verdict)
    assert hit and RP.parse_kv_tokens(hit) == int(12.5 * 32_768) == 409_600
    assert RP.next_rung(hit, [16, 14, 12]) == 12
    assert vs.kv_coverage_problem({**hybrid, "kv_max_concurrency": 18.0}, max_num_seqs=18, max_model_len=32_768) is None
    # a walk entry refused on the header alone is refused by the writer too
    with pytest.raises(RP.LadderError, match="KV-coverage verdict"):
        RP.walked_entry(18, RP.WALK_REFUSED, "VllmServerError: resolved scheduler state does not match the composed "
                                             "no-chunking invariant\n- worker 0: enable_prefix_caching resolved to False")


def test_the_one_per_task_validator_is_shared_by_the_launcher_and_the_dispatch(tmp_path):
    from sisteer.capeval.tinycap import completion
    task = BFCL
    ids = list(PIN_F["ids"][:2])
    assert RP.ladder_task_partial_reason(tmp_path, task, log_ids=ids, cap=16384, n_expected_full=200) == "no partial marker"
    n = len(ids)
    summary = {"task": task, C.METRIC_ACC: 0.5, C.METRIC_N: n, "n_requests": n, "item_set": "pinned200",
               "thinking": "off", "think_split_mode": C.THINK_SPLIT_RECORD_ONLY, "unterminated_policy": "score",
               "max_tokens": 16384, "samples_per_item": 1, "n_scored": n, "n_unscored": 0, "n_scored_samples": n,
               "n_unscored_samples": 0, "n_stop_reason": {"stop": n}, "n_at_cap": 0, "n_think_block_present": 0,
               "n_think_block_unterminated": 0, "n_unterminated_not_at_cap": 0, "n_completion_tokens": 40}
    items = [{"item_id": i, "sample_idx": 0, "attempt": 1, "scored": True, "value": 1.0, "unparseable": False,
              "stop_reason": "stop", "at_cap": False, "think_block_present": False, "think_block_unterminated": False,
              "stray_open_tag": False, "content_empty_with_reasoning": False, "prompt_tokens": 80,
              "completion_tokens": 20, "seed": 0} for i in ids]
    outcomes = {**{f: summary[f] for f in C.TASK_OUTCOMES_SUMMARY_FIELDS}, "n_attempted_items": n,
                "n_attempted_samples": n, "producer_tree_sha": "f" * 40, "items": items}
    completion.finish_task(tmp_path, task=task, n_items=n, n_expected_full=200, limits=dict(C.LADDER_LIMITS),
                           n_rows=n, n_requests=n, n_rejected=0, wall_clock_s=1.0, summary=summary, outcomes=outcomes)
    assert RP.ladder_task_partial_reason(tmp_path, task, log_ids=ids, cap=16384, n_expected_full=200) is None
    assert "n_items" in RP.ladder_task_partial_reason(tmp_path, task, log_ids=ids[:1], cap=16384, n_expected_full=200)
    assert "max_tokens" in RP.ladder_task_partial_reason(tmp_path, task, log_ids=ids, cap=4096, n_expected_full=200)
    assert "n_expected_full" in RP.ladder_task_partial_reason(tmp_path, task, log_ids=ids, cap=16384, n_expected_full=199)
    why = RP.ladder_task_partial_reason(tmp_path, task, log_ids=[ids[0], "other"], cap=16384, n_expected_full=200)
    assert why and "outcomes items != the selection" in why
    # the launcher's per-task rule and the driver's dispatch both read THIS function
    launcher = (REPO / "scripts/modal/tinycap_served.py").read_text()
    assert "return RP.ladder_task_partial_reason(" in launcher
    provider = (REPO / "src/sisteer/capeval/tiers/inspect_served.py").read_text()
    assert "_RP.ladder_task_partial_reason(store.cell_dir, plan.row_task, log_ids=plan.selection" in provider


def test_a_refusal_must_be_a_verdict_about_its_own_rung(tmp_path):
    """A walk entry saying S = 18 over a REAL verdict produced for
    S = 16 is evidence about another geometry — refused by the writer
    (`walked_entry`) and by the reader (`certificate_is_valid`), uniform and
    hybrid alike; a verdict with no shortfall is refused too."""
    prov = _prov()
    kv = {"kv_num_cache_groups": 1, "kv_num_gpu_blocks": 33_000, "kv_block_size": 16}
    for_16 = "- worker 0: " + vs.kv_coverage_problem({**kv, "kv_num_gpu_blocks": 30_000}, max_num_seqs=16, max_model_len=32_768)
    assert RP.kv_coverage_refusal(for_16) and RP.parse_kv_tokens(for_16) == 480_000
    assert RP.verdict_geometry_reason(REFUSAL_18, 18) is None
    why = RP.verdict_geometry_reason(for_16, 18)
    assert why and "names S 16 x L 32768, not this rung's 18 x 32768" in why
    with pytest.raises(RP.LadderError, match="its OWN KV-coverage verdict"):
        RP.walked_entry(18, RP.WALK_REFUSED, for_16)
    # the reader: a hand-built walk carrying S = 16's verdict under the S = 18 entry
    forged = _cert(prov, S=12, walked=[{"S": 18, "outcome": "refused", "kv_tokens": 480_000, "refusal_tail": for_16},
                                       RP.walked_entry(12, RP.WALK_ACCEPTED)])
    why = RP.certificate_is_valid(forged, prov)
    assert why and "walked[0] (S 18): the verdict names S 16" in why
    # another window in the verdict
    at_16k = "- worker 0: " + vs.kv_coverage_problem({**kv, "kv_num_gpu_blocks": 10_000}, max_num_seqs=18,
                                                      max_model_len=16_384)          # 160,000 < 18 x 16,384
    assert "L 16384" in RP.verdict_geometry_reason(at_16k, 18)
    # a verdict whose numbers state no shortfall (an edited tail)
    assert "no shortfall" in RP.verdict_geometry_reason(REFUSAL_18.replace("528000", "600000"), 18)
    assert "budget" in RP.verdict_geometry_reason(REFUSAL_18.replace("589824", "500000"), 18)
    # hybrid: the allocator's verdict for S = 16 under an S = 18 entry
    hybrid = {"kv_num_cache_groups": 2,
              "kv_cache_groups": [{"num_layers": 8, "spec": "FullAttentionSpec", "block_size": 16},
                                  {"num_layers": 24, "spec": "MambaSpec", "block_size": 544}],
              "kv_max_concurrency": 12.5}
    h18 = "- worker 0: " + vs.kv_coverage_problem(hybrid, max_num_seqs=18, max_model_len=32_768)
    h16 = "- worker 0: " + vs.kv_coverage_problem(hybrid, max_num_seqs=16, max_model_len=32_768)
    assert RP.verdict_geometry_reason(h18, 18) is None and RP.walked_entry(18, RP.WALK_REFUSED, h18)["kv_tokens"] == 409_600
    assert "names S 16" in RP.verdict_geometry_reason(h16, 18)
    with pytest.raises(RP.LadderError):
        RP.walked_entry(18, RP.WALK_REFUSED, h16)
    forged_h = _cert(prov, S=12, walked=[{"S": 18, "outcome": "refused", "kv_tokens": 409_600, "refusal_tail": h16},
                                         RP.walked_entry(12, RP.WALK_ACCEPTED)])
    assert "the verdict names S 16" in RP.certificate_is_valid(forged_h, prov)
    # a legal hybrid walk passes
    ok = _cert(prov, S=12, walked=[RP.walked_entry(18, RP.WALK_REFUSED, h18), RP.walked_entry(12, RP.WALK_ACCEPTED)])
    assert RP.certificate_is_valid(ok, prov) is None
    # the certificate binds the probe target: a launch probing another cell is not admitted
    assert "probed_cell" in RP.certificate_is_valid(_cert(prov), _prov(probed_cell="none"))
    assert "suffix" in RP.certificate_is_valid(_cert(prov), _prov(suffix="ladder-2"))


def test_the_strict_validator_holds_the_summary_and_the_outcomes_to_each_other(tmp_path):
    """A PARTIAL whose summary disagrees with its outcome record
    (a torn or edited pair) is never PARTIAL_COMPLETE — the canonical
    `verify_outcomes_against_summary` runs inside the ONE validator."""
    from sisteer.capeval.tinycap import completion
    task, ids = BFCL, list(PIN_F["ids"][:2])
    n = len(ids)
    summary = {"task": task, C.METRIC_ACC: 0.5, C.METRIC_N: n, "n_requests": n, "item_set": "pinned200",
               "thinking": "off", "think_split_mode": C.THINK_SPLIT_RECORD_ONLY, "unterminated_policy": "score",
               "max_tokens": 16384, "samples_per_item": 1, "n_scored": n, "n_unscored": 0, "n_scored_samples": n,
               "n_unscored_samples": 0, "n_stop_reason": {"stop": n}, "n_at_cap": 0, "n_think_block_present": 0,
               "n_think_block_unterminated": 0, "n_unterminated_not_at_cap": 0, "n_completion_tokens": 40}
    items = [{"item_id": i, "sample_idx": 0, "attempt": 1, "scored": True, "value": 1.0, "unparseable": False,
              "stop_reason": "stop", "at_cap": False, "think_block_present": False, "think_block_unterminated": False,
              "stray_open_tag": False, "content_empty_with_reasoning": False, "prompt_tokens": 80,
              "completion_tokens": 20, "seed": 0} for i in ids]
    outcomes = {**{f: summary[f] for f in C.TASK_OUTCOMES_SUMMARY_FIELDS}, "n_attempted_items": n,
                "n_attempted_samples": n, "producer_tree_sha": "f" * 40, "items": items}
    completion.finish_task(tmp_path, task=task, n_items=n, n_expected_full=200, limits=dict(C.LADDER_LIMITS),
                           n_rows=n, n_requests=n, n_rejected=0, wall_clock_s=1.0, summary=summary, outcomes=outcomes)
    assert RP.ladder_task_partial_reason(tmp_path, task, log_ids=ids, cap=16384, n_expected_full=200) is None
    sp = completion.task_summary_path(tmp_path, task)
    # (the verifier names the FIRST disagreement: a moved sample count surfaces as the derived n_attempted_samples)
    for field, value, named in (("n_scored_samples", n + 1, "n_attempted_samples"), ("n_at_cap", 1, "n_at_cap"),
                                ("n_completion_tokens", 41, "n_completion_tokens")):
        doc = dict(summary)                                  # ONE field moved from the pristine summary
        doc[field] = value
        sp.write_text(json.dumps(doc))
        why = RP.ladder_task_partial_reason(tmp_path, task, log_ids=ids, cap=16384, n_expected_full=200)
        assert why and "outcomes disagree with the summary" in why and named in why, (field, why)
    sp.write_text(json.dumps(summary))
    assert RP.ladder_task_partial_reason(tmp_path, task, log_ids=ids, cap=16384, n_expected_full=200) is None


# --------------------------------------------------------------------------- #
# the MODEL-AWARE ladder, the per-cell certificate + re-probe at the common    #
# rung, the wave record                                                       #
# --------------------------------------------------------------------------- #


def test_the_ladder_spec_is_sized_per_model_and_a_provisional_budget_refuses():
    g = RP.ladder_spec("gemma4e4b")
    assert (g.model_key, g.window, g.seqs, g.identity_S) == ("gemma4e4b", 32_768, (18, 16, 14, 12), 18)
    assert g.budget is sg.BUDGET_BY_MODEL["gemma4e4b"] and RP.ladder_budget(g) is g.budget
    assert RP.ladder_spec() == g and RP.identity_S(g) == RP.identity_S() == 18       # the default IS gemma4e4b's ladder
    n = RP.ladder_spec("nemotron3nano4b")
    assert (n.seqs, n.identity_S) == ((24, 22, 20, 18), 24) and n.budget.batched_token_budget == 786_432 == 24 * 32_768
    assert n.block() == {"model_key": "nemotron3nano4b", "window": 32_768, "seqs": [24, 22, 20, 18], "identity_S": 24}
    assert n.as_dict()["budget_source"] == "measured"
    assert RP.accepted_block(22, spec=n) == {"max_model_len": 32_768, "max_num_seqs": 22, "seqs_source": "override",
                                             "rung_source": "probe"}
    assert RP.accepted_block(24, spec=n)["rung_source"] == "identity"
    with pytest.raises(RP.LadderError, match="not on the ladder"):
        RP.accepted_block(16, spec=n)                                   # gemma's rung is not nemotron's
    with pytest.raises(RP.LadderError, match="not on the ladder"):
        RP.accepted_block(22)                                            # and vice versa
    # qwen35_4b and llama8b follow the SHARED table's rows (per-wave served-stack measurements stay recorded in their
    # budget.json outputs, NOT in the table). qwen35_4b is the cluster-swept MEASURED row, so its ladder sizes AT THE
    # LADDER WINDOW 32,768 from that row (720,896 -> identity S 22, rungs (22, 20, 18, 16), S 44 at 16,384 — spelled
    # from the table, never a literal). llama8b's row was a PROVISIONAL ceiling until its MEASURED standup budget
    # landed on the table (2026-09-12; 475,136 at 32,768 -> identity S 14, rungs (14, 12)), so its ladder sizes the
    # same way now; the rows still PROVISIONAL (qwen35_9b, llama3b) REFUSE, naming the measurement
    qrow = sg.BUDGET_BY_MODEL["qwen35_4b"]
    q_ident = min(qrow.batched_token_budget // 32_768, sg.HYBRID_SEQS_CAP)
    q = RP.ladder_spec("qwen35_4b")
    assert qrow.source == "measured" and q.budget is qrow
    assert (q.window, q.seqs, q.identity_S, q.budget.batched_token_budget) == (
        32_768, RP.ladder_seqs_for(q_ident), q_ident, qrow.batched_token_budget)
    assert q.seqs == tuple(range(q_ident, max(RP.LADDER_SEQS_FLOOR, q_ident - RP.LADDER_SEQS_DEPTH) - 1, -RP.LADDER_SEQS_STEP))
    assert q.budget.seqs_at(16_384) == min(qrow.batched_token_budget // 16_384, sg.HYBRID_SEQS_CAP)
    assert RP.accepted_block(q.seqs[1], spec=q)["rung_source"] == "probe" and RP.accepted_block(q.seqs[0], spec=q)["rung_source"] == "identity"
    with pytest.raises(RP.LadderError, match="not on the ladder"):
        RP.accepted_block(q.seqs[-1] - RP.LADDER_SEQS_STEP, spec=q)          # nothing below the ladder's last rung
    # llama8b's shared-table row is MEASURED since its standup measurement landed (475,136 batched tokens, 2026-09-12;
    # every standup attests it), so its ladder sizes from that row exactly like qwen35_4b's — spelled from
    # the table AND pinned as the literal (475,136 -> identity S 14, rungs (14, 12)) so a silent table change is caught; the
    # provisional-ceiling refusal below is the two rows still provisional (qwen35_9b, llama3b)
    lrow = sg.BUDGET_BY_MODEL["llama8b"]
    assert lrow.source == "measured" and lrow.source != sg.PROVISIONAL_BUDGET_SOURCE
    l_ident = min(lrow.batched_token_budget // 32_768, sg.HYBRID_SEQS_CAP)
    lad = RP.ladder_spec("llama8b")
    assert lad.budget is lrow and (lad.window, lad.identity_S, lad.budget.batched_token_budget) == (32_768, l_ident, lrow.batched_token_budget)
    assert lad.seqs == tuple(range(l_ident, max(RP.LADDER_SEQS_FLOOR, l_ident - RP.LADDER_SEQS_DEPTH) - 1, -RP.LADDER_SEQS_STEP)) == RP.ladder_seqs_for(l_ident)
    assert (lrow.batched_token_budget, lad.identity_S, lad.seqs) == (475_136, 14, (14, 12))   # the measured row, pinned
    assert lad.as_dict()["budget_source"] == "measured"
    assert RP.accepted_block(12, spec=lad)["rung_source"] == "probe" and RP.accepted_block(14, spec=lad)["rung_source"] == "identity"
    with pytest.raises(RP.LadderError, match="not on the ladder"):
        RP.accepted_block(16, spec=lad)                                  # gemma's rung is not llama's
    for provisional in ("qwen35_9b", "llama3b"):
        assert sg.BUDGET_BY_MODEL[provisional].source == sg.PROVISIONAL_BUDGET_SOURCE
        with pytest.raises(RP.LadderError, match=f"MEASURED budget row of '{provisional}'.*tinycap_budget_measure"):
            RP.ladder_spec(provisional)
    with pytest.raises(RP.LadderError, match="MEASURED budget row"):
        RP.ladder_spec("no_such_model")
    # the rung rule: down by 2 from the identity S to max(12, identity - 6)
    assert RP.ladder_seqs_for(18) == (18, 16, 14, 12) and RP.ladder_seqs_for(24) == (24, 22, 20, 18)
    assert RP.ladder_seqs_for(22) == (22, 20, 18, 16) and RP.ladder_seqs_for(13) == (13,) and RP.ladder_seqs_for(14) == (14, 12)
    assert RP.ladder_seqs_for(44) == (44, 42, 40, 38)
    with pytest.raises(RP.LadderError):
        RP.ladder_seqs_for(0)
    # code review: an identity BELOW the floor (12) has no admissible rung — it used to return
    # `(identity,)`, a rung under the documented floor, instead of refusing
    for below in (10, 11, 1):
        with pytest.raises(RP.LadderError, match="below the ladder floor 12"):
            RP.ladder_seqs_for(below)
    assert RP.ladder_seqs_for(12) == (12,)                                # the floor itself is a one-rung ladder
    assert RP.next_rung("KV cache holds 700000 tokens < the 786432-token budget (= 24 x 32768)", [22, 20, 18], window=32_768) == 20
    # per-cell certificate paths; the legacy single path still reads
    assert RP.certificate_path("/g", "m", "s", DERIVED).name == f"rung_probe.{DERIVED}.json"
    assert RP.certificate_path("/g", "m", "s").name == "rung_probe.json" and RP.admission_path("/g", "m", "s").name == "rung_admission.json"
    with pytest.raises(RP.LadderError):
        RP.certificate_filename("a/b")


def _nemo_prov(**over):
    n = RP.ladder_spec("nemotron3nano4b")
    kw = dict(model_slug="nemotron3nano4b", hf="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
              cfg_path="configs/tinycap/bcbw_nemotron3nano4b_main.yaml",
              batched_token_budget=n.budget.batched_token_budget, budget_source=n.budget.source, ladder=n)
    kw.update(over)
    return _prov(**kw)


def test_a_nemotron_certificate_walks_its_own_rungs_and_never_admits_a_gemma_launch():
    prov = _nemo_prov()
    n = prov.spec
    assert prov.identity_S == 24 and prov.as_dict()["ladder"] == n.block()
    tail24 = ("VllmServerError: resolved scheduler state does not match the composed no-chunking invariant:\n- worker 0: "
              + str(vs.kv_coverage_problem({"kv_num_cache_groups": 1, "kv_num_gpu_blocks": 46_000, "kv_block_size": 16},
                                           max_num_seqs=24, max_model_len=32_768)) + "\n")
    walked = [RP.walked_entry(24, RP.WALK_REFUSED, tail24), RP.walked_entry(22, RP.WALK_ACCEPTED)]
    cert = RP.build_certificate(accepted=RP.accepted_block(22, spec=n), walked=walked, provenance=prov,
                                probe_run_id="r", standup_s=400.0)
    assert RP.certificate_is_valid(cert, prov) is None and cert["ladder"]["seqs"] == [24, 22, 20, 18]
    assert cert["walked"][0]["kv_tokens"] == 736_000
    # the identity rung
    c24 = RP.build_certificate(accepted=RP.accepted_block(24, spec=n), walked=[RP.walked_entry(24, RP.WALK_ACCEPTED)],
                               provenance=prov, probe_run_id="r", standup_s=1.0)
    assert RP.certificate_is_valid(c24, prov) is None and c24["accepted"]["rung_source"] == "identity"
    # a gemma provenance never admits it (the ladder block, the budget, the rungs all differ)
    why = RP.certificate_is_valid(cert, _prov(probed_cell=DERIVED))
    assert why and ("ladder" in why or "not on the ladder" in why or "budget" in why)
    # a gemma-walked certificate under a nemotron provenance: refused on the rungs / ladder block
    g = _cert(_prov(probed_cell=DERIVED))
    why = RP.certificate_is_valid(g, prov)
    assert why and ("not on the ladder" in why or "ladder" in why)
    # a walk that leaves nemotron's ladder
    with pytest.raises(RP.LadderError, match="not on the ladder"):
        RP.build_certificate(accepted=RP.accepted_block(16, spec=n), walked=[], provenance=prov, probe_run_id="r", standup_s=1.0)


def test_a_re_probe_certificate_walks_one_rung_at_the_common_footprint_and_names_its_origin():
    prov = _prov()
    first = _cert(prov, S=18, walked=[RP.walked_entry(18, RP.WALK_ACCEPTED)])            # the LoRA accepted 18
    origin = {"certificate_sha256": RP.certificate_sha256(first), "accepted_S": 18}
    re = RP.build_certificate(accepted=RP.accepted_block(16), walked=[RP.walked_entry(16, RP.WALK_ACCEPTED)],
                              provenance=prov, probe_run_id="re", standup_s=300.0, reprobed_from=origin)
    assert RP.certificate_is_valid(re, prov) is None and re["reprobed_from"] == origin and re["accepted"]["rung_source"] == "probe"
    # a re-probe that walked two rungs, or accepted a rung not below its origin, or an operator re-probe: refused
    bad_walk = dict(re, walked=[RP.walked_entry(18, RP.WALK_REFUSED, REFUSAL_18), RP.walked_entry(16, RP.WALK_ACCEPTED)])
    assert "walks exactly its one accepted rung" in RP.certificate_is_valid(bad_walk, prov)
    not_below = dict(re, reprobed_from={**origin, "accepted_S": 16})
    assert "not above the re-probed S" in RP.certificate_is_valid(not_below, prov)
    off = dict(re, reprobed_from={**origin, "accepted_S": 17})
    assert "not on the ladder" in RP.certificate_is_valid(off, prov)
    op = _cert(prov, S=14, walked=[], operator=True)
    op_re = dict(op, reprobed_from=origin)
    assert "never a re-probe" in RP.certificate_is_valid(op_re, prov, operator_rung=(32_768, 14))
    assert RP.certificate_sha256(re) != RP.certificate_sha256(first) and len(RP.certificate_sha256(re)) == 64


def test_the_wave_record_admits_only_final_certificates_at_one_common_rung(tmp_path):
    """The HETEROGENEOUS wave: the LoRA's probe accepted 18, the PSR's 16 -> the
    common rung is 16; the record REFUSES while the LoRA's certificate still says 18 and admits
    once the LoRA is re-probed at 16 (a single-rung certificate naming its origin)."""
    lora, psr = "constv2-mix-full__lora_r4_ce", "constv2-mix-full__psr_ce_all"
    p_lora, p_psr = _prov(probed_cell=lora, object_digests={lora: "ab" * 32}), _prov(probed_cell=psr, object_digests={psr: "cd" * 32})
    c_lora = _cert(p_lora, S=18, walked=[RP.walked_entry(18, RP.WALK_ACCEPTED)])
    c_psr = _cert(p_psr, S=16)
    assert RP.common_rung({lora: c_lora, psr: c_psr}) == 16
    common = RP.accepted_block(16)
    spec = RP.ladder_spec("gemma4e4b")
    # without the re-probe: the LoRA's certificate accepted 18 != the common 16 -> refused by name
    doc = RP.build_admission(suffix="ladder-1", common=common, certificates={lora: c_lora, psr: c_psr}, at="2026-09-11T00:00:00+00:00")
    assert tuple(doc) == RP.ADMISSION_FIELDS and doc["cells"][lora] == {"certificate_sha256": RP.certificate_sha256(c_lora),
                                                                       "first_accepted_S": 18, "reprobed": False}
    why = RP.admission_is_valid(doc, suffix="ladder-1", spec=spec, certificates={lora: c_lora, psr: c_psr})
    assert why and "re-probe it at the common footprint" in why and lora in why
    # the re-probe at the common footprint: the LoRA's FINAL certificate accepts 16 and names the first
    c_lora2 = RP.build_certificate(accepted=common, walked=[RP.walked_entry(16, RP.WALK_ACCEPTED)], provenance=p_lora,
                                   probe_run_id="re", standup_s=300.0,
                                   reprobed_from={"certificate_sha256": RP.certificate_sha256(c_lora), "accepted_S": 18})
    final = {lora: c_lora2, psr: c_psr}
    doc = RP.build_admission(suffix="ladder-1", common=common, certificates=final, at="2026-09-11T00:00:00+00:00")
    assert doc["cells"][lora] == {"certificate_sha256": RP.certificate_sha256(c_lora2), "first_accepted_S": 18, "reprobed": True}
    assert doc["cells"][psr] == {"certificate_sha256": RP.certificate_sha256(c_psr), "first_accepted_S": 16, "reprobed": False}
    assert RP.admission_is_valid(doc, suffix="ladder-1", spec=spec, certificates=final) is None
    path = RP.write_admission(RP.admission_path(str(tmp_path), "gemma4e4b-f22it", "ladder-1"), doc)
    back, err = RP.read_admission(path)
    assert back == doc and err is None
    assert RP.admission_state(back, None, suffix="ladder-1", spec=spec, certificates=final) == "valid"
    assert RP.admission_state(None, None, suffix="ladder-1", spec=spec, certificates=final) == "absent"
    # every refusal named: another suffix; a cell recorded but unprobed / probed but unrecorded; a stale sha;
    # a common rung off the ladder; a reprobed flag that disagrees; a first S below the common
    assert "suffix" in RP.admission_is_valid(doc, suffix="ladder-2", spec=spec, certificates=final)
    assert "every probed cell is recorded" in RP.admission_is_valid(doc, suffix="ladder-1", spec=spec, certificates={psr: c_psr})
    stale = copy.deepcopy(doc); stale["cells"][psr]["certificate_sha256"] = "0" * 64
    assert "certificate_sha256" in RP.admission_is_valid(stale, suffix="ladder-1", spec=spec, certificates=final)
    off = copy.deepcopy(doc); off["common"]["max_num_seqs"] = 17
    assert "not a rung of the ladder" in RP.admission_is_valid(off, suffix="ladder-1", spec=spec, certificates=final)
    flag = copy.deepcopy(doc); flag["cells"][psr]["reprobed"] = True
    assert "disagrees with the certificate" in RP.admission_is_valid(flag, suffix="ladder-1", spec=spec, certificates=final)
    first = copy.deepcopy(doc); first["cells"][lora]["first_accepted_S"] = 16
    assert "first_accepted_S" in RP.admission_is_valid(first, suffix="ladder-1", spec=spec, certificates=final)
    # a nemotron record is held to nemotron's ladder
    assert "not a rung of the ladder" in RP.admission_is_valid(doc, suffix="ladder-1", spec=RP.ladder_spec("nemotron3nano4b"), certificates=final)
    assert "not an object" in RP.admission_is_valid([], suffix="ladder-1", spec=spec, certificates=final)
    with pytest.raises(RP.LadderError):
        RP.common_rung({})


def test_the_launcher_asks_the_requested_rung_a_served_rung_stands_for_and_a_native_block_re_derives(monkeypatch):
    """``requested_rung_for``: the S the launcher ASKS the standup for
    a rung — on the lens the rung itself; on the native engine the LARGEST requested rung the engine serves at it (gemma's
    served 15 <- (18, 16) is asked as 18, the budget's identity, so the engine's own kernel step lowers and records it;
    nemotron's one served rung 12 as 24; llama's stand), an S off the ladder refused. ``spec_from_block`` RE-DERIVES a
    native block from its requested rungs through the same step (never restates it), so a forged mapping or a stale
    ceiling refuses by name; the geometry module's two reduction sources are the ladder's below-identity words."""
    import json as _json
    gemma = RP.ladder_spec("gemma4e4b")
    native = RP.native_ladder_spec(gemma)
    assert [RP.requested_rung_for(native, s) for s in native.seqs] == [18, 14, 12]
    assert [RP.requested_rung_for(gemma, s) for s in gemma.seqs] == [18, 16, 14, 12]
    for spec, S in ((native, 16), (native, 18), (gemma, 15), (native, 13)):
        with pytest.raises(RP.LadderError, match=f"ladder rung S {S} is not on the ladder"):
            RP.requested_rung_for(spec, S)
    nemo = RP.native_ladder_spec(RP.ladder_spec("nemotron3nano4b"))
    assert (nemo.seqs, RP.requested_rung_for(nemo, 12)) == ((12,), 24)
    llama = RP.native_ladder_spec(RP.ladder_spec("llama8b"))
    assert [RP.requested_rung_for(llama, s) for s in llama.seqs] == [14, 12]
    assert tuple(sg.LADDER_REDUCTION_SOURCES) == (RP.RUNG_SOURCE_PROBE, RP.RUNG_SOURCE_OPERATOR) == ("probe", "operator")
    # every requested rung asked serves the rung it stands for through the ONE §1.8 step
    for S in native.seqs:
        assert sg.adjusted_seqs("gemma4e4b", native.window, RP.requested_rung_for(native, S)) == S
    # a recorded native block re-derives (the readers' fallback): equal when honest, refused by name when not
    assert RP.spec_from_block(native.block(), gemma.budget) == native
    assert RP.spec_from_block(_json.loads(_json.dumps(native.block())), None).block() == native.block()
    forged = _json.loads(_json.dumps(native.block()))
    forged["served_rungs"][0]["requested"] = [18]
    with pytest.raises(RP.LadderError, match="!= the ladder the lora_native engine serves for its requested rungs"):
        RP.spec_from_block(forged, gemma.budget)
    stale = _json.loads(_json.dumps(native.block()))
    stale["kernel_ceiling"]["max_S"] = 16
    with pytest.raises(RP.LadderError, match="!= the ladder the lora_native engine serves"):
        RP.spec_from_block(stale, gemma.budget)
    monkeypatch.delitem(sg.LORA_MAX_MODULE_DIM_BY_MODEL, "gemma4e4b")
    with pytest.raises(RP.LadderError, match="does not re-derive: .*no pinned LoRA max module dim"):
        RP.spec_from_block(native.block(), gemma.budget)
    assert RP.spec_from_block(gemma.block(), gemma.budget) == gemma            # the lens block needs no width
