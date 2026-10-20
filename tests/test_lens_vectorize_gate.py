"""scripts/lens_vectorize_gate.py — the CPU-testable half of the GPU
bit-identity gate for the vectorized lens apply, after the adversarial
review that found the first version could produce a false green:

- G1: the reference is a SEPARATE checkout of the base commit — the tree
  facts are checked (reference HEAD == the base sha, candidate != it, both
  clean) and a violation is a tree refusal (exit 2), never a verdict;
- G2: five hard checks, each proven here to FAIL the gate: the attested
  apply paths (reference per_row / RPC absent on the base commit;
  candidate vectorized), the worker's own steering evidence (matched
  observations AND modifying applications > 0), steering-bites (steered
  rows differ from the passthrough oracle), the untouched control rows
  (passthrough AND scale-0 rows equal the oracle), and full response
  evidence;
- G3: the required cases are DERIVED by probing every lens-served arm's
  adapter over its save format's value domain (positions over the hooks
  vocabulary, skip_bos both ways) and keeping what the adapter accepts —
  vector_all yields all/response x skip_bos on/off, psr and both LoRA forms
  all + response, `last` is dropped, antidote/alphasteer/loreft keep their
  one regime — unioned with the adapted signatures of the real artifacts
  present; a real object serves a case only when its adapted signature IS
  the case, else a synthetic written WITH that signature; an uncovered case
  (or a missing multigroup) makes the report incomplete (exit 3), and
  ``complete`` is authoritative in the exit code;
- N1: the live ``multigroup`` case — two or three distinct nonzero cells in
  ONE interleaved batch with passthrough, scale-0, plain and declared-seeded
  rows: per-row comparison, control rows vs the oracle, steering-bites per
  cell, same-prompt cross-cell distinctness, and per-(cell, scale) worker
  evidence, each proven to fail the gate;
- G4: a response without token ids, misaligned records, a non-numeric
  logprob or empty top-logprobs is a SignatureError, not a signature;
- THE REDUCTION CONTRACT (2026-09-04: stacked reductions are the served
  default; 2026-09-06: per op CLASS): the shadow compare is held to a
  bounded bf16-kernel-variance tolerance for REDUCTION ops under the
  stacked mode — the pooled fraction of unequal elements <= 1e-4 and the row /
  group concentration bounds for every reduction op; for LINEAR ops (LoRA,
  LoReFT) every unequal element within 4 bf16 ulps of the per-row EDIT beyond
  one output ulp; for GATED ops (the gated-vector and low-rank pairs) the
  OBSERVED stacked gate within 16 x 2^-24 of the observed per-row gate in
  units of sum|h_i w_i|, a ReLU flip only within that bound, and the gate-edit
  identity applied - ref = eff (g_stacked - g_per_row) v within 4 ulps — and
  to exactness for elementwise ops and for every op under the per-row mode;
  flag disagreements, shadow / record errors, unequal groups without the
  tolerance fields and gated groups without observed gate stats FAIL; the
  gate SETS the mode on every candidate standup and holds the attested apply
  path and ``lens_reduction_mode`` to it; B=1 stays exact in either mode; the
  eight recorded gate runs replay through the rule (TestRealGateRecords).

The sbatch is pinned to the module (family order, the reference clone
from the mirror at the base sha, the gemma tf5141 venv)."""

import importlib.util
import copy
import json
import math
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
# scripts/ goes to the FRONT unconditionally: scripts/modal/lens_vectorize_gate.py shares the bare module name with the
# reviewed gate in scripts/, and 52 test modules put scripts/modal on sys.path — whichever directory is in front at the
# FIRST bare import wins for the whole process. A conditional insert left scripts/ wherever an earlier module had put
# it, and a sharded gate whose shard collected such a module first imported the Modal launcher here (AttributeError:
# module 'modal' has no attribute 'App' under a modal stub; cpu-suite-8968828 shard 1, 2026-09-16).
_SCRIPTS_DIR = str(REPO / "scripts")
if _SCRIPTS_DIR in sys.path:
    sys.path.remove(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)   # the gate lives in scripts/ (as its sibling tests do)

import lens_vectorize_gate as g  # noqa: E402
assert Path(g.__file__).resolve() == (REPO / "scripts" / "lens_vectorize_gate.py").resolve(), \
    f"the bare name resolved to {g.__file__}, not the reviewed gate in scripts/ (sys.path order)"
from sisteer.capeval.serving import lens_adapters as la  # noqa: E402
from sisteer.capeval.serving import lens_worker as lw  # noqa: E402

HEADER = (7, 8, 9, 10)
BASE = g.DEFAULT_REFERENCE_SHA


def _choice(tokens, logprobs=None, ids=None, top=None):
    """A well-formed chat-completion choice (the shape the server returns
    with logprobs + top_logprobs + return_token_ids on)."""
    n = len(tokens)
    logprobs = logprobs or [-0.5] * n
    ids = ids if ids is not None else [100 + i for i in range(n)]
    content = []
    for i, t in enumerate(tokens):
        tl = top[i] if top else [{"token": t, "logprob": logprobs[i]},
                                 {"token": "x", "logprob": logprobs[i] - 1.0}]
        content.append({"token": t, "logprob": logprobs[i], "top_logprobs": tl})
    return {"finish_reason": "length", "token_ids": ids,
            "message": {"content": "".join(tokens)}, "logprobs": {"content": content}}


def _sig(tokens, logprobs=None, ids=None):
    return g.signature(_choice(list(tokens), logprobs, ids))


# --------------------------------------------------------------------------
# G4 — signatures require full evidence
# --------------------------------------------------------------------------


class TestStrictSignature:
    def test_well_formed_signature(self):
        s = _sig("hi", [-0.5, -0.25], ids=[5, 6])
        assert s["token_ids"] == [5, 6] and s["tokens"] == ["h", "i"]
        assert s["logprobs"] == [-0.5, -0.25]
        assert s["top"][0] == sorted([("h", -0.5), ("x", -1.5)])

    def test_top_logprobs_compare_order_free(self):
        a = _choice(["h"], top=[[{"token": "h", "logprob": -0.5}, {"token": "x", "logprob": -2.0}]])
        b = _choice(["h"], top=[[{"token": "x", "logprob": -2.0}, {"token": "h", "logprob": -0.5}]])
        assert g.signatures_equal([g.signature(a)], [g.signature(b)]) == (True, None)

    @pytest.mark.parametrize("mutate,match", [
        (lambda c: c.pop("token_ids"), "token_ids"),
        (lambda c: c.__setitem__("token_ids", []), "token_ids"),
        (lambda c: c.__setitem__("token_ids", ["5", "6"]), "token_ids"),
        (lambda c: c["logprobs"]["content"].pop(), "aligned record"),
        (lambda c: c.__setitem__("logprobs", None), "aligned record"),
        (lambda c: c["logprobs"]["content"][0].__setitem__("logprob", None), "finite number"),
        (lambda c: c["logprobs"]["content"][1].__setitem__("logprob", float("nan")), "finite number"),
        (lambda c: c["logprobs"]["content"][0].__setitem__("top_logprobs", []), "top_logprobs"),
        (lambda c: c["logprobs"]["content"][0].pop("top_logprobs"), "top_logprobs"),
        (lambda c: c["logprobs"]["content"][1]["top_logprobs"][0].__setitem__("logprob", "x"),
         "malformed"),
    ])
    def test_missing_evidence_is_a_signature_error(self, mutate, match):
        c = _choice(["h", "i"])
        mutate(c)
        with pytest.raises(g.SignatureError, match=match):
            g.signature(c)

    def test_signature_errors_fail_the_case(self):
        case = _ok_case()
        case["signature_errors"] = ["candidate: response carries no usable token_ids"]
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert not checks["signatures"]["ok"]
        assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 2


# --------------------------------------------------------------------------
# verdicts (unchanged semantics)
# --------------------------------------------------------------------------


class TestVerdicts:
    def test_pass_fail_unstable(self):
        a = [_sig("ab"), _sig("cd", [-1.0, -2.0])]
        b = [_sig("ab"), _sig("cd", [-1.0, -2.0000001])]
        assert g.verdict([a, a], [a, a])["verdict"] == g.PASS
        v = g.verdict([a, a], [b, b])
        assert v["verdict"] == g.FAIL and v["detail"] == "request 1: logprobs differs at position 1"
        assert g.verdict([a, b], [a, a])["verdict"] == g.UNSTABLE_REF
        assert g.verdict([a, a], [a, b])["verdict"] == g.UNSTABLE_VEC
        assert g.verdict([a], [a])["verdict"] == g.UNSTABLE_REF   # one run is not repeat evidence

    def test_differ_somewhere(self):
        a, b = [_sig("ab")], [_sig("ab", [-0.5, -0.4])]
        assert g.signatures_differ_somewhere(a, b) and not g.signatures_differ_somewhere(a, a)

    def test_mixed_roles_and_targets(self):
        assert g.mixed_roles(1) == ["steered"]
        r = g.mixed_roles(32)
        assert (r.count("steered"), r.count("passthrough"), r.count("scale0")) == (16, 8, 8)
        assert g.prompt_targets(8) == [48, 96, 200, 400, 800, 1200, 48, 96]


# --------------------------------------------------------------------------
# G2 — the hard checks, each proven to fail the gate
# --------------------------------------------------------------------------


HIDDEN_ELEMS = 4096            # elements per compared row in the synthetic shadow records


PSR_KEY = "psr:relu_gated_vector:gemv:response:skipbos0:both"          # a GATED reduction op
LINEAR_KEY = "lora_all:lora_oproj:rankK:response:skipbos0:both"         # a LINEAR reduction op
#: the per-key gate fields a current worker keeps (ShadowCompareRecord._entry)
GATE_KEY_ZERO = {"gate_max_rel_diff": 0.0, "gate_n_rows_unequal": 0, "gate_n_relu_flips": 0,
                 "gate_max_diff_rel_sum": 0.0, "gate_max_flip_rel_sum": None,
                 "gate_n_groups_with_stats": 0, "gate_n_groups_observed": 0, "gate_n_errors": 0,
                 "gate_edit_consistency_max_ulps": None, "gate_exact_rows_n": 0,
                 "gate_exact_rows_unequal_elems": 0, "gate_lowrank_zero_delta_max_ulps_edit": None,
                 "gate_n_groups_recon": 0, "gate_write_recon_mismatch_elems": 0}


def _is_gated_key(key):
    try:
        return g.shadow_key_op_class(f"{key}:B32") == g.OP_CLASS_REDUCTION_GATED
    except g.GateError:
        return False


def _shadow_snapshot(batches=(1, 8, 32), unequal=0, key=PSR_KEY, *, measured=None, flags=0):
    """A ShadowCompareRecord.as_dict() twin: 32 counted comparisons per ladder
    size under ``key``, every group ``B * HIDDEN_ELEMS`` elements. ``unequal``
    groups land on the largest size; WITHOUT ``measured`` they carry the
    zeros a record keeps when the worker recorded no ulp-scale fields (the
    gate reads that as "lacks the tolerance fields"); ``measured`` =
    ``{"n_unequal_elems", "max_ulps_edit"[, "n_elems_compared", "max_frac_row",
    "max_frac_group", gate fields]}`` files a measured deviation on that key
    (and pooled overall) — the two concentration fields default to the
    measured class (2 elements in one 4096-wide row, 3 in one 32-row group)
    unless given, and on a GATED key the gate fields default to the measured
    class too (stats on every unequal group, |delta gate|/sum|hw| 0.8 x 2^-24,
    no ReLU flips, no gate errors) unless given — pass one explicitly (None
    included) to shape a missing-evidence or over-the-bound record. ``flags``
    marks that many of the unequal groups as modified-flag disagreements."""
    by_key = {f"{key}:B{b}": {"n_compared": 32, "n_equal": 32, "n_unequal": 0, "n_unmodified": 0,
                              "n_shadow_errors": 0, "max_abs_diff": 0.0, "max_ulps": 0.0, "max_abs_ref": 0.0,
                              "n_unequal_elems": 0, "n_elems_compared": 32 * b * HIDDEN_ELEMS,
                              "max_ulps_edit": 0.0, "max_frac_row": 0.0, "max_frac_group": 0.0,
                              "n_flag_disagreements": 0, "first_unequal": None, "worst_unequal": None,
                              **GATE_KEY_ZERO}
              for b in batches}
    snap = {"enabled": True, "n_groups_compared": 32 * len(batches), "n_equal": 32 * len(batches),
            "n_unequal": 0, "n_unmodified_groups": 64, "n_shadow_errors": 0, "max_abs_diff": 0.0,
            "max_ulps": 0.0, "max_abs_ref": 0.0, "n_unequal_elems": 0,
            "n_elems_compared": sum(e["n_elems_compared"] for e in by_key.values()), "max_ulps_edit": 0.0,
            "max_frac_row": 0.0, "max_frac_group": 0.0,
            "n_flag_disagreements": 0,
            "first_unequal": None, "worst_unequal": None, "first_shadow_error": None, "by_key": by_key,
            "n_keys_dropped": 0,
            "gate_max_abs_diff": 0.0, "gate_max_rel_diff": 0.0, "gate_min_abs_at_diff": None,
            "gate_n_rows_unequal": 0, "gate_n_relu_flips": 0, "gate_max_diff_rel_sum": 0.0,
            "gate_max_flip_rel_sum": None, "gate_n_groups_with_stats": 0, "gate_n_groups_observed": 0,
            "gate_n_errors": 0, "gate_edit_consistency_max_ulps": None, "gate_exact_rows_n": 0,
            "gate_exact_rows_unequal_elems": 0, "gate_lowrank_zero_delta_max_ulps_edit": None,
            "gate_n_groups_recon": 0, "gate_write_recon_mismatch_elems": 0,
            "first_gate_error": None,
            "n_layer_calls": 96, "n_layer_calls_with_ge2_cells_compared": 0,
            "max_distinct_cells_compared_in_one_layer_call": 1,
            "cells_compared": ["si_constitution__psr_kl_all"], "n_record_errors": 0,
            "first_record_error": None}
    if unequal:
        fu = {"key": f"{key}:B{batches[-1]}", "layer": 7, "cell": "si_constitution__psr_kl_all",
              "site": "block", "rows": batches[-1], "max_abs_diff": 0.03125, "note": "",
              "n_unequal_elems": None, "n_elems": batches[-1] * HIDDEN_ELEMS, "max_ulps_edit": None,
              "max_frac_row": None, "frac_group": None, "max_ulps": None, "max_abs_ref": None}
        e = by_key[fu["key"]]
        snap.update(n_unequal=unequal, n_equal=snap["n_equal"] - unequal, max_abs_diff=0.03125,
                    first_unequal=fu, n_flag_disagreements=flags)
        e.update(n_unequal=unequal, n_equal=32 - unequal, max_abs_diff=0.03125, first_unequal=fu,
                 n_flag_disagreements=flags)
        if flags:
            fu["note"] = "modified-flag disagreement: grouped True, per-row False"
        if measured:
            m = dict(measured)
            if "n_elems_compared" in m:
                e["n_elems_compared"] = int(m["n_elems_compared"])
                snap["n_elems_compared"] = sum(x["n_elems_compared"] for x in by_key.values())
            row = m.get("max_frac_row", 2 / HIDDEN_ELEMS)
            grp = m.get("max_frac_group", 3 / (batches[-1] * HIDDEN_ELEMS))
            e.update(n_unequal_elems=m["n_unequal_elems"], max_ulps_edit=m["max_ulps_edit"],
                     max_frac_row=row, max_frac_group=grp,
                     max_ulps=m.get("max_ulps", m["max_ulps_edit"]), max_abs_ref=m.get("max_abs_ref", 3.5))
            snap.update(n_unequal_elems=m["n_unequal_elems"], max_ulps_edit=m["max_ulps_edit"],
                        max_frac_row=row, max_frac_group=grp,
                        max_ulps=e["max_ulps"], max_abs_ref=e["max_abs_ref"])
            fu.update(n_unequal_elems=m["n_unequal_elems"], max_ulps_edit=m["max_ulps_edit"],
                      max_frac_row=row, frac_group=grp,
                      max_ulps=e["max_ulps"], max_abs_ref=e["max_abs_ref"])
            if _is_gated_key(key):
                # the gate fields (2026-09-06): stats on every unequal group, the measured class
                gate = {"gate_max_diff_rel_sum": m.get("gate_max_diff_rel_sum", 0.8 * g.REDUCTION_GATE_UNIT),
                        "gate_max_flip_rel_sum": m.get("gate_max_flip_rel_sum", None),
                        "gate_edit_consistency_max_ulps": m.get("gate_edit_consistency_max_ulps", 0.5),
                        "gate_n_groups_observed": m.get("gate_n_groups_observed", unequal),
                        "gate_exact_rows_n": m.get("gate_exact_rows_n", 31 * unequal),
                        "gate_exact_rows_unequal_elems": m.get("gate_exact_rows_unequal_elems", 0),
                        "gate_lowrank_zero_delta_max_ulps_edit": m.get("gate_lowrank_zero_delta_max_ulps_edit", None),
                        "gate_n_groups_recon": m.get("gate_n_groups_recon", unequal),
                        "gate_write_recon_mismatch_elems": m.get("gate_write_recon_mismatch_elems", 0),
                        "gate_n_groups_with_stats": m.get("gate_n_groups_with_stats", unequal),
                        "gate_n_relu_flips": m.get("gate_n_relu_flips", 0),
                        "gate_n_errors": m.get("gate_n_errors", 0),
                        "gate_n_rows_unequal": m.get("gate_n_rows_unequal", 3 * unequal),
                        "gate_max_rel_diff": m.get("gate_max_rel_diff", 0.02)}
                e.update(gate)
                snap.update(gate, gate_max_abs_diff=m.get("gate_max_abs_diff", 1.7e-6),
                            gate_min_abs_at_diff=m.get("gate_min_abs_at_diff", 1.4e-5))
            else:
                for k_, v_ in m.items():
                    if k_.startswith("gate_"):
                        e[k_] = v_
                        snap[k_] = v_
    return snap


#: the measured class (microbench 388414 / A/B 388415): a few elements per million,
#: one edit ulp, 2 elements in one 4096-wide row, 3 in one 32-row group
MEASURED = {"n_unequal_elems": 3, "max_ulps_edit": 1.0, "n_elems_compared": 2_000_000,
            "max_frac_row": 2 / HIDDEN_ELEMS, "max_frac_group": 3 / (32 * HIDDEN_ELEMS)}


def _shadow_waves(batches=(1, 8, 32), compared=32, rows=None):
    """Per-wave shadow attribution as drive_shadow records it: one accepted wave
    per ladder size carrying ``compared`` counted comparisons over groups of
    ``rows`` (default: exactly B rows — admission did not split)."""
    return [{"B": b, "attempt": 1, "compared_in_wave": compared, "unequal_in_wave": 0,
             "unmodified_in_wave": 0, "shadow_errors_in_wave": 0,
             "rows_compared_in_wave": (list(rows[i]) if rows else ([b] if compared else [])),
             "accepted": True, "n_unequal_so_far": 0}
            for i, b in enumerate(batches)]


def _paths(*, per_slot_mod=0, grouped_mod=3200, per_slot_app=0, grouped_app=3200,
           unattributed=0, recorded=True):
    """The SelectionRecord's per-path counters as selection_evidence sums them."""
    return {"paths_recorded": recorded, "n_applications_per_slot": per_slot_app,
            "n_applications_grouped": grouped_app, "n_applications_unattributed": unattributed,
            "n_modifying_per_slot": per_slot_mod, "n_modifying_grouped": grouped_mod,
            "n_modifying_unattributed": 0}


def _ev(n_mod=3200, n_obs=3200, matched=True, mode=None, **paths):
    out = {"n_modifying": n_mod, "n_observations": n_obs, "all_observations_matched": matched}
    if paths:
        out.update(_paths(**paths))
    if mode is not None:
        out["lens_reduction_mode"] = mode          # the worker's attested reduction mode
        out["lens_reduction_modes_seen"] = [mode]
    return out


def _bi_ok(intent=0):
    return {"ok": True, "detail": f"1 worker(s) attest BI={intent}", "intent": intent,
            "attested": [{"batch_invariant_env": bool(intent), "batch_invariant_active": bool(intent),
                          "batch_invariant_surface": "_batch_invariant_MODE (vllm 0.25.1)",
                          "batch_invariant_error": None}]}


def _proxy_decision(policy, cell="si_constitution__psr_kl_all", ihash="cd" * 8):
    """The PROXY HALF's block of a green shared-prefill record: the decision
    ``proxy_salt_decision`` reads out of the shipped ``proxy.RoutingProxy``
    against the policy this standup's workers attested, plus the live probe
    through the proxy. Built with the REAL minting so the fixture can never
    describe a salt the proxy would not produce."""
    from sisteer.capeval.serving import proxy as px
    entry = px.RouteEntry(virtual_model="petri__" + cell, cell_id=cell,
                          cell_class="gate", intervention_hash=ihash,
                          serving_identity="proxy_vllm+lens+interleaved")
    regime = px.RoutingProxy.salt_regime_for(lens=True, resumed_prefill_policy=policy)
    mints = [px.RoutingProxy.cache_salt_for(entry, lens=True, resumed_prefill_policy=policy)
             for _ in range(2)]
    forced = regime != px.SALT_REGIME_CELL_STABLE
    return {"attested_policy": policy, "attest_error": None, "policy_cells_present": True,
            "cache_salt_regime": regime, "salt": mints[0], "second_mint": mints[1],
            "salt_is_stable_across_mints": mints[0] == mints[1],
            "unhooked_salt": px.RoutingProxy.cache_salt_for(entry, lens=False),
            "regime_vocabulary": list(px.SALT_REGIMES),
            "cell_stable": px.SALT_REGIME_CELL_STABLE,
            "request_unique": px.SALT_REGIME_REQUEST_UNIQUE,
            "forced_stable_salt": forced,
            "drive_salt": (f"gate|{cell}|{ihash}|u" if forced else mints[0]),
            "live_probe": {"status": 200, "cache_salt_regime": regime,
                           "resumed_prefill_policy": policy, "cell_id": cell}}


def _resumed_ok_record(resumable=None, policy=None):
    """A green shared-prefill record (the ``candidate_resumed`` drive) for one
    resumed-prefill POLICY: the response case of ``_ok_case`` is CONDITIONAL
    (the default here — the safe single-turn resume served and attested, the
    unsafe multi-turn one contained and claimed, the engine serving after),
    ``always`` is the all-positions served-and-attested protocol (repeat and
    multi-turn resume served too), ``never`` the refused-with-marker one.
    ``resumable`` is the pre-policy bool spelling (True = always, False =
    never), kept for the older tests."""
    if policy is None:
        policy = (g.RESUMED_CONDITIONAL if resumable is None
                  else g.RESUMED_ALWAYS if resumable else g.RESUMED_NEVER)
    sig_pass_cold = _sig("abcd", [-0.5, -0.6, -0.7, -0.8])
    sig_pass_warm = _sig("abcd", [-0.5, -0.6, -0.7, -0.8001])
    sig_steer_cold = _sig("abxd", [-0.4, -0.6, -0.9, -0.8])
    rec = {"policy": policy, "resumable": policy == g.RESUMED_ALWAYS,
           "harness_cell": "si_constitution__psr_kl_all",
           "proxy_decision": _proxy_decision(policy),
           "salts": {"passthrough": "gate|passthrough|u",
                     "cell": _proxy_decision(policy)["drive_salt"]},
           "n_prompt_tokens": {"a": 1207, "b": 1209},
           "requests": {"passthrough_cold_b": {"status": 200, "signature": sig_pass_cold, "steered": False},
                        "passthrough_writer_a": {"status": 200, "signature": _sig("wxyz"), "steered": False},
                        "passthrough_warm_b": {"status": 200, "signature": sig_pass_warm, "steered": False},
                        # every STEERED entry carries the worker's failure ledger, CLAIMED (None = nothing
                        # contained: served under the production contract)
                        "steered_cold_b": {"status": 200, "signature": sig_steer_cold, "steered": True,
                                           "id": "chatcmpl-0", "claimed_failure": None},
                        "steered_writer_a": {"status": 200, "signature": _sig("wxyq"), "steered": True,
                                             "id": "chatcmpl-1", "claimed_failure": None}},
           "attest": {"steered_cold_b": {"n_entries": 1, "req_id": "r1", "prefill_resumed": False,
                                         "prefill_cached_positions": 0, "prefill_computed_positions": 0,
                                         "n_applications_modifying": 32, "n_applications_modifying_resumed": 0}},
           "signature_errors": [], "closeness": {"passthrough": g.signature_closeness(sig_pass_warm, sig_pass_cold)}}
    if policy == g.RESUMED_NEVER:
        rec["requests"]["steered_warm_b"] = {"status": None, "steered": True,
                                             "error": "URLError: connection refused",
                                             "claimed_failure": {"error": "the response carried no id to claim "
                                                                          "a failure for"}}
        rec["server_died"] = True
        rec["refusal_marker_in_log"] = True
        return rec
    sig_steer_warm = _sig("abxd", [-0.4, -0.6, -0.9001, -0.8])
    rec["requests"]["steered_warm_b"] = {"status": 200, "signature": sig_steer_warm, "id": "chatcmpl-2",
                                         "steered": True, "claimed_failure": None}
    warm = {"n_entries": 1, "req_id": "chatcmpl-2", "prefill_resumed": True, "prefill_cached_positions": 1200,
            "prefill_computed_positions": 9, "n_applications_modifying": 32,
            "n_applications_modifying_resumed": 32}
    rec["attest"]["steered_warm_b"] = warm
    rec["warm_claimed_failure"] = None
    rec["closeness"]["steered"] = g.signature_closeness(sig_steer_warm, sig_steer_cold)
    rec["n_prompt_tokens"].update(multiturn_a=2031, multiturn_b=2033)
    rec["requests"]["steered_multiturn_writer"] = {"status": 200, "signature": _sig("mnop"), "id": "chatcmpl-4",
                                                   "steered": True, "claimed_failure": None}
    if policy == g.RESUMED_ALWAYS:
        rec["requests"]["steered_warm_b_repeat"] = {"status": 200, "signature": sig_steer_warm, "id": "chatcmpl-3",
                                                    "steered": True, "claimed_failure": None}
        rec["attest"]["steered_warm_b_repeat"] = dict(warm, req_id="chatcmpl-3", prefill_cached_positions=1208,
                                                      prefill_computed_positions=1)
        rec["repeat_claimed_failure"] = None
        rec["requests"]["steered_multiturn_warm"] = {"status": 200, "signature": _sig("mnoq"), "id": "chatcmpl-5",
                                                     "steered": True, "claimed_failure": None}
        rec["attest"]["steered_multiturn_warm"] = dict(warm, req_id="chatcmpl-5", prefill_cached_positions=2016,
                                                       prefill_computed_positions=17)
        rec["multiturn_claimed_failure"] = None
        return rec
    # conditional: the repeat is evidence (here: unsafe — the header straddled a block boundary — and
    # contained), the multi-turn resume CONTAINED — THE one request that must carry a claimed
    # failure — and the engine alive afterwards
    repeat_claim = {"req_id": "chatcmpl-3", "error_type": "ResumedPrefillRefusal",
                    "error": "INSIDE the assistant header"}
    rec["requests"]["steered_warm_b_repeat"] = {"status": 200, "signature": _sig("abcd"), "id": "chatcmpl-3",
                                                "steered": True, "claimed_failure": repeat_claim}
    rec["attest"]["steered_warm_b_repeat"] = {"n_entries": 0, "error": "expected exactly one request entry"}
    rec["repeat_claimed_failure"] = repeat_claim
    mt_claim = {"req_id": "chatcmpl-5", "cell_id": "si_constitution__psr_kl_all",
                "layer": 1, "site": "block", "error_type": "ResumedPrefillRefusal",
                "error": "prior writer's response boundary"}
    rec["requests"]["steered_multiturn_warm"] = {"status": 200, "signature": _sig("mnop"), "id": "chatcmpl-5",
                                                 "steered": True, "claimed_failure": mt_claim}
    rec["attest"]["steered_multiturn_warm"] = {"n_entries": 0, "error": "expected exactly one request entry"}
    rec["multiturn_claimed_failure"] = mt_claim
    rec["requests"]["steered_after_refusal"] = {"status": 200, "signature": sig_steer_cold, "id": "chatcmpl-6",
                                                "steered": True, "claimed_failure": None}
    rec["after_refusal_claimed_failure"] = None
    return rec


def _ok_case():
    return {"case_id": "psr:relu_gated_vector:gemv:response:skipbos0:both", "arm": "psr_kl_all",
            "objects": "real", "batches": [1, 8, 32], "routing": g.ROUTING_VECTORIZED,
            "resumed_prefill": _resumed_ok_record(policy=g.RESUMED_CONDITIONAL),
            "apply_path": {g.REFERENCE: {"rpc": "absent", "status": 500, "detail": "AttributeError"},
                           g.CANDIDATE: g.APPLY_PATH_VECTORIZED_STACKED, g.SHADOW: g.APPLY_PATH_SHADOW},
            "steering_evidence": {
                g.REFERENCE: _ev(),                                    # the base commit: no path counters, no mode
                g.CANDIDATE: _ev(grouped_mod=3200, grouped_app=3200, mode="stacked"),
                g.SHADOW: _ev(grouped_mod=3200, grouped_app=3200, mode="stacked")},
            "controls_B1": {g.REFERENCE: {"controls_match": True, "steered_differs": True},
                            g.CANDIDATE: {"controls_match": True, "steered_differs": True}},
            "bi_attested": {g.REFERENCE: {"ok": True, "detail": "no worker attests its batch-invariance "
                                                                "state (accepted: base-commit reference at BI=0)"},
                            g.CANDIDATE: _bi_ok(), g.SHADOW: _bi_ok()},
            "shadow": _shadow_snapshot(),
            "shadow_waves": _shadow_waves(),
            "signature_errors": [],
            "verdicts": {"pure_B1": {"verdict": g.PASS},
                         "pure_B8": {"verdict": g.EVIDENCE, "evidence": True, "equal": False,
                                     "underlying": g.FAIL},
                         "mixed_B32": {"verdict": g.EVIDENCE, "evidence": True, "equal": True,
                                       "underlying": g.PASS}}}


def _exit_with(case, **over):
    case = dict(case, **over)
    case["checks"] = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
    return g.overall_exit_code({"cases": [case], "complete": True}), case["checks"]


class TestHardChecks:
    def test_all_ok_is_green(self):
        code, checks = _exit_with(_ok_case())
        assert code == 0 and all(c["ok"] for c in checks.values())

    def test_reference_apply_path(self):
        # (a) absent RPC is fine ONLY on the base commit; per_row is fine anywhere
        assert g.apply_path_ok_reference({"rpc": "absent"}, reference_sha=BASE, base_sha=BASE)
        assert g.apply_path_ok_reference("per_row", reference_sha="f" * 40, base_sha=BASE)
        assert not g.apply_path_ok_reference({"rpc": "absent"}, reference_sha="f" * 40, base_sha=BASE)
        assert not g.apply_path_ok_reference("vectorized", reference_sha=BASE, base_sha=BASE)
        assert not g.apply_path_ok_reference(None, reference_sha=BASE, base_sha=BASE)
        code, checks = _exit_with(_ok_case(), apply_path={g.REFERENCE: "vectorized",
                                                          g.CANDIDATE: "vectorized"})
        assert code == 2 and not checks["reference_apply_path"]["ok"]

    def test_candidate_apply_path(self):
        # (b) both sides per_row = the same path on both sides -> FAIL
        code, checks = _exit_with(_ok_case(), apply_path={g.REFERENCE: "per_row",
                                                          g.CANDIDATE: "per_row"})
        assert code == 2 and not checks["candidate_apply_path"]["ok"]
        code, checks = _exit_with(_ok_case(), apply_path={g.REFERENCE: {"rpc": "absent"}})
        assert code == 2 and not checks["candidate_apply_path"]["ok"]

    def test_legacy_ab_path_when_run(self):
        code, checks = _exit_with(_ok_case(), apply_path={
            g.REFERENCE: {"rpc": "absent"}, g.CANDIDATE: "vectorized", g.LEGACY: "vectorized"})
        assert code == 2 and not checks["legacy_ab_apply_path"]["ok"]

    @pytest.mark.parametrize("side", [g.REFERENCE, g.CANDIDATE])
    def test_evidence_requires_matched_and_modifying(self, side):
        # (c) nothing steered by the worker's own count -> FAIL
        ev = dict(_ok_case()["steering_evidence"])
        ev[side] = {"n_modifying": 0, "n_observations": 3200, "all_observations_matched": True}
        code, checks = _exit_with(_ok_case(), steering_evidence=ev)
        assert code == 2 and not checks[f"evidence_{side}"]["ok"]
        ev[side] = {"n_modifying": 10, "n_observations": 10, "all_observations_matched": False}
        code, checks = _exit_with(_ok_case(), steering_evidence=ev)
        assert code == 2 and not checks[f"evidence_{side}"]["ok"]
        code, checks = _exit_with(_ok_case(), steering_evidence={})   # missing = failed
        assert code == 2

    @pytest.mark.parametrize("side", [g.REFERENCE, g.CANDIDATE])
    def test_steering_bites_B1(self, side):
        # (d) the B=1 steered singleton bit-identical to the oracle -> FAIL; unknown -> FAIL
        c1 = _ok_case()["controls_B1"]
        c1[side] = dict(c1[side], steered_differs=False)
        code, checks = _exit_with(_ok_case(), controls_B1=c1)
        assert code == 2 and "nothing was applied" in checks[f"steering_bites_B1_{side}"]["detail"]
        c1[side] = {"controls_match": True}
        code, _ = _exit_with(_ok_case(), controls_B1=c1)
        assert code == 2

    @pytest.mark.parametrize("side", [g.REFERENCE, g.CANDIDATE])
    def test_controls_B1(self, side):
        # (e) the B=1 passthrough / scale-0 singleton moved -> FAIL
        c1 = _ok_case()["controls_B1"]
        c1[side] = dict(c1[side], controls_match=False, controls_detail="scale0: request 0: logprobs")
        code, checks = _exit_with(_ok_case(), controls_B1=c1)
        assert code == 2 and "scale0" in checks[f"controls_B1_{side}"]["detail"]

    def test_no_oracle_fails_the_oracle_checks(self):
        case = _ok_case()
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=False)
        assert not checks["controls_B1_candidate"]["ok"]
        assert not checks["steering_bites_B1_reference"]["ok"]

    def test_shadow_apply_path_and_bi_attestation(self):
        ap = dict(_ok_case()["apply_path"], **{g.SHADOW: g.APPLY_PATH_VECTORIZED})
        code, checks = _exit_with(_ok_case(), apply_path=ap)
        assert code == 2 and not checks["shadow_apply_path"]["ok"]
        bi = dict(_ok_case()["bi_attested"])
        bi[g.CANDIDATE] = {"ok": False, "detail": "intent BI=0 but a worker attests env=True active=True"}
        code, checks = _exit_with(_ok_case(), bi_attested=bi)
        assert code == 2 and not checks["batch_invariant_candidate"]["ok"]
        del bi[g.SHADOW]
        code, checks = _exit_with(_ok_case(), bi_attested=bi)
        assert code == 2 and "no attestation" in checks["batch_invariant_candidate_shadow"]["detail"]

    def test_shadow_compare_check(self):
        # a counted comparison in some wave at every ladder size, all equal -> ok
        code, checks = _exit_with(_ok_case())
        assert code == 0 and checks["shadow_compare"]["ok"] is True
        assert "counted comparisons per submitted wave size: {1: 32, 8: 32, 32: 32}" in checks["shadow_compare"]["detail"]
        # one unequal group whose record carries NO tolerance fields -> FAIL (missing evidence,
        # never kernel variance) with the first unequal named
        code, checks = _exit_with(_ok_case(), shadow=_shadow_snapshot(unequal=1))
        assert code == 2 and checks["shadow_compare"]["ok"] is False
        assert "lacks the tolerance fields" in checks["shadow_compare"]["detail"]
        assert "max |diff| 0.03125" in checks["shadow_compare"]["detail"]
        assert "'layer': 7" in checks["shadow_compare"]["detail"]
        # a ladder size whose waves carried no counted comparison -> INCONCLUSIVE (exit 3),
        # naming the size — even though the cumulative record has comparisons
        code, checks = _exit_with(_ok_case(), shadow_waves=_shadow_waves(batches=(1, 8)))
        assert code == 3 and checks["shadow_compare"]["ok"] is None
        assert "B=[32]" in checks["shadow_compare"]["detail"]
        waves = _shadow_waves()
        waves[2]["compared_in_wave"] = 0
        code, checks = _exit_with(_ok_case(), shadow_waves=waves)
        assert code == 3 and "B=[32]" in checks["shadow_compare"]["detail"]
        # no per-wave attribution at all -> INCONCLUSIVE, never a pass on the cumulative count
        code, checks = _exit_with(_ok_case(), shadow_waves=None)
        assert code == 3 and "no per-wave shadow attribution" in checks["shadow_compare"]["detail"]
        # shadow errors / record errors -> FAIL
        snap = _shadow_snapshot()
        snap.update(n_shadow_errors=1, first_shadow_error={"cell": "x", "error": "AdapterError: y"})
        code, checks = _exit_with(_ok_case(), shadow=snap)
        assert code == 2 and "shadow errors" in checks["shadow_compare"]["detail"]
        # the shadow standup did not enable the shadow / no record -> FAIL
        code, checks = _exit_with(_ok_case(), shadow=dict(_shadow_snapshot(), enabled=False))
        assert code == 2
        code, checks = _exit_with(_ok_case(), shadow={"rpc": "absent"})
        assert code == 2 and "no sisteer_shadow_compare" in checks["shadow_compare"]["detail"]
        # group row counts are kernel-shape EVIDENCE, never coverage
        assert g.compared_at(_shadow_snapshot(), 8) == 32 and g.compared_at(_shadow_snapshot(), 64) == 0
        assert g.rows_compared(_shadow_snapshot()) == [1, 8, 32]
        assert g.rows_compared({"rpc": "absent"}) == [] and g.rows_compared(None) == []

    def test_admission_split_groups_still_cover_the_wave(self):
        """A B=32 HTTP wave that vLLM admitted as groups of 3 and 5 rows (no
        ``:B32`` key anywhere) is COVERED: the comparisons are attributed to the
        wave that submitted them; the row counts are reported as evidence."""
        snap = _shadow_snapshot(batches=(1, 3, 5))                     # keys B1, B3, B5 only
        waves = _shadow_waves(batches=(1, 8, 32), compared=32, rows=([1], [3, 5], [3, 5]))
        code, checks = _exit_with(_ok_case(), shadow=snap, shadow_waves=waves)
        assert code == 0 and checks["shadow_compare"]["ok"] is True
        assert "group rows compared (kernel-shape evidence): [1, 3, 5]" in checks["shadow_compare"]["detail"]
        assert g.compared_at(snap, 32) == 0                            # and that is fine
        # compared_per_wave_size sums over every wave of a size, re-driven ones included
        waves = _shadow_waves(batches=(1, 8), compared=0) + _shadow_waves(batches=(8,), compared=4)
        assert g.compared_per_wave_size(waves, (1, 8, 32)) == {1: 0, 8: 4, 32: 0}
        assert g.compared_per_wave_size(None, (1,)) == {1: 0}
        assert g.compared_per_wave_size([{"B": "x"}, "junk"], (1,)) == {1: 0}

    def test_shadow_wave_delta_attributes_to_the_wave(self):
        key = "psr:relu_gated_vector:gemv:response:skipbos0:both"
        before = {"n_groups_compared": 10, "n_unequal": 0, "n_unmodified_groups": 2, "n_shadow_errors": 0,
                  "by_key": {f"{key}:B1": {"n_compared": 10}}}
        after = {"n_groups_compared": 17, "n_unequal": 1, "n_unmodified_groups": 3, "n_shadow_errors": 0,
                 "by_key": {f"{key}:B1": {"n_compared": 10}, f"{key}:B3": {"n_compared": 4},
                            f"{key}:B4": {"n_compared": 3}, "junk": {"n_compared": 9}}}
        d = g.shadow_wave_delta(before, after)
        assert d == {"compared": 7, "unequal": 1, "unmodified": 1, "shadow_errors": 0,
                     "rows_compared": [3, 4]}
        assert g.shadow_wave_delta({"rpc": "absent"}, after)["compared"] == 17
        assert g.shadow_wave_delta(None, None) == {"compared": 0, "unequal": 0, "unmodified": 0,
                                                  "shadow_errors": 0, "rows_compared": []}

    def test_b_gt_1_comparisons_are_evidence_only(self):
        """The re-scoped cross-server comparison: B>1 entries never produce a
        PASS or FAIL exit-code effect, whatever they say; B=1 stays hard."""
        a = [_sig("ab", [-1.0, -2.0])]
        b = [_sig("ab", [-1.0, -2.0000001])]
        ev = g.evidence_verdict([a, a], [b, b])
        assert ev["verdict"] == g.EVIDENCE and ev["evidence"] is True
        assert ev["equal"] is False and ev["underlying"] == g.FAIL
        ev2 = g.evidence_verdict([a, b], [a, a])
        assert ev2["underlying"] == g.UNSTABLE_REF and ev2["reference_stable"] is False
        case = _ok_case()
        case["verdicts"] = {"pure_B1": {"verdict": g.PASS}, "pure_B8": ev, "pure_B32": ev2,
                            "mixed_B32": g.evidence_verdict([a, a], [a, a])}
        case["checks"] = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 0
        case["verdicts"]["pure_B1"] = {"verdict": g.FAIL, "detail": "request 0: logprobs"}
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 2
        text = g.render_report({"model": dict(key="llama8b", hf="x", rev="r" * 40, cfg="c", bank="b",
                                              branch="plain"),
                                "geometry": {"max_model_len": 4096, "max_num_seqs": 64},
                                "batches": [1, 8, 32], "timing_batches": [8], "cases": [case],
                                "inventory": {}, "uncovered_cases": [], "complete": True,
                                "batch_invariant": 0, "exit_code": 2})
        assert "differs(FAIL)" in text and "differs(UNSTABLE_REFERENCE)" in text and "| equal |" in text

    def test_compare_case_b1_hard_rest_evidence(self):
        ctx = type("C", (), {"batches": (1, 4)})()
        roles = g.mixed_roles(4)
        oracle = [_sig("ab", [-0.1 * (i + 1), -0.2]) for i in range(4)]
        o1 = _sig("ab", [-0.9, -0.9])
        steered1 = _sig("ab", [-0.4, -0.9])
        rec = {"batches": {"pure_B1": [oracle[:1]] * 2, "pure_B4": [oracle] * 2, "mixed_B4": [oracle] * 2},
               "mixed_roles": roles,
               "controls_B1": {"passthrough": o1, "scale0": o1, "steered": steered1}}
        pt = {"batches": {"passthrough_B4": [oracle] * 2, "passthrough_B1": [[o1]] * 2}}
        out = g.compare_case(ctx, rec, rec, pt)
        assert out["verdicts"]["pure_B1"]["verdict"] == g.PASS
        assert out["verdicts"]["pure_B4"]["evidence"] and out["verdicts"]["mixed_B4"]["evidence"]
        assert out["controls_B1"][g.CANDIDATE] == {"controls_match": True, "steered_differs": True,
                                                   "cells_distinct": True}
        assert out["oracle_B1_stable"]["stable"] is True
        # a B=1 difference is a hard FAIL; a B=4 difference is evidence only
        cand = dict(rec, batches=dict(rec["batches"], pure_B1=[[steered1]] * 2, pure_B4=[oracle[1:] + oracle[:1]] * 2))
        out = g.compare_case(ctx, rec, cand, pt)
        assert out["verdicts"]["pure_B1"]["verdict"] == g.FAIL
        assert out["verdicts"]["pure_B4"]["verdict"] == g.EVIDENCE and out["verdicts"]["pure_B4"]["equal"] is False
        # ... and a ONE-bf16-ulp logprob difference at B=1 is that hard FAIL too: no reduction
        # tolerance leaks into B=1 (a one-row group is one call in either mode)
        one_ulp = _sig("ab", [-0.1, -0.2 + 2 ** -10])                    # 1 bf16 ulp at |0.2|
        cand = dict(rec, batches=dict(rec["batches"], pure_B1=[[one_ulp]] * 2))
        out = g.compare_case(ctx, rec, cand, pt)
        assert out["verdicts"]["pure_B1"]["verdict"] == g.FAIL and "logprob" in out["verdicts"]["pure_B1"]["detail"]
        case = dict(_ok_case(), **out)
        for mode in g.REDUCTION_MODES:
            checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True, reduction_mode=mode)
            assert checks["pure_B1_present"]["ok"] is True
            assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 2, mode
        # the B=1 controls: a moved scale-0 singleton / a steered singleton equal to passthrough
        facts = g.controls_B1_facts({"passthrough": o1, "scale0": steered1, "steered": steered1}, o1, ["steered"])
        assert facts["controls_match"] is False and "scale0" in facts["controls_detail"]
        facts = g.controls_B1_facts({"passthrough": o1, "scale0": o1, "steered": o1}, o1, ["steered"])
        assert facts["steered_differs"] is False and facts["cells_distinct"] is False
        facts = g.controls_B1_facts({"passthrough": o1, "scale0": o1, "cA": steered1, "cB": steered1}, o1, ["cA", "cB"])
        assert facts["steered_differs"] is True and facts["cells_distinct"] is False
        assert "cA == cB" in facts["cells_detail"]
        assert g.controls_B1_facts({}, None, ["steered"])["controls_match"] is None

    def test_compare_case_records_b_gt_1_row_facts_as_evidence(self):
        """The B=4 control rows / bites vs the oracle are recorded under
        evidence_B_gt_1 — no check reads them (the composition confound)."""
        ctx = type("C", (), {"batches": (1, 4)})()
        roles = g.mixed_roles(4)          # passthrough, scale0, steered, steered
        oracle = [_sig("ab", [-0.1 * (i + 1), -0.2]) for i in range(4)]
        steered = [dict(s) for s in oracle]
        steered[2] = _sig("ab", [-0.9, -0.2])
        steered[3] = _sig("ab", [-0.8, -0.2])
        rec = {"batches": {"pure_B1": [oracle[:1]] * 2, "pure_B4": [oracle] * 2,
                           "mixed_B4": [steered] * 2}, "mixed_roles": roles, "controls_B1": {}}
        pt = {"batches": {"passthrough_B4": [oracle] * 2}}
        out = g.compare_case(ctx, rec, rec, pt)
        ev = out["evidence_B_gt_1"][g.CANDIDATE]
        assert ev["control_rows_match"] is True and ev["steering_bites"] is True
        moved = [dict(s) for s in steered]
        moved[1] = _sig("ab", [-0.3, -0.7])                                    # the scale-0 row
        out = g.compare_case(ctx, rec, dict(rec, batches=dict(rec["batches"], mixed_B4=[moved] * 2)), pt)
        assert out["evidence_B_gt_1"][g.CANDIDATE]["control_rows_match"] is False
        assert "request 1" in out["evidence_B_gt_1"][g.CANDIDATE]["control_rows_detail"]
        # no B=1 oracle in this fixture: the B=1 control facts say so (None), never pass
        assert out["controls_B1"][g.CANDIDATE]["controls_match"] is None
        case = dict(_ok_case(), **out)
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert not checks["controls_B1_candidate"]["ok"]

    def test_exit_codes(self):
        ok = _ok_case()
        ok["checks"] = g.case_checks(ok, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert g.overall_exit_code({"cases": [ok], "complete": True}) == 0
        assert g.overall_exit_code({"cases": [ok], "complete": True, "uncovered_cases": ["x"]}) == 3
        assert g.overall_exit_code({"cases": [dict(ok, error="boom")], "complete": True}) == 3
        assert g.overall_exit_code({"cases": [ok], "complete": True, "tree_error": "same commit"}) == 2
        bad = dict(ok, verdicts={"pure_B1": {"verdict": g.UNSTABLE_REF}})
        assert g.overall_exit_code({"cases": [bad], "complete": True}) == 3
        bad = dict(ok, verdicts={"pure_B1": {"verdict": g.UNSTABLE_VEC}})
        assert g.overall_exit_code({"cases": [bad], "complete": True}) == 2

    def test_complete_field_is_authoritative(self):
        """All verdicts PASS, every check ok, no uncovered list — still exit 3
        whenever ``complete`` is false or absent, however it was derived."""
        ok = _ok_case()
        ok["checks"] = g.case_checks(ok, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert g.overall_exit_code({"cases": [ok], "complete": False}) == 3
        assert g.overall_exit_code({"cases": [ok]}) == 3
        assert g.overall_exit_code({"cases": [ok], "complete": True}) == 0


# --------------------------------------------------------------------------
# THE REDUCTION CONTRACT — stacked reductions are the served default
# --------------------------------------------------------------------------


def _measured_case(key=None, **measured):
    """An _ok_case whose shadow record carries a measured-class deviation on
    the psr (GATED reduction-op) key — or on ``key`` (``LINEAR_KEY`` for the
    edit-ulp rule's class; the case id follows the key): 3 of 2e6 elements,
    one bf16 ulp, and on a gated key |delta gate|/sum|hw| 0.8 x 2^-24 by default."""
    m = dict(MEASURED, **measured)
    case = _ok_case()
    if key is not None:
        case["case_id"] = key
        return dict(case, shadow=_shadow_snapshot(unequal=3, key=key, measured=m))
    return dict(case, shadow=_shadow_snapshot(unequal=3, measured=m))


def _per_row_mode_case():
    """The same case as run under ``--reduction-mode per_row``: the worker
    attests the per-row apply paths and mode; the record is exact."""
    case = _ok_case()
    case["apply_path"] = {g.REFERENCE: {"rpc": "absent"}, g.CANDIDATE: g.APPLY_PATH_VECTORIZED_PER_ROW,
                          g.SHADOW: g.APPLY_PATH_SHADOW_PER_ROW}
    for side in (g.CANDIDATE, g.SHADOW):
        case["steering_evidence"][side] = _ev(grouped_mod=3200, grouped_app=3200, mode="per_row")
    return case


class TestReductionContract:
    """Stacked reductions (one [B, K] call per group) are the default served
    arithmetic since 2026-09-04, accepting bf16-ulp-level differences. The
    gate's contract for the in-worker shadow compare is therefore a BOUNDED
    tolerance for reduction ops under the stacked mode and exactness
    everywhere else, with every defect / missing-evidence shape still a
    FAIL — each rule proven here to fail or pass the gate."""

    def test_constants_mirror_the_core(self):
        # the module keeps its runtime imports lazy, so these are literals pinned to la / lw
        assert g.REDUCTION_MODE_STACKED == la.REDUCTION_MODE_STACKED == "stacked"
        assert g.REDUCTION_MODE_PER_ROW == la.REDUCTION_MODE_PER_ROW == "per_row"
        assert g.REDUCTION_MODES == la.REDUCTION_MODES and g.DEFAULT_REDUCTION_MODE == "stacked"
        assert g.STACKED_ENV == la.STACKED_REDUCTIONS_ENV
        assert g.LENS_REDUCTION_MODE_KEY == lw.LENS_REDUCTION_MODE_KEY == "lens_reduction_mode"
        assert la.reduction_mode() == la.parse_reduction_mode(None) == "stacked"   # the core's default
        assert g.APPLY_PATH_VECTORIZED == lw.APPLY_PATH_VECTORIZED == "vectorized"
        assert g.APPLY_PATH_PER_ROW == lw.APPLY_PATH_PER_ROW == "per_row"
        assert g.APPLY_PATH_VECTORIZED_STACKED == lw.APPLY_PATH_VECTORIZED_STACKED == "vectorized+stacked"
        assert g.APPLY_PATH_VECTORIZED_PER_ROW == lw.APPLY_PATH_VECTORIZED_PER_ROW == "vectorized+per_row"
        assert g.APPLY_PATH_SHADOW == lw.APPLY_PATH_SHADOW == "vectorized+stacked+shadow(serves per_row)"
        assert g.APPLY_PATH_SHADOW_PER_ROW == lw.APPLY_PATH_SHADOW_PER_ROW
        for mode in g.REDUCTION_MODES:
            assert g.expected_apply_path(mode) == lw.apply_path_string(mode)
            assert g.expected_apply_path(mode, shadow=True) == lw.apply_path_string(mode, shadow=True)
        with pytest.raises(g.GateError, match="unknown reduction mode"):
            g.expected_apply_path("sideways")
        assert g.REDUCTION_FRAC_TOL == 1e-4 and g.REDUCTION_ULPS_EDIT_TOL == 4.0
        assert g.REDUCTION_GROUP_FRAC_TOL == 2e-3 and g.REDUCTION_ROW_FRAC_TOL == 2e-2
        # the gated bound (2026-09-06): C = 16 on the unit 2^-24
        assert g.REDUCTION_GATE_C == 16.0 and g.REDUCTION_GATE_UNIT == 2.0 ** -24
        assert g.REDUCTION_GATE_REL_SUM_TOL == 16.0 * 2.0 ** -24
        assert g.REDUCTION_GATE_CONSISTENCY_ULPS_TOL == 4.0                  # the gate-edit identity
        assert g.REDUCTION_GATE_EXACT_ROWS_UNEQUAL_TOL == 0                  # exact where the gates agree (P1-b)
        assert g.REDUCTION_GATE_WRITE_RECON_MISMATCH_TOL == 0                # the bit-exact reconstruction (round 4 P1)
        # the class vocabulary mirrors the adapters' ONE definition
        assert g.GATED_REDUCTION_OPS == la.GATED_REDUCTION_OPS and g.LINEAR_REDUCTION_OPS == la.LINEAR_REDUCTION_OPS
        assert (g.OP_CLASS_REDUCTION_GATED, g.OP_CLASS_REDUCTION_LINEAR, g.OP_CLASS_ELEMENTWISE) == (
            la.OP_CLASS_REDUCTION_GATED, la.OP_CLASS_REDUCTION_LINEAR, la.OP_CLASS_ELEMENTWISE)
        assert not hasattr(g, "OP_CLASS_REDUCTION")                 # the undivided class is gone
        assert "1e-4" in g.__doc__ and "4 bf16 ulps" in g.__doc__ and "3.9e-5" in g.__doc__
        for phrase in ("bf16-kernel-variance equivalence", "<= 0.0001", "within 4 bf16 ulps of the per-row edit",
                       "beyond one bf16 ulp of the per-row output", "in any single group <= 0.002",
                       "in any single row <= 0.02", "10x the measured maxima, not mechanism",
                       "LINEAR reduction ops (affine_subspace, lora_oproj)",
                       "GATED reduction ops (linear_gated_vector, lowrank_linear_gated, lowrank_relu_gated, "
                       "relu_gated_vector, relu_gated_vector_pair)",
                       "|g_stacked - g_per_row| <= 16 * 2^-24 * sum_i |h_i w_i|",
                       "a ReLU flip only when the side that fired is within that same bound",
                       "the gates OBSERVED from the two applies on every unequal group (never recomputed)",
                       "the stacked-minus-per-row edit equal to eff * (g_stacked - g_per_row) * v within 4 bf16 ulps "
                       "of the output + the edit",
                       "for the gated-VECTOR pair, the two writes EXACT on every row whose observed gates agree or which "
                       "is masked off",
                       "each gated-vector side's output the BIT-EXACT reconstruction base + eff * g * v from that side's "
                       "observed gate (any other element is a wrong write, whatever the gate delta)",
                       "the low-rank pair is not reconstructed and never held exact where it writes",
                       "its zero-delta rows are BOUNDED at the linear 4-edit-ulp unit, a one-output-ulp flip reading 0 "
                       "excess",
                       "the edit-ulp unit is recorded there but is not a rule",
                       "lacks observed gate stats", "gate_error",
                       "EVERY op under the per-row mode: exact", "never for a defect or for missing evidence"):
            assert phrase in g.REDUCTION_CONTRACT, phrase

    def test_shadow_key_op_class_is_the_adapters_classification(self):
        tail = "gemv:response:skipbos0:both:B32"
        for op in la.GATED_REDUCTION_OPS:
            assert g.shadow_key_op_class(f"kind:{op}:{tail}") == g.OP_CLASS_REDUCTION_GATED, op
        for op in la.LINEAR_REDUCTION_OPS:
            assert g.shadow_key_op_class(f"kind:{op}:{tail}") == g.OP_CLASS_REDUCTION_LINEAR, op
        for op in la.ELEMENTWISE_OPS:
            assert g.shadow_key_op_class(f"kind:{op}:{tail}") == g.OP_CLASS_ELEMENTWISE, op
        assert {g.shadow_key_op_class(f"kind:{op}:{tail}") for op in la.OPS} == set(la.OP_CLASSES)
        # several ops on one key: GATED when ANY op is gated, LINEAR when any reduces and none is gated
        assert g.shadow_key_op_class(f"k:add_vector+relu_gated_vector:{tail}") == g.OP_CLASS_REDUCTION_GATED
        assert g.shadow_key_op_class(f"k:add_vector+lora_oproj:{tail}") == g.OP_CLASS_REDUCTION_LINEAR
        assert g.shadow_key_op_class(f"k:affine_subspace+lowrank_linear_gated:{tail}") == g.OP_CLASS_REDUCTION_GATED
        assert g.shadow_key_op_class(f"k:add_vector+prefill_last_matmul:{tail}") == g.OP_CLASS_ELEMENTWISE
        assert set(g.OP_CLASSES_REDUCTION) == {g.OP_CLASS_REDUCTION_GATED, g.OP_CLASS_REDUCTION_LINEAR}
        for bad in ("psr:mystery_op:" + tail, "garbage", "k:-:" + tail, "k::" + tail,
                    "psr:relu_gated_vector:gemv:response:skipbos0:both"):      # no B<rows> tail
            with pytest.raises(g.GateError):
                g.shadow_key_op_class(bad)
        # the adapters' own classifier is what decides, and it refuses an unclassified op
        with pytest.raises(la.AdapterError):
            la.op_reduces_in_group("mystery_op")
        with pytest.raises(la.AdapterError):
            la.reduction_op_class("mystery_op")

    def test_measured_class_is_inside_the_contract(self):
        U = g.REDUCTION_GATE_UNIT
        tol = g.reduction_tolerance(_measured_case()["shadow"], reduction_mode="stacked")
        assert tol["ok"] is True
        assert tol["n_unequal_elems"] == 3 and tol["n_elems_compared"] == 2_000_000 + 32 * 1 * HIDDEN_ELEMS + 32 * 8 * HIDDEN_ELEMS
        assert tol["frac_unequal_elems"] == pytest.approx(3 / tol["n_elems_compared"])
        assert tol["max_ulps_edit"] == 1.0 and tol["n_reduction_unequal"] == 3 and tol["n_unequal"] == 3
        assert tol["max_ulps_edit_gated"] == 1.0 and tol["max_ulps_edit_linear"] is None
        assert tol["max_gate_diff_rel_sum"] == 0.8 * U and tol["max_gate_flip_rel_sum"] is None
        assert tol["reduction_keys_unequal"] == [f"{PSR_KEY}:B32"] == tol["gated_keys_unequal"]
        assert tol["linear_keys_unequal"] == [] and tol["keys_lacking_gate_stats"] == []
        assert tol["elementwise_keys_unequal"] == [] and tol["n_flag_disagreements"] == 0
        assert [r["class"] for r in tol["per_key"]] == [g.OP_CLASS_REDUCTION_GATED] * 3
        assert tol["per_key"][-1]["gate_max_diff_rel_sum"] == 0.8 * U and tol["per_key"][-1]["gate_n_groups_with_stats"] == 3
        # the detail names the numbers AND the bounds, in the same units
        d = tol["detail"]
        assert "within the reduction contract (stacked mode)" in d
        assert "3 unequal reduction-op group(s) of 96" in d and "<= 0.0001" in d and "'layer': 7" in d
        assert ("gated ops: |delta gate|/sum|hw| = 0.8 x 2^-24 <= 16 x 2^-24, no ReLU flips, gate-edit identity "
                "within 0.5 ulps (<= 4; gates observed on all 3 unequal group(s)), writes exact on the 93 row(s) whose "
                "gates agree, both outputs the bit-exact reconstruction from the observed gates on all 3 vector-gated "
                "group(s) (edit-ulps 1 recorded, not a rule)") in d
        assert tol["max_gate_edit_consistency_ulps"] == 0.5
        assert tol["n_gate_groups_observed"] == 3 and tol["gate_exact_rows_n"] == 93 and tol["gate_exact_rows_unequal_elems"] == 0
        assert tol["max_gate_lowrank_zero_delta_ulps_edit"] is None
        assert tol["n_gate_groups_recon"] == 3 and tol["gate_write_recon_mismatch_elems"] == 0
        assert tol["vector_gated_keys_unequal"] == [f"{PSR_KEY}:B32"]
        assert "linear ops" not in d
        assert "max in one group 2.29e-05 <= 0.002, in one row 4.88e-04 <= 0.02" in d
        assert tol["max_frac_row"] == 2 / HIDDEN_ELEMS and tol["max_frac_group"] == 3 / (32 * HIDDEN_ELEMS)
        # the LINEAR twin of the measured class: the (unchanged) edit-ulp rule, named
        lin = g.reduction_tolerance(_measured_case(key=LINEAR_KEY)["shadow"], reduction_mode="stacked")
        assert lin["ok"] is True and lin["linear_keys_unequal"] == [f"{LINEAR_KEY}:B32"] and lin["gated_keys_unequal"] == []
        assert "linear ops: largest 1 bf16 ulps of the per-row edit beyond one output ulp (<= 4)" in lin["detail"]
        assert "gated ops" not in lin["detail"]
        assert lin["max_ulps_edit_linear"] == 1.0 and lin["max_ulps_edit_gated"] is None and lin["max_gate_diff_rel_sum"] is None
        assert [r["class"] for r in lin["per_key"]] == [g.OP_CLASS_REDUCTION_LINEAR] * 3
        # the boundary is inclusive on every bound
        n = tol["n_elems_compared"]
        at = g.reduction_tolerance(_measured_case(n_unequal_elems=int(n * 1e-4), gate_max_diff_rel_sum=16 * U)["shadow"],
                                   reduction_mode="stacked")
        assert at["ok"] is True and at["frac_unequal_elems"] <= 1e-4
        at_lin = g.reduction_tolerance(_measured_case(key=LINEAR_KEY, n_unequal_elems=int(n * 1e-4), max_ulps_edit=4.0)["shadow"],
                                       reduction_mode="stacked")
        assert at_lin["ok"] is True

    def test_fraction_over_the_bound_fails_naming_it(self):
        n = g.reduction_tolerance(_measured_case()["shadow"], reduction_mode="stacked")["n_elems_compared"]
        bad = _measured_case(n_unequal_elems=round(n * 2e-4))
        tol = g.reduction_tolerance(bad["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False and tol["frac_unequal_elems"] == pytest.approx(2e-4, rel=1e-2)
        assert tol["frac_unequal_elems"] > g.REDUCTION_FRAC_TOL
        assert "fraction of unequal reduction-op elements 2.00e-04 > 0.0001" in tol["detail"]
        code, checks = _exit_with(bad)
        assert code == 2 and checks["shadow_compare"]["ok"] is False
        assert "> 0.0001" in checks["shadow_compare"]["detail"]

    def test_linear_ulps_over_the_bound_fail_and_gated_ulps_are_informational(self):
        """The edit-ulp bound is the LINEAR class's rule, unchanged; on a GATED
        key the unit is recorded and reported but judges nothing (2026-09-06)."""
        bad = _measured_case(key=LINEAR_KEY, max_ulps_edit=4.5)
        tol = g.reduction_tolerance(bad["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False and tol["max_ulps_edit"] == 4.5 == tol["max_ulps_edit_linear"]
        assert tol["linear_keys_unequal"] == [f"{LINEAR_KEY}:B32"] and tol["gated_keys_unequal"] == []
        assert ("largest unequal element of a LINEAR reduction op 4.5 bf16 ulps of the per-row edit beyond one "
                "output ulp > 4") in tol["detail"]
        code, checks = _exit_with(bad)
        assert code == 2 and "> 4" in checks["shadow_compare"]["detail"]
        assert g.reduction_tolerance(_measured_case(key=LINEAR_KEY, max_ulps_edit=1.0)["shadow"],
                                     reduction_mode="stacked")["ok"] is True
        # a sign flip / row shift / transposed operand lands far outside: hundreds of ulps, frac ~ 1
        planted = _measured_case(key=LINEAR_KEY, n_unequal_elems=32 * 32 * HIDDEN_ELEMS, max_ulps_edit=512.0)
        tol = g.reduction_tolerance(planted["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False and "> 0.0001" in tol["detail"] and "> 4" in tol["detail"]
        # GATED: nemotron3nano4b's 23 edit-ulps at a gate deviation of 0.6 x 2^-24 PASSES — the
        # unit is recorded, reported, and not a rule there
        gated = _measured_case(max_ulps_edit=23.0, gate_max_diff_rel_sum=0.6 * g.REDUCTION_GATE_UNIT)
        tol = g.reduction_tolerance(gated["shadow"], reduction_mode="stacked")
        assert tol["ok"] is True and tol["max_ulps_edit_gated"] == 23.0 == tol["max_ulps_edit"]
        assert tol["max_ulps_edit_linear"] is None
        assert "edit-ulps 23 recorded, not a rule" in tol["detail"] and "> 4" not in tol["detail"]
        assert "|delta gate|/sum|hw| = 0.6 x 2^-24 <= 16 x 2^-24" in tol["detail"]
        code, checks = _exit_with(gated)
        assert code == 0 and checks["shadow_compare"]["ok"] is True

    def test_gated_gate_bound_fails_naming_it_in_the_contracts_unit(self):
        """THE GATED RULE (2026-09-06): |g_stacked - g_per_row| <= 16 x 2^-24 x
        sum|h_i w_i| per gate element; the FAIL names the bound and the value in
        the same unit."""
        U = g.REDUCTION_GATE_UNIT
        bad = _measured_case(gate_max_diff_rel_sum=20 * U)
        tol = g.reduction_tolerance(bad["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False and tol["max_gate_diff_rel_sum"] == 20 * U
        assert "largest |delta gate|/sum|hw| of a GATED reduction op 20 x 2^-24 > 16 x 2^-24" in tol["detail"]
        assert "edit-ulps 1, informational" in tol["detail"]
        assert "> 4" not in tol["detail"] and "> 0.0001" not in tol["detail"]      # only the gate rule fired
        code, checks = _exit_with(bad)
        assert code == 2 and "20 x 2^-24 > 16 x 2^-24" in checks["shadow_compare"]["detail"]
        # inclusive at the bound; just over fails
        assert g.reduction_tolerance(_measured_case(gate_max_diff_rel_sum=16 * U)["shadow"], reduction_mode="stacked")["ok"] is True
        assert g.reduction_tolerance(_measured_case(gate_max_diff_rel_sum=16.5 * U)["shadow"], reduction_mode="stacked")["ok"] is False
        assert tol["gate_c"] == 16.0 and tol["gate_unit"] == 2 ** -24 and tol["gate_rel_sum_tol"] == 16 * 2 ** -24
        # a misplaced or wrong gate reads O(1) of sum|hw| — ~1e6 in these units
        wrong = g.reduction_tolerance(_measured_case(gate_max_diff_rel_sum=0.5)["shadow"], reduction_mode="stacked")
        assert wrong["ok"] is False and "8.39e+06 x 2^-24 > 16 x 2^-24" in wrong["detail"]
        # the concentration and pooled-fraction bounds still fire on a gated key
        conc = g.reduction_tolerance(_measured_case(max_frac_row=0.5)["shadow"], reduction_mode="stacked")
        assert conc["ok"] is False and "in one ROW 5.00e-01 > 0.02" in conc["detail"]
        n = tol["n_elems_compared"]
        frac = g.reduction_tolerance(_measured_case(n_unequal_elems=round(n * 2e-4))["shadow"], reduction_mode="stacked")
        assert frac["ok"] is False and "> 0.0001" in frac["detail"]
        # per-row mode: exact everywhere, gate stats or not
        assert g.reduction_tolerance(_measured_case()["shadow"], reduction_mode="per_row")["ok"] is False

    def test_pooled_frac_bound_is_a_keyword_defaulting_to_the_contract_and_never_vacuous(self):
        """``frac_tol``: the production constant by default — the gate's own
        callers never pass it — and the scale a small fixture is honestly
        judged at (test_lens_stacked_reductions._case_frac_tol: a handful of
        groups' pooled fraction is a host-dependent flip count, not a rate).
        The verdict records the bound it was judged at; every field but the
        pooled clause's own is identical across scales; a bound above
        REDUCTION_GROUP_FRAC_TOL (vacuous: the pooled fraction never exceeds
        the largest group's) or non-positive is refused."""
        n = g.reduction_tolerance(_measured_case()["shadow"], reduction_mode="stacked")["n_elems_compared"]
        over = _measured_case(n_unequal_elems=round(n * 2e-4))["shadow"]
        default = g.reduction_tolerance(over, reduction_mode="stacked")
        assert default["ok"] is False and default["frac_tol"] == g.REDUCTION_FRAC_TOL == 1e-4
        assert default["detail"].startswith("fraction of unequal reduction-op elements") and "> 0.0001" in default["detail"]
        assert default == g.reduction_tolerance(over, reduction_mode="stacked", frac_tol=g.REDUCTION_FRAC_TOL)
        scaled = g.reduction_tolerance(over, reduction_mode="stacked", frac_tol=g.REDUCTION_GROUP_FRAC_TOL)
        assert scaled["ok"] is True and scaled["frac_tol"] == g.REDUCTION_GROUP_FRAC_TOL and "<= 0.002" in scaled["detail"]
        assert {k for k in default if default[k] != scaled[k]} == {"ok", "detail", "frac_tol"}
        assert g.reduction_tolerance(over, reduction_mode="stacked", frac_tol=1.5e-4)["ok"] is False   # 2e-4 > 1.5e-4
        for bad in (g.REDUCTION_GROUP_FRAC_TOL * 1.01, 1.0, 0.0, -1e-4):
            with pytest.raises(g.GateError):
                g.reduction_tolerance(over, reduction_mode="stacked", frac_tol=bad)

    def test_gated_relu_flip_allowed_only_within_the_bound(self):
        U = g.REDUCTION_GATE_UNIT
        ok = _measured_case(gate_n_relu_flips=2, gate_max_flip_rel_sum=3 * U)
        tol = g.reduction_tolerance(ok["shadow"], reduction_mode="stacked")
        assert tol["ok"] is True and tol["n_gate_relu_flips"] == 2 and tol["max_gate_flip_rel_sum"] == 3 * U
        assert "2 ReLU flip(s) with fired side 3 x 2^-24 <= 16 x 2^-24" in tol["detail"]
        over = _measured_case(gate_n_relu_flips=1, gate_max_flip_rel_sum=40 * U)
        tol = g.reduction_tolerance(over["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False
        assert "a ReLU flip of a GATED reduction op whose fired side is 40 x 2^-24 of sum|hw| > 16 x 2^-24" in tol["detail"]
        assert "> 16 x 2^-24 (the stacked gate" not in tol["detail"]              # only the flip rule fired
        code, checks = _exit_with(over)
        assert code == 2
        # a flip whose fired side was not recorded cannot be judged: missing evidence
        blind = _measured_case(gate_n_relu_flips=1, gate_max_flip_rel_sum=None)
        tol = g.reduction_tolerance(blind["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False and "lacks gate stats" in tol["detail"]
        # no flips: the ratio is None and the detail says so
        none = g.reduction_tolerance(_measured_case()["shadow"], reduction_mode="stacked")
        assert none["max_gate_flip_rel_sum"] is None and "no ReLU flips" in none["detail"]
        # the boundary: exactly at 16 x 2^-24 passes, the next representable value fails
        at = _measured_case(gate_n_relu_flips=1, gate_max_flip_rel_sum=16 * U)
        assert g.reduction_tolerance(at["shadow"], reduction_mode="stacked")["ok"] is True
        just_over = _measured_case(gate_n_relu_flips=1, gate_max_flip_rel_sum=math.nextafter(16 * U, math.inf))
        tol = g.reduction_tolerance(just_over["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False and "a ReLU flip of a GATED reduction op whose fired side is 16 x 2^-24 of sum|hw| > 16 x 2^-24" in tol["detail"]
        # (the rendered 3-significant-figure value reads 16; the comparison is on the raw float)
        assert tol["max_gate_flip_rel_sum"] > g.REDUCTION_GATE_REL_SUM_TOL

    def test_gated_gate_edit_identity_bound(self):
        """Review P1 (2026-09-06): the observed gate delta must EXPLAIN the
        observed output delta — applied - ref = eff (g_stacked - g_per_row) v
        up to bf16 rounding, recorded as gate_edit_consistency_max_ulps and
        bounded at 4; a record without it (a recomputation-only worker) is
        missing evidence."""
        ok = _measured_case(gate_edit_consistency_max_ulps=4.0)
        tol = g.reduction_tolerance(ok["shadow"], reduction_mode="stacked")
        assert tol["ok"] is True and tol["max_gate_edit_consistency_ulps"] == 4.0
        assert "gate-edit identity within 4 ulps (<= 4; gates observed on all 3 unequal group(s))" in tol["detail"]
        bad = _measured_case(gate_edit_consistency_max_ulps=4.5)
        tol = g.reduction_tolerance(bad["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False
        assert ("the gate-edit identity of a GATED reduction op breaks by 4.5 bf16 ulps (of the output + the edit) > 4: "
                "applied - ref is not eff * (g_stacked - g_per_row) * v") in tol["detail"]
        assert "largest |delta gate|" not in tol["detail"]                  # the gate delta itself was fine
        code, checks = _exit_with(bad)
        assert code == 2 and "gate-edit identity" in checks["shadow_compare"]["detail"]
        # a wrong write reads tens to hundreds of ulps; the misplaced-gate defect reads it too
        assert g.reduction_tolerance(_measured_case(gate_edit_consistency_max_ulps=75.0)["shadow"],
                                     reduction_mode="stacked")["ok"] is False
        # missing: a recomputation-only record carries no identity residual -> missing evidence
        tol = g.reduction_tolerance(_measured_case(gate_edit_consistency_max_ulps=None)["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False and tol["keys_lacking_gate_stats"] == [f"{PSR_KEY}:B32"]
        assert "gates observed on every unequal group, not a recomputation" in tol["detail"]
        assert "gates observed on all" not in tol["detail"]                # the PASS claim is never made
        # a linear key carries none and needs none
        lin = g.reduction_tolerance(_measured_case(key=LINEAR_KEY, gate_edit_consistency_max_ulps=None)["shadow"],
                                    reduction_mode="stacked")
        assert lin["ok"] is True and lin["max_gate_edit_consistency_ulps"] is None

    def test_gated_provenance_is_enforced_per_group(self):
        """Review P1-a: a gated key passes only when EVERY unequal group's gate
        stats were observed (gate_n_groups_observed == n_unequal) — a
        recomputed, unsourced or mixed record with perfect numbers is missing
        evidence, and the detail never claims 'gates observed' for it."""
        mixed = g.reduction_tolerance(_measured_case(gate_n_groups_observed=2)["shadow"], reduction_mode="stacked")
        assert mixed["ok"] is False and mixed["keys_lacking_gate_stats"] == [f"{PSR_KEY}:B32"]
        assert "gate_n_groups_observed == n_unequal" in mixed["detail"] and "gates observed on all" not in mixed["detail"]
        recomputed = g.reduction_tolerance(_measured_case(gate_n_groups_observed=0)["shadow"], reduction_mode="stacked")
        assert recomputed["ok"] is False and "gates observed on all" not in recomputed["detail"]
        assert "writes exact on the" not in recomputed["detail"]            # no PASS clause on missing evidence
        absent = g.reduction_tolerance(_measured_case(gate_n_groups_observed=None)["shadow"], reduction_mode="stacked")
        assert absent["ok"] is False and absent["keys_lacking_gate_stats"] == [f"{PSR_KEY}:B32"]
        for m in ({"gate_n_groups_observed": 2}, {"gate_n_groups_observed": 0}, {"gate_n_groups_observed": None}):
            code, checks = _exit_with(_measured_case(**m))
            assert code == 2 and checks["shadow_compare"]["ok"] is False, m
        ok = g.reduction_tolerance(_measured_case()["shadow"], reduction_mode="stacked")
        assert ok["ok"] is True and ok["n_gate_groups_observed"] == 3

    def test_gated_writes_must_be_exact_where_the_gates_agree(self):
        """Review P1-b: any unequal element in a row whose observed vector
        gates are bit-identical (or masked off) is a wrong write — FAIL at
        one element; a low-rank zero-delta row takes the linear unit."""
        one = g.reduction_tolerance(_measured_case(gate_exact_rows_unequal_elems=1)["shadow"], reduction_mode="stacked")
        assert one["ok"] is False and one["gate_exact_rows_unequal_elems"] == 1
        assert ("1 unequal element(s) of a GATED reduction op in rows whose observed gates are bit-identical or masked "
                "off (93 such row(s)): after an equal gate a gated-vector op is elementwise, so the two writes must be "
                "exact — a wrong write, never kernel variance") in one["detail"]
        assert "gate-edit identity of a GATED reduction op breaks" not in one["detail"]   # exactness alone fired
        code, checks = _exit_with(_measured_case(gate_exact_rows_unequal_elems=1))
        assert code == 2 and "bit-identical or masked off" in checks["shadow_compare"]["detail"]
        # missing: a record without the count is missing evidence
        none = g.reduction_tolerance(_measured_case(gate_exact_rows_unequal_elems=None)["shadow"], reduction_mode="stacked")
        assert none["ok"] is False and none["keys_lacking_gate_stats"] == [f"{PSR_KEY}:B32"]
        # the low-rank zero-delta rows: the LINEAR edit-ulp unit, inclusive at 4
        lz_ok = g.reduction_tolerance(_measured_case(gate_lowrank_zero_delta_max_ulps_edit=4.0)["shadow"], reduction_mode="stacked")
        assert lz_ok["ok"] is True and "low-rank zero-delta rows within 4 edit ulps (<= 4)" in lz_ok["detail"]
        lz_bad = g.reduction_tolerance(_measured_case(gate_lowrank_zero_delta_max_ulps_edit=4.5)["shadow"], reduction_mode="stacked")
        assert lz_bad["ok"] is False and lz_bad["max_gate_lowrank_zero_delta_ulps_edit"] == 4.5
        assert ("the low-rank write on rows whose observed latents are bit-identical departs by 4.5 bf16 ulps of the "
                "per-row edit beyond one output ulp > 4") in lz_bad["detail"]
        assert g.reduction_tolerance(_measured_case()["shadow"], reduction_mode="stacked")["max_gate_lowrank_zero_delta_ulps_edit"] is None

    def test_gated_vector_outputs_must_be_the_reconstruction_from_the_observed_gate(self):
        """Round 4 P1: any element of a gated-vector group that is not the
        bit-exact reconstruction base + eff g v from that side's OBSERVED gate
        is a wrong write, whatever the gate delta (a legitimate nonzero delta
        exempted its row from exactness and the 4-ulp identity envelope hid a
        one-ulp write where the direction is zero); a vector-gated key whose
        unequal groups were not all rebuilt is missing evidence; a low-rank
        key needs no reconstruction."""
        one = g.reduction_tolerance(_measured_case(gate_write_recon_mismatch_elems=1)["shadow"], reduction_mode="stacked")
        assert one["ok"] is False and one["gate_write_recon_mismatch_elems"] == 1 and one["n_gate_groups_recon"] == 3
        assert ("1 element(s) of a GATED reduction op where the grouped or the per-row output is not the bit-exact "
                "reconstruction base + eff * g * v from that side's OBSERVED gate (write consistency, 3 group(s) rebuilt): "
                "a wrong write, never kernel variance") in one["detail"]
        assert "a nonzero gate delta cannot write where the direction is zero" in one["detail"]
        for absent in ("gate-edit identity of a GATED reduction op breaks", "bit-identical or masked off", "largest |delta gate|"):
            assert absent not in one["detail"], absent                     # write consistency alone
        code, checks = _exit_with(_measured_case(gate_write_recon_mismatch_elems=1))
        assert code == 2 and "write consistency" in checks["shadow_compare"]["detail"]
        # not every unequal group rebuilt, or the count missing: missing evidence for a vector-gated key
        for m in ({"gate_n_groups_recon": 2}, {"gate_n_groups_recon": None}, {"gate_write_recon_mismatch_elems": None}):
            tol = g.reduction_tolerance(_measured_case(**m)["shadow"], reduction_mode="stacked")
            assert tol["ok"] is False and tol["keys_lacking_gate_stats"] == [f"{PSR_KEY}:B32"], m
            assert "gate_n_groups_recon == n_unequal" in tol["detail"]
        # a LOW-RANK key is never rebuilt and needs none: the identity + the zero-delta linear unit rule there
        low_key = "lowrank_gated:lowrank_relu_gated:rankK:response:skipbos0:both"
        lo = g.reduction_tolerance(_measured_case(key=low_key, gate_n_groups_recon=0, gate_write_recon_mismatch_elems=None)["shadow"],
                                   reduction_mode="stacked")
        assert lo["ok"] is True and lo["n_gate_groups_recon"] == 0 and lo["vector_gated_keys_unequal"] == []
        assert "bit-exact reconstruction" not in lo["detail"]
        assert g.shadow_key_has_vector_gate(f"{PSR_KEY}:B32") and not g.shadow_key_has_vector_gate(f"{low_key}:B32")
        assert g.shadow_key_has_vector_gate("k:add_vector+linear_gated_vector:x:all:skipbos0:both:B4")
        assert not g.shadow_key_has_vector_gate("garbage")

    def test_gated_key_lacking_gate_stats_or_with_a_gate_error_is_missing_evidence(self):
        key = f"{PSR_KEY}:B32"
        # fewer groups with stats than unequal groups
        tol = g.reduction_tolerance(_measured_case(gate_n_groups_with_stats=2)["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False and tol["keys_lacking_gate_stats"] == [key]
        assert "lacks gate stats on every unequal group" in tol["detail"]
        assert "missing evidence, never kernel variance" in tol["detail"]
        assert "> 16 x 2^-24" not in tol["detail"] and "> 0.0001" not in tol["detail"]   # no numeric verdict
        # no ratio at all
        tol = g.reduction_tolerance(_measured_case(gate_max_diff_rel_sum=None)["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False and tol["keys_lacking_gate_stats"] == [key]
        # a record from a worker before the aggregates existed (the ff35913 shape): the keys are absent
        snap = _measured_case()["shadow"]
        for k in ("gate_max_diff_rel_sum", "gate_max_flip_rel_sum", "gate_n_groups_with_stats", "gate_n_errors"):
            for e in snap["by_key"].values():
                e.pop(k, None)
            snap.pop(k, None)
        tol = g.reduction_tolerance(snap, reduction_mode="stacked")
        assert tol["ok"] is False and tol["keys_lacking_gate_stats"] == [key]
        # a gate_error on the key
        tol = g.reduction_tolerance(_measured_case(gate_n_errors=1)["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False and tol["keys_with_gate_errors"] == [key] and "gate_error" in tol["detail"]
        # a gate_error anywhere in the record (case-wide), every key clean
        snap = _measured_case()["shadow"]
        snap["gate_n_errors"] = 1
        snap["first_gate_error"] = {"key": "dropped:B2", "layer": 0, "cell": "c", "error": "RuntimeError: x"}
        tol = g.reduction_tolerance(snap, reduction_mode="stacked")
        assert tol["ok"] is False and "1 gate recomputation(s) raised" in tol["detail"]
        # a LINEAR key never needs gate stats
        tol = g.reduction_tolerance(_measured_case(key=LINEAR_KEY, gate_n_groups_with_stats=0,
                                                   gate_max_diff_rel_sum=None)["shadow"], reduction_mode="stacked")
        assert tol["ok"] is True and tol["keys_lacking_gate_stats"] == []
        for m in ({"gate_n_groups_with_stats": 2}, {"gate_max_diff_rel_sum": None}, {"gate_n_errors": 1}):
            code, checks = _exit_with(_measured_case(**m))
            assert code == 2 and checks["shadow_compare"]["ok"] is False, m

    def test_fail_detail_carries_the_evidence_where_the_unit_peaked(self):
        """2026-09-05: a record that names WHERE max_ulps_edit peaked (the
        worker's ``worst_unequal`` with its ``worst_elem`` / ``gate``
        sub-records, and the gate summary) has that evidence surfaced in the
        FAIL detail and the tolerance dict — reading only. Since 2026-09-06 the
        gated FAIL is the gate bound, and the summary's rel_sum totals are the
        rule's own inputs; the brief still reads them, in the contract's unit."""
        U = g.REDUCTION_GATE_UNIT
        bad = _measured_case(gate_max_diff_rel_sum=20 * U)["shadow"]
        plain = g.reduction_tolerance(bad, reduction_mode="stacked")
        assert plain["ok"] is False and plain["worst_unequal"] is None
        assert "worst (" not in plain["detail"]
        assert "case-wide gate:" in plain["detail"] and "max_diff_rel_sum 20 x 2^-24" in plain["detail"]
        bad["worst_unequal"] = {
            "key": f"{PSR_KEY}:B32", "layer": 14, "cell": "c", "rows": 32, "max_ulps_edit": 4.5,
            "worst_elem": {"row": 3, "col": 77, "base": 0.5, "ref": 0.75, "applied": 0.78125,
                           "edit_ref": 0.25, "edit_applied": 0.28125, "abs_diff": 0.03125, "ulps_out": 8.0,
                           "ulps_edit": 4.5, "row_n_unequal": 3, "row_n_elems": 3136, "row_edit_norm": 2.0,
                           "row_edit_ratio": 1.125, "row_edit_cos": 0.99999, "row_out_norm": 400.0},
            "gate": {"gate_ops": ["relu_gated_vector"], "gate_source": "observed", "gate_n_rows": 32,
                     "gate_n_rows_unequal": 29,
                     "gate_n_relu_flips": 1, "gate_max_abs_diff": 0.002, "gate_max_rel_diff": 0.125,
                     "gate_min_abs_at_diff": 0.016, "gate_max_abs_ref": 3.0, "gate_max_diff_rel_sum": 20 * U,
                     "gate_max_flip_rel_sum": 2 * U, "gate_edit_consistency_max_ulps": 0.7,
                     "gate_edit_consistency_worst": {"row": 3, "col": 77},
                     "gate_exact_rows_n": 3, "gate_exact_rows_unequal_elems": 0,
                     "gate_write_recon_elems": 2 * 32 * 3136, "gate_write_recon_mismatch_elems": 0,
                     "gate_rows_unequal_sample": [[0.016, 0.018, 900.0]]}}
        bad.update(gate_max_abs_diff=0.002, gate_max_rel_diff=0.125, gate_min_abs_at_diff=0.016,
                   gate_n_rows_unequal=29, gate_n_relu_flips=0)
        tol = g.reduction_tolerance(bad, reduction_mode="stacked")
        assert tol["ok"] is False and tol["max_gate_diff_rel_sum"] == 20 * U and tol["max_ulps_edit"] == 1.0
        assert tol["worst_unequal"]["layer"] == 14 and tol["gate"]["gate_max_rel_diff"] == 0.125
        d = tol["detail"]
        assert "20 x 2^-24 > 16 x 2^-24" in d and "worst (ulps_edit 4.5)" in d and "layer 14" in d
        assert "element [3, 77] base 0.5 per-row 0.75 stacked 0.78125" in d
        assert "||stacked edit||/||per-row edit|| 1.125" in d and "cosine 0.99999" in d
        assert "gate (relu_gated_vector): 29/32 rows differ" in d and "smallest gate that differed 0.016" in d
        assert "|delta gate|/sum|hw| 20 x 2^-24" in d and "ReLU flips 1 (fired side 2 x 2^-24 of sum|hw|)" in d
        assert "gate-edit identity 0.7 ulps (worst {'row': 3, 'col': 77})" in d and "gates observed" in d
        assert "exact rows 3 with 0 unequal element(s)" in d
        assert f"write reconstruction 0 mismatch(es) of {2 * 32 * 3136}" in d
        assert "write reconstruction skipped: other ops" in g.worst_brief(
            {"key": "k", "max_ulps_edit": 5.0, "gate": {"gate_ops": ["relu_gated_vector"], "gate_write_recon_skipped": "other ops"}}, {})
        assert "case-wide gate:" in d and "min_abs_at_diff 0.016" in d and "n_groups_with_stats 3" in d
        assert "edit_consistency_max_ulps 0.5" in d and "n_groups_observed 3" in d and "exact_rows_unequal_elems 0" in d
        # the gate-magnitude histogram (over EVERY compared gated group) rides in the brief when the
        # record carries it — case-wide, never a rule
        assert "gate magnitudes" not in d
        bad.update(gate_hist_n_groups=96, gate_hist_zero=40, gate_hist_near=3, gate_hist_mid=400, gate_hist_far=500,
                   gate_dose_share_near_threshold=0.002, gate_g_over_f_min_nonzero=2.5, gate_g_over_f_max=3e6)
        told = g.reduction_tolerance(bad, reduction_mode="stacked")
        assert told["ok"] is False and told["max_gate_diff_rel_sum"] == 20 * U         # nothing moved in the verdict
        d = told["detail"]
        assert ("case-wide gate magnitudes over 96 compared group(s), equal and unequal (per-row, in noise floors f): "
                "zero 40, near (0 < g <= 10 f) 3, mid (<= 1000 f) 400, far 500; near-threshold dose share 0.002; "
                "g/f nonzero min 2.5, max 3e+06") in d
        assert "hist_zero 40" not in d                                                   # not repeated as raw keys
        # an unsourced nested gate record says so in the brief
        assert "gates UNSOURCED" in g.worst_brief({"key": "k", "max_ulps_edit": 5.0, "gate": {"gate_ops": ["relu_gated_vector"]}}, {})
        # the same evidence on a PASSING record changes nothing but the dict
        ok = _measured_case()["shadow"]
        ok["worst_unequal"] = dict(bad["worst_unequal"], max_ulps_edit=1.0)
        told = g.reduction_tolerance(ok, reduction_mode="stacked")
        assert told["ok"] is True and told["worst_unequal"]["max_ulps_edit"] == 1.0
        assert "worst (" not in told["detail"]
        # a LINEAR FAIL carries the brief too (no gate sub-record on a linear group)
        lin = _measured_case(key=LINEAR_KEY, max_ulps_edit=4.5)["shadow"]
        lin["worst_unequal"] = {"key": f"{LINEAR_KEY}:B32", "layer": 3, "cell": "c", "rows": 32, "max_ulps_edit": 4.5,
                                "worst_elem": dict(bad["worst_unequal"]["worst_elem"]), "gate": None}
        told = g.reduction_tolerance(lin, reduction_mode="stacked")
        assert told["ok"] is False and "> 4" in told["detail"] and "worst (ulps_edit 4.5)" in told["detail"]
        assert "gate (" not in told["detail"]
        # the brief itself tolerates a record with an error sub-field, and an empty gate
        assert g.worst_brief({"key": "k", "max_ulps_edit": 5.0, "worst_elem": {"error": "x"}, "gate": {"gate_error": "y"}}, {}) \
            .startswith("; worst (ulps_edit 5): key k")
        assert g.worst_brief(None, {}) == ""

    def test_concentration_bounds_catch_a_row_level_defect_the_case_fraction_hides(self):
        """Review round 13: one row's edit dropped in one 32-row group is 4096
        of ~4.2e6 compared elements here (1e-3 -- and 4e-6 on a real case),
        every element moved by one output ulp (max_ulps_edit 0.0): the case
        fraction and the per-element bound cannot see it; the row and group
        concentration bounds do, in either check, naming the bound."""
        n = g.reduction_tolerance(_measured_case()["shadow"], reduction_mode="stacked")["n_elems_compared"]
        dropped_row = _measured_case(n_unequal_elems=HIDDEN_ELEMS, max_ulps_edit=0.0,
                                     max_frac_row=1.0, max_frac_group=1 / 32,
                                     n_elems_compared=int(HIDDEN_ELEMS / 5e-6))     # case frac 5e-6
        tol = g.reduction_tolerance(dropped_row["shadow"], reduction_mode="stacked")
        assert tol["frac_unequal_elems"] < g.REDUCTION_FRAC_TOL and tol["max_ulps_edit"] == 0.0
        assert tol["ok"] is False
        assert "largest unequal fraction in one ROW 1.00e+00 > 0.02" in tol["detail"]
        assert "largest unequal fraction in one GROUP 3.12e-02 > 0.002" in tol["detail"]
        assert "row's edit dropped, scaled or misplaced" in tol["detail"]
        assert "> 0.0001" not in tol["detail"] and "> 4" not in tol["detail"]      # only concentration fired
        assert "> 16 x 2^-24" not in tol["detail"]
        # the LINEAR class: the same defect, the same two bounds
        lin = _measured_case(key=LINEAR_KEY, n_unequal_elems=HIDDEN_ELEMS, max_ulps_edit=0.0,
                             max_frac_row=1.0, max_frac_group=1 / 32, n_elems_compared=int(HIDDEN_ELEMS / 5e-6))
        tol_lin = g.reduction_tolerance(lin["shadow"], reduction_mode="stacked")
        assert tol_lin["ok"] is False and "in one ROW 1.00e+00 > 0.02" in tol_lin["detail"]
        assert "in one GROUP 3.12e-02 > 0.002" in tol_lin["detail"] and "> 4" not in tol_lin["detail"]
        code, checks = _exit_with(dropped_row)
        assert code == 2 and "in one ROW" in checks["shadow_compare"]["detail"]
        assert g.shadow_multigroup_check(dict(dropped_row["shadow"], n_layer_calls_with_ge2_cells_compared=1),
                                         reduction_mode="stacked")[0] is False
        # the bounds are inclusive; a record exactly at them passes
        at = _measured_case(max_frac_row=g.REDUCTION_ROW_FRAC_TOL, max_frac_group=g.REDUCTION_GROUP_FRAC_TOL)
        assert g.reduction_tolerance(at["shadow"], reduction_mode="stacked")["ok"] is True
        # a group-level defect alone (a whole small group off by one output ulp): the group bound
        grp = _measured_case(max_frac_group=0.5, max_frac_row=1e-3)
        tol = g.reduction_tolerance(grp["shadow"], reduction_mode="stacked")
        assert tol["ok"] is False and "in one GROUP 5.00e-01 > 0.002" in tol["detail"] and "in one ROW" not in tol["detail"]
        # and a max_ulps_edit of exactly 0.0 with the concentration fields present is NOT "lacks the fields"
        zero = _measured_case(max_ulps_edit=0.0)
        tol = g.reduction_tolerance(zero["shadow"], reduction_mode="stacked")
        assert tol["ok"] is True and tol["keys_lacking_fields"] == []

    def test_elementwise_op_must_be_exact_in_the_stacked_mode(self):
        key = "vector_all:add_vector:vec:response:skipbos0:both"
        snap = _shadow_snapshot(unequal=1, key=key, measured={"n_unequal_elems": 1, "max_ulps_edit": 1.0})
        tol = g.reduction_tolerance(snap, reduction_mode="stacked")
        assert tol["ok"] is False and tol["elementwise_keys_unequal"] == [f"{key}:B32"]
        assert "ELEMENTWISE-op key(s) with unequal groups (exact required in every mode)" in tol["detail"]
        assert tol["n_unequal_elems"] == 0 and tol["frac_unequal_elems"] is None    # nothing pooled as reduction
        case = dict(_ok_case(), case_id="vector_all:add_vector:vec:response:skipbos0:both", shadow=snap)
        code, checks = _exit_with(case)
        assert code == 2 and checks["shadow_compare"]["ok"] is False

    def test_per_row_mode_requires_exactness_for_every_op(self):
        snap = _measured_case(n_unequal_elems=1)["shadow"]                       # 1 elem of 2e6, 1 ulp
        assert g.reduction_tolerance(snap, reduction_mode="stacked")["ok"] is True
        tol = g.reduction_tolerance(snap, reduction_mode="per_row")
        assert tol["ok"] is False and "per_row mode: 3 unequal group(s) of 96" in tol["detail"]
        assert "must be EXACT under the per-row mode" in tol["detail"]
        case = _per_row_mode_case()
        case["shadow"] = snap
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True, reduction_mode="per_row")
        assert checks["shadow_compare"]["ok"] is False
        assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 2
        assert g.shadow_check(snap, (1, 8, 32), routing=g.ROUTING_VECTORIZED, waves=_shadow_waves(),
                              reduction_mode="per_row")[0] is False

    def test_flag_disagreement_fails_in_either_mode(self):
        snap = _shadow_snapshot(unequal=3, measured=MEASURED, flags=1)
        for mode in g.REDUCTION_MODES:
            tol = g.reduction_tolerance(snap, reduction_mode=mode)
            assert tol["ok"] is False and tol["n_flag_disagreements"] == 1, mode
            assert "disagreed on whether the rows were modified at all" in tol["detail"]
            assert "routing/masking defect, never kernel variance" in tol["detail"]
        code, checks = _exit_with(dict(_ok_case(), shadow=snap))
        assert code == 2 and checks["shadow_compare"]["ok"] is False

    def test_unequal_reduction_group_without_the_tolerance_fields_fails(self):
        # (1) the zeros a record keeps when the worker recorded no ulp-scale fields
        tol = g.reduction_tolerance(_shadow_snapshot(unequal=1), reduction_mode="stacked")
        assert tol["ok"] is False and "lacks the tolerance fields" in tol["detail"]
        assert tol["keys_lacking_fields"] == ["psr:relu_gated_vector:gemv:response:skipbos0:both:B32"]
        # (2) an older record without the keys at all
        snap = _shadow_snapshot(unequal=1)
        for k in ("n_elems_compared", "n_unequal_elems", "max_ulps_edit", "max_frac_row", "max_frac_group"):
            for e in snap["by_key"].values():
                e.pop(k, None)
            snap.pop(k, None)
        assert g.reduction_tolerance(snap, reduction_mode="stacked")["ok"] is False
        # (3) n_elems_compared == 0 with elements differing; (4) max_ulps_edit missing; (5) a
        # concentration field missing or 0.0 while elements differed (a record from before round 13)
        for m in ({"n_unequal_elems": 3, "max_ulps_edit": 1.0, "n_elems_compared": 0},
                  {"n_unequal_elems": 3, "max_ulps_edit": None},
                  {"n_unequal_elems": 3, "max_ulps_edit": 1.0, "max_frac_row": None},
                  {"n_unequal_elems": 3, "max_ulps_edit": 1.0, "max_frac_row": 0.0},
                  {"n_unequal_elems": 3, "max_ulps_edit": 1.0, "max_frac_group": None},
                  {"n_unequal_elems": 3, "max_ulps_edit": 1.0, "max_frac_group": 0.0},
                  {"n_unequal_elems": 0, "max_ulps_edit": 1.0},
                  {"n_unequal_elems": None, "max_ulps_edit": 1.0}):
            tol = g.reduction_tolerance(_shadow_snapshot(unequal=1, measured=m), reduction_mode="stacked")
            assert tol["ok"] is False and "lacks the tolerance fields" in tol["detail"], m
            assert "missing evidence, never kernel variance" in tol["detail"]
        code, checks = _exit_with(dict(_ok_case(), shadow=_shadow_snapshot(unequal=1)))
        assert code == 2

    def test_unclassifiable_or_unattributed_unequal_groups_fail(self):
        snap = _shadow_snapshot()
        snap["by_key"]["psr:mystery_op:gemv:response:skipbos0:both:B8"] = dict(
            snap["by_key"]["psr:relu_gated_vector:gemv:response:skipbos0:both:B8"])
        tol = g.reduction_tolerance(snap, reduction_mode="stacked")
        assert tol["ok"] is False and "do not classify" in tol["detail"] and len(tol["unclassified_keys"]) == 1
        # an unequal group that landed on no key (the key cap) cannot be classified either
        snap = _shadow_snapshot()
        snap.update(n_unequal=1, n_keys_dropped=1, first_unequal={"key": "dropped", "max_abs_diff": 0.5})
        tol = g.reduction_tolerance(snap, reduction_mode="stacked")
        assert tol["ok"] is False and "1 unequal group(s) attributed to no key (1 key(s) dropped)" in tol["detail"]

    def test_an_op_less_key_fails_in_both_modes_even_with_nothing_unequal(self):
        """The 2026-09-04 llama8b multigroup FAIL, pinned from the gate's side
        so the rule cannot be softened: a ``by_key`` entry keyed with
        placeholder dashes (``psr:-:-:-:-:-:B6`` — a group whose cell served
        no op at that hook site, recorded by the pre-fix worker as
        "unmodified on both sides" under an op-less key) fails
        ``reduction_tolerance`` in EVERY reduction mode, with ZERO unequal
        groups anywhere, because the classification runs over every entry
        before the "exact" early return. So the defect was latent on main
        (the dash keys date from the shadow compare's introduction) and what
        exposed it is this loop, added with the stacked-reductions contract —
        NOT the stacked mode's ulp-level unequal groups: a per_row-mode
        multigroup run under this gate fails identically. The fixed worker
        records such a group as UNSERVED (fields the tolerance ignores) and
        never keys it."""
        dash = {"n_compared": 0, "n_equal": 0, "n_unequal": 0, "n_unmodified": 4032,
                "n_shadow_errors": 0, "max_abs_diff": 0.0, "max_ulps": 0.0, "max_abs_ref": 0.0,
                "n_unequal_elems": 0, "n_elems_compared": 0, "max_ulps_edit": 0.0,
                "max_frac_row": 0.0, "max_frac_group": 0.0, "n_flag_disagreements": 0,
                "first_unequal": None}
        for mode in g.REDUCTION_MODES:
            snap = _shadow_snapshot()
            snap["by_key"]["psr:-:-:-:-:-:B6"] = dict(dash)
            snap["n_unmodified_groups"] += 4032
            tol = g.reduction_tolerance(snap, reduction_mode=mode)
            assert tol["ok"] is False and tol["n_unequal"] == 0, mode
            assert "names no op" in tol["detail"] and "never a default class" in tol["detail"]
            assert len(tol["unclassified_keys"]) == 1 and "psr:-:-:-:-:-:B6" in tol["unclassified_keys"][0]
            assert [e["class"] for e in tol["per_key"] if e["key"] == "psr:-:-:-:-:-:B6"] == [None]
            ok, detail = g.shadow_multigroup_check(
                dict(snap, n_layer_calls_with_ge2_cells_compared=64,
                     max_distinct_cells_compared_in_one_layer_call=3), reduction_mode=mode)
            assert ok is False and "names no op" in detail, mode
        # the fixed worker's record: the same groups counted UNSERVED per kind@site,
        # no op-less key — the tolerance reads nothing of it and the case is exact
        snap = _shadow_snapshot()
        snap.update(n_unserved_groups=36448, unserved={"psr@oproj": 12128, "lora_all@block": 16192},
                    first_unserved={"kind": "lora_all", "cell": "si_constitution__lora_r4", "site": "block",
                                    "layer": 0, "rows": 2},
                    n_unserved_modified=0, first_unserved_modified=None)
        for mode in g.REDUCTION_MODES:
            assert g.reduction_tolerance(snap, reduction_mode=mode)["ok"] is True, mode
        # ... and a side that modified rows it serves no op at rides in as a flag
        # disagreement, which fails in every mode too
        snap.update(n_unserved_modified=1, n_flag_disagreements=1)
        for mode in g.REDUCTION_MODES:
            tol = g.reduction_tolerance(snap, reduction_mode=mode)
            assert tol["ok"] is False and "disagreed on whether the rows were modified" in tol["detail"], mode

    def test_exact_records_and_missing_records(self):
        for mode in g.REDUCTION_MODES:
            tol = g.reduction_tolerance(_shadow_snapshot(), reduction_mode=mode)
            assert tol["ok"] is True and tol["detail"] == f"exact ({mode} mode): every compared group equal"
            assert tol["n_unequal"] == 0 and tol["frac_unequal_elems"] == 0.0 and tol["max_ulps_edit"] is None
            assert tol["max_frac_row"] is None and tol["max_frac_group"] is None
            assert tol["max_gate_diff_rel_sum"] is None and tol["max_gate_flip_rel_sum"] is None
            assert tol["max_ulps_edit_linear"] is None and tol["max_ulps_edit_gated"] is None
            assert tol["max_gate_edit_consistency_ulps"] is None
            assert tol["n_gate_groups_observed"] == 0 and tol["gate_exact_rows_unequal_elems"] == 0
            assert tol["max_gate_lowrank_zero_delta_ulps_edit"] is None
        assert g.reduction_tolerance({"rpc": "absent"}, reduction_mode="stacked")["ok"] is None
        assert g.reduction_tolerance(None, reduction_mode="stacked")["ok"] is None
        with pytest.raises(g.GateError, match="unknown reduction mode"):
            g.reduction_tolerance(_shadow_snapshot(), reduction_mode="sideways")

    def test_shadow_check_routes_through_the_tolerance_in_both_modes(self):
        case = _measured_case()
        ok, detail = g.shadow_check(case["shadow"], (1, 8, 32), routing=g.ROUTING_VECTORIZED,
                                    waves=_shadow_waves(), reduction_mode="stacked")
        assert ok is True
        assert "within the reduction contract (stacked mode)" in detail and "<= 0.0001" in detail
        assert "counted comparisons per submitted wave size: {1: 32, 8: 32, 32: 32}" in detail
        ok, detail = g.shadow_check(case["shadow"], (1, 8, 32), routing=g.ROUTING_VECTORIZED,
                                    waves=_shadow_waves(), reduction_mode="per_row")
        assert ok is False and "per_row mode" in detail
        # the default mode is stacked, and the ok path still needs wave coverage
        assert g.shadow_check(case["shadow"], (1, 8, 32), routing=g.ROUTING_VECTORIZED, waves=_shadow_waves())[0] is True
        assert g.shadow_check(case["shadow"], (1, 8, 32), routing=g.ROUTING_VECTORIZED,
                              waves=_shadow_waves(batches=(1, 8)))[0] is None
        # a per-slot case with a counted (and unequal) comparison is still a routing disagreement
        assert g.shadow_check(case["shadow"], (1, 8, 32), routing=g.ROUTING_PER_SLOT, waves=None)[0] is False
        # through case_checks: the measured class is green under stacked, exit 0
        code, checks = _exit_with(case)
        assert code == 0 and checks["shadow_compare"]["ok"] is True
        assert "frac" in checks["shadow_compare"]["detail"] or "elements unequal" in checks["shadow_compare"]["detail"]

    def test_shadow_multigroup_check_routes_through_the_tolerance_in_both_modes(self):
        mg = _mg_case()
        measured = _shadow_snapshot(batches=(32,), unequal=3, measured=MEASURED)
        mg["shadow"] = dict(measured, n_layer_calls_with_ge2_cells_compared=64,
                            max_distinct_cells_compared_in_one_layer_call=3, cells_compared=mg["cells"])
        ok, detail = g.shadow_multigroup_check(mg["shadow"], reduction_mode="stacked")
        assert ok is True and "within the reduction contract (stacked mode)" in detail and "64 layer call(s)" in detail
        ok, detail = g.shadow_multigroup_check(mg["shadow"], reduction_mode="per_row")
        assert ok is False and "per_row mode" in detail
        assert g.shadow_multigroup_check(mg["shadow"])[0] is True                      # default: stacked
        checks = g.multigroup_checks(mg, g.multigroup_spec(32), mg["cells"], reference_sha=BASE, base_sha=BASE,
                                     have_oracle=True)
        assert checks["shadow_compare"]["ok"] is True
        assert g.overall_exit_code({"cases": [dict(mg, checks=checks)], "complete": True}) == 0
        checks = g.multigroup_checks(mg, g.multigroup_spec(32), mg["cells"], reference_sha=BASE, base_sha=BASE,
                                     have_oracle=True, reduction_mode="per_row")
        assert checks["shadow_compare"]["ok"] is False
        # still INCONCLUSIVE without the two-cell layer call, whatever the tolerance says
        mg["shadow"] = dict(mg["shadow"], n_layer_calls_with_ge2_cells_compared=0)
        assert g.shadow_multigroup_check(mg["shadow"])[0] is None

    def test_case_checks_expect_the_modes_apply_paths_and_attested_mode(self):
        # stacked (default): the stacked paths and the attested stacked mode -> green
        code, checks = _exit_with(_ok_case())
        assert code == 0 and checks["reduction_mode_candidate"]["ok"] and checks["reduction_mode_candidate_shadow"]["ok"]
        assert "lens_reduction_mode='stacked'" in checks["reduction_mode_candidate"]["detail"]
        # per_row: the per-row paths and mode are what a per_row-mode gate expects
        case = _per_row_mode_case()
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True, reduction_mode="per_row")
        assert all(c["ok"] for c in checks.values()), {k: v for k, v in checks.items() if not v["ok"]}
        assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 0
        # the SAME per-row standups judged by a stacked-mode gate: paths and modes both fail (exit 2)
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True, reduction_mode="stacked")
        for k in ("candidate_apply_path", "shadow_apply_path", "reduction_mode_candidate",
                  "reduction_mode_candidate_shadow"):
            assert checks[k]["ok"] is False, k
        assert "expects 'vectorized+stacked'" in checks["candidate_apply_path"]["detail"]
        assert "expects 'vectorized+stacked+shadow(serves per_row)'" in checks["shadow_apply_path"]["detail"]
        assert "attests lens_reduction_mode='per_row'; the gate runs the 'stacked' mode" in checks["reduction_mode_candidate"]["detail"]
        assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 2
        # and stacked standups judged by a per_row-mode gate fail the same four
        checks = g.case_checks(_ok_case(), reference_sha=BASE, base_sha=BASE, have_oracle=True, reduction_mode="per_row")
        assert not any(checks[k]["ok"] for k in ("candidate_apply_path", "shadow_apply_path",
                                                 "reduction_mode_candidate", "reduction_mode_candidate_shadow"))
        # the bare token / the pre-landing strings never pass either mode
        for stale in (g.APPLY_PATH_VECTORIZED, "vectorized+shadow(serves per_row)"):
            code, checks = _exit_with(_ok_case(), apply_path={g.REFERENCE: {"rpc": "absent"}, g.CANDIDATE: stale,
                                                              g.SHADOW: stale})
            assert code == 2 and not checks["candidate_apply_path"]["ok"] and not checks["shadow_apply_path"]["ok"]
        # a missing attested mode fails, naming the missing key; workers that disagree fail too
        case = _ok_case()
        del case["steering_evidence"][g.CANDIDATE]["lens_reduction_mode"]
        code, checks = _exit_with(case)
        assert code == 2 and checks["reduction_mode_candidate"]["ok"] is False
        assert "attests no lens_reduction_mode" in checks["reduction_mode_candidate"]["detail"]
        assert checks["reduction_mode_candidate_shadow"]["ok"] is True
        case = _ok_case()
        case["steering_evidence"][g.SHADOW].update(lens_reduction_mode=None,
                                                   lens_reduction_modes_seen=["per_row", "stacked"])
        code, checks = _exit_with(case)
        assert code == 2 and "workers disagree: ['per_row', 'stacked']" in checks["reduction_mode_candidate_shadow"]["detail"]
        case = _ok_case()
        del case["steering_evidence"][g.SHADOW]
        code, checks = _exit_with(case)
        assert code == 2 and checks["reduction_mode_candidate_shadow"]["ok"] is False
        # the reference (base commit) is never asked for a mode
        assert "reduction_mode_reference" not in checks
        assert g.reduction_mode_evidence_ok(None, "stacked")[0] is False
        with pytest.raises(g.GateError):
            g.reduction_mode_evidence_ok({}, "sideways")

    def test_multigroup_checks_hold_the_mode_too(self):
        mg = _mg_case()
        checks = g.multigroup_checks(mg, g.multigroup_spec(32), mg["cells"], reference_sha=BASE, base_sha=BASE,
                                     have_oracle=True)
        assert checks["reduction_mode_candidate"]["ok"] and checks["reduction_mode_candidate_shadow"]["ok"]
        assert checks["candidate_apply_path"]["ok"] and checks["shadow_apply_path"]["ok"]
        checks = g.multigroup_checks(mg, g.multigroup_spec(32), mg["cells"], reference_sha=BASE, base_sha=BASE,
                                     have_oracle=True, reduction_mode="per_row")
        for k in ("candidate_apply_path", "shadow_apply_path", "reduction_mode_candidate",
                  "reduction_mode_candidate_shadow"):
            assert checks[k]["ok"] is False, k
        mg = _mg_case()
        del mg["steering_evidence"][g.SHADOW]
        checks = g.multigroup_checks(mg, g.multigroup_spec(32), mg["cells"], reference_sha=BASE, base_sha=BASE,
                                     have_oracle=True)
        assert checks["reduction_mode_candidate_shadow"]["ok"] is False and checks["reduction_mode_candidate"]["ok"]
        assert g.overall_exit_code({"cases": [dict(mg, checks=checks)], "complete": True}) == 2

    def test_standup_env_sets_the_mode_explicitly_and_scrubs_when_none(self):
        for inherited in ("1", "0", "garbage"):
            base = {"PATH": "/bin", g.STACKED_ENV: inherited, g.SHADOW_ENV: "1", g.PER_ROW_ENV: "1"}
            assert g.STACKED_ENV not in g.standup_env(base, Path("/ref"), batch_invariant=0)   # None: scrubbed
            assert g.standup_env(base, Path("/c"), batch_invariant=0, reduction_mode="stacked")[g.STACKED_ENV] == "1"
            assert g.standup_env(base, Path("/c"), batch_invariant=0, reduction_mode="per_row")[g.STACKED_ENV] == "0"
        env = g.standup_env(base, Path("/c"), batch_invariant=0, reduction_mode="stacked", env_extra={g.SHADOW_ENV: "1"})
        assert env[g.STACKED_ENV] == "1" and env[g.SHADOW_ENV] == "1" and g.PER_ROW_ENV not in env
        # env_extra is applied last (the A/B driver sets the switch itself, no mode given)
        import lens_stacked_ab as ab
        env = g.standup_env(base, Path("/c"), batch_invariant=0, env_extra=ab.SIDES[ab.PER_ROW][0])
        assert env[g.STACKED_ENV] == "0"
        with pytest.raises(g.GateError, match="unknown reduction mode"):
            g.standup_env(base, Path("/c"), batch_invariant=0, reduction_mode="sideways")
        # GateContext carries the mode, stacked by default
        import dataclasses
        mode_field = {f.name: f for f in dataclasses.fields(g.GateContext)}["reduction_mode"]
        assert mode_field.default == "stacked"
        # start_standup / run_gate thread it: every candidate-tree standup passes the mode, the
        # reference passes None (source pins — the runtime needs a GPU)
        import inspect
        src = inspect.getsource(g.run_gate)
        assert src.count("reduction_mode=ctx.reduction_mode") >= 5      # passthrough, bi0 probe, checks x2, kr x2 ...
        assert "(REFERENCE, ctx.reference, ref_sha, None, None)" in src
        assert "(CANDIDATE, ctx.candidate, cand_sha, None, ctx.reduction_mode)" in src
        assert "(SHADOW, ctx.candidate, cand_sha, {SHADOW_ENV: \"1\"}, ctx.reduction_mode)" in src
        assert "(LEGACY, ctx.candidate, cand_sha, {PER_ROW_ENV: \"1\"}, ctx.reduction_mode)" in src
        assert "reduction_mode=rmode" in src
        assert "reduction_mode=reduction_mode)" in inspect.getsource(g.start_standup)

    def test_start_standup_refuses_an_endpoint_that_attests_another_mode(self):
        ep = {"lens_reduction_mode": "stacked", "lens_reduction_mode_error": None,
              "stacked_reductions_env": "1"}
        ok, why = g.reduction_mode_endpoint_ok(ep, "stacked")
        assert ok and "attests lens_reduction_mode='stacked' = the gate's mode" in why
        ok, why = g.reduction_mode_endpoint_ok(ep, "per_row")
        assert not ok and "the gate set the 'per_row' mode" in why and "attests lens_reduction_mode='stacked'" in why
        assert "does not run" in why
        # the helper could not read a mode (an older worker / a disagreement): refused, naming why
        ep_none = {"lens_reduction_mode": None, "stacked_reductions_env": "1",
                   "lens_reduction_mode_error": "worker 0: no lens_reduction_mode in its scheduler-config report"}
        ok, why = g.reduction_mode_endpoint_ok(ep_none, "stacked")
        assert not ok and "worker 0: no lens_reduction_mode" in why
        assert not g.reduction_mode_endpoint_ok({}, "stacked")[0]
        assert not g.reduction_mode_endpoint_ok(None, "stacked")[0]
        # the base-commit reference is asked for nothing; what is there is recorded
        ok, why = g.reduction_mode_endpoint_ok(ep_none, None)
        assert ok and "no reduction mode required" in why and "worker 0" in why
        assert g.reduction_mode_endpoint_ok(None, None)[0] is True
        with pytest.raises(g.GateError, match="unknown reduction mode"):
            g.reduction_mode_endpoint_ok(ep, "sideways")
        # start_standup applies it after the prefix-caching refusal and records the verdict;
        # the A/B driver applies it per side
        import inspect
        src = inspect.getsource(g.start_standup)
        assert "reduction_mode_endpoint_ok(endpoint, reduction_mode)" in src
        assert src.index("pc_check(endpoint)") < src.index("reduction_mode_endpoint_ok(endpoint")
        assert 'endpoint["reduction_mode_check"]' in src
        import lens_stacked_ab as ab
        assert ab.SIDE_MODE == {ab.PER_ROW: "per_row", ab.STACKED: "stacked", ab.STACKED_SHADOW: "stacked"}
        assert "g.reduction_mode_endpoint_ok(su.endpoint, SIDE_MODE[side])" in inspect.getsource(ab.drive_side)
        for fn in (g.drive_standup, g.drive_shadow, g.drive_multigroup):
            assert '"reduction_mode_endpoint": su.endpoint.get("reduction_mode_check")' in inspect.getsource(fn), fn

    def test_cli_flag_and_env_seam(self, monkeypatch, tmp_path):
        assert g.parse_reduction_mode_spec(None) == g.parse_reduction_mode_spec("") == "stacked"
        assert g.parse_reduction_mode_spec("stacked") == g.parse_reduction_mode_spec("1") == "stacked"
        assert g.parse_reduction_mode_spec(" PER_ROW ") == g.parse_reduction_mode_spec("0") == "per_row"
        for bad in ("true", "off", "sideways", "2"):
            with pytest.raises(g.GateError, match="not one of"):
                g.parse_reduction_mode_spec(bad)
        captured = {}

        def fake_run_gate(ctx, **kw):
            captured["ctx"] = ctx
            return {"exit_code": 0}
        monkeypatch.setattr(g, "run_gate", fake_run_gate)
        monkeypatch.setattr(g, "plan_inventory", lambda *a, **k: {})
        monkeypatch.setattr(g, "header_len_for", lambda *a, **k: 4)
        monkeypatch.delenv(g.REDUCTION_MODE_ENV, raising=False)
        args = ["--model", "llama8b", "--reference-repo", str(tmp_path), "--results-root",
                str(tmp_path / "r"), "--objects-root", str(tmp_path / "o"), "--candsi-root",
                str(tmp_path / "c")]
        assert g.main(args) == 0 and captured["ctx"].reduction_mode == "stacked"          # the default
        assert g.main(args + ["--reduction-mode", "per_row"]) == 0 and captured["ctx"].reduction_mode == "per_row"
        monkeypatch.setenv(g.REDUCTION_MODE_ENV, "0")                                    # the sbatch seam
        assert g.main(args) == 0 and captured["ctx"].reduction_mode == "per_row"
        monkeypatch.setenv(g.REDUCTION_MODE_ENV, "per_row")
        assert g.main(args + ["--reduction-mode", "stacked"]) == 0 and captured["ctx"].reduction_mode == "stacked"
        monkeypatch.setenv(g.REDUCTION_MODE_ENV, "sideways")
        with pytest.raises(g.GateError, match="not one of"):
            g.main(args)
        with pytest.raises(SystemExit):
            g.main(args + ["--reduction-mode", "sideways"])                             # argparse choices
        assert g.REDUCTION_MODE_ENV == "GATE_REDUCTION_MODE"
        sb = (REPO / "scripts/sbatch/lens_vectorize_gate.sbatch").read_text()
        assert "GATE_REDUCTION_MODE" in sb and "bit for bit" not in sb
        assert "bounded bf16-kernel-variance tolerance" in sb

    def test_report_states_the_contract_and_the_numbers(self):
        ms = g.MODELS["llama8b"]
        measured = _measured_case()
        measured["checks"] = g.case_checks(measured, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        measured["reduction_tolerance"] = g.reduction_tolerance(measured["shadow"], reduction_mode="stacked")
        exact = _ok_case()
        exact["case_id"] = "lora_all:lora_oproj:rankK:response:skipbos0:both"
        exact["checks"] = g.case_checks(exact, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        doc = {"model": dict(key=ms.key, hf=ms.hf, rev=ms.rev, cfg=ms.cfg, bank=ms.bank, branch=ms.branch),
               "geometry": {"max_model_len": 4096, "max_num_seqs": 64}, "batches": [1, 8, 32],
               "timing_batches": [8], "cases": [measured, exact], "inventory": {}, "uncovered_cases": [],
               "complete": True, "batch_invariant": 0, "exit_code": 0, "reduction_mode": "stacked",
               "kind_coverage": {"lowrank_gated": {"servable": False, "decode_capable": True,
                                                 "decode_capable_cases": ["lowrank_gated:lowrank_linear_gated:rankK:all:skipbos0:both"],
                                                 "cases": [], "routing": ["vectorized"], "reason": "no Petri arm"}}}
        text = g.render_report(doc)
        # the header line: the mode, the switch, the contract in the module's words
        assert "reduction mode: stacked (SISTEER_LENS_STACKED_REDUCTIONS=1 set explicitly on every candidate-tree standup" in text
        assert g.REDUCTION_CONTRACT in text
        assert "B=1 check (pure_B1, controls_B1, steering_bites_B1) stays EXACT in either mode" in text
        # the shadow column: compared/unequal, and the pooled numbers when a group was unequal
        mt = measured["reduction_tolerance"]
        frac, row, grp = (g._fmt_frac(mt["frac_unequal_elems"]), g._fmt_frac(mt["max_frac_row"]),
                          g._fmt_frac(mt["max_frac_group"]))
        # a GATED key's column carries the gate rule's numbers in the contract's unit, then the evidence read
        assert (f"| 96/3 frac={frac} row={row} group={grp} ulps_edit=1 gate_rel_sum=0.8 x 2^-24 flip_rel_sum=n/a "
                "relu_flips=0 gate_cons=0.5 exact_ne=0 recon_ne=0 gate_rel=0.02 gate_min=1.4e-05 |") in text
        assert "| 96/0 |" in text
        assert ("shadow (compared/unequal[ frac row group ulps_edit[ gate_rel_sum flip_rel_sum relu_flips gate_cons "
                "exact_ne recon_ne]])") in text
        # the per-case shadow line carries the pooled numbers, the concentration maxima, the class rule and the verdict
        assert (f"reduction-op elements unequal 3/{mt['n_elems_compared']} (frac {frac} <= 0.0001; max in one "
                f"row {row} <= 0.02, in one group {grp} <= 0.002), max_ulps_edit 1 (<= 4 for linear ops, informational "
                "for gated ops); gated ops: |delta gate|/sum|hw| 0.8 x 2^-24 (<= 16 x 2^-24), ReLU flips 0 (fired side "
                "n/a), gate-edit identity 0.5 ulps (<= 4), exact-row unequal elements 0 (must be 0), write reconstruction "
                "mismatches 0 (must be 0; 3 group(s) rebuilt), observed on 3 group(s); flag disagreements 0; contract: ok") in text
        # a LINEAR key's column carries no gate numbers
        lin = _measured_case(key=LINEAR_KEY)
        lin["checks"] = g.case_checks(lin, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        lin["reduction_tolerance"] = g.reduction_tolerance(lin["shadow"], reduction_mode="stacked")
        text_lin = g.render_report(dict(doc, cases=[lin]))
        assert f"| 96/3 frac={frac} row={row} group={grp} ulps_edit=1 |" in text_lin
        assert "max_ulps_edit 1 (<= 4 for linear ops, informational for gated ops); flag disagreements 0; contract: ok" in text_lin
        assert "gated ops: |delta gate|/sum|hw|" not in text_lin           # no per-case gate numbers on a linear case
        # the footer restates the class-dependent contract
        assert "gated ops: |g_stacked - g_per_row| <= 16 x 2^-24 x sum|h_i w_i| per gate element" in text
        assert ("gates observed on both applies and applied - ref = eff x (g_stacked - g_per_row) x v within 4 ulps, "
                "writes exact where the gates agree and each gated-vector side the bit-exact reconstruction from its "
                "observed gate") in text
        # the histogram appears in the case line only when the record carries it, after the verdict, informational
        assert "gate magnitudes" not in text
        hist = measured["shadow"]
        hist.update(gate_hist_n_groups=96, gate_hist_zero=40, gate_hist_near=3, gate_hist_mid=400, gate_hist_far=500,
                    gate_dose_share_near_threshold=0.002, gate_g_over_f_min_nonzero=2.5, gate_g_over_f_max=3e6)
        measured["reduction_tolerance"] = g.reduction_tolerance(hist, reduction_mode="stacked")
        assert measured["reduction_tolerance"]["ok"] is True
        text_h = g.render_report(doc)
        assert ("contract: ok; gate magnitudes over 96 compared group(s), equal and unequal (per-row, in noise floors f): "
                "zero 40, near (0 < g <= 10 f) 3, mid (<= 1000 f) 400, far 500; near-threshold dose share 0.002; g/f nonzero "
                "min 2.5, max 3e+06 (informational)") in text_h
        # the kind-coverage note no longer claims certification by construction
        assert "certified by construction" not in text and "every reduction per row inside a group" not in text
        assert "their call shapes and exact integer arithmetic are certified by the CPU call-shape / torch.equal tests only" in text
        assert "the reduction-op tolerance is certified on hardware only for the cases this gate sees" in text
        # the footer
        assert ("reduction ops: bounded bf16-kernel-variance tolerance (fraction of unequal elements <= 0.0001 over "
                "the case, <= 0.002 in any group, <= 0.02 in any row; linear ops: every unequal element <= 4 bf16 ulps "
                "of the per-row edit beyond one output ulp; gated ops: |g_stacked - g_per_row| <= 16 x 2^-24 x "
                "sum|h_i w_i| per gate element, a ReLU flip only when the fired side is within that bound, gates "
                "observed on both applies and applied - ref = eff x (g_stacked - g_per_row) x v within 4 ulps, writes "
                "exact where the gates agree and each gated-vector side the bit-exact reconstruction from its observed "
                "gate)") in text
        assert "elementwise + per-row mode: exact." in text and "a missing pure_B1" in text
        # the per-row mode renders its own switch value
        assert "reduction mode: per_row (SISTEER_LENS_STACKED_REDUCTIONS=0" in g.render_report(dict(doc, reduction_mode="per_row"))
        # a doc without the field (an older JSON) renders under the default
        assert "reduction mode: stacked" in g.render_report({k: v for k, v in doc.items() if k != "reduction_mode"})

    def test_docstring_states_the_new_truth_and_drops_the_old(self):
        doc = g.__doc__
        assert "THE REDUCTION CONTRACT" in doc
        for phrase in ("fraction of unequal ELEMENTS\nis <= 1e-4", "REDUCTION_FRAC_TOL", "REDUCTION_ULPS_EDIT_TOL",
                       "REDUCTION_GROUP_FRAC_TOL", "REDUCTION_ROW_FRAC_TOL", "MEASURED,\nnot mechanism",
                       "4 * ulp(|ref - base|) + 1 * ulp(|ref|)", "ANY SINGLE ROW <= 2e-2", "ANY SINGLE GROUP <= 2e-3",
                       "4 bf16 ulps", "3.9e-5", "predicts <= 2", "lacks the tolerance fields",
                       "never for a defect or for missing evidence", "x.shape[0] <= 1",
                       "pure_B1=PASS on all 26 cases", "five families", "reduction_mode_*",
                       "vectorized+<mode>", "shadow_key_op_class", "op_reduces_in_group",
                       # the 2026-09-06 contract change
                       "reduction_op_class", "REDUCTION_GATE_C", "GATED_REDUCTION_OPS", "LINEAR_REDUCTION_OPS",
                       "|g_stacked - g_per_row| <= C * 2^-24 * sum_i |h_i w_i| with C = 16",
                       "gate_max_diff_rel_sum", "gate_max_flip_rel_sum", "gate_n_groups_with_stats",
                       "is NOT a rule there", "lacks observed gate stats",
                       # observed gates, the gate-edit identity
                       "THE GATES ARE OBSERVED, NOT\nRECOMPUTED", "GateCapture", "_gate_stats_observed",
                       "gate_edit_consistency_max_ulps", "REDUCTION_GATE_CONSISTENCY_ULPS_TOL",
                       # provenance enforced, exact where the gates agree
                       "gate_n_groups_observed", "gate_exact_rows_unequal_elems",
                       "REDUCTION_GATE_EXACT_ROWS_UNEQUAL_TOL", "gate_lowrank_zero_delta_max_ulps_edit",
                       # the gate-magnitude histogram: reported, never ruled on; over every compared group
                       "GATE-MAGNITUDE HISTOGRAM", "gate_dose_share_near_threshold", "NEVER ruled on",
                       "gate_hist_n_groups", "equal or unequal",
                       # round 4 P1 / P2-b: the write reconstruction; the truthful cost
                       "WRITE RECONSTRUCTION", "gated_vector_write", "gate_write_recon_mismatch_elems",
                       "REDUCTION_GATE_WRITE_RECON_MISMATCH_TOL", "one module-global\nlookup and one branch",
                       "three\nbranches in the stacked mode", "B + 2 in the per-row mode", "not zero",
                       # round 5 P3: low-rank zero-delta rows are bounded, never exact
                       "never held exact, identical latents or not", "BOUNDED (<= 4 edit ulps",
                       "exactness-at-zero belongs to the vector pair alone"):
            assert phrase in doc, phrase
        for gone in ("never a tolerance", "bit-identical BY CONSTRUCTION", "a group issued per row",
                     "certified by construction", "every reduction inside a group is issued per row"):
            assert gone not in doc, gone
        # and the A/B driver's table names the explicit per-row switch
        import lens_stacked_ab as ab
        assert ab.SIDES[ab.PER_ROW] == ({g.STACKED_ENV: "0"}, "vectorized+per_row")
        assert ab.SIDES[ab.STACKED] == ({g.STACKED_ENV: "1"}, "vectorized+stacked")
        assert ab.SIDES[ab.STACKED_SHADOW] == ({g.STACKED_ENV: "1", g.SHADOW_ENV: "1"},
                                               "vectorized+stacked+shadow(serves per_row)")
        assert "the landed path (flag unset)" not in ab.__doc__ and 'apply_path "vectorized"\n' not in ab.__doc__
        assert "SISTEER_LENS_STACKED_REDUCTIONS=0 apply_path \"vectorized+per_row\"" in ab.__doc__

    def test_ab_report_files_the_gate_contract_verdict(self):
        import lens_stacked_ab as ab
        snap = _shadow_snapshot(batches=(8, 32), unequal=3, measured=MEASURED)
        rows = ab.shadow_rows_table(snap)
        assert rows[1]["max_ulps_edit"] == 1.0 and rows[1]["n_elems_compared"] == 2_000_000
        assert rows[0]["max_ulps_edit"] == 0.0 and rows[0]["n_flag_disagreements"] == 0
        doc = {"model": {"key": "m", "hf": "h", "rev": "r" * 12}, "tree": "/t", "sha": "s" * 12,
               "geometry": {"max_model_len": 1, "max_num_seqs": 1}, "batches": [8], "repeats": 2, "max_tokens": 64,
               "ignore_eos": True, "timing_batches": [8],
               "cases": [{"case_id": "c", "objects": "synthetic", "cell": "x",
                          "apply_path": {s_: ab.SIDES[s_][1] for s_ in ab.SIDES}, "b1_sanity": {"equal": True},
                          "comparisons": {"8": {n: ab.pool([]) for n in ab.COMPARISONS}},
                          "shadow": snap, "shadow_rows": rows, "timing": {}}]}
        md = ab.render_report(doc)
        assert "gate contract (stacked mode; lens_vectorize_gate.reduction_tolerance): ok — within the reduction contract" in md
        assert "| max ulps edit |" in md and "3 of " in md
        snap_bad = _shadow_snapshot(batches=(8, 32), unequal=3,
                                    measured=dict(MEASURED, gate_max_diff_rel_sum=20 * g.REDUCTION_GATE_UNIT))
        doc["cases"][0]["shadow"] = snap_bad
        assert "reduction_tolerance): FAIL — " in ab.render_report(doc)


# --------------------------------------------------------------------------
# G1 — two checkouts
# --------------------------------------------------------------------------


class TestTrees:
    def test_check_trees(self):
        ref = {"path": "/r", "sha": BASE, "dirty": False}
        cand = {"path": "/c", "sha": "a" * 40, "dirty": False}
        assert g.check_trees(cand, ref, reference_sha=BASE) is None
        assert "SAME commit" in g.check_trees(dict(ref, path="/c"), ref, reference_sha=BASE)
        assert "not the base commit" in g.check_trees(cand, dict(ref, sha="b" * 40),
                                                       reference_sha=BASE)
        assert "modified tracked" in g.check_trees(cand, dict(ref, dirty=True), reference_sha=BASE)
        assert "modified tracked" in g.check_trees(dict(cand, dirty=True), ref, reference_sha=BASE)
        assert "not a git checkout" in g.check_trees(cand, {"path": "/r", "sha": None},
                                                      reference_sha=BASE)

    def test_tree_refusal_is_recorded_and_exit_2(self):
        doc = {"cases": [], "complete": True,
               "tree_error": "candidate and reference are the SAME commit"}
        assert g.overall_exit_code(doc) == 2

    def test_tree_facts_on_a_real_checkout(self, tmp_path):
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        (tmp_path / "f").write_text("x")
        subprocess.run(["git", "-C", str(tmp_path), "add", "f"], check=True)
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                        "commit", "-q", "-m", "c"], check=True)
        facts = g.tree_facts(tmp_path)
        assert len(facts["sha"]) == 40 and facts["dirty"] is False
        (tmp_path / "f").write_text("y")
        assert g.tree_facts(tmp_path)["dirty"] is True
        assert g.tree_facts(tmp_path / "nope")["sha"] is None

    def test_standup_helper_verifies_the_tree(self, tmp_path):
        spec = importlib.util.spec_from_file_location(
            "lens_gate_standup", REPO / "scripts" / "lens_gate_standup.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        (tmp_path / "f").write_text("x")
        subprocess.run(["git", "-C", str(tmp_path), "add", "f"], check=True)
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                        "commit", "-q", "-m", "c"], check=True)
        sha = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        assert helper.verify_tree(tmp_path, sha) == sha
        with pytest.raises(SystemExit, match="expected"):
            helper.verify_tree(tmp_path, "0" * 40)
        (tmp_path / "f").write_text("y")
        with pytest.raises(SystemExit, match="modified tracked"):
            helper.verify_tree(tmp_path, sha)
        # the helper imports nothing from its own directory
        text = (REPO / "scripts" / "lens_gate_standup.py").read_text()
        assert "import capctrl_serving_up" not in text and "_drop_own_dir()" in text


# --------------------------------------------------------------------------
# G3 — the inventory
# --------------------------------------------------------------------------


class TestInventory:
    def test_servable_arms_are_the_lens_served_vocabulary(self):
        arms = g.servable_arms()
        for arm in ("psr_kl_all", "psr_mse_all", "vector_all", "lora_r1", "lora_r4", "loreft",
                    "antidote", "alphasteer"):
            assert arm in arms, arm
        for arm in ("lofit", "token_bos", "cast", "circuit_breaker"):
            assert arm not in arms, arm

    def test_inventory_is_the_adapters_accepted_variants(self):
        """The probe over each save format's value domain, filtered by the
        REAL adapter: both LoRA forms and psr get all + response (never
        `last`, which their adapters refuse); vector_all gets all/response x
        skip_bos on/off (its class default is all + skip_bos True); loreft,
        antidote and alphasteer keep the one regime their classes fix;
        rank-1 and rank-k LoRA are distinct cases."""
        inv = g.served_inventory()
        want = {
            "psr:relu_gated_vector:gemv:all:skipbos0:both",
            "psr:relu_gated_vector:gemv:response:skipbos0:both",
            "vector_all:add_vector:vec:all:skipbos1:both",
            "vector_all:add_vector:vec:all:skipbos0:both",
            "vector_all:add_vector:vec:response:skipbos1:both",
            "vector_all:add_vector:vec:response:skipbos0:both",
            "lora:lora_oproj:rank1:all:skipbos0:both",
            "lora:lora_oproj:rank1:response:skipbos0:both",
            "lora_all:lora_oproj:rankK:all:skipbos0:both",
            "lora_all:lora_oproj:rankK:response:skipbos0:both",
            # the sdf-base-steering DERIVED difference object (ported 2026-09-10):
            # the SAME op and case signature as its lora_all parents — which is
            # the point, and is why the gate needs no new op class for it. It is
            # a separate CASE because its blob schema (both parents' factors) is
            # a separate translation the gate must exercise.
            "lora_subtraction:lora_oproj:rankK:all:skipbos0:both",
            "lora_subtraction:lora_oproj:rankK:response:skipbos0:both",
            # the paired write (kind psr_paired_write, 2026-09-17): the DERIVED difference
            # of two psr parents, served as ONE op per layer (OP_RELU_GATED_VECTOR_PAIR —
            # two same-shape gemv gates on the shared pre-steering hidden, so shape class
            # `gemv`). The `all` case alone: the class refuses a masked parent and the
            # adapter a masked leg, so — unlike psr — there is no `response` twin.
            "psr_paired_write:relu_gated_vector_pair:gemv:all:skipbos0:both",
            "loreft:affine_subspace:rankK:all:skipbos0:prefill",
            "antidote:add_vector:vec:last:skipbos0:both",
            "alphasteer:prefill_last_matmul:dxd:all:skipbos0:prefill",
            # caa_static_r (E45 H2): the Class B CAA floor's ONE deployment —
            # a fixed vector added at EVERY prompt position, prefill only
            "caa_static_r:add_vector:vec:all:skipbos0:prefill",
        }
        assert set(inv) == want
        assert not any(":last:" in k and not k.startswith("antidote") for k in inv)
        assert set(inv["psr:relu_gated_vector:gemv:response:skipbos0:both"]["arms"]) >= {"psr_kl_all", "psr_mse_all"}
        assert inv["vector_all:add_vector:vec:all:skipbos1:both"]["variants"]["vector_all"] == {
            "positions": "all", "skip_bos": True}
        covered = {arm for e in inv.values() for arm in e["arms"]}
        assert covered == set(g.servable_arms())                       # every servable arm filed
        assert all(e["sources"] == ["adapter-probe"] for e in inv.values())

    def test_format_axes_come_from_the_class_vocabularies(self):
        from sisteer.hooks import POSITION_MODES
        assert g.format_variant_axes("psr") == {"positions": tuple(POSITION_MODES)}
        assert g.format_variant_axes("vector_all")["skip_bos"] == (True, False)
        assert g.format_variant_axes("loreft") == {"positions": ("prompt", "response")}
        assert g.format_variant_axes("antidote") == {} and g.format_variant_axes("alphasteer") == {}

    def test_case_id_carries_skip_bos_and_parses(self):
        c = g.parse_case_id("vector_all:add_vector:vec:all:skipbos1:both")
        assert c == {"kind": "vector_all", "op": "add_vector", "shape": "vec", "positions": "all",
                     "skip_bos": True, "phase": "both"}
        assert g.timing_for_case("psr:relu_gated_vector:gemv:response:skipbos0:both")
        assert not g.timing_for_case("psr:relu_gated_vector:gemv:all:skipbos0:both")
        assert not g.timing_for_case("vector_all:add_vector:vec:response:skipbos1:both")
        assert not g.timing_for_case("antidote:add_vector:vec:last:skipbos0:both")
        assert g.timing_for_case(g.MULTIGROUP)

    def test_real_artifacts_join_the_inventory(self, tmp_path):
        """A real all-fit, skip_bos vector object in the bank: its adapted
        signature is already a probed case (and would be added if not) — the
        union records the artifact as that case's source, and the response
        case stays covered by a synthetic. Nothing real is rejected into
        oblivion any more."""
        bank = tmp_path / "bank"
        cell = "si_constitution__vector_all"
        (bank / cell).mkdir(parents=True)
        torch.save({"kind": "vector_all", "vectors": {"0": torch.zeros(16), "1": torch.ones(16)},
                    "positions": "all", "skip_bos": True, "fit_positions": "all",
                    "fit_response_header_len": None}, bank / cell / "object.pt")
        ms = g.MODELS["llama8b"]
        real = g.real_artifact_cases(ms, bank_dir=bank, extra_dirs=(), candsi_dir=None,
                                     header_len=len(HEADER))
        assert list(real) == ["vector_all:add_vector:vec:all:skipbos1:both"]
        inv = g.plan_inventory(ms, bank_dir=bank, extra_dirs=(), candsi_dir=None,
                               header_len=len(HEADER))
        e = inv["vector_all:add_vector:vec:all:skipbos1:both"]
        assert "adapter-probe" in e["sources"] and str(bank / cell / "object.pt") in e["sources"]
        out = g.resolve_case_object(ms, "vector_all:add_vector:vec:all:skipbos1:both", ["vector_all"],
                                    bank_dir=bank, extra_dirs=(), candsi_dir=None,
                                    stage_root=tmp_path / "stage", header_len=len(HEADER),
                                    dims_fn=lambda: TestObjectResolution.DIMS, allow_synthetic=False)
        assert out["objects"] == "real" and out["cell"] == cell
        # a real object that does not adapt is a NOTE, never a case
        bad = bank / "si_constitution__psr_kl_all"
        bad.mkdir()
        torch.save({"kind": "psr", "layers": [0]}, bad / "object.pt")
        inv = g.plan_inventory(ms, bank_dir=bank, extra_dirs=(), candsi_dir=None,
                               header_len=len(HEADER))
        assert inv["__unadaptable__"]["sources"] and "__unadaptable__" not in g.served_inventory()

    def test_synthetic_variants_are_written_with_the_cases_signature(self, tmp_path):
        ms = g.MODELS["llama8b"]
        for cid in ("vector_all:add_vector:vec:all:skipbos1:both",
                    "psr:relu_gated_vector:gemv:all:skipbos0:both",
                    "lora_all:lora_oproj:rankK:all:skipbos0:both",
                    "loreft:affine_subspace:rankK:all:skipbos0:prefill"):
            out = g.resolve_case_object(ms, cid, g.served_inventory()[cid]["arms"], bank_dir=None,
                                        extra_dirs=(), candsi_dir=None,
                                        stage_root=tmp_path / "stage", header_len=len(HEADER),
                                        dims_fn=lambda: TestObjectResolution.DIMS,
                                        allow_synthetic=True)
            assert out["objects"] == "synthetic" and out["case_id"] == cid, cid

    @pytest.mark.parametrize("arm", sorted(g.servable_arms()))
    def test_every_servable_arm_has_a_synthetic_object_the_real_adapter_hydrates(self, tmp_path, arm):
        cell, ddir = g.write_synthetic_object(arm, g.INVENTORY_DIMS, tmp_path, len(HEADER))
        kind, ops = g.adapt_object_dir(ddir / cell, arm=arm, header_len=len(HEADER))
        assert ops and kind == g.servable_arms()[arm]
        cid = g.case_id_of(kind, ops)
        assert cid.startswith(f"{kind}:")
        from sisteer.eval.defense_cell import object_path_for
        assert Path(object_path_for(ddir, cell)).is_file()             # where the registry looks

    def test_refused_variants_are_not_cases(self, tmp_path):
        # `last` for psr / lora_all: written fine, refused by the adapter -> absent from the probe
        cell, ddir = g.write_synthetic_object("psr_kl_all", g.INVENTORY_DIMS, tmp_path, len(HEADER),
                                              positions="last")
        with pytest.raises(la.AdapterError):
            g.adapt_object_dir(ddir / cell, arm="psr_kl_all", header_len=len(HEADER))
        variants = dict(g.probe_variants("psr_kl_all", "psr", len(HEADER), tmp_path / "probe"))
        assert set(variants) == {"psr:relu_gated_vector:gemv:all:skipbos0:both",
                                 "psr:relu_gated_vector:gemv:response:skipbos0:both"}
        # loreft response: refused (the class's prompt-only translation)
        cell, ddir = g.write_synthetic_object("loreft", g.INVENTORY_DIMS, tmp_path / "l", len(HEADER),
                                              positions="response")
        with pytest.raises(la.AdapterError, match="response"):
            g.adapt_object_dir(ddir / cell, arm="loreft", header_len=len(HEADER))

    def test_case_id_refuses_a_mixed_object(self):
        ops = (la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.zeros(4)),
               la.HookOp(op=la.OP_ADD_VECTOR, layer=1, vector=torch.zeros(4), phase=la.PHASE_PREFILL))
        with pytest.raises(g.GateError, match="several signatures"):
            g.case_id_of("vector_all", ops)

    def test_op_shape_classes(self):
        r1 = la.HookOp(op=la.OP_LORA_OPROJ, layer=0, lowrank_a=torch.zeros(1, 6),
                       lowrank_b=torch.zeros(8, 1))
        r4 = la.HookOp(op=la.OP_LORA_OPROJ, layer=0, lowrank_a=torch.zeros(4, 6),
                       lowrank_b=torch.zeros(8, 4))
        assert g.op_shape_class(r1) == "rank1" and g.op_shape_class(r4) == "rankK"

    def test_uncovered_case_is_incomplete(self):
        ok = _ok_case()
        ok["checks"] = g.case_checks(ok, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        doc = {"cases": [ok], "uncovered_cases": ["lora:lora_oproj:rank1:response:skipbos0:both"],
               "inventory": {"lora:lora_oproj:rank1:response:skipbos0:both": {"kind": "lora", "arms": ["lora_r1"]}},
               "model": dict(key="llama8b", hf="x", rev="r" * 40, cfg="c", bank="b", branch="plain"),
               "geometry": {"max_model_len": 4096, "max_num_seqs": 64}, "batches": [1],
               "timing_batches": [8], "exit_code": 3}
        assert g.overall_exit_code(doc) == 3
        text = g.render_report(doc)
        assert "complete=False" in text and "UNCOVERED: lora:lora_oproj:rank1" in text


class TestKindCoverage:
    """Round-8 review: the gate's inventory (Petri arms) had prefill-only
    ``loreft`` but no both-phase ``loreft_both`` case, so the five-family
    gate could not see the grouped affine-subspace class at all. The
    inventory IS derived from the Petri arm vocabulary (a standup serves what
    a cell id names) and loreft_both is a different lineage, so the
    decode-capable KINDS are derived from the adapters' registered table:
    every registered kind is covered by a Petri case or minted + adapted
    under an explicit NON_PETRI_KINDS entry, and a kind with neither
    refuses."""

    def test_every_registered_kind_is_covered_or_filed(self):
        cov = g.kind_coverage(len(HEADER))
        assert set(cov) == set(la.registered_kinds()) - set(g.INTERNAL_KINDS)
        assert set(g.INTERNAL_KINDS) == {la.SYNTHETIC_KIND, la.ZERO_KIND, la.STACKED_KIND}
        for kind, v in cov.items():
            if kind in la.UNSERVABLE_KIND_REASONS:
                assert v["servable"] is False and v["cases"] == [] and v["decode_capable"] is None
                continue
            assert v["cases"], kind                              # a case for every servable kind
            for cid in v["cases"]:
                assert g.parse_case_id(cid)["kind"] == kind
                assert v["routing_by_case"][cid] == g.case_routing(cid)
            assert v["decode_capable_cases"] == [c for c in v["cases"]
                                                 if v["routing_by_case"][c] == g.ROUTING_VECTORIZED]
            assert v["decode_capable"] is bool(v["decode_capable_cases"])
        # the Petri arms: served cases, servable on a standup
        assert cov["psr"]["servable"] is True and "psr_kl_all" in cov["psr"]["petri_arms"]
        assert cov["psr"]["cases"] == ["psr:relu_gated_vector:gemv:all:skipbos0:both",
                                       "psr:relu_gated_vector:gemv:response:skipbos0:both"]
        assert cov["psr"]["decode_capable"] is True and cov["loreft"]["decode_capable"] is False
        # lora_r4_ce / psr_ce_all (sdf-base-steering, 2026-09-04): the document-CE arms train
        # lora_r4's / psr_kl_all's EXACT object classes (MultiLayerLoRAObject / PSRObject — only
        # the loss and the data differ), so they serve through the same lens kind and are a
        # deliberate widening of the audited arm set (R5), not a new kernel shape: the kind's
        # CASES are unchanged, only the arm names that reach them.
        assert cov["lora_all"]["petri_arms"] == ["lora_r4", "lora_r4_ce"]
        assert cov["psr"]["petri_arms"] == ["psr_ce_all", "psr_kl_all"] or \
            {"psr_kl_all", "psr_ce_all"} <= set(cov["psr"]["petri_arms"])
        assert cov["antidote"]["servable"] is True
        # the non-Petri lineages: probed over their WHOLE variant domain, filed as UNSEEN
        j = cov["loreft_both"]
        # rank 1 and rank k are DISTINCT kernel-shape classes (op_shape_class)
        assert j["cases"] == ["loreft_both:affine_subspace:rank1:all:skipbos0:both",
                              "loreft_both:affine_subspace:rankK:all:skipbos0:both"]
        assert j["variants"]["loreft_both:affine_subspace:rank1:all:skipbos0:both"] == {"rank": 1}
        assert j["servable"] is False and j["decode_capable"] is True
        assert j["routing"] == [g.ROUTING_VECTORIZED] and "BOTH phases" in j["reason"]
        # both clamp_gate classes of the two gated lineages, x both ranks ...
        assert cov["lowrank_gated"]["cases"] == ["lowrank_gated:lowrank_linear_gated:rank1:all:skipbos0:both",
                                               "lowrank_gated:lowrank_linear_gated:rankK:all:skipbos0:both",
                                               "lowrank_gated:lowrank_relu_gated:rank1:all:skipbos0:both",
                                               "lowrank_gated:lowrank_relu_gated:rankK:all:skipbos0:both"]
        assert cov["lowrank_gated"]["variants"]["lowrank_gated:lowrank_linear_gated:rankK:all:skipbos0:both"] == {
            "clamp_gate": False, "rank": 4}
        assert cov["lowrank_gated"]["variants"]["lowrank_gated:lowrank_relu_gated:rank1:all:skipbos0:both"] == {
            "clamp_gate": True, "rank": 1}
        assert cov["alphakl"]["cases"] == ["alphakl:linear_gated_vector:gemv:all:skipbos0:both",
                                           "alphakl:relu_gated_vector:gemv:all:skipbos0:both"]
        assert cov["lowrank_gated"]["decode_capable"] is True and cov["alphakl"]["decode_capable"] is True
        # ... and static_r's native prefill-only modes PLUS its response-mode serving, which is
        # phase both and therefore a grouped signature the GPU gate cannot see
        sr = cov["static_r"]
        assert sr["cases"] == ["static_r:add_vector:vec:all:skipbos0:prefill",
                               "static_r:add_vector:vec:last:skipbos0:prefill",
                               "static_r:add_vector:vec:response:skipbos0:both"]
        assert sr["decode_capable_cases"] == ["static_r:add_vector:vec:response:skipbos0:both"]
        assert sr["decode_capable"] is True and sr["routing"] == [g.ROUTING_PER_SLOT, g.ROUTING_VECTORIZED]
        assert sr["variants"]["static_r:add_vector:vec:response:skipbos0:both"]["positions"] == "response"
        assert sr["variants"]["static_r:add_vector:vec:all:skipbos0:prefill"] == {
            "positions": None, "steer_positions": "prefill_all"}
        # the container arm: every served positions mode x skip_bos the adapter accepts
        # (`last`, `prompt` and `front` are refused by the adapter, like the psr / lora probes)
        assert cov["vector"]["cases"] == ["vector:add_vector:vec:all:skipbos0:both",
                                          "vector:add_vector:vec:all:skipbos1:both",
                                          "vector:add_vector:vec:response:skipbos0:both",
                                          "vector:add_vector:vec:response:skipbos1:both"]
        assert cov["vector"]["decode_capable"] is True
        # cosine_gated_r: the response override is REFUSED by its adapter (derivation locality)
        assert cov["cosine_gated_r"]["cases"] == [
            "cosine_gated_r:prefill_last_cosine_gated:other:last:skipbos0:prefill"]
        assert cov["cosine_gated_r"]["decode_capable"] is False
        assert all(k in cov for k in g.NON_PETRI_KINDS)
        # the unseen decode-capable SIGNATURES are exactly the grouped classes the GPU gate cannot see
        unseen = sorted(c for k, v in cov.items() if v["servable"] is False for c in v["decode_capable_cases"])
        assert len(unseen) == 13
        assert unseen == [
            "alphakl:linear_gated_vector:gemv:all:skipbos0:both",
            "alphakl:relu_gated_vector:gemv:all:skipbos0:both",
            "loreft_both:affine_subspace:rank1:all:skipbos0:both",
            "loreft_both:affine_subspace:rankK:all:skipbos0:both",
            "lowrank_gated:lowrank_linear_gated:rank1:all:skipbos0:both",
            "lowrank_gated:lowrank_linear_gated:rankK:all:skipbos0:both",
            "lowrank_gated:lowrank_relu_gated:rank1:all:skipbos0:both",
            "lowrank_gated:lowrank_relu_gated:rankK:all:skipbos0:both",
            "static_r:add_vector:vec:response:skipbos0:both",
            "vector:add_vector:vec:all:skipbos0:both", "vector:add_vector:vec:all:skipbos1:both",
            "vector:add_vector:vec:response:skipbos0:both", "vector:add_vector:vec:response:skipbos1:both"]

    def test_variant_axes_come_from_the_adapters_and_refusals_are_not_cases(self, tmp_path):
        from sisteer.hooks import POSITION_MODES
        assert g.kind_variant_axes("static_r") == {"steer_positions": ("prompt_last", "prefill_all"),
                                                   "positions": (None, "response")}
        assert g.kind_variant_axes("alphakl") == {"clamp_gate": (True, False)}
        assert g.kind_variant_axes("lowrank_gated") == {"clamp_gate": (True, False), "rank": (1, 4)}
        assert g.kind_variant_axes("loreft_both") == {"rank": (1, 4)}
        assert g.kind_variant_axes("vector") == {"positions": (None,) + tuple(POSITION_MODES),
                                                 "skip_bos": (True, False)}
        assert "positions" in g.KIND_SPEC_AXES and "rank" not in g.KIND_SPEC_AXES
        probed = g.probe_kind_variants("static_r", len(HEADER), tmp_path)
        assert sorted(c for c, _, _ in probed) == ["static_r:add_vector:vec:all:skipbos0:prefill",
                                                    "static_r:add_vector:vec:last:skipbos0:prefill",
                                                    "static_r:add_vector:vec:response:skipbos0:both"]
        resp = next(ops for c, _, ops in probed if c.endswith(":response:skipbos0:both"))
        assert all(o.phase == la.PHASE_BOTH and o.positions == la.POSITIONS_RESPONSE for o in resp)
        assert lw.ops_vectorize_in_decode(resp) is True
        # the spec override is a RESOLVED-SPEC key, never an artifact field
        d = g.write_synthetic_kind("static_r", g.INVENTORY_DIMS, tmp_path / "s", len(HEADER),
                                   steer_positions="prompt_last")
        assert "positions" not in torch.load(d / "object.pt", weights_only=True)
        with pytest.raises(la.AdapterError, match="derivation locality|DERIVATION LOCALITY"):
            g.adapt_kind_dir(g.write_synthetic_kind("cosine_gated_r", g.INVENTORY_DIMS, tmp_path / "c",
                                                    len(HEADER)),
                             kind="cosine_gated_r", header_len=len(HEADER),
                             spec_overrides={"positions": "response"})
        # a variant the adapter refuses (non-served position modes) is recorded nowhere
        assert not any(":prompt:" in c or ":front:" in c or ":last:" in c
                       for c, _, _ in g.probe_kind_variants("vector", len(HEADER), tmp_path / "v"))

    def test_a_kind_with_zero_cases_refuses(self, monkeypatch, tmp_path):
        # a Petri-armed kind the served inventory does not file
        real_inv = g.served_inventory(len(HEADER))
        monkeypatch.setattr(g, "served_inventory",
                            lambda header_len=4: {k: v for k, v in real_inv.items() if v["kind"] != "psr"})
        with pytest.raises(g.GateError, match="Petri-armed lens kind 'psr' .* ZERO inventory cases"):
            g.kind_coverage(len(HEADER))
        monkeypatch.setattr(g, "served_inventory", lambda header_len=4: real_inv)
        # a non-Petri kind whose every variant the adapter refuses
        real_adapt = g.adapt_kind_dir

        def refusing(obj_dir, *, kind, header_len, spec_overrides=None):
            if kind == "loreft_both":
                raise la.AdapterError("synthetic refusal")
            return real_adapt(obj_dir, kind=kind, header_len=header_len, spec_overrides=spec_overrides)
        monkeypatch.setattr(g, "adapt_kind_dir", refusing)
        with pytest.raises(g.GateError, match="non-Petri lens kind 'loreft_both': the adapter refused EVERY"):
            g.kind_coverage(len(HEADER))

    def test_a_registered_kind_with_neither_arm_nor_entry_refuses(self, monkeypatch):
        real = la.registered_kinds()
        monkeypatch.setattr(la, "registered_kinds", lambda: real + ("mystery_kind",))
        with pytest.raises(g.GateError, match="neither a Petri arm .* nor a NON_PETRI_KINDS entry"):
            g.kind_coverage(len(HEADER))

    @pytest.mark.parametrize("rank,shape", [(4, "rankK"), (1, "rank1")])
    def test_loreft_both_mint_adapts_to_the_both_phase_affine_op(self, tmp_path, rank, shape, monkeypatch):
        dims = {"hidden_size": 16, "num_hidden_layers": 6}
        d = g.write_synthetic_kind("loreft_both", dims, tmp_path / "j", len(HEADER), rank=rank)
        ops = g.adapt_kind_dir(d, kind="loreft_both", header_len=len(HEADER))
        assert [o.layer for o in ops] == [2, 3]
        for o in ops:
            assert o.op == la.OP_AFFINE_SUBSPACE and o.phase == la.PHASE_BOTH
            assert o.rotate.shape == (rank, 16) and o.affine_w.shape == (rank, 16)
            assert o.affine_b.shape == (rank,)
        assert lw.ops_vectorize_in_decode(ops) is True
        cid = g.case_id_of("loreft_both", ops)
        assert cid == f"loreft_both:affine_subspace:{shape}:all:skipbos0:both"   # the SAME op class
        assert g.case_routing(cid) == g.ROUTING_VECTORIZED
        # its decode rows group; under the PER-ROW mode the group's reductions run per row and
        # equal the single-row calls exactly, under the STACKED default (one [3, K] call) they
        # agree to fp32 rounding — the bf16 deviation is the GPU gate's contract, not a CPU fact
        rows = torch.randn(3, 16)
        singles = [la.apply_hook_ops(rows[i:i + 1], ops, 1.0, layer=2,
                                     ctx=la.HookContext(la.PHASE_DECODE, (7,), token_ids=(42,)))
                   for i in range(3)]
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", False)
        grouped = la.apply_hook_ops(rows, ops, 1.0, layer=2, ctx=la.DecodeRows())
        for i in range(3):
            assert torch.equal(grouped[i:i + 1], singles[i])
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", True)
        stacked = la.apply_hook_ops(rows, ops, 1.0, layer=2, ctx=la.DecodeRows())
        for i in range(3):
            assert torch.allclose(stacked[i:i + 1], singles[i], rtol=1e-5, atol=1e-6)
        # lowrank_gated's rank-1 mint is the rank1 class of ITS op too
        d2 = g.write_synthetic_kind("lowrank_gated", dims, tmp_path / "l", len(HEADER), rank=rank,
                                    clamp_gate=False)
        ops2 = g.adapt_kind_dir(d2, kind="lowrank_gated", header_len=len(HEADER))
        assert g.case_id_of("lowrank_gated", ops2) == f"lowrank_gated:lowrank_linear_gated:{shape}:all:skipbos0:both"
        assert all(o.lowrank_a.shape == (rank, 16) and o.lowrank_b.shape == (16, rank) for o in ops2)
        with pytest.raises(g.GateError, match="no synthetic writer"):
            g.write_synthetic_kind("lofit", dims, tmp_path / "x", len(HEADER))

    def test_report_lists_the_unseen_classes(self):
        ms = g.MODELS["llama8b"]
        doc = {"model": dict(key=ms.key, hf=ms.hf, rev=ms.rev, cfg=ms.cfg, bank=ms.bank, branch=ms.branch),
               "geometry": {"max_model_len": 4096, "max_num_seqs": 64}, "batches": [1],
               "timing_batches": [8], "cases": [], "inventory": {}, "uncovered_cases": [],
               "complete": True, "batch_invariant": 0, "exit_code": 0,
               "kind_coverage": {"psr": {"servable": True, "petri_arms": ["psr_kl_all"], "cases": ["x"],
                                         "decode_capable_cases": ["x"],
                                         "decode_capable": True, "routing": ["vectorized"], "reason": None},
                                 "loreft_both": {"servable": False, "petri_arms": [],
                                                   "cases": ["loreft_both:affine_subspace:rankK:all:skipbos0:both"],
                                                   "decode_capable_cases": ["loreft_both:affine_subspace:rankK:all:skipbos0:both"],
                                                   "decode_capable": True, "routing": ["vectorized"],
                                                   "reason": "the earlier development branch's lineage"},
                                 "static_r": {"servable": False, "petri_arms": [],
                                              "cases": ["static_r:add_vector:vec:last:skipbos0:prefill",
                                                        "static_r:add_vector:vec:response:skipbos0:both"],
                                              "decode_capable_cases": ["static_r:add_vector:vec:response:skipbos0:both"],
                                              "decode_capable": True, "routing": ["per_slot", "vectorized"],
                                              "reason": "caa_static_r"}}}
        text = g.render_report(doc)
        assert "2 decode-capable signature(s) of 2 kind(s) ['loreft_both', 'static_r'] have NO Petri-servable cell" in text
        assert "- UNSEEN grouped signature: loreft_both:affine_subspace:rankK:all:skipbos0:both" in text
        assert "- UNSEEN grouped signature: static_r:add_vector:vec:response:skipbos0:both" in text
        assert "UNSEEN grouped signature: static_r:add_vector:vec:last" not in text   # prefill-only: not a grouped class
        assert "UNSEEN kind loreft_both: cases ['loreft_both:affine_subspace:rankK:all:skipbos0:both']" in text
        assert "UNSEEN kind psr" not in text


# --------------------------------------------------------------------------
# F1 — every worker RPC read goes through unwrap_rpc (the 378517 false negative)
# --------------------------------------------------------------------------

#: the candidate's co-scheduling reply, VERBATIM from
#: lens-vectorize-gate_378517_2.out (qwen35_4b, 2026-09-04): the per-worker
#: list under "results" — 63 of 64 forwards held a same-render pair of
#: distinct cells, yet the gate read the fields off the wrapper and logged
#: "not co-scheduled" three times.
PRODUCTION_COSCHEDULE_PAYLOAD = {'results': [{'note': 'co-scheduling evidence: per classified forward, the distinct NONZERO vectorized groups (cell, expect_hash, scale, row context) that shared it; observation only, one per forward on the real view', 'n_forwards': 64, 'n_forwards_with_groups': 63, 'max_distinct_nonzero_groups_in_one_forward': 5, 'max_distinct_nonzero_cells_in_one_forward': 3, 'n_forwards_with_ge2_groups': 63, 'n_forwards_with_ge2_cells': 63, 'n_forwards_with_same_render_pair': 63, 'n_record_errors': 0, 'first_record_error': None}]}


class TestRpcUnwrap:
    def test_production_shape_attests(self):
        recs = g.unwrap_rpc(PRODUCTION_COSCHEDULE_PAYLOAD)
        assert len(recs) == 1 and recs[0]["n_forwards_with_same_render_pair"] == 63
        rec = g.single_worker(recs, "sisteer_coschedule_record")
        assert g.coschedule_attested(rec) is True
        # the pre-fix read: the wrapper itself -> None -> "not attested"
        assert g.coschedule_attested(PRODUCTION_COSCHEDULE_PAYLOAD) is False or \
            g.coschedule_attested(PRODUCTION_COSCHEDULE_PAYLOAD) is None
        # re-derive the multigroup verdict from it: one wave, attested
        waves = [{"wave": 1, "repeat": 1, "attempt": 1, "attested": g.coschedule_attested(rec),
                  "accepted": True, "record": rec}]
        agg = g.aggregate_coschedule(waves)
        assert agg["attested_waves"] == [1] and g.coschedule_attested(agg) is True
        case = _mg_case()
        case["coschedule"][g.CANDIDATE] = agg
        case["coschedule_waves"][g.CANDIDATE] = waves
        case["coschedule_attested"] = True
        checks = g.multigroup_checks(case, g.multigroup_spec(32), case["cells"], reference_sha=BASE,
                                     base_sha=BASE, have_oracle=True)
        assert checks["coschedule_candidate"]["ok"] is True
        case["checks"] = checks
        assert g.overall_exit_code({"cases": [case], "complete": True, "batch_invariant": 1}) == 0

    def test_shapes(self):
        assert g.unwrap_rpc([{"a": 1}, {"a": 1}]) == [{"a": 1}, {"a": 1}]      # bare list
        assert g.unwrap_rpc({"a": 1}) == [{"a": 1}]                          # bare dict = one worker
        assert g.unwrap_rpc({"results": ["vectorized"]}) == ["vectorized"]
        for bad in ("vectorized", 3, None, {"results": "vectorized"}, {"results": None}):
            with pytest.raises(g.GateError, match="collective_rpc payload"):
                g.unwrap_rpc(bad)
        assert g.single_worker(["per_row", "per_row"], "x") == "per_row"
        with pytest.raises(g.GateError, match="disagree"):
            g.single_worker(["per_row", "vectorized"], "sisteer_apply_path")
        with pytest.raises(g.GateError, match="no worker results"):
            g.single_worker([], "x")

    def test_every_read_uses_the_worker_list(self, monkeypatch, tmp_path):
        """apply_path_of / coschedule_snapshot / selection_evidence over the
        production wrapper shape, through a stubbed rpc()."""
        su = g.Standup(tag="t", side=g.CANDIDATE, tree=REPO, out=tmp_path, proc=None,
                       stop_file=tmp_path / "STOP", endpoint={"proxy_url": "x", "server_base_url": "y"},
                       api_key=None)
        record = {"n_observations": 320, "all_observations_matched": True,
                  "requests": {"r0": {"requested": "si_safety__psr_kl_all", "n_applications": 32,
                                      "n_applications_modifying": 32, "applied_effective_scale": 1.0},
                               "r1": {"requested": "none__lens0", "n_applications": 32,
                                      "n_applications_modifying": 0, "applied_effective_scale": 0.0}}}
        replies = {"sisteer_apply_path": {"results": ["vectorized"]},
                   "sisteer_coschedule_record": PRODUCTION_COSCHEDULE_PAYLOAD,
                   "sisteer_selection_record": {"results": [record]},
                   "sisteer_shadow_compare": {"results": [_shadow_snapshot()]}}
        monkeypatch.setattr(g, "rpc", lambda su_, method, args=(): replies[method])
        assert g.apply_path_of(su) == "vectorized"
        assert g.coschedule_attested(g.coschedule_snapshot(su, reset=False)) is True
        assert g.shadow_check(g.shadow_snapshot(su, reset=True), (1, 8, 32), routing=g.ROUTING_VECTORIZED,
                              waves=_shadow_waves())[0] is True
        ev = g.selection_evidence(su)
        assert ev["n_modifying"] == 32 and ev["n_observations"] == 320 and ev["all_observations_matched"]
        assert ev["by_cell"]["si_safety__psr_kl_all"]["n_modifying"] == 32
        assert ev["by_cell"]["none__lens0"]["n_modifying"] == 0
        # an absent RPC (the base commit) is still the recorded absence
        def absent(su_, method, args=()):
            raise g.RpcError(500, "AttributeError: sisteer_coschedule_record")
        monkeypatch.setattr(g, "rpc", absent)
        assert g.apply_path_of(su) == {"rpc": "absent", "status": 500,
                                       "detail": "AttributeError: sisteer_coschedule_record"}
        assert g.coschedule_snapshot(su, reset=True)["rpc"] == "absent"
        assert g.shadow_snapshot(su, reset=True)["rpc"] == "absent"
        # a malformed worker record refuses instead of reading as zero evidence
        monkeypatch.setattr(g, "rpc", lambda su_, method, args=(): {"results": ["garbage"]})
        with pytest.raises(g.GateError, match="not a record"):
            g.selection_evidence(su)


# --------------------------------------------------------------------------
# F2 — batch invariance on both standups
# --------------------------------------------------------------------------


class TestBatchInvariance:
    def test_standup_env(self):
        base = {"PATH": "/bin", "PYTHONPATH": "/cand/src:/cand/scripts",
                g.PER_ROW_ENV: "1", g.SHADOW_ENV: "1", g.BATCH_INVARIANT_ENV: "1", "HF_HOME": "/hf"}
        env = g.standup_env(base, Path("/ref"), batch_invariant=0)
        assert env["PYTHONPATH"] == "/ref/src:/ref/scripts"          # only the target tree
        assert g.PER_ROW_ENV not in env and g.SHADOW_ENV not in env
        assert g.BATCH_INVARIANT_ENV not in env                      # production: unset
        assert env["HF_HOME"] == "/hf" and env["PATH"] == "/bin"
        shadow = g.standup_env(base, Path("/cand"), batch_invariant=0, env_extra={g.SHADOW_ENV: "1"})
        assert shadow[g.SHADOW_ENV] == "1" and g.PER_ROW_ENV not in shadow
        exp = g.standup_env(base, Path("/cand"), batch_invariant=1)  # the opt-in experiment
        assert exp[g.BATCH_INVARIANT_ENV] == "1"
        # the reduction-mode switch: scrubbed when no mode is given (the base-commit reference),
        # SET explicitly — never inherited — when one is
        base[g.STACKED_ENV] = "1"
        assert g.STACKED_ENV not in g.standup_env(base, Path("/ref"), batch_invariant=0)
        assert g.standup_env(base, Path("/cand"), batch_invariant=0, reduction_mode="per_row")[g.STACKED_ENV] == "0"
        assert g.standup_env(base, Path("/cand"), batch_invariant=0, reduction_mode="stacked")[g.STACKED_ENV] == "1"

    def test_bi_attestation(self):
        good0 = {"batch_invariant_env": False, "batch_invariant_active": False,
                 "batch_invariant_surface": "_batch_invariant_MODE (vllm 0.25.1)",
                 "batch_invariant_error": None}
        ok, why = g.bi_attestation_ok([good0], 0, allow_absent=False)
        assert ok and "_batch_invariant_MODE (vllm 0.25.1)" in why
        ok, why = g.bi_attestation_ok([{"batch_invariant_env": True, "batch_invariant_active": True}], 0,
                                      allow_absent=False)
        assert not ok and "intent BI=0" in why
        ok, why = g.bi_attestation_ok([{"batch_invariant_env": True, "batch_invariant_active": False}], 1,
                                      allow_absent=False)
        assert not ok and "intent BI=1" in why
        ok, why = g.bi_attestation_ok([{"batch_invariant_env": False, "batch_invariant_active": True}], 0,
                                      allow_absent=False)
        assert not ok and "intent BI=0" in why                       # env and active must BOTH agree
        ok, _ = g.bi_attestation_ok([{"batch_invariant_env": True, "batch_invariant_active": True}], 1,
                                    allow_absent=False)
        assert ok
        # the base commit's workers carry no fields: accepted only as the reference at BI=0
        assert g.bi_attestation_ok([{}], 0, allow_absent=True)[0]
        assert not g.bi_attestation_ok([{}], 0, allow_absent=False)[0]
        assert not g.bi_attestation_ok([{}], 1, allow_absent=True)[0]
        # NO worker records at all is a broken attestation, not an absent field
        assert not g.bi_attestation_ok([], 0, allow_absent=True)[0]
        assert not g.bi_attestation_ok([], 1, allow_absent=True)[0]
        assert not g.bi_attestation_ok(None, 0, allow_absent=True)[0]
        assert not g.bi_attestation_ok(["garbage"], 0, allow_absent=True)[0]
        # a mixed set (one attests, one does not) is not accepted
        assert not g.bi_attestation_ok([good0, {}], 0, allow_absent=True)[0]

    def test_unreadable_bi_state_is_refused_under_both_intents(self):
        """A report whose runtime read FAILED (``batch_invariant_error`` set,
        ``active`` None) used to satisfy BI=0 because ``(False, None)`` is
        falsy. Now any error, missing field or non-boolean value refuses — for
        intent 0 AND 1, allow_absent or not."""
        errored = {"batch_invariant_env": False, "batch_invariant_active": None,
                   "batch_invariant_surface": None,
                   "batch_invariant_error": "active: LensWorkerError: ... unverified private flag"}
        for intent in (0, 1):
            for allow in (False, True):
                ok, why = g.bi_attestation_ok([errored], intent, allow_absent=allow)
                assert not ok and "could not be read" in why, (intent, allow)
        env_err = {"batch_invariant_env": None, "batch_invariant_active": None,
                   "batch_invariant_error": "env: ModuleNotFoundError: No module named 'vllm'"}
        assert not g.bi_attestation_ok([env_err], 0, allow_absent=False)[0]
        # None / missing active with NO error string: still not a state
        ok, why = g.bi_attestation_ok([{"batch_invariant_env": False, "batch_invariant_active": None,
                                        "batch_invariant_error": None}], 0, allow_absent=False)
        assert not ok and "non-boolean" in why
        ok, why = g.bi_attestation_ok([{"batch_invariant_env": False}], 0, allow_absent=False)
        assert not ok and "non-boolean" in why
        ok, why = g.bi_attestation_ok([{"batch_invariant_env": 0, "batch_invariant_active": 0}], 0,
                                      allow_absent=False)
        assert not ok and "non-boolean" in why                       # ints are not the worker's bools
        # the helper could not read the attestation file at all: refused even for the reference
        ok, why = g.bi_attestation_ok([{"error": "FileNotFoundError: steering_attestation.json"}], 0,
                                      allow_absent=True)
        assert not ok and "could not be read" in why

    def test_pinned_surface_report_feeds_the_attestation(self):
        """End to end on the CPU: the worker's report off a module shaped like
        vllm 0.25.1's batch_invariant (no public reader; the private flag
        settled by init_batch_invariance) attests BI=0 for the candidate; the
        same module at an unverified version yields an error that the gate
        REFUSES for the candidate and for the reference alike."""
        from test_lens_vectorized_apply import Vllm0251BatchInvariant
        envs = types.SimpleNamespace(VLLM_BATCH_INVARIANT=False)
        bi = Vllm0251BatchInvariant(envs)
        bi.init_batch_invariance()
        rep = lw.batch_invariant_report(envs, bi, version="0.25.1")
        assert g.bi_attestation_ok([rep], 0, allow_absent=False)[0]
        assert not g.bi_attestation_ok([rep], 1, allow_absent=False)[0]
        rep_bad = lw.batch_invariant_report(envs, bi, version="0.26.0")
        assert rep_bad["batch_invariant_error"] and rep_bad["batch_invariant_active"] is None
        assert not g.bi_attestation_ok([rep_bad], 0, allow_absent=False)[0]   # candidate / shadow
        assert not g.bi_attestation_ok([rep_bad], 0, allow_absent=True)[0]    # even the reference
        assert not g.bi_attestation_ok([rep_bad], 1, allow_absent=False)[0]
        # and a case carrying such a refusal exits 2 (start_standup raises before any request;
        # case_checks would fail it too)
        bi_att = dict(_ok_case()["bi_attested"])
        bi_att[g.SHADOW] = dict(zip(("ok", "detail"), g.bi_attestation_ok([rep_bad], 0, allow_absent=False)))
        code, checks = _exit_with(_ok_case(), bi_attested=bi_att)
        assert code == 2 and not checks["batch_invariant_candidate_shadow"]["ok"]

    def test_batch_invariant_ok(self):
        assert g.batch_invariant_ok("1", 1) and not g.batch_invariant_ok(None, 1)
        assert not g.batch_invariant_ok("0", 1)
        assert g.batch_invariant_ok(None, 0) and g.batch_invariant_ok("0", 0)
        assert not g.batch_invariant_ok("1", 0)

    def test_unstable_reference_is_a_failure_under_bi1_only(self):
        ok = _ok_case()
        ok["checks"] = g.case_checks(ok, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        bad = dict(ok, verdicts={"pure_B32": {"verdict": g.UNSTABLE_REF}})
        assert g.overall_exit_code({"cases": [bad], "complete": True, "batch_invariant": 1}) == 2
        assert g.overall_exit_code({"cases": [bad], "complete": True, "batch_invariant": 0}) == 3
        assert g.overall_exit_code({"cases": [bad], "complete": True}) == 3
        assert g.overall_exit_code({"cases": [ok], "complete": True, "batch_invariant": 1}) == 0

    def test_report_and_sbatch_name_the_regime(self):
        text = (REPO / "scripts/sbatch/lens_vectorize_gate.sbatch").read_text()
        assert '--batch-invariant "${GATE_BATCH_INVARIANT:-0}"' in text
        assert "unset VLLM_BATCH_INVARIANT SISTEER_LENS_SHADOW_COMPARE" in text
        ms = g.MODELS["llama8b"]
        doc = {"model": dict(key=ms.key, hf=ms.hf, rev=ms.rev, cfg=ms.cfg, bank=ms.bank, branch=ms.branch),
               "geometry": {"max_model_len": 4096, "max_num_seqs": 64}, "batches": [1],
               "timing_batches": [8], "cases": [], "inventory": {}, "uncovered_cases": [],
               "complete": True, "batch_invariant": 0, "exit_code": 0}
        text = g.render_report(doc)
        assert "VLLM_BATCH_INVARIANT=0 on every standup" in text and "shadow compare" in text
        assert "cand+shadow ms/step" in text


# --------------------------------------------------------------------------
# N1 — the multigroup case
# --------------------------------------------------------------------------


def _mg_case(cells=("si_constitution__psr_kl_all", "si_constitution__lora_r4",
                    "safety_refusal__psr_kl_all")):
    by_cell = {c: {"n_modifying": 640, "n_applications": 640, "n_requests": 8, "scales": [1.0]}
               for c in cells}
    by_cell["none__lens0"] = {"n_modifying": 0, "n_applications": 320, "n_requests": 8, "scales": [0.0]}
    ev = {"n_modifying": 1920, "n_observations": 2240, "all_observations_matched": True,
          "by_cell": by_cell}
    facts = {"control_rows_match": True, "bites_by_cell": {c: True for c in cells},
             "cross_cell_distinct": True, "cross_cell_pairs": 6}
    attested = {"n_forwards": 70, "n_forwards_with_groups": 70,
                "max_distinct_nonzero_groups_in_one_forward": 5,
                "max_distinct_nonzero_cells_in_one_forward": 3,
                "n_forwards_with_ge2_groups": 70, "n_forwards_with_ge2_cells": 70,
                "n_forwards_with_same_render_pair": 70, "n_record_errors": 0,
                "first_record_error": None}
    unattested = dict(attested, n_forwards_with_same_render_pair=0, n_forwards_with_ge2_groups=0,
                      n_forwards_with_ge2_cells=0, max_distinct_nonzero_groups_in_one_forward=1,
                      max_distinct_nonzero_cells_in_one_forward=1)
    waves = [{"wave": 1, "repeat": 1, "attempt": 1, "attested": True, "accepted": True,
              "record": attested},
             {"wave": 2, "repeat": 2, "attempt": 1, "attested": False, "accepted": False,
              "record": unattested},
             {"wave": 3, "repeat": 2, "attempt": 2, "attested": True, "accepted": True,
              "record": attested}]
    mg_shadow = dict(_shadow_snapshot(batches=(32,)), n_layer_calls_with_ge2_cells_compared=64,
                     max_distinct_cells_compared_in_one_layer_call=3,
                     cells_compared=list(cells))
    ev_cand = dict(json.loads(json.dumps(ev)), lens_reduction_mode="stacked", lens_reduction_modes_seen=["stacked"])
    return {"case_id": g.MULTIGROUP, "cells": list(cells),
            "shadow": mg_shadow,
            "controls_B1": {g.REFERENCE: {"controls_match": True, "steered_differs": True,
                                          "cells_distinct": True},
                            g.CANDIDATE: {"controls_match": True, "steered_differs": True,
                                          "cells_distinct": True}},
            "bi_attested": {g.REFERENCE: {"ok": True, "detail": "absent accepted"},
                            g.CANDIDATE: _bi_ok(), g.SHADOW: _bi_ok()},
            "coschedule": {g.REFERENCE: {"rpc": "absent"},
                           g.CANDIDATE: g.aggregate_coschedule(waves)},
            "coschedule_waves": {g.REFERENCE: [{"wave": 1, "attested": None,
                                                "record": {"rpc": "absent"}}],
                                 g.CANDIDATE: waves},
            "coschedule_attested": True,
            "coschedule_attempts": 3,
            "apply_path": {g.REFERENCE: {"rpc": "absent"}, g.CANDIDATE: g.APPLY_PATH_VECTORIZED_STACKED,
                           g.SHADOW: g.APPLY_PATH_SHADOW},
            "steering_evidence": {g.REFERENCE: json.loads(json.dumps(ev)),
                                  g.CANDIDATE: json.loads(json.dumps(ev_cand)),
                                  g.SHADOW: json.loads(json.dumps(ev_cand))},
            "row_facts": {g.REFERENCE: json.loads(json.dumps(facts)),
                          g.CANDIDATE: json.loads(json.dumps(facts))},
            "signature_errors": [],
            "verdicts": {"multigroup_B32": {"verdict": g.EVIDENCE, "evidence": True, "equal": True,
                                            "underlying": g.PASS}}}


class TestMultigroup:
    def test_spec_interleaves_every_role_and_repeats_prompts(self):
        spec = g.multigroup_spec(32)
        roles = [(r["role"], r["seeded"]) for r in spec]
        assert ("passthrough", False) in roles and ("scale0", False) in roles
        assert ("cellA", False) in roles and ("cellB", False) in roles and ("cellC", False) in roles
        assert ("cellA", True) in roles and ("cellB", True) in roles and ("scale0", True) in roles
        assert [r["prompt"] for r in spec] == [i % 16 for i in range(32)]
        # every prompt appears twice, under DIFFERENT roles; every nonzero
        # cell's row has a same-prompt, same-render twin under ANOTHER cell
        by_prompt = {}
        for r in spec:
            by_prompt.setdefault(r["prompt"], []).append((r["role"], r["seeded"]))
        assert all(len(v) == 2 and v[0] != v[1] for v in by_prompt.values())
        cross = [(a, b) for a, b in by_prompt.values()
                 if a[0].startswith("cell") and b[0].startswith("cell") and a[1] == b[1]]
        assert len(cross) >= 8 and all(a[0] != b[0] for a, b in cross)
        assert all(g.MG_TWIN_ROLE[r] != r for r in g.MG_ROLES)
        assert g.multigroup_spec(1) == [{"i": 0, "role": "passthrough", "seeded": False, "prompt": 0}]

    def test_roles_resolve_to_cells_with_a_two_cell_fallback(self):
        cells = ["c1", "c2"]
        assert g.resolve_mg_role("cellA", cells) == "c1" and g.resolve_mg_role("cellB", cells) == "c2"
        assert g.resolve_mg_role("cellC", cells) == "c1"          # no third cell
        assert g.resolve_mg_role("cellC", ["c1", "c2", "c3"]) == "c3"
        assert g.resolve_mg_role("scale0", cells) == "scale0"

    def test_seeded_requests_declare_only_on_lens_routes(self):
        msgs = [{"role": "user", "content": "hi"}]
        m, ex = g.seeded_request(msgs, lens_route=True)
        assert m[-1] == {"role": "assistant", "content": g.SEED_TEXT}
        assert ex == {"continue_final_message": True, "add_generation_prompt": False,
                      "petri_response_prefill": True}
        m, ex = g.seeded_request(msgs, lens_route=False)
        assert "petri_response_prefill" not in ex and ex["continue_final_message"] is True
        spec = g.multigroup_spec(16)
        prompts = [([{"role": "user", "content": f"p{i}"}], 5) for i in range(8)]
        vm = {"passthrough": "petri__base", "scale0": "petri__none__lens0",
              "cA": "petri__cA", "cB": "petri__cB"}
        rows = g.spec_requests(spec, prompts, ["cA", "cB"], vm)
        # first half: the eight roles in order (cellC -> cA with two cells);
        # second half: each prompt's twin role
        assert [r[0] for r in rows[:8]] == ["petri__base", "petri__none__lens0", "petri__cA",
                                            "petri__cB", "petri__cA", "petri__cA", "petri__cB",
                                            "petri__none__lens0"]
        assert [r[0] for r in rows[8:]] == ["petri__cA", "petri__cB", "petri__cB", "petri__cA",
                                            "petri__cA", "petri__cB", "petri__cA", "petri__cA"]
        seeded_rows = [r for r, sp in zip(rows, spec) if sp["seeded"]]
        assert len(seeded_rows) == 6
        assert all(r[2].get("petri_response_prefill") is True for r in seeded_rows)   # lens routes
        assert all(r[1][-1] == {"role": "assistant", "content": g.SEED_TEXT} for r in seeded_rows)
        assert all("continue_final_message" not in r[2]
                   for r, sp in zip(rows, spec) if not sp["seeded"])
        oracle = g.spec_requests(spec, prompts, (), vm, all_passthrough=True)
        assert all(r[0] == "petri__base" for r in oracle)
        assert all("petri_response_prefill" not in r[2] for r in oracle)
        assert [r[1] for r in oracle] == [r[1] for r in rows]                      # same renders
        with pytest.raises(g.GateError, match="no virtual model"):
            g.spec_requests(spec, prompts, ["cA", "missing"], vm)

    def test_row_facts_control_bites_and_cross_cell(self):
        spec = g.multigroup_spec(8)   # rows 0-3: pt, s0, A, B; rows 4-7 (twins): A, B, B, C->A
        cells = ["cA", "cB"]
        oracle = [_sig("ab", [-0.1 * (i + 1), -0.2]) for i in range(8)]
        mixed = [dict(s) for s in oracle]
        for r in spec:
            c = g.resolve_mg_role(r["role"], cells)
            if c == "cA":
                mixed[r["i"]] = _sig("ab", [-0.5, -0.2 - 0.01 * r["i"]])
            elif c == "cB":
                mixed[r["i"]] = _sig("ab", [-0.7, -0.2 - 0.01 * r["i"]])
        facts = g.multigroup_row_facts(spec, cells, mixed, oracle)
        assert facts["control_rows_match"] is True
        assert facts["bites_by_cell"] == {"cA": True, "cB": True}
        assert facts["cross_cell_distinct"] is True and facts["cross_cell_pairs"] >= 1
        # cell B's op applied to cell A's rows: same-prompt rows identical -> not distinct
        merged = [dict(s) for s in mixed]
        for r in spec:
            if g.resolve_mg_role(r["role"], cells) == "cA" and not r["seeded"]:
                twin = next(q for q in spec if q["prompt"] == r["prompt"] and q["i"] != r["i"])
                if g.resolve_mg_role(twin["role"], cells) == "cB":
                    merged[r["i"]] = dict(mixed[twin["i"]])
        facts = g.multigroup_row_facts(spec, cells, merged, oracle)
        assert facts["cross_cell_distinct"] is False and "identical" in facts["cross_cell_detail"]
        # a moved scale-0 row / an unsteered cell
        moved = [dict(s) for s in mixed]
        moved[1] = _sig("ab", [-0.3, -0.9])
        assert g.multigroup_row_facts(spec, cells, moved, oracle)["control_rows_match"] is False
        dead = [dict(s) for s in mixed]
        for r in spec:
            if g.resolve_mg_role(r["role"], cells) == "cB":
                dead[r["i"]] = dict(oracle[r["i"]])
        assert g.multigroup_row_facts(spec, cells, dead, oracle)["bites_by_cell"]["cB"] is False
        assert g.multigroup_row_facts(spec, cells, mixed[:4], oracle)["control_rows_match"] is None

    def _checks(self, case, **over):
        case = json.loads(json.dumps(case))
        case.update(over)
        return g.multigroup_checks(case, g.multigroup_spec(32), case["cells"], reference_sha=BASE,
                                   base_sha=BASE, have_oracle=True)

    def test_multigroup_all_ok_is_green(self):
        case = _mg_case()
        checks = self._checks(case)
        assert all(c["ok"] for c in checks.values()), checks
        case["checks"] = checks
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 0

    def test_multigroup_evidence_per_cell(self):
        case = _mg_case()
        ev = case["steering_evidence"][g.CANDIDATE]
        ev["by_cell"]["si_constitution__lora_r4"]["n_modifying"] = 0          # one group never applied
        checks = self._checks(case)
        assert not checks["evidence_candidate"]["ok"] and checks["evidence_reference"]["ok"]
        case = _mg_case()
        case["steering_evidence"][g.REFERENCE]["by_cell"]["none__lens0"]["n_modifying"] = 3   # twin steered
        assert not self._checks(case)["evidence_reference"]["ok"]
        case = _mg_case()
        del case["steering_evidence"][g.CANDIDATE]["by_cell"]["none__lens0"]                  # no twin rows seen
        assert not self._checks(case)["evidence_candidate"]["ok"]
        case = _mg_case()
        case["steering_evidence"][g.CANDIDATE]["all_observations_matched"] = False
        assert not self._checks(case)["evidence_candidate"]["ok"]

    def test_multigroup_row_facts_are_evidence_and_b1_controls_are_hard(self):
        # B=32 row facts (cross-cell, bites, control rows vs oracle) no longer gate
        case = _mg_case()
        case["row_facts"][g.CANDIDATE]["cross_cell_distinct"] = False
        case["row_facts"][g.REFERENCE]["control_rows_match"] = False
        checks = self._checks(case)
        assert "cross_cell_distinct_candidate" not in checks and "control_rows_reference" not in checks
        assert all(c["ok"] for c in checks.values()), checks
        # the B=1 singletons are hard: cells identical at B=1 / a moved scale-0 singleton
        case = _mg_case()
        case["controls_B1"][g.CANDIDATE] = {"controls_match": True, "steered_differs": True,
                                            "cells_distinct": False,
                                            "cells_detail": "identical B=1 outputs: a == b"}
        checks = self._checks(case)
        assert not checks["cells_B1_candidate"]["ok"] and "a == b" in checks["cells_B1_candidate"]["detail"]
        case = _mg_case()
        case["controls_B1"][g.REFERENCE] = {"controls_match": False, "controls_detail": "scale0: request 0",
                                            "steered_differs": True, "cells_distinct": True}
        assert not self._checks(case)["controls_B1_reference"]["ok"]
        case = _mg_case()
        checks = g.multigroup_checks(case, g.multigroup_spec(32), case["cells"], reference_sha=BASE,
                                     base_sha=BASE, have_oracle=False)
        assert not checks["controls_B1_reference"]["ok"] and not checks["cells_B1_candidate"]["ok"]
        for bad in ({g.REFERENCE: "vectorized", g.CANDIDATE: "vectorized", g.SHADOW: g.APPLY_PATH_SHADOW},
                    {g.REFERENCE: {"rpc": "absent"}, g.CANDIDATE: "per_row", g.SHADOW: g.APPLY_PATH_SHADOW},
                    {g.REFERENCE: {"rpc": "absent"}, g.CANDIDATE: "vectorized", g.SHADOW: "vectorized"}):
            case = _mg_case()
            case["apply_path"] = bad
            checks = self._checks(case)
            assert not (checks["reference_apply_path"]["ok"] and checks["candidate_apply_path"]["ok"]
                        and checks["shadow_apply_path"]["ok"])
        case = _mg_case()
        case["signature_errors"] = ["candidate: no token_ids"]
        assert not self._checks(case)["signatures"]["ok"]

    def test_multigroup_shadow_requirement(self):
        case = _mg_case()
        assert self._checks(case)["shadow_compare"]["ok"] is True
        case["checks"] = self._checks(case)
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 0
        # equal everywhere but never two cells in one layer call -> INCONCLUSIVE (exit 3)
        case = _mg_case()
        case["shadow"] = dict(case["shadow"], n_layer_calls_with_ge2_cells_compared=0,
                              max_distinct_cells_compared_in_one_layer_call=1)
        checks = self._checks(case)
        assert checks["shadow_compare"]["ok"] is None and ">= 2 distinct cells" in checks["shadow_compare"]["detail"]
        case["checks"] = checks
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 3
        # one unequal group attributed to no key (and without the tolerance fields) -> FAIL
        case = _mg_case()
        case["shadow"] = dict(case["shadow"], n_unequal=1,
                              first_unequal={"key": "k", "layer": 3, "cell": "c", "max_abs_diff": 0.5})
        checks = self._checks(case)
        assert checks["shadow_compare"]["ok"] is False
        assert "attributed to no key" in checks["shadow_compare"]["detail"]
        case["checks"] = checks
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 2
        assert g.shadow_multigroup_check({"rpc": "absent"})[0] is False
        assert g.shadow_multigroup_check(dict(_mg_case()["shadow"], enabled=False))[0] is False

    def test_coschedule_attestation_drives_the_verdict(self):
        """N1 item 1: outputs agreeing is not enough — the CANDIDATE worker
        must attest a same-render pair of distinct cells in one forward."""
        rec = _mg_case()["coschedule"][g.CANDIDATE]
        assert g.coschedule_attested(rec) is True
        assert g.coschedule_attested(dict(rec, n_forwards_with_same_render_pair=0)) is False
        assert g.coschedule_attested(dict(rec, max_distinct_nonzero_groups_in_one_forward=1)) is False
        assert g.coschedule_attested({"rpc": "absent"}) is None and g.coschedule_attested(None) is None
        # attested -> ok
        case = _mg_case()
        assert self._checks(case)["coschedule_candidate"]["ok"] is True
        # answered but never attested after the retries -> INCONCLUSIVE (exit 3), never PASS
        case = _mg_case()
        case["coschedule"][g.CANDIDATE] = dict(rec, n_forwards_with_same_render_pair=0)
        case["coschedule_attested"] = False
        for w in case["coschedule_waves"][g.CANDIDATE]:
            w["attested"] = False
        case["coschedule_attempts"] = 6
        checks = self._checks(case)
        assert checks["coschedule_candidate"]["ok"] is None
        assert "6 wave attempt" in checks["coschedule_candidate"]["detail"]
        case["checks"] = checks
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 3
        case["verdicts"] = {"multigroup_B32": {"verdict": g.INCONCLUSIVE}}
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 3
        # the B=32 cross-server comparison of the multigroup is evidence: a
        # "differs" there changes no exit code
        case = _mg_case()
        case["verdicts"] = {"multigroup_B32": {"verdict": g.EVIDENCE, "evidence": True, "equal": False,
                                               "underlying": g.UNSTABLE_REF}}
        case["checks"] = self._checks(case)
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 0
        # the candidate has no RPC at all -> FAIL (exit 2). In production the
        # accumulated boolean derives from the same per-wave records, so an
        # absent RPC yields attested=None per wave and the boolean False.
        case = _mg_case()
        absent = {"rpc": "absent", "status": 500}
        case["coschedule_waves"][g.CANDIDATE] = [
            {"wave": 1, "repeat": 1, "attempt": 1, "attested": None, "accepted": True, "record": absent}]
        case["coschedule"][g.CANDIDATE] = g.aggregate_coschedule(case["coschedule_waves"][g.CANDIDATE])
        case["coschedule_attested"] = False
        assert case["coschedule"][g.CANDIDATE]["rpc"] == "absent"
        checks = self._checks(case)
        assert checks["coschedule_candidate"]["ok"] is False
        case["checks"] = checks
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 2
        # the reference's record is informational: absent is fine
        assert "coschedule_reference" not in self._checks(_mg_case())

    def test_an_early_attestation_survives_a_later_unattested_repeat(self):
        """With two mandatory repeats, the first wave attests and the last
        repeat exhausts its retries without co-scheduling (vLLM admission
        varies). The evidence was obtained: the check must be ok and the
        exit-0 path reachable — never a nondeterministic exit 3."""
        rec = _mg_case()["coschedule_waves"][g.CANDIDATE][0]["record"]
        unatt = dict(rec, n_forwards_with_same_render_pair=0, n_forwards_with_ge2_groups=0,
                     max_distinct_nonzero_groups_in_one_forward=1)
        waves = [{"wave": 1, "repeat": 1, "attempt": 1, "attested": True, "accepted": True, "record": rec}]
        waves += [{"wave": k, "repeat": 2, "attempt": k - 1, "attested": False,
                   "accepted": k == 4, "record": unatt} for k in (2, 3, 4)]
        agg = g.aggregate_coschedule(waves)
        assert agg["attested_waves"] == [1] and agg["n_waves"] == 4
        assert agg["n_forwards_with_same_render_pair"] == rec["n_forwards_with_same_render_pair"]
        assert agg["max_distinct_nonzero_groups_in_one_forward"] == 5     # the max, not the last
        assert agg["n_forwards"] == 4 * rec["n_forwards"]                   # the sum
        assert g.coschedule_attested(agg) is True
        case = _mg_case()
        case["coschedule"][g.CANDIDATE] = agg
        case["coschedule_waves"][g.CANDIDATE] = waves
        case["coschedule_attested"] = any(w["attested"] for w in waves)
        case["coschedule_attempts"] = 4
        checks = self._checks(case)
        assert checks["coschedule_candidate"]["ok"] is True
        assert "wave(s) [1]" in checks["coschedule_candidate"]["detail"]
        assert all(c["ok"] for c in checks.values())
        case["checks"] = checks
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 0
        # the boolean alone carries it even if the aggregate were stale
        case["coschedule"][g.CANDIDATE] = unatt
        assert self._checks(case)["coschedule_candidate"]["ok"] is True
        # and with NO attesting wave anywhere (every record unattested): inconclusive, exit 3
        none = [dict(w, attested=False, record=unatt) for w in waves]
        case["coschedule"][g.CANDIDATE] = g.aggregate_coschedule(none)
        case["coschedule_waves"][g.CANDIDATE] = none
        case["coschedule_attested"] = False
        checks = self._checks(case)
        assert checks["coschedule_candidate"]["ok"] is None
        case["checks"] = checks
        assert g.overall_exit_code({"cases": [case], "complete": True}) == 3

    def test_aggregate_coschedule_absent_and_errors(self):
        assert g.aggregate_coschedule([{"wave": 1, "attested": None, "record": {"rpc": "absent"}}]) == {
            "rpc": "absent", "n_waves": 1}
        assert g.aggregate_coschedule([]) == {"rpc": "absent", "n_waves": 0}
        agg = g.aggregate_coschedule([{"wave": 1, "attested": False,
                                       "record": {"n_forwards": 3, "n_record_errors": 1,
                                                  "first_record_error": "boom",
                                                  "max_distinct_nonzero_groups_in_one_forward": 1}}])
        assert agg["first_record_error"] == "boom" and agg["n_record_errors"] == 1
        assert agg["attested_waves"] == []

    def test_harness_ids_are_used_for_routes_requests_and_evidence(self, tmp_path):
        """N1 item 2: the standup serves TrainTest cells under HARNESS ids
        (safety_refusal__psr_kl_all -> si_safety__psr_kl_all); every route
        lookup, request, row check and evidence check must go through the
        helper's explicit cell_map — integration over a routing table keyed
        by the harness id."""
        from sisteer.capeval.petri.registry_adapter import harness_cell_id
        cells = ["si_constitution__psr_kl_all", "si_constitution__lora_r4", "safety_refusal__psr_kl_all"]
        spec = importlib.util.spec_from_file_location(
            "lens_gate_standup", REPO / "scripts" / "lens_gate_standup.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        cmap = helper.cell_map(cells, harness_cell_id)
        assert cmap == {"si_constitution__psr_kl_all": "si_constitution__psr_kl_all",
                        "si_constitution__lora_r4": "si_constitution__lora_r4",
                        "safety_refusal__psr_kl_all": "si_safety__psr_kl_all"}
        assert helper.harness_cells(cells, harness_cell_id) == ["none"] + list(cmap.values())
        table = tmp_path / "routing_table.json"
        routes = [{"cell_id": "none", "virtual_model": "petri__base"},
                  {"cell_id": "none__lens0", "virtual_model": "petri__none__lens0"}]
        routes += [{"cell_id": h, "virtual_model": f"petri__{h}"} for h in cmap.values()]
        table.write_text(json.dumps({"version": 1, "routes": routes}))
        su = g.Standup(tag="mg", side=g.CANDIDATE, tree=REPO, out=tmp_path, proc=None,
                       stop_file=tmp_path / "STOP",
                       endpoint={"routing_table": str(table), "cell_map": cmap, "proxy_url": "x",
                                 "server_base_url": "y"}, api_key=None)
        vms = g.vmodels_of(su)
        hmap = g.harness_map_of(su.endpoint, cells)
        hcells = [hmap[c] for c in cells]
        assert hcells == ["si_constitution__psr_kl_all", "si_constitution__lora_r4", "si_safety__psr_kl_all"]
        assert all(vms.get(h) == f"petri__{h}" for h in hcells)
        assert vms.get("safety_refusal__psr_kl_all") is None          # the TrainTest id has NO route
        with pytest.raises(g.GateError, match="not in the standup's cell_map"):
            g.harness_map_of(su.endpoint, ["si_irrelevant__psr_kl_all"])
        vm = {"passthrough": vms["none"], "scale0": vms["none__lens0"], **{h: vms[h] for h in hcells}}
        mg_spec = g.multigroup_spec(16)
        prompts = [([{"role": "user", "content": f"p{i}"}], 5) for i in range(8)]
        rows = g.spec_requests(mg_spec, prompts, hcells, vm)
        assert any(r[0] == "petri__si_safety__psr_kl_all" for r in rows)   # cellC reaches the routes
        # the checks key evidence and row facts by the HARNESS ids the worker reports
        case = _mg_case(cells=tuple(hcells))
        assert all(c["ok"] for c in g.multigroup_checks(case, mg_spec, hcells, reference_sha=BASE,
                                                        base_sha=BASE, have_oracle=True).values())
        # keyed by TrainTest ids instead, the evidence check fails loudly
        wrong = _mg_case(cells=tuple(cells))
        checks = g.multigroup_checks(wrong, mg_spec, hcells, reference_sha=BASE, base_sha=BASE,
                                     have_oracle=True)
        assert not checks["evidence_candidate"]["ok"]

    def test_multigroup_runs_first(self):
        inv = list(g.served_inventory())
        order = g.gate_case_order(inv, None)
        assert order[0] == g.MULTIGROUP and order[1:] == inv
        assert g.gate_case_order(inv, [inv[3], g.MULTIGROUP]) == [g.MULTIGROUP, inv[3]]
        assert g.gate_case_order(inv, [inv[0]]) == [inv[0]]

    def test_multigroup_is_mandatory_for_complete(self):
        ok = _ok_case()
        ok["checks"] = g.case_checks(ok, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        doc = {"cases": [ok], "uncovered_cases": [g.MULTIGROUP], "complete": False,
               "inventory": {ok["case_id"]: {"kind": "psr", "arms": ["psr_kl_all"], "sources": ["adapter-probe"]}},
               "model": dict(key="llama8b", hf="x", rev="r" * 40, cfg="c", bank="b", branch="plain"),
               "geometry": {"max_model_len": 4096, "max_num_seqs": 64}, "batches": [1],
               "timing_batches": [8], "exit_code": 3}
        assert g.overall_exit_code(doc) == 3
        assert f"UNCOVERED: {g.MULTIGROUP}" in g.render_report(doc)

    def test_build_multigroup_members(self, tmp_path):
        """Two synthetic psr objects under different SIs plus a lora_all in ONE
        symlinked defenses dir; fewer than two distinct cells refuses."""
        dims = TestObjectResolution.DIMS
        stage = tmp_path / "stage"
        ms = g.MODELS["llama8b"]
        inv = g.served_inventory()
        resolved = {}
        for cid in ("psr:relu_gated_vector:gemv:response:skipbos0:both",
                    "lora_all:lora_oproj:rankK:response:skipbos0:both"):
            resolved[cid] = g.resolve_case_object(ms, cid, inv[cid]["arms"], bank_dir=None, extra_dirs=(),
                                                  candsi_dir=None, stage_root=stage, header_len=len(HEADER),
                                                  dims_fn=lambda: dims, allow_synthetic=True)
        mg = g.build_multigroup_members(ms, resolved, bank_dir=None, stage_root=stage,
                                        header_len=len(HEADER), dims_fn=lambda: dims, allow_synthetic=True)
        assert mg["cells"] == ["si_constitution__psr_kl_all", "si_constitution__lora_r4",
                               "safety_refusal__psr_kl_all"]
        assert mg["members"][2]["objects"] == "synthetic(seed 1)"
        d = Path(mg["defenses_dir"])
        for c in mg["cells"]:
            assert (d / c / "object.pt").is_symlink()
        # the two psr objects are DIFFERENT objects (distinct groups, distinct outputs)
        a = torch.load(d / "si_constitution__psr_kl_all" / "object.pt", weights_only=True)
        b = torch.load(d / "safety_refusal__psr_kl_all" / "object.pt", weights_only=True)
        assert not torch.equal(a["directions"]["0"], b["directions"]["0"])
        # every member adapts under the multigroup dir exactly as in its own
        for m in mg["members"]:
            k, ops = g.adapt_object_dir(d / m["cell"], arm=m["arm"], header_len=len(HEADER))
            assert g.case_id_of(k, ops) == m["case_id"]
        # a real second-SI psr in a bank is preferred over the synthetic
        bank = tmp_path / "bank"
        cell2, _ = g.write_synthetic_object("psr_kl_all", dims, bank, len(HEADER), seed=7,
                                            positions="response", si="si_irrelevant")
        mg2 = g.build_multigroup_members(ms, resolved, bank_dir=bank, stage_root=tmp_path / "s2",
                                         header_len=len(HEADER), dims_fn=lambda: dims, allow_synthetic=False)
        assert "si_irrelevant__psr_kl_all" in mg2["cells"] and mg2["members"][2]["objects"] == "real"
        with pytest.raises(g.GateError, match="at least two"):
            g.build_multigroup_members(ms, {}, bank_dir=None, stage_root=tmp_path / "s3",
                                       header_len=len(HEADER), dims_fn=lambda: dims, allow_synthetic=False)

    def test_helper_serves_several_cells_from_one_registry(self):
        spec = importlib.util.spec_from_file_location(
            "lens_gate_standup", REPO / "scripts" / "lens_gate_standup.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        from sisteer.capeval.petri.registry_adapter import harness_cell_id
        cells = helper.harness_cells(["si_constitution__psr_kl_all", "si_constitution__lora_r4",
                                      "safety_refusal__psr_kl_all", "si_constitution__psr_kl_all"],
                                     harness_cell_id)
        assert cells == ["none", "si_constitution__psr_kl_all", "si_constitution__lora_r4",
                         "si_safety__psr_kl_all"]
        assert helper.harness_cells(["none"], harness_cell_id) == ["none"]
        assert 'action="append"' in (REPO / "scripts" / "lens_gate_standup.py").read_text()


# --------------------------------------------------------------------------
# objects: real bank -> extra dirs -> staged candsi -> synthetic
# --------------------------------------------------------------------------


class TestObjectResolution:
    DIMS = {"hidden_size": 16, "num_hidden_layers": 6, "oproj_shapes": {1: (12, 16), 4: (12, 16)}}

    def test_bank_cell_found_in_si_order_and_class_b_bare(self, tmp_path):
        bank = tmp_path / "bank"
        for d in ("safety_refusal__psr_kl_all", "si_irrelevant__lora_r1_all", "antidote"):
            (bank / d).mkdir(parents=True)
            (bank / d / ("object.safetensors" if d == "antidote" else "object.pt")).write_bytes(b"x")
        assert g.find_bank_cell(bank, "psr_kl_all") == ("safety_refusal__psr_kl_all", bank)
        assert g.find_bank_cell(bank, "lora_r4") == ("si_irrelevant__lora_r4", bank)  # legacy dir
        assert g.find_bank_cell(bank, "antidote") == ("antidote", bank)
        assert g.find_bank_cell(bank, "lora_r1") is None and g.find_bank_cell(None, "x") is None

    def test_legacy_lora_r1_all_dir_is_a_rank4_object_not_lora_r1(self, tmp_path):
        """The kind check that keeps rank classes honest: a lora_all blob
        staged under the legacy lora_r1_all spelling never passes as the
        rank-1 `lora` arm."""
        cell, ddir = g.write_synthetic_object("lora_r4", self.DIMS, tmp_path, len(HEADER))
        legacy = tmp_path / "si_constitution__lora_r1_all"
        (ddir / cell).rename(legacy)
        with pytest.raises(g.GateError, match="!= 'lora'"):
            g.adapt_object_dir(legacy, arm="lora_r1", header_len=len(HEADER))

    def test_resolve_case_object_prefers_a_real_object_that_matches_the_case(self, tmp_path):
        bank = tmp_path / "bank"
        cell, _ = g.write_synthetic_object("psr_kl_all", self.DIMS, bank, len(HEADER))
        (bank / cell / "COMPLETE").write_text("ok")
        ms = g.MODELS["llama8b"]
        inv = g.served_inventory()
        cid = "psr:relu_gated_vector:gemv:response:skipbos0:both"
        out = g.resolve_case_object(ms, cid, inv[cid]["arms"], bank_dir=bank, extra_dirs=(),
                                    candsi_dir=None, stage_root=tmp_path / "stage",
                                    header_len=len(HEADER), dims_fn=lambda: self.DIMS,
                                    allow_synthetic=False)
        assert out["objects"] == "real" and out["cell"] == cell and out["case_id"] == cid

    def test_resolve_case_object_falls_to_synthetic_and_reports_rejections(self, tmp_path):
        # a real vector_all fit at "all" positions is a DIFFERENT case (its own,
        # required, covered by it — test_real_artifacts_join_the_inventory); for
        # the response case it is rejected, the synthetic covers it, recorded
        bank = tmp_path / "bank"
        cell = "si_constitution__vector_all"
        (bank / cell).mkdir(parents=True)
        torch.save({"kind": "vector_all", "vectors": {"0": torch.zeros(16)}, "positions": "all",
                    "skip_bos": True, "fit_positions": "all", "fit_response_header_len": None},
                   bank / cell / "object.pt")
        ms = g.MODELS["llama8b"]
        cid = "vector_all:add_vector:vec:response:skipbos0:both"
        out = g.resolve_case_object(ms, cid, ["vector_all"], bank_dir=bank, extra_dirs=(),
                                    candsi_dir=None, stage_root=tmp_path / "stage",
                                    header_len=len(HEADER), dims_fn=lambda: self.DIMS,
                                    allow_synthetic=True)
        assert out["objects"] == "synthetic" and out["case_id"] == cid
        assert any("adapts to case vector_all:add_vector:vec:all:skipbos1:both" in r
                   for r in out["real_candidates_rejected"])
        with pytest.raises(g.GateError, match="synthetic objects are disabled"):
            g.resolve_case_object(ms, cid, ["vector_all"], bank_dir=bank, extra_dirs=(),
                                  candsi_dir=None, stage_root=tmp_path / "stage2",
                                  header_len=len(HEADER), dims_fn=lambda: self.DIMS,
                                  allow_synthetic=False)

    def test_candsi_staging_checks_provenance(self, tmp_path):
        ms = g.MODELS["glm46vflash"]
        c = tmp_path / "candsi"
        d = c / "A10__psr_kl_all"
        cell, ddir = g.write_synthetic_object("psr_kl_all", self.DIMS, tmp_path / "w", len(HEADER))
        d.mkdir(parents=True)
        (d / "object.pt").write_bytes((ddir / cell / "object.pt").read_bytes())
        (d / "COMPLETE").write_text("ok")
        (d / "fit_provenance.json").write_text(json.dumps(
            {"resolved_model_id": ms.hf, "resolved_revision": "0" * 40}))
        assert g.find_candsi_cell(c, "psr_kl_all") == d
        with pytest.raises(g.GateError, match="another model's object"):
            g.check_fit_provenance(d, ms)
        (d / "fit_provenance.json").write_text(json.dumps(
            {"resolved_model_id": ms.hf, "resolved_revision": ms.rev, "positions_at_fit": "response"}))
        out = g.resolve_case_object(ms, "psr:relu_gated_vector:gemv:response:skipbos0:both", ["psr_kl_all"],
                                    bank_dir=None, extra_dirs=(), candsi_dir=c,
                                    stage_root=tmp_path / "stage", header_len=len(HEADER),
                                    dims_fn=lambda: self.DIMS, allow_synthetic=False)
        assert out["objects"].startswith("real(staged:candsi/A10__psr_kl_all")
        assert (Path(out["defenses_dir"]) / out["cell"] / "object.pt").is_symlink()
        assert out["fit_provenance"]["positions_at_fit"] == "response"

    def test_synthetic_lora_refuses_without_attention_layers(self, tmp_path):
        dims = dict(self.DIMS, oproj_shapes=None, oproj_layers_error="meta skeleton failed")
        for arm in ("lora_r4", "lora_r1"):
            with pytest.raises(g.GateError, match="o_proj shapes"):
                g.write_synthetic_object(arm, dims, tmp_path, len(HEADER))

    def test_synthetic_lora_r1_targets_one_attention_layer(self, tmp_path):
        cell, ddir = g.write_synthetic_object("lora_r1", self.DIMS, tmp_path, len(HEADER))
        kind, ops = g.adapt_object_dir(ddir / cell, arm="lora_r1", header_len=len(HEADER))
        assert kind == "lora" and [o.layer for o in ops] == [4]
        assert ops[0].lowrank_a.shape == (1, 12) and ops[0].lowrank_b.shape == (16, 1)


class TestExtraObjectProvenance:
    """Real-object selection is PER MODEL (re-gate at aa2b8a1, array 384162):
    the extra dirs were searched unkeyed, so stage3-wave lane A's Llama-3.1-8B
    rank-1 LoRA was served to all five families — AdapterError 500s on
    gemma4e4b / qwen35_4b, a dead reference server on nemotron3nano4b, and a
    silently served PASS on glm46vflash (whose o_proj is also 4096x4096). An
    extra-dir object serves only the family its own fit-time record names
    (train_budget_meta.json — the blob carries no model name), the configured
    origin cross-checks it, and an unattributable object refuses loudly."""
    DIMS = TestObjectResolution.DIMS
    CID = "lora:lora_oproj:rank1:all:skipbos0:both"
    TEMPLATE_HASH = "b48c47f644389271"

    def _lane(self, tmp_path, *, meta=True, hf=None, rev=None):
        """stage3-wave-shaped extra dir: one lora_r1 cell fit at ``all`` with
        the training pipeline's sidecar naming (by default) llama8b."""
        llama = g.MODELS["llama8b"]
        lane = tmp_path / "stage3-wave" / "A"
        cell, ddir = g.write_synthetic_object("lora_r1", self.DIMS, lane, len(HEADER), positions="all")
        cell_dir = ddir / cell
        kind, ops = g.adapt_object_dir(cell_dir, arm="lora_r1", header_len=len(HEADER))
        assert g.case_id_of(kind, ops) == self.CID
        if meta:
            (cell_dir / "train_budget_meta.json").write_text(json.dumps({
                "arm": "lora_r1", "cell_id": cell, "model_id": hf or llama.hf,
                "model_revision": rev or llama.rev, "template_hash": self.TEMPLATE_HASH,
                "positions": "all", "positions_at_fit": "all", "site": "o_proj_out"}))
        return lane, cell, cell_dir

    def _resolve(self, ms, lane, tmp_path, *, origins=None, allow_synthetic=True):
        return g.resolve_case_object(ms, self.CID, ["lora_r1"], bank_dir=None, extra_dirs=(lane,),
                                     candsi_dir=None, stage_root=tmp_path / "stage" / ms.key,
                                     header_len=len(HEADER), dims_fn=lambda: self.DIMS,
                                     allow_synthetic=allow_synthetic, extra_origins=origins)

    def test_extra_object_serves_only_the_family_its_record_names(self, tmp_path):
        lane, cell, cell_dir = self._lane(tmp_path)
        llama = g.MODELS["llama8b"]
        out = self._resolve(llama, lane, tmp_path, allow_synthetic=False)
        assert out["objects"] == "real(provenance: llama8b via train_budget_meta)"
        assert out["cell"] == cell and out["source"] == str(cell_dir / "object.pt")
        assert out["fit_provenance"]["resolved_model_id"] == llama.hf
        assert out["fit_provenance"]["resolved_revision"] == llama.rev
        assert out["fit_provenance"]["origin"] == "llama8b"
        for key, ms in g.MODELS.items():
            if key == "llama8b":
                continue
            out = self._resolve(ms, lane, tmp_path)
            assert out["objects"] == "synthetic", key
            assert out["source"] != str(cell_dir / "object.pt")
            rejected = out["real_candidates_rejected"]
            assert any(f"real(A) {cell}: fit on {llama.hf}@{llama.rev} (llama8b) per train_budget_meta, "
                       f"not {key}" == r for r in rejected), (key, rejected)
            with pytest.raises(g.GateError, match="synthetic objects are disabled"):
                self._resolve(ms, lane, tmp_path, allow_synthetic=False)

    def test_inventory_sources_are_this_familys_objects_only(self, tmp_path):
        lane, cell, cell_dir = self._lane(tmp_path)
        src = str(cell_dir / "object.pt")
        real = g.real_artifact_cases(g.MODELS["llama8b"], bank_dir=None, extra_dirs=(lane,),
                                     candsi_dir=None, header_len=len(HEADER))
        assert real[self.CID]["sources"] == [src] and "__foreign__" not in real
        for key in ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b"):
            ms = g.MODELS[key]
            real = g.real_artifact_cases(ms, bank_dir=None, extra_dirs=(lane,), candsi_dir=None,
                                         header_len=len(HEADER))
            assert self.CID not in real, key
            assert [n for n in real["__foreign__"]["sources"] if n.startswith(src + ": fit on ")], key
            inv = g.plan_inventory(ms, bank_dir=None, extra_dirs=(lane,), candsi_dir=None,
                                   header_len=len(HEADER))
            assert inv[self.CID]["sources"] == ["adapter-probe"], key      # probed, never foreign-sourced
            assert inv["__foreign__"]["sources"] == real["__foreign__"]["sources"]
            assert set(inv) & set(g.INVENTORY_NOTES) == {"__foreign__"}

    def test_a_record_contradicting_the_configured_origin_refuses(self, tmp_path):
        lane, cell, _ = self._lane(tmp_path)
        for key in ("llama8b", "gemma4e4b"):
            with pytest.raises(g.GateError, match="contradicts the object's own record"):
                self._resolve(g.MODELS[key], lane, tmp_path, origins={lane: "gemma4e4b"})
        with pytest.raises(g.GateError, match="not a gate family"):
            self._resolve(g.MODELS["llama8b"], lane, tmp_path, origins={lane: "llama70b"})
        # an agreeing origin is a no-op cross-check
        out = self._resolve(g.MODELS["llama8b"], lane, tmp_path, origins={lane: "llama8b"})
        assert out["objects"] == "real(provenance: llama8b via train_budget_meta)"

    def test_a_recordless_object_needs_a_configured_origin(self, tmp_path):
        lane, cell, cell_dir = self._lane(tmp_path, meta=False)
        with pytest.raises(g.GateError, match="unattributable object is never served to any family"):
            self._resolve(g.MODELS["llama8b"], lane, tmp_path)
        with pytest.raises(g.GateError, match="unattributable"):        # the inventory walk refuses too
            g.real_artifact_cases(g.MODELS["llama8b"], bank_dir=None, extra_dirs=(lane,),
                                  candsi_dir=None, header_len=len(HEADER))
        out = self._resolve(g.MODELS["llama8b"], lane, tmp_path, origins={lane: "llama8b"})
        assert out["objects"] == ("real(provenance: llama8b, configured origin; the object carries no "
                                  "fit-time record)")
        assert out["fit_provenance"]["status"] == "unverified"
        out = self._resolve(g.MODELS["qwen35_4b"], lane, tmp_path, origins={lane: "llama8b"})
        assert out["objects"] == "synthetic"
        assert any(r == f"real(A) {cell}: configured origin llama8b, not qwen35_4b (the object carries "
                   "no fit-time record)" for r in out["real_candidates_rejected"])

    def test_a_record_at_another_revision_is_another_models_object(self, tmp_path):
        llama = g.MODELS["llama8b"]
        lane, cell, _ = self._lane(tmp_path, rev="0" * 40)
        out = self._resolve(llama, lane, tmp_path)
        assert out["objects"] == "synthetic"
        assert any(r == f"real(A) {cell}: fit on {llama.hf}@{'0' * 40} per train_budget_meta, but the "
                   f"gate serves @{llama.rev}" for r in out["real_candidates_rejected"])

    def test_the_save_sidecars_model_name_is_partial_provenance(self, tmp_path):
        llama = g.MODELS["llama8b"]
        lane, cell, cell_dir = self._lane(tmp_path, meta=False)
        (cell_dir / "object.pt.json").write_text(json.dumps({"model_name": llama.hf}))
        out = self._resolve(llama, lane, tmp_path)
        assert out["objects"] == "real(provenance: llama8b via object_json, revision unrecorded)"
        assert out["fit_provenance"]["status"] == "partial"
        out = self._resolve(g.MODELS["glm46vflash"], lane, tmp_path)
        assert out["objects"] == "synthetic"
        assert any("per object_json, not glm46vflash" in r for r in out["real_candidates_rejected"])

    def test_defaults_and_cli_record_the_stage3_wave_origin(self, monkeypatch, tmp_path):
        assert set(g.DEFAULT_EXTRA_OBJECTS.values()) == {"llama8b"}
        assert sorted(Path(k).name for k in g.DEFAULT_EXTRA_OBJECTS) == ["A", "C"]
        assert all("stage3-wave" in k for k in g.DEFAULT_EXTRA_OBJECTS)
        monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "shared" / "someone"))   # the job's shared root
        dirs, origins = g.parse_extra_objects(None, "someone")
        assert [str(d) for d in dirs] == [str(tmp_path / "shared/someone/exp/objects/stage3-wave/A"),
                                          str(tmp_path / "shared/someone/exp/objects/stage3-wave/C")]
        assert origins == {d: "llama8b" for d in dirs}
        dirs, origins = g.parse_extra_objects(["/x/A=gemma4e4b", "/y/B"], "someone")
        assert dirs == [Path("/x/A"), Path("/y/B")] and origins == {Path("/x/A"): "gemma4e4b"}
        with pytest.raises(g.GateError, match="not a gate family"):
            g.parse_extra_objects(["/x/A=lama8b"], "someone")
        sb = TestFamilyTableAndSbatch.SBATCH.read_text()
        assert '--extra-objects-dir "$SHARED/exp/objects/stage3-wave/A=llama8b"' in sb
        assert '--extra-objects-dir "$SHARED/exp/objects/stage3-wave/C=llama8b"' in sb

    def test_a_cases_subset_run_lands_in_its_own_dir_and_labels_itself(self, monkeypatch, tmp_path):
        """The per-family re-run of the one failed case must not overwrite the
        family's full report, and must never read as a certification."""
        assert g.subset_slug([self.CID]) == "lora_lora_oproj_rank1_all_skipbos0_both"
        assert g.subset_slug([self.CID, "psr:relu_gated_vector:gemv:all:skipbos0:both"]) == \
            "lora_lora_oproj_rank1_all_skipbos0_both+psr_relu_gated_vector_gemv_all_skipbos0_both"
        captured = {}

        def fake_run_gate(ctx, **kw):
            captured["ctx"], captured["kw"] = ctx, kw
            return {"exit_code": 3}
        monkeypatch.setattr(g, "run_gate", fake_run_gate)
        monkeypatch.setattr(g, "plan_inventory", lambda *a, **k: {})
        monkeypatch.setattr(g, "header_len_for", lambda *a, **k: 4)
        args = ["--model", "gemma4e4b", "--reference-repo", str(tmp_path), "--results-root",
                str(tmp_path / "r"), "--objects-root", str(tmp_path / "o"), "--candsi-root",
                str(tmp_path / "c")]
        assert g.main(args) == 3
        assert captured["ctx"].results_dir == tmp_path / "r" / "gemma4e4b"
        assert captured["kw"]["only_cases"] is None
        assert g.main(args + ["--cases", self.CID]) == 3
        assert captured["ctx"].results_dir == tmp_path / "r" / "gemma4e4b" / "subset" / g.subset_slug([self.CID])
        assert captured["kw"]["only_cases"] == [self.CID]
        assert captured["kw"]["extra_origins"] == {d: "llama8b" for d in captured["kw"]["extra_dirs"]}
        sb = TestFamilyTableAndSbatch.SBATCH.read_text()
        subset_line = 'SUBSET="subset/$(printf ' + "'%s'" + ' "$GATE_CASES" | tr ' + "':,' '_+'" + ')"'
        assert subset_line in sb                                    # the bash twin of subset_slug
        assert 'RESULTS_DIR="$RESULTS_DIR/$SUBSET"; FIGS_DIR="$FIGS_DIR/$SUBSET"' in sb
        assert '${GATE_CASES:+--cases "$GATE_CASES"}' in sb
        doc = {"model": dict(key="gemma4e4b", hf="x", rev="0" * 40, cfg="c", bank="b", branch="br"),
               "geometry": {"max_model_len": 4096, "max_num_seqs": 64}, "trees": {},
               "batches": [1], "timing_batches": [8], "exit_code": 3, "batch_invariant": 0,
               "uncovered_cases": ["psr:relu_gated_vector:gemv:all:skipbos0:both"], "complete": False,
               "inventory": {}, "cases": [], "only_cases": [self.CID]}
        text = g.render_report(doc)
        assert f"SUBSET RUN — cases {[self.CID]} only: complete=False by construction" in text
        assert "NOT a certification" in text and "complete=False" in text
        assert "SUBSET RUN" not in g.render_report(dict(doc, only_cases=None))

    def test_report_names_the_provenance_and_the_foreign_note(self):
        ms = g.MODELS["gemma4e4b"]
        doc = {"model": dict(key=ms.key, hf=ms.hf, rev=ms.rev, cfg=ms.cfg, bank=ms.bank, branch=ms.branch),
               "geometry": {"max_model_len": 4096, "max_num_seqs": 64},
               "trees": {"reference": {"path": "/ref", "sha": BASE}, "candidate": {"path": "/c", "sha": "a" * 40}},
               "node": "TESTHOST", "job_id": "1", "batches": [1, 8], "timing_batches": [8],
               "exit_code": 3, "batch_invariant": 0, "uncovered_cases": [], "complete": True,
               "inventory": {self.CID: {"kind": "lora", "arms": ["lora_r1"], "sources": ["adapter-probe"],
                                        "routing": "vectorized"}},
               "inventory_notes": {"__foreign__": {"kind": None, "arms": [], "sources": [
                   "/objects/stage3-wave/A/safety_refusal__lora_r1/object.pt: fit on x/y@0 (llama8b) "
                   "per train_budget_meta, not gemma4e4b"]}},
               "cases": [{"case_id": self.CID, "arm": "lora_r1", "objects": "synthetic", "error": "boom"},
                         {"case_id": "psr:relu_gated_vector:gemv:all:skipbos0:both", "arm": "psr_kl_all",
                          "objects": "real(provenance: gemma4e4b via train_budget_meta)", "error": "boom"}]}
        text = g.render_report(doc)
        assert "| lora_r1 | synthetic | ERROR: boom" in text
        assert "| psr_kl_all | real(provenance: gemma4e4b via train_budget_meta) | ERROR: boom" in text
        assert ("- FOREIGN real artifact (another family's object; never served here): "
                "/objects/stage3-wave/A/safety_refusal__lora_r1/object.pt: fit on x/y@0 (llama8b) per "
                "train_budget_meta, not gemma4e4b") in text
        assert "real(unverified provenance" not in text and "unverified provenance" not in g.__doc__


# --------------------------------------------------------------------------
# G5 — the standup's registry vocabulary covers the served cells (job 382416)
# --------------------------------------------------------------------------


def _helper():
    spec = importlib.util.spec_from_file_location(
        "lens_gate_standup", REPO / "scripts" / "lens_gate_standup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(ms):
    import yaml
    return yaml.safe_load((REPO / ms.cfg).read_text(encoding="utf-8"))


def _registry_arms(cfg):
    """The arms the standup registry would carry from ``cfg`` alone (what
    traintest_cell_source -> build_registry filters to)."""
    from sisteer.eval import defense_cell as tt
    return set(tt._registered(tuple(cfg.get("class_a_arms", tt.CLASS_A_ARMS)))) | set(
        tt._registered(tuple(cfg.get("class_b_arms", tt.CLASS_B_ARMS))))


def _cells_for(arms):
    from sisteer.capeval.petri import defense_cells as dc
    return [arm if arm in dc.CLASS_B_IDS else f"si_constitution__{arm}" for arm in sorted(arms)]


class TestRegistryVocabulary:
    """Job array 382416 (2026-09-04): the reference standup of every synthetic
    vector_all case, of lora_r1 / loreft on gemma-4 / Qwen3.5 / GLM / Nemotron,
    of antidote / alphasteer on Qwen3.5 and of lora_r4 + the multigroup on GLM
    / Nemotron died with ``absent cells: [...]`` before the candidate stood
    up — 27 of 70 case standups. Not a tree difference (defense_cells /
    registry_adapter are byte-identical in both trees): the helper built the
    registry from the model's two-SI config, whose class arm lists name the
    arms THAT campaign fitted, while the inventory is the adapters' whole
    servable vocabulary. The helper now derives the registry config from the
    cells it is asked to serve."""

    QWEN_CELLS = ["si_constitution__vector_all", "antidote", "safety_refusal__lora_r1",
                  "si_constitution__lora_r4", "si_constitution__loreft", "alphasteer",
                  "si_constitution__psr_kl_all"]

    def test_gate_config_adds_exactly_the_served_arms(self):
        helper = _helper()
        cfg = _cfg(g.MODELS["qwen35_4b"])
        assert cfg["class_a_arms"] == ["psr_kl_all", "lora_r4"] and cfg["class_b_arms"] == []
        out, added = helper.gate_config(cfg, ["none"] + self.QWEN_CELLS)
        assert out["class_a_arms"] == ["psr_kl_all", "lora_r4", "vector_all", "lora_r1", "loreft"]
        assert out["class_b_arms"] == ["antidote", "alphasteer"]
        assert added == {"class_a_arms": ["vector_all", "lora_r1", "loreft"],
                         "class_b_arms": ["antidote", "alphasteer"]}
        # nothing else in the config moved, and the input is untouched
        assert {k: v for k, v in out.items() if k not in ("class_a_arms", "class_b_arms")} == \
            {k: v for k, v in cfg.items() if k not in ("class_a_arms", "class_b_arms")}
        assert cfg["class_a_arms"] == ["psr_kl_all", "lora_r4"]
        # the legacy spelling: a config naming lora_r1_all (filtered out by build_registry
        # as unregistered) still gets the canonical lora_r4 the cell names
        legacy = dict(cfg, class_a_arms=["psr_kl_all", "lora_r1_all"])
        out, added = helper.gate_config(legacy, ["si_constitution__lora_r4"])
        assert out["class_a_arms"] == ["psr_kl_all", "lora_r1_all", "lora_r4"]
        assert "lora_r4" in _registry_arms(out) and "lora_r4" not in _registry_arms(legacy)
        # a config without the keys starts from the module defaults, like the registry does
        from sisteer.eval import defense_cell as tt
        bare = {k: v for k, v in cfg.items() if k not in ("class_a_arms", "class_b_arms")}
        out, added = helper.gate_config(bare, ["si_constitution__vector_all", "antidote"])
        assert out["class_a_arms"][:len(tt.CLASS_A_ARMS)] == list(tt.CLASS_A_ARMS)
        assert out["class_b_arms"][:len(tt.CLASS_B_ARMS)] == list(tt.CLASS_B_ARMS)
        assert "vector_all" in out["class_a_arms"] and "antidote" in out["class_b_arms"]
        assert set(added["class_a_arms"]) == {"vector_all"} - set(tt.CLASS_A_ARMS)
        assert set(added["class_b_arms"]) == {"antidote"} - set(tt.CLASS_B_ARMS)
        with pytest.raises(RuntimeError, match="names no arm"):
            helper.gate_config(cfg, ["prompted__safety_refusal"])

    @pytest.mark.parametrize("key", sorted(g.MODELS))
    def test_every_family_config_lacks_servable_arms_and_the_gate_config_covers_them(self, key):
        """The finding, per family: the config's registry vocabulary misses
        inventory arms (so the derivation is load-bearing everywhere), and
        the derived config carries every servable arm."""
        helper = _helper()
        cfg = _cfg(g.MODELS[key])
        servable = set(g.servable_arms())
        missing = servable - _registry_arms(cfg)
        assert "vector_all" in missing, key                  # absent from EVERY family config
        if key in ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b"):
            assert {"lora_r1", "loreft"} <= missing, key
        if key == "qwen35_4b":
            assert {"antidote", "alphasteer"} <= missing
        if key in ("glm46vflash", "nemotron3nano4b"):
            assert "lora_r4" in missing, key                  # spelled lora_r1_all there
        out, added = helper.gate_config(cfg, _cells_for(servable))
        assert servable <= _registry_arms(out), key
        assert set(added["class_a_arms"]) | set(added["class_b_arms"]) == missing, key

    def test_enumerate_gate_cells_serves_what_the_raw_config_reports_absent(self, tmp_path):
        """End to end over synthetic objects: the SAME defenses dir and cells
        report ABSENT through the raw Qwen config (the 382416 death) and
        serve completely through the helper's derived config."""
        helper = _helper()
        from sisteer.capeval import manifest as cm
        from sisteer.capeval.petri import defense_cells as dc
        from sisteer.capeval.petri.registry_adapter import harness_cell_id, traintest_cell_source
        from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
        ms = g.MODELS["qwen35_4b"]
        cfg = _cfg(ms)
        ddir = tmp_path / "defenses"
        cells = []
        for arm in ("vector_all", "antidote", "lora_r1", "lora_r4", "loreft", "alphasteer", "psr_kl_all"):
            cell, _ = g.write_synthetic_object(arm, TestObjectResolution.DIMS, ddir, len(HEADER))
            cells.append(cell)
        manifest = "configs/capeval/manifests/petri_twosi.yaml"
        out = tmp_path / "standup"
        out.mkdir()
        spec, src, cset, gate_cfg_path, added = helper.enumerate_gate_cells(
            repo=REPO, cfg=cfg, cells_tt=cells, defenses_dir=ddir, manifest=manifest,
            chat_template_hash="ab" * 8, out=out)
        assert cset.absent == ()
        served = {c.cell_id for c in cset.all_served()}
        assert {harness_cell_id(c) for c in cells} <= served and "none" in served
        assert gate_cfg_path == out / "two_si_gate.yaml" and gate_cfg_path.is_file()
        assert added == {"class_a_arms": ["vector_all", "lora_r1", "loreft"],
                         "class_b_arms": ["antidote", "alphasteer"]}
        import yaml
        written = yaml.safe_load(gate_cfg_path.read_text())
        assert written["class_a_arms"] == ["psr_kl_all", "lora_r4", "vector_all", "lora_r1", "loreft"]
        assert written["model"] == ms.hf and written["model_revision"] == ms.rev
        # the negative control: the raw config over the same dir and spec
        raw = traintest_cell_source(spec, repo_root=REPO, config_path=REPO / ms.cfg, defenses_dir=ddir)
        bf = {k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)}
        pf = resolve_prompt_files(spec, override_root=REPO / "configs/capeval/petri/prompts")
        absent = dc.enumerate_cells(spec, raw, behavior_files=bf, prompt_files=pf).absent
        assert sorted(a.cell_id for a in absent) == sorted(
            harness_cell_id(c) for c in cells if c not in ("si_constitution__lora_r4",
                                                          "si_constitution__psr_kl_all"))
        assert all("not in the defense-cell registry" in a.reason for a in absent)
        # and the helper's own refusal names each absent cell WITH its reason
        (ddir / "si_constitution__vector_all" / "object.pt").unlink()
        with pytest.raises(RuntimeError, match=r"absent cells: si_constitution__vector_all: object artifact missing"):
            helper.enumerate_gate_cells(repo=REPO, cfg=cfg, cells_tt=cells, defenses_dir=ddir,
                                        manifest=manifest, chat_template_hash="ab" * 8, out=out)


class TestSyntheticOprojShapes:
    """Job array 382416, gemma-4 E4B: the reference standup served the
    synthetic lora_all object and refused it at layer 5 — "read basis dim
    2048 != o_proj input dim 4096 — the object was fit on a different
    architecture" (HTTP 500 through the proxy, the case errored). A gate
    FIXTURE bug: model_dims minted every layer's read basis at
    num_attention_heads * head_dim = 8 * 256 = 2048, but gemma-4's global
    attention layers project 8 heads x global_head_dim 512 = 4096 (o_proj
    weights [2560, 2048] on sliding layers, [2560, 4096] on global ones in
    the pinned snapshot). The mint now reads each layer's o_proj shape off
    the model's own skeleton."""

    @staticmethod
    def _skeleton(widths, d_out):
        import torch.nn as nn

        class Attn(nn.Module):
            def __init__(self, d_in):
                super().__init__()
                self.q_proj = nn.Linear(d_out, d_in, bias=False)
                self.k_proj = nn.Linear(d_out, d_in // 4, bias=False)
                self.v_proj = nn.Linear(d_out, d_in // 4, bias=False)
                self.o_proj = nn.Linear(d_in, d_out, bias=False)

        class Block(nn.Module):
            def __init__(self, d_in):
                super().__init__()
                self.self_attn = Attn(d_in)

        class MambaBlock(nn.Module):           # a mixer with NO attention o_proj
            def __init__(self):
                super().__init__()
                self.mixer = nn.Module()
                self.mixer.in_proj = nn.Linear(d_out, 8, bias=False)
                self.mixer.out_proj = nn.Linear(8, d_out, bias=False)

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MambaBlock() if w is None else Block(w) for w in widths])

        class Skel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Inner()
        with torch.device("meta"):
            return Skel()

    def test_shapes_are_read_per_layer_off_the_skeleton(self):
        skel = self._skeleton([2048, 2048, None, 4096, 2048], 2560)
        assert g.oproj_shapes_from_skeleton(skel) == {0: (2048, 2560), 1: (2048, 2560),
                                                      3: (4096, 2560), 4: (2048, 2560)}
        # a module without in/out_features but with a 2-D weight is read from the weight
        import torch.nn as nn
        skel2 = self._skeleton([12], 16)
        blk = skel2.model.layers[0]
        blk.self_attn.o_proj = nn.Module()
        with torch.device("meta"):
            blk.self_attn.o_proj.weight = nn.Parameter(torch.zeros(16, 12))
        assert g.oproj_shapes_from_skeleton(skel2) == {0: (12, 16)}
        blk.self_attn.o_proj = nn.Module()          # neither: refuses, never guesses
        with pytest.raises(g.GateError, match="refusing to guess"):
            g.oproj_shapes_from_skeleton(skel2)
        assert g._oproj_shapes({"oproj_shapes": {"3": [4096, 2560]}}) == {3: (4096, 2560)}
        assert g._oproj_shapes({}) == {}

    def test_synthetic_lora_is_minted_with_each_layers_width(self, tmp_path):
        dims = {"hidden_size": 2560, "num_hidden_layers": 6,
                "oproj_shapes": g.oproj_shapes_from_skeleton(
                    self._skeleton([2048, 2048, None, 4096, 2048, 4096], 2560))}
        cell, ddir = g.write_synthetic_object("lora_r4", dims, tmp_path, len(HEADER))
        kind, ops = g.adapt_object_dir(ddir / cell, arm="lora_r4", header_len=len(HEADER))
        by_layer = {o.layer: o for o in ops}
        assert kind == "lora_all" and sorted(by_layer) == [0, 1, 3, 4, 5]     # no o_proj on layer 2
        assert by_layer[0].lowrank_a.shape == (4, 2048) and by_layer[3].lowrank_a.shape == (4, 4096)
        assert by_layer[5].lowrank_b.shape == (2560, 4)
        # the global layer's op applies to a 4096-wide o_proj input ...
        x, out = torch.zeros(2, 4096), torch.zeros(2, 2560)
        assert la.apply_oproj_ops(x, out, (by_layer[3],), 1.0, layer=3, ctx=la.DecodeRows()).shape == out.shape
        # ... and a sliding layer's 2048-wide basis on that input is EXACTLY the 382416 refusal
        with pytest.raises(la.AdapterError, match="read basis dim 2048 != o_proj input dim 4096"):
            la.apply_oproj_ops(x, out, (by_layer[0],), 1.0, layer=0, ctx=la.DecodeRows())
        # rank-1 targets the middle attention layer with THAT layer's shapes
        cell, ddir = g.write_synthetic_object("lora_r1", dims, tmp_path / "r1", len(HEADER))
        kind, ops = g.adapt_object_dir(ddir / cell, arm="lora_r1", header_len=len(HEADER))
        assert kind == "lora" and [o.layer for o in ops] == [3]
        assert ops[0].lowrank_a.shape == (1, 4096) and ops[0].lowrank_b.shape == (2560, 1)
        # uniform widths (llama, GLM, Qwen, Nemotron) are the same mint as before
        dims_u = {"hidden_size": 16, "num_hidden_layers": 4, "oproj_shapes": {1: (12, 16), 3: (12, 16)}}
        cell, ddir = g.write_synthetic_object("lora_r4", dims_u, tmp_path / "u", len(HEADER))
        _, ops = g.adapt_object_dir(ddir / cell, arm="lora_r4", header_len=len(HEADER))
        assert {o.layer: tuple(o.lowrank_a.shape) for o in ops} == {1: (4, 12), 3: (4, 12)}

    def test_model_dims_reads_the_skeleton_not_a_heads_formula(self):
        import inspect
        src = inspect.getsource(g.model_dims)
        assert "oproj_shapes_from_skeleton" in src and "num_attention_heads" not in src


class TestPrefixCaching:
    """Job array 382416: controls_B1 (scale-0 == passthrough == oracle at
    B=1) FAILED on both trees for every case of llama, gemma-4 and GLM and
    PASSED on Qwen3.5 and Nemotron — exactly the families where
    build_server_launch resolves prefix caching ON vs OFF. The scale-0
    ``none__lens0`` request is a lens route with the proxy's request-unique
    cache_salt (a cold prefill); the passthrough and the oracle were warm
    cache hits on prompt 0. Every gate standup now runs cold."""

    def test_prefix_caching_ok(self):
        """Round-8 review: an unreadable / missing attestation used to collapse
        into None and PASS. Only composed False AND attested boolean False
        with no read error passes; everything else refuses, by name."""
        assert g.prefix_caching_ok({"prefix_caching": False, "prefix_caching_attested": False}) == (
            True, "prefix caching OFF: composed False, attested False")
        assert g.prefix_caching_ok({"prefix_caching": False, "prefix_caching_attested": False,
                                    "prefix_caching_attest_error": None})[0] is True
        # absent attestation: refused, never a pass by silence
        ok, why = g.prefix_caching_ok({"prefix_caching": False})
        assert ok is False and "no attested value at all" in why
        ok, why = g.prefix_caching_ok({"prefix_caching": False, "prefix_caching_attested": None})
        assert ok is False and "no attested value at all" in why
        # a read / parse error string: refused, quoting it
        ok, why = g.prefix_caching_ok({"prefix_caching": False, "prefix_caching_attested": None,
                                       "prefix_caching_attest_error": "JSONDecodeError: Expecting value"})
        assert ok is False and "JSONDecodeError" in why and "unverifiable" in why
        # non-boolean spellings of False are not False
        for bad in ("False", 0, "off"):
            ok, why = g.prefix_caching_ok({"prefix_caching": False, "prefix_caching_attested": bad})
            assert ok is False and "not the boolean" in why, bad
        for ep in ({"prefix_caching": True, "prefix_caching_attested": False},
                   {"prefix_caching": None, "prefix_caching_attested": False}, {}, None,
                   {"prefix_caching": False, "prefix_caching_attested": True}):
            ok, why = g.prefix_caching_ok(ep)
            assert ok is False, ep

    def test_helper_composes_caching_off_and_the_gate_refuses_otherwise(self):
        import inspect
        text = (REPO / "scripts" / "lens_gate_standup.py").read_text()
        # OFF by default (every fidelity standup); the per-model default ONLY under
        # --prefix-caching default (the shared-prefill standup, 2026-09-06)
        assert 'prefix_caching = False if args.prefix_caching == "off" else None' in text
        assert "prefix_caching=prefix_caching)" in text and '"prefix_caching": launch.prefix_caching' in text
        assert '"prefix_caching_requested": args.prefix_caching' in text
        assert 'choices=("off", "default"), default="off"' in text
        assert '"prefix_caching_attested": prefix_attested' in text
        assert '"prefix_caching_attest_error": prefix_attest_error' in text
        # a read exception lands in the endpoint as an error string, never as a silent None
        body = text[text.index("prefix_attested, prefix_attest_error = None, None"):text.index('auth = out / "server_auth.json"')]
        assert 'prefix_attest_error = f"{type(e).__name__}: {e}"' in body
        # the kwarg exists on BOTH trees' launch composer (vllm_server.py is identical there)
        from sisteer.capeval.serving import vllm_server as vs
        assert "prefix_caching" in inspect.signature(vs.build_server_launch).parameters
        base = subprocess.run(["git", "-C", str(REPO), "show",
                               f"{BASE}:src/sisteer/capeval/serving/vllm_server.py"],
                              capture_output=True, text=True)
        if base.returncode == 0:                      # the base commit is present in this clone
            assert "prefix_caching: Optional[bool] = None" in base.stdout
        # start_standup refuses a standup that did not run cold — except the shared-prefill
        # standup, which it holds to ON with the same strictness
        src = inspect.getsource(g.start_standup)
        assert "prefix_caching_on_ok if prefix_caching == PREFIX_CACHING_DEFAULT else prefix_caching_ok" in src
        assert '"--prefix-caching", prefix_caching' in src

    def test_helper_reads_the_resolved_cache_state_strictly(self):
        """The attestation shape observed in 382416 (llama passthrough standup):
        scheduler_geometry.expected.enable_prefix_caching plus one
        enable_prefix_caching per worker report -- not a top-level key. Read
        STRICTLY: a boolean on the launch AND every worker report, at least
        one report; anything else is an error string the gate refuses, never
        None."""
        helper = _helper()
        att = {"scheduler_geometry": {"expected": {"enable_prefix_caching": False, "max_num_seqs": 64},
                                      "worker_reports": [{"enable_prefix_caching": False,
                                                          "batch_invariant_env": None}]}}
        assert helper.prefix_caching_attested(att) == (False, None)
        att["scheduler_geometry"]["expected"]["enable_prefix_caching"] = True
        assert helper.prefix_caching_attested(att) == (True, None)
        att["scheduler_geometry"]["expected"]["enable_prefix_caching"] = False
        att["scheduler_geometry"]["worker_reports"][0]["enable_prefix_caching"] = True
        assert helper.prefix_caching_attested(att) == (True, None)   # a worker ON is ON
        # every malformed shape is an error, and the gate refuses each
        bad = [
            ({}, "no scheduler_geometry"),
            ({"scheduler_geometry": {"worker_reports": [{"enable_prefix_caching": False}]}},
             "expected is not a mapping"),
            ({"scheduler_geometry": {"expected": {"enable_prefix_caching": "False"},
                                     "worker_reports": [{"enable_prefix_caching": False}]}},
             "not a boolean"),
            ({"scheduler_geometry": {"expected": {"max_num_seqs": 64},
                                     "worker_reports": [{"enable_prefix_caching": False}]}},
             "not a boolean"),
            ({"scheduler_geometry": {"expected": {"enable_prefix_caching": False}, "worker_reports": []}},
             "empty or not a list"),
            ({"scheduler_geometry": {"expected": {"enable_prefix_caching": False},
                                     "worker_reports": [{"batch_invariant_env": None}]}},
             "worker report 0 carries no boolean"),
            ({"scheduler_geometry": {"expected": {"enable_prefix_caching": False},
                                     "worker_reports": [{"enable_prefix_caching": False},
                                                        {"enable_prefix_caching": 0}]}},
             "worker report 1 carries no boolean"),
            ("not a dict", "not a JSON object"),
        ]
        for att_bad, needle in bad:
            value, err = helper.prefix_caching_attested(att_bad)
            assert value is None and err and needle in err, (att_bad, err)
            ok, why = g.prefix_caching_ok({"prefix_caching": False, "prefix_caching_attested": value,
                                           "prefix_caching_attest_error": err})
            assert ok is False and needle in why, (att_bad, why)
        assert "prefix_caching_attested(att)" in (REPO / "scripts" / "lens_gate_standup.py").read_text()

    def test_report_names_the_cache_regime(self):
        ms = g.MODELS["glm46vflash"]
        doc = {"model": dict(key=ms.key, hf=ms.hf, rev=ms.rev, cfg=ms.cfg, bank=ms.bank, branch=ms.branch),
               "geometry": {"max_model_len": 4096, "max_num_seqs": 64}, "batches": [1],
               "timing_batches": [8], "cases": [], "inventory": {}, "uncovered_cases": [],
               "complete": True, "batch_invariant": 0, "exit_code": 0}
        text = g.render_report(doc)
        assert "prefix caching: OFF on every fidelity standup" in text and "cache_salt" in text
        assert f"caching-ON standup per case is the shared-prefill case's ({g.RESUMED}" in text


# --------------------------------------------------------------------------
# the family table + the sbatch pins
# --------------------------------------------------------------------------


class TestFamilyTableAndSbatch:
    SBATCH = REPO / "scripts/sbatch/lens_vectorize_gate.sbatch"

    def test_sbatch_models_pin_the_module_order(self):
        m = re.search(r"^MODELS=\(([^)]*)\)", self.SBATCH.read_text(), re.M)
        assert m and tuple(m.group(1).split()) == g.MODEL_ORDER

    def test_sbatch_clones_the_base_commit_from_the_mirror_and_passes_it(self):
        text = self.SBATCH.read_text()
        assert f'REF_SHA="${{GATE_REFERENCE_SHA:-{g.DEFAULT_REFERENCE_SHA}}}"' in text
        assert 'git clone -q --no-checkout "$MIR" "$REF_DIR"' in text
        assert 'git -C "$REF_DIR" checkout -q "$REF_SHA"' in text
        assert '[ "$REF_HEAD" = "$REF_SHA" ]' in text
        assert 'git -C "$REF_DIR" diff --quiet HEAD --' in text
        assert '[ "$REF_SHA" != "$GATE_SHA" ]' in text
        assert '--reference-repo "$REF_DIR" --reference-sha "$REF_SHA"' in text
        assert "unset SISTEER_LENS_PER_ROW_APPLY" in text
        assert re.search(r"gemma4e4b\|gemma4e2b\)\s+GEOM_REQS=.*tf5141", text)

    def test_configs_pin_the_table(self):
        import yaml
        for ms in g.MODELS.values():
            cfg = yaml.safe_load((REPO / ms.cfg).read_text())
            assert cfg["model"] == ms.hf and cfg["model_revision"] == ms.rev, ms.key
            assert ms.branch

    def test_base_sha_is_the_branch_base(self):
        assert g.DEFAULT_REFERENCE_SHA == "09be969fb676db26a532917e7eb1673de40605f5"
        assert len(g.DEFAULT_REFERENCE_SHA) == 40

    def test_env_lines(self):
        lines = g.shell_env_lines(g.MODELS["llama8b"], REPO)
        assert any(l.startswith("GATE_HF_SNAPSHOT_REL=") and "snapshots/d10aef79" in l for l in lines)


class TestRoutingClass:
    """Prefill-only cases could never satisfy a shadow coverage that demanded
    a counted comparison whose key ends in every exact ladder size. Now each
    case has a ROUTING CLASS derived from its op phase through the worker's
    own predicate: a per-slot case passes with zero counted comparisons and
    is held to the fallback path directly (the SelectionRecord's per-path
    counters); a vectorized case needs a counted comparison attributed to
    some wave at every ladder size."""

    def test_routing_is_derived_from_the_op_phase_not_a_list(self):
        inv = g.served_inventory(4)
        per_slot = {cid for cid in inv if g.case_routing(cid) == g.ROUTING_PER_SLOT}
        assert per_slot == {cid for cid in inv if cid.endswith(":prefill")}
        # caa_static_r joined this set with E45 H2: its op is prefill-only
        # (the class's hook returns early on any seq_len<=1 forward)
        assert {g.parse_case_id(c)["kind"] for c in per_slot} == {
            "alphasteer", "loreft", "caa_static_r"}
        # antidote's positions=last op is phase BOTH: all-ones in decode, so it vectorizes —
        # the review's example list named it prefill-only; the phase field is the authority
        assert g.case_routing("antidote:add_vector:vec:last:skipbos0:both") == g.ROUTING_VECTORIZED
        assert "antidote:add_vector:vec:last:skipbos0:both" in inv
        # the rule IS the worker's
        for cid in inv:
            assert (g.case_routing(cid) == g.ROUTING_VECTORIZED) == \
                lw.phase_vectorizes_in_decode(g.parse_case_id(cid)["phase"]), cid
        for bad in ("garbage", "k:op:shape:all:skipbos0:sometimes", "k:op:shape:all:skipbos0"):
            with pytest.raises(g.GateError, match="routing class"):
                g.case_routing(bad)

    @staticmethod
    def _per_slot_case():
        case = _ok_case()
        case["case_id"] = "alphasteer:prefill_last_matmul:dxd:all:skipbos0:prefill"
        case["routing"] = g.ROUTING_PER_SLOT
        key = "alphasteer:prefill_last_matmul:dxd:all:skipbos0:prefill"
        snap = _shadow_snapshot(batches=())
        snap.update(n_groups_compared=0, n_equal=0, n_unmodified_groups=96, n_layer_calls=96,
                    cells_compared=[],
                    by_key={f"{key}:B{b}": {"n_compared": 0, "n_equal": 0, "n_unequal": 0,
                                            "n_unmodified": 32, "n_shadow_errors": 0,
                                            "max_abs_diff": 0.0, "first_unequal": None}
                            for b in (1, 8, 32)})
        case["shadow"] = snap
        waves = _shadow_waves(compared=0)
        for w in waves:
            w["unmodified_in_wave"] = 32
        case["shadow_waves"] = waves
        # the prefill slots carried every modification (per slot); the decode groups were
        # applied-and-unmodified (grouped applications, zero modifying)
        for side in (g.CANDIDATE, g.SHADOW):
            case["steering_evidence"][side] = _ev(n_mod=100, n_obs=6500, per_slot_mod=100,
                                                  per_slot_app=100, grouped_mod=0, grouped_app=6400,
                                                  mode="stacked")
        return case

    def test_per_slot_case_passes_with_zero_counted_comparisons(self):
        code, checks = _exit_with(self._per_slot_case())
        assert code == 0, {k: v for k, v in checks.items() if not v["ok"]}
        assert checks["shadow_compare"]["ok"] is True
        assert "per-slot routing" in checks["shadow_compare"]["detail"]
        assert "96 decode group(s) unmodified" in checks["shadow_compare"]["detail"]
        for side in (g.CANDIDATE, g.SHADOW):
            assert checks[f"routing_{side}"]["ok"] is True
            assert "100 modifying per-slot" in checks[f"routing_{side}"]["detail"]
        assert g.shadow_check(self._per_slot_case()["shadow"], (1, 8, 32), routing=g.ROUTING_PER_SLOT,
                              waves=None)[0] is True                  # no wave attribution needed

    def test_per_slot_case_is_held_to_the_fallback_path(self):
        case = self._per_slot_case()
        case["steering_evidence"][g.CANDIDATE] = _ev(n_mod=0, per_slot_mod=0, per_slot_app=100,
                                                    grouped_mod=0, grouped_app=6400, mode="stacked")
        code, checks = _exit_with(case)
        assert code == 2 and checks["routing_candidate"]["ok"] is False
        assert "must be > 0" in checks["routing_candidate"]["detail"]
        assert checks["routing_candidate_shadow"]["ok"] is True
        # a grouped modification on a per-slot case is a routing failure
        case = self._per_slot_case()
        case["steering_evidence"][g.SHADOW] = _ev(per_slot_mod=100, per_slot_app=100,
                                                 grouped_mod=5, grouped_app=6400, mode="stacked")
        code, checks = _exit_with(case)
        assert code == 2 and checks["routing_candidate_shadow"]["ok"] is False
        assert "must be 0" in checks["routing_candidate_shadow"]["detail"]
        # no per-path counters (the base commit's record shape) -> fails, never inferred
        case = self._per_slot_case()
        case["steering_evidence"][g.CANDIDATE] = _ev(n_mod=100, mode="stacked")
        code, checks = _exit_with(case)
        assert code == 2 and "no per-path counters" in checks["routing_candidate"]["detail"]
        # an unattributed application is a missing attribution, not a path
        case = self._per_slot_case()
        case["steering_evidence"][g.CANDIDATE] = _ev(per_slot_mod=100, per_slot_app=100,
                                                    grouped_app=6400, unattributed=1, mode="stacked")
        code, checks = _exit_with(case)
        assert code == 2 and "named no apply path" in checks["routing_candidate"]["detail"]
        # the shadow standup's evidence is missing entirely -> fails
        case = self._per_slot_case()
        del case["steering_evidence"][g.SHADOW]
        code, checks = _exit_with(case)
        assert code == 2 and checks["routing_candidate_shadow"]["ok"] is False

    def test_per_slot_case_with_a_counted_comparison_is_a_routing_disagreement(self):
        case = self._per_slot_case()
        case["shadow"] = _shadow_snapshot()                            # 96 counted comparisons
        code, checks = _exit_with(case)
        assert code == 2 and checks["shadow_compare"]["ok"] is False
        assert "disagrees with the worker's grouping" in checks["shadow_compare"]["detail"]

    def test_vectorized_case_fails_with_zero_counted_comparisons(self):
        case = _ok_case()
        snap = _shadow_snapshot()
        snap.update(n_groups_compared=0, n_equal=0)
        for e in snap["by_key"].values():
            e.update(n_compared=0, n_equal=0)
        case["shadow"] = snap
        case["shadow_waves"] = _shadow_waves(compared=0)
        code, checks = _exit_with(case)
        assert code == 3 and checks["shadow_compare"]["ok"] is None
        assert "B=[1, 8, 32]" in checks["shadow_compare"]["detail"]
        # and a candidate whose grouped path modified nothing fails the routing check outright
        case = _ok_case()
        case["steering_evidence"][g.CANDIDATE] = _ev(grouped_mod=0, grouped_app=3200, per_slot_mod=3200,
                                                    per_slot_app=3200, mode="stacked")
        code, checks = _exit_with(case)
        assert code == 2 and checks["routing_candidate"]["ok"] is False
        assert "must be > 0" in checks["routing_candidate"]["detail"]
        # the reference (base commit) is never asked for path counters
        assert "routing_reference" not in checks

    def test_malformed_case_id_fails_the_routing_class(self):
        code, checks = _exit_with(_ok_case(), case_id="garbage")
        assert code == 2 and checks["routing_class"]["ok"] is False
        assert checks["routing_candidate"]["ok"] is False

    def test_selection_evidence_sums_the_path_counters(self, monkeypatch):
        su = types.SimpleNamespace(proxy_url="http://x", server_base_url="http://y", api_key="k",
                                   side=g.CANDIDATE, endpoint={})
        rec = {"n_observations": 10, "all_observations_matched": True,
               "requests": {"r0": {"requested": "c", "n_applications": 10, "n_applications_modifying": 4,
                                   "applied_effective_scale": 1.0}},
               "n_applications_per_slot": 2, "n_applications_grouped": 8, "n_applications_unattributed": 0,
               "n_modifying_per_slot": 1, "n_modifying_grouped": 3, "n_modifying_unattributed": 0}
        monkeypatch.setattr(g, "rpc", lambda su_, method, args=(): {"results": [rec, rec]})
        ev = g.selection_evidence(su)
        assert ev["paths_recorded"] is True
        assert ev["n_applications_per_slot"] == 4 and ev["n_applications_grouped"] == 16
        assert ev["n_modifying_per_slot"] == 2 and ev["n_modifying_grouped"] == 6
        assert g.routing_evidence_ok(ev, g.ROUTING_VECTORIZED)[0] is True
        assert g.routing_evidence_ok(ev, g.ROUTING_PER_SLOT)[0] is False
        # a record without the counters (the base commit) reads as not recorded, never as zeros
        old = {k: v for k, v in rec.items() if not k.startswith("n_applications_") and not k.startswith("n_modifying_")}
        monkeypatch.setattr(g, "rpc", lambda su_, method, args=(): {"results": [old]})
        ev = g.selection_evidence(su)
        assert ev["paths_recorded"] is False and ev["n_modifying_grouped"] is None
        assert g.routing_evidence_ok(ev, g.ROUTING_VECTORIZED) == (
            False, "the worker's SelectionRecord carries no per-path counters (n_applications_per_slot / "
                   "n_applications_grouped)")
        assert g.routing_evidence_ok(None, g.ROUTING_VECTORIZED)[0] is False
        assert g.routing_evidence_ok(_paths(), "sideways")[0] is False

    def test_drive_standup_records_the_whole_selection_evidence(self, monkeypatch):
        """Job 382416: routing_candidate read "the worker's SelectionRecord
        carries no per-path counters" on EVERY family (llama included) while
        routing_candidate_shadow passed -- drive_standup summarized the record
        down to three totals and dropped paths_recorded / the counters that
        routing_evidence_ok reads. The candidate's recorded evidence is the
        whole selection_evidence record, and case_checks passes routing on it."""
        rec_full = {"n_observations": 12, "all_observations_matched": True,
                    "requests": {"r0": {"requested": "c", "n_applications": 12,
                                        "n_applications_modifying": 9, "applied_effective_scale": 1.0}},
                    "n_applications_per_slot": 3, "n_applications_grouped": 9,
                    "n_applications_unattributed": 0, "n_modifying_per_slot": 3,
                    "n_modifying_grouped": 6, "n_modifying_unattributed": 0,
                    "lens_reduction_mode": "stacked"}
        rec_base = {k: v for k, v in rec_full.items()
                    if not k.startswith("n_applications_") and not k.startswith("n_modifying_")
                    and k != "lens_reduction_mode"}
        state = {"rec": rec_full}

        def fake_rpc(su_, method, args=()):
            if method == "sisteer_selection_record":
                return {"results": [state["rec"]]}
            if method == "sisteer_apply_path":
                return {"results": [g.APPLY_PATH_VECTORIZED_STACKED]}
            raise AssertionError(method)
        monkeypatch.setattr(g, "rpc", fake_rpc)
        monkeypatch.setattr(g, "run_batch", lambda url, reqs, **kw: [])
        monkeypatch.setattr(g, "controls_B1", lambda *a, **k: {})
        monkeypatch.setattr(g, "vmodels_of", lambda su_: {"none": "n", "none__lens0": "z", "c": "vc"})
        monkeypatch.setattr(g, "measure_timing", lambda *a, **k: {"skipped": "t"})
        su = types.SimpleNamespace(proxy_url="http://x", side=g.CANDIDATE, commit="c" * 40,
                                   endpoint={"cell_map": {"cell": "c"}, "served_from": {}, "bi_attested": {}})
        ctx = types.SimpleNamespace(batches=(1, 8), max_tokens=4, ignore_eos=True, top_logprobs=1,
                                    repeats=2)
        prompts = [([{"role": "user", "content": "p"}], 5)] * 8
        rec = g.drive_standup(ctx, su, cell="cell", prompts=prompts, timing=False)
        ev = rec["evidence"]
        assert ev["paths_recorded"] is True and ev["n_modifying"] == 9 and ev["n_observations"] == 12
        assert ev["n_modifying_grouped"] == 6 and ev["n_modifying_per_slot"] == 3
        assert ev["n_applications_unattributed"] == 0 and "by_cell" in ev
        assert ev["lens_reduction_mode"] == "stacked"      # the attested mode rides along
        case = _ok_case()
        case["steering_evidence"][g.CANDIDATE] = ev
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert checks["routing_candidate"]["ok"] is True and checks["evidence_candidate"]["ok"] is True
        assert checks["reduction_mode_candidate"]["ok"] is True
        assert g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True,
                             reduction_mode="per_row")["reduction_mode_candidate"]["ok"] is False
        # the base commit's record (the reference) still reads as unrecorded, never as zeros
        state["rec"] = rec_base
        su.side = g.REFERENCE
        ev = g.drive_standup(ctx, su, cell="cell", prompts=prompts, timing=False)["evidence"]
        assert ev["paths_recorded"] is False and ev["n_modifying_grouped"] is None
        assert ev["lens_reduction_mode"] is None and ev["lens_reduction_modes_seen"] == []
        assert ev["n_modifying"] == 9                       # the evidence_* check still has its totals
        # WRITER -> READER, end to end: the payload is the worker's OWN serializer
        # (SelectionRecord.as_dict(), what the sisteer_selection_record RPC returns,
        # wrapped in the pinned route's {"results": [...]}) after grouped and per-slot
        # applications -- the gate's reader files it with the counters intact
        live = lw.SelectionRecord(reduction_mode=la.REDUCTION_MODE_STACKED)
        obj = types.SimpleNamespace(cell_id="c", expect_hash="h", effective_scale=1.0)
        for i in range(3):
            live.observe(req_id=f"r{i}", layer=1, requested="c", requested_hash="h", requested_scale=1.0,
                         applied=obj, modified=True, path=lw.SELECTION_PATH_GROUPED)
        live.observe(req_id="p", layer=1, requested="c", requested_hash="h", requested_scale=1.0,
                     applied=obj, modified=True, path=lw.SELECTION_PATH_PER_SLOT)
        state["rec"] = json.loads(json.dumps(live.as_dict()))
        su.side = g.CANDIDATE
        ev = g.drive_standup(ctx, su, cell="cell", prompts=prompts, timing=False)["evidence"]
        assert ev["paths_recorded"] is True
        assert (ev["n_modifying_grouped"], ev["n_modifying_per_slot"], ev["n_applications_unattributed"]) == (3, 1, 0)
        assert ev["lens_reduction_mode"] == la.REDUCTION_MODE_STACKED == lw.SelectionRecord.as_dict(live)["lens_reduction_mode"]
        assert g.reduction_mode_evidence_ok(ev, "stacked") == (True, "the worker attests lens_reduction_mode='stacked'; "
                                                                      "the gate runs the 'stacked' mode")
        assert g.routing_evidence_ok(ev, g.ROUTING_VECTORIZED)[0] is True
        assert g.routing_evidence_ok(ev, g.ROUTING_PER_SLOT)[0] is False
        # and the worker's RPC is that serializer, not a projection of it
        src = (REPO / "src/sisteer/capeval/serving/lens_worker.py").read_text()
        body = src[src.index("def sisteer_selection_record("):src.index("def sisteer_apply_path(")]
        assert "steering.selection_record.as_dict()" in body

    def test_drive_shadow_attributes_each_wave_and_reads_the_paths(self, monkeypatch):
        """drive_shadow snapshots the record around every wave: the per-wave
        entries carry that wave's counted comparisons and group rows, a
        vectorized size is re-driven until a wave counts, a per-slot case
        drives each size once, and the SelectionRecord is reset then read."""
        calls = []
        state = {"n": 0, "keys": {}}

        def fake_rpc(su_, method, args=()):
            calls.append((method, tuple(args)))
            if method == "sisteer_shadow_compare":
                snap = {"enabled": True, "n_groups_compared": state["n"], "n_unequal": 0,
                        "n_unmodified_groups": 0, "n_shadow_errors": 0, "n_record_errors": 0,
                        "by_key": {k: {"n_compared": v} for k, v in state["keys"].items()}}
                if args and args[0]:
                    state["n"] = 0
                    state["keys"] = {}
                return {"results": [snap]}
            if method == "sisteer_selection_record":
                return {"results": [{"n_observations": 5, "all_observations_matched": True, "requests": {},
                                     "lens_reduction_mode": "stacked", **{k: 1 for k in g.PATH_COUNTERS}}]}
            if method == "sisteer_apply_path":
                return {"results": [g.APPLY_PATH_SHADOW]}
            raise AssertionError(method)

        waves_run = []

        def fake_run_batch(url, reqs, **kw):
            waves_run.append(len(reqs))
            B = len(reqs)
            # the FIRST B=8 wave counts nothing (all requests failed); every other wave counts,
            # and admission splits B=32 into groups of 3 and 5 rows
            if B == 8 and waves_run.count(8) == 1:
                return []
            groups = {1: {1: 1}, 8: {8: 1}, 32: {3: 4, 5: 2}}[B]
            for rows, n in groups.items():
                state["n"] += n
                k = f"psr:relu_gated_vector:gemv:response:skipbos0:both:B{rows}"
                state["keys"][k] = state["keys"].get(k, 0) + n
            return []
        monkeypatch.setattr(g, "rpc", fake_rpc)
        monkeypatch.setattr(g, "run_batch", fake_run_batch)
        monkeypatch.setattr(g, "vmodels_of", lambda su_: {"none": "n", "none__lens0": "z", "c": "vc"})
        monkeypatch.setattr(g, "measure_timing", lambda *a, **k: {"skipped": "t"})
        su = types.SimpleNamespace(proxy_url="http://x", side=g.SHADOW, commit="c" * 40,
                                   endpoint={"cell_map": {"cell": "c"}, "served_from": {}, "bi_attested": {}})
        ctx = types.SimpleNamespace(batches=(1, 8, 32), max_tokens=4, ignore_eos=True, top_logprobs=1)
        prompts = [([{"role": "user", "content": "p"}], 5)] * 32
        rec = g.drive_shadow(ctx, su, cell="cell", prompts=prompts, timing=False, routing=g.ROUTING_VECTORIZED)
        assert waves_run == [1, 8, 8, 32]                              # B=8 re-driven once
        w = rec["shadow_waves"]
        assert [(x["B"], x["attempt"], x["compared_in_wave"], x["accepted"]) for x in w] == [
            (1, 1, 1, True), (8, 1, 0, False), (8, 2, 1, True), (32, 1, 6, True)]
        assert w[3]["rows_compared_in_wave"] == [3, 5]
        assert g.compared_per_wave_size(w, ctx.batches) == {1: 1, 8: 1, 32: 6}
        assert rec["routing"] == g.ROUTING_VECTORIZED and rec["evidence"]["paths_recorded"] is True
        assert rec["evidence"]["n_modifying_grouped"] == 1
        assert rec["evidence"]["lens_reduction_mode"] == "stacked"
        assert rec["reduction_tolerance"]["ok"] is True and rec["reduction_tolerance"]["reduction_mode"] == "stacked"
        # the SelectionRecord was reset before the waves and read after them
        sel = [c for c in calls if c[0] == "sisteer_selection_record"]
        assert sel[0] == ("sisteer_selection_record", (True,)) and len(sel) == 2
        assert g.shadow_check(rec["shadow"], ctx.batches, routing=rec["routing"],
                              waves=rec["shadow_waves"])[0] is True
        # a per-slot case: one wave per size, accepted with zero comparisons
        waves_run.clear()
        state.update(n=0, keys={})
        monkeypatch.setattr(g, "run_batch", lambda url, reqs, **kw: waves_run.append(len(reqs)))
        rec = g.drive_shadow(ctx, su, cell="cell", prompts=prompts, timing=False, routing=g.ROUTING_PER_SLOT)
        assert waves_run == [1, 8, 32]
        assert all(x["compared_in_wave"] == 0 and x["accepted"] for x in rec["shadow_waves"])
        assert g.shadow_check(rec["shadow"], ctx.batches, routing=g.ROUTING_PER_SLOT,
                              waves=rec["shadow_waves"])[0] is True


class TestPureB1Unconditional:
    """``--batches 8,32`` was accepted and the only hard cross-server
    comparison (pure_B1) silently never ran, leaving exit 0 reachable. B=1
    now runs outside the configurable ladder, a case without a filed B=1
    verdict fails a hard check, and a malformed ladder refuses loudly."""

    def test_gate_batches_always_starts_with_one(self):
        assert g.gate_batches("8,32") == (1, 8, 32)
        assert g.gate_batches("1,8,32") == (1, 8, 32)
        assert g.gate_batches("32, 8,8") == (1, 8, 32)
        assert g.gate_batches("") == (1,) and g.gate_batches("1") == (1,)
        assert g.gate_batches((8, 32)) == (1, 8, 32)
        for bad in ("8,x", "0,8", "-1", "8,1.5", "8,,3.0", "1e3"):
            with pytest.raises(g.GateError, match="not a positive integer"):
                g.gate_batches(bad)
        with pytest.raises(g.GateError):
            g.gate_batches((True, 8))
        assert g.parse_int_list("64,8,32", "--timing-batches") == (8, 32, 64)
        with pytest.raises(g.GateError, match="--timing-batches"):
            g.parse_int_list("8,zz", "--timing-batches")
        assert g.DEFAULT_BATCHES[0] == 1 and g.gate_batches(g.DEFAULT_BATCHES) == g.DEFAULT_BATCHES

    def test_main_runs_b1_whatever_the_ladder_says(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run_gate(ctx, **kw):
            captured["ctx"] = ctx
            return {"exit_code": 0}
        monkeypatch.setattr(g, "run_gate", fake_run_gate)
        monkeypatch.setattr(g, "plan_inventory", lambda *a, **k: {})
        monkeypatch.setattr(g, "header_len_for", lambda *a, **k: 4)
        args = ["--model", "llama8b", "--reference-repo", str(tmp_path), "--results-root",
                str(tmp_path / "r"), "--objects-root", str(tmp_path / "o"), "--candsi-root",
                str(tmp_path / "c")]
        assert g.main(args + ["--batches", "8,32"]) == 0
        assert captured["ctx"].batches == (1, 8, 32)
        monkeypatch.setenv("GATE_BATCHES", "8,32")                     # the sbatch seam, same parser
        assert g.main(args + ["--batches", "32"]) == 0
        assert captured["ctx"].batches == (1, 32)
        with pytest.raises(g.GateError, match="not a positive integer"):
            g.main(args + ["--batches", "8,x"])
        with pytest.raises(g.GateError, match="--timing-batches"):
            g.main(args + ["--timing-batches", "8,x"])

    def test_exit_0_is_impossible_without_the_b1_verdict(self):
        case = _ok_case()
        del case["verdicts"]["pure_B1"]
        code, checks = _exit_with(case)
        assert code == 2 and checks["pure_B1_present"]["ok"] is False
        assert "not configurable away" in checks["pure_B1_present"]["detail"]
        # a B=1 entry that is only evidence does not count
        case = _ok_case()
        case["verdicts"]["pure_B1"] = g.evidence_verdict([[_sig("a")]] * 2, [[_sig("a")]] * 2)
        code, checks = _exit_with(case)
        assert code == 2 and checks["pure_B1_present"]["ok"] is False
        # a filed FAIL / UNSTABLE is present (and fails / is inconclusive on its own)
        for v, want in ((g.FAIL, 2), (g.UNSTABLE_VEC, 2), (g.UNSTABLE_REF, 3), (g.PASS, 0)):
            case = _ok_case()
            case["verdicts"]["pure_B1"] = {"verdict": v}
            code, checks = _exit_with(case)
            assert checks["pure_B1_present"]["ok"] is True and code == want, v
        # compare_case refuses a ladder without B=1 outright
        rec = {"batches": {"pure_B8": [[_sig("a")]] * 2, "mixed_B8": [[_sig("a")]] * 2}, "controls_B1": {}}
        with pytest.raises(g.GateError, match="omits B=1"):
            g.compare_case(types.SimpleNamespace(batches=(8,)), rec, rec, None)

    def test_docstring_exit_table_and_hard_checks_match_the_code(self):
        doc = g.__doc__
        m = re.search(r"EXIT CODES \(``overall_exit_code``.*?\n(.*?)\n\nRuns from", doc, re.S)
        assert m, "the EXIT CODES paragraph is gone from the module docstring"
        documented = {int(x) for x in re.findall(r"^  (\d)  ", m.group(1), re.M)}
        assert documented == {0, 2, 3}
        ok = _ok_case()
        ok["checks"] = g.case_checks(ok, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        observed = {g.overall_exit_code({"cases": [ok], "complete": True}),
                    g.overall_exit_code({"cases": [ok], "complete": False}),
                    g.overall_exit_code({"cases": [ok], "complete": True, "tree_error": "x"})}
        assert observed == documented
        # the docstring's HARD CHECKS list names every check key case_checks emits
        with_legacy = _ok_case()
        with_legacy["apply_path"][g.LEGACY] = g.APPLY_PATH_PER_ROW
        keys = set(g.case_checks(with_legacy, reference_sha=BASE, base_sha=BASE, have_oracle=True))
        keys |= set(g.case_checks(dict(_ok_case(), case_id="garbage"), reference_sha=BASE, base_sha=BASE,
                                  have_oracle=True))
        hard = doc[doc.index("HARD CHECKS"):doc.index("THE AUTHORITATIVE IDENTITY CHECK")]
        for key in sorted(keys):
            stem = re.sub(r"_(reference|candidate_shadow|candidate|legacy_ab)$", "", key)
            assert re.search(rf"(?<![A-Za-z_]){re.escape(stem)}(_\*)?(?![A-Za-z_])", hard), key
        assert {"routing_candidate", "routing_candidate_shadow", "pure_B1_present", "shadow_compare",
                "batch_invariant_candidate_shadow", "reduction_mode_candidate",
                "reduction_mode_candidate_shadow"} <= keys
        # and the report footer states the same table
        assert "a missing pure_B1" in g.render_report(
            {"model": dict(key="llama8b", hf="x", rev="r" * 40, cfg="c", bank="b", branch="p"),
             "geometry": {"max_model_len": 1, "max_num_seqs": 1}, "batches": [1], "timing_batches": [],
             "cases": [], "inventory": {}, "uncovered_cases": [], "complete": True, "batch_invariant": 0,
             "exit_code": 0})


class TestReport:
    def test_render_report_smoke(self):
        ms = g.MODELS["qwen35_4b"]
        case = _ok_case()
        case.update(mixed_B=32, timing={g.REFERENCE: {"8": {"step_ms": 70.7}, "32": {"step_ms": 257.9}},
                                        g.CANDIDATE: {"8": {"step_ms": 31.0}, "32": {"step_ms": 33.0},
                                                      "64": {"step_ms": 36.0}}})
        case["verdicts"] = {"pure_B1": {"verdict": g.FAIL, "detail": "request 3: logprobs"},
                            "pure_B8": g.evidence_verdict([[_sig("a")]] * 2, [[_sig("a")]] * 2),
                            "pure_B32": g.evidence_verdict([[_sig("a")]] * 2, [[_sig("b")]] * 2),
                            "mixed_B32": g.evidence_verdict([[_sig("a")]] * 2, [[_sig("a")]] * 2)}
        case["timing"][g.SHADOW] = {"8": {"step_ms": 60.0}}
        case["checks"] = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        doc = {"model": dict(key=ms.key, hf=ms.hf, rev=ms.rev, cfg=ms.cfg, bank=ms.bank, branch=ms.branch),
               "geometry": {"max_model_len": 4096, "max_num_seqs": 64},
               "trees": {"reference": {"path": "/ref", "sha": BASE},
                         "candidate": {"path": "/cand", "sha": "a" * 40}},
               "node": "TESTHOST", "job_id": "1", "batches": [1, 8, 32], "timing_batches": [8, 32, 64],
               "passthrough_timing": {"8": {"step_ms": 29.4}, "32": {"step_ms": 27.5}, "64": {}},
               "exit_code": 2, "batch_invariant": 0,
               "inventory": {case["case_id"]: {"kind": "psr", "arms": ["psr_kl_all"],
                                               "sources": ["adapter-probe"], "routing": "vectorized"}},
               "uncovered_cases": [], "complete": True,
               "cases": [case, {"case_id": "loreft:affine_subspace:rankK:all:skipbos0:prefill",
                                "objects": "synthetic", "error": "boom"}]}
        text = g.render_report(doc)
        assert "hybrid GDN" in text and "70.7/257.9" in text and "FAIL" in text
        assert "ERROR: boom" in text and "exit code: 2" in text and "complete=True" in text
        assert "09be969fb676" in text and "pure_B1: FAIL" in text and "pure_B32: evidence" in text
        assert "| 96/0 |" in text                                     # the shadow column
        assert "reduction mode: stacked (SISTEER_LENS_STACKED_REDUCTIONS=1" in text
        assert g.REDUCTION_CONTRACT in text                          # the header states the contract
        assert "elementwise + per-row mode: exact" in text            # ... and so does the footer
        assert "reduction-op elements unequal 0/" in text
        assert "routing vectorized" in text
        assert "group rows compared (kernel-shape evidence, never coverage): [1, 8, 32]" in text
        assert "counted per submitted wave size {1: 32, 8: 32, 32: 32}" in text
        assert "a missing pure_B1" in text                            # the footer's exit table


# --------------------------------------------------------------------------
# the recorded gate runs, replayed through the 2026-09-06 rule
# --------------------------------------------------------------------------

RECORDS_DIR = REPO / "tests/fixtures/lens_gate_shadow_records"
PSR_RESPONSE_CASE = "psr:relu_gated_vector:gemv:response:skipbos0:both"
FAMILIES = ("gemma4e4b", "glm46vflash", "llama8b", "nemotron3nano4b", "qwen35_4b")


def _records():
    out = {}
    for fn in sorted(RECORDS_DIR.glob("*.json")):
        d = json.loads(fn.read_text())
        out[(d["candidate_sha"], d["family"])] = d
    return out


class TestRealGateRecords:
    """The shadow records of the eight recorded gate runs (numeric fixtures:
    five families at c8e0f45 — the five-family gate the ff35913 evidence
    analysed — and the three ff35913 evidence runs), replayed through the
    2026-09-06 rule. What holds: every LINEAR key reads as before (ok, edit-
    ulps <= 1); every exact and elementwise case is ok; the multigroup cases
    still FAIL on their unclassifiable dash keys. What does NOT hold, on
    purpose: no recorded run carries the per-key gate aggregates the gated
    rule reads — c8e0f45 recorded no gate stats at all, and ff35913 recorded
    |delta gate|/sum|hw| only inside each key's first_unequal / worst_unequal
    sub-records — so every gated key with unequal groups is MISSING EVIDENCE,
    never a pass and never a numeric verdict; the qwen35_4b / nemotron3nano4b
    PSR cases the old rule failed are certified by the five-family re-gate at
    the new sha, not by these records. The rule's arithmetic is checked on
    the sub-samples ff35913 did record."""

    RECORDS = _records()

    def _tol(self, sha, family, case_id):
        return g.reduction_tolerance(self.RECORDS[(sha, family)]["cases"][case_id]["shadow"], reduction_mode="stacked")

    def test_fixture_inventory(self):
        assert {k for k in self.RECORDS if k[0] == "c8e0f45"} == {("c8e0f45", f) for f in FAMILIES}
        assert {k for k in self.RECORDS if k[0] == "ff35913"} == {("ff35913", f) for f in ("gemma4e4b", "nemotron3nano4b", "qwen35_4b")}
        for (sha, fam), d in self.RECORDS.items():
            assert d["reduction_mode"] == "stacked"
            assert len(d["cases"]) == (14 if sha == "c8e0f45" else 1), (sha, fam)
            if sha == "ff35913":
                assert set(d["cases"]) == {PSR_RESPONSE_CASE}
            for body in d["cases"].values():
                assert "node" not in body["shadow"] and "job_id" not in body["shadow"]

    def test_linear_keys_pass_the_unchanged_rule_on_every_family(self):
        n_linear_cases = 0
        for fam in FAMILIES:
            d = self.RECORDS[("c8e0f45", fam)]
            for cid, body in d["cases"].items():
                if cid == "multigroup":
                    continue
                tol = self._tol("c8e0f45", fam, cid)
                if tol["gated_keys_unequal"]:
                    continue                                   # the gated cases: the next test
                assert tol["ok"] is True, (fam, cid, tol["detail"])
                assert body["recorded_shadow_compare_ok"] is True, (fam, cid)     # the old rule agreed
                if tol["linear_keys_unequal"]:
                    n_linear_cases += 1
                    assert tol["max_ulps_edit_linear"] <= 1.0 and tol["max_ulps_edit_gated"] is None
                    assert "linear ops: largest" in tol["detail"] and "(<= 4)" in tol["detail"]
                    assert all(r["class"] == g.OP_CLASS_REDUCTION_LINEAR for r in tol["per_key"] if r["n_unequal"])
                    # every bound holds with >= 5x margin on every family (pooled 1.1e-5 at worst,
                    # llama8b lora_all response; row 7.8e-4; group 1.5e-4)
                    assert tol["frac_unequal_elems"] <= g.REDUCTION_FRAC_TOL / 5
                    assert tol["max_frac_row"] <= g.REDUCTION_ROW_FRAC_TOL / 5
                    assert tol["max_frac_group"] <= g.REDUCTION_GROUP_FRAC_TOL / 5
                else:
                    assert tol["n_unequal"] == 0 and tol["detail"].startswith("exact (stacked mode)")
        assert n_linear_cases >= 12                              # lora_all on all five, lora rank-1 on two

    def test_gated_keys_without_gate_aggregates_are_missing_evidence_never_a_verdict(self):
        """Every recorded run predates BOTH the per-key aggregates and the
        observed-gate seam (their gate sub-records are recomputations), so
        every gated key with unequal groups is missing evidence."""
        seen = []
        for (sha, fam), d in self.RECORDS.items():
            for cid in d["cases"]:
                if cid == "multigroup":
                    continue
                tol = self._tol(sha, fam, cid)
                if not tol["gated_keys_unequal"]:
                    continue
                seen.append((sha, fam, cid))
                assert tol["ok"] is False, (sha, fam, cid)
                assert sorted(tol["keys_lacking_gate_stats"]) == sorted(tol["gated_keys_unequal"]), (sha, fam, cid)
                assert "lacks gate stats on every unequal group" in tol["detail"]
                assert "missing evidence, never kernel variance" in tol["detail"]
                # no numeric verdict on missing evidence — neither the gate bound nor the edit unit
                assert "> 16 x 2^-24" not in tol["detail"]
                assert "bf16 ulps of the per-row edit beyond one output ulp > 4" not in tol["detail"]
                assert tol["max_gate_diff_rel_sum"] is None
                assert all(r["class"] == g.OP_CLASS_REDUCTION_GATED for r in tol["per_key"] if r["n_unequal"])
        # the c8e0f45 PSR cases with unequal groups (gemma / nemotron / qwen: 'all' and 'response'),
        # and the three ff35913 evidence runs
        assert sorted(seen) == sorted(
            [("c8e0f45", fam, cid) for fam in ("gemma4e4b", "nemotron3nano4b", "qwen35_4b")
             for cid in ("psr:relu_gated_vector:gemv:all:skipbos0:both", PSR_RESPONSE_CASE)]
            + [("ff35913", fam, PSR_RESPONSE_CASE) for fam in ("gemma4e4b", "nemotron3nano4b", "qwen35_4b")])
        # what the OLD rule recorded on those runs: the edit-ulp FAILs the change is about (qwen 5,
        # nemotron 23) and gemma's pass — the new rule replaces both verdicts with "re-gate"
        for sha in ("c8e0f45", "ff35913"):
            for fam, old_ok, ulps in (("qwen35_4b", False, 5.0), ("nemotron3nano4b", False, 23.0), ("gemma4e4b", True, 1.0)):
                body = self.RECORDS[(sha, fam)]["cases"][PSR_RESPONSE_CASE]
                assert body["recorded_shadow_compare_ok"] is old_ok, (sha, fam)
                assert self._tol(sha, fam, PSR_RESPONSE_CASE)["max_ulps_edit_gated"] == ulps, (sha, fam)

    def test_multigroup_cases_still_fail_on_unclassifiable_dash_keys(self):
        for fam in FAMILIES:
            tol = self._tol("c8e0f45", fam, "multigroup")
            assert tol["ok"] is False and tol["unclassified_keys"], fam
            assert "names no op" in tol["detail"] and "never a default class" in tol["detail"]
            assert any(":-:-:-:-:-:B" in k for k in tol["unclassified_keys"]), fam
            assert self.RECORDS[("c8e0f45", fam)]["cases"]["multigroup"]["recorded_shadow_compare_ok"] is False

    def test_rule_arithmetic_on_the_gate_subsamples_ff35913_did_record(self):
        """Every gate sub-record ff35913 kept (first / worst per key, the
        case-wide worst) sits inside the bound with no ReLU flip; and a record
        whose per-key aggregates are patched in from those sub-records passes
        the rule with the edit-ulp unit informational. This is the rule's
        arithmetic on real magnitudes, NOT a certification: the aggregates
        over EVERY unequal group are what the re-gate records."""
        U = g.REDUCTION_GATE_UNIT
        expect = {"qwen35_4b": (4.24, 5.0), "nemotron3nano4b": (0.616, 23.0), "gemma4e4b": (2.65, 1.0)}
        for fam, (rel_sum_units, ulps) in expect.items():
            sh = copy.deepcopy(self.RECORDS[("ff35913", fam)]["cases"][PSR_RESPONSE_CASE]["shadow"])
            subs = [sh["worst_unequal"]["gate"]]
            for e in sh["by_key"].values():
                if not e.get("n_unequal"):
                    continue
                keyed = [x["gate"] for x in (e.get("first_unequal"), e.get("worst_unequal")) if x and x.get("gate")]
                assert keyed, fam
                subs += keyed
                e["gate_max_diff_rel_sum"] = max(x["gate_max_diff_rel_sum"] for x in keyed)
                e["gate_max_flip_rel_sum"] = None
                e["gate_n_groups_with_stats"] = e["n_unequal"]
                e["gate_n_errors"] = 0
                # the gate-edit identity, the observed provenance and the exact-rows count were NOT
                # measured by ff35913 (recomputation-based worker): placeholders so the gate-bound
                # arithmetic can be exercised on the real ratios
                e["gate_edit_consistency_max_ulps"] = 0.0
                e["gate_n_groups_observed"] = e["n_unequal"]
                e["gate_exact_rows_unequal_elems"] = 0
                e["gate_n_groups_recon"] = e["n_unequal"]
                e["gate_write_recon_mismatch_elems"] = 0
            for x in subs:
                assert x["gate_max_diff_rel_sum"] <= g.REDUCTION_GATE_REL_SUM_TOL and x["gate_n_relu_flips"] == 0
                assert x["gate_ops"] == ["relu_gated_vector"]
            tol = g.reduction_tolerance(sh, reduction_mode="stacked")
            assert tol["ok"] is True, (fam, tol["detail"])
            assert tol["max_gate_diff_rel_sum"] / U == pytest.approx(rel_sum_units, abs=0.02), fam
            assert tol["max_ulps_edit_gated"] == ulps and tol["max_ulps_edit_linear"] is None
            assert f"edit-ulps {ulps:g} recorded, not a rule" in tol["detail"]
            assert tol["max_gate_flip_rel_sum"] is None and tol["n_gate_relu_flips"] == 0
        # the 4.24 x 2^-24 is qwen's B31 first-unequal group, not the 0.8 at the worst-by-edit-ulps
        # element the evidence note quoted: the case-wide maximum over every group was never recorded
        assert expect["qwen35_4b"][0] > 4 * 0.8


# --------------------------------------------------------------------------
# THE DECLARED-RENDER CASES ride the gate (the case bodies are pinned in
# tests/test_lens_declared_render_gate.py)
# --------------------------------------------------------------------------


class TestDeclaredRenderCasesRideTheGate:
    def test_they_are_not_inventory_cases_and_do_not_move_the_inventory_order(self):
        inv = ["psr:relu_gated_vector:gemv:all:skipbos0:both", "lora:lora_oproj:rank1:all:skipbos0:both"]
        assert g.gate_case_order(inv, None) == [g.MULTIGROUP] + inv
        # a selection naming a declared-render case beside an inventory case: the inventory
        # order is unchanged, the declared-render case is supplied by declared_render_order
        only = [inv[1], "all_turns_render_check"]
        assert g.gate_case_order(inv, only) == [inv[1]]
        assert g.declared_render_order(g.MODELS["gemma4e4b"], only) == ["all_turns_render_check"]
        for cid in g.DECLARED_RENDER_CASES:
            assert cid not in g.served_inventory(4)
        assert set(g.DECLARED_RENDER_FAMILIES) == set(g.DECLARED_RENDER_CASES) == set(g.DECLARED_RENDER_CHECKERS)

    def test_run_gate_skips_the_oracle_for_a_declared_render_only_run_and_threads_the_cases(self):
        import inspect
        src = inspect.getsource(g.run_gate)
        assert "declared_order = declared_render_order(ms, only_cases)" in src
        assert 'doc["uncovered_cases"] += list(declared_order)' in src
        assert "if declared_render_only(only_cases):" in src
        assert src.count("_declared_render_case(ctx, case_id, cand_sha=cand_sha, bank_dir=bank_dir") == 2
        # the oracle standup and the inventory cases are exactly as before
        assert 'cell="none", defenses_dir=empty, tag="passthrough"' in src
        assert "(REFERENCE, ctx.reference, ref_sha, None, None)" in src

    def test_their_checks_and_errors_ride_overall_exit_code(self):
        kr = {"case_id": "declared_thinking", "kind": "declared_render",
              "checks": {"thinking_on_served": {"ok": True, "detail": "200, 64 tokens"}}}
        assert g.overall_exit_code({"cases": [kr], "complete": True}) == 0
        bad = {"case_id": "declared_thinking", "checks": {"thinking_on_served": {"ok": False, "detail": "status 400"}}}
        assert g.overall_exit_code({"cases": [bad], "complete": True}) == 2
        err = {"case_id": "all_turns_render_check", "error": "GateError: the bank holds no artifact"}
        assert g.overall_exit_code({"cases": [err], "complete": True}) == 3
        # a selected-but-never-run case stays uncovered: exit 3, never a pass by silence
        assert g.overall_exit_code({"cases": [], "complete": False, "uncovered_cases": ["declared_thinking"]}) == 3

    def test_start_standup_argv_is_unchanged_unless_a_case_asks_for_a_salt_policy(self):
        import inspect
        src = inspect.getsource(g.start_standup)
        assert inspect.signature(g.start_standup).parameters["lens_salt_policy"].default is None
        assert 'if lens_salt_policy is not None:' in src and 'cmd += ["--lens-salt-policy", lens_salt_policy]' in src
        assert src.index('"--prefix-caching", prefix_caching]') < src.index('"--lens-salt-policy"')
        # the declared-render flag (S-G2) rides after it, only when a declared-render case asks
        assert inspect.signature(g.start_standup).parameters["declared_render"].default is False
        assert 'if declared_render:' in src and 'cmd += ["--declared-render"]' in src
        assert src.index('"--lens-salt-policy"') < src.index('"--declared-render"')
