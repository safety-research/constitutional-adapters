"""THE SHARED-PREFILL CASE of scripts/lens_vectorize_gate.py — the CPU half
(2026-09-06; the lens half of resumed prefills for per-position ops,
tests/test_lens_resumed_prefill.py).

Pins: the ``resumed_prefill`` hard check's truth table over the drive's
record (served + attested for a resumable case, refused + typed marker for
a non-resumable one, inconclusive when not run, not-applicable on a hybrid
family, every missing fact a FAIL); the case-id rule vs the adapters'
predicate; the refusal marker against the refusals lens_batch / lens_worker
actually raise; the upstream request shape (the proxy's selector key and
cache_salt); the closeness evidence; the standup helper's regime flag and
``prefix_caching_on_ok``; that the check rides into ``case_checks`` and the
exit code; and that the Modal wrapper needs no change (the case is a per-
inventory-case standup inside ``run_gate``, driven by the inventory).
"""

import inspect
import json
import sys
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
sys.path.insert(0, _SCRIPTS_DIR)

import lens_vectorize_gate as g  # noqa: E402
assert Path(g.__file__).resolve() == (REPO / "scripts" / "lens_vectorize_gate.py").resolve(), \
    f"the bare name resolved to {g.__file__}, not the reviewed gate in scripts/ (sys.path order)"
from sisteer.capeval.serving import lens_adapters as la  # noqa: E402
from sisteer.capeval.serving import lens_batch as lb  # noqa: E402
from sisteer.capeval.serving import lens_worker as lw  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402

from test_lens_vectorize_gate import BASE, _ok_case, _resumed_ok_record, _sig  # noqa: E402
from test_petri_lens_worker import HIDDEN, _obj, _registry, _sel  # noqa: E402


class TestLiteralMirrors:
    def test_selector_key_and_attest_fields_mirror_the_modules(self):
        assert g.UPSTREAM_EXTRA_ARGS_KEY == px.UPSTREAM_EXTRA_ARGS_KEY == "vllm_xargs"
        rec = lw.SelectionRecord()
        rec.observe(req_id="r", layer=1, requested="c", requested_hash="h", requested_scale=None,
                    applied=None, modified=True, path=lw.SELECTION_PATH_PER_SLOT,
                    resumed=lw.ResumedPrefill(cached=5, computed=4))
        entry = rec.as_dict()["requests"]["r"]
        assert set(g.RESUMED_ATTEST_FIELDS) <= set(entry)
        assert entry["prefill_resumed"] is True and entry["prefill_cached_positions"] == 5

    def test_refusal_marker_is_in_both_refusals(self):
        seg = lb.Segment(start=0, end=4, q=4, computed=5, seq_len=9)
        with pytest.raises(lb.LensBatchError) as ei:
            lb.check_tracked_segment(seg, prompt_len=9, req_id="r")
        assert g.RESUMED_REFUSAL_MARKER in str(ei.value)
        twin = _obj("static__scale0", [la.HookOp(op=la.OP_ADD_VECTOR, layer=1, vector=torch.ones(HIDDEN),
                                                  phase=la.PHASE_PREFILL)], scale=0.0)
        with pytest.raises(lw.LensWorkerError) as ei:
            lw.build_request_slots(
                query_start_loc=[0, 4], seq_lens=[9], num_actual_tokens=4, positions=[5, 6, 7, 8],
                req_ids=["twin"], raw_selectors=[_sel(twin, scale=0.0)],
                token_lookup={"twin": list(range(200, 209))}.__getitem__,
                prompt_len_lookup={"twin": 9}.__getitem__, registry=_registry(twin), allow_resumed=True)
        assert g.RESUMED_REFUSAL_MARKER in str(ei.value)
        # the apply-level typed refusal spells it too
        with pytest.raises(la.ResumedPrefillRefusal) as ei:
            la.apply_hook_ops(torch.zeros(4, HIDDEN),
                              [la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(HIDDEN),
                                         positions=la.POSITIONS_LAST)], 1.0, layer=0,
                              ctx=la.HookContext(phase=la.PHASE_PREFILL, positions=(5, 6, 7, 8), cached=5))
        assert g.RESUMED_REFUSAL_MARKER in str(ei.value)


class TestCaseRule:
    @pytest.mark.parametrize("cid, ok", [
        ("psr:relu_gated_vector:gemv:all:skipbos0:both", True),
        ("psr:relu_gated_vector:gemv:response:skipbos0:both", False),
        ("vector_all:add_vector:vec:all:skipbos1:both", True),
        ("lora_all:lora_oproj:rankK:all:skipbos0:both", True),
        ("lora_all:lora_oproj:rankK:response:skipbos0:both", False),
        ("antidote:add_vector:vec:last:skipbos0:both", False),
        ("alphasteer:prefill_last_matmul:dxd:all:skipbos0:prefill", False),
        ("loreft:affine_subspace:rankK:all:skipbos0:prefill", False),
        ("cosine_gated_r:prefill_last_cosine_gated:other:last:skipbos0:prefill", False),
    ])
    def test_rule_from_case_id(self, cid, ok):
        assert g.resumed_rule_from_case_id(cid) is ok

    def test_unclassified_op_refuses(self):
        with pytest.raises(la.AdapterError, match="classified neither"):
            g.resumed_rule_from_case_id("k:brand_new_op:vec:all:skipbos0:both")

    @pytest.mark.parametrize("cid, policy", [
        ("psr:relu_gated_vector:gemv:all:skipbos0:both", "always"),
        ("psr:relu_gated_vector:gemv:response:skipbos0:both", "conditional"),
        ("lora_all:lora_oproj:rankK:response:skipbos0:both", "conditional"),
        ("static_r:add_vector:vec:response:skipbos0:both", "conditional"),
        ("antidote:add_vector:vec:last:skipbos0:both", "never"),
        ("loreft:affine_subspace:rankK:all:skipbos0:prefill", "never"),
        ("alphasteer:prefill_last_matmul:dxd:all:skipbos0:prefill", "never"),
    ])
    def test_policy_from_case_id(self, cid, policy):
        assert g.resumed_policy_from_case_id(cid) == policy
        assert g.RESUMED_POLICIES == la.RESUMED_POLICIES
        assert (g.RESUMED_NEVER, g.RESUMED_ALWAYS, g.RESUMED_CONDITIONAL) == (
            la.RESUMED_NEVER, la.RESUMED_ALWAYS, la.RESUMED_CONDITIONAL)
        assert g.RESUMED_CONTAINED_ERROR_TYPE == la.ResumedPrefillRefusal.__name__

    def test_case_resumed_policy_reads_the_served_objects_ops_and_cross_checks(self, tmp_path):
        header_len = 4
        for arm, positions, want in (("vector_all", "all", "always"), ("vector_all", "response", "conditional"),
                                     ("alphasteer", None, "never")):
            kw = {} if positions is None else {"positions": positions}
            cell, ddir = g.write_synthetic_object(arm, g.INVENTORY_DIMS, tmp_path / f"{arm}-{positions}",
                                                  header_len, **kw)
            kind, ops = g.adapt_object_dir(ddir / cell, arm=arm, header_len=header_len)
            obj = {"arm": arm, "cell": cell, "source": str(ddir / cell / "object.pt"),
                   "case_id": g.case_id_of(kind, ops)}
            assert g.case_resumed_policy(obj, header_len=header_len) == want
            assert g.case_resumable(obj, header_len=header_len) is (want == "always")
            # a misfiled case id (the rule disagrees with the ops) refuses: flip the
            # positions / phase fields of the id while the served ops stay what they are
            c = g.parse_case_id(obj["case_id"])
            flipped = dict(c, positions=("response" if want == "always" else "all"), phase="both")
            bad_id = (f"{flipped['kind']}:{flipped['op']}:{flipped['shape']}:{flipped['positions']}:"
                      f"skipbos{int(flipped['skip_bos'])}:{flipped['phase']}")
            if g.resumed_policy_from_case_id(bad_id) != want:
                with pytest.raises(g.GateError, match="disagree"):
                    g.case_resumed_policy(dict(obj, case_id=bad_id), header_len=header_len)


class TestUpstreamShape:
    def test_payload_mirrors_the_proxy_rewrite(self):
        ctx = _ctx()
        sel = {"petri_object": "si_constitution__psr_kl_all", "expect_hash": "h" * 16}
        p = g.upstream_payload(ctx, "base/model", [{"role": "user", "content": "x"}], cache_salt="s|1", selector=sel)
        assert p["model"] == "base/model" and p["cache_salt"] == "s|1" and p[g.UPSTREAM_EXTRA_ARGS_KEY] == sel
        assert p["temperature"] == 0.0 and p["logprobs"] is True and p["return_token_ids"] is True
        assert p["max_tokens"] == ctx.max_tokens and p["top_logprobs"] == ctx.top_logprobs and p["stream"] is False
        q = g.upstream_payload(ctx, "base/model", [{"role": "user", "content": "x"}], cache_salt="s|2")
        assert g.UPSTREAM_EXTRA_ARGS_KEY not in q and q["cache_salt"] == "s|2"
        # a client-supplied cache_salt / selector is exactly what the PROXY forbids: the gate
        # bypasses the proxy on purpose (its lens salts are request-unique until the proxy half lands)
        assert "cache_salt" in px.FORBIDDEN_CLIENT_KEYS

    def test_prompt_pair_shares_the_longest_passage(self):
        prompts = [([{"role": "user", "content": "Passage 1. short"}], 5),
                   ([{"role": "user", "content": "Passage 5. " + "mill " * 250}], 800),
                   ([{"role": "user", "content": "Passage 6. " + "river " * 400}], 1200)]
        a, b = g.resumed_prompt_pair(prompts)
        base = prompts[-1][0][0]["content"]
        assert a[0]["content"].startswith(base) and b[0]["content"].startswith(base)
        assert a[0]["content"] != b[0]["content"] and a[0]["role"] == b[0]["role"] == "user"
        # the multi-turn pair: the same first turn (user + assistant), a second user turn sharing
        # the second-longest passage and diverging in its tail — the first turn's header is deep
        # inside the shared prefix
        ma, mb = g.resumed_multiturn_pair(prompts)
        assert [m["role"] for m in ma] == [m["role"] for m in mb] == ["user", "assistant", "user"]
        assert ma[:2] == mb[:2] and ma[0]["content"].startswith(base) and ma[1]["content"] == g.MULTITURN_ANSWER
        second = prompts[-2][0][0]["content"]
        assert ma[2]["content"].startswith(second) and mb[2]["content"].startswith(second)
        assert ma[2]["content"] != mb[2]["content"]


class TestCloseness:
    def test_equal_and_divergent(self):
        s1 = _sig("abcd", [-0.5, -0.6, -0.7, -0.8])
        assert g.signature_closeness(s1, _sig("abcd", [-0.5, -0.6, -0.7, -0.8])) == {
            "equal": True, "first_token_divergence": None, "n_tokens": [4, 4],
            "max_abs_dlogprob_before_divergence": 0.0, "pos0_max_abs_dlogprob_top": 0.0,
            "pos0_top_tokens_compared": 2}
        c = g.signature_closeness(s1, _sig("abXd", [-0.5, -0.61, -0.7, -0.8], ids=[100, 101, 999, 103]))
        assert c["equal"] is False and c["first_token_divergence"] == 2
        assert abs(c["max_abs_dlogprob_before_divergence"] - 0.01) < 1e-9
        # same token ids, a logprob moved: not equal, no token divergence, the delta recorded
        c = g.signature_closeness(s1, _sig("abcd", [-0.5, -0.6, -0.7, -0.8001]))
        assert c["equal"] is False and c["first_token_divergence"] is None
        assert abs(c["max_abs_dlogprob_before_divergence"] - 0.0001) < 1e-9
        c = g.signature_closeness(s1, _sig("abc", [-0.5, -0.6, -0.7]))
        assert c["first_token_divergence"] == 3 and c["n_tokens"] == [4, 3]

    def test_one_request_attestation(self):
        entry = {"prefill_resumed": True, "prefill_cached_positions": 1200, "prefill_computed_positions": 9,
                 "n_applications_modifying": 32, "n_applications_modifying_resumed": 32, "layers": [1]}
        a = g.one_request_attestation({"chatcmpl-1": entry})
        assert a == {"n_entries": 1, "req_id": "chatcmpl-1", **{k: entry[k] for k in g.RESUMED_ATTEST_FIELDS}}
        assert g.one_request_attestation({})["error"] and g.one_request_attestation({"a": {}, "b": {}})["n_entries"] == 2
        assert g.one_request_attestation(None)["n_entries"] is None


def _ctx(**over):
    kw = dict(ms=g.MODELS["llama8b"], candidate=Path("/c"), reference=Path("/r"), reference_sha=BASE,
              results_dir=Path("/tmp/x"), max_model_len=4096, max_num_seqs=64, gpu_mem_util="0.9",
              ready_timeout_s=10, batches=(1, 8, 32), timing_batches=(8,), max_tokens=64, repeats=2,
              top_logprobs=5, ignore_eos=True, warmup_s=1.0, measure_s=1.0, manifest="m", legacy_ab=False,
              batch_invariant=0)
    kw.update(over)
    return g.GateContext(**kw)


class TestResumedPrefillCheck:
    def test_green_records(self):
        ok, detail = g.resumed_prefill_check(_resumed_ok_record(policy=g.RESUMED_ALWAYS))
        assert ok is True and detail.startswith("always") and "attested RESUMED" in detail
        assert "cached 1200" in detail and "repeat resumed too" in detail and "multi-turn resume served" in detail
        ok, detail = g.resumed_prefill_check(_resumed_ok_record(policy=g.RESUMED_CONDITIONAL))
        assert ok is True and detail.startswith("conditional") and "CONTAINED" in detail
        assert "identical-prompt repeat" in detail and "ResumedPrefillRefusal" in detail
        ok, detail = g.resumed_prefill_check(_resumed_ok_record(policy=g.RESUMED_NEVER))
        assert ok is True and detail.startswith("never") and "typed refusal" in detail
        assert "the caching floor" in detail             # the passthrough closeness rides as evidence
        # the pre-policy bool spelling still reads
        assert g.resumed_prefill_check(_resumed_ok_record(resumable=True))[0] is True
        assert g.resumed_prefill_check(_resumed_ok_record(resumable=False))[0] is True
        rec = _resumed_ok_record(policy=g.RESUMED_NEVER)
        del rec["policy"], rec["resumable"]
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "does not carry a resumed-prefill policy" in detail

    def test_conditional_protocol_failures(self):
        # the unsafe multi-turn resume served STEERED (no claimed refusal): the classifier did not decide
        rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        rec["requests"]["steered_multiturn_warm"]["claimed_failure"] = None
        rec["multiturn_claimed_failure"] = None
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "no contained refusal was claimed" in detail
        # a claimed record of the wrong type is not the containment
        rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        rec["requests"]["steered_multiturn_warm"]["claimed_failure"] = {"error_type": "AdapterError",
                                                                        "error": "header not located"}
        assert g.resumed_prefill_check(rec)[0] is False
        # the check reads the ENTRY's claim; the top-level alias alone cannot clear it
        rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        rec["multiturn_claimed_failure"] = None            # alias cleared, entry still carries the refusal
        assert g.resumed_prefill_check(rec)[0] is True
        # an engine-fatal refusal is the WRONG class for a conditional cell
        rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        rec["requests"]["steered_multiturn_warm"] = {"status": None, "error": "URLError: connection refused"}
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "engine-fatal refusal is the wrong class" in detail
        # the engine did not keep serving afterwards
        rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        rec["requests"]["steered_after_refusal"] = {"status": 500, "error": "EngineCore died"}
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "did not keep serving" in detail
        # the containment leaked onto the next request
        rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        rec["requests"]["steered_after_refusal"]["claimed_failure"] = {"error_type": "ResumedPrefillRefusal"}
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "steered_after_refusal" in detail and "CONTAINED a failure" in detail
        # the SAFE shape must still serve and attest on a conditional cell
        rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        rec["attest"]["steered_warm_b"]["prefill_resumed"] = False
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "did not attest a RESUMED prefill" in detail
        # the repeat is evidence on a conditional cell: a served-and-attested repeat is green too
        rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        rec["attest"]["steered_warm_b_repeat"] = {"n_entries": 1, "req_id": "chatcmpl-3", "prefill_resumed": True,
                                                  "prefill_cached_positions": 1200, "prefill_computed_positions": 9,
                                                  "n_applications_modifying": 32, "n_applications_modifying_resumed": 32}
        rec["repeat_claimed_failure"] = None
        assert g.resumed_prefill_check(rec)[0] is True
        # an always cell must serve the multi-turn resume
        rec = _resumed_ok_record(policy=g.RESUMED_ALWAYS)
        rec["attest"]["steered_multiturn_warm"]["prefill_resumed"] = False
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "steered_multiturn_warm" in detail

    def test_not_run_is_inconclusive_and_skipped_is_not_applicable(self):
        ok, detail = g.resumed_prefill_check(None)
        assert ok is None and "--no-resumed-prefill" in detail
        ok, detail = g.resumed_prefill_check({"skipped": "hybrid-recurrent target"})
        assert ok is True and detail.startswith("not applicable")
        ok, detail = g.resumed_prefill_check({"error": "GateError: standup died", "resumable": True})
        assert ok is False and "errored" in detail
        assert g.resumed_prefill_check("junk")[0] is False

    def test_resumable_protocol_failures(self):
        # the warm request refused where it should be served
        rec = _resumed_ok_record(resumable=True)
        assert rec["policy"] == g.RESUMED_ALWAYS
        rec["requests"]["steered_warm_b"] = {"status": 500, "error": "boom"}
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "was not served" in detail
        # served but not attested resumed (the resume never happened: e.g. the salt differed)
        rec = _resumed_ok_record(resumable=True)
        rec["attest"]["steered_warm_b"]["prefill_resumed"] = False
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "did not attest a RESUMED prefill" in detail
        # attested resumed but nothing modified the resumed rows (a silently unsteered tail)
        rec = _resumed_ok_record(resumable=True)
        rec["attest"]["steered_warm_b"]["n_applications_modifying_resumed"] = 0
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "no op modified the resumed rows" in detail
        # the warm steered output equals the warm passthrough: steering did not bite
        rec = _resumed_ok_record(resumable=True)
        rec["requests"]["steered_warm_b"]["signature"] = rec["requests"]["passthrough_warm_b"]["signature"]
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "bit-identical to passthrough_warm_b" in detail
        # the cold twin attested resumed: the unique salt did not isolate it
        rec = _resumed_ok_record(resumable=True)
        rec["attest"]["steered_cold_b"]["prefill_resumed"] = True
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "expected False" in detail
        # the repeat missing
        rec = _resumed_ok_record(resumable=True)
        del rec["requests"]["steered_warm_b_repeat"]
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "avg@k shape" in detail
        # several entries in the reset record: no attestation can be attributed
        rec = _resumed_ok_record(resumable=True)
        rec["attest"]["steered_warm_b"] = {"n_entries": 2, "error": "expected exactly one request entry"}
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "no single SelectionRecord entry" in detail
        # a missing cached / computed count is missing evidence
        rec = _resumed_ok_record(resumable=True)
        rec["attest"]["steered_warm_b"]["prefill_cached_positions"] = None
        assert g.resumed_prefill_check(rec)[0] is False

    def test_non_resumable_protocol_failures(self):
        rec = _resumed_ok_record(resumable=False)
        rec["requests"]["steered_warm_b"] = {"status": 200, "signature": _sig("abcd")}
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "backstop did not fire" in detail
        rec = _resumed_ok_record(resumable=False)
        rec["refusal_marker_in_log"] = False
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "typed refusal" in detail and g.RESUMED_REFUSAL_MARKER in detail
        rec = _resumed_ok_record(resumable=False)
        del rec["resumable"], rec["policy"]
        assert g.resumed_prefill_check(rec)[0] is False
        rec = _resumed_ok_record(resumable=False)
        rec["signature_errors"] = ["steered_cold_b: no token_ids"]
        assert g.resumed_prefill_check(rec)[0] is False
        rec = _resumed_ok_record(resumable=False)
        rec["requests"]["steered_cold_b"]["status"] = 500
        assert g.resumed_prefill_check(rec)[0] is False

    def test_unsafe_multiturn_positive_stays_green_and_is_the_only_required_claim(self):
        rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        assert rec["requests"]["steered_multiturn_warm"]["claimed_failure"]["error_type"] == "ResumedPrefillRefusal"
        case = dict(_ok_case(), resumed_prefill=rec)
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert checks["resumed_prefill"]["ok"] is True
        assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 0
        # every OTHER steered entry of that green record carries an EMPTY claim
        for name, e in rec["requests"].items():
            if name.startswith("steered_") and name not in ("steered_multiturn_warm", "steered_warm_b_repeat"):
                assert "claimed_failure" in e and e["claimed_failure"] is None, name
        # the repeat is evidence: a served-and-empty repeat is green, a block-alignment refusal is green,
        # a failure of another class is not, an unclaimed repeat is missing evidence
        ok_rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        ok_rec["requests"]["steered_warm_b_repeat"]["claimed_failure"] = None
        ok_rec["repeat_claimed_failure"] = None
        assert g.resumed_prefill_check(ok_rec)[0] is True
        bad = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        bad["requests"]["steered_warm_b_repeat"]["claimed_failure"] = {"error_type": "AdapterError", "error": "x"}
        ok, detail = g.resumed_prefill_check(bad)
        assert ok is False and "another class" in detail
        bad = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        del bad["requests"]["steered_warm_b_repeat"]["claimed_failure"], bad["repeat_claimed_failure"]
        ok, detail = g.resumed_prefill_check(bad)
        assert ok is False and "never claimed" in detail

    @pytest.mark.parametrize("policy, name", [
        (g.RESUMED_CONDITIONAL, "steered_cold_b"),
        (g.RESUMED_CONDITIONAL, "steered_writer_a"),
        (g.RESUMED_CONDITIONAL, "steered_warm_b"),
        (g.RESUMED_CONDITIONAL, "steered_multiturn_writer"),
        (g.RESUMED_CONDITIONAL, "steered_after_refusal"),
        (g.RESUMED_ALWAYS, "steered_cold_b"),
        (g.RESUMED_ALWAYS, "steered_writer_a"),
        (g.RESUMED_ALWAYS, "steered_warm_b"),
        (g.RESUMED_ALWAYS, "steered_warm_b_repeat"),
        (g.RESUMED_ALWAYS, "steered_multiturn_writer"),
        (g.RESUMED_ALWAYS, "steered_multiturn_warm"),
        (g.RESUMED_NEVER, "steered_cold_b"),
        (g.RESUMED_NEVER, "steered_writer_a"),
    ])
    def test_a_claimed_failure_on_an_expected_success_request_fails_the_gate(self, policy, name):
        """THE PRODUCTION CONTRACT (proxy._refuse_poisoned_lens_response): a
        worker-contained hook failure returns 200 from vLLM and the proxy 500s
        it from the claimed ledger — so 200 + a signature is not "served". A
        claimed failure on ANY expected-success steered request rejects, and
        the whole hard gate exits 2 (never merely the helper returning
        False); a steered entry whose ledger was never claimed is missing
        evidence and rejects too. Worst shape covered: a failed WRITER that
        seeded the cache."""
        for claim in ({"req_id": "x", "error_type": "ResumedPrefillRefusal", "error": "boundary"},
                      {"req_id": "x", "error_type": "AdapterError", "error": "header not located"}):
            rec = _resumed_ok_record(policy=policy)
            rec["requests"][name]["claimed_failure"] = claim
            case = dict(_ok_case(), resumed_prefill=rec)
            checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
            assert checks["resumed_prefill"]["ok"] is False, (policy, name, claim)
            assert name in checks["resumed_prefill"]["detail"] and "CONTAINED a failure" in checks["resumed_prefill"]["detail"]
            assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 2
        # unclaimed: missing evidence
        rec = _resumed_ok_record(policy=policy)
        del rec["requests"][name]["claimed_failure"]
        case = dict(_ok_case(), resumed_prefill=rec)
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert checks["resumed_prefill"]["ok"] is False and "never claimed" in checks["resumed_prefill"]["detail"]
        assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 2
        # and the untouched green record of the same policy exits 0
        case = dict(_ok_case(), resumed_prefill=_resumed_ok_record(policy=policy))
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 0

    def test_the_drive_claims_the_ledger_for_every_steered_response(self):
        src = inspect.getsource(g.drive_resumed_prefill)
        body = src[src.index("def one("):src.index("# the passthrough floor")]
        assert "if steered:" in body and 'entry["claimed_failure"] = claim_request_failure(su, entry.get("id"))' in body
        # the aliases the report reads are the entries' claims, never a second claim
        assert 'rec["warm_claimed_failure"] = warm.get("claimed_failure")' in src
        assert 'rec["multiturn_claimed_failure"] = mt.get("claimed_failure")' in src
        assert 'rec["after_refusal_claimed_failure"] = after.get("claimed_failure")' in src
        assert src.count("claim_request_failure(") == 1      # one claim per response, inside one()

    def test_check_rides_into_case_checks_and_the_exit_code(self):
        ok = _ok_case()
        checks = g.case_checks(ok, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert checks["resumed_prefill"]["ok"] is True
        assert g.overall_exit_code({"cases": [dict(ok, checks=checks)], "complete": True}) == 0
        # not run: inconclusive, exit 3 — never a pass
        case = dict(_ok_case(), resumed_prefill=None)
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert checks["resumed_prefill"]["ok"] is None
        assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 3
        # a failed protocol: exit 2
        rec = _resumed_ok_record(resumable=False)
        rec["refusal_marker_in_log"] = False
        case = dict(_ok_case(), resumed_prefill=rec)
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert checks["resumed_prefill"]["ok"] is False
        assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 2
        # a conditional case's unsafe resume served without a claim: exit 2
        rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        rec["requests"]["steered_multiturn_warm"]["claimed_failure"] = None
        rec["multiturn_claimed_failure"] = None
        case = dict(_ok_case(), resumed_prefill=rec)
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        assert checks["resumed_prefill"]["ok"] is False
        assert g.overall_exit_code({"cases": [dict(case, checks=checks)], "complete": True}) == 2
        rec = _resumed_ok_record(policy=g.RESUMED_NEVER)
        rec["refusal_marker_in_log"] = False
        case = dict(_ok_case(), resumed_prefill=rec)
        checks = g.case_checks(case, reference_sha=BASE, base_sha=BASE, have_oracle=True)
        # the report names the case's verdict
        doc = {"model": dict(key="llama8b", hf="x", rev="r" * 40, cfg="c", bank="b", branch="p"),
               "geometry": {"max_model_len": 1, "max_num_seqs": 1}, "batches": [1], "timing_batches": [],
               "cases": [dict(case, checks=checks)], "inventory": {}, "uncovered_cases": [], "complete": True,
               "batch_invariant": 0, "exit_code": 2}
        text = g.render_report(doc)
        assert f"shared prefill ({g.RESUMED}): FAIL" in text and g.RESUMED in text
        assert "resumed prefills for per-position ops" in text


class TestProxyDecision:
    """THE PROXY HALF inside the gate: which namespace the shipped proxy would
    mint for the case's cell, whether the case drove it, and whether the LIVE
    proxy agreed. Pinned because the shared-prefill drive otherwise proves only
    what the WORKER does with a salt the gate invented — it must also prove the
    proxy hands that salt out (``always``) or refuses to (``never`` /
    ``conditional``)."""

    def _endpoint(self, policy, cell="c1"):
        return {"resumed_prefill_policy_cells": {cell: policy},
                "resumed_prefill_policy_error": None}

    def test_the_decision_comes_from_the_shipped_proxy(self):
        for policy, regime in ((g.RESUMED_ALWAYS, px.SALT_REGIME_CELL_STABLE),
                               (g.RESUMED_CONDITIONAL, px.SALT_REGIME_REQUEST_UNIQUE),
                               (g.RESUMED_NEVER, px.SALT_REGIME_REQUEST_UNIQUE)):
            d = g.proxy_salt_decision(self._endpoint(policy), harness_cell="c1",
                                      intervention_hash="ab" * 8, virtual_model="petri__c1")
            assert d["attested_policy"] == policy and d["cache_salt_regime"] == regime
            # the salt IS the proxy's, not a mirror
            assert d["salt"] == px.RoutingProxy.cache_salt_for(
                px.RouteEntry(virtual_model="petri__c1", cell_id="c1", cell_class="gate",
                              serving_identity="proxy_vllm+lens+interleaved",
                              intervention_hash="ab" * 8),
                lens=True, resumed_prefill_policy=policy) or regime == px.SALT_REGIME_REQUEST_UNIQUE
            assert d["salt_is_stable_across_mints"] is (regime == px.SALT_REGIME_CELL_STABLE)
            assert d["salt"] != d["unhooked_salt"]

    def test_an_unattested_standup_reads_as_no_policy(self):
        d = g.proxy_salt_decision({"resumed_prefill_policy_cells": None,
                                   "resumed_prefill_policy_error": "predates the landing"},
                                  harness_cell="c1", intervention_hash="ab" * 8,
                                  virtual_model="petri__c1")
        assert d["policy_cells_present"] is False and d["attested_policy"] is None
        assert d["cache_salt_regime"] == px.SALT_REGIME_REQUEST_UNIQUE
        # ...and the check refuses such a record rather than reading it as a pass
        rec = _resumed_ok_record(policy=g.RESUMED_ALWAYS)
        rec["proxy_decision"] = dict(rec["proxy_decision"], policy_cells_present=False,
                                     attest_error="predates the landing")
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "published no attested resumed_prefill_policy_cells" in detail

    def test_a_record_without_the_decision_fails(self):
        rec = _resumed_ok_record(policy=g.RESUMED_ALWAYS)
        rec.pop("proxy_decision")
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "no proxy_decision" in detail

    def test_a_never_cell_offered_a_shared_namespace_fails(self):
        """E62 constraint 2: the PROXY must not route a broadcast-kind cell onto
        a shared prefix; a decision that says it would is a hard failure even
        though the worker would still refuse."""
        rec = _resumed_ok_record(policy=g.RESUMED_NEVER)
        rec["proxy_decision"] = dict(rec["proxy_decision"],
                                     cache_salt_regime=px.SALT_REGIME_CELL_STABLE)
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "never be routed onto a shared prefix" in detail

    def test_an_always_case_must_drive_the_proxys_own_salt(self):
        rec = _resumed_ok_record(policy=g.RESUMED_ALWAYS)
        rec["salts"] = dict(rec["salts"], cell="gate|made-up|salt")
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "the proxy mints" in detail
        rec = _resumed_ok_record(policy=g.RESUMED_ALWAYS)
        rec["proxy_decision"] = dict(rec["proxy_decision"], forced_stable_salt=True)
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "FORCED gate salt" in detail

    def test_a_conditional_case_must_force_its_salt(self):
        """The proxy declines a shared namespace for a conditional cell, so the
        case can only reach the worker's per-request decision by forcing one —
        a case that claims it drove the proxy's salt is describing something
        that cannot have happened."""
        rec = _resumed_ok_record(policy=g.RESUMED_CONDITIONAL)
        rec["proxy_decision"] = dict(rec["proxy_decision"], forced_stable_salt=False)
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "must force one to reach the worker" in detail

    def test_the_attested_policy_must_match_the_served_ops(self):
        rec = _resumed_ok_record(policy=g.RESUMED_ALWAYS)
        rec["proxy_decision"] = dict(rec["proxy_decision"],
                                     attested_policy=g.RESUMED_CONDITIONAL)
        ok, detail = g.resumed_prefill_check(rec)
        assert ok is False and "would disagree about this cell" in detail

    def test_the_live_probe_must_have_served_and_agreed(self):
        for probe, want in (({"status": None, "error": "URLError"}, "did not serve"),
                            ({"status": 500}, "did not serve"),
                            ({"status": 200, "cache_salt_regime": px.SALT_REGIME_REQUEST_UNIQUE,
                              "resumed_prefill_policy": g.RESUMED_ALWAYS},
                             "out of process"),
                            ({"status": 200, "cache_salt_regime": px.SALT_REGIME_CELL_STABLE,
                              "resumed_prefill_policy": g.RESUMED_CONDITIONAL},
                             "the standup attested")):
            rec = _resumed_ok_record(policy=g.RESUMED_ALWAYS)
            rec["proxy_decision"] = dict(rec["proxy_decision"], live_probe=probe)
            ok, detail = g.resumed_prefill_check(rec)
            assert ok is False and want in detail, probe

    def test_the_probe_prompt_cannot_seed_the_cases_own_prefix(self):
        """The one request that crosses the proxy must not share a prefix with
        the A/B pair, or reading the decision would perturb the measurement."""
        prompts = [([{"role": "user", "content": "Passage 1. short"}], 5),
                   ([{"role": "user", "content": "Passage 6. " + "river " * 400}], 1200)]
        a, b = g.resumed_prompt_pair(prompts)
        probe = g.PROXY_PROBE_MESSAGES[0]["content"]
        for msgs in (a, b):
            assert not msgs[0]["content"].startswith(probe)
            assert not probe.startswith(msgs[0]["content"][:20])
        assert len(probe) < 60

    def test_the_probe_is_sent_before_anything_engine_fatal(self):
        """A ``never`` cell's forced warm request kills the engine by design, so
        the probe has to precede every request of the drive."""
        src = inspect.getsource(g.drive_resumed_prefill)
        assert src.index("proxy_regime_probe") < src.index('one("passthrough_cold_b"')

    def test_a_green_record_passes_for_every_policy(self):
        for policy in (g.RESUMED_ALWAYS, g.RESUMED_CONDITIONAL, g.RESUMED_NEVER):
            ok, detail = g.resumed_prefill_check(_resumed_ok_record(policy=policy))
            assert ok is True, (policy, detail)


class TestStandupRegime:
    def test_prefix_caching_on_ok(self):
        good = {"prefix_caching_requested": "default", "prefix_caching": True, "prefix_caching_attested": True,
                "prefix_caching_attest_error": None}
        assert g.prefix_caching_on_ok(good) == (True, "prefix caching ON: requested default, composed True, "
                                                      "attested True")
        for bad, why in ((dict(good, prefix_caching_requested="off"), "regime"),
                         (dict(good, prefix_caching=False), "not applicable"),
                         (dict(good, prefix_caching=None), "not applicable"),
                         (dict(good, prefix_caching_attested=False), "attests enable_prefix_caching=False"),
                         (dict(good, prefix_caching_attested=None), "attests enable_prefix_caching=None"),
                         (dict(good, prefix_caching_attest_error="JSONDecodeError"), "could not be read"),
                         ({}, "regime"), (None, "no endpoint")):
            ok, detail = g.prefix_caching_on_ok(bad)
            assert ok is False and why in detail, (bad, detail)
        # the OFF check is unchanged and refuses the ON endpoint
        assert g.prefix_caching_ok(good)[0] is False

    def test_start_standup_threads_the_regime(self):
        src = inspect.getsource(g.start_standup)
        assert '"--prefix-caching", prefix_caching' in src
        assert "prefix_caching_on_ok if prefix_caching == PREFIX_CACHING_DEFAULT else prefix_caching_ok" in src
        assert g.PREFIX_CACHING_OFF == "off" and g.PREFIX_CACHING_DEFAULT == "default"
        assert inspect.signature(g.start_standup).parameters["prefix_caching"].default == g.PREFIX_CACHING_OFF
        # run_gate stands the resumed case up on the CANDIDATE tree only, caching default, mode set,
        # and drives it by the served object's POLICY
        rsrc = inspect.getsource(g.run_gate)
        assert "side=RESUMED, tree=ctx.candidate" in rsrc and "prefix_caching=PREFIX_CACHING_DEFAULT" in rsrc
        assert "resumed_case_applicable(ms)" in rsrc and "case_resumed_policy(obj, header_len=header_len)" in rsrc
        assert "if ctx.resumed_prefill:" in rsrc and "policy=policy" in rsrc
        dsrc = inspect.getsource(g.drive_resumed_prefill)
        # the production contract: EVERY steered response's failure ledger is CLAIMED inside one()
        # exactly as the proxy claims it, and a fresh request after the contained refusal proves the
        # engine alive
        assert 'entry["claimed_failure"] = claim_request_failure(su, entry.get("id"))' in dsrc
        assert '"steered_after_refusal"' in dsrc
        assert "sisteer_take_request_failure" in inspect.getsource(g.claim_request_failure)
        # the helper's flag exists with the right vocabulary and default
        helper = (REPO / "scripts" / "lens_gate_standup.py").read_text()
        assert '"--prefix-caching", choices=("off", "default"), default="off"' in helper
        assert 'prefix_caching = False if args.prefix_caching == "off" else None' in helper

    def test_resumed_case_applicable_reads_the_targets_recurrence(self, monkeypatch):
        from sisteer.capeval.serving import vllm_server as vs
        monkeypatch.setattr(vs, "hybrid_recurrent_model", lambda hf, rev: hf.startswith("Qwen"))
        ok, why = g.resumed_case_applicable(g.MODELS["qwen35_4b"])
        assert ok is False and "structurally OFF" in why
        ok, why = g.resumed_case_applicable(g.MODELS["llama8b"])
        assert ok is True and "resolves ON" in why

        def boom(hf, rev):
            raise OSError("no config cached")
        monkeypatch.setattr(vs, "hybrid_recurrent_model", boom)
        ok, why = g.resumed_case_applicable(g.MODELS["llama8b"])
        assert ok is True and "could not resolve" in why      # never a silent skip

    def test_cli_flag_and_context_default(self):
        import dataclasses
        f = {x.name: x for x in dataclasses.fields(g.GateContext)}["resumed_prefill"]
        assert f.default is True
        src = inspect.getsource(g.main)
        assert '"--no-resumed-prefill"' in src and "resumed_prefill=not args.no_resumed_prefill" in src

    def test_docstring_names_the_check_and_the_case(self):
        doc = g.__doc__
        hard = doc[doc.index("HARD CHECKS"):doc.index("THE AUTHORITATIVE IDENTITY CHECK")]
        assert "resumed_prefill" in hard
        assert "THE SHARED-PREFILL CASE" in doc and "candidate_resumed" in doc
        # the docstring states plainly that warm-vs-cold is not bit-identical even unhooked
        assert "NOT\nbit-identical even without a hook" in doc or "NOT bit-identical even without a hook" in doc.replace("\n", " ")


class TestModalWrapperNeedsNoChange:
    def test_the_case_is_inside_run_gate_and_the_wrapper_passes_no_new_flag(self):
        """The Modal wrapper builds the gate's argv (gate_argv) and runs the
        whole inventory; the shared-prefill case is a per-inventory-case
        standup inside run_gate, on by default, so the wrapper is untouched:
        its argv carries no resumed flag, and the helper it stages is the
        candidate tree's (which has the new --prefix-caching flag)."""
        import ast
        src = (REPO / "scripts/modal/lens_vectorize_gate.py").read_text()
        assert "--no-resumed-prefill" not in src and "resumed_prefill" not in src
        assert "scripts/lens_gate_standup.py" in src
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "gate_argv")
        assert fn is not None


# --------------------------------------------------------------------------
# the cache-isolation gate's sharing case (step 3b) — the pure parts
# --------------------------------------------------------------------------


import importlib.util  # noqa: E402


def _cache_iso():
    spec = importlib.util.spec_from_file_location("petri_cache_isolation_gate",
                                                  REPO / "scripts" / "petri_cache_isolation_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCacheIsolationSharingCase:
    def test_pick_sharing_cell_prefers_a_live_resumable_arm_and_states_the_skip(self):
        ci = _cache_iso()
        cells = {"static_r_matched": {"resumed_prefill_ok": False, "forced_scale": None},
                 "static_r_matched__scale0": {"resumed_prefill_ok": False, "forced_scale": 0.0},
                 "alphasteer_official": {"resumed_prefill_ok": False, "forced_scale": None}}
        cell, why = ci.pick_sharing_cell(cells)
        assert cell is None and "no live registry cell has resumable ops" in why
        assert "alphasteer_official" in why and "static_r_matched" in why and "__scale0" not in why
        cells["si_safety__psr_all"] = {"resumed_prefill_ok": True, "forced_scale": None}
        cells["si_safety__psr_all__scale0"] = {"resumed_prefill_ok": True, "forced_scale": 0.0}
        assert ci.pick_sharing_cell(cells) == ("si_safety__psr_all", None)   # the twin is never picked

    def test_resumable_registry_cells_reads_the_loaded_registrys_ops(self, tmp_path, monkeypatch):
        ci = _cache_iso()
        from sisteer.capeval.serving import lens_steer as ls
        arm = _obj("arm-all", [la.HookOp(op=la.OP_ADD_VECTOR, layer=1, vector=torch.ones(HIDDEN), skip_bos=True)])
        last = _obj("arm-last", [la.HookOp(op=la.OP_ADD_VECTOR, layer=1, vector=torch.ones(HIDDEN),
                                           positions=la.POSITIONS_LAST)])
        twin = _obj("arm-all__scale0", arm.ops, scale=0.0)
        reg = _registry(arm, last, twin)
        (tmp_path / ls.REGISTRY_INDEX).write_text(json.dumps({"version": 1, "entries": [
            {"cell_id": "arm-all", "forced_scale": None}, {"cell_id": "arm-last", "forced_scale": None},
            {"cell_id": "arm-all__scale0", "forced_scale": 0.0}]}))
        monkeypatch.setattr(ls.WorkerObjectRegistry, "load", classmethod(lambda cls, d: reg))
        cells = ci.resumable_registry_cells(tmp_path)
        assert cells == {"arm-all": {"resumed_prefill_ok": True, "forced_scale": None},
                         "arm-last": {"resumed_prefill_ok": False, "forced_scale": None},
                         "arm-all__scale0": {"resumed_prefill_ok": True, "forced_scale": 0.0}}
        assert ci.pick_sharing_cell(cells) == ("arm-all", None)

    def test_sharing_case_verdict(self):
        ci = _cache_iso()
        good = {"cell": "c", "warm_status": 200, "warm_bites": True,
                "warm_attestation": {"n_entries": 1, "req_id": "r", "prefill_resumed": True,
                                     "prefill_cached_positions": 128, "prefill_computed_positions": 11,
                                     "n_applications_modifying": 10, "n_applications_modifying_resumed": 10}}
        ci.sharing_case_verdict(good)          # no raise
        for over, why in (({"warm_status": 500}, "was not served"),
                          ({"warm_attestation": {"n_entries": 2}}, "exactly one"),
                          ({"warm_attestation": dict(good["warm_attestation"], prefill_resumed=False)}, "RESUMED"),
                          ({"warm_attestation": dict(good["warm_attestation"], prefill_cached_positions=0)}, "implausible"),
                          ({"warm_attestation": dict(good["warm_attestation"], n_applications_modifying_resumed=0)},
                           "unsteered"),
                          ({"warm_bites": False}, "did not bite")):
            with pytest.raises(RuntimeError, match=why):
                ci.sharing_case_verdict(dict(good, **over))
        # the RPC reply shapes the attestation is read from
        assert ci._request_entries({"results": [{"requests": {"a": {"x": 1}}}, {"requests": {"b": {"y": 2}}}]}) == {
            "a": {"x": 1}, "b": {"y": 2}}
        assert ci._request_entries({"requests": {"a": {}}}) == {"a": {}}
        assert ci._request_entries(None) == {}

    def test_prompts_share_a_long_prefix(self):
        ci = _cache_iso()
        a, b = ci.SHARING_PROMPTS
        assert a.startswith(ci.SHARED_PASSAGE) and b.startswith(ci.SHARED_PASSAGE) and a != b
        assert len(ci.SHARED_PASSAGE.split()) > 200        # comfortably more than one KV block
