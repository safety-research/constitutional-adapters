"""The steer-is-live policy of
``sisteer.eval.attacks.bites_policy``.

What these pin, and why each one exists:

* The DEFAULT is today's reading. Nothing about any other lane moves until a
  job body exports ``SISTEER_BITES_POLICY=harmful_continuous``.
* The continuous verdict is the EXISTING checker's statistic on the
  INTERSECTION of the three legs' top-K supports, with nothing filled. The
  load-bearing test here is
  ``test_topk_membership_change_alone_manufactures_no_divergence``: the
  earlier design (a UNION of supports, missing entries filled) would have
  turned a pure top-K membership change into a large fabricated divergence and
  certified a dead steer. It must read ~0 and refuse.
* Thin common support is INCONCLUSIVE and refuses, with a diagnosis
  distinguishable from inert -- the two route to different places in the
  positive control's verdict matrix.
* Greedy TEXT counts never decide, in either direction. Greedy decoding is
  many-to-one, so 0/8 text differences with a strong continuous signal is an
  ADMIT and 8/8 with a signal below the bar is a REFUSE.
* The frozen probe set is the held-out harmful slice's first 8 distinct rows
  and is disjoint from the 259-goal confirmatory eval pool.

No network, no GPU: the transport is a stub, and every fixture is synthetic
logprob arithmetic.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.serving.fidelity import FidelityError  # noqa: E402
from sisteer.eval.attacks import bites_policy as bp  # noqa: E402


# --------------------------------------------------------------------------- #
# a stub transport                                                             #
# --------------------------------------------------------------------------- #

def _resp(text, logprobs=None):
    """One chat-completion dict. ``logprobs`` is ``{token_id: logprob}`` at the
    first generated position, rendered the way the pinned frontend renders
    ``return_tokens_as_token_ids`` + ``top_logprobs``."""
    out = {"choices": [{"message": {"content": text}}]}
    if logprobs is not None:
        items = sorted(logprobs.items(), key=lambda kv: -kv[1])
        top = [{"token": f"token_id:{t}", "logprob": lp} for t, lp in items]
        out["choices"][0]["logprobs"] = {"content": [
            {"token": top[0]["token"], "logprob": top[0]["logprob"],
             "top_logprobs": top}]}
    return out


class Chat:
    """A stub proxy transport keyed on (model, prompt). ``legs`` maps a vmodel
    to a callable ``(prompt, call_index) -> (text, logprobs|None)``; the call
    index is what lets the twin answer differently on its SECOND call (the
    noise leg)."""

    def __init__(self, legs):
        self.legs = legs
        self.calls = []
        self._n = {}

    def __call__(self, payload):
        model = payload["model"]
        prompt = payload["messages"][-1]["content"]
        self.calls.append(dict(payload))
        k = (model, prompt)
        i = self._n.get(k, 0)
        self._n[k] = i + 1
        text, lps = self.legs[model](prompt, i)
        if not payload.get("logprobs"):
            lps = None
        return _resp(text, lps)


def _flat(base=-1.0, n=6, start=1000):
    return {start + i: base - 0.1 * i for i in range(n)}


def _legs(*, steered_lp, scale0_lp, repeat_lp=None, steered_text="S",
          scale0_text="Z", repeat_text=None):
    """The three legs as a vmodel map. The twin serves ``scale0`` first and
    ``scale0_repeat`` second."""
    repeat_lp = scale0_lp if repeat_lp is None else repeat_lp
    repeat_text = scale0_text if repeat_text is None else repeat_text

    def steered(prompt, i):
        return steered_text, dict(steered_lp)

    def twin(prompt, i):
        return ((scale0_text, dict(scale0_lp)) if i == 0
                else (repeat_text, dict(repeat_lp)))

    return {"petri__alphasteer": steered, "petri__none__lens0": twin}


def _verdict(chat, **kw):
    kw.setdefault("policy", bp.POLICY_HARMFUL_CONTINUOUS)
    kw.setdefault("raise_on_refusal", False)
    return bp.liveness_verdict(chat, steered_vmodel="petri__alphasteer",
                               twin_vmodel="petri__none__lens0", **kw)


# --------------------------------------------------------------------------- #
# the frozen probe set                                                         #
# --------------------------------------------------------------------------- #

class TestProbeSet:
    def test_eight_distinct_probes_at_the_pinned_digest(self):
        assert len(bp.HARMFUL_PROBE_PROMPTS) == 8
        assert len(set(bp.HARMFUL_PROBE_PROMPTS)) == 8
        assert len(bp.HARMFUL_PROBE_LIST_INDICES) == 8
        assert (bp.probe_set_sha256(bp.HARMFUL_PROBE_PROMPTS)
                == bp.HARMFUL_PROBE_PROMPTS_SHA256), (
            "the frozen probe set changed. That redefines what 'live' was "
            "measured on for every rung and every leg -- update "
            "HARMFUL_PROBE_PROMPTS_SHA256 deliberately, never to make this "
            "pass")

    def test_neutral_five_are_the_same_five_the_two_call_sites_use(self):
        """The literal copy cannot drift from ``served_target`` /
        ``unified_sweep_run`` (it is a copy only to avoid an import cycle;
        served_target imports this module)."""
        from sisteer.eval.attacks import served_target as st
        assert bp.NEUTRAL_PROBE_PROMPTS == st.PROBE_PROMPTS
        assert bp.NEUTRAL_PROBE_MAX_TOKENS == st.PROBE_MAX_TOKENS
        spec = importlib.util.spec_from_file_location(
            "_bp_unified_sweep_run", REPO / "scripts" / "unified_sweep_run.py")
        text = (REPO / "scripts" / "unified_sweep_run.py").read_text()
        # module-level import of the script pulls heavy deps; read the literal
        start = text.index("_SERVED_PROBE_PROMPTS = (")
        end = text.index(")", start) + 1
        listed = eval(text[start + len("_SERVED_PROBE_PROMPTS = "):end])  # noqa: S307
        assert tuple(listed) == bp.NEUTRAL_PROBE_PROMPTS
        assert spec is not None

    def test_probes_are_heldout_harmful_rows_never_fit_rows(self):
        """Provenance, against the real pinned inputs when they are staged on
        this machine (they are cluster/state files, not repo files)."""
        root = Path("~/.local/state/alphasteer_f22").expanduser()
        lst, split = root / "harmful_alphasteer.json", root / "split_v1.json"
        if not (lst.is_file() and split.is_file()):
            pytest.skip(f"pinned harmful list / split_v1.json not staged at "
                        f"{root} (cluster inputs; see the module docstring's "
                        "provenance)")
        rows = json.loads(lst.read_text())
        sides = json.loads(split.read_text())["sides"]["harmful"]
        held, fit = set(sides["heldout_idx"]), set(sides["fit_idx"])
        held_texts = {rows[i] for i in held}
        fit_texts = {rows[i] for i in fit}
        for idx, prompt in zip(bp.HARMFUL_PROBE_LIST_INDICES,
                               bp.HARMFUL_PROBE_PROMPTS):
            assert rows[idx] == prompt, f"list index {idx} is not this probe"
            assert idx in held and idx not in fit
            assert prompt in held_texts
            assert prompt not in fit_texts, (
                f"probe {prompt!r} also appears at a FIT index -- the split is "
                "content-grouped, so this would be a split defect")

    def test_probes_are_disjoint_from_the_259_goal_eval_pool(self):
        """Never from the 259 eval goals. Exact (normalised) AND
        the campaign's own near-duplicate reading (word-3-gram Jaccard >= 0.8,
        the Phase 0 method)."""
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from sisteer.eval import jailbreak_data as jd
            pool = jd.load_harmbench("test") + jd.load_jbb("harmful")
        except Exception as e:  # noqa: BLE001 -- a cache miss is not a failure
            pytest.skip(f"pinned eval datasets not in the local HF cache "
                        f"({type(e).__name__}: {e})")
        assert len(pool) == 259, f"expected the 259-goal pool, got {len(pool)}"

        def norm(s):
            return " ".join(s.lower().split()).strip(" .!?\"'")

        def grams(s, n=3):
            w = norm(s).split()
            return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}

        by_text = {norm(g.text): g.goal_id for g in pool}
        for prompt in bp.HARMFUL_PROBE_PROMPTS:
            assert norm(prompt) not in by_text, (
                f"probe {prompt!r} IS eval goal {by_text[norm(prompt)]!r}")
            gp = grams(prompt)
            worst = max(
                ((len(gp & gg) / len(gp | gg)) if (gp | gg) else 0.0, g.goal_id)
                for g, gg in ((g, grams(g.text)) for g in pool))
            assert worst[0] < 0.8, (
                f"probe {prompt!r} is a near-duplicate (J={worst[0]:.3f}) of "
                f"eval goal {worst[1]!r}")


# --------------------------------------------------------------------------- #
# policy resolution and the default                                            #
# --------------------------------------------------------------------------- #

class TestPolicyResolution:
    def test_default_is_text_neutral_when_unset_or_empty(self):
        assert bp.resolve_policy(env={}) == bp.POLICY_TEXT_NEUTRAL
        assert bp.resolve_policy(env={bp.POLICY_ENV: ""}) == bp.POLICY_TEXT_NEUTRAL

    def test_env_selects_and_an_explicit_argument_wins(self):
        env = {bp.POLICY_ENV: bp.POLICY_HARMFUL_CONTINUOUS}
        assert bp.resolve_policy(env=env) == bp.POLICY_HARMFUL_CONTINUOUS
        assert (bp.resolve_policy(bp.POLICY_TEXT_NEUTRAL, env=env)
                == bp.POLICY_TEXT_NEUTRAL)

    def test_an_unknown_policy_refuses_rather_than_defaulting(self):
        """A typo must not silently run the OLD reading and file it under the
        requested name."""
        with pytest.raises(ValueError, match="not a known steer-is-live"):
            bp.resolve_policy(env={bp.POLICY_ENV: "harmful-continuous"})

    def test_default_policy_runs_the_neutral_text_check_only(self, monkeypatch):
        monkeypatch.delenv(bp.POLICY_ENV, raising=False)
        chat = Chat(_legs(steered_lp=_flat(), scale0_lp=_flat(),
                          steered_text="different", scale0_text="base"))
        block = bp.liveness_verdict(chat, steered_vmodel="petri__alphasteer",
                                    twin_vmodel="petri__none__lens0")
        assert block["policy"] == bp.POLICY_TEXT_NEUTRAL
        assert block["live"] is True
        # today's requests exactly: five prompts x two legs, no logprobs asked
        assert len(chat.calls) == 2 * len(bp.NEUTRAL_PROBE_PROMPTS)
        assert all("logprobs" not in c for c in chat.calls)
        assert {c["max_tokens"] for c in chat.calls} == {bp.NEUTRAL_PROBE_MAX_TOKENS}
        assert {c["temperature"] for c in chat.calls} == {0.0}
        assert {c["seed"] for c in chat.calls} == {0}


# --------------------------------------------------------------------------- #
# the continuous reading                                                       #
# --------------------------------------------------------------------------- #

class TestHarmfulContinuous:
    def test_a_clear_signal_over_a_measured_noise_floor_is_live(self):
        steered = {t: lp + 0.5 for t, lp in _flat().items()}
        noise = {t: lp + 1e-6 for t, lp in _flat().items()}
        chat = Chat(_legs(steered_lp=steered, scale0_lp=_flat(),
                          repeat_lp=noise))
        b = _verdict(chat)
        assert b["verdict"] == bp.VERDICT_LIVE and b["live"] is True
        assert b["n_conclusive"] == 8
        assert b["median"] == pytest.approx(0.5)
        assert b["noise_ceiling"] == pytest.approx(1e-6, abs=1e-9)
        assert b["bar"] == pytest.approx(1e-3)   # the abs floor binds
        assert b["ratio"] > 100

    def test_noise_sized_divergence_is_refused_not_certified(self):
        steered = {t: lp + 4e-4 for t, lp in _flat().items()}
        noise = {t: lp + 3e-4 for t, lp in _flat().items()}
        chat = Chat(_legs(steered_lp=steered, scale0_lp=_flat(),
                          repeat_lp=noise))
        b = _verdict(chat)
        assert b["verdict"] == bp.VERDICT_INERT and b["live"] is False
        assert "does not clear the bar" in b["error"]

    def test_the_three_legs_are_read_on_the_common_support_and_nothing_is_filled(self):
        """The steered leg returns two tokens the twin never did, and the twin
        two the steered never did. The item must carry ONLY the intersection,
        ordered by token id, with every logprob a value some leg actually
        returned."""
        base = _flat(n=6)                       # ids 1000..1005
        steered = dict(base)
        steered.pop(1004); steered.pop(1005)
        steered[2001] = -0.2; steered[2002] = -0.3
        twin = dict(base)
        twin.pop(1000); twin.pop(1001)
        twin[3001] = -0.25
        seen = {}

        def _cap(ev, diag, **kw):
            seen.update(ev)
            raise FidelityError("captured")

        chat = Chat(_legs(steered_lp=steered, scale0_lp=twin))
        import sisteer.capeval.serving.fidelity as fid
        real = fid.check_logprob_steering_bites
        try:
            fid.check_logprob_steering_bites = _cap
            _verdict(chat)
        finally:
            fid.check_logprob_steering_bites = real
        item = seen["items"][0]
        assert item["letter_ids"] == [1002, 1003], (
            "the item must be the INTERSECTION of the three supports, ordered "
            "by token id")
        for leg in ("steered", "scale0", "scale0_repeat"):
            assert len(item[leg]) == 2
            assert all(isinstance(v, float) and math.isfinite(v)
                       for v in item[leg])
        assert item["steered"] == [base[1002], base[1003]]

    def test_topk_membership_change_alone_manufactures_no_divergence(self):
        """THE test the earlier design would have failed.

        Three legs whose distributions are IDENTICAL on the common support but
        whose top-K membership differs at the boundary: a union-with-fill would
        have scored the fabricated entries as a huge divergence and certified a
        dead steer. On the intersection ds == 0 exactly, and the checker's
        bit-identical branch refuses."""
        shared = _flat(n=5)
        steered = dict(shared); steered[9001] = -9.10   # its own boundary token
        scale0 = dict(shared); scale0[9002] = -9.11     # a different one
        repeat = dict(shared); repeat[9003] = -9.12     # and a third
        chat = Chat(_legs(steered_lp=steered, scale0_lp=scale0,
                          repeat_lp=repeat))
        b = _verdict(chat)
        assert b["n_conclusive"] == 8, "the common support is 5 tokens wide"
        assert b["median"] == 0.0
        assert b["verdict"] == bp.VERDICT_INERT and b["live"] is False
        assert "BIT-IDENTICAL" in b["error"]
        assert b["supports_disjoint_probes"] == []

    def test_thin_common_support_is_inconclusive_not_inert(self):
        """Fewer than 4 conclusive probes refuses -- with a diagnosis that says
        *insufficient common support*, which routes to the positive control's
        escalation branch rather than being filed as inertness."""
        steered = {5000 + i: -1.0 - i for i in range(6)}
        twin = {7000 + i: -1.0 - i for i in range(6)}   # fully disjoint
        chat = Chat(_legs(steered_lp=steered, scale0_lp=twin))
        b = _verdict(chat)
        assert b["verdict"] == bp.VERDICT_INCONCLUSIVE and b["live"] is False
        assert b["n_conclusive"] == 0
        assert b["n_common_by_probe"] == [0] * 8
        assert len(b["supports_disjoint_probes"]) == 8
        assert "INSUFFICIENT COMMON SUPPORT" in b["error"]
        assert "inert" in b["error"]        # names what it is NOT
        assert "does not clear the bar" not in b["error"]

    def test_a_probe_with_one_common_token_contributes_no_item(self):
        """The checker itself refuses a leg shorter than 2, so a 1-token
        overlap is inconclusive, never a one-element comparison."""
        shared = {4242: -0.5}
        steered = {**shared, **{5000 + i: -2.0 - i for i in range(5)}}
        twin = {**shared, **{7000 + i: -2.0 - i for i in range(5)}}
        chat = Chat(_legs(steered_lp=steered, scale0_lp=twin))
        b = _verdict(chat)
        assert b["n_common_by_probe"] == [1] * 8
        assert b["n_conclusive"] == 0
        assert b["verdict"] == bp.VERDICT_INCONCLUSIVE

    def test_exactly_four_conclusive_probes_is_enough(self):
        """The boundary, since it decides between two different refusals."""
        strong = {t: lp + 0.5 for t, lp in _flat().items()}

        def _rank(prompt):
            # the neutral text pass also reaches these legs; only the harmful
            # probes are thinned
            return (bp.HARMFUL_PROBE_PROMPTS.index(prompt)
                    if prompt in bp.HARMFUL_PROBE_PROMPTS else 0)

        def steered(prompt, i):
            return "S", (dict(strong) if _rank(prompt) < 4 else {9000: -1.0})

        def twin(prompt, i):
            return "Z", (dict(_flat()) if _rank(prompt) < 4 else {8000: -1.0})

        b = _verdict(Chat({"petri__alphasteer": steered,
                           "petri__none__lens0": twin}))
        assert b["n_conclusive"] == 4
        assert b["verdict"] == bp.VERDICT_LIVE


class TestTextCountsNeverDecide:
    def test_zero_text_difference_with_a_strong_signal_is_admitted(self):
        """Greedy decode is many-to-one: an object that pushes a refusal from
        0.9 to 0.999 emits the identical text. 0/8 must not refuse."""
        steered = {t: lp + 0.5 for t, lp in _flat().items()}
        chat = Chat(_legs(steered_lp=steered, scale0_lp=_flat(),
                          steered_text="I can't help with that.",
                          scale0_text="I can't help with that."))
        b = _verdict(chat)
        assert b["verdict"] == bp.VERDICT_LIVE
        assert b["text_diff_harmful"] == "0/8"
        assert b["text_counts_decide"] is False

    def test_every_text_differing_with_a_weak_signal_is_refused(self):
        steered = {t: lp + 1e-5 for t, lp in _flat().items()}
        chat = Chat(_legs(steered_lp=steered, scale0_lp=_flat(),
                          steered_text="wholly different prose",
                          scale0_text="base prose"))
        b = _verdict(chat)
        assert b["text_diff_harmful"] == "8/8"
        assert b["verdict"] == bp.VERDICT_INERT

    def test_both_text_counts_are_recorded_as_measurements(self):
        steered = {t: lp + 0.5 for t, lp in _flat().items()}
        chat = Chat(_legs(steered_lp=steered, scale0_lp=_flat(),
                          steered_text="A", scale0_text="B"))
        b = _verdict(chat)
        assert b["text_diff_harmful"] == "8/8"
        assert b["text_diff_neutral"] == "5/5"
        assert b["probe_set_sha256"] == bp.HARMFUL_PROBE_PROMPTS_SHA256


class TestRefusalContract:
    def test_an_inert_object_refuses_under_both_policies(self):
        """The guard is not weakened by the new policy."""
        chat_t = Chat(_legs(steered_lp=_flat(), scale0_lp=_flat(),
                            steered_text="same", scale0_text="same"))
        with pytest.raises(FidelityError, match="NO difference"):
            bp.liveness_verdict(chat_t, steered_vmodel="petri__alphasteer",
                                twin_vmodel="petri__none__lens0",
                                policy=bp.POLICY_TEXT_NEUTRAL)
        chat_c = Chat(_legs(steered_lp=_flat(), scale0_lp=_flat(),
                            steered_text="same", scale0_text="same"))
        with pytest.raises(FidelityError):
            bp.liveness_verdict(chat_c, steered_vmodel="petri__alphasteer",
                                twin_vmodel="petri__none__lens0",
                                policy=bp.POLICY_HARMFUL_CONTINUOUS)

    def test_the_refusal_carries_the_evidence_that_caused_it(self):
        chat = Chat(_legs(steered_lp=_flat(), scale0_lp=_flat()))
        with pytest.raises(FidelityError) as ei:
            bp.liveness_verdict(chat, steered_vmodel="petri__alphasteer",
                                twin_vmodel="petri__none__lens0",
                                policy=bp.POLICY_HARMFUL_CONTINUOUS)
        assert ei.value.evidence["verdict"] == bp.VERDICT_INERT
        assert ei.value.evidence["n_conclusive"] == 8

    def test_raise_on_refusal_false_still_reports_not_live(self):
        """The ladder's mode: it RECORDS the verdict (liveness is an admission
        guard observed at every rung), it never serves on it."""
        chat = Chat(_legs(steered_lp=_flat(), scale0_lp=_flat()))
        b = _verdict(chat)
        assert b["live"] is False and b["required"] is True and "error" in b


class TestRequestShape:
    def test_the_probe_payload_is_greedy_seeded_and_token_id_keyed(self):
        p = bp.harmful_probe_payload("petri__alphasteer", "x")
        assert p["temperature"] == 0.0 and p["seed"] == 0
        assert p["logprobs"] is True and p["top_logprobs"] == 20
        assert p["return_tokens_as_token_ids"] is True
        # a FREELY generated first position: no teacher forcing, so the lens
        # route's response mask stays anchored where it is
        assert "continue_final_message" not in p
        assert "petri_response_prefill" not in p

    def test_three_legs_per_harmful_probe_and_the_twin_is_asked_twice(self):
        steered = {t: lp + 0.5 for t, lp in _flat().items()}
        chat = Chat(_legs(steered_lp=steered, scale0_lp=_flat()))
        _verdict(chat)
        lp_calls = [c for c in chat.calls if c.get("logprobs")]
        assert len(lp_calls) == 24
        by_model = {}
        for c in lp_calls:
            by_model[c["model"]] = by_model.get(c["model"], 0) + 1
        assert by_model == {"petri__alphasteer": 8, "petri__none__lens0": 16}

    def test_a_response_without_logprobs_refuses_rather_than_scoring_text(self):
        def steered(prompt, i):
            return "S", None       # server ignored the flags

        def twin(prompt, i):
            return "Z", dict(_flat())

        chat = Chat({"petri__alphasteer": steered,
                     "petri__none__lens0": twin})
        chat_call = chat.__call__

        def no_lp(payload):
            r = chat_call(payload)
            if payload["model"] == "petri__alphasteer":
                r["choices"][0].pop("logprobs", None)
            return r

        with pytest.raises(RuntimeError, match="did not honor logprobs"):
            _verdict(no_lp)

    def test_a_non_token_id_surface_refuses(self):
        """``return_tokens_as_token_ids`` not honored: matching by surface
        string could silently compare different tokens across legs."""
        with pytest.raises(RuntimeError, match="token_id:<int> form"):
            bp.first_position_logprobs(
                {"choices": [{"logprobs": {"content": [
                    {"token": " A", "logprob": -0.5, "top_logprobs": [
                        {"token": " A", "logprob": -0.5}]}]}}]},
                what="probe")

    def test_a_non_finite_logprob_refuses(self):
        with pytest.raises(RuntimeError, match="non-finite"):
            bp.first_position_logprobs(
                {"choices": [{"logprobs": {"content": [
                    {"token": "token_id:5", "logprob": -0.5, "top_logprobs": [
                        {"token": "token_id:5", "logprob": -0.5},
                        {"token": "token_id:6", "logprob": float("-inf")}]}]}}]},
                what="probe")

    def test_token_id_reader_matches_the_controls_drivers_twin(self):
        """Anti-drift: this module reimplements the served drivers' token-id
        reader. When this policy was written that was ``served_capability_run
        ._token_id_of`` (a library must not import a script); commit 34675760
        folded it into the serving tree as
        ``proxy_engine.first_position_logprobs_by_id`` behind
        ``TOKEN_ID_PREFIX`` — and a policy module OUTSIDE the serving tree
        must not import it either (``serving_impl_commit`` would then depend
        on this module). So the two are pinned to behave identically on
        the same payloads: the id read for every well-formed entry, a refusal
        (each side's own typed error) for every malformed one."""
        from sisteer.capeval.serving import proxy_engine as pe

        assert pe.TOKEN_ID_PREFIX == "token_id:"

        def theirs(token):
            resp = {"choices": [{"logprobs": {"content": [
                {"token": token, "logprob": -0.5, "top_logprobs": []}]}}]}
            (tid,) = pe.first_position_logprobs_by_id(resp, what="twin")
            return tid

        for good in ("token_id:0", "token_id:12345"):
            assert bp._token_id_of(good) == theirs(good) == int(good[len("token_id:"):])
        for bad in (" A", "", None, "tokenid:3", 5):
            with pytest.raises(RuntimeError):
                bp._token_id_of(bad)
            with pytest.raises(pe.ProxyEngineError):
                theirs(bad)


class TestNotWiredYet:
    def test_the_policy_module_is_outside_the_serving_stack(self):
        """It must not move ``serving_impl_commit``."""
        import sisteer.eval.attacks.bites_policy as m
        p = Path(m.__file__).resolve()
        assert "capeval/serving" not in p.as_posix()
        assert p.parent.as_posix().endswith("sisteer/eval/attacks")

    def test_the_four_production_call_sites_are_untouched_in_this_change(self):
        """Rewiring them is plan Phase G, deliberately NOT in this change: it
        would put a new decision function in front of four live drivers in the
        same commit that introduces it. Until then the callers keep their own
        code and the default policy keeps them byte-identical."""
        for rel in ("src/sisteer/eval/attacks/served_target.py",
                    "src/sisteer/eval/attacks/bon_served.py",
                    "scripts/unified_sweep_run.py",
                    "scripts/served_capability_run.py"):
            assert "bites_policy" not in (REPO / rel).read_text(), (
                f"{rel} now imports bites_policy -- if the Phase G rewiring "
                "has landed, this test should be replaced by the wiring "
                "tests, not deleted")
