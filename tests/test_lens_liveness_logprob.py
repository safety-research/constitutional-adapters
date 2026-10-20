"""The LOGPROB liveness gate of the served driver's positive control.

Array 495677 refused the gemma (lambda 0.057) and nemotron (lambda 0.049) AlphaSteer legs as "inert": the steering-bites
control demands SOME greedy-text divergence over five 48-token bites against the scale-0 twin, and a small correct edit
flips no argmax there (the qwen point at lambda 0.548 diverged on 2/5 and passed). The campaign's ladder had already
retired text-diff liveness for LOGPROB DISTANCE with a measured repeat-noise floor; the ruling applies that instrument in
the driver for the two Class B deployment-shape kinds (``alphasteer`` / ``static_r``) under ``--lens-liveness logprob``:

  (1) the floor is MEASURED on the same probe prompts in the same run (the scale-0 twin against itself), and the route
      refuses when the steered divergence sits inside it — the bar is max(abs_floor, min_ratio x the measured ceiling), the
      record says which bound applied; never a constant alone;
  (2) the text bites statistic still runs and is REPORTED beside the ruling one (``text_divergence``);
  (3) the record states that logprob distance is NOT a dose axis, verbatim.

Everything synthetic: RouteEntry fixtures, a scripted chat that answers text probes with canned text and one-token logprob
probes with canned top-K distributions per leg, a stub provenance. The REAL ``_served_evidence`` runs end to end (salt
isolation, the engine stamp, ``assert_steering_bites``, the new gate); the wiring into ``cmd_generate`` and the launcher is
pinned as text where a live server would be needed to reach it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import served_capability_run as srun  # noqa: E402
import unified_sweep_run as usr  # noqa: E402
from sisteer.capeval.serving.proxy import RouteEntry  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402

LENS = "proxy_vllm+lens+interleaved"
PASS = "proxy_vllm+passthrough+interleaved"
TEMPLATE_HASH = "0a2c8073c878ab1d"
PROMPTS = usr._SERVED_PROBE_PROMPTS


def _route(cell, ident, *, vmodel=None, cell_class="class_b", ih="cc" * 8, **kw):
    return RouteEntry(virtual_model=vmodel or f"petri__{cell}", cell_id=cell, cell_class=cell_class,
                      serving_identity=ident, intervention_hash=ih, **kw)


def _routes(cell_id: str):
    base = _route("none", PASS, vmodel="petri__none", cell_class="base", ih="aa" * 8)
    twin = _route(tc.TWIN_CELL_ID, LENS, vmodel=tc.twin_vmodel(), cell_class="control", ih="aa" * 8, control_of="none")
    cell = _route(cell_id, LENS, ih="bb" * 8)
    return cell, twin, [base, twin, cell]


def _cell(cell_id: str, arm: str):
    return SimpleNamespace(id=cell_id, arm=arm, object_path=Path("/synthetic/object.pt"))


PROV = SimpleNamespace(model_id="synthetic/model", revision="d" * 40, trained_template_hash=TEMPLATE_HASH,
                       object_sha="ab" * 8, status="verified", source="fit_provenance.json", positions_at_fit="prefill")
TOK_EV = {"served_template_hash": TEMPLATE_HASH, "serve_header_ids": [1, 2, 3], "serve_header_len": 3}


def _lp_response(by_id: dict) -> dict:
    top = max(by_id, key=by_id.__getitem__)
    entries = [{"token": f"token_id:{tid}", "logprob": lp} for tid, lp in sorted(by_id.items())]
    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": " x"},
                         "logprobs": {"content": [{"token": f"token_id:{top}", "logprob": by_id[top],
                                                   "top_logprobs": entries}]}}]}


class Chat:
    """A scripted proxy client. TEXT probes (max_tokens 48) answer canned text per vmodel. LOGPROB probes (max_tokens 1
    + logprobs) answer a 20-id top-K per leg: the twin's base distribution; the steered leg = the twin's shifted by
    ``delta`` (sign alternating per id); the twin's SECOND draw per prompt = the base jittered by ``jitter`` (the engine's
    repeat noise). ``steered_ids`` moves the steered leg onto other token ids for ``disjoint_probes`` prompts (no common
    support); ``no_logprobs`` answers logprob probes with no logprobs block (a server not honoring the request)."""

    def __init__(self, *, steered_text="steered voice says alpha", twin_text="plain voice says beta",
                 delta=0.5, jitter=0.0, disjoint_probes=(), no_logprobs=False):
        self.steered_text, self.twin_text = steered_text, twin_text
        self.delta, self.jitter = delta, jitter
        self.disjoint_probes, self.no_logprobs = set(disjoint_probes), no_logprobs
        self.calls: list = []
        self.twin_draws: dict = {}

    @staticmethod
    def _base(prompt_index: int) -> dict:
        return {100 + i: -0.3 - 0.2 * i - 0.01 * prompt_index for i in range(20)}

    def __call__(self, payload):
        self.calls.append(json.loads(json.dumps(payload)))
        vm = payload["model"]
        steered = vm != tc.twin_vmodel()
        if payload.get("logprobs"):
            if self.no_logprobs:
                return {"choices": [{"finish_reason": "length", "message": {"content": " x"}}]}
            i = PROMPTS.index(payload["messages"][-1]["content"])
            base = self._base(i)
            if steered:
                if i in self.disjoint_probes:
                    return _lp_response({tid + 500: lp for tid, lp in base.items()})
                return _lp_response({tid: lp + self.delta * (1 if k % 2 else -1) for k, (tid, lp) in enumerate(base.items())})
            n = self.twin_draws.get(i, 0)
            self.twin_draws[i] = n + 1
            if n % 2 == 1:      # the SECOND twin draw of this prompt: the repeat leg, jittered
                return _lp_response({tid: lp + self.jitter * (1 if k % 2 else -1) for k, (tid, lp) in enumerate(base.items())})
            return _lp_response(base)
        text = self.steered_text if steered else self.twin_text
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


def _evidence(chat, *, cell_id="alphasteer", arm="alphasteer", lens_liveness=tc.LENS_LIVENESS_LOGPROB,
              logprob_chat=None, **kw):
    route, twin, routes = _routes(cell_id)
    return srun._served_evidence(cell=_cell(cell_id, arm), route=route, routes=routes, chat=chat,
                                 model_id="synthetic/model", model_revision="d" * 40, prov=PROV, vllm_log=None,
                                 tok_ev=TOK_EV, lens_liveness=lens_liveness, logprob_chat=logprob_chat, **kw)


# --------------------------------------------------------------------------------------------------------------------- #
# the gate on the two kinds                                                                                               #
# --------------------------------------------------------------------------------------------------------------------- #


def test_a_live_route_passes_by_logprob_while_text_is_inert_and_both_statistics_are_on_the_record():
    """0/5 text bites (steered and twin answer the SAME text) but a first-token divergence of 0.5 nats
    against a bit-stable engine (repeat noise 0) -> LIVE by logprob; the text result is an observation beside it; the
    floor, the divergence, the binding bound and the not-a-dose-axis sentence are all recorded; no top-level error."""
    chat = Chat(steered_text="same words", twin_text="same words", delta=0.5, jitter=0.0)
    ev, err = _evidence(chat)
    b = ev["steering_bites"]
    assert err is None and b["verified"] is True and b["applicable"] is True
    assert b[tc.OOS_EVIDENCE_BITES_LIVENESS_GATE] == "logprob" and b["object_kind"] == "alphasteer"
    assert "error" not in b
    # (2) the text statistic, computed and REPORTED, not the ruler
    tx = b[tc.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE]
    assert tx["verified"] is False and tx["n_differing"] == 0 and tx["n_probes"] == len(PROMPTS)
    assert "NO difference" in tx["error"] and "REPORTED" in tx["role"]
    assert b["n_differing"] == 0 and b["n_probes"] == len(PROMPTS)        # the analysis's count fields still read
    # (1) the ruling statistic with its MEASURED floor
    lp = b[tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]
    assert lp["verified"] is True and lp["kind"] == "alphasteer" and lp["twin"] == tc.TWIN_CELL_ID
    assert lp["n_probes"] == len(PROMPTS) == lp["n_conclusive"] and lp["top_logprobs"] == 20 and lp["max_tokens"] == 1
    assert lp["noise_floor"]["measured"] == 0.0 == lp["diag"]["noise_ceiling"]
    assert lp["noise_floor"]["bar"] == pytest.approx(1e-3) and lp["noise_floor"]["binding"] == "abs_floor"
    assert "scale-0 twin against ITSELF" in lp["noise_floor"]["how"]
    assert lp["divergence"]["steered_vs_scale0_median"] == pytest.approx(0.5) and lp["divergence"]["clears_bar"] is True
    assert all(r["conclusive"] and r["n_common"] == 20 and not r["supports_disjoint"] for r in lp["per_probe"])
    assert lp["legs"] == ["steered", "scale0", "scale0_repeat"]
    # (3) not a dose axis, verbatim
    assert lp["not_a_dose_axis"] == tc.LOGPROB_LIVENESS_NOT_A_DOSE_AXIS
    assert "NOT linear in dose" in lp["not_a_dose_axis"] and "never a dose axis" in lp["not_a_dose_axis"]


def test_the_measured_floor_binds_when_the_engine_is_noisy_and_the_bar_is_never_a_constant_alone():
    """Repeat jitter 5e-3 -> the measured ceiling 5e-3, the bar 2 x 5e-3 = 1e-2 > abs_floor: the MEASURED bound applies
    and the record says so. A 0.5-nat edit still clears it."""
    chat = Chat(delta=0.5, jitter=5e-3)
    ev, err = _evidence(chat)
    lp = ev["steering_bites"][tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]
    assert err is None and lp["verified"] is True
    assert lp["noise_floor"]["measured"] == pytest.approx(5e-3) and lp["noise_floor"]["bar"] == pytest.approx(1e-2)
    assert lp["noise_floor"]["binding"] == "measured" and lp["noise_floor"]["min_ratio"] == 2.0 and lp["noise_floor"]["abs_floor"] == 1e-3


def test_a_dead_route_refuses_bit_identical_and_inside_the_floor_alike():
    """Twin-vs-twin: the steered leg IS the twin (delta 0) -> bit-identical, refused. A divergence of 1e-4 under a repeat
    noise of 1e-3 sits inside the floor (bar 2e-3) -> refused. Both are recorded with the error at the top level and the
    caller's SystemExit names the LOGPROB statistic. Text bites are inert either way and reported."""
    ev, err = _evidence(Chat(steered_text="w", twin_text="w", delta=0.0, jitter=0.0))
    b = ev["steering_bites"]
    assert err is not None and "BIT-IDENTICAL" in err and b["verified"] is False and b["error"] == err
    assert b[tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]["verified"] is False and b[tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]["error"] == err
    assert b[tc.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE]["n_differing"] == 0
    ev, err = _evidence(Chat(steered_text="w", twin_text="w", delta=1e-4, jitter=1e-3))
    b = ev["steering_bites"]
    assert err is not None and "does not clear the bar" in err and b["verified"] is False
    lp = b[tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]
    assert lp["noise_floor"]["measured"] == pytest.approx(1e-3) and lp["noise_floor"]["bar"] == pytest.approx(2e-3)
    assert lp["divergence"]["steered_vs_scale0_median"] == pytest.approx(1e-4) and lp["divergence"]["clears_bar"] is False
    assert lp["not_a_dose_axis"] == tc.LOGPROB_LIVENESS_NOT_A_DOSE_AXIS       # on a failed record too


def test_text_live_and_logprob_live_records_both_and_text_live_but_logprob_dead_still_refuses():
    """The ruler is the logprob statistic for these kinds in BOTH directions: text divergence alone (5/5 differing) does
    not certify a route whose first-token logprobs are bit-identical to the twin's."""
    ev, err = _evidence(Chat(delta=0.5))
    b = ev["steering_bites"]
    assert err is None and b["verified"] is True
    assert b[tc.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE]["verified"] is True and b["n_differing"] == len(PROMPTS)
    ev, err = _evidence(Chat(delta=0.0))                # text differs (different canned words), logprobs identical
    b = ev["steering_bites"]
    assert err is not None and "BIT-IDENTICAL" in err and b["verified"] is False
    assert b[tc.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE]["verified"] is True and b[tc.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE]["n_differing"] == len(PROMPTS)


def test_static_r_takes_the_gate_too_through_the_arditi_arm():
    """The Arditi cell's arm is `arditi`; the blob's kind is `static_r` (registry_adapter.ARM_OBJECT_KINDS) — in the
    ruling's set."""
    ev, err = _evidence(Chat(steered_text="w", twin_text="w", delta=0.4), cell_id="arditi", arm="arditi")
    b = ev["steering_bites"]
    assert err is None and b["verified"] is True and b["object_kind"] == "static_r" and b[tc.OOS_EVIDENCE_BITES_LIVENESS_GATE] == "logprob"
    assert srun._object_kind_of(_cell("arditi", "arditi")) == "static_r"
    assert srun._object_kind_of(_cell("alphasteer", "alphasteer")) == "alphasteer"
    assert srun._object_kind_of(_cell("si_safety__psr_kl_all", "psr_kl_all")) == "psr"
    assert srun._object_kind_of(SimpleNamespace(id="none", arm=None)) is None


# --------------------------------------------------------------------------------------------------------------------- #
# the boundary: every other kind, and the default mode, are byte-identical                                                #
# --------------------------------------------------------------------------------------------------------------------- #


def test_other_kinds_keep_the_text_gate_under_the_logprob_mode_and_send_no_logprob_probe():
    """A psr cell under --lens-liveness logprob: the text statistic rules (inert text refuses), no logprob block, not one
    logprob request on the wire."""
    chat = Chat(steered_text="w", twin_text="w", delta=0.5)
    ev, err = _evidence(chat, cell_id="si_safety__psr_kl_all", arm="psr_kl_all")
    b = ev["steering_bites"]
    assert err is not None and "NO difference" in err and b["verified"] is False and b["error"] == err
    assert b[tc.OOS_EVIDENCE_BITES_LIVENESS_GATE] == "text" and b["object_kind"] == "psr"
    assert tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS not in b and tc.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE not in b
    assert not any(c.get("logprobs") for c in chat.calls)
    # and a live psr cell under the logprob mode passes by text exactly as before
    ev, err = _evidence(Chat(delta=0.5), cell_id="si_safety__psr_kl_all", arm="psr_kl_all")
    assert err is None and ev["steering_bites"]["verified"] is True and ev["steering_bites"]["n_differing"] == len(PROMPTS)


def test_the_default_mode_is_text_for_the_two_kinds_as_well_and_the_block_is_unchanged_but_for_the_gate_name():
    """Without --lens-liveness logprob an alphasteer cell is ruled by text — the historical behaviour, so an inert text
    bite refuses even though a logprob read would have passed; nothing logprob-shaped is sent. The block carries the
    same keys as before plus the additive facts (the gate name, the kind, the standup's thinking
    mode, the probe channel policy and the render-arm reason)."""
    chat = Chat(steered_text="w", twin_text="w", delta=0.5)
    ev, err = _evidence(chat, lens_liveness=tc.LENS_LIVENESS_TEXT)
    b = ev["steering_bites"]
    assert err is not None and "NO difference" in err and b[tc.OOS_EVIDENCE_BITES_LIVENESS_GATE] == "text"
    assert not any(c.get("logprobs") for c in chat.calls)
    ev, err = _evidence(Chat(delta=0.5), lens_liveness=tc.LENS_LIVENESS_TEXT)
    b = ev["steering_bites"]
    assert err is None and b["verified"] is True
    assert set(b) == {"applicable", "verified", "twin", "arm", "n_probes", "n_differing", "diag",
                      tc.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM, "probe_extra_body", "stamps_by_leg", "stamp_errors_by_leg",
                      "channels_by_leg", tc.OOS_EVIDENCE_BITES_LIVENESS_GATE, "object_kind",
                      # the probe renders in the standup's attested regime — three additive facts
                      'standup_thinking_mode', 'probe_channel_policy', 'probe_render_arm_reason'}


# --------------------------------------------------------------------------------------------------------------------- #
# fail closed: an unreadable or inconclusive instrument refuses, never a text-only pass                                   #
# --------------------------------------------------------------------------------------------------------------------- #


def test_a_server_not_honoring_logprobs_refuses_by_name():
    chat = Chat(delta=0.5, no_logprobs=True)             # text bites 5/5 differing — would have passed by text
    ev, err = _evidence(chat)
    b = ev["steering_bites"]
    assert err is not None and "could not be read" in err and "no logprobs" in err and b["verified"] is False
    assert b[tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]["verified"] is False and b[tc.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE]["verified"] is True


def test_too_few_conclusive_probes_is_unevaluable_and_refuses_while_a_minority_of_disjoint_probes_is_recorded():
    """3 of 5 probes with NO common top-K support (a large edit moved the whole top-20) -> 2 conclusive < 3 -> refused
    by name with every probe's support listed. 2 of 5 disjoint -> 3 conclusive: the statistic runs over those three, the
    two disjoint probes are recorded as inconclusive with `supports_disjoint: true` (the shape of a large edit)."""
    ev, err = _evidence(Chat(delta=0.5, disjoint_probes=(0, 1, 2)))
    assert err is not None and "only 2 of 5 probe(s) are conclusive" in err and "UNEVALUABLE" in err
    lp = ev["steering_bites"][tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]
    assert lp["n_conclusive"] == 2 and lp["min_conclusive_probes"] == 3 and lp["diag"] == {} and lp["noise_floor"]["measured"] is None
    ev, err = _evidence(Chat(delta=0.5, disjoint_probes=(0, 1)))
    lp = ev["steering_bites"][tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]
    assert err is None and lp["verified"] is True and lp["n_conclusive"] == 3
    assert [r["supports_disjoint"] for r in lp["per_probe"]] == [True, True, False, False, False]
    assert [r["conclusive"] for r in lp["per_probe"]] == [False, False, True, True, True]
    assert lp["diag"]["n_items"] == 3


# --------------------------------------------------------------------------------------------------------------------- #
# the wire: what the logprob probes send, and through which labelled client                                              #
# --------------------------------------------------------------------------------------------------------------------- #


def test_the_logprob_probes_are_one_greedy_token_top_20_by_token_id_three_legs_per_prompt_in_order():
    chat = Chat(delta=0.5)
    _evidence(chat)
    lp_calls = [c for c in chat.calls if c.get("logprobs")]
    assert len(lp_calls) == 3 * len(PROMPTS)
    for c in lp_calls:
        assert c["max_tokens"] == tc.LOGPROB_BITES_MAX_TOKENS == 1 and c["temperature"] == 0.0 and c["seed"] == 0
        assert c["top_logprobs"] == 20 and c["return_tokens_as_token_ids"] is True
        assert set(c) == {"model", "messages", "temperature", "max_tokens", "seed", "logprobs", "top_logprobs", "return_tokens_as_token_ids"}
        assert "cache_salt" not in c
    # per prompt: steered, scale0, scale0_repeat — sequential, in that order, on the bites prompts
    route_vm, twin_vm = "petri__alphasteer", tc.twin_vmodel()
    for i, p in enumerate(PROMPTS):
        trio = lp_calls[3 * i: 3 * i + 3]
        assert [c["model"] for c in trio] == [route_vm, twin_vm, twin_vm]
        assert all(c["messages"] == [{"role": "user", "content": p}] for c in trio)
    # the text bites still fired on both legs before them
    text_calls = [c for c in chat.calls if not c.get("logprobs")]
    assert len(text_calls) == 2 * len(PROMPTS) and all(c["max_tokens"] == 48 for c in text_calls)


def test_the_logprob_probes_ride_the_bites_render_and_go_through_the_labelled_client_when_given(monkeypatch):
    """Under a thinking arm the bites probe adds the arm's kwargs on both legs; the logprob probes carry the SAME body
    (the statistic is read under the same render as the text bite). ``logprob_chat`` — the probe/logprob_bites-labelled
    client — carries them when given; the bites chat otherwise."""
    monkeypatch.setattr(srun, "_probe_stamp_problem", lambda *a, **k: None)   # the leg stamps are the world tests' business
    text_chat, lp_chat = Chat(delta=0.5), Chat(delta=0.5)
    ev, err = _evidence(text_chat, logprob_chat=lp_chat, probe_render_arm=tc.THINKING_OFF, probe_header_located=False)
    assert err is None
    off = tc.render_fragment(lens=True, arm=tc.THINKING_OFF, header_located=False)
    assert off == {"chat_template_kwargs": {"enable_thinking": False}}
    assert not any(c.get("logprobs") for c in text_chat.calls) and all(c.get("logprobs") for c in lp_chat.calls)
    assert len(lp_chat.calls) == 3 * len(PROMPTS)
    assert all(c["chat_template_kwargs"] == off["chat_template_kwargs"] for c in lp_chat.calls)
    assert all(c["chat_template_kwargs"] == off["chat_template_kwargs"] for c in text_chat.calls)
    lp = ev["steering_bites"][tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]
    assert lp["probe_extra_body"] == off == ev["steering_bites"]["probe_extra_body"]


# --------------------------------------------------------------------------------------------------------------------- #
# the wiring the tests above cannot reach live: cmd_generate, the flag, the launcher, the recorder's kind               #
# --------------------------------------------------------------------------------------------------------------------- #


def test_cmd_generate_passes_the_mode_and_the_labelled_client_and_the_abort_names_the_statistic():
    t = (REPO / "scripts/served_capability_run.py").read_text()
    call = t.index("evidence, bites_error = _served_evidence(")
    assert 'lens_liveness=getattr(args, "lens_liveness", tc.LENS_LIVENESS_TEXT)' in t[call: call + 2500]
    assert "logprob_chat=_probe_chat(tc.PROBE_KIND_LOGPROB_BITES)" in t[call: call + 2500]
    # the flag: default text, the two modes
    assert 'g.add_argument(tc.FLAG_LENS_LIVENESS, dest="lens_liveness", choices=list(tc.LENS_LIVENESS_MODES),' in t
    assert "default=tc.LENS_LIVENESS_TEXT," in t
    p = srun.build_parser()
    a = p.parse_args(["generate", "--proxy-url", "x", "--routing-table", "t", "--cell", "c", "--config", "y",
                      "--defenses-dir", "d", "--out-root", "o", "--controls", "math500", "--thinking", "on"])
    assert a.lens_liveness == "text"
    a = p.parse_args(["generate", "--proxy-url", "x", "--routing-table", "t", "--cell", "c", "--config", "y",
                      "--defenses-dir", "d", "--out-root", "o", "--controls", "math500", "--thinking", "on",
                      "--lens-liveness", "logprob"])
    assert a.lens_liveness == "logprob"
    with pytest.raises(SystemExit):
        p.parse_args(["generate", "--proxy-url", "x", "--routing-table", "t", "--cell", "c", "--config", "y",
                      "--defenses-dir", "d", "--out-root", "o", "--controls", "math500", "--lens-liveness", "rho"])
    # the abort: the logprob gate's refusal names the statistic and the text observation; the text gate's is unchanged
    abort = t.index("if bites_error is not None:\n        if bites.get(tc.OOS_EVIDENCE_BITES_LIVENESS_GATE) == tc.LENS_LIVENESS_LOGPROB:")
    assert "the LOGPROB liveness statistic" in t[abort: abort + 1200] and "MEASURED repeat-noise floor" in t[abort: abort + 1200]
    assert "an inert serve; nothing was generated." in t[abort: abort + 1600]
    # the pin threads into the provenance mirror (class A's fourth door)
    assert "prov = _mirror_provenance(cell, expected_object_sha256=(args.expect_object_sha256 or None))" in t


def test_the_launcher_passes_the_flag_on_external_object_rows_only_and_the_recorder_knows_the_kind():
    sh = (REPO / "scripts/sdfbase/math500/m5t_common.sh").read_text()
    assert '[ -z "${M5T_OBJECT_SOURCE:-}" ] || DRV+=(--lens-liveness logprob)' in sh
    body = sh[sh.index("m5t_driver_argv(){"):]
    body = body[: body.index("\n}\n")]
    assert body.count("--lens-liveness") == 1 and 'if [ "$FITTED" = 1 ]; then' in body
    assert body.index('if [ "$FITTED" = 1 ]; then') < body.index("--lens-liveness")       # inside the FITTED block
    sys.path.insert(0, str(REPO / "scripts/sdfbase"))
    import record_resolved_sampler as rrs
    assert rrs.PROBE_KIND_TABLE[tc.PROBE_KIND_LOGPROB_BITES]["caps"] == (tc.LOGPROB_BITES_MAX_TOKENS,)
    assert tc.probe_task_label(tc.PROBE_KIND_LOGPROB_BITES) == "probe/logprob_bites"
    rec = tc.request_evidence_record(json.dumps({"model": "m", "messages": [], "temperature": 0.0, "max_tokens": 1,
                                                 "logprobs": True, "top_logprobs": 20}).encode(),
                                     headers={tc.HEADER_TASK: tc.probe_task_label(tc.PROBE_KIND_LOGPROB_BITES),
                                              tc.HEADER_SAMPLE: "m/0", tc.HEADER_REQUEST_ID: "0" * 32})
    assert rec["kind"] == "probe" and rec["task"] == "probe/logprob_bites" and rec["max_tokens"] == 1


def test_the_no_switch_inventory_counts_the_logprob_probes_only_when_told():
    base = tc.no_switch_probe_inventory("petri__arditi", fitted=True, n_probes=5)
    assert tc.PROBE_KIND_LOGPROB_BITES not in {k[2] for k in base}
    inv = tc.no_switch_probe_inventory("petri__arditi", fitted=True, n_probes=5, n_logprob_probes=5)
    na = tc.THINKING_NOT_AVAILABLE
    assert inv == {**base, ("petri__arditi", na, "logprob_bites"): 5, (tc.twin_vmodel(), na, "logprob_bites"): 10}
    with pytest.raises(ValueError, match="logprob_liveness.n_probes"):
        tc.no_switch_probe_inventory("petri__arditi", fitted=True, n_probes=5, n_logprob_probes=0)


def test_the_constants_are_the_rulings_and_the_ladders():
    # the three CONSTANT residual kinds joined 2026-09-16: the Arditi arm in the
    # paper's own configuration is a `vector_all` object at MODE_ALL, which both of
    # logprob_liveness_cells' branches missed, so a correct constant add at a small
    # norm read inert on the greedy statistic. `psr` is deliberately absent: the
    # stage-1 PSR certificates were admitted under the greedy gate.
    assert tc.LOGPROB_LIVENESS_KINDS == ("alphasteer", "static_r", "vector_all", "vector", "diffmean")
    assert srun.LOGPROB_LIVENESS_TOP_LOGPROBS == 20 and srun.LOGPROB_LIVENESS_MIN_COMMON_TOKENS == 2
    assert srun.LOGPROB_LIVENESS_MIN_CONCLUSIVE_PROBES == 3 and len(PROMPTS) == 5
    assert srun.LOGPROB_BITES_MIN_RATIO == 2.0 and srun.LOGPROB_BITES_ABS_FLOOR == 1e-3     # the instrument's own
    assert re.search(r"NOT linear in dose", tc.LOGPROB_LIVENESS_NOT_A_DOSE_AXIS)
    assert tc.LENS_LIVENESS_MODES == ("text", "logprob") and tc.FLAG_LENS_LIVENESS == "--lens-liveness"
