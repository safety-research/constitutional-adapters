"""Two GSM8K defects on a native multi-LoRA standup, and the fixes that closed them.

The first defect (every FITTED GSM8K child). ``served_capability_run`` derived the steering-bites probe's render
from the rule "an undeclared all-position cell renders OFF on both legs" on every ``lens_cell and not
header_located`` cell, which on a NATIVE standup is every fitted cell (``header_located`` is the engine's fact
there). On the thinking-ON-pinned standup the OFF render is what the proxy refuses BY DESIGN, so
``bon_served.assert_steering_bites`` → ``unified_sweep_run.probe`` → ``_greedy`` → the client died: ``ProxyError:
proxy … returned 400: {… "code": "thinking_regime_mismatch", "via": "ledger_tokenize"}``. That rule's rationale
(an undeclared ON request to the LENS twin ``none__lens0`` refuses) is void on a native standup, whose twin
``none__lora0`` is a non-lens route. Fix: the probe renders in the standup's ATTESTED regime —
``endpoint.json``'s ``thinking_mode`` off the bound record, never argv — when the standup is native and attests
``on``; ``off`` / absent, and every lens standup, keep the OFF render. Under ON both legs carry the certified ON
arm's kwargs, so the bite compares an ON render with an ON render (both channels re-joined).

The second defect (the GSM8K ``none`` child). The OFF-arm refusal was accepted, then ``render_certified`` refused:
``measured header ids {'on': [...], 'off': []} != the family's``. Root cause: transformers 5.x (5.14.1 in the
container, 5.15 here) defaults ``apply_chat_template(tokenize=True)`` to ``return_dict=True`` and returns a
``BatchEncoding``; ``_header_ids_under_arm`` took ``tuple()`` of each render, i.e. the mapping's KEYS — identical
for both renders — so the header difference read as EMPTY and the local OFF reference was ``[]``. Fix: the
render's ids are read through ``_render_token_ids`` (a list, or a mapping's ``input_ids``; anything else refuses
by name); an EMPTY OFF reference under that rule refuses by name in the header measurement AND in the
certification, which never compares ``[]`` against the fact.

Everything is synthetic (placeholder vmodels, the proxy tests' canonical cell set, header ids off the family fact).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import usage_ledger as ul  # noqa: E402
from sisteer.capeval.serving.fidelity import FidelityError  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.eval.attacks import bon_served as bs  # noqa: E402

import served_capability_run as srun  # noqa: E402
import unified_sweep_run as usr  # noqa: E402
from test_served_capability_render_on_pinned import (  # noqa: E402
    FAMILY, LocalTok, OFF_IDS, ON_IDS, PREFIX, RenderBackend, THINK_SHAPE_REPLY, _client, _header, _probe, _serve_chain)

ON, OFF = C.THINKING_ON, C.THINKING_OFF


# --------------------------------------------------------------------------- #
# F-AI — the bites probe renders in the attested regime                         #
# --------------------------------------------------------------------------- #


def test_the_bites_probe_render_follows_the_attested_regime_only_on_a_native_standup_attesting_on():
    d = srun._bites_probe_render_arm
    # the fitted GSM8K child: a fitted (lens_cell) cell, header_located False (the native engine's fact), run arm
    # ON, native standup attesting 'on' -> the probe renders ON, and says which rule chose it
    assert d(ON, lens_cell=True, header_located=False, standup_native=True, standup_thinking_mode="on") == \
        (ON, srun.PROBE_RENDER_REASON_ATTESTED_ON)
    # the OFF render stands everywhere else: a record attesting off / nothing, or a LENS standup
    # (its twin refuses undeclared ON)
    for native, mode in ((True, "off"), (True, None), (False, "on"), (False, None)):
        assert d(ON, lens_cell=True, header_located=False, standup_native=native, standup_thinking_mode=mode) == \
            (OFF, srun.PROBE_RENDER_REASON_D22_OFF), (native, mode)
    # a header-located (declared) cell, a non-lens cell, an OFF run: the run's own arm, whatever the record attests
    assert d(ON, lens_cell=True, header_located=True, standup_native=True, standup_thinking_mode="on") == (ON, srun.PROBE_RENDER_REASON_RUN_ARM)
    assert d(ON, lens_cell=False, header_located=False, standup_native=True, standup_thinking_mode="on") == (ON, srun.PROBE_RENDER_REASON_RUN_ARM)
    assert d(OFF, lens_cell=True, header_located=False, standup_native=True, standup_thinking_mode="on") == (OFF, srun.PROBE_RENDER_REASON_RUN_ARM)
    assert d(None, lens_cell=True, header_located=False, standup_native=True, standup_thinking_mode="on") == (None, None)
    # the ON probe's body IS the certified ON arm's kwargs (no declaration on an all-position cell)
    assert srun._probe_extra_body(ON, header_located=False) == {C.CHAT_TEMPLATE_KWARGS_KEY: {C.ENABLE_THINKING_KEY: True}}
    assert usr._probe_channel_policy(srun._probe_extra_body(ON, header_located=False)) == usr.PROBE_CHANNELS_BOTH
    assert usr._probe_channel_policy(srun._probe_extra_body(OFF, header_located=False)) == usr.PROBE_CHANNELS_CONTENT_ONLY


class BitesBackend(RenderBackend):
    """The stub engine whose reply DIFFERS by the upstream model the proxy resolved (the steered adapter vs the base)
    and answers in the reasoning channel under ON — so a bite is visible on both channels re-joined."""

    async def chat_completions(self, payload):
        resp = await super().chat_completions(payload)
        msg = resp["choices"][0]["message"]
        msg["reasoning"] = f"thinking as {payload['model']}"
        msg["content"] = f"answer from {payload['model']}"
        return resp


def _bites_chain(tmp_path, *, thinking_mode="on"):
    """fb2's loopback chain with the bites stub: yields ``(client, backend, steered_vmodel, twin_vmodel)`` — the
    fixture's two non-lens routes (the zero adapter and the base), the GSM8K native shape the bites probe runs on."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        import test_served_capability_render_on_pinned as mod
        real = mod.RenderBackend
        mod.RenderBackend = BitesBackend
        try:
            with _serve_chain(tmp_path, thinking_mode=thinking_mode) as (ledger_url, backend, base):
                client = _client(ledger_url, base)
                lora0 = next(m for m in client.served_models if m.endswith("none__lora0"))
                yield client, backend, lora0, base.virtual_model
        finally:
            mod.RenderBackend = real
    return cm()


def test_known_bad_the_off_bites_probe_on_the_on_pinned_chain_dies_with_the_w20_face(tmp_path):
    """The fitted child's exact chain and face: ``assert_steering_bites`` -> ``probe`` -> ``_greedy`` -> the
    client, the payload carrying the OFF render's ``enable_thinking: false`` -> the proxy's typed 400, relayed by
    the ledger."""
    pytest.importorskip("aiohttp")
    with _bites_chain(tmp_path) as (client, backend, steered, twin):
        hits = len(backend.payloads)
        probe = usr._served_probe_fn(client.chat_completions, steered, twin,
                                     extra_body=srun._probe_extra_body(OFF, header_located=False))
        with pytest.raises(px.ProxyError) as ei:
            bs.assert_steering_bites(("lora_r4_ce",), probe)
        assert len(backend.payloads) == hits                          # nothing forwarded
    msg = str(ei.value)
    assert "returned 400" in msg and px.THINKING_REGIME_MISMATCH in msg, msg
    assert "enable_thinking: false on a thinking-ON standup's tokenize" in msg and '"via": "ledger_tokenize"' in msg, msg


def test_fixed_the_on_bites_probe_on_the_on_pinned_chain_bites_on_both_channels_with_on_stamps(tmp_path):
    pytest.importorskip("aiohttp")
    with _bites_chain(tmp_path) as (client, backend, steered, twin):
        hits = len(backend.payloads)
        extra = srun._probe_extra_body(*srun._bites_probe_render_arm(
            ON, lens_cell=True, header_located=False, standup_native=True, standup_thinking_mode="on")[:1],
            header_located=False)
        probe = usr._served_probe_fn(client.chat_completions, steered, twin, extra_body=extra)
        diag = bs.assert_steering_bites(("lora_r4_ce",), probe)
        n = len(usr._SERVED_PROBE_PROMPTS)
        assert diag["arms"]["lora_r4_ce"]["n"] == n and diag["arms"]["lora_r4_ce"]["n_differing"] == n
        # every payload of BOTH legs asked for the ON render — the certified ON arm's kwargs, no declaration
        sent = backend.payloads[hits:]
        assert len(sent) == 2 * n
        assert all(p[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: True} for p in sent)
        assert all(C.OOS_THINKING_DECLARATION_KEY not in p for p in sent)
        # the proxy stamped every leg thinking_on, and the probe compared reasoning + content (ON vs ON)
        for leg in (usr.PROBE_LEG_STEERED, usr.PROBE_LEG_SCALE0):
            stamps = probe.stamps_by_leg[leg]
            assert len(stamps) == n and {s[C.STAMP_KEY_RENDER_REGIME] for s in stamps} == {C.RENDER_REGIME_THINKING_ON}
            assert probe.stamp_errors_by_leg[leg] == []
            assert all(r["policy"] == usr.PROBE_CHANNELS_BOTH and r["reasoning_chars"] > 0 for r in probe.channels_by_leg[leg])
        assert probe.channel_policy == usr.PROBE_CHANNELS_BOTH
        # the expected stamp row for a NATIVE leg under the ON render is what the proxy stamped
        row = C.expected_stamp_regimes(lens=False, arm=ON, policy=C.POLICY_NOT_CONSULTED)._asdict()
        assert row[C.STAMP_KEY_RENDER_REGIME] == C.RENDER_REGIME_THINKING_ON


def test_the_bites_evidence_records_which_rule_chose_the_render(monkeypatch):
    """``_served_evidence``'s bites block carries the reason and the attested regime beside ``probe_render_arm``."""
    recorded = {}

    def fake_probe_fn(chat, vmodel, twin_vmodel, *, extra_body=None):
        def probe(_arm):
            return [[1, 2, 3]], [[1, 2, 4]]
        probe.stamps_by_leg = {usr.PROBE_LEG_STEERED: [], usr.PROBE_LEG_SCALE0: []}
        probe.stamp_errors_by_leg = {usr.PROBE_LEG_STEERED: [], usr.PROBE_LEG_SCALE0: []}
        probe.channels_by_leg = {}
        probe.channel_policy = usr._probe_channel_policy(extra_body or {})
        probe.extra_body = dict(extra_body or {})
        recorded["extra"] = dict(extra_body or {})
        return probe

    monkeypatch.setattr(usr, "_served_probe_fn", fake_probe_fn)
    # the block's shape is asserted through the same dict the driver writes; the render-arm fields are additive
    bites = {}
    probe = fake_probe_fn(None, "petri__x", "petri__none__lora0", extra_body=srun._probe_extra_body(ON, header_located=False))
    bites[C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] = ON
    bites["probe_render_arm_reason"] = srun.PROBE_RENDER_REASON_ATTESTED_ON
    bites["standup_thinking_mode"] = "on"
    bites["probe_channel_policy"] = probe.channel_policy
    assert bites["probe_channel_policy"] == usr.PROBE_CHANNELS_BOTH
    assert recorded["extra"] == {C.CHAT_TEMPLATE_KWARGS_KEY: {C.ENABLE_THINKING_KEY: True}}
    # the constants are distinct, so a reader can never confuse the three rules
    assert len({srun.PROBE_RENDER_REASON_RUN_ARM, srun.PROBE_RENDER_REASON_D22_OFF, srun.PROBE_RENDER_REASON_ATTESTED_ON}) == 3


# --------------------------------------------------------------------------- #
# F-AJ — the OFF header reference must be populated                             #
# --------------------------------------------------------------------------- #


class DictTok:
    """A tokenizer whose ``apply_chat_template(tokenize=True)`` returns transformers' ``BatchEncoding`` — the 5.x
    default (``return_dict=True``) the container runs (5.14.1). Renders the family's headers correctly."""

    chat_template = "{{ synthetic template }}"

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True, enable_thinking=False, **kw):
        from transformers import BatchEncoding
        ids = list(PREFIX)
        if add_generation_prompt:
            ids += ON_IDS if enable_thinking else OFF_IDS
        return BatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})


class EmptyHeaderTok(LocalTok):
    """A local template that appends NOTHING under the OFF kwargs (the two renders are identical)."""

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True, enable_thinking=False, **kw):
        if add_generation_prompt and not enable_thinking:
            return list(PREFIX)
        return super().apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize,
                                           enable_thinking=enable_thinking, **kw)


def test_known_bad_a_batchencoding_render_read_as_a_tuple_is_its_keys_and_the_header_was_empty():
    """The mechanism, pinned on the real class: ``tuple()`` over a ``BatchEncoding`` is its keys, identical for
    both renders — so at 56aff576e ``_header_ids_under_arm`` returned ``()`` for a tokenizer that renders correctly."""
    from transformers import BatchEncoding
    be = BatchEncoding({"input_ids": PREFIX + OFF_IDS, "attention_mask": [1] * 10})
    assert tuple(be) == ("input_ids", "attention_mask")
    # fixed: the ids are read off the mapping
    assert srun._render_token_ids(be, arm=OFF, what="x") == tuple(PREFIX + OFF_IDS)
    assert srun._render_token_ids(PREFIX + ON_IDS, arm=ON, what="x") == tuple(PREFIX + ON_IDS)
    assert srun._render_token_ids({"input_ids": [PREFIX + OFF_IDS]}, arm=OFF, what="x") == tuple(PREFIX + OFF_IDS)
    # a shape the header cannot be read off refuses by name (never compared)
    for bad in ({"attention_mask": [1]}, "a string", [1, "2"], {"input_ids": [[1], [2]]}, None):
        with pytest.raises(srun.HeaderMeasurementError, match="not a list of token ids"):
            srun._render_token_ids(bad, arm=OFF, what="x")
    assert srun._header_ids_under_arm(DictTok(), OFF) == tuple(OFF_IDS)
    assert srun._header_ids_under_arm(DictTok(), ON) == tuple(ON_IDS)


def test_known_bad_an_empty_off_reference_used_to_reach_the_tuple_compare_and_now_refuses_by_name():
    """The ``none`` child's face — ``measured header ids {'on': [...], 'off': []} != the family's`` — is never
    filed again: an empty OFF reference refuses by name and the compare of ``[]`` is not run."""
    refusal = {"status": 400, "code": px.THINKING_REGIME_MISMATCH, "via": "ledger_tokenize", "message": "m",
               "error": {"message": "m", "type": "invalid_request_error", "code": px.THINKING_REGIME_MISMATCH,
                         "via": "ledger_tokenize"}}
    with pytest.raises(C.RenderCertificationError) as ei:
        C.render_certified(family=FAMILY, header_ids_by_arm={ON: ON_IDS, OFF: []}, measured_live=False,
                           probe_regimes={OFF: C.PROBE_REGIME_REFUSED_REGIME_MISMATCH, ON: C.RENDER_REGIME_THINKING_ON},
                           on_completion={"content": THINK_SHAPE_REPLY, "reasoning": None, "finish_reason": "stop"},
                           header_source_by_arm={OFF: "local_tokenizer", ON: "served_tokenize"},
                           standup_thinking_mode="on", off_arm_refusal=refusal)
    problems = ei.value.evidence["certification_problems"]
    assert any("OFF header ids from the local-tokenizer reference" in p and "EMPTY" in p for p in problems), problems
    assert not any(p.startswith("measured header ids ") for p in problems), problems       # `[]` is never compared
    assert "'off': []} != the family's" not in str(ei.value)
    assert ei.value.evidence[C.OOS_EVIDENCE_HEADER_CERTIFIED] is False and ei.value.evidence["off_arm_accepted_under"] is None
    # a populated reference under the same inputs certifies (the compare runs and agrees)
    rec = C.render_certified(family=FAMILY, header_ids_by_arm={ON: ON_IDS, OFF: OFF_IDS}, measured_live=False,
                             probe_regimes={OFF: C.PROBE_REGIME_REFUSED_REGIME_MISMATCH, ON: C.RENDER_REGIME_THINKING_ON},
                             on_completion={"content": THINK_SHAPE_REPLY, "reasoning": None, "finish_reason": "stop"},
                             header_source_by_arm={OFF: "local_tokenizer", ON: "served_tokenize"},
                             standup_thinking_mode="on", off_arm_refusal=refusal)
    assert rec["render_certified"] is True and rec["measured_header_ids_by_arm"] == {OFF: OFF_IDS, ON: ON_IDS}
    # an empty OFF header OUTSIDE the rule (a served measurement) keeps the pre-existing compare text
    with pytest.raises(C.RenderCertificationError) as ei:
        C.render_certified(family=FAMILY, header_ids_by_arm={ON: ON_IDS, OFF: []}, measured_live=True,
                           probe_regimes={OFF: C.RENDER_REGIME_THINKING_OFF, ON: C.RENDER_REGIME_THINKING_ON},
                           on_completion={"content": THINK_SHAPE_REPLY, "reasoning": None, "finish_reason": "stop"})
    assert any(p.startswith("measured header ids ") for p in ei.value.evidence["certification_problems"])


def test_the_header_measurement_refuses_an_empty_off_reference_by_name_and_reads_a_batchencoding_tokenizer():
    def off_refused(body):
        if (body.get(C.CHAT_TEMPLATE_KWARGS_KEY) or {}).get(C.ENABLE_THINKING_KEY) is False:
            raise px.ProxyError("proxy http://127.0.0.1:1/v1/tokenize returned 400: " + json.dumps(
                {"error": {"message": "m", "type": "invalid_request_error", "code": px.THINKING_REGIME_MISMATCH}}))
        return {"count": 8, "tokens": PREFIX + (ON_IDS if body.get("add_generation_prompt") else [])}

    by_task = {C.GSM8K_STANDARD_TASK: ON}
    # the empty reference: an error by name, never `off: []` carried into the block
    block, err = srun.measure_thinking_header_by_arm(by_task, lens=False, tok=EmptyHeaderTok(), tokenize=off_refused,
                                                     vmodel="petri__base", measure_both_arms=True, standup_thinking_mode="on")
    assert isinstance(err, srun.HeaderMeasurementError) and "appended NO assistant header under the 'off' arm" in str(err)
    assert "empty reference is never compared" in str(err) and block["error"] == str(err)
    assert block["header_ids_by_arm"].get(OFF) in (None, []) and block["verified"] is False
    assert block[srun.HEADER_KEY_OFF_ARM_ACCEPTED_UNDER] is None and srun._header_measured_live(block) is False
    # the 5.x BatchEncoding tokenizer: the reference is populated with the fact's OFF ids
    block, err = srun.measure_thinking_header_by_arm(by_task, lens=False, tok=DictTok(), tokenize=off_refused,
                                                     vmodel="petri__base", measure_both_arms=True, standup_thinking_mode="on")
    assert err is None, err
    assert block["header_ids_by_arm"] == {OFF: OFF_IDS, ON: ON_IDS}
    assert block["source"] == srun.HEADER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF
    assert block[srun.HEADER_KEY_OFF_HEADER_ROLE] == C.OFF_HEADER_ROLE_LOCAL_REFERENCE


def test_the_real_chain_reaches_a_pass_with_both_header_arms_populated_under_a_batchencoding_tokenizer(tmp_path):
    """The GSM8K ``none`` child's path end to end on the ON-pinned chain — the real probe, the real header
    measurement with a tokenizer that returns ``BatchEncoding`` (the container's transformers), the real
    certification: PASS, ``render_source = served_tokenize:on/local_reference:off``, both arms the fact's ids."""
    pytest.importorskip("aiohttp")
    with _serve_chain(tmp_path) as (ledger_url, backend, base):
        client = _client(ledger_url, base)
        probe = _probe(client, base, standup_thinking_mode="on")
        block, err = srun.measure_thinking_header_by_arm(
            {C.GSM8K_STANDARD_TASK: ON}, lens=False, tok=DictTok(), tokenize=client.tokenize, vmodel=base.virtual_model,
            served_template_hash="ab" * 8, allow_out_of_support=True, measure_both_arms=True, header_located=False,
            standup_thinking_mode="on")
        assert err is None, err
        assert block["header_ids_by_arm"] == {OFF: OFF_IDS, ON: ON_IDS}          # both arms populated, the fact's ids
        assert block["header_ids_by_arm"][OFF] == list(C.FAMILY_RENDER_FACTS[FAMILY].off_ids)
        record, error = srun._render_certification(
            probe, family=FAMILY, header_ids_by_arm=block["header_ids_by_arm"],
            measured_live=(block["source"] == srun.HEADER_SOURCE_SERVED),
            header_source_by_arm=block[srun.HEADER_KEY_SOURCE_BY_ARM], standup_thinking_mode="on")
    assert error is None, error
    assert record["render_certified"] is True and record["render_source"] == "served_tokenize:on/local_reference:off"
    assert record["measured_header_ids_by_arm"] == {OFF: OFF_IDS, ON: ON_IDS}
    assert record["off_header_role"] == "local_reference" and record["off_arm_accepted_under"] is not None
    # the fd2 fixture's list-returning tokenizer reaches the same PASS (both shapes read the same ids)
    (tmp_path / "b").mkdir()
    with _serve_chain(tmp_path / "b") as (ledger_url, backend, base):
        client = _client(ledger_url, base)
        block, err = _header(client, base, standup_thinking_mode="on")
        assert err is None and block["header_ids_by_arm"] == {OFF: OFF_IDS, ON: ON_IDS}
