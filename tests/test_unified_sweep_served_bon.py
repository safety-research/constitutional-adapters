"""The sweep driver's served-BoN seam (opt-in --bon-engine served).

CPU-only, no server: the routes come from the same canonical petri cell
set the proxy's own tests build, and the probe is driven with a stub chat
callable. What must hold:

1. CELL TRANSLATION. The sweep's cell ids map onto the petri harness ids
   the routing table speaks — including the sweep's
   ``stacked__<cell>`` grammar (harness stacked ids carry the SI without
   the ``si_`` object prefix) — and anything the petri matrix cannot
   express refuses loudly (never a silent fall-back to the HF engine).
2. ROUTE RESOLUTION. A cell the serving launch did not export is a loud
   BonServedError naming both ids; a mapped cell resolves to its route.
3. THE PROBE goes through the SAME chat surface as the run, greedy
   (temperature 0, fixed seed), never sends a proxy-owned key
   (cache_salt), and encodes the two legs' texts as the id rows
   fidelity._check_steering_bites compares.
4. SETUP REFUSALS. --bon-engine served without the proxy URL or the
   routing table is a loud SystemExit before anything generates.
"""

import importlib.util
from pathlib import Path

import pytest

from sisteer.capeval.serving import proxy as px
from sisteer.eval.attacks.bon_served import BonServedError

from test_petri_proxy import _cells, _source, _spec

_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "unified_sweep_run.py"


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location("unified_sweep_run", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _routes(tmp_path):
    return px.routes_from_cells(_cells(_spec(serving_mode="interleaved"),
                                       _source(tmp_path)))


# --------------------------------------------------------------------------- #
# 1. cell-id translation (sweep -> harness), stacked grammar included           #
# --------------------------------------------------------------------------- #


def test_served_harness_cell_id_maps_the_vocabularies(drv):
    f = drv._served_harness_cell_id
    assert f("none") == "none"
    assert f("prompted__safety_refusal") == "prompted__safety"
    assert f("prompted__si_constitution") == "prompted__constitution"
    assert f("safety_refusal__psr_kl_all") == "si_safety__psr_kl_all"
    assert f("si_constitution__vector_all") == "si_constitution__vector_all"
    assert f("antidote") == "antidote"          # bare Class B coincides
    # the sweep's stacked grammar: harness stacked ids carry the SI WITHOUT
    # the si_ object prefix (defense_cells.parse_cell_id)
    assert f("stacked__safety_refusal__psr_kl_all") == "stacked__safety__psr_kl_all"
    assert f("stacked__antidote") == "stacked__antidote"


def test_served_harness_cell_id_parses_as_petri_grammar(drv):
    """The translated stacked ids are REAL petri ids — parse_cell_id accepts
    them with the stacked class, so a routing table exported from a stacked
    cell set speaks exactly these strings."""
    from sisteer.capeval.petri.defense_cells import parse_cell_id

    parsed = parse_cell_id(
        drv._served_harness_cell_id("stacked__safety_refusal__psr_kl_all"))
    assert parsed.cell_class == "stacked"
    assert parsed.si_id == "safety" and parsed.method == "psr_kl_all"


def test_served_harness_cell_id_refuses_unmappable(drv):
    """A sweep-only cell (the alphakl ladder) has no harness id — loud,
    never silently renamed or dropped."""
    from sisteer.capeval.petri.registry_adapter import RegistryAdapterError

    with pytest.raises(RegistryAdapterError, match="alphakl"):
        drv._served_harness_cell_id("safety_refusal__alphakl_null")


# --------------------------------------------------------------------------- #
# 2. route resolution over the exported table                                  #
# --------------------------------------------------------------------------- #


def test_served_route_found_for_mapped_cell(drv, tmp_path):
    routes = _routes(tmp_path)
    route = drv._served_route_for("safety_refusal__psr_kl_all", routes)
    assert route.cell_id == "si_safety__psr_kl_all"
    assert route.virtual_model  # a served identity exists


def test_served_route_missing_is_loud(drv, tmp_path):
    routes = _routes(tmp_path)
    with pytest.raises(BonServedError, match="no route in"):
        drv._served_route_for("stacked__antidote", routes)


# --------------------------------------------------------------------------- #
# 3. the probe: same surface, greedy, no proxy-owned keys, byte rows           #
# --------------------------------------------------------------------------- #


def test_served_probe_discipline(drv):
    sent = []

    def chat(payload):
        sent.append(dict(payload))
        # steered vs scale0 legs answer differently on every prompt
        text = f"{payload['model']}::{payload['messages'][-1]['content']}"
        return {"choices": [{"message": {"content": text}}]}

    probe = drv._served_probe_fn(chat, "petri__cell", "petri__twin")
    steered, scale0 = probe("psr_kl_all")
    n = len(drv._SERVED_PROBE_PROMPTS)
    assert len(steered) == len(scale0) == n
    assert all(isinstance(row, list) and all(isinstance(b, int) for b in row)
               for row in steered + scale0)
    # every request is greedy, seeded, and NEVER carries a proxy-owned key
    assert len(sent) == 2 * n
    for req in sent:
        assert req["temperature"] == 0.0 and req["seed"] == 0
        assert not (set(req) & set(px.FORBIDDEN_CLIENT_KEYS))
    assert {r["model"] for r in sent} == {"petri__cell", "petri__twin"}
    # the rows feed assert_steering_bites: differing legs pass...
    from sisteer.eval.attacks import bon_served as bs
    diag = bs.assert_steering_bites(["psr_kl_all"], lambda arm: (steered, scale0))
    assert diag["arms"]["psr_kl_all"]["n_differing"] == n

    def same_chat(payload):
        return {"choices": [{"message": {
            "content": payload["messages"][-1]["content"]}}]}

    # ...and identical legs (the silently-unhooked server) refuse
    from sisteer.capeval.serving.fidelity import FidelityError
    inert = drv._served_probe_fn(same_chat, "petri__cell", "petri__twin")
    with pytest.raises(FidelityError, match="NO"):
        bs.assert_steering_bites(["psr_kl_all"], inert)


# --------------------------------------------------------------------------- #
# 3b. the probe reads BOTH channels (gsm8k-paired200 rows 9 / 10)              #
# --------------------------------------------------------------------------- #
#
# The declared header-located gemma cells (`safety_refusal__{psr_kl_all,lora_r4}@thinking`, served positions
# `response`) are the ONLY cells whose bites probe carries the thinking-ON declaration on both legs
# (`_probe_extra_body(render_arm, header_located=True)`); under ON gemma's server reasoning parser moves the
# probe's 48 tokens into `reasoning` and leaves `content` None, and `_greedy` crashed on `None.encode`. The
# undeclared all-position cells probe thinking-OFF and never saw it.

_ON_EXTRA = {"chat_template_kwargs": {"enable_thinking": True}, "petri_thinking_on": True}


def _thinking_on_chat(sent, *, steered_reasoning=True, twin_reasoning=True):
    """A stub proxy whose replies are shaped like the served thinking-ON probe: `content` None, the reasoning
    channel under this vLLM's `reasoning` key on the steered leg and the older `reasoning_content` on the twin."""
    def chat(payload):
        sent.append(dict(payload))
        prompt = payload["messages"][-1]["content"]
        if payload["model"] == "petri__cell":
            msg = {"content": None, "reasoning": f"steered thought about: {prompt}" if steered_reasoning else None}
        else:
            msg = {"content": None, "reasoning_content": f"twin thought about: {prompt}" if twin_reasoning else None}
        return {"choices": [{"message": msg, "finish_reason": "length"}]}
    return chat


def test_a_thinking_on_probe_compares_on_the_reasoning_channel_when_content_is_none(drv):
    """(1) steered `content` None + reasoning present -> the probe compares on the reasoning, no crash, and the
    bite is ruled on (the legs' reasoning differs)."""
    from sisteer.eval.attacks import bon_served as bs

    sent = []
    probe = drv._served_probe_fn(_thinking_on_chat(sent), "petri__cell", "petri__twin", extra_body=_ON_EXTRA)
    steered, scale0 = probe("psr_kl_all")
    n = len(drv._SERVED_PROBE_PROMPTS)
    assert len(steered) == len(scale0) == n and all(row for row in steered + scale0)      # every row carries bytes
    assert bytes(steered[0]).decode() == f"steered thought about: {drv._SERVED_PROBE_PROMPTS[0]}"
    assert bytes(scale0[0]).decode() == f"twin thought about: {drv._SERVED_PROBE_PROMPTS[0]}"   # either spelling
    diag = bs.assert_steering_bites(["psr_kl_all"], lambda arm: (steered, scale0))
    assert diag["arms"]["psr_kl_all"]["n_differing"] == n
    # the declaration rode every payload on BOTH legs (ONE shape), and the channel facts are recorded per leg
    assert all(req["petri_thinking_on"] is True and req["chat_template_kwargs"] == {"enable_thinking": True}
               for req in sent) and len(sent) == 2 * n
    for leg in (drv.PROBE_LEG_STEERED, drv.PROBE_LEG_SCALE0):
        rows = probe.channels_by_leg[leg]
        assert [r["prompt_index"] for r in rows] == list(range(n))
        assert all(r["content_chars"] == 0 and r["reasoning_chars"] > 0 and r["finish_reason"] == "length"
                   and r["policy"] == drv.PROBE_CHANNELS_BOTH for r in rows)
    assert probe.channel_policy == drv.PROBE_CHANNELS_BOTH


def test_a_thinking_on_probe_joins_reasoning_and_content_in_emission_order(drv):
    """Steering that changes ONLY the reasoning still bites: identical `content` on both legs, different reasoning."""
    from sisteer.eval.attacks import bon_served as bs

    def chat(payload):
        who = "steered" if payload["model"] == "petri__cell" else "twin"
        return {"choices": [{"message": {"content": "43", "reasoning": f"{who} working"}, "finish_reason": "stop"}]}

    probe = drv._served_probe_fn(chat, "petri__cell", "petri__twin", extra_body=_ON_EXTRA)
    steered, scale0 = probe("psr_kl_all")
    assert bytes(steered[0]).decode() == "steered working43" and bytes(scale0[0]).decode() == "twin working43"
    assert bs.assert_steering_bites(["psr_kl_all"], lambda arm: (steered, scale0))["arms"]["psr_kl_all"]["n_differing"] \
        == len(drv._SERVED_PROBE_PROMPTS)


def test_a_probe_draw_with_neither_channel_fails_closed_by_name(drv):
    """(2) both channels None -> FidelityError naming the leg and the prompt index (never the text), raised out of
    the bites control so `_served_evidence` records it as an unverified bite and the driver aborts — no
    certification on an empty draw, and no AttributeError."""
    from sisteer.capeval.serving.fidelity import FidelityError
    from sisteer.eval.attacks import bon_served as bs

    sent = []
    probe = drv._served_probe_fn(_thinking_on_chat(sent, steered_reasoning=False), "petri__cell", "petri__twin",
                                 extra_body=_ON_EXTRA)
    with pytest.raises(FidelityError, match=r"leg 'steered' prompt 0: the completion carries NO text in either channel"):
        bs.assert_steering_bites(["psr_kl_all"], probe)
    assert len(sent) == 1                                                  # refused on the first empty draw
    assert probe.channels_by_leg[drv.PROBE_LEG_STEERED] == [
        {"prompt_index": 0, "content_chars": 0, "reasoning_chars": 0, "finish_reason": "length",
         "policy": drv.PROBE_CHANNELS_BOTH}]
    # a JSON null / missing key on the TWIN leg refuses the same way, naming that leg
    probe2 = drv._served_probe_fn(_thinking_on_chat([], twin_reasoning=False), "petri__cell", "petri__twin",
                                  extra_body=_ON_EXTRA)
    with pytest.raises(FidelityError, match=r"leg 'scale0' prompt 0"):
        bs.assert_steering_bites(["psr_kl_all"], probe2)


def test_the_thinking_off_probe_bytes_are_unchanged(drv):
    """(3) the OFF path: a content-only reply (the 36 legs, the MATH-500 runs, the battery) encodes exactly
    `content` — byte-identical to the pre-fix probe — with or without a `None` reasoning key, and the request
    body is untouched."""
    sent = []

    def chat(payload):
        sent.append(dict(payload))
        text = f"{payload['model']}::{payload['messages'][-1]['content']}"
        msg = {"content": text} if payload["model"] == "petri__cell" else {"content": text, "reasoning": None}
        return {"choices": [{"message": msg, "finish_reason": "stop"}]}

    probe = drv._served_probe_fn(chat, "petri__cell", "petri__twin")
    steered, scale0 = probe("psr_kl_all")
    for i, prompt in enumerate(drv._SERVED_PROBE_PROMPTS):
        assert steered[i] == list(f"petri__cell::{prompt}".encode())
        assert scale0[i] == list(f"petri__twin::{prompt}".encode())
    for req in sent:
        assert set(req) == {"model", "messages", "temperature", "max_tokens", "seed"}
        assert req["temperature"] == 0.0 and req["seed"] == 0 and req["max_tokens"] == drv._SERVED_PROBE_MAX_TOKENS
    assert all(r["reasoning_chars"] == 0 and r["content_chars"] > 0 and r["policy"] == drv.PROBE_CHANNELS_CONTENT_ONLY
               for leg in probe.channels_by_leg.values() for r in leg)
    assert probe.channel_policy == drv.PROBE_CHANNELS_CONTENT_ONLY


_OFF_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("key", ["reasoning", "reasoning_content"])
def test_an_off_probe_never_reads_a_populated_reasoning_channel(drv, key):
    """An OFF probe runs on thinking-ON standups too (the undeclared all-position cells' bites probes
    render OFF, with gemma's reasoning parser installed), and nothing says its reasoning channel is
    empty. The policy follows the REQUESTED render: OFF compares content only, so two legs with identical
    `content` and DIFFERENT non-empty reasoning are an inert arm exactly as at HEAD — under either spelling."""
    from sisteer.capeval.serving.fidelity import FidelityError
    from sisteer.eval.attacks import bon_served as bs

    def chat(payload):
        who = "steered" if payload["model"] == "petri__cell" else "twin"
        return {"choices": [{"message": {"content": "43", key: f"{who} channel text"}, "finish_reason": "stop"}]}

    for extra in (_OFF_EXTRA, None):                                          # the OFF probe and the legacy probe
        probe = drv._served_probe_fn(chat, "petri__cell", "petri__twin", extra_body=extra)
        assert probe.channel_policy == drv.PROBE_CHANNELS_CONTENT_ONLY
        steered, scale0 = probe("psr_kl_all")
        assert steered == scale0 == [list(b"43")] * len(drv._SERVED_PROBE_PROMPTS)     # HEAD's bytes: content alone
        with pytest.raises(FidelityError, match="NO difference"):
            bs.assert_steering_bites(["psr_kl_all"], lambda arm, s=steered, z=scale0: (s, z))
        assert all(r["reasoning_chars"] > 0 and r["policy"] == drv.PROBE_CHANNELS_CONTENT_ONLY
                   for leg in probe.channels_by_leg.values() for r in leg)         # seen, recorded, not compared
    # ... and the same replies under the ON request compare on both channels and bite
    on_probe = drv._served_probe_fn(chat, "petri__cell", "petri__twin", extra_body=_ON_EXTRA)
    steered, scale0 = on_probe("psr_kl_all")
    assert bytes(steered[0]).decode() == "steered channel text43" and bytes(scale0[0]).decode() == "twin channel text43"


def test_the_channel_policy_is_read_off_the_requested_render_only(drv):
    """`enable_thinking` must be the JSON true inside `chat_template_kwargs` — the proxy's own key; the declaration
    alone, a string, a bare top-level key or nothing at all is content-only."""
    assert drv._probe_channel_policy({"chat_template_kwargs": {"enable_thinking": True}}) == drv.PROBE_CHANNELS_BOTH
    assert drv._probe_channel_policy({"chat_template_kwargs": {"enable_thinking": True}, "petri_thinking_on": True}) == drv.PROBE_CHANNELS_BOTH
    for off in ({}, None, {"chat_template_kwargs": {"enable_thinking": False}}, {"petri_thinking_on": True},
                {"enable_thinking": True}, {"chat_template_kwargs": {"enable_thinking": "true"}},
                {"chat_template_kwargs": {"enable_thinking": 1}}, {"chat_template_kwargs": None}):
        assert drv._probe_channel_policy(off) == drv.PROBE_CHANNELS_CONTENT_ONLY, off


# --------------------------------------------------------------------------- #
# 4. setup refusals (no URL / no table -> SystemExit before generating)        #
# --------------------------------------------------------------------------- #


class _Cond:
    cell = "safety_refusal__psr_kl_all"


class _Args:
    bon_proxy_url = None
    bon_routing_table = None


def test_setup_refuses_without_proxy_url_and_table(drv):
    with pytest.raises(SystemExit, match="--bon-proxy-url"):
        drv._bon_served_setup(_Cond(), _Args(), defense=None)


def test_setup_refuses_with_only_one_of_the_two(drv, tmp_path):
    a = _Args()
    a.bon_proxy_url = "http://localhost:8814"
    with pytest.raises(SystemExit, match="--bon-routing-table"):
        drv._bon_served_setup(_Cond(), a, defense=None)


def test_cli_defaults_keep_the_hf_engine(drv):
    """The served path is OPT-IN: --gen-engine and --bon-engine both default to
    None (no default proxy/table), and _effective_engines resolves an
    invocation that says nothing to HF for the WHOLE fixed-prompt leg (whose
    bon_generate defaults to width DEFAULT_BON_BATCH_SIZE=64 — engine unchanged,
    width defaulted up). --bon-engine served ALONE still means BoN-only served
    (the pre-generalisation behaviour is preserved)."""
    import argparse
    f = drv._effective_engines
    # default (nothing said) -> HF everywhere.
    assert f(argparse.Namespace(gen_engine=None, bon_engine=None)) == ("hf", "hf")
    # bon-only served preserved; the general knob serves the whole leg.
    assert f(argparse.Namespace(gen_engine=None,
                                bon_engine="served")) == ("hf", "served")
    assert f(argparse.Namespace(gen_engine="served",
                                bon_engine=None)) == ("served", "served")
    src = _DRIVER.read_text()
    assert '"--gen-engine", choices=("hf", "served"), default=None' in src
    assert '"--bon-engine", choices=("hf", "served"), default=None' in src
    assert '"--bon-proxy-url", default=None' in src
    assert '"--bon-routing-table", default=None' in src
    assert '"--bon-max-concurrency", type=int, default=None' in src


def test_stamp_records_effective_bon_width():
    """bon_generate defaults to DEFAULT_BON_BATCH_SIZE (64) when
    --bon-batch-size is absent, so the HF-engine stamp must fall back to that
    default — a stamp keyed only on the CLI arg would understate BoN's width
    for every invocation that says nothing."""
    src = _DRIVER.read_text()
    assert ('getattr(args, "bon_batch_size", None) or DEFAULT_BON_BATCH_SIZE'
            in src)
