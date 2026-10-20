"""The thinking-ON standup's ``--reasoning-parser`` knob (Lane B).

The tinycap matrix runs each test cell twice — Gemma-4's thinking mode ON and
OFF — and the mode travels as a per-request ``chat_template_kwargs``
override that vLLM 0.25.1 merges over the server default. That is enough
to make the model think, and NOT enough to make the numbers readable:
with no reasoning parser composed, the thought text stays in ``content``
and every extractor in the suite (``ANSWER:``, the numeric match, the
choice scorer, the code-fence split) scores the scratchpad.

So ``build_server_launch`` grows one opt-in knob, and the tests below pin
the three things that make it safe rather than merely present:

1. **Absent by default** — a launch that does not ask composes no
   ``--reasoning-parser``, byte-for-byte the argv every launch before
   this change composed (thinking-OFF is still the whole published
   suite);
2. **Held to the family's pin** — ``gemma4`` on a gemma-4 target and
   nothing else; a name that is not the pin refuses, and a family with
   NO pin refuses rather than composing a guess. This is stricter than
   the tool parser's rule (which composes an unknown family's manifest
   pin unchecked) on purpose: a wrong tool parser 500s on the first
   tools-carrying request, while a wrong reasoning parser returns 200s
   forever with the answer field quietly wrong;
3. **Still manifest-forbidden** — ``--reasoning-parser`` stays in
   ``_WRAPPER_OWNED_OPTIONS``, so the ONE caller that flips the render
   also names the parser, through the wrapper.

Plus the ``capctrl_serving_up.py`` passthrough: the flag reaches
``build_server_launch`` and the composed name is recorded in
``endpoint.json`` so the cell manifest states what it was served by.

CPU only; no GPU, no weights, no network (a synthetic template and the
name-marker family fallback).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.tinycap import contracts as tc

REPO = Path(__file__).resolve().parents[1]
#: A synthetic gemma-shaped template carrying the thinking switch, so the
#: composer's THINKING_OFF render pin fires the same way the real one does
#: (the point here is the parser, not the template bytes).
THINK_TEMPLATE_BYTES = (b"{# synthetic gemma-4-shaped template #}"
                        b"{% if enable_thinking %}<|think|>\n{% endif %}"
                        b"{% for m in messages %}{{ m['content'] }}{% endfor %}")
GEMMA = "google/gemma-4-E4B-it"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
UNKNOWN = "TESTORG/mystery-7b"
REV = "0" * 40


def _spec_for(target: str, *, tool_parser: str = "gemma4"):
    spec0 = cm.load_petri_spec(
        REPO / "configs/capeval/manifests/petri_twosi.yaml")
    return dataclasses.replace(
        spec0, target_model=target, target_revision=REV,
        serving_mode="interleaved", cells=["none", "prompted__safety"],
        serving_launch_flags=["--enforce-eager"],
        scenario_bundle_hash="ab" * 8, serving_impl_commit="fedcba9876" * 4,
        chat_template_hash=cm.sha256_hex16(THINK_TEMPLATE_BYTES),
        tool_parser=tool_parser)


@pytest.fixture()
def tpl(tmp_path):
    p = tmp_path / "chat_template.jinja"
    p.write_bytes(THINK_TEMPLATE_BYTES)
    return p


def _launch(target: str, tpl, **kw):
    return vs.build_server_launch(_spec_for(target), port=8801,
                                  chat_template_path=tpl, **kw)


# --------------------------------------------------------------------------
# 1. absent by default — the existing launch identity is untouched
# --------------------------------------------------------------------------


def test_no_reasoning_parser_by_default(tpl):
    """The default launch composes nothing: no ``--reasoning-parser``, no
    plugin loader, and the thinking-OFF render pin still fires from the
    template bytes. This is the shape of every launch the published suite
    was served by, so existing controls stay untouched for
    every standup that does not ask."""
    cmd = list(_launch(GEMMA, tpl).cmd)
    assert "--reasoning-parser" not in cmd
    assert "--reasoning-parser-plugin" not in cmd
    assert (cmd[cmd.index("--default-chat-template-kwargs") + 1]
            == vs.THINKING_OFF_TEMPLATE_KWARGS)


def test_explicit_none_is_the_same_argv_as_omitting_it(tpl):
    """``reasoning_parser=None`` is not a third state: it composes the
    identical command, so a launcher that always passes the (possibly
    None) knob cannot change a thinking-OFF standup's identity."""
    assert list(_launch(GEMMA, tpl).cmd) == list(
        _launch(GEMMA, tpl, reasoning_parser=None).cmd)


# --------------------------------------------------------------------------
# 2. composed only at the family's pin
# --------------------------------------------------------------------------


def test_gemma4_composes_the_pinned_parser_once(tpl):
    """The thinking-ON standup: ``--reasoning-parser gemma4``, exactly
    once, with the name read from the contracts' single definition — and
    the render pin unchanged (the mode is a per-REQUEST override; the
    server default stays OFF, which is what makes the thinking-OFF arm of
    the same image a legitimate arm)."""
    cmd = list(_launch(GEMMA, tpl, reasoning_parser="gemma4").cmd)
    assert cmd.count("--reasoning-parser") == 1
    assert (cmd[cmd.index("--reasoning-parser") + 1]
            == tc.REASONING_PARSER_BY_FAMILY["gemma4"] == "gemma4")
    assert (cmd[cmd.index("--default-chat-template-kwargs") + 1]
            == vs.THINKING_OFF_TEMPLATE_KWARGS)


def test_a_name_that_is_not_the_family_pin_refuses(tpl):
    """A parser name off the table is refused at compose time, naming the
    pin. ``deepseek_r1`` is a real registered 0.25.1 reasoning parser, so
    the server would come up healthy and mis-split every completion."""
    with pytest.raises(vs.VllmServerError) as e:
        _launch(GEMMA, tpl, reasoning_parser="deepseek_r1")
    assert "REASONING_PARSER_BY_FAMILY" in str(e.value)
    assert "'gemma4'" in str(e.value)


@pytest.mark.parametrize("target", [LLAMA, UNKNOWN])
def test_a_family_without_a_pin_refuses_rather_than_guessing(target, tpl):
    """Llama resolves to a family with no reasoning pin; the synthetic id
    resolves to no family at all. Both refuse — this is where the rule is
    deliberately stricter than the tool parser's, which composes an
    unknown family's manifest pin unchecked. A mis-fitting reasoning
    parser never 500s; it just moves the answer."""
    with pytest.raises(vs.VllmServerError) as e:
        vs.build_server_launch(
            _spec_for(target, tool_parser="llama3_json"), port=8801,
            chat_template_path=tpl, reasoning_parser="gemma4")
    msg = str(e.value)
    assert "NO" in msg and "pinned reasoning parser" in msg
    assert "gemma4" in msg


@pytest.mark.parametrize("bad", ["", "   ", 7, True])
def test_a_non_name_refuses(bad, tpl):
    with pytest.raises(vs.VllmServerError):
        _launch(GEMMA, tpl, reasoning_parser=bad)


def test_the_pin_table_is_the_contracts_one_and_is_tool_parser_shaped():
    """ONE definition (P24): ``vllm_server`` imports the mapping from
    ``contracts`` rather than mirroring it, and every family that can
    serve thinking-ON must also be a family whose tool parser is pinned —
    the two surfaces are verified together per model."""
    assert dict(tc.REASONING_PARSER_BY_FAMILY) == {"gemma4": "gemma4"}
    assert set(tc.REASONING_PARSER_BY_FAMILY) <= set(vs.TOOL_PARSER_BY_FAMILY)


# --------------------------------------------------------------------------
# 3. still manifest-forbidden, and composable alongside the agent modality
# --------------------------------------------------------------------------


def test_reasoning_parser_stays_wrapper_owned_for_manifests():
    """The knob does not open the surface to manifests: both spellings
    stay owned, so the thinking decision cannot be made twice."""
    assert "--reasoning-parser" in vs._WRAPPER_OWNED_OPTIONS
    assert "--reasoning-parser-plugin" in vs._WRAPPER_OWNED_OPTIONS
    assert "--reasoning-parser" not in vs._PIN_RESTATEMENTS


def test_agent_and_thinking_compose_together(tpl):
    """The prompted thinking-ON container is BOTH: BFCL needs the agent
    modality and the arm needs the reasoning parser. The two pins are
    independent and both land."""
    cmd = list(_launch(GEMMA, tpl, tool_calls=True,
                       reasoning_parser="gemma4").cmd)
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "gemma4"
    assert cmd[cmd.index("--reasoning-parser") + 1] == "gemma4"


def test_contracts_standup_argv_matches_what_the_composer_accepts():
    """``contracts.standup_extra_argv`` builds the flags the launcher
    appends; the name it resolves is the one the composer will accept,
    and a family with no pin refuses there too (both ends, one table)."""
    assert tc.standup_extra_argv(tool_calls=False, thinking=tc.THINKING_OFF) == []
    assert tc.standup_extra_argv(tool_calls=False, thinking=tc.THINKING_ON) == [
        tc.FLAG_REASONING_PARSER, "gemma4"]
    with pytest.raises(ValueError):
        tc.standup_extra_argv(tool_calls=False, thinking=tc.THINKING_ON,
                              family="llama")


# --------------------------------------------------------------------------
# the capctrl_serving_up passthrough
# --------------------------------------------------------------------------


def _standup_parser():
    """The standup's argv parser, without importing its heavy body: the
    script is not a package module, so it is loaded by path (the repo's
    existing convention for script-level tests)."""
    import importlib.util
    import sys

    path = REPO / "scripts" / "capctrl_serving_up.py"
    spec = importlib.util.spec_from_file_location("_rp_capctrl_serving_up", path)
    mod = importlib.util.module_from_spec(spec)
    # registered BEFORE exec: Standup is a dataclass under ``from __future__
    # import annotations``, resolved through sys.modules[cls.__module__]
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_standup_flag_defaults_to_none_and_parses_a_name():
    """``--reasoning-parser`` exists, defaults to None (the thinking-OFF
    standup), and carries the name through as a plain string — ``main`` hands
    it to ``bring_up``, ``bring_up`` to ``build_server_launch`` (which
    validates it against the family's pin) — the script never second-guesses
    the table; and the composed pin is recorded in endpoint.json, so the cell
    manifest's thinking.reasoning_parser states what actually served."""
    from pathlib import Path as _P
    from types import SimpleNamespace

    mod = _standup_parser()
    base = ["--repo-dir", "/r", "--config", "c.yaml", "--cell", "none",
            "--defenses-dir", "/d", "--out", "/o", "--stop-file", "/o/STOP"]
    assert mod.build_parser().parse_args(base).reasoning_parser is None
    assert mod.build_parser().parse_args(
        base + ["--reasoning-parser", "gemma4"]).reasoning_parser == "gemma4"
    src = (REPO / "scripts" / "capctrl_serving_up.py").read_text()
    assert '"--reasoning-parser", default=None' in src
    assert "reasoning_parser=args.reasoning_parser" in src          # main -> bring_up
    # bring_up -> build_server_launch: the kwarg travels by name; it is now
    # followed by sampling_clamp_override=, so the pin reads the kwarg, not the closing paren
    assert "reasoning_parser=reasoning_parser," in src and "sampling_clamp_override=sampling_clamp_override)" in src
    assert '"reasoning_parser": self.reasoning_parser' in src       # Standup.endpoint -> endpoint.json
    assert '"tool_calls": bool(self.tool_calls)' in src
    # behaviourally: a live Standup's endpoint record carries both modality pins,
    # and a Standup built without them is the ordinary thinking-OFF / no-tools one
    geom = SimpleNamespace(max_model_len=16, max_num_seqs=2, as_dict=lambda: {"L": 16})
    common = {"out": _P("/o"), "model": "m", "revision": "r", "cell": "none",
              "harness_cell": "none", "geometry": geom, "launch": None, "server": None,
              "proxy_loop": None, "proxy_url": "http://127.0.0.1:1/v1",
              "table_path": _P("/o/rt.json"), "tokenizer": None}
    ep = mod.Standup(**common, tool_calls=True, reasoning_parser="gemma4").endpoint("capability")
    assert ep["reasoning_parser"] == "gemma4" and ep["tool_calls"] is True
    assert ep["geometry"] == {"L": 16} and ep["eval"] == "capability"
    ep0 = mod.Standup(**common).endpoint(None)
    assert ep0["reasoning_parser"] is None and ep0["tool_calls"] is False
    # PW18 (U4(b)): the record states the proxy's Petri-facing agent-loop entry
    # — ON by default (the proxy's own default), False under the stated opt-out.
    assert ep0["agent_loop_entry"] is True and ep["agent_loop_entry"] is True
    assert mod.Standup(**common, petri_agent_loop=False).endpoint(None)["agent_loop_entry"] is False
    # The lens reduction mode is NO LONGER passed through from a captured
    # ``/v1/models`` block (E25's ``proxy_identity`` field, dropped in the
    # stacked-landing merge): ``endpoint()`` reads the workers' own steering
    # attestation, so the key is always present — ``None`` plus a reason when
    # nothing could be attested, never a guess and never an absent key. There
    # is no attestation under this fake ``out``, so that is what this Standup
    # records; the attested-value cases are pinned in
    # ``tests/test_lens_reduction_mode_attestation.py``.
    assert ep0["lens_reduction_mode"] is None
    assert "FileNotFoundError" in ep0["lens_reduction_mode_error"]
    # the E25 pass-through is gone: no dataclass field, no constructor kwarg
    # (the merge comment that explains WHY still names it, so pin the
    # constructs rather than the word)
    assert "proxy_identity: object" not in src and "proxy_identity=" not in src
    assert '"agent_loop_entry": bool(self.petri_agent_loop)' in src     # Standup.endpoint -> endpoint.json
    assert "petri_agent_loop=bool(petri_agent_loop)," in src           # bring_up -> Standup
    assert mod.build_parser().parse_args(base).petri_agent_loop is True
    assert mod.build_parser().parse_args(base + ["--no-petri-agent-loop"]).petri_agent_loop is False
