"""native-serving-default: the TEXT that states the serving default, and the generated files that
stopped claiming the lens serves every LoRA cell.

1. CLAUDE.md's `### Serving default: native LoRA engine` subsection sits right after
   `### The four model families` and carries the statements the spec requires (native default for every LoRA
   cell; PSR and the plain-vector cells serving natively through the residual slot, with the lens the explicit flag
   for AlphaSteer and the other lens-only kinds; the sampling caveat; never re-run an existing eval, the engine
   identity separating the eras; where the certification records live; ONE native standup per view serving
   `none`, the prompted cells and every LoRA cell).
2. No prose surface still says "every lora cell serves via the LENS o_proj site" — the full22 README's §7 and
   CLAUDE.md — and no generated wave config / manifest, none of the files the downstream generators derive from
   them, and not the generator's own template.
3. The wave CONFIG headers were deliberately left byte-identical: `tests/test_sdfbase_ifeval_overlays.py` pins the
   llama8b main / it campaign configs by git blob sha, so their lens-era serving sentence stays as it is and the
   engine a wave actually ran on is read from the standup's stamps (`endpoint.json.engine`, `serving_engine` on
   every row), never from a config header. Only the manifests' `serving_launch_flags` comment and derived-twin
   sentence changed.
4. PSR and the plain-vector kinds (`vector`, `vector_all`, `diffmean`, `static_r`) serve NATIVELY through the
   residual slot, so every prose surface that said they are "NOT served" / "skipped by name" on a native standup
   was re-pinned (CLAUDE.md's subsection and README §7). The known-bads: the retired sentences must be ABSENT,
   not merely outnumbered.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from published_repo import claude_md_text

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CLAUDE_MD = REPO / "CLAUDE.md"
README = REPO / "scripts/sdfbase/full22/README.md"
README_ENGINE_HEADING = "## Serving engine"   # the full22 README section that states the serving engine
TINYCAP_DOC = REPO / "docs/tinycap_interfaces.md"
GENERATOR = REPO / "scripts/sdfbase/make_full22_configs.py"

LENS_SENTENCE = "every lora cell serves via the LENS o_proj site"
HEADING = "### Serving default: native LoRA engine (user ruling 2026-09-13)"
PRECEDING_HEADING = "### The four model families"

#: the statements the spec requires of CLAUDE.md, each as phrases that must all appear INSIDE the subsection
SIX_STATEMENTS = {
    "native default for every LoRA cell": ("Every LoRA cell", "native multi-LoRA engine", "by default", "--serving-path lora_native"),
    "PSR and plain-vector cells serve natively through the residual slot; the lens is the explicit flag for the lens-only kinds":
        ("PSR and plain-vector cells serve NATIVELY by default", "residual slot", "`native-psr-serving`",
         "NATIVE_STAGE2_ARMED", "AlphaSteer stays on the lens pending its own spec", "--serving-path lens",
         "F22_SERVING_PATH=lens"),
    "the sampling caveat": ("Production decodes by SAMPLING", "greedy-token disagreement", "DIAGNOSTIC", "never an admission gate"),
    "never re-run existing evals; the engine identity separates the eras":
        ("Never re-run an existing eval", "proxy_vllm+lens+interleaved", "proxy_vllm+lora+interleaved", "serving_engine"),
    "where the certification records live":
        ("exp/multilora_cert/", "multilora_cert_report.py", "native-serving-default/decisions.md"),
    "ONE native standup per view serving none, prompted and every LoRA cell":
        ("ONE standup per view", "`none` cell", "prompted", "every LoRA cell", "none__lora0"),
}


def _section(text: str, heading: str) -> str:
    """The body from `heading` (a whole line) to the next heading of the same or a higher level."""
    lines = text.splitlines(keepends=True)
    level = len(heading) - len(heading.lstrip("#"))
    starts = [i for i, l in enumerate(lines) if l.rstrip("\n") == heading]
    assert len(starts) == 1, (heading, len(starts))
    i = starts[0]
    j = next((k for k in range(i + 1, len(lines))
              if re.match(r"^#{1,%d} " % level, lines[k])), len(lines))
    return "".join(lines[i:j])


# --- 1: CLAUDE.md -------------------------------------------------------------------------------------------------

def test_claude_md_states_the_native_default_right_after_the_model_families():
    text = claude_md_text()
    assert text.count(HEADING + "\n") == 1
    h3 = [l.rstrip("\n") for l in text.splitlines() if l.startswith("### ")]
    assert h3[h3.index(HEADING) - 1] == PRECEDING_HEADING, h3[:6]
    body = _section(text, HEADING)
    for statement, phrases in SIX_STATEMENTS.items():
        missing = [ph for ph in phrases if ph not in body]
        assert not missing, (statement, missing)
    # the eras are separated by the identity, not by re-running anything; the caveat is about sampling, not greedy
    assert LENS_SENTENCE.lower() not in body.lower()
    assert "lora_kernel_index_ceiling" in body and "121→107" in body and "148→107" in body and "161→86" in body


#: a position-masked LoRA fit is SKIPPED BY NAME on a native standup, never a refusal of the whole standup — the
#: rule every prose surface must state, with the two views that carried the landmine named
MASKED_SKIP_PHRASES = ("position-masked", "safety_refusal__lora_r4", "D97")


def test_prose_surfaces_state_that_a_position_masked_lora_fit_is_skipped_by_name_not_refused():
    body = _section(claude_md_text(), HEADING)
    for ph in MASKED_SKIP_PHRASES + ("skipped by name", "gemma4e4b / qwen35_4b `main` views", "`positions` missing"):
        assert ph in body, ph
    assert "never a refusal of the whole standup" in body
    readme7 = _section(README.read_text(encoding="utf-8"), README_ENGINE_HEADING)
    assert "position-masked LoRA fit" in readme7 and "skipped by name rather than refusing the whole standup" in readme7
    # and no surface still says a masked fit REFUSES the standup
    for surface in (body, readme7):
        assert "cannot honestly" not in surface, "the pre-D97 refusal phrasing"


#: the statements every prose surface must make about PSR and the plain-vector kinds, and the retired sentences that
#: must be gone (a known-bad each)
PSR_NATIVE_CLAUDE_PHRASES = (
    "PSR and plain-vector cells serve NATIVELY by default",
    "`vector`, `vector_all`, `diffmean`, `static_r`", "residual slot", "rank-1 ZERO o_proj LoRA",
    "`functional-2026-09-13`", "not_measured",
    # position-restricted objects: the flag, both halves armed together, and the landing rule
    "`defense_cells.NATIVE_STAGE2_ARMED` is True", "`native_residual.STAGE2_POSITION_CLASSES`",
    "native-psr-serving D65", "ADMITTED stage-2 certificate", "`gemma4e4b_psr2cert`", "flip\nFalse before the branch lands",
    # AlphaSteer and the lens-only kinds
    "AlphaSteer stays on the lens pending its own spec", "per-REQUEST broadcast op",
    "skipped by name on a mixed native launch", "antidote", "the stacked\nwrapper",
    # lens-era records stand; the identity separates the eras
    "Lens-era PSR / vector rows stand and are never re-run", "differ by the engine\nidentity",
    # what still refuses by name for a residual kind
    "or a residual-kind record it cannot", "`static_r` `site=pre` at block 0",
    # the kind-aware selector's consequence and how to force the lens
    "AlphaSteer + Arditi now selects\n`lora_native` and skips AlphaSteer by name", "GSM8K's\nunpinned PSR rows AUTO-select native",
    "`--serving-path lens` (`F22_SERVING_PATH=lens` for a\nwave job)",
)
#: the sentences the subsection carried before PSR served natively — each must be ABSENT (known-bad)
PSR_RETIRED_CLAUDE_SENTENCES = (
    "NOT served", "We do not expect to run more PSR", "have no PEFT form", "PSR and steering-vector cells",
    "Anyone who still needs a PSR cell",
    # the earlier lens-only list of the kind-aware paragraph
    "Arditi = `static_r`, PSR, vector, the stacked wrapper", "(alphasteer×2, static_r×1)",
)


def test_prose_surfaces_state_that_psr_and_plain_vector_cells_serve_natively():
    """The prose surfaces state the new world and no longer state the old one. A stale statement here would send
    a reader to `--serving-path lens` for a cell the native standup serves."""
    text = claude_md_text()
    body = _section(text, HEADING)
    missing = [ph for ph in PSR_NATIVE_CLAUDE_PHRASES if ph not in body]
    assert not missing, missing
    stale = [s for s in PSR_RETIRED_CLAUDE_SENTENCES if s in body]
    assert not stale, ("the retired wording is still in the subsection", stale)
    # the sampling caveat says a residual-bearing standup measures no greedy-token agreement at all
    assert "residual-bearing standup" in body and "`numerics.mode =\n  not_measured`" in body
    # the pinned LoRA statements keep holding beside the new ones (the masked-fit skip, the refusal rule)
    for ph in MASKED_SKIP_PHRASES + ("skipped by name", "never a refusal of the whole standup", "`positions` missing"):
        assert ph in body, ph

    readme7 = _section(README.read_text(encoding="utf-8"), README_ENGINE_HEADING)
    for ph in ("residual slot", "`NATIVE_STAGE2_ARMED`",
               "AlphaSteer, the other lens-only kinds and any position-masked LoRA fit", "skips AlphaSteer by name",
               "force a PSR / vector cell back onto the lens"):
        assert ph in readme7, ph
    for stale_ph in ("PSR / vector cells and any position-masked", "for anyone who still needs a PSR cell",
                     "Arditi = `static_r`, PSR, vector —", "(alphasteer×2, static_r×1)"):
        assert stale_ph not in readme7, stale_ph
    # the README §7 LoRA pins keep holding
    assert "position-masked LoRA fit" in readme7 and "skipped by name rather than refusing the whole standup" in readme7


# --- 2: the prose surfaces ------------------------------------------------------------------------------------------

def test_prose_surfaces_no_longer_say_the_lens_serves_every_lora_cell():
    readme7 = _section(README.read_text(encoding="utf-8"), README_ENGINE_HEADING)
    assert readme7.startswith(README_ENGINE_HEADING)
    for token in ("endpoint.json.engine", "petri_proxy.engine", "F22_SERVING_PATH=lens", "sdfbase_full22_native",
                  "TARGET_ENGINE", "SERVED_CELLS", "cells_skipped"):
        assert token in readme7, token
    for surface in (readme7, claude_md_text()):
        assert LENS_SENTENCE.lower() not in surface.lower()
    tiny = TINYCAP_DOC.read_text(encoding="utf-8")
    assert "serves an object on the lens route" not in tiny
    assert "BFCL runs on the lens route" not in tiny
    assert tiny.count("TWIN_CELL_ID_LORA") >= 2 and "twin_cell_id_for" in tiny


# --- 3: the generated files -----------------------------------------------------------------------------------------

def _generator():
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
    import make_full22_configs as gen  # noqa: E402
    return gen


def test_no_generated_file_says_the_lens_serves_every_lora_cell():
    """The generator's own outputs, its template, and every file the downstream generators derive from those outputs
    (the math500 / gsm8k / ifeval overlay manifests, the misalign and alphasteer manifests; the configs and everything
    derived from them are byte-untouched — see the module docstring) — found by their `serving_launch_flags:` key or
    their `GENERATED by` header, not by a list."""
    gen = _generator()
    roster = gen.load_roster()
    outs = [dst for dst, *_ in gen.outputs(roster)]
    assert len(outs) >= 70 and all(p.is_file() for p in outs)
    offenders = [str(p.relative_to(REPO)) for p in outs if LENS_SENTENCE in p.read_text(encoding="utf-8")]
    assert not offenders, offenders
    assert LENS_SENTENCE not in GENERATOR.read_text(encoding="utf-8")
    derived = [p for p in (REPO / "configs").rglob("*.yaml")
               if "serving_launch_flags:" in p.read_text(encoding="utf-8") or "GENERATED by" in p.read_text(encoding="utf-8")]
    assert len(derived) > len(outs)
    offenders = [str(p.relative_to(REPO)) for p in derived if LENS_SENTENCE in p.read_text(encoding="utf-8")]
    assert not offenders, offenders
    # the manifest comment now names both engines and the recorded drop
    man = next(dst for dst, kind, *_ in gen.outputs(roster) if kind == "manifest").read_text(encoding="utf-8")
    assert "manifest_flags_dropped" in man and "--serving-path lora_native" in man and "--serving-path lens" in man
    assert "`none__lora0` on a native standup, `none__lens0` on a lens one" in man
