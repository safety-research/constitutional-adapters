"""The misalign manifests' and the gsm8k overlays' Petri fallback chains are their OWN -- never the seed
views' roster knobs.

The misalign manifest for a seed view is generated from that view's Petri manifest text (make_misalign_manifest.render_manifest:
byte-for-byte except the cells / audit / namespace). The Petri s1/s2/s3 manifests render the battery's per-view chains
(`views.<view>.petri_auditor_fallback_models` / `petri_judge_fallback_models`: opus-4-8 before grok, both roles), so without a
rule the knobs leaked into the misalign render and `--check` reported DRIFT against the committed manifests -- whose canonical
misalign rows must not re-key. Pinned here: the generator's render for s1-s3 carries the committed one-leg auditor chain and
NO judge chain, byte-identical to the committed file, while the Petri s1 manifest it is generated from carries the two-leg
chains for both roles; the scoping helper refuses a source without exactly one auditor chain block.

The u0 misalign manifest drifted when the u0 view gained the PER-REQUEST refusal-fallback knobs
(`auditor_refusal_fallback_models` / `judge_refusal_fallback_models`) -- the helper scoped only the two sticky
chains, so the knobs leaked. The rule is now PATTERN-based in code and here: the misalign render's petri block carries exactly one
`*_fallback_models` key (the misalign auditor chain) and no Petri-role key beyond the set the render is allowed to carry, over the
generator's renders AND every committed misalign manifest; every committed misalign manifest stays byte-stable (`--check`)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sisteer.capeval import manifest as cm  # noqa: E402

GROK = "openrouter/x-ai/grok-4.6"
#: the Petri-role keys the misalign render is ALLOWED to carry in its petri block (option (A)): the roles themselves and the one
#: misalign chain -- any other `auditor_*` / `judge_*` / `scenarios_*` key, and any other `*_fallback_models` key, is a leaked knob
ALLOWED_ROLE_KEYS = {"auditor_model", "auditor_fallback_models", "judge_model", "scenarios_model"}
MISALIGN_MANIFESTS = sorted((REPO / "configs" / "capeval" / "manifests").glob("misalign_full22_*.yaml"))


def _leaked_knobs(petri_block: dict) -> list[str]:
    """The keys of a misalign petri block the pattern rule rejects: a `*_fallback_models` other than the misalign chain, or a
    Petri-role key (auditor_* / judge_* / scenarios_*) outside ALLOWED_ROLE_KEYS."""
    return sorted(k for k in petri_block
                  if (k.endswith("_fallback_models") and k != "auditor_fallback_models")
                  or (k.startswith(("auditor_", "judge_", "scenarios_")) and k not in ALLOWED_ROLE_KEYS))
OPUS48 = "anthropic/claude-opus-4-8"


def _gen():
    spec = importlib.util.spec_from_file_location("make_misalign_manifest", REPO / "scripts/sdfbase/full22/misalign/make_misalign_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("view", ["s1", "s2", "s3"])
def test_misalign_render_carries_the_misalign_chains_not_the_petri_knobs(view):
    mm = _gen()
    text, _notes = mm.render_manifest(f"qwen35_4b_{view}")
    committed = (REPO / f"configs/capeval/manifests/misalign_full22_qwen35_4b_{view}.yaml").read_text(encoding="utf-8")
    assert text == committed, "the committed misalign manifest is authoritative (D129) and the generator renders it byte for byte"
    got = yaml.safe_load(text)["petri"]
    assert got["auditor_fallback_models"] == mm.MISALIGN_AUDITOR_CHAIN == [GROK]
    assert "judge_fallback_models" not in got
    spec = cm.load_petri_spec(REPO / f"configs/capeval/manifests/misalign_full22_qwen35_4b_{view}.yaml")
    assert spec.auditor_fallback_models == [GROK] and spec.judge_fallback_models == []
    assert _leaked_knobs(got) == [] and spec.auditor_refusal_fallback_models == [] and spec.judge_refusal_fallback_models == []   # option (A): pattern rule
    # ... while the Petri manifest it is generated FROM spells the battery's two-leg chains for both roles
    petri = cm.load_petri_spec(REPO / f"configs/capeval/manifests/petri_full22_qwen35_4b_{view}.yaml")
    assert petri.auditor_fallback_models == petri.judge_fallback_models == [OPUS48, GROK]
    src = (REPO / f"configs/capeval/manifests/petri_full22_qwen35_4b_{view}.yaml").read_text(encoding="utf-8")
    assert "judge_fallback_models:" in src and OPUS48 in src and OPUS48 not in text.split("  scenarios_model:")[0].split("  auditor_model:")[1]


def test_scoping_helper_rewrites_both_blocks_and_refuses_a_source_without_one_auditor_chain():
    mm = _gen()
    src = ("petri:\n  auditor_model: a\n  auditor_fallback_models:   # knob\n    - x/one\n    - y/two\n"
           "  judge_fallback_models:   # knob\n    - x/one\n  judge_model: j\n")
    out = mm.scope_fallback_chains(src)
    assert out == "petri:\n  auditor_model: a\n  auditor_fallback_models:\n    - openrouter/x-ai/grok-4.6\n  judge_model: j\n"
    assert mm.scope_fallback_chains(out) == out, "idempotent on an already-scoped text"
    with pytest.raises(SystemExit, match="exactly one"):
        mm.scope_fallback_chains(src.replace("  auditor_fallback_models:", "  other:"))
    # option (A): the per-request refusal-fallback knobs (and any future `*_fallback_models` knob) are stripped too, by pattern; idempotent
    knobs = ("petri:\n  auditor_model: a\n  auditor_fallback_models:   # knob\n    - x/one\n  judge_model: j\n"
             "  auditor_refusal_fallback_models:  # D354\n    - anthropic/claude-sonnet-4-6\n"
             "  judge_refusal_fallback_models:  # D354/D869\n    - anthropic/claude-opus-4-8\n    - openrouter/x-ai/grok-4.6-20260810\n"
             "  future_fallback_models:\n    - z/next\n  scenarios_model: s\n")
    out2 = mm.scope_fallback_chains(knobs)
    assert out2 == "petri:\n  auditor_model: a\n  auditor_fallback_models:\n    - openrouter/x-ai/grok-4.6\n  judge_model: j\n  scenarios_model: s\n"
    assert mm.scope_fallback_chains(out2) == out2


def test_every_committed_misalign_manifest_carries_only_the_misalign_chain_and_the_helper_strips_every_real_petri_source():
    """Option (A), pattern-based over the real files: every committed misalign manifest's petri block has exactly one
    `*_fallback_models` key (the misalign chain, MISALIGN_AUDITOR_CHAIN) and no Petri-role key beyond ALLOWED_ROLE_KEYS; and the
    scoping helper applied to EVERY committed Petri manifest text (the sources the renders copy) yields a block the same rule
    accepts -- so the next knob a seed view gains cannot leak into any misalign render. Byte-stability of the committed misalign
    manifests against their generator is the misalign `--check` in test_the_five_generator_checks_are_green."""
    mm = _gen()
    assert MISALIGN_MANIFESTS, "no committed misalign manifests found"
    for p in MISALIGN_MANIFESTS:
        got = yaml.safe_load(p.read_text(encoding="utf-8"))["petri"]
        assert _leaked_knobs(got) == [], (p.name, _leaked_knobs(got))
        assert got["auditor_fallback_models"] == mm.MISALIGN_AUDITOR_CHAIN, p.name
    sources = sorted((REPO / "configs" / "capeval" / "manifests").glob("petri_full22_*.yaml"))
    assert sources
    leaked_before = 0
    for p in sources:
        text = p.read_text(encoding="utf-8")
        if "refusal_fallback_models" in text or "judge_fallback_models" in text:
            leaked_before += 1
        scoped = yaml.safe_load(mm.scope_fallback_chains(text))["petri"]
        assert _leaked_knobs(scoped) == [], (p.name, _leaked_knobs(scoped))
        assert scoped["auditor_fallback_models"] == mm.MISALIGN_AUDITOR_CHAIN, p.name
    assert leaked_before >= 1, "no committed Petri source carries a knob the helper must strip -- the rule is untested here"


# --- the gsm8k overlays (fk2e): the same rule, the second generator that copies a campaign manifest's text --------------------

def _gsm8k_gen():
    spec = importlib.util.spec_from_file_location("make_gsm8k_overlays", REPO / "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("view", ["s1", "s2", "s3"])
def test_gsm8k_overlay_carries_the_campaign_chains_not_the_petri_knobs(view):
    """One gsm8k overlay per seed view (the `none` anchor overlay): the generator's campaign text for the view carries the
    campaign template's one-leg auditor chain and NO judge chain, the committed overlay is byte-identical to that text
    outside its header/cells, and the Petri campaign manifest it is cut from spells the two-leg chains."""
    g = _gsm8k_gen()
    tag = f"qwen35_4b_{view}"
    scoped = g.campaign_manifest_text(REPO, tag)
    got = yaml.safe_load(scoped)["petri"]
    assert got["auditor_fallback_models"] == g.CAMPAIGN_AUDITOR_CHAIN == [GROK] and "judge_fallback_models" not in got
    overlay = REPO / f"configs/capeval/manifests/petri_full22_{tag}.gsm8k_none.yaml"
    text = overlay.read_text(encoding="utf-8")
    spec = cm.load_petri_spec(overlay)
    assert spec.auditor_fallback_models == [GROK] and spec.judge_fallback_models == []
    assert spec.cells[0] == "none"          # the cells block is the ONE thing an overlay changes (the anchor + its companions)
    # byte identity outside the overlay's header + cells: the scoped campaign text from `scenario_bundle_hash:` on equals the overlay's
    tail = lambda s: s[s.index("  scenario_bundle_hash:"):]
    assert tail(text) == tail(scoped)
    petri = cm.load_petri_spec(REPO / f"configs/capeval/manifests/petri_full22_{tag}.yaml")
    assert petri.auditor_fallback_models == petri.judge_fallback_models == [OPUS48, GROK]
    assert OPUS48 not in tail(text).split("  scenarios_model:")[0]


def test_gsm8k_scoping_helper_rewrites_both_blocks_and_refuses_a_source_without_one_auditor_chain():
    g = _gsm8k_gen()
    src = ("petri:\n  auditor_model: a\n  auditor_fallback_models:   # knob\n    - x/one\n    - y/two\n"
           "  judge_fallback_models:   # knob\n    - x/one\n  judge_model: j\n")
    out = g.scope_fallback_chains(src)
    assert out == "petri:\n  auditor_model: a\n  auditor_fallback_models:\n    - openrouter/x-ai/grok-4.6\n  judge_model: j\n"
    assert g.scope_fallback_chains(out) == out
    with pytest.raises(SystemExit, match="exactly one"):
        g.scope_fallback_chains(src.replace("  auditor_fallback_models:", "  other:"))


# --- the FIVE generators' --check, green on this tree ------------------------------------------------------------------------

FIVE_CHECKS = [
    ("misalign", ["scripts/sdfbase/full22/misalign/make_misalign_manifest.py", "--check"]),
    ("bcbw", ["scripts/bcbw_configs.py", "--check"]),
    ("full22", ["scripts/sdfbase/make_full22_configs.py", "--check"]),
    ("arditi", ["scripts/sdfbase/seedrep/render_arditi_seedrep_configs.py", "--check"]),
    ("gsm8k overlays", ["scripts/sdfbase/gsm8k/make_gsm8k_overlays.py", "--matrix", "qwen_seedrep", "--check"]),
]


@pytest.mark.parametrize("name,argv", FIVE_CHECKS, ids=[n for n, _ in FIVE_CHECKS])
def test_the_five_generator_checks_are_green(name, argv):
    """Acceptance: every generator that renders a committed config reproduces it byte for byte on this tree -- the
    seed views' roster knobs reach the four Petri plan manifests and nothing else."""
    import os
    import subprocess
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{REPO / 'scripts'}")
    r = subprocess.run([sys.executable, *argv], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "DRIFT" not in r.stdout + r.stderr, (name, (r.stdout + r.stderr)[-1500:])
