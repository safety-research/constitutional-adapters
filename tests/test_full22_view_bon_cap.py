"""tests/test_full22_view_bon_cap.py — the PER-VIEW BoN cap of the full22 generator (the Qwen base seed-replicate campaign,
2026-09-15).

The roster grammar gained an optional ``views.<view>.caps: {bon: N}``: a view of a target WITHOUT a caps table writes its own
``bon.max_new_tokens`` (the NATIVE best-of-75 pass at N — no 512 store, no climb; the sdfbase-llama8b route) while every
other view of the target keeps the campaign's 512 store byte-for-byte. The first users are qwen35_4b's ``s1`` / ``s2`` / ``s3``
views — the batch-64 seed replicates of the base->instruct lane. Pinned here:

  1. the three ``s<s>`` configs render ``bon.max_new_tokens: 4096`` (and carry the header line that says why), deterministically
     from the roster + pins (their five ADMITTED cells staged + pinned 2026-09-17; the sixth, the distilled PSR difference, was
     dropped by decisions O20 and is on no view), and the drift guard admits them and
     refuses the same text at 512;
  2. EVERY other qwen35_4b config and manifest (``main``, ``it``, both dose ladders) is byte-identical to what the generator
     renders from the merged roster with the three views REMOVED — "before the views" is re-derived at test time, never a
     snapshot: the first cut pinned sha16s taken at ``reseed`` f47715f1, and main's strength-frontier landing (#156,
     0e21d6dc) regenerated the ``it_d*`` headers and added the bare ``d*`` ladder under this target within the day;
  3. a view cap on a target that also declares ``caps.bon`` REFUSES (two authorities for one cap), on a dose view REFUSES,
     and the block's grammar refuses any key but ``bon``, a missing / non-int / ``pending`` value;
  4. the request stream: a synthetic ``bon_served_run``-style config load of the rendered ``s1`` config (``jailbreak_run._load_cfg``,
     the loader bon_served_run.py calls; ``load_check.cap_for`` reads the same key) drives ``ServedDefense`` to send
     ``max_tokens 4096`` on every best-of-N request, where the ``main`` config's load sends 512 through the same path.
"""
from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import make_full22_configs as gen  # noqa: E402
import f22_roster as fr  # noqa: E402

ROSTER = gen.load_roster()
PINS = gen.load_pins()
BODY = (REPO / "configs" / "sdfbase_transfer_qwen35_4b_r3.yaml").read_text(encoding="utf-8")
TARGET = "qwen35_4b"
SEED_VIEWS = ("s1", "s2", "s3")
PAIRED_VIEWS = ("s1p", "s2p", "s3p")   # the composed difference_relu views
HEADER_LINE = "bon.max_new_tokens 4096: the view's caps.bon (native best-of-75 pass, no 512 store, no climb — the sdfbase-llama8b route)"
#: the six DECLARED cells of every seed view, in the campaign's order (the four parents, then the derived pair LAST, as main)
SEED_CELLS = ("constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce",
              "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce",
              "constv2-mix-full__psr_ce_all_minus_ctl_distilled", "constv2-mix-full__lora_r4_ce_minus_ctl")
#: decisions O20 (2026-09-17): the distilled PSR difference failed the lane's admission for EVERY seed (RS/reseed/admission/
#: qwen35_4b_base_eb64_s<s>__constv2-mix-full__psr_ce_all_minus_ctl_distilled.json) and is DROPPED from every view — the five
#: ADMITTED cells are staged + pinned + served, no pending_fit is left
DROPPED = "constv2-mix-full__psr_ce_all_minus_ctl_distilled"
ADMITTED_CELLS = tuple(c for c in SEED_CELLS if c != DROPPED)
#: 2026-09-17: the two PROMPTED baselines every seed view lists AFTER its objects (the roster's
#: `prompted__<si>: null` grammar, the it-view order — safety, then the constitution SI last): no object, no staged id, no pin
PROMPTED_CELLS = ("prompted__safety_refusal", "prompted__si_constitution")
SERVED_CELLS = ADMITTED_CELLS + PROMPTED_CELLS
DERIVED_SUFFIXES = ("_minus_ctl", "_minus_ctl_distilled")


def _bon_at(text: str, cap: int) -> str:
    """The rendered text with the BON block's cap (not attacks.target_max_new_tokens above it) moved to `cap`."""
    head, sep, tail = text.partition("\nbon:")
    assert sep, "no bon block"
    return head + sep + tail.replace("max_new_tokens: 4096", f"max_new_tokens: {cap}", 1)


# --- 1: the three seed views -------------------------------------------------------------------------

@pytest.mark.parametrize("view", SEED_VIEWS)
def test_seed_view_declares_the_cap_and_renders_4096_without_pins(view):
    v = ROSTER["targets"][TARGET]["views"][view]
    assert v["caps"] == {"bon": 4096}
    assert gen.view_declared_bon_cap(ROSTER, TARGET, view) == 4096 == gen._view_bon_cap(ROSTER, TARGET, view)
    assert gen.view_dose(ROSTER, TARGET, view) is None and fr.target_caps(ROSTER, TARGET) is None
    text = (REPO / v["config"]).read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    assert cfg["bon"]["max_new_tokens"] == 4096
    assert cfg["bon"]["climb_cap"] == ROSTER["bon_climb_cap"] == 4096   # the wave-wide climb keys ride every view (the guard requires them)
    assert f"  # {HEADER_LINE}\n  max_new_tokens: 4096\n" in text        # written where the body's 512 was, labelled
    assert f"#   * {HEADER_LINE}" in text                                # and said in the header
    assert "the 512 draws the canonical instrument climbs FROM" not in text   # the body's climb label is false on the native route
    # deterministic from the roster + pins: the FIVE ADMITTED cells (the four parents + the LoRA difference) are staged (pinned
    # from the clusters' staging manifest, 2026-09-17) and served; the distilled PSR difference was NOT admitted for any seed
    # (decisions O20) and is DROPPED from the view — nothing is left under pending_fit, and the dropped id is nowhere on the view
    s = view[1:]
    # ... + the two PROMPTED baselines LAST: served, null staged id, never pinned
    assert "pending_fit" not in v and list(v["cells"]) == list(SERVED_CELLS)
    assert DROPPED not in v["cells"] and f"derived_base64s{s}__{DROPPED}" not in PINS[TARGET]
    assert {sid for sid in v["cells"].values() if sid is not None} <= set(PINS[TARGET])
    assert [cid for cid, sid in v["cells"].items() if sid is None] == list(PROMPTED_CELLS)
    assert gen.render_config(ROSTER, PINS, TARGET, view) == text
    assert gen.render_manifest(ROSTER, TARGET, view) == (REPO / v["manifest"]).read_text(encoding="utf-8")
    assert cfg["explicit_cells"] == ["none", *SERVED_CELLS]
    for cid in ADMITTED_CELLS:
        assert f"#       {cid:<32} <- {v['cells'][cid]}" in text
        assert f"object_sha256 {PINS[TARGET][v['cells'][cid]]['object_sha256']}" in text
        assert f"#   pending_fit      {cid:<32}" not in text
    for cid in PROMPTED_CELLS:
        assert f"#       {cid:<32} <- (prompted baseline: the SI as the system prompt; no object)" in text
    assert "#   pending_fit      (none)" in text and DROPPED not in cfg["explicit_cells"]
    man = yaml.safe_load((REPO / v["manifest"]).read_text(encoding="utf-8"))
    assert [c for c in man["petri"]["cells"] if c.startswith("prompted__")] == ["prompted__safety", "prompted__constitution"] == man["petri"]["cells"][-2:]
    # the five OBJECT cells, in main's order (parents then the derived), each staged id carrying the seed in its lane prefix and
    # the campaign grammar in its cell id
    for cid, sid in v["cells"].items():
        if cid in PROMPTED_CELLS:
            continue
        lane = "derived_base64s" if cid.endswith(DERIVED_SUFFIXES) else "base64s"
        assert sid == f"{lane}{s}__{cid}", (view, cid, sid)
    # the guard: admits the rendered text; refuses the same text at the body's 512 — the cap is the VIEW's declaration
    gen._refuse_bon_drift(TARGET, view, text, BODY, roster=ROSTER)
    with pytest.raises(SystemExit, match="the view's caps.bon 4096"):
        gen._refuse_bon_drift(TARGET, view, _bon_at(text, 512), BODY, roster=ROSTER)
    with pytest.raises(SystemExit, match="the view's caps.bon 4096"):
        gen._refuse_bon_drift(TARGET, view, _bon_at(text, 8192), BODY, roster=ROSTER)


def test_the_seed_views_are_the_only_view_caps_in_the_roster_and_are_exactly_once():
    declared = {(t, vn): gen.view_declared_bon_cap(ROSTER, t, vn)
                for t, tt in ROSTER["targets"].items() for vn in tt["views"]}
    assert {k for k, cap in declared.items() if cap is not None} == {(TARGET, vn) for vn in SEED_VIEWS}
    # the 15 staged ids (five admitted per view x 3; the three distilled ids are on NO view — decisions O20) are named exactly
    # once across the target (every_staged_id's domain: cells + pending_fit, and no pending_fit is left)
    ids = fr.every_staged_id(ROSTER, TARGET)
    ours = [i for i in ids if i.startswith(("base64s", "derived_base64s"))]
    # the PAIRED views s<s>p RE-SERVE two of the seed view's ids each — the r=4
    # constitution LoRA (the GSM8K standup companion) and the LoRA difference (ReLU's difference-vs-difference
    # read) — the way the dose views re-serve itdoc__ ids; they stage nothing of their own beyond the composed paired write.
    paired_views = {vn: v for vn, v in ROSTER["targets"][TARGET]["views"].items() if vn in PAIRED_VIEWS}
    assert set(paired_views) == set(PAIRED_VIEWS)
    reserved = [sid for v in paired_views.values() for sid in v["cells"].values() if sid.startswith(("base64s", "derived_base64s"))]
    assert len(reserved) == 6 and len(set(reserved)) == 6
    once = [i for i in ours if i not in reserved] + sorted(set(reserved))
    assert len(once) == 15 == len(set(once)) and not any(i.endswith(DROPPED) for i in once)
    assert all(ours.count(i) == 1 for i in ours if i not in reserved)   # the seed views' own ids appear once
    assert all(ours.count(i) == 2 for i in set(reserved))               # each re-served id: its seed view + its paired view
    for vn, v in paired_views.items():
        s = vn[1]
        assert v["cells"]["constv2-mix-full__psr_ce_all_minus_ctl"] == f"paired_eb64s{s}__constv2-mix-full__psr_ce_all_minus_ctl"
        assert v["cells"]["constv2-mix-full__lora_r4_ce"] == ROSTER["targets"][TARGET]["views"][f"s{s}"]["cells"]["constv2-mix-full__lora_r4_ce"]
        assert v["cells"]["constv2-mix-full__lora_r4_ce_minus_ctl"] == ROSTER["targets"][TARGET]["views"][f"s{s}"]["cells"]["constv2-mix-full__lora_r4_ce_minus_ctl"]
        assert gen.view_declared_bon_cap(ROSTER, TARGET, vn) is None   # no BoN cap / Petri knobs on a GSM8K + agentic view
    # and every other view of the target renders the body's 512 through the SAME reader
    for vn in ROSTER["targets"][TARGET]["views"]:
        if vn not in SEED_VIEWS:
            assert gen.view_declared_bon_cap(ROSTER, TARGET, vn) is None and gen._view_bon_cap(ROSTER, TARGET, vn) == 512


# --- 2: nothing else moved --------------------------------------------------------------------------

def _roster_before_the_views() -> dict:
    """The merged roster with the three seed views REMOVED: the target as it stood before the per-view cap had a user.
    Derived from the live roster at test time, never a snapshot — main keeps adding views under this target (the bare
    ``d*`` ladder landed after the first cut of this test pinned sha16s at f47715f1, and the pins were stale within the
    day), so "before" always means THIS roster minus ``s1`` / ``s2`` / ``s3``."""
    before = copy.deepcopy(ROSTER)
    for vn in SEED_VIEWS + PAIRED_VIEWS:      # a paired view is an ALIAS of its seed view (`paired_of`): it cannot outlive it
        del before["targets"][TARGET]["views"][vn]
    return before


def test_every_other_qwen_config_and_manifest_is_byte_identical_to_before_the_views():
    before = _roster_before_the_views()
    # "before" declares no view cap anywhere in the roster: the extension is inert on it, so its renders ARE the pre-cap text
    assert all(gen.view_declared_bon_cap(before, t, vn) is None for t, tt in before["targets"].items() for vn in tt["views"]), \
        "a view cap survives the seed views' removal — they are not the only per-view caps, so this roster is not 'before'"
    others = [vn for vn in ROSTER["targets"][TARGET]["views"] if vn not in SEED_VIEWS + PAIRED_VIEWS]
    assert {"main", "it"} <= set(others)                                            # the walk is not vacuous
    checked: set[str] = set()
    for vn in others:
        v = ROSTER["targets"][TARGET]["views"][vn]
        assert before["targets"][TARGET]["views"][vn] == v                          # its own declaration is untouched
        for k, render in (("config", lambda r: gen.render_config(r, PINS, TARGET, vn)),
                          ("manifest", lambda r: gen.render_manifest(r, TARGET, vn))):
            disk = (REPO / v[k]).read_text(encoding="utf-8")
            assert render(before) == disk, f"{v[k]}: the committed text is not the generator's render WITHOUT the seed views"
            assert render(ROSTER) == disk, f"{v[k]}: the committed text is not the generator's render WITH them (--check)"
            checked.add(v[k])
        # the campaign's 512 reaches every one of them on both rosters, and the cap's header line reaches none
        assert gen._view_bon_cap(before, TARGET, vn) == 512 == gen._view_bon_cap(ROSTER, TARGET, vn)
        text = (REPO / v["config"]).read_text(encoding="utf-8")
        assert yaml.safe_load(text)["bon"]["max_new_tokens"] == 512 and HEADER_LINE not in text
    # the checked set IS every qwen35_4b file but the three seed views' and their three paired aliases' — nothing rides along
    # unchecked, none of theirs is in it
    assert len(checked) == 2 * len(others)
    assert checked.isdisjoint(ROSTER["targets"][TARGET]["views"][sv][k] for sv in SEED_VIEWS + PAIRED_VIEWS for k in ("config", "manifest"))
    # a caps-table target (llama8b) and a dose view of this target still pass the drift guard unchanged
    gen._refuse_bon_drift("llama8b", "main", (REPO / "configs" / "sdfbase_full22_llama8b.yaml").read_text(encoding="utf-8"), BODY, roster=ROSTER)
    gen._refuse_bon_drift(TARGET, "it_d0p25", (REPO / "configs" / "sdfbase_full22_qwen35_4b_it_d0p25.yaml").read_text(encoding="utf-8"), BODY, roster=ROSTER)


# --- 3: the refusals --------------------------------------------------------------------------------

def test_a_view_cap_on_a_caps_table_target_refuses():
    r = copy.deepcopy(ROSTER)
    assert fr.target_caps(r, "llama8b")["bon"] == 4096
    r["targets"]["llama8b"]["views"]["main"]["caps"] = {"bon": 4096}     # even the SAME value: two authorities for one cap
    with pytest.raises(SystemExit, match="caps table whose caps.bon is WRITTEN for every view"):
        gen.view_declared_bon_cap(r, "llama8b", "main")
    with pytest.raises(SystemExit, match="caps table whose caps.bon is WRITTEN for every view"):
        gen.render_config(r, PINS, "llama8b", "main")
    text = (REPO / "configs" / "sdfbase_full22_llama8b.yaml").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="caps table whose caps.bon is WRITTEN for every view"):
        gen._refuse_bon_drift("llama8b", "main", text, BODY, roster=r)   # the guard refuses it too, whatever the text says


@pytest.mark.parametrize("view", ["it_d1", "it_d0p25", "it_dm1"])
def test_a_view_cap_on_a_dose_view_refuses(view):
    r = copy.deepcopy(ROSTER)
    assert gen.view_dose(r, TARGET, view) is not None
    r["targets"][TARGET]["views"][view]["caps"] = {"bon": 4096}
    with pytest.raises(SystemExit, match="a dose view moves the OBJECT, never a cap"):
        gen.view_declared_bon_cap(r, TARGET, view)
    with pytest.raises(SystemExit, match="a dose view moves the OBJECT, never a cap"):
        gen.render_config(r, PINS, TARGET, view)
    text = (REPO / ROSTER["targets"][TARGET]["views"][view]["config"]).read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="a dose view moves the OBJECT, never a cap"):
        gen._refuse_bon_drift(TARGET, view, text, BODY, roster=r)
    # a dose-NAMED view is a dose view to this grammar even before the dose key is checked (the name is the spelling rule)
    r2 = copy.deepcopy(ROSTER)
    r2["targets"][TARGET]["views"]["it_d2"] = {"config": "x.yaml", "manifest": "y.yaml", "caps": {"bon": 4096}, "cells": {}}
    with pytest.raises(SystemExit, match="a dose view moves the OBJECT, never a cap"):
        gen.view_declared_bon_cap(r2, TARGET, "it_d2")


@pytest.mark.parametrize("caps,match", [
    ({"static": 4096}, "unknown keys"),
    ({"bon": 4096, "petri": 1024}, "unknown keys"),
    ({}, "names no cap"),
    ({"bon": "pending"}, "positive int"),
    ({"bon": 0}, "positive int"),
    ({"bon": True}, "positive int"),
    ("4096", "must be a mapping"),
    ([4096], "must be a mapping"),
])
def test_the_view_caps_grammar_refuses_anything_but_a_positive_bon(caps, match):
    r = copy.deepcopy(ROSTER)
    r["targets"][TARGET]["views"]["s1"]["caps"] = caps
    with pytest.raises(SystemExit, match=match):
        gen.view_declared_bon_cap(r, TARGET, "s1")
    with pytest.raises(SystemExit, match=match):
        gen.render_config(r, PINS, TARGET, "s1")


def test_a_view_without_the_block_is_the_campaigns_512_and_a_moved_cap_still_refuses():
    r = copy.deepcopy(ROSTER)
    del r["targets"][TARGET]["views"]["s1"]["caps"]
    assert gen.view_declared_bon_cap(r, TARGET, "s1") is None and gen._view_bon_cap(r, TARGET, "s1") == 512
    text = gen.render_config(r, PINS, TARGET, "s1")
    assert yaml.safe_load(text)["bon"]["max_new_tokens"] == 512 and HEADER_LINE not in text
    committed = (REPO / "configs" / "sdfbase_full22_qwen35_4b_s1.yaml").read_text(encoding="utf-8")
    # the committed 4,096 text against a roster with no view cap: the block differs from the body's (the 4,096 is nobody's
    # declaration now), so the guard refuses it as drift — the cap is admitted ONLY through the view's `caps.bon`
    with pytest.raises(SystemExit, match="bon block drifted from the body|is not 512"):
        gen._refuse_bon_drift(TARGET, "s1", committed, BODY, roster=r)


# --- 4: the request stream ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.sent = []

    async def __call__(self, req):
        self.sent.append(dict(req))
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


def _load_like_bon_served_run(tmp_path: Path, rel: str) -> dict:
    """bon_served_run.py: ``cfg = jr._load_cfg(args.config)`` then ``bcfg = cfg["bon"]`` and
    ``max_new_tokens=int(bcfg["max_new_tokens"])`` on every draw — the SAME loader over a synthetic copy of the config."""
    import jailbreak_run as jr
    p = tmp_path / Path(rel).name
    p.write_text((REPO / rel).read_text(encoding="utf-8"), encoding="utf-8")
    return jr._load_cfg(str(p))


@pytest.mark.parametrize("rel,cap", [("configs/sdfbase_full22_qwen35_4b_s1.yaml", 4096),
                                     ("configs/sdfbase_full22_qwen35_4b_s2.yaml", 4096),
                                     ("configs/sdfbase_full22_qwen35_4b_s3.yaml", 4096),
                                     ("configs/sdfbase_full22_qwen35_4b.yaml", 512)])
def test_the_rendered_config_drives_the_bon_request_streams_max_tokens(tmp_path, rel, cap):
    import load_check as lc
    from sisteer.eval.attacks import bon_served as bs
    cfg = _load_like_bon_served_run(tmp_path, rel)
    bcfg = cfg["bon"]
    assert int(bcfg["max_new_tokens"]) == cap and bcfg["n_max"] == 75
    got, src = lc.cap_for("bon", cfg, None)              # the BoN standup's load check reads the same key
    assert got == cap and "bon" in src
    rec = _Recorder()
    sd = bs.ServedDefense(rec, virtual_model="petri__si_constv2_mix_full__lora_r4_ce", cell_id="constv2-mix-full__lora_r4_ce",
                          serving_engine="lora_native")
    # the sampler is the caller's to name since the 2026-09-17 ruling (main #216: no model-independent default temperature);
    # this test is about the CAP the rendered config drives, so it names the grandfathered T=1 row BY LABEL — the wire shape
    # the assertion below pins ({temperature: 1.0} alone) — exactly as tests/test_bon_decode_seed_policy.py does
    decode = bs.resolve_bon_sampler(None, bs.BON_HARNESS_DEFAULT).decode
    turns = [asyncio.run(sd.generate_one_detailed(f"variant {i}", sampling_seed=1000 + i, max_new_tokens=int(bcfg["max_new_tokens"]),
                                                  decode=decode))
             for i in range(3)]
    assert [r["max_tokens"] for r in rec.sent] == [cap] * 3
    assert all(t.max_tokens == cap and t.max_tokens_clamped_from is None for t in turns)
    assert all(r["temperature"] == 1.0 and set(r) == {"model", "messages", "temperature", "max_tokens", "seed"} for r in rec.sent)
