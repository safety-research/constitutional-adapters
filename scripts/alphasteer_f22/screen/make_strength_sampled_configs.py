#!/usr/bin/env python
"""alphasteer-full22 STRENGTH FRONTIER — the SAMPLED variants of the AlphaSteer strength views' serving configs, generated
the way scripts/sdfbase/full22/sampled/make_sampled_configs.py generates the full22 views' (that module is IMPORTED and its
block assembly, rendering and data-verification are reused; nothing is respelled here), for the screen scope of
as22_screen (three paper targets x five strengths; glm46vflash refused).

Per (target, view, trial T in 0,1) three files, exactly the wave's set per TAG:
  configs/alphasteer_f22_serve_<target>_<view>_sampled_t<T>.yaml            base cap (static 4096 / controls 512)
  configs/alphasteer_f22_serve_<target>_<view>_sampled_t<T>_cap8192.yaml    the static 8192 rung, chained from the base
  configs/alphasteer_f22_serve_<target>_<view>_sampled_t<T>_cap16384.yaml   the static 16384 rung, chained from the 8192 rung
The screen (the controls phase) reads the base config; the wave's job-side belt (`common_sampled.sh f22s_check_config`,
run UNCHANGED by the wrapper) requires all three to exist and to chain, so all three are written.

Each output is the strength view's serving config (`configs/alphasteer_f22_serve_<target>_<view>.yaml`, the AlphaSteer
lane's, itself the campaign's config + [none, alphasteer]) as TEXT with exactly the wave generator's edits:
  * a header naming this generator and the sha256 of the source config's bytes;
  * `out_dir` / `private_run` -> `alphasteer_f22_<target>_<view>_sampled_t<T>[_cap<rung>]` (as22_screen.sampled_name);
  * the top-level `sampled_decode` block APPENDED at the end — make_sampled_configs.expected_block for the served MODEL's
    preset (a view inherits its target's preset: the sampler is the model's, not the view's), with ONE field respelled:
    a rung's `escalated_from.private_run` names THIS lane's previous rung (the wave's names `sdfbase_full22_<key>_…`);
  * a rung additionally `attacks.static_max_new_tokens: <rung>` and `sampled_decode.static_cap: <rung>`.
Everything else is byte-for-byte the source's, so the two are equal AS DATA except those keys — verified by the wave
generator's own `_verify` before anything is written (the same refusals: a drifted block, an unrenamed store, a key that
moved, static_cap != attacks.static_max_new_tokens, controls_cap != attacks.target_max_new_tokens).

CONSEQUENCE FOR COMPARABILITY: a strength view's sampled config differs from a LoRA dose view's sampled config of the
same target AS DATA in exactly {defenses_dir, explicit_cells, out_dir, private_run} — the same four keys the two serving
configs differ in — and in nothing the drivers read for decoding (preset, seed_base, trial, caps, readouts identical).
tests/test_alphasteer_f22_screen.py pins this against the wave's generated dose config where the tree carries one.

The Petri manifest is NOT duplicated: the wrapper serves the sampled config with the strength view's own manifest
(`configs/capeval/manifests/alphasteer_f22_<target>_<view>.yaml`), as the wave pairs its sampled config with the wave manifest.

Usage:
  python scripts/alphasteer_f22/screen/make_strength_sampled_configs.py            # write the 90 files
  python scripts/alphasteer_f22/screen/make_strength_sampled_configs.py --check    # exit 1 on any drift
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (str(_HERE), str(_HERE.parent), str(REPO / "scripts" / "sdfbase" / "full22" / "sampled")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_sampled_configs as msc  # noqa: E402  (the wave's generator @ f38d055d, byte-identical on this tree)
from as22_screen import (REPO as _REPO, SCREEN_TARGETS, SCREEN_VIEWS, ScreenRefusal, check_scope, load_roster,  # noqa: E402
                         sampled_config_rel, sampled_name, tag_of)
from as22_strength import serve_config_rel, strength_of_view  # noqa: E402

assert _REPO == REPO
GENERATOR = "scripts/alphasteer_f22/screen/make_strength_sampled_configs.py"
WAVE_GENERATOR = msc.GENERATOR
#: the wave generator's rungs / caps, read from it (never respelled): base 4096 -> 8192 -> 16384; controls 512
STATIC_RUNGS = msc.STATIC_RUNGS
STATIC_CAP_BASE = msc.STATIC_CAP_BASE


def check_pins(presets: dict, roster: dict) -> None:
    """The preset the block carries was read from the SNAPSHOT the strength roster serves (the wave's check_pins, against
    OUR roster; the wave's own reads the full22 roster and its SAMPLED_VIEWS, neither of which is this scope)."""
    for key in SCREEN_TARGETS:
        p = presets["presets"].get(key)
        if p is None:
            raise ScreenRefusal(f"{msc.PRESETS.name} has no preset for screen target {key}")
        t = roster["targets"].get(key)
        if t is None:
            raise ScreenRefusal(f"the strength roster names no target {key}")
        if (p["model"], p["revision"]) != (t["model"], t["model_revision"]):
            raise ScreenRefusal(f"preset {key} was read from {p['model']}@{p['revision']} but the strength roster serves "
                                f"{t['model']}@{t['model_revision']}")
        for view in SCREEN_VIEWS[key]:
            if view not in (t.get("views") or {}):
                raise ScreenRefusal(f"the strength roster has no view {key}/{view}")


def expected_block(presets: dict, target: str, view: str, trial: int, cap=None) -> dict:
    """The wave's block for the served MODEL (make_sampled_configs.expected_block on the model's main-view TAG, which IS the
    bare key), with the escalation chain renamed to this lane's rungs — the ONE deviation, and only on a rung."""
    esc = False if cap is None else int(cap)
    block = msc.expected_block(presets, target, trial, esc)
    if esc:
        prev_cap, prev_esc = msc.previous_rung(esc)
        block["escalated_from"] = {"kind": "static", "cap": prev_cap,
                                   "private_run": sampled_name(target, view, trial, None if prev_esc is False else prev_cap)}
    return block


def render(presets: dict, target: str, view: str, trial: int, cap=None) -> str:
    check_scope(target, view)   # the fifteen paper views only: a descoped target (glm46vflash) or a foreign view refuses here too
    src_path = REPO / serve_config_rel(target, strength_of_view(view))
    src_bytes = src_path.read_bytes()
    src_sha = hashlib.sha256(src_bytes).hexdigest()
    text = src_bytes.decode("utf-8")
    tag = tag_of(target, view)
    name = sampled_name(target, view, trial, cap)
    text = msc._sub_line(text, rf"^out_dir: out/alphasteer_f22_{msc.re.escape(tag)}$", f"out_dir: out/{name}")
    text = msc._sub_line(text, rf"^private_run: alphasteer_f22_{msc.re.escape(tag)}$", f"private_run: {name}")
    if cap:
        text = msc._sub_line(text, rf"^  static_max_new_tokens: {STATIC_CAP_BASE}$", f"  static_max_new_tokens: {int(cap)}")
    block = expected_block(presets, target, view, trial, cap)
    p = presets["presets"][target]
    extra = ""
    if cap:
        prev_cap, prev_esc = msc.previous_rung(int(cap))
        prev_name = sampled_name(target, view, trial, None if prev_esc is False else prev_cap)
        prev_what = "the base run" if prev_esc is False else f"the {prev_cap} rung"
        extra = (f"#   * `attacks.static_max_new_tokens` {prev_cap} -> {int(cap)} and `sampled_decode.static_cap` "
                 f"{int(cap)}: the static {int(cap)} RUNG of trial {trial}, a SEPARATE store\n"
                 f"#     (`sampled_decode.escalated_from` names {prev_what} {prev_name}; only its at-cap rows are\n"
                 f"#     re-drawn, DESIGN.md 'The primary comparison is SAME CAP vs SAME CAP').\n")
    header = (f"# {name}.yaml — STRENGTH FRONTIER (paper Figure 5), AlphaSteer lane: the SAMPLED variant of strength view\n"
              f"# `{view}` of {target}, trial {trial}" + (f" @ the static {int(cap)} rung" if cap else "") + " (the strength-frontier screen;\n"
              f"# scripts/sdfbase/full22/sampled/DESIGN.md). GENERATED by {GENERATOR} from\n"
              f"# configs/{src_path.name} (source config sha256 {src_sha}) and\n"
              f"# configs/sdfbase_sampled_decode.yaml, THE WAY {WAVE_GENERATOR} generates the full22\n"
              f"# views' (that generator is imported: same block, same validation, same refusals) — never hand-edited; `--check`\n"
              f"# refuses drift and the test pins byte-identity.\n"
              f"# The source config's text follows BYTE-FOR-BYTE (its own header included), EXCEPT:\n"
              f"#   * `out_dir` / `private_run` -> `{name}` (a NEW store; nothing greedy is touched).\n"
              f"{extra}"
              f"#   * the top-level `sampled_decode` block APPENDED at the end: {p['model']} @ {p['revision'][:8]}'s\n"
              f"#     vendor preset ({', '.join(f'{k} {v}' for k, v in p['preset'].items())}; source_kind {p['source_kind']}) — the served\n"
              f"#     MODEL's preset, identical to the one the LoRA dose views of this target decode under (a view inherits its\n"
              f"#     target's preset) — seed_base {presets['seed_base']}, trial {trial} of {presets['n_trials']}. The Petri manifest stays the\n"
              f"#     strength view's configs/capeval/manifests/alphasteer_f22_{tag}.yaml (the wrapper pairs them).\n"
              f"# Nominal strength (`base_scale`) is a LABEL in the served op's own units, not a dose comparable across object\n"
              f"# types or layer sets; the physical comparison is the measured realised dose.\n"
              f"#\n")
    tail = ("\n# -- full22 SAMPLED re-run (DESIGN.md): the vendor-recommended sampler, read by jailbreak_run.py generate\n"
            "# (--trial must equal `trial`; static_cap must equal attacks.static_max_new_tokens) and by\n"
            "# served_capability_run.py (controls_gen_config); every row records `decode` + `sampling_seed` --\n"
            + msc.render_block(block))
    out = header + text + tail
    msc._verify(out, src_bytes.decode("utf-8"), block, name, None if cap is None else int(cap))
    return out


def outputs() -> List[Tuple[Path, str, str, int, object]]:
    """[(path, target, view, trial, cap)] in generation order: target, view (ladder order), trial, base / 8192 / 16384."""
    out = []
    n_trials = int(msc.load_presets()["n_trials"])
    for target in SCREEN_TARGETS:
        for view in SCREEN_VIEWS[target]:
            for trial in range(n_trials):
                for cap in (None, *STATIC_RUNGS):
                    out.append((REPO / sampled_config_rel(target, view, trial, cap), target, view, trial, cap))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed output drifts")
    args = ap.parse_args(argv)
    presets = msc.load_presets()
    check_pins(presets, load_roster(REPO))
    rc = 0
    for dst, target, view, trial, cap in outputs():
        text = render(presets, target, view, trial, cap)
        if args.check:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != text:
                print(f"DRIFT: {dst}"); rc = 1
            else:
                print(f"ok: {dst}")
        else:
            dst.write_text(text, encoding="utf-8"); print(f"wrote {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
