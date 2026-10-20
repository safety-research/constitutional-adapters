#!/usr/bin/env python
"""alphasteer-full22 STRENGTH FRONTIER — the serving CONFIGS and Petri MANIFESTS of the AlphaSteer strength
views, generated from the CAMPAIGN's committed files; `--check` requires byte-identity (like
make_alphasteer_configs.py). Never hand-edit the outputs.

Per (target, strength) — as22_strength names the view (`as_dm0p5 as_d0p1 as_d0p3 as_d0p5 as_d1`), the run tag
(`alphasteer_f22_<target>_<view>`) and both paths:

  configs/alphasteer_f22_serve_<target>_<view>.yaml
      DERIVED from configs/alphasteer_f22_serve_<target>.yaml (the campaign's serving config for this target —
      itself the wave's config for the target's `it`/`main` view + [none, alphasteer]) byte-for-byte as data,
      EXCEPT exactly three keys and the header:
        * `defenses_dir`  = out/alphasteer_f22/<target>/strength/cells_view/<view>   (the lane's OWN bank; on
                            the cluster the job's TT_TWO_SI_DEFENSES_DIR export resolves it)
        * `out_dir`       = out/alphasteer_f22_<target>_<view>
        * `private_run`   = alphasteer_f22_<target>_<view>
      `bon.max_new_tokens` is WRITTEN (as22_strength.BON_MAX_NEW_TOKENS, through a guarded substitution) and
      equals the campaign's 512: BoN is the 512 STORE the campaign's 512 -> 4,096 climb tooling then escalates
      (superseding the native 4,096 — one best-of-N estimand across the frontier's three method lanes; each
      strength lane then needs a row in the climb's lanes manifest).
      `explicit_cells` stays [none, alphasteer]; `attacks.static_max_new_tokens` (4096), `attacks.target_max_new_tokens`
      (the roster's wave-wide value — 4,096 since main's 2026-09-11 multi-turn ruling) and `attacks.controls_max_new_tokens`
      (the judged controls' own cap, 512) stay EXACTLY the campaign config's this view is rendered from: a strength view moves
      the object, never a cap. The generator re-reads its own render and refuses if any of those facts does not hold.

  configs/capeval/manifests/alphasteer_f22_<target>_<view>.yaml
      DERIVED from configs/capeval/manifests/alphasteer_f22_<target>.yaml (the campaign's PRODUCTION manifest)
      with EXACTLY one key changed — `audit` (alphasteer_f22_<target>_<view>) — plus the header. `cells` stays
      [none, alphasteer]; EVERY serving pin is byte-identical, INCLUDING `serving_impl_commit` (the wave's
      94e294c4… line, comment and all — asserted equal to the strength roster's, never computed).

A glm46vflash config / manifest IS generated and IS meant to serve: glm has a lambda=1 parent fit (the campaign's
`no_admissible_operating_point` is a statement about its rho-grid LADDER SELECTION), and the strength frontier
runs its ladder at the low end where the stop rule decides.

usage:
  python scripts/alphasteer_f22/make_strength_configs.py            # write the 20 configs + 20 manifests
  python scripts/alphasteer_f22/make_strength_configs.py --check    # exit 1 on any drift
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from as22_strength import (BON_MAX_NEW_TOKENS, DEFAULT_LANE, LANES, STATIC_MAX_NEW_TOKENS, STRENGTHS,  # noqa: E402
                           TARGET_MAX_NEW_TOKENS, Lane, defenses_dir_placeholder, lane as lane_of, lane_targets,
                           layer_gaps, load_roster, manifest_rel, require_target, roster_parent_layers, run_tag,
                           serve_config_rel, view_name)
from as22_targets import CELL_ID, LADDER_CELLS, PRODUCTION_SERVING_IMPL_COMMIT  # noqa: E402
from make_ladder_configs import _BODY_MARKER, _EXPLICIT_MARKER, _sub_line, _value  # noqa: E402

GENERATOR = "scripts/alphasteer_f22/make_strength_configs.py"


def roster_serving_impl_commit(roster: dict, where: str) -> str:
    v = roster.get("serving_impl_commit")
    if not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{40}", v):
        raise SystemExit(f"{where}: serving_impl_commit {v!r} is not a full 40-hex sha")
    return v


def _template(tinfo: dict, key: str, target: str, ln: Lane) -> str:
    rel = tinfo.get(key)
    if not isinstance(rel, str) or not (REPO / rel).is_file():
        raise SystemExit(f"lane {ln.name} / {target}: roster `{key}` {rel!r} is not an existing file — a target's TEMPLATE "
                         "config / manifest is named in its roster entry (the campaign's for the four campaign targets)")
    return rel


def _layers_sentence(tinfo: dict) -> str:
    pl = roster_parent_layers(tinfo)
    if pl is None:
        return "the object's as fit"
    gaps = layer_gaps(pl)
    span = f"{pl[0]}-{pl[-1]}" + (f" skipping {gaps}" if gaps else "")
    return f"{len(pl)} layers, {span} ({'NON-CONTIGUOUS' if gaps else 'contiguous'}; the roster's parent_layers, asserted against the blob at bake time)"


def _bon_block_span(body: str) -> Tuple[int, int]:
    """(start, end) offsets of the top-level `bon:` block in a rendered config body."""
    m = re.search(r"^bon:\n", body, flags=re.M)
    if not m:
        raise SystemExit("config body has no top-level bon: block")
    nxt = re.compile(r"^[A-Za-z_][^\n]*:.*$", re.M).search(body, m.end())
    return m.start(), (nxt.start() if nxt else len(body))


# --------------------------------------------------------------------------- configs

def render_config(target: str, s: float, ln: Lane = None, *, roster: dict = None) -> str:
    ln = ln or lane_of(DEFAULT_LANE)
    roster = roster or load_roster(ln)
    tinfo = require_target(ln, target, roster)
    view = view_name(s)
    tag = run_tag(target, s, ln)
    src_rel = _template(tinfo, "campaign_config", target, ln)
    src = (REPO / src_rel).read_text(encoding="utf-8")
    if _EXPLICIT_MARKER not in src or _BODY_MARKER not in src:
        raise SystemExit(f"{src_rel} is not a generated full22-style serving config (markers missing)")
    # the explicit cells block is OURS ([none, alphasteer]), whatever the template listed; the body is the template's
    body = src[src.index(_BODY_MARKER):]
    body = re.sub(r"^# -- everything below: .*$",
                  f"# -- everything below: {Path(src_rel).name}'s data, unchanged, EXCEPT the keys the header names --",
                  body, count=1, flags=re.M)
    body = f"{_EXPLICIT_MARKER} (ordered; the registry is EXACTLY this) ---\nexplicit_cells:\n" + "".join(f"- {c}\n" for c in LADDER_CELLS) + "\n" + body
    body = _sub_line(body, r"^defenses_dir: .*$", f"defenses_dir: {defenses_dir_placeholder(target, s, ln)}")
    body = _sub_line(body, r"^out_dir: .*$", f"out_dir: out/{tag}")
    body = _sub_line(body, r"^private_run: .*$", f"private_run: {tag}")
    # the BoN cap: the ONE `  max_new_tokens:` line, and it must sit inside the bon: block
    b0, b1 = _bon_block_span(body)
    hits = [m for m in re.finditer(r"^  max_new_tokens: (\S+)$", body, flags=re.M)]
    if len(hits) != 1 or not (b0 <= hits[0].start() < b1):
        raise SystemExit(f"{src_rel}: expected exactly one `  max_new_tokens:` line, inside bon:, found {len(hits)}")
    body = body[:hits[0].start()] + f"  max_new_tokens: {BON_MAX_NEW_TOKENS}" + body[hits[0].end():]
    model, rev = _value(body, "model"), _value(body, "model_revision")
    cfg = yaml.safe_load(body)
    # The caps are HELD TO THE CAMPAIGN SOURCE, not to constants: main moved the multi-turn target cap to the roster's
    # wave-wide 4,096 and gave the judged controls their own `controls_max_new_tokens`, so "a strength view moves no cap"
    # is asserted against the campaign config this view is rendered from. The BoN cap is the one WRITTEN key and must
    # still equal the campaign's (the 512 store the climb tooling escalates).
    src_cfg = yaml.safe_load(src[src.index(_BODY_MARKER):])
    src_atk = src_cfg["attacks"]
    if src_cfg["bon"]["max_new_tokens"] != BON_MAX_NEW_TOKENS:
        raise SystemExit(f"{src_rel}: the campaign's bon.max_new_tokens is {src_cfg['bon']['max_new_tokens']!r}, not the "
                         f"{BON_MAX_NEW_TOKENS} store this lane writes (D48) — the campaign moved; restate the lane before regenerating")
    facts = {"explicit_cells": cfg.get("explicit_cells") == list(LADDER_CELLS),
             "bon.max_new_tokens": cfg["bon"]["max_new_tokens"] == BON_MAX_NEW_TOKENS,
             "attacks.static_max_new_tokens": cfg["attacks"]["static_max_new_tokens"] == src_atk["static_max_new_tokens"] == STATIC_MAX_NEW_TOKENS,
             "attacks.target_max_new_tokens": cfg["attacks"]["target_max_new_tokens"] == src_atk["target_max_new_tokens"],
             "attacks.controls_max_new_tokens": cfg["attacks"].get("controls_max_new_tokens") == src_atk.get("controls_max_new_tokens"),
             "defenses_dir": cfg["defenses_dir"] == defenses_dir_placeholder(target, s, ln),
             "private_run": cfg["private_run"] == tag and cfg["out_dir"] == f"out/{tag}",
             "model": (cfg.get("model"), cfg.get("model_revision")) == (tinfo.get("model"), tinfo.get("model_revision"))}
    bad = [k for k, ok in facts.items() if not ok]
    if bad:
        raise SystemExit(f"{tag}: the rendered config does not hold {bad}; refusing to write")
    parent_desc = ("the campaign's Phase B lambda=1 PARENT fit" if ln.parent_lineage == "campaign_lambda1"
                   else "the RE-BAKED lambda=1 PARENT fit (paper-sized, threshold-selected layer set; the parallel root "
                        f"$SHARED/{ln.bank_root_rel}, the original objects never mutated)")
    cview = tinfo.get("campaign_view")
    tmpl_desc = (f"the campaign's serving config for this target, itself the full22 wave's `{cview}` config + [none, {CELL_ID}]"
                 if cview else "this target's template serving config, named by the lane roster")
    header = f"""# {Path(serve_config_rel(target, s, ln)).name} — STRENGTH FRONTIER (paper Figure 5), AlphaSteer lane `{ln.name}`: the
# SERVING config for target {target}, strength view `{view}` (nominal base_scale {s!r}; the
# strengths are read as NOMINAL base_scale, not as multiples of any selected rung; nominal strengths are not
# comparable across object types and need not be, the realised dose is measured separately). The multi-layer
# AlphaSteer Class B cell of {parent_desc}{(', THE PARENT OBJECT ITSELF: base_scale 1.0, the cell dir symlinks its object.pt (scripts/alphasteer_f22/bake_strength.py bake_anchor), never re-saved, never re-scaled' if s == 1.0 else f', re-scaled to base_scale {s!r} by scripts/alphasteer_f22/bake_strength.py (bake_rungs.scaled_blob: matrices bit-identical, ONLY the scalar rewritten)')},
# served on the vLLM lens target under the STACKED reduction mode next to a FRESH same-standup bare anchor.
# GENERATED by {GENERATOR} --lane {ln.name} — never hand-edited; the test
# requires byte-identity. DERIVED from {src_rel} ({tmpl_desc}): every key below is that file's,
# byte-for-byte as data, EXCEPT —
#
#   * `defenses_dir` names the lane's OWN bank view, $SHARED/{ln.bank_root_rel}/{target}/strength/cells_view/{view}
#     (bake_strength.py --lane {ln.name} lays cells/{view}__{CELL_ID}/ out; scripts/sdfbase/full22/make_bank_view.py builds
#     the view with {ln.roster_rel} + {ln.pins_rel}; on the
#     cluster the job's TT_TWO_SI_DEFENSES_DIR export resolves it — the defenses-dir order). Nothing of the parent lane's
#     cells/, cells_view/, ladder/, selected__alphasteer, lambda_ladder.json, roster or pins is a write target.
#   * out_dir / private_run get the `{tag}` namespace (Tag = <lane>_<target>_<view>; the dose parses
#     from the view name: strip `as_`, drop `d`, `p` -> `.`, leading `m` -> `-`).
#   * `bon.max_new_tokens` = {BON_MAX_NEW_TOKENS}, the campaign's value (superseding a native-4096 override): BoN is the 512
#     STORE the campaign's 512 -> 4,096 climb tooling then escalates, so the frontier's three method lanes share
#     ONE best-of-N estimand (a native 4,096 pass for one method next to a climb counterfactual for another would
#     make the methods incomparable — the one comparison Figure 5 exists to make). A 512 store is NOT the final
#     reading: this view expects a row in the climb's lanes manifest (NOTES_strength.md section 9).
#     `attacks.static_max_new_tokens` ({STATIC_MAX_NEW_TOKENS}) and `attacks.target_max_new_tokens` ({TARGET_MAX_NEW_TOKENS}) are
#     UNCHANGED, so the static families and multi-turn keep the campaign's caps.
#
# `explicit_cells` is [none, {CELL_ID}]: `none` first is the same-standup bare anchor; `{CELL_ID}` is the bare
# Class B id served from <defenses_dir>/{CELL_ID}/object.pt with fit_provenance.json (the PARENT's, byte-identical)
# as its ONE identity sidecar. The path-matched no-op twin `none__lens0` is DERIVED by defense_cells.enumerate_cells.
# The LAYER SET is the parent's as fit: {_layers_sentence(tinfo)}. The bake asserts
# every copy's layer list equals the parent's element for element; this lane never re-bakes a layer set, so the
# strength ladder measures THIS layer set.
#
# Serving manifest: {manifest_rel(target, s, ln)} (same generator; its cells are
# EXACTLY these explicit_cells in the harness vocabulary — scripts/sdfbase/mt_serving_up.py refuses a mismatch).
#
# serving target: {model} @ {rev}

"""
    return header + body


# --------------------------------------------------------------------------- manifests

def render_manifest(target: str, s: float, ln: Lane = None, *, roster: dict = None) -> str:
    ln = ln or lane_of(DEFAULT_LANE)
    roster = roster or load_roster(ln)
    tinfo = require_target(ln, target, roster)
    view = view_name(s)
    tag = run_tag(target, s, ln)
    src_rel = _template(tinfo, "campaign_manifest", target, ln)
    src = (REPO / src_rel).read_text(encoding="utf-8")
    if "\npetri:\n" not in src:
        raise SystemExit(f"{src_rel} has no top-level petri: block")
    body = src[src.index("\npetri:\n") + 1:]
    body = _sub_line(body, r"^  audit: .*$", f"  audit: {tag}")
    m = re.search(r"^  cells:\n((?:    - .*\n)+)", body, flags=re.M)
    if not m:
        raise SystemExit(f"{src_rel}: cells block not found")
    body = body[:m.start()] + "  cells:\n" + "".join(f"    - {c}\n" for c in LADDER_CELLS) + body[m.end():]
    sic = re.search(r"^  serving_impl_commit: (\S+)", body, flags=re.M)
    if not sic:
        raise SystemExit(f"{src_rel}: no serving_impl_commit line")
    carried, want = sic.group(1), roster_serving_impl_commit(roster, ln.roster_rel)
    # The campaign's PRODUCTION manifests carry the loader's `UNFROZEN` literal since main's native-serving landing
    # (make_ladder_configs.LADDER_SERVING_IMPL_COMMIT: the ladder records W_L's rho-emit commit, which is W_L itself, and a
    # tree cannot carry its own sha). A derived manifest carries EXACTLY what the campaign's carries — that is the
    # byte-identity this generator promises — so UNFROZEN is accepted as-is; a frozen sha must still be the roster's AND
    # the production sha, and the serving identity is proven at the head of every job by `as22_assert_serving_identity`.
    if carried != "UNFROZEN" and (carried != want or carried != PRODUCTION_SERVING_IMPL_COMMIT):
        raise SystemExit(f"{src_rel} carries serving_impl_commit {carried} but the strength roster says {want} and the "
                         f"production sha is {PRODUCTION_SERVING_IMPL_COMMIT} — resolve it deliberately, never here")
    sic_line = (f"# `serving_impl_commit` is the campaign's `UNFROZEN` literal UNCHANGED (the loader's marker: the ladder tree cannot carry\n"
                f"# its own sha); the run tree's serving stack is proven byte-for-byte at the head of every job by\n"
                f"# `as22_assert_serving_identity production` (the roster's production sha {want[:8]}…); never served from a patched tree."
                if carried == "UNFROZEN" else
                f"# `serving_impl_commit` is the wave's {carried[:8]}… UNCHANGED — the campaign's invariant: the run\n"
                f"# tree serves the wave's serving stack byte-for-byte, proven at the head of every job by\n"
                f"# `as22_assert_serving_identity production`; this file must never be served from a patched tree.")
    target_model = re.search(r"^  target_model: (.*)$", body, re.M).group(1).strip()
    target_rev = re.search(r"^  target_revision: (.*)$", body, re.M).group(1).strip()
    header = f"""# Petri-manifest serving pins — STRENGTH FRONTIER (paper Figure 5), AlphaSteer lane `{ln.name}`, target {target},
# strength view `{view}` (nominal base_scale {s!r}): the manifest every leg of this view stands its server up on through
# scripts/sdfbase/mt_serving_up.py. GENERATED by {GENERATOR} --lane {ln.name} — never
# hand-edited (byte-identity is tested). DERIVED from {src_rel} (this target's
# template manifest, named by the lane roster): EVERY serving pin below is that file's, byte-identical — target_model /
# target_revision, chat_template_hash, tool_parser, serving_packages, serving_launch_flags (incl. the
# --limit-mm-per-prompt text-only pin where the campaign carries it), the frozen scenario bundle, behaviours,
# auditor chain, bloom options, `cells` ([none, {CELL_ID}]) AND serving_impl_commit — EXCEPT exactly one key:
#
#   * `audit` = {tag}
#
{sic_line}
#
# Objects: <defenses_dir>/{CELL_ID}/object.pt = $SHARED/{ln.bank_root_rel}/{target}/strength/cells_view/{view}/{CELL_ID}
# -> ../../cells/{view}__{CELL_ID} (scripts/alphasteer_f22/bake_strength.py --lane {ln.name}; pin in {ln.pins_rel}),
# identity sidecar fit_provenance.json (the lambda=1 parent's, lineage `{ln.parent_lineage}`). Serving target: {target_model} @ {target_rev}
"""
    return header + body


# --------------------------------------------------------------------------- driver

def outputs(ln: Lane = None) -> List[Tuple[Path, str, str, float]]:
    """One config + one manifest per (target of the lane's roster, ladder strength)."""
    ln = ln or lane_of(DEFAULT_LANE)
    out = []
    for target in lane_targets(ln):
        for s in STRENGTHS:
            out.append((REPO / serve_config_rel(target, s, ln), "config", target, s))
            out.append((REPO / manifest_rel(target, s, ln), "manifest", target, s))
    return out


def render(kind: str, target: str, s: float, ln: Lane = None, *, roster: dict = None) -> str:
    return (render_config(target, s, ln, roster=roster) if kind == "config"
            else render_manifest(target, s, ln, roster=roster))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed file drifts")
    ap.add_argument("--lane", default=DEFAULT_LANE, choices=sorted(LANES),
                    help="which lane's roster names the targets and templates (default: the campaign-parent lane)")
    args = ap.parse_args(argv)
    ln = lane_of(args.lane)
    roster = load_roster(ln)
    rc = 0
    for dst, kind, target, s in outputs(ln):
        text = render(kind, target, s, ln, roster=roster)
        if args.check:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != text:
                print(f"DRIFT: {dst}"); rc = 1
            else:
                print(f"ok: {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".tmp-{dst.name}")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(dst)
            print(f"wrote {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
