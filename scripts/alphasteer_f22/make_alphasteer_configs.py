#!/usr/bin/env python
"""alphasteer-full22 Phase F/G — the serving CONFIGS and PRODUCTION MANIFESTS of the served legs,
generated from the wave's own generated files; `--check` requires byte-identity (like
scripts/sdfbase/make_full22_configs.py). Never hand-edit the outputs.

Per served target (scripts/alphasteer_f22/as22_targets.py names the view: it / it / main / main):

  configs/alphasteer_f22_serve_<target>.yaml   (UNCHANGED from Phase C — reproduced byte-for-byte)
      The wave config for the SAME (target, view), byte-for-byte as data, EXCEPT exactly five things:
      `explicit_cells` ([none, alphasteer]), `defenses_dir` (our root's view), `out_dir`,
      `private_run` (`alphasteer_f22_<target>`) and the header. Rendered by
      make_ladder_configs.render_config — IMPORTED, not copied — because the four committed files
      already satisfy that contract exactly (verified before this generator existed: the only keys
      that differ from the wave body are those four + the header) and the SAME config serves both
      the ladder standups (Phase C, complete) and the Phase G legs. This generator adds the check,
      not a second spelling.

  configs/capeval/manifests/alphasteer_f22_<target>.yaml   (PRODUCTION — this generator's own)
      The wave manifest for the same (target, view) with EXACTLY two keys changed — `audit`
      (alphasteer_f22_<target>) and `cells` ([none, alphasteer] in the harness vocabulary) — plus
      the header. EVERYTHING else is byte-identical, INCLUDING `serving_impl_commit`: the wave's
      94e294c4… line, comment and all. That is an INVARIANT, not a value this file computes:
      every Phase G leg runs the unpatched tree W_A whose serving stack is byte-identical to the
      wave's own serving tree, which `as22_assert_serving_identity production` proves at the head of
      every job; the last commit touching capeval/serving/** + tiers/petri_runner.py on W_A is
      therefore the wave's, and a production manifest recording anything else would be a lie the
      harness hash would then bake into every audit id. The generator ASSERTS the carried value
      equals the roster's `serving_impl_commit` and is a full sha (never `UNFROZEN`, never abbreviated).

  WHAT CHANGED AND WHY (recorded before the change was made): until this
  generator, the four manifest files at these paths were the LADDER manifests written by
  make_ladder_configs.py, which carried `serving_impl_commit: UNFROZEN` — the loader's sentinel, by
  necessity, because the ladder tree W_L's serving_impl_commit was W_L itself and a tree
  cannot name its own sha. The harness hash REFUSES that sentinel, so mt_serving_up.py (the wave's
  standup, every Phase G leg's) could not serve on them. The ladder is COMPLETE and its
  standups never read the field (lens_gate_standup replaced it), so the manifests at these paths are
  now the production ones; make_ladder_configs.py keeps its config half and its historical manifest
  renderer, and no longer claims these manifest paths as outputs.

  A glm46vflash manifest is still generated: the manifest is a serving pin set, not a decision to
  serve — the roster's `not_served` marker, the null pin and every launcher's refusal are what keep
  glm off the GPUs. Generating it keeps the four-target byte-identity guard uniform.

usage:
  python scripts/alphasteer_f22/make_alphasteer_configs.py            # write the four manifests (configs are verified, never rewritten here)
  python scripts/alphasteer_f22/make_alphasteer_configs.py --check    # exit 1 on any drift, configs AND manifests
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

from as22_targets import (CELL_ID, LADDER_CELLS, PRODUCTION_SERVING_IMPL_COMMIT,  # noqa: E402
                          TARGETS, manifest_rel, serve_config_rel, view_of, wave_manifest_rel)
from make_ladder_configs import _sub_line, render_config  # noqa: E402  (the configs' ONE renderer)

ROSTER = REPO / "configs" / "alphasteer_f22_roster.yaml"
GENERATOR = "scripts/alphasteer_f22/make_alphasteer_configs.py"
#: the cells every production config / manifest lists (the same trio the ladder listed)
SERVED_CELLS = LADDER_CELLS


def roster_serving_impl_commit(roster_path: Path = ROSTER) -> str:
    v = yaml.safe_load(Path(roster_path).read_text(encoding="utf-8"))["serving_impl_commit"]
    if not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{40}", v):
        raise SystemExit(f"{roster_path}: serving_impl_commit {v!r} is not a full 40-hex sha")
    return v


def render_manifest(target: str, *, roster_path: Path = ROSTER) -> str:
    """The PRODUCTION manifest for ``target``: the wave manifest with `audit` and `cells` changed and
    a new header; `serving_impl_commit` carried across UNCHANGED and asserted."""
    view = view_of(target)
    src_rel = wave_manifest_rel(target)
    src = (REPO / src_rel).read_text(encoding="utf-8")
    if "\npetri:\n" not in src:
        raise SystemExit(f"{src_rel} has no top-level petri: block")
    body = src[src.index("\npetri:\n") + 1:]
    body = _sub_line(body, r"^  audit: .*$", f"  audit: alphasteer_f22_{target}")
    m = re.search(r"^  cells:\n(?:    - .*\n)+", body, flags=re.M)
    if not m:
        raise SystemExit(f"{src_rel}: cells block not found")
    body = body[:m.start()] + "  cells:\n" + "".join(f"    - {c}\n" for c in SERVED_CELLS) + body[m.end():]
    # serving_impl_commit: carried UNCHANGED, asserted (never substituted, never computed)
    sic = re.search(r"^  serving_impl_commit: (\S+)", body, flags=re.M)
    if not sic:
        raise SystemExit(f"{src_rel}: no serving_impl_commit line")
    carried, want = sic.group(1), roster_serving_impl_commit(roster_path)
    if carried != want:
        raise SystemExit(f"{src_rel} carries serving_impl_commit {carried} but the campaign roster says {want} — "
                         "the wave body and our roster disagree; resolve it deliberately (D75 ruling 4), never here")
    if carried != PRODUCTION_SERVING_IMPL_COMMIT or carried == "UNFROZEN":
        raise SystemExit(f"{src_rel}: serving_impl_commit {carried} is not the production sha "
                         f"{PRODUCTION_SERVING_IMPL_COMMIT} (as22_targets)")
    target_model = re.search(r"^  target_model: (.*)$", body, re.M).group(1).strip()
    target_rev = re.search(r"^  target_revision: (.*)$", body, re.M).group(1).strip()
    header = f"""# Petri-manifest serving pins — alphasteer-full22 campaign, target {target}, view `{view}`:
# the PRODUCTION manifest every Phase G leg (static, BoN, controls, multi-turn, the smokes, the Phase E
# lens-gate case) stands its server up on through scripts/sdfbase/mt_serving_up.py. GENERATED by
# {GENERATOR} — never hand-edited (byte-identity is tested). DERIVED from
# {src_rel}: EVERY serving pin below is that file's, byte-identical —
# target_model / target_revision, chat_template_hash, tool_parser, serving_packages,
# serving_launch_flags (incl. the --limit-mm-per-prompt text-only pin where the wave carries it:
# without it the multimodal profiling OOM-kills the 200G host cgroup), the frozen scenario bundle,
# behaviours, auditor chain, bloom options AND serving_impl_commit — EXCEPT exactly two keys:
#
#   * `audit`  = alphasteer_f22_{target}
#   * `cells`  = [none, {CELL_ID}] — EXACTLY the harness ids of configs/alphasteer_f22_serve_{target}.yaml's
#     explicit_cells (`{CELL_ID}` is a bare Class B id: registry_adapter.traintest_cell_id /
#     harness_cell_id are both the identity on it; mt_serving_up.py refuses a manifest/config
#     mismatch in either direction). The path-matched no-op leg `none__lens0` is DERIVED by
#     defense_cells.enumerate_cells, never listed.
#
# `serving_impl_commit` is the wave's {carried[:8]}… UNCHANGED — an invariant of every production leg (D75
# ruling 4): the run tree W_A serves the wave's serving stack byte-for-byte (D10), proven at the head of
# every job by `as22_assert_serving_identity production`. The ladder tree W_L was the one exception
# (D24/D25); its manifests recorded W_L per rung and the ladder is complete (D72–D74) — this file is
# NOT a ladder manifest and must never be served from a patched tree.
#
# Objects: <defenses_dir>/{CELL_ID}/object.pt = $SHARED/exp/objects/alphasteer_f22/{target}/cells_view/{view}/{CELL_ID}
# -> ../../cells/selected__alphasteer (the ladder's SELECTED rung, staged by stage_alphasteer.sh; pin in
# configs/alphasteer_f22_pins.json), identity sidecar fit_provenance.json. Serving target: {target_model} @ {target_rev}
"""
    return header + body


def outputs() -> List[Tuple[Path, str, str]]:
    out = []
    for target in TARGETS:
        out.append((REPO / serve_config_rel(target), "config", target))
        out.append((REPO / manifest_rel(target), "manifest", target))
    return out


def render(kind: str, target: str) -> str:
    return render_config(target) if kind == "config" else render_manifest(target)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed file drifts")
    args = ap.parse_args(argv)
    rc = 0
    for dst, kind, target in outputs():
        text = render(kind, target)
        if args.check or kind == "config":
            # configs are VERIFIED only: their renderer is make_ladder_configs.render_config and the
            # committed files must already equal it (this generator never rewrites them)
            if not dst.is_file() or dst.read_text(encoding="utf-8") != text:
                print(f"DRIFT: {dst}" + ("" if args.check else " (a config drifted; regenerate with make_ladder_configs.py)"))
                rc = 1
            else:
                print(f"ok: {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".tmp-{dst.name}")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(dst)
        print(f"wrote {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
