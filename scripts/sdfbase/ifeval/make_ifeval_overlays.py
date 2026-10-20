#!/usr/bin/env python3
"""Generate (or check) the ifeval-canonical-battery config + manifest OVERLAYS — one pair per TAG of the frozen matrix
(the ``make_gsm8k_overlays.py`` pattern). The campaign files are UNTOUCHED; each overlay
is the campaign file with ONE change:

  configs/sdfbase/ifeval/sdfbase_full22_<TAG>.ifeval.yaml
      the campaign config configs/sdfbase_full22_<TAG>.yaml with `explicit_cells` replaced by EXACTLY that TAG's matrix
      rows' cells, in matrix order — the launch AND held rows (a held row's overlay is in place for its later wave),
      NEVER the deferred slots (an AlphaSteer / Arditi slot has no object yet; its cell joins the overlay when
      its record fills and the matrix is regenerated). The prompted cells (safety, constitution) are matrix decisions
      the campaign configs' `explicit_cells` may not list — a prompted cell needs no object; a FITTED matrix cell the
      campaign config does not list refuses (the roster and the config disagree);
  configs/capeval/manifests/petri_full22_<TAG>.ifeval.yaml
      the campaign manifest with `cells:` replaced by those rows' harness ids (mt_serving_up.py / capctrl_serving_up.py
      refuse a config / manifest cell mismatch), content otherwise identical — the SAME chat_template_hash,
      serving_launch_flags, tool_parser, prompted_prefix_hashes (every campaign manifest carries the safety AND the
      constitution SI hashes, so `prompted__constitution` verifies per request without an edit here).

The two Llama TAGs' campaign files (configs/sdfbase_full22_llama8b{,_it}.yaml, petri_full22_llama8b{,_it}.yaml) are
byte-identical copies of those on origin/llama-paper-rows, so a later landing of that branch merges clean.

  make_ifeval_overlays.py [--check]                               this experiment's frozen matrix, this tree
  make_ifeval_overlays.py --matrix <m.json> --configs-root <tree>  a peer's matrix over their campaign files, written under
                                                                  <tree>/configs/... (what the --configs-root dry run drives)
  make_ifeval_overlays.py --matrix configs/sdfbase/ifeval/matrix_qwen_seedrep.json [--check]
                                                                  the SIBLING seed document's six TAG pairs:
                                                                  a `lane_extra` fitted cell the campaign config does not
                                                                  list (the Arditi arditi_allpos object the document declares
                                                                  staged in the view's bank) is ADMITTED; a TAG whose rows
                                                                  carry no `none` row (the paired views) lists the
                                                                  bare anchor FIRST anyway — the standup's select_cells
                                                                  admits only enumerated cells and every overlay lists it

Never hand-edit the outputs: tests/test_sdfbase_ifeval_overlays.py requires byte-identity (`--check`).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "gsm8k", HERE.parents[2] / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import yaml

import if_matrix as M
import if_views as V
from sisteer.capeval.tinycap import contracts as C
from sisteer.eval.cell_controls import MULTI_LAYER_ARMS_KEY
# the campaign-file grammar (the `explicit_cells:` / `cells:` blocks, the campaign paths, the per-TAG grouping) is the
# GSM8K overlay generator's, imported rather than re-spelled: one grammar for both overlay lanes
from make_gsm8k_overlays import MANIFEST_DIR_REL, _CELLS_BLOCK, _MANIFEST_CELLS, campaign_config, campaign_manifest, rows_by_tag  # noqa: F401

REPO = HERE.parents[2]
_PREFIX_HASHES = re.compile(r"^  prompted_prefix_hashes:.*\n(?:    \w+: [0-9a-f]{16}\n)+", re.MULTILINE)
GENERATOR = "scripts/sdfbase/ifeval/make_ifeval_overlays.py"


def overlay_rows(rows: list[dict]) -> list[dict]:
    """The rows an overlay lists: launch + held (in matrix order); a DEFERRED slot has no object and is left out."""
    return [r for r in rows if r["status"] in (M.STATUS_LAUNCH, M.STATUS_HELD)]


def is_lane_extra(row: dict) -> bool:
    """A fitted row whose object ANOTHER lane fit, declared staged in the view's bank by the document (the
    Arditi ``arditi_allpos`` object inside each ``s<s>`` view) — the campaign config never lists it, and the overlay admits it."""
    return row.get("object_source") == V.OBJECT_SOURCE_EXTERNAL and isinstance(row.get("lane_extra"), dict)


def has_own_anchor(rows: list[dict]) -> bool:
    """Whether the TAG's rows carry a ``none`` row of their own (a paired view's do not — its composed row
    pairs with ``s<s>/none``)."""
    return any(r["cell"] == M.CELL_NONE for r in rows)


def overlay_ids(rows: list[dict], key: str) -> list[str]:
    """The overlay's cell ids under ``key`` (``cell`` for the config, ``harness_cell`` for the manifest) in matrix order,
    with the bare anchor ``none`` FIRST when the TAG's rows carry none of their own — the standup's select_cells admits only
    enumerated cells and every overlay lists ``none`` (nothing launches it on such a TAG)."""
    ids = [r[key] for r in rows]
    return ids if has_own_anchor(rows) else [M.CELL_NONE, *ids]


def overlay_cells(rows: list[dict]) -> list[str]:
    """The overlay's ``explicit_cells`` (:func:`overlay_ids` under ``cell``)."""
    return overlay_ids(rows, "cell")


def overlay_paths(root: Path, trows: list[dict]) -> tuple[Path, Path]:
    """The TAG's overlay config / manifest, read off the rows themselves (every row of a TAG names the same pair)."""
    cfgs = {r["config"] for r in trows}
    mans = {r["manifest"] for r in trows}
    if len(cfgs) != 1 or len(mans) != 1:
        raise SystemExit(f"TAG {trows[0]['tag']}: its rows name {sorted(cfgs)} / {sorted(mans)} — one overlay pair per TAG")
    return root / cfgs.pop(), root / mans.pop()


def multi_layer_arms_for_tag(root: Path, rows: list[dict]) -> dict[str, list[int]]:
    """The declared multi-layer sets of the TAG's overlay rows' FITTED cells, read off the ONE declaration the repo
    carries for the same (model, view) bank: the bcb-bfcl-200-widen serving config (``contracts.bcbw_config_path``;
    cd8927d6 / S1-f — the Nemotron LoRA objects are fit on the hybrid's four attention layers [12, 17, 24, 32], and F9
    requires the declared set to EQUAL the object's layers). The campaign config carries a single ``layer: 20`` and no
    declaration, so an overlay that copied it alone made every Nemotron fitted cell refuse at the identity check and
    would have at the standup (the wave shakedown of 2026-09-13). A bank with no bcbw config, or one whose declaration
    names none of the TAG's cells, contributes no block; a declared cell the TAG does not serve is left out."""
    keys = {(r["model_key"], r["view"]) for r in rows}
    if len(keys) != 1:
        raise SystemExit(f"overlay rows span {sorted(keys)} — one (model, view) bank per TAG")
    ((key, view),) = keys
    # (a `lane_extra` row — the served-only Class B Arditi object — declares NOTHING here: its identity is derived from the
    # object and its fit_provenance.json by object_identity's served-only branch, which takes the layer from
    # the object itself, and the native standup serves it at its own layers. An earlier cut declared
    # `multi_layer_arms: {arditi_allpos: [12]}` to feed arm_spec_for_cell(arm=vector_all); that path hashes a RESPONSE-masked
    # serve under load_config's forced steer_positions and was the wrong derivation for an all-positions arm — retired.)
    path = root / C.bcbw_config_path(key, view)
    if not path.is_file():
        return {}
    declared = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(MULTI_LAYER_ARMS_KEY) or {}
    return {r["cell"]: [int(x) for x in declared[r["cell"]]] for r in rows if r["fitted"] and r["cell"] in declared}


def multi_layer_arms_block(root: Path, rows: list[dict]) -> str:
    """The YAML lines inserted right after ``explicit_cells`` — empty when the bank declares nothing."""
    ml = multi_layer_arms_for_tag(root, rows)
    if not ml:
        return ""
    ((key, view),) = {(r["model_key"], r["view"]) for r in rows}
    return (f"# declared multi-layer objects, from {C.bcbw_config_path(key, view)} (cd8927d6 / S1-f): F9 requires each set to\n"
            "# EQUAL the object's target_layers; the campaign config's single `layer` cannot express them\n"
            f"{MULTI_LAYER_ARMS_KEY}:\n" + "".join(f"  {c}: [{', '.join(str(x) for x in ls)}]\n" for c, ls in ml.items()))


def render_config(root: Path, tag: str, trows: list[dict]) -> str:
    src_path = campaign_config(root, tag)
    if not src_path.is_file():
        raise SystemExit(f"no campaign config for TAG {tag!r} at {src_path} (the roster's view names it)")
    src = src_path.read_text(encoding="utf-8")
    m = _CELLS_BLOCK.search(src)
    if not m:
        raise SystemExit(f"{src_path}: no explicit_cells block")
    listed = [line[2:].strip() for line in m.group(0).splitlines()[1:]]
    rows = overlay_rows(trows)
    for r in rows:
        if r["fitted"] and r["cell"] not in listed and not is_lane_extra(r):
            raise SystemExit(f"{src_path}: the campaign config does not list the fitted cell {r['cell']} of the matrix — "
                             "the roster and the config disagree; regenerate the campaign configs first")
    cells = overlay_cells(rows)
    extra_rows = [r for r in rows if is_lane_extra(r)]
    n_held = sum(r["status"] == M.STATUS_HELD for r in rows)
    n_deferred = sum(r["status"] == M.STATUS_DEFERRED for r in trows)
    prompted = [r["cell"] for r in rows if r["cell"].startswith("prompted__")]
    lane_note = ("" if not extra_rows else
                 f"# LANE EXTRA (b64-capability-agentic-queue): {[r['cell'] for r in extra_rows]} is a fitted Class B cell ANOTHER lane fit, which the\n"
                 f"# campaign config does not list — the matrix document declares it STAGED in this view's bank on the volume (an extra\n"
                 f"# bare-cell dir beside the roster's cells; staged id {[r['staged_id'] for r in extra_rows]}, pinned from\n"
                 f"# {sorted({r['pin']['source'] for r in extra_rows})}); the explicit-cell registry validates it there.\n")
    anchor_note = ("" if has_own_anchor(rows) else
                   "# NO OWN ANCHOR: this TAG's matrix rows carry no `none` row — its cell pairs with the seed view's\n"
                   "# `none` (the document's anchor_view); `none` is listed FIRST here only because the standup's select_cells admits\n"
                   "# only enumerated cells and every overlay lists the bare anchor. Nothing launches it on this TAG.\n")
    serving_note = ("" if not any(r.get("serving_path") for r in rows) else
                    f"# SERVING PATH: the matrix document declares serving_path {sorted({r.get('serving_path') for r in rows})} — the\n"
                    "# dispatcher passes it to the launcher; the store's manifest attestation is the fact, never this header.\n")
    header = f"""# sdfbase_full22_{tag}.ifeval.yaml — the ifeval-canonical-battery OVERLAY of configs/sdfbase_full22_{tag}.yaml
# GENERATED by {GENERATOR} from the frozen matrix — never hand-edited
# (tests/test_sdfbase_ifeval_overlays.py requires byte-identity). The campaign config's body follows BYTE FOR BYTE except
# `explicit_cells` (and, right after it, a `multi_layer_arms` block for the fitted cells the bank's bcbw serving config
# declares as multi-layer — F9 equality; none when it declares none), which is EXACTLY this TAG's matrix rows in matrix order: the same-wave bare anchor, the prompted
# cells the matrix places on this view ({', '.join(prompted) or 'none here'}), and every served object of the view
# ({n_held} held row(s) included: a held row's overlay is in place for its later wave; {n_deferred} deferred
# AlphaSteer / Arditi slot(s) left out until their object record fills and the matrix is regenerated). Served with
# capctrl_serving_up.py --cell <cell> per Modal container, thinking OFF (no @thinking spelling), sampling vendor_preset,
# k = 2, at the cap's geometry entry ({trows[0]['geometry_eval_by_cap']['2048']} at pass 1 for the short prompt class;
# the constitution-prompted cell at capability:ifeval_longsi_<cap>), --upstream-timeout-s 7200.
{lane_note}{anchor_note}{serving_note}#
"""
    body = (src[:m.start()] + "explicit_cells:\n" + "".join(f"- {c}\n" for c in cells)
            + multi_layer_arms_block(root, rows) + src[m.end():])
    return header + body


def render_manifest(root: Path, tag: str, trows: list[dict]) -> str:
    src_path = campaign_manifest(root, tag)
    if not src_path.is_file():
        raise SystemExit(f"no campaign manifest for TAG {tag!r} at {src_path}")
    src = src_path.read_text(encoding="utf-8")
    m = _MANIFEST_CELLS.search(src)
    if not m:
        raise SystemExit(f"{src_path}: no cells block")
    rows = overlay_rows(trows)
    hashes = _PREFIX_HASHES.search(src)
    carried = set(re.findall(r"^    (\w+): [0-9a-f]{16}$", hashes.group(0), re.MULTILINE)) if hashes else set()
    for r in rows:
        if r["harness_cell"] is None:
            raise SystemExit(f"{tag}: row {r['index']} ({r['cell']}) has no harness spelling — not an overlay row")
        if r["harness_cell"].startswith("prompted__") and r["harness_cell"][len("prompted__"):] not in carried:
            raise SystemExit(f"{src_path}: prompted_prefix_hashes carries no hash for the SI of {r['harness_cell']} "
                             f"(carried: {sorted(carried)}) — the prompted intervention could not be verified per request")
    header = (f"# petri_full22_{tag}.ifeval.yaml — the ifeval-canonical-battery OVERLAY of petri_full22_{tag}.yaml: the\n"
              f"# campaign manifest BYTE FOR BYTE except `cells`, which names EXACTLY the harness ids of\n"
              f"# configs/sdfbase/ifeval/sdfbase_full22_{tag}.ifeval.yaml's explicit_cells (the standup refuses a\n"
              f"# config / manifest cell mismatch; prompted__constitution verifies against the constitution hash the\n"
              f"# campaign manifest already carries). GENERATED by {GENERATOR}.\n")
    # the harness ids in the SAME order as the config's explicit_cells (the bare anchor first when the TAG lists none of its own)
    return header + src[:m.start()] + "  cells:\n" + "".join(f"    - {h}\n" for h in overlay_ids(rows, "harness_cell")) + src[m.end():]


def outputs(root: Path, rows: list[dict]):
    for tag, trows in rows_by_tag(rows).items():
        cfg_path, man_path = overlay_paths(root, trows)
        yield cfg_path, render_config(root, tag, trows)
        yield man_path, render_manifest(root, tag, trows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--matrix", type=Path, default=None, help="the matrix (default: <configs-root>/configs/sdfbase/ifeval/matrix.json)")
    ap.add_argument("--configs-root", type=Path, default=REPO, help="the tree whose campaign configs / manifests are overlaid, and where the overlays land")
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed overlay drifts from the generator")
    a = ap.parse_args(argv)
    root = a.configs_root.resolve()
    matrix = a.matrix if a.matrix is not None else root / M.MATRIX_PATH.relative_to(M.REPO)
    rows = M.load_matrix(matrix)["rows"]
    drift = 0
    n = 0
    for path, text in outputs(root, rows):
        n += 1
        if a.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                print(f"DRIFT {path}")
                drift += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(root) if path.is_relative_to(root) else path}")
    if a.check and not drift:
        print(f"OK: {len(rows_by_tag(rows))} TAGs' overlays ({n} files) byte-identical to the generator")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
