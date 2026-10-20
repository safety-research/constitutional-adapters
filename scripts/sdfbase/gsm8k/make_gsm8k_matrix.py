#!/usr/bin/env python3
"""Generate (or check) a gsm8k-paired200-paper-objects MATRIX through the ONE view-selection interface
(``g8k_views.select_rows``).

  make_gsm8k_matrix.py                                   this experiment: configs/sdfbase/gsm8k/views.yaml + the FROZEN roster
                                                         + pins copies it names (`frozen_inputs:`, P16) -> configs/sdfbase/
                                                         gsm8k/matrix.json (FROZEN; the live registry may grow, this does not)
  make_gsm8k_matrix.py --check                           exit 1 if the committed matrix drifts from the generator
  make_gsm8k_matrix.py --matrix-name llama [--check]     the Llama rows (llama-paper-rows): views_llama.yaml + ITS frozen roster
                                                         + pins copies (`frozen_inputs:`, since 2026-09-14) -> configs/sdfbase/gsm8k/matrix_llama.json (schema g8k_matrix/2)
  make_gsm8k_matrix.py --views "gemma4e4b=it_d0p1" --configs-root <a peer's checkout> --out <tmp>/matrix.json
                                                         a peer's matrix from THEIR roster + pins (their dose views), the
                                                         SAME selector — what g8k_dry_run.py --configs-root drives
  make_gsm8k_matrix.py --matrix-name frontdoor_<key> [--check]
                                                         THE FRONT DOOR: ONE matrix per
                                                         front-door model of reproduce/models.yaml (g8k_cells.front_door_facts:
                                                         the family, arm, sampling row from the table), schema g8k_matrix/2 under
                                                         its OWN wave root (`<published root>_frontdoor_<key>`, wave `frontdoor_<key>`),
                                                         rows = the table's PHASE VIEWS the LIVE roster names (05_stage_roster.sh
                                                         --step roster pastes the block) x the roster's cells — DERIVED, no views
                                                         file; refuses by name while the roster carries none of the views ->
                                                         configs/sdfbase/gsm8k/matrix_frontdoor_<key>.json. The frozen default
                                                         matrix and every paper row are untouched.

Never hand-edit the output: tests/test_sdfbase_gsm8k_sbatch.py requires byte-identity (`--check`) and derives every
count from it. The document carries no timestamp, so a regeneration over the same inputs is byte-identical.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import g8k_cells as M
import g8k_views as V

REPO = HERE.parents[2]


def build(*, configs_root: Path, views_yaml: Path | None, views_spec: str | None,
          matrix_name: str = M.MATRIX_DEFAULT_NAME) -> tuple[str, dict]:
    """The rendered matrix text and document for the named inputs. A views file naming `frozen_inputs:` is regenerated from
    those IMMUTABLE copies (their sha256 asserted; the document records the published registry paths and the copies'
    hashes — equal to the pin's, byte for byte); otherwise the configs root's live roster + pins are the inputs."""
    if views_spec is not None and views_yaml is not None:
        raise SystemExit("--views and --views-yaml are exclusive")
    if views_spec is not None:
        spec = V.ViewsSpec(views=V.parse_views_spec(views_spec))
        views_file = None
    elif matrix_name.startswith(M.FRONT_DOOR_MATRIX_PREFIX):
        if views_yaml is not None:
            raise SystemExit(f"--matrix-name {matrix_name}: a front-door matrix is DERIVED from the models table + the roster; it takes no --views-yaml")
        spec, views_file = front_door_spec(matrix_name, configs_root), None
    else:
        views_file = views_yaml if views_yaml is not None else configs_root / M.VIEWS_BY_MATRIX[matrix_name].relative_to(M.REPO)
        spec = V.load_views_yaml(views_file)
    live_roster, live_pins = V.roster_path(configs_root), V.pins_path(configs_root)
    frozen = V.frozen_input_paths(spec, configs_root)
    roster_file, pins_file = frozen if frozen is not None else (live_roster, live_pins)
    rel = (lambda p: str(Path(p).resolve().relative_to(Path(configs_root).resolve())))
    # E2 (plan P2 / P8): the appended waves — the `extensions:` prompted rows need no pin and no roster entry; the `steering:`
    # rows are generated from the P8 admission record of the SAME configs root (the record the launcher's three doors assert)
    admission_file = V.admission_path(configs_root) if spec.steering else None
    admission = V.load_admission(configs_root) if spec.steering else None
    # E3: the `itdoc:` rows are generated from the it-doc admission record of the SAME configs root (the record the bank door re-reads)
    itdoc_file = V.itdoc_admission_path(configs_root) if spec.itdoc else None
    itdoc_admission = V.load_itdoc_admission(configs_root) if spec.itdoc else None
    rows = V.select_rows(V.load_roster(roster_file), V.load_pins(pins_file), spec.views, holds=spec.holds, exclude=spec.exclude,
                         schema=spec.schema, trigger_threshold=spec.trigger_threshold, where=f"make_gsm8k_matrix (configs root {configs_root})",
                         extensions=spec.extensions, steering=spec.steering, admission=admission, itdoc=spec.itdoc, itdoc_admission=itdoc_admission,
                         row_overlays=spec.row_overlays, object_generation=spec.object_generation, wave=spec.wave,
                         inspect_seed_policy=spec.inspect_seed_policy, inspect_seed_base_by_view=spec.inspect_seed_base_by_view,
                         standup_companions=spec.standup_companions, geometry_class=spec.geometry_class)
    doc = V.matrix_document(rows, spec=spec, roster_file=roster_file, pins_file=pins_file, views_file=views_file,
                            configs_root=configs_root,
                            roster_rel=rel(live_roster) if frozen is not None else None,
                            pins_rel=rel(live_pins) if frozen is not None else None, admission_file=admission_file,
                            itdoc_admission_file=itdoc_file)
    return V.render_matrix(doc), doc


def front_door_spec(matrix_name: str, configs_root: Path) -> "V.ViewsSpec":
    """The FRONT DOOR's views spec for `frontdoor_<key>` (module docstring): the key's phase views from the models table
    (g8k_cells.front_door_facts) that the configs root's LIVE roster names, in the table's order; schema 2 under the key's own wave
    root and wave; the campaign trigger. Refuses by name while the roster names none of them (the block is pasted by
    05_stage_roster.sh --step roster; the pins by --step pins) — a matrix with no row is never written."""
    key = M.front_door_key_of(matrix_name)
    facts = M.front_door_facts()[key]
    roster = V.load_roster(V.roster_path(configs_root))
    named = ((roster.get("targets") or {}).get(key) or {}).get("views") or {}
    views = [v for v in facts["views"] if v in named]
    if not views:
        raise SystemExit(f"--matrix-name {matrix_name}: the roster {V.roster_path(configs_root)} does not carry {key}'s phase views "
                         f"{list(facts['views'])} yet (05_stage_roster.sh --step roster pastes the block; --step configs renders the wave "
                         "pair); a front-door matrix is rendered once it does — nothing to write")
    return V.ViewsSpec(views={key: views}, schema=M.MATRIX_SCHEMA_2, wave_root_rel=M.front_door_wave_root_rel(key),
                       trigger_threshold=M.TRIGGER_THRESHOLD, wave=M.front_door_wave(key))


def matrix_out_path(matrix_name: str, configs_root: Path) -> Path:
    """Where a registered matrix lands under a configs root: MATRICES' path for the frozen names, the front door's for `frontdoor_<key>`."""
    if matrix_name.startswith(M.FRONT_DOOR_MATRIX_PREFIX):
        M.front_door_key_of(matrix_name)                  # an unknown front-door key refuses BY NAME here, before any lookup
        return Path(configs_root) / M.front_door_matrices()[matrix_name].relative_to(M.REPO)
    return Path(configs_root) / M.MATRICES[matrix_name].relative_to(M.REPO)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--views", default=None, help="key=view[,view…];… (the CLI spec form; exclusive with --views-yaml)")
    ap.add_argument("--views-yaml", type=Path, default=None, help="a views.yaml (default: <configs-root>/configs/sdfbase/gsm8k/views.yaml)")
    ap.add_argument("--configs-root", type=Path, default=REPO, help="the tree whose roster + pins (+ views.yaml) are read (default: this tree)")
    ap.add_argument("--out", type=Path, default=None, help="where to write the matrix (default: <configs-root>/configs/sdfbase/gsm8k/matrix.json)")
    ap.add_argument("--check", action="store_true", help="exit 1 if the matrix at --out drifts from the generator")
    ap.add_argument("--matrix-name", default=M.MATRIX_DEFAULT_NAME,
                    help=f"which registered matrix: its views file and output path (default: the published matrix; one of {sorted(M.MATRICES)}, "
                         f"or {M.FRONT_DOOR_MATRIX_PREFIX}<models.yaml key> — the front door's derived matrix)")
    a = ap.parse_args(argv)
    if a.matrix_name not in M.MATRICES and not a.matrix_name.startswith(M.FRONT_DOOR_MATRIX_PREFIX):
        raise SystemExit(f"--matrix-name {a.matrix_name!r}: not one of {sorted(M.MATRICES)} and not {M.FRONT_DOOR_MATRIX_PREFIX}<models.yaml key>")
    configs_root = a.configs_root.resolve()
    out = a.out if a.out is not None else matrix_out_path(a.matrix_name, configs_root)
    text, doc = build(configs_root=configs_root, views_yaml=a.views_yaml, views_spec=a.views, matrix_name=a.matrix_name)
    c = doc["counts"]
    if a.check:
        if not out.is_file() or out.read_text(encoding="utf-8") != text:
            print(f"DRIFT {out}")
            return 1
        print(f"OK {out}: {c['rows']} rows / {c['held']} held / {c['launched']} launched byte-identical to the generator")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}: {c['rows']} rows / {c['held']} held / {c['launched']} launched / {c['fitted_launched']} fitted launched / "
          f"{c['in_support']} in support / {c['declared']} declared / {c['servable']} servable / {c['n_tags']} TAGs "
          f"({', '.join(c['tags'])}); waves {c['waves']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
