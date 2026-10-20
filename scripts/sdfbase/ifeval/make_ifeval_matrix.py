#!/usr/bin/env python3
"""Generate (or check) the ifeval-canonical-battery MATRIX through the IFEval row builder (``if_views.select_rows``;
views.yaml's exclude / prompted_extra / deferred policy).

  make_ifeval_matrix.py                 this experiment: configs/sdfbase/ifeval/views.yaml + the FROZEN roster / pins
                                        copies it names (`frozen_inputs:`, pin 59cedcca — the shared registry as
                                        committed when the matrix was frozen; the LIVE registry has since gained a
                                        llama8b target and the constitution cell) + the Llama roster
                                        overlay + the Llama pins overlay of THIS tree
                                        -> configs/sdfbase/ifeval/matrix.json (FROZEN)
  make_ifeval_matrix.py --check         exit 1 if the committed matrix drifts from the generator OR a promised
                                        IFEVAL_CONTRASTS leg does not resolve among the launch rows (P32)
  make_ifeval_matrix.py --views "llama8b=main,it" --configs-root <tree> --out <tmp>/matrix.json
                                        a matrix over another tree's LIVE roster + pins (no views.yaml, no frozen
                                        inputs), the SAME builder
  make_ifeval_matrix.py --views-yaml configs/sdfbase/ifeval/views_qwen_seedrep.yaml \
                        --out configs/sdfbase/ifeval/matrix_qwen_seedrep.json [--check] [--table]
                                        the SIBLING document of the batch-64 seed lane views: 30 rows (24 seed +
                                        3 composed + 3 Arditi), native serving, its own frozen roster / pins
                                        copies; `--table` prints the (view, cell) table the launch-set derivation
                                        pins by value

Never hand-edit the output: tests/test_sdfbase_ifeval_matrix.py requires byte-identity (`--check`) and derives every
count from it. The document carries no timestamp, so a regeneration over the same inputs is byte-identical. The
HELD rows print as `HELD:` lines (P41) and the DEFERRED slots as `DEFERRED:` lines; neither is counted as a launch row.
The generator ASSERTS the constitution SI text's sha256 (views.yaml prompted_extra.text_sha256) on every run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import if_matrix as IM
import if_views as V

REPO = HERE.parents[2]


def build(*, configs_root: Path, views_yaml: Path | None, views_spec: str | None) -> tuple[str, dict]:
    """The rendered matrix text and document for the named inputs (the contrast resolution asserted)."""
    if views_spec is not None and views_yaml is not None:
        raise SystemExit("--views and --views-yaml are exclusive")
    if views_spec is not None:
        spec = V.ViewsSpec(views=V.parse_views_spec(views_spec))
        views_file = None
    else:
        views_file = views_yaml if views_yaml is not None else V.views_path(configs_root)
        spec = V.load_views_yaml(views_file)
    policy = V.load_ifeval_policy(views_file)
    # the FROZEN copies when views.yaml names them (their sha256 asserted), else this tree's live registry
    roster, pins, source = V.load_inputs(configs_root, spec)
    rows = V.select_rows(roster, pins, spec.views, holds=spec.holds, policy=policy, pins_source=source,
                         where=f"make_ifeval_matrix (configs root {configs_root})", configs_root=configs_root)
    V.check_contrasts(rows, where="make_ifeval_matrix", anchor_view=policy.anchor_view)
    doc = V.matrix_document(rows, spec=spec, source=source, views_file=views_file, configs_root=configs_root, policy=policy)
    return V.render_matrix(doc), doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--views", default=None, help="key=view[,view…];… (the CLI spec form; exclusive with --views-yaml)")
    ap.add_argument("--views-yaml", type=Path, default=None, help="a views.yaml (default: <configs-root>/configs/sdfbase/ifeval/views.yaml)")
    ap.add_argument("--configs-root", type=Path, default=REPO, help="the tree whose roster + pins (+ overlays, views.yaml) are read")
    ap.add_argument("--out", type=Path, default=None, help="where to write the matrix (default: <configs-root>/configs/sdfbase/ifeval/matrix.json)")
    ap.add_argument("--check", action="store_true", help="exit 1 if the matrix at --out drifts from the generator or a contrast does not resolve")
    ap.add_argument("--table", action="store_true", help="also print the (view, cell) table of the LAUNCH rows, one `view<TAB>cell` per line, in matrix order")
    a = ap.parse_args(argv)
    configs_root = a.configs_root.resolve()
    out = a.out if a.out is not None else configs_root / IM.MATRIX_PATH.relative_to(IM.REPO)
    text, doc = build(configs_root=configs_root, views_yaml=a.views_yaml, views_spec=a.views)
    c = doc["counts"]
    if a.table:
        for r in doc["rows"]:
            if r["status"] == IM.STATUS_LAUNCH:
                print(f"TABLE\t{r['view']}\t{r['cell']}")
    for r in doc["rows"]:
        if r["status"] == IM.STATUS_HELD:
            print(f"HELD: index={r['index']} job_id={r['job_id']} reason={r['held_reason']}")
        elif r["status"] == IM.STATUS_DEFERRED:
            print(f"DEFERRED: index={r['index']} job_id={r['job_id']} record={r['deferred']['record']} "
                  f"pending={r['deferred']['pending']}")
    if a.check:
        if not out.is_file() or out.read_text(encoding="utf-8") != text:
            print(f"DRIFT {out}")
            return 1
        print(f"OK {out}: {c['rows']} rows / {c['held']} held / {c['launched']} launched / {c['deferred']} deferred "
              f"byte-identical to the generator; "
              f"contrasts: {doc['contrasts'][V.IC.CONTRAST_VS_NONE]['n_resolved']}/{doc['contrasts'][V.IC.CONTRAST_VS_NONE]['n']} "
              f"cell_minus_none legs resolve, noise floor on {doc['contrasts'][V.IC.CONTRAST_NOISE]['models_with_floor']}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}: {c['rows']} rows / {c['held']} held / {c['launched']} launched / {c['deferred']} deferred / "
          f"{c['fitted_launched']} fitted launched / {c['in_support']} in support / {c['prompted_constitution']} constitution-prompted / "
          f"{c['header_located']} header-located / {c['n_tags']} TAGs ({', '.join(c['tags'])}) / "
          f"{c['n_slugs']} slugs / {c['distinct_paths']} distinct paths of {c['n_paths']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
