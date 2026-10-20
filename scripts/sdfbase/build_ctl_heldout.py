#!/usr/bin/env python
"""Build the CONTROL HELD-OUT pool ``constv2ctl-heldout`` — the remainder of
the control corpus the ``constv2ctl-100k`` draw never trained on ("loses
the control-corpus CE gain" needs control-register text OUTSIDE the
ctl-100k fit; the ~59k-token unused remainder of the 159k control corpus
is the only such text, ~20 docs — the power caveat is stated where the
number is read).

Mechanics, all reused from scripts/sdfbase/build_pools.py so the rows are
byte-identical to what the control cell's own sampler would have emitted:

1. the two control finals are sha-checked against the ctl cell manifest's
   ``corpus_shas`` (a drifted input refuses — the remainder must be the
   remainder of THE corpus the draw was cut from);
2. docs render through ``make_doc`` (renderer parity with the corpus repo),
   the SAME draw order is rebuilt (``sha_shuffle`` under ``SALT_CTL`` +
   ``greedy_interleave`` on the natural shares), and the recorded
   ``doc_ids_in_order`` must appear in that order in the same sequence
   (proof it is the same draw);
3. the remainder = every doc not in ``doc_ids_in_order`` (the truncated
   boundary doc counts as USED — its untrained tail is dropped rather than
   leaked as "held out" text; a boundary-skipped doc was never trained on
   and stays in the remainder), taken WHOLE (``cut_cell(budget=None)``);
4. ``build_cell`` writes ``pairs_constv2ctl-heldout.jsonl`` + manifest with
   lane ``control_heldout``, the ``ctl-`` id prefix, and a ``remainder_of``
   block naming the ctl cell, its manifest sha and the excluded ids.

Run the max-length preflight on the result before staging it
(``build_pools.py preflight`` on a subset dir), then
``modal volume put twosi-fits <pool> pool/<key>/pairs_constv2ctl-heldout.jsonl``
for each model key (+ the manifest).

  python scripts/sdfbase/build_ctl_heldout.py \\
      --pools $STAGE/sdfbase/pools \\
      --control-advice corpus_generator/data/control/final-control-20260901-advice.jsonl \\
      --control-stories corpus_generator/data/control/final-control-20260901-stories.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import build_heldout_split as hsplit  # noqa: E402
import build_pools as bp  # noqa: E402

CTL_CELL = "constv2ctl-100k"
OUT_CELL = "constv2ctl-heldout"
LANE = "control_heldout"


def build_control_heldout(pools_dir, *, control_advice=None, control_stories=None,
                          ctl_cell: str = CTL_CELL, out_cell: str = OUT_CELL,
                          rig=None, generator: dict | None = None) -> dict:
    pools_dir = Path(pools_dir)
    mpath = pools_dir / f"pairs_{ctl_cell}.manifest.json"
    if not mpath.is_file():
        raise bp.SamplerError(f"{mpath} missing — the control cell must be "
                              "built (and staged here) first")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if control_advice is None and control_stories is None:
        raise bp.SamplerError("no --control-advice/--control-stories input")
    recorded = manifest.get("corpus_shas") or {}
    ctl_shas: dict[str, str] = {}
    for key, path in (("control_advice", control_advice),
                      ("control_stories", control_stories)):
        if path is None:
            continue
        got = hsplit.sha256_file(Path(path))
        if recorded.get(key) != got:
            raise bp.SamplerError(
                f"{path} hashes to {got} but {mpath} records "
                f"{recorded.get(key)!r} for {key} — not the corpus the "
                f"{ctl_cell} draw was cut from; refusing to build a "
                "remainder of a different corpus")
        ctl_shas[key] = got
    if set(ctl_shas) != set(recorded):
        raise bp.SamplerError(
            f"{mpath} records corpus files {sorted(recorded)} but only "
            f"{sorted(ctl_shas)} were given — the remainder needs every "
            "family the draw interleaved")

    tokens_mod = hsplit.generator_tokens()
    rig = rig or bp.StagedCountRig()
    fams: dict[str, list] = {}
    if control_advice is not None:
        fams["advice"] = [bp.make_doc(r, "advice", tokens_mod)
                          for r in hsplit.load_jsonl(Path(control_advice))]
    if control_stories is not None:
        fams["stories"] = [bp.make_doc(r, "control_stories", tokens_mod)
                           for r in hsplit.load_jsonl(Path(control_stories))]
    # build_pools' (nested) fam_official_total, formula for formula: the
    # family's official total is min over tokenizers of the summed counts.
    def fam_official_total(docs) -> int:
        tot = {k: 0 for k in bp.TOK_NAMES}
        for d in docs:
            for k in bp.TOK_NAMES:
                tot[k] += d.token_counts[k]
        return min(tot.values())

    totals = {f: fam_official_total(ds) for f, ds in fams.items()}
    grand = sum(totals.values())
    shares = {f: t / grand for f, t in totals.items()}
    order = bp.greedy_interleave(
        {f: bp.sha_shuffle(ds, bp.SALT_CTL) for f, ds in fams.items()}, shares)

    used_ids = list(manifest["doc_ids_in_order"])
    pos = {bp.CTL_ID_PREFIX + d.doc_id: i for i, d in enumerate(order)}
    missing = [u for u in used_ids if u not in pos]
    if missing:
        raise bp.SamplerError(
            f"{len(missing)} doc id(s) of {ctl_cell} are not in the rebuilt "
            f"control draw (e.g. {missing[:3]}) — different corpus or "
            "prefix convention")
    seq = [pos[u] for u in used_ids]
    if any(b <= a for a, b in zip(seq, seq[1:])):
        raise bp.SamplerError(
            f"{ctl_cell}'s recorded doc order is not a subsequence of the "
            "rebuilt draw order — the draw salts or shares differ; refusing")
    used = set(used_ids)
    remainder = [d for d in order if bp.CTL_ID_PREFIX + d.doc_id not in used]
    if not remainder:
        raise bp.SamplerError(f"{ctl_cell} consumed the whole control corpus "
                              "— no remainder to hold out")
    docs, cum, skips = bp.cut_cell(remainder, None, rig)
    assert not skips  # budget=None never skips
    docs = [replace(d, doc_id=bp.CTL_ID_PREFIX + d.doc_id) for d in docs]
    generator = generator or hsplit.generator_provenance()
    m = bp.build_cell(
        out_cell, LANE, None, docs, cum,
        salts={"draw": bp.SALT_CTL, "remainder_of": ctl_cell},
        natural_ratio=shares.get("advice", 0.0), out_dir=pools_dir, rig=rig,
        split_sha=manifest["heldout_split_sha"], corpus_shas=ctl_shas,
        generator=generator, ratio_gate_mode="record",
        constdatagen_provenance=manifest.get("constdatagen_provenance"))
    m["remainder_of"] = {
        "cell": ctl_cell,
        "manifest_sha256": hsplit.sha256_file(mpath),
        "n_excluded": len(used),
        "excluded_doc_ids": used_ids,
        "truncated_boundary_doc_counted_as_used":
            (manifest.get("truncation") or {}).get("doc_id"),
        "n_control_corpus_docs": len(order),
    }
    out_ids = set(m["doc_ids_in_order"])
    if out_ids & used:
        raise bp.SamplerError("internal: remainder overlaps the trained draw")
    if len(out_ids) + len(used) != len(order):
        raise bp.SamplerError("internal: remainder + trained != corpus")
    (pools_dir / f"pairs_{out_cell}.manifest.json").write_text(
        json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools", type=Path, default=bp.DEFAULT_OUT)
    ap.add_argument("--control-advice", type=Path, default=None)
    ap.add_argument("--control-stories", type=Path, default=None)
    ap.add_argument("--ctl-cell", default=CTL_CELL)
    ap.add_argument("--out-cell", default=OUT_CELL)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    args.pools = bp.site_path(args.pools, "--pools")   # the default sits under the staging directory, $STAGE
    try:
        generator = hsplit.refuse_dirty_generator_tree(args.allow_dirty,
                                                      tag="build-ctl-heldout")
        m = build_control_heldout(args.pools, control_advice=args.control_advice,
                                  control_stories=args.control_stories,
                                  ctl_cell=args.ctl_cell, out_cell=args.out_cell,
                                  generator=generator)
    except (bp.SamplerError, hsplit.SplitError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"{args.out_cell}: {m['n_docs']} docs, official "
          f"{m['official_tokens']:,} ({m['limiting_tokenizer']}), families "
          f"{m['family_counts']}, remainder of {m['remainder_of']['cell']} "
          f"({m['remainder_of']['n_excluded']} trained docs excluded) -> "
          f"{args.pools / f'pairs_{args.out_cell}.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
