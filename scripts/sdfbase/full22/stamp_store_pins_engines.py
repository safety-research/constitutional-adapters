#!/usr/bin/env python3
"""STAMP each admitted STEER leg's serving ENGINE into a store-pins record — `_meta.serving_engine_by_leg` — DERIVED from the leg's
PULLED store, never typed: the launchers' serving-path pin follows the store, and "the store on record" must include the admitted
record that travels with the launch tree, not only the launching cluster's disk.

For every EXTERNAL row of a launcher's matrix (the AlphaSteer / Arditi steer rows) and every admitted-leg key of that row present in
the pins record (`leg_keys(row)`: the GSM8K row's `@8192` and `@16384` legs, the MATH-500 row's one leg), the leg's pulled store is
located under one of the `--pulled-root` directories at the row's out root (the cap leg) or rung sibling root (the `@16384` leg):
its engine markers are read with the launchers' own reader (`steer_store_pin.read_store_engines`: identity + engine stamp off
`evidence.json` and `cells/*/manifest.json`), its `cells/*/manifest.json` `producer.tree_sha` is REQUIRED to equal the leg's pinned
tree (a pulled store of another tree is not this leg's), and the entry is written with its provenance:

    {"engine": "lens" | "lora_native", "producer_tree_sha": "<40 hex>", "pulled_store": "<pulled root>/<rel>",
     "readings": [["<file>", "<field>", "<value>"], ...], "stamped": "<UTC date>"}

A leg whose pulled store is absent from every `--pulled-root` is REFUSED by name unless `--allow-missing` names it (the record then
carries no engine for it and the launchers' pin reads it as unknown). Writes the record back with `_meta.schema` bumped to `/2` and a
`serving_engine_source` note; the leg -> tree entries are byte-identical (every store-pins reader skips `_meta`).

  stamp_store_pins_engines.py --launcher gsm8k --matrix default --pulled-root <pull dir> [--pulled-root …] [--pins PATH] [--dry-run]
  stamp_store_pins_engines.py --launcher gsm8k --matrix llama  --pulled-root <pull dir>
  stamp_store_pins_engines.py --launcher gsm8k --matrix arditi_allpos --pulled-root <pull dir>   # the all-positions record (both rung legs)
  stamp_store_pins_engines.py --launcher math500               --pulled-root <pull dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import steer_store_pin as SP  # noqa: E402

SCHEMA_BUMP = {"g8k_store_pins/1": "g8k_store_pins/2", "m5t_store_pins/1": "m5t_store_pins/2"}
SOURCE_NOTE = ("_meta.serving_engine_by_leg: each admitted STEER leg's serving engine (lens | lora_native), DERIVED by "
               "scripts/sdfbase/full22/stamp_store_pins_engines.py from the leg's PULLED store (evidence.json / cells/*/manifest.json "
               "route: serving_identity + serving_engine stamp; steer_store_pin.read_store_engines), the store's producer tree verified "
               "equal to the leg's pin; read by the launchers' serving-path pin (g8k_cells.serving_path_pin / math500 cells.serving_path_pin) "
               "as the PINS source of 'a store on record' beside the launching cluster's disk — the pin follows the store (2026-09-15/16)")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                     # the launchers declare dataclasses, which resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _rows(launcher: str, matrix: str | None):
    if launcher == "gsm8k":
        g = _load_module("g8k_cells_stamp", REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_cells.py")
        mpath = g.resolve_matrix(matrix)
        rows = [r for r in g.matrix_rows(mpath) if r.get("object_source") == g.OBJECT_SOURCE_EXTERNAL]
        pins = REPO / "configs" / "sdfbase" / "gsm8k" / f"store_pins_{g.matrix_name_of(mpath)}.json"
        # leg key -> the out root (rel) whose store IS that leg: the cap leg's root, the rung leg's sibling root
        legs = {}
        for r in rows:
            for key, rel in zip(g.leg_keys(r), g.store_roots_rel(r)):
                legs[key] = (r, rel)
        return legs, pins
    if launcher == "math500":
        m = _load_module("m5t_cells_stamp", REPO / "scripts" / "sdfbase" / "math500" / "cells.py")
        rows = [r for r in m.M5T_CELLS if r.get("object_source") == m.OBJECT_SOURCE_EXTERNAL]
        legs = {m.leg_keys(r)[0]: (r, m.store_roots_rel(r)[0]) for r in rows}
        return legs, m.STORE_PINS_PATHS[0]
    raise SystemExit(f"--launcher must be gsm8k or math500, got {launcher!r}")


def _producer_tree(root: Path) -> str | None:
    trees = set()
    for mp in sorted(glob.glob(str(root / "cells" / "*" / "manifest.json"))):
        prod = json.loads(Path(mp).read_text()).get("producer") or {}
        if prod.get("tree_sha"):
            trees.add(str(prod["tree_sha"]))
    if len(trees) > 1:
        raise SystemExit(f"REFUSED: {root}: the cell manifests name {len(trees)} producer trees {sorted(trees)} — not one leg's store")
    return next(iter(trees)) if trees else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--launcher", required=True, choices=("gsm8k", "math500"))
    ap.add_argument("--matrix", default=None, help="the GSM8K matrix (default | llama | itdoc | arditi_allpos); ignored for math500")
    ap.add_argument("--pulled-root", action="append", required=True, help="a grading host pull of the wave root (its controls_steer/… tree); repeatable")
    ap.add_argument("--pins", default=None, help="the store-pins record to stamp (default: the launcher's for the matrix)")
    ap.add_argument("--allow-missing", action="append", default=[], metavar="LEG_KEY", help="a leg whose pulled store may be absent (recorded as such, no engine)")
    ap.add_argument("--dry-run", action="store_true", help="print the block, write nothing")
    a = ap.parse_args(argv)
    legs, pins_path = _rows(a.launcher, a.matrix)
    pins_path = Path(a.pins) if a.pins else pins_path
    raw = pins_path.read_text(encoding="utf-8")
    doc = json.loads(raw)
    # the records are written `indent=1`, `_meta` first then the legs in their own order, ASCII-escaped or not per file: the re-dump
    # must reproduce the untouched bytes (the leg -> tree entries are the analysis' admission; only _meta grows)
    ensure_ascii = "\\u" in raw
    if json.dumps(doc, indent=1, ensure_ascii=ensure_ascii) + "\n" != raw:
        raise SystemExit(f"REFUSED: {pins_path} is not a canonical indent=1 dump (a re-write would move bytes outside _meta) — normalise it first")
    meta = doc.get(SP.PINS_META_KEY)
    if not isinstance(meta, dict):
        raise SystemExit(f"REFUSED: {pins_path}: no _meta block")
    block: dict[str, dict] = {}
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    for key in sorted(legs):
        if key not in doc:
            continue                                    # the pins do not admit this leg (no rung store, say): nothing to stamp
        pinned = str(doc[key])
        row, rel = legs[key]
        found = [Path(pr) / rel for pr in a.pulled_root if (Path(pr) / rel).is_dir()]
        stores = [(root, SP.read_store_engines(root)) for root in found]
        stores = [(root, s) for root, s in stores if s is not None]
        if not stores:
            if key in a.allow_missing:
                print(f"{key}: pinned {pinned[:8]} — NO pulled store under {a.pulled_root} at {rel} (allowed missing: no engine recorded)")
                continue
            raise SystemExit(f"REFUSED: {key}: pinned {pinned[:8]} but no pulled store at {rel} under {a.pulled_root} — pull it first, or --allow-missing {key}")
        if len(stores) > 1:
            raise SystemExit(f"REFUSED: {key}: {len(stores)} pulled stores at {rel} under {a.pulled_root} — one pull per leg")
        root, s = stores[0]
        tree = _producer_tree(root)
        if tree is None:
            raise SystemExit(f"REFUSED: {key}: the pulled store {root} names no producer tree (cells/*/manifest.json producer.tree_sha)")
        if not (tree == pinned or tree.startswith(pinned) or pinned.startswith(tree)):
            raise SystemExit(f"REFUSED: {key}: the pulled store {root} was produced at {tree[:8]}, the pins admit {pinned[:8]} — not this leg's admitted store")
        if len(s.engines) != 1:
            raise SystemExit(f"REFUSED: {key}: the pulled store names engines {s.engines} — " + s.describe())
        engine = SP.SERVING_ENGINE_LENS if s.engines[0] == SP.ENGINE_LENS else SP.SERVING_ENGINE_LORA_NATIVE
        # provenance: the pull's NAME + the store's path inside it (never an absolute path — a workstation's home is not the record's business)
        pulled = f"{Path([pr for pr in a.pulled_root if root.is_relative_to(pr)][0]).name}/{rel}"
        block[key] = {SP.PINS_ENGINE_FIELD: engine, "producer_tree_sha": tree, "pulled_store": pulled,
                      "readings": [[f, field, value] for f, field, value, _e in s.readings], "stamped": today}
        print(f"{key}: {engine} (tree {tree[:8]} == pin {pinned[:8]}; {len(s.readings)} readings from {root})")
    meta["schema"] = SCHEMA_BUMP.get(str(meta.get("schema")), str(meta.get("schema")))
    meta["serving_engine_source"] = SOURCE_NOTE
    meta[SP.PINS_ENGINE_BLOCK] = block
    if a.dry_run:
        print(json.dumps({SP.PINS_META_KEY: {k: meta[k] for k in ("schema", "serving_engine_source", SP.PINS_ENGINE_BLOCK)}}, indent=1))
        return 0
    pins_path.write_text(json.dumps(doc, indent=1, ensure_ascii=ensure_ascii) + "\n", encoding="utf-8")
    print(f"wrote {pins_path}: {len(block)} steer legs stamped; schema {meta['schema']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
