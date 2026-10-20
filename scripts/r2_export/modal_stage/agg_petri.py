"""Petri (alignment audit) per-cell aggregation for the full22 wave, every audited view. Reads cells.json (run key -> cell)
and each cell's results.json written by the Petri runner; nothing is recomputed from the eval logs.

Roots are ENUMERATED from ~/.local/state/petri/petri_full22_* (2026-09-14): the earlier hardcoded list of twelve view tags plus a
hand-written `_prompted` fold silently dropped every root whose name it did not know — the eight arm roots, the three llama
roots and the `_newcells` root (sixteen directories, several holding finished results) were invisible to the export with no
warning. A root that lands later now folds in with no code edit, and every root that is NOT read is named on stdout with the
reason.

Two kinds of root, told apart from the records rather than from a suffix list:
  * a VIEW: a root that audits its own unsteered anchor — its `none` cell has a record under cells/<key>/ for the key of any
    of its enumerations (cells.json or a cells.json.* snapshot). Output key = the tag (dir name minus `petri_full22_`). The
    base lanes, `_it`, `_halfdose`, `_shortconst` and the llama roots are views.
  * a SIDECAR: a root `petri_full22_<view>_<suffix>` that carries NO record of its own `none` — the drivers that write these
    (the `_prompted` / `_newcells` out-suffix legs, the `_arditi` / `_alphasteer` arm legs) deliberately never re-run the
    anchor and pair their cells against the VIEW's frozen `none`. Its canonical cells fold into the longest-prefix view that
    exists as a root; a suffix root whose prefix is not a root stands as a view of its own.
Sidecars fold in the order their enumerations were written (cells.json `written_at`), so an earlier record keeps its place
in the view (base, then `_prompted`, then `_newcells`, then the arms — the order the hand-written folds produced).
Per cell id the base root's record comes first, then the sidecars'; a record that has results.json beats an unaudited
placeholder, and two roots both holding results for one cell id are reported as a CONFLICT (the first still wins, loudly).

Provenance that the canonical schema does not carry (which root each cell was read from, every root's classification, every
skip) is written beside the output as <out>.roots.json and printed as the summary at the end.

usage: agg_petri.py [--state DIR] [--out FILE]      (no arguments = the canonical output path, as before)
"""
import argparse, glob, json, os, sys

PREFIX = "petri_full22_"
DEFAULT_STATE = os.path.expanduser("~/.local/state/petri")
DEFAULT_OUT = "$STAGE/sdfbase/fullscale/consolidated/petri_agg.json"   # under the staging directory; expanded after parsing


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_root(root, notes):
    """cells.json + its cells.json.* snapshots for one root; None (with a note) when the root cannot be read."""
    cj = f"{root}/cells.json"
    if not os.path.exists(cj):
        notes.append((os.path.basename(root), "no cells.json (never enumerated; %s)" % (", ".join(sorted(os.listdir(root))) or "empty")))
        return None
    try:
        cells = load_json(cj)
    except Exception as e:  # a half-written or corrupt record: skip, loudly
        notes.append((os.path.basename(root), f"cells.json unreadable: {type(e).__name__}: {e}"))
        return None
    if not isinstance(cells, dict) or not isinstance(cells.get("canonical"), list):
        notes.append((os.path.basename(root), "cells.json has no `canonical` list"))
        return None
    snaps = []
    for f in sorted(glob.glob(f"{root}/cells.json.*")):
        try:
            snaps.append(load_json(f))
        except Exception as e:
            print(f"WARNING {os.path.basename(root)}: snapshot {os.path.basename(f)} unreadable ({type(e).__name__}); ignoring that snapshot")
    return {"root": root, "name": os.path.basename(root), "tag": os.path.basename(root)[len(PREFIX):], "cells": cells, "snaps": snaps,
            "written_at": cells.get("written_at")}


def enumeration_keys(rec, c):
    """the run keys under which this cell may hold results in ITS root: the current enumeration's key first, then the keys the
    frozen snapshots gave the same cell id. A later enumeration can re-key a cell whose audit already ran under an earlier
    cells.json snapshot (Nemotron's five wave cells after its derived + prompted cells were added), so the run key is read
    from whichever enumeration actually has results, the current one first, then the snapshots."""
    return [c["petri_run_key"]] + [x["petri_run_key"] for sn in rec["snaps"] for x in sn.get("canonical", [])
                                   if x["cell_id"] == c["cell_id"] and x["petri_run_key"] != c["petri_run_key"]]


def resolve_key(rec, c):
    return next((k for k in enumeration_keys(rec, c) if os.path.exists(f"{rec['root']}/cells/{k}/results.json")), c["petri_run_key"])


def has_own_anchor(rec):
    """does this root hold any record of auditing its own `none` (under the key of any of its enumerations)?"""
    for c in rec["cells"]["canonical"]:
        if c["cell_id"] == "none":
            return any(os.path.isdir(f"{rec['root']}/cells/{k}") for k in enumeration_keys(rec, c))
    return False


def classify(recs):
    """tag -> ('view', None) | ('sidecar', view_tag). A root without its own anchor folds into the longest proper prefix tag
    that is itself a root (resolved through to a view when that prefix is a sidecar too)."""
    kind = {}
    for tag, rec in recs.items():
        if has_own_anchor(rec):
            kind[tag] = ("view", None); continue
        parts = tag.split("_")
        target = next(("_".join(parts[:i]) for i in range(len(parts) - 1, 0, -1) if "_".join(parts[:i]) in recs), None)
        kind[tag] = ("sidecar", target) if target else ("view", None)
    resolved = {}
    for tag in recs:
        k, tgt = kind[tag]; hops = 0
        while k == "sidecar" and kind[tgt][0] == "sidecar" and hops < len(recs):
            tgt = kind[tgt][1]; hops += 1
        resolved[tag] = (k, tgt)
    return resolved


def cell_record(rec, c):
    key = resolve_key(rec, c)
    croot = rec["root"]
    rf = f"{croot}/cells/{key}/results.json"
    inc = f"{croot}/cells/{key}/petri_incomplete.json"
    out = {"cell_id": c["cell_id"], "run_key": key, "kind": c["kind"], "cell_class": c["cell_class"],
           "complete": os.path.exists(rf), "incomplete_marker": os.path.exists(inc)}
    try:
        if os.path.exists(rf):
            r = load_json(rf)
            out["audited"] = r["audited"]; out["auditor_refused"] = r["auditor_refused"]
            out["auditor_chain"] = r["auditor_chain"]; out["unscored"] = r.get("unscored")
            out["behaviors"] = {b: {"audited": v["audited"], "mean": v["mean"], "n_items": v["n_items"],
                                    "incomplete": v.get("incomplete"), "unresolved": v.get("unresolved")}
                                for b, v in r["behaviors"].items()}
        elif os.path.exists(inc):
            out["incomplete"] = load_json(inc)
    except Exception as e:  # a results.json mid-write while an audit lands must not take the whole aggregate down
        print(f"WARNING {rec['name']}/{c['cell_id']}: {os.path.basename(rf if os.path.exists(rf) else inc)} unreadable "
              f"({type(e).__name__}: {e}); recording the cell as NOT complete")
        out = {**{k: out[k] for k in ("cell_id", "run_key", "kind", "cell_class", "incomplete_marker")}, "complete": False,
               "read_error": f"{type(e).__name__}: {e}"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=DEFAULT_STATE, help="the Petri state dir holding the petri_full22_* roots")
    ap.add_argument("--out", default=DEFAULT_OUT, help="petri_agg.json path (the canonical one by default; <out>.roots.json is written beside it)")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from site_paths import site_path   # an unset $STAGE ends the run by name
    args.out = str(site_path(args.out, "--out"))
    P = os.path.expanduser(args.state)

    entries = sorted(os.listdir(P))
    in_scope = [e for e in entries if e.startswith(PREFIX)]
    outside = [e for e in entries if not e.startswith(PREFIX)]
    skipped, recs = [], {}
    for name in in_scope:
        root = f"{P}/{name}"
        if not os.path.isdir(root):
            skipped.append((name, "not a directory")); continue
        rec = load_root(root, skipped)
        if rec is not None:
            recs[rec["tag"]] = rec

    kinds = classify(recs)
    views = {t: rec for t, rec in recs.items() if kinds[t][0] == "view"}
    sidecars_of = {t: [] for t in views}
    for t, (k, tgt) in kinds.items():
        if k == "sidecar":
            sidecars_of[tgt].append(recs[t])
    for lst in sidecars_of.values():  # chronological by enumeration record; name breaks ties (a record without written_at goes last)
        lst.sort(key=lambda r: (r["written_at"] is None, r["written_at"] or 0, r["name"]))

    out, provenance, conflicts, folds = {}, {}, [], {}
    for tag in sorted(views):
        base = views[tag]
        # candidates per cell id, in fold order: the base root's canonical cells, then each sidecar's
        cands = {}
        for rec in [base] + sidecars_of[tag]:
            for c in rec["cells"]["canonical"]:
                cands.setdefault(c["cell_id"], []).append((rec, c))
        view, prov = {}, {}
        for cid, lst in cands.items():
            with_results = [(rec, c) for rec, c in lst if os.path.exists(f"{rec['root']}/cells/{resolve_key(rec, c)}/results.json")]
            with_marker = [(rec, c) for rec, c in lst if os.path.exists(f"{rec['root']}/cells/{resolve_key(rec, c)}/petri_incomplete.json")]
            rec, c = (with_results or with_marker or lst)[0]
            if len(with_results) > 1:
                conflicts.append((tag, cid, [r["name"] for r, _ in with_results]))
            view[cid] = cell_record(rec, c)
            prov[cid] = {"root": rec["name"], "run_key": view[cid]["run_key"], "also_enumerated_in": [r["name"] for r, _ in lst if r is not rec]}
            if rec is not base:
                folds.setdefault(rec["name"], {"view": tag, "cells": []})["cells"].append(cid)
        for sc in sidecars_of[tag]:  # a sidecar every one of whose cells lost to another root still gets a line
            folds.setdefault(sc["name"], {"view": tag, "cells": []})
        # the scale-0 serving controls (none__lens0: the lens engaged at scale 0 — a no-op that measures the serving path's own
        # noise on the audit), kept OUT of `cells` so nothing that iterates the roster picks them up (2026-09-09). Read from the
        # view's own root only, as before: a sidecar's control leg is never run (it audits its arm cell alone).
        ctrl = {}
        for c in base["cells"].get("controls", []):
            rf = f"{base['root']}/cells/{c['petri_run_key']}/results.json"
            rec_ = {"cell_id": c["cell_id"], "control_of": c.get("control_of"), "run_key": c["petri_run_key"], "complete": os.path.exists(rf)}
            if os.path.exists(rf):
                try:
                    r = load_json(rf)
                    rec_["audited"] = r["audited"]; rec_["auditor_refused"] = r["auditor_refused"]
                    rec_["behaviors"] = {b: {"audited": v["audited"], "mean": v["mean"], "n_items": v["n_items"]} for b, v in r["behaviors"].items()}
                except Exception as e:
                    print(f"WARNING {base['name']}/{c['cell_id']} (control): results.json unreadable ({type(e).__name__}); recording as NOT complete")
                    rec_["complete"] = False; rec_["read_error"] = f"{type(e).__name__}: {e}"
            ctrl[c["cell_id"]] = rec_
        out[tag] = {"cells": view, "n_cells": len(view), "n_complete": sum(v["complete"] for v in view.values()), "controls": ctrl}
        provenance[tag] = {"root": base["name"], "sidecars": [s["name"] for s in sidecars_of[tag]], "cells": prov}

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    roots_path = out_path[:-len(".json")] + ".roots.json" if out_path.endswith(".json") else out_path + ".roots.json"
    json.dump({"state": P, "out": out_path, "views": provenance, "folded": folds, "skipped": [{"root": n, "reason": r} for n, r in skipped],
               "conflicts": [{"view": v, "cell_id": c, "roots_with_results": r} for v, c, r in conflicts],
               "outside_namespace": outside}, open(roots_path, "w"), indent=1)

    for tag, v in out.items():
        print("=====", tag, v["n_complete"], "/", v["n_cells"])
        for cid, rec in v["cells"].items():
            if rec["complete"]:
                m = {b: rec["behaviors"][b]["mean"][b] for b in rec["behaviors"]}
                print(f"  {cid:48s} audited={rec['audited']['items']}/{rec['audited']['expected']} refused={rec['auditor_refused']['total']} unscored={rec['unscored']} means={m}")
            else:
                print(f"  {cid:48s} INCOMPLETE marker={rec['incomplete_marker']} {json.dumps(rec.get('incomplete'))[:300]}")

    print("\n===== roots")
    print(f"seen: {len(entries)} entries under {P}; {len(in_scope)} in the {PREFIX}* namespace, {len(recs)} readable; "
          f"{len(outside)} outside it ignored ({', '.join(outside) if outside else 'none'})")
    print(f"views ({len(views)}):")
    for tag in sorted(views):
        sc = sidecars_of[tag]
        print(f"  {tag:32s} <- {views[tag]['name']}" + (f"  + folded {len(sc)}: " + ", ".join(s['name'] for s in sc) if sc else ""))
    print(f"folded ({len(folds)}):")
    for name in sorted(folds, key=lambda n: (folds[n]['view'], n)):
        f = folds[name]
        print(f"  {name:45s} -> {f['view']:24s} cells added: {', '.join(f['cells']) if f['cells'] else '(none: every cell id already held by an audited record)'}")
    print(f"skipped ({len(skipped)}):")
    for name, reason in skipped:
        print(f"  {name!r}: {reason}")
    print(f"conflicts ({len(conflicts)}):" + ("" if conflicts else " none"))
    for v, c, r in conflicts:
        print(f"  {v}/{c}: results.json in more than one root {r} — the first was used; two audits of one cell need a decision")
    print(f"wrote {out_path} ({len(out)} views, {sum(v['n_cells'] for v in out.values())} cells, "
          f"{sum(v['n_complete'] for v in out.values())} complete) and {roots_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
