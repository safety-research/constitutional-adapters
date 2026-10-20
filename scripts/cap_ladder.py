#!/usr/bin/env python
"""The standing truncation rule's CLI (``sisteer.capeval.tiers.cap_ladder``;
``docs/cap_ladder.md``): size a cap before a run, measure a store after a
pass, report the ladder's exits across rungs.

  size     the config-time half. From the measured completion-length arrays
           (``~/evals-geometry-measure/raw/cap__<family>__<task>.npy``, censored
           at the cap they were generated under) — or, with
           ``--from-length-quantiles``, from the sampled run's UNCENSORED
           ``length_quantiles.json`` (per cell, at-cap loops set aside the way
           the ladder's reclassify decision did; the task takes the largest
           recommendation across its cells) — name the smallest rung whose
           truncated share is at or below the threshold, per task, and write a
           ``cap_sizing/1`` JSON the drivers take as ``--cap-sizing`` (they
           refuse a cap below the recommendation). The task's CONFIGURED default
           is read from the harness (``--config`` -> ``controls_gen_config``;
           ``inspect_served.MAX_TOKENS_OFF`` for the Inspect tasks, which reads
           ``tinycap.contracts.MATH500_MAX_TOKENS_DEFAULT``) and compared — never
           copied here.
  measure  the check half, on a store the drivers did not measure (a store that
           predates the rule): re-tokenize each judged spool under
           ``carry_over.AT_CAP_RULE``, classify, write the sidecar + the
           ``cap_ladder.json`` record (no manifest change).
  report   the exit half: walk cell stores under the given roots, group their
           ``cap_ladder.json`` records by (cell, readout), order by cap, and
           write the report — every pair's exit and residual, the threshold
           exits that still carry a non-zero residual, the ceiling exits split
           into "climbed and ran out" and "at the ceiling from the outset".

    HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts .venv/bin/python scripts/cap_ladder.py size \\
        --family gemma4e4b --tasks xstest_v2,or_bench_hard_1k --out sizing.json [--config cfg.yaml]
    ... size --family glm46vflash --tasks xstest_v2,or_bench_hard_1k \\
        --from-length-quantiles <report dir>/length_quantiles.json [--cells a,b] [--trial 0] --out sizing.json
    ... measure --store <cell dir> [--tokenizer <repo or dir>] [--climbed]
    ... report <roots...> --out report.json [--md report.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts
REPO = Path(__file__).resolve().parents[1]

#: the judged tasks the sizing knows the measurement name of
DEFAULT_TASKS = ("xstest_v2", "or_bench_hard_1k")
SPOOL_OF = {"xstest": "gens_xstest_v2.jsonl", "or_bench_hard_1k": "gens_or_bench_hard_1k.jsonl"}


def _refuse(msg: str) -> "None":
    raise SystemExit(f"[cap-ladder] {msg}")


# --------------------------------------------------------------------------- #
# size                                                                         #
# --------------------------------------------------------------------------- #


def configured_cap_for(task: str, *, cfg: Optional[dict], thinking: str = "off") -> Optional[Dict[str, Any]]:
    """The cap the harness would run ``task`` at today, READ from the place
    that owns it: the judged controls' cap from the config
    (``cell_controls.controls_gen_config``), an Inspect task's from
    ``inspect_served.MAX_TOKENS_OFF/ON`` (which reads
    ``contracts.MATH500_MAX_TOKENS_DEFAULT``). None when neither owns it."""
    from sisteer.capeval.tiers import cap_ladder as cl

    judged = {v: k for k, v in cl.MEASURED_TASK_OF.items()}
    if task in judged:
        if cfg is None:
            return {"owner": "cell_controls.controls_gen_config(config)", "cap": None,
                    "note": "pass --config to read the configured judged cap"}
        from sisteer.eval.cell_controls import controls_gen_config
        return {"owner": "cell_controls.controls_gen_config(config)",
                "cap": int(controls_gen_config(cfg)["max_new_tokens"])}
    try:
        from sisteer.capeval.tiers import inspect_served as isv
    except ImportError:
        # this tree carries no Inspect legs (the tiny-capability lineage's module is
        # not here): no table owns an Inspect task's cap, so nothing is configured
        return None
    name = task.split("/", 1)[-1]
    table = isv.MAX_TOKENS_ON if thinking == "on" else isv.MAX_TOKENS_OFF
    if name in table:
        return {"owner": f"inspect_served.MAX_TOKENS_{'ON' if thinking == 'on' else 'OFF'}[{name!r}]",
                "cap": table[name]}
    return None


def _size_task_from_length_quantiles(path: str, *, lane: str, task: str, cells: str, trial: Optional[str],
                                     threshold: float, ceiling: int) -> Dict[str, Any]:
    """One task's sizing from the sampled run's report: the task's readout is
    the judged control that :data:`cap_ladder.MEASURED_TASK_OF` maps to it;
    every cell present (or ``--cells``) is sized and the task takes the
    LARGEST recommendation, since a run's cap must cover every cell it
    serves. A task with no readout in the report is refused, not skipped."""
    from sisteer.capeval.tiers import cap_ladder as cl

    readout_of = {v: k for k, v in cl.MEASURED_TASK_OF.items()}
    readout = readout_of.get(task)
    if readout is None:
        raise cl.CapLadderError(
            f"{task!r} is not a judged control's measured task ({sorted(readout_of)}); the length "
            "quantiles report has no readout for it — drop it from --tasks or size it from the arrays")
    names = [c.strip() for c in (cells or "").split(",") if c.strip()] \
        or cl.cells_in_length_quantiles(Path(path), lane, readout)
    if not names:
        raise cl.CapLadderError(f"length quantiles: no cells for {lane}|{readout} in {path}")
    per: Dict[str, Dict[str, Any]] = {}
    for c in names:
        per[c] = cl.size_cap_from_length_quantiles(Path(path), lane, readout, c, trial=trial,
                                                   threshold=threshold, ceiling=ceiling)
    worst = max(names, key=lambda c: (int(per[c]["recommended_cap"]), bool(per[c]["censored"]), c))
    s = dict(per[worst])
    s["task"] = task
    s["sized_on_cell"] = worst
    s["per_cell"] = {c: {k: per[c][k] for k in cl._PER_TRIAL_KEYS + ("trials", "pooled", "sized_on_trial")}
                     for c in names}
    return s


def cmd_size(args) -> None:
    from sisteer.capeval.tiers import cap_ladder as cl

    raw_dir = Path(os.path.expanduser(args.raw_dir))
    cfg = None
    if args.config:
        import jailbreak_run
        cfg = jailbreak_run._load_cfg(args.config)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out: Dict[str, Any] = {"schema": cl.SIZING_SCHEMA, "family": args.family,
                           "source": str(raw_dir), "threshold": args.threshold,
                           "ceiling": args.ceiling, "ladder": list(cl.CAP_LADDER),
                           "config": args.config or None, "tasks": {}}
    lq = args.from_length_quantiles
    if lq:
        out["source"] = str(Path(os.path.expanduser(lq)))
        out["source_kind"] = "length_quantiles"
        out["lane"] = args.lane or args.family
    for task in tasks:
        try:
            if lq:
                s = _size_task_from_length_quantiles(lq, lane=args.lane or args.family, task=task,
                                                     cells=args.cells, trial=args.trial,
                                                     threshold=args.threshold, ceiling=args.ceiling)
            else:
                s = cl.size_cap_from_measurement(args.family, task, raw_dir=raw_dir,
                                                 measured_cap=args.measured_cap,
                                                 threshold=args.threshold, ceiling=args.ceiling)
        except cl.CapLadderError as e:
            _refuse(str(e))
        conf = configured_cap_for(task, cfg=cfg)
        s["configured"] = conf
        if conf and conf.get("cap") is not None:
            s["configured_ok"] = int(conf["cap"]) >= int(s["recommended_cap"])
            s["configured_at_ceiling"] = int(conf["cap"]) >= int(args.ceiling)
        out["tasks"][task] = s
        flag = ("CENSORED" if s["censored"] else "resolvable")
        conf_s = (f"configured {conf['cap']} ({'ok' if s.get('configured_ok') else 'UNDER'}"
                  f"{'; at the ceiling: nothing above it' if s.get('configured_at_ceiling') else ''})"
                  if conf and conf.get("cap") is not None else "configured: n/a")
        cell_s = (f" [worst cell {s['sized_on_cell']} of {len(s['per_cell'])}]" if "per_cell" in s else "")
        print(f"[cap-ladder] size {args.family}/{task}{cell_s}: n={s['n']} measured cap {s['measured_cap']} "
              f"at-cap {s['share_at_measured_cap']:.2%} truncated "
              f"{s.get('share_truncated_at_measured_cap', s['share_at_measured_cap']):.2%} -> {flag}; "
              f"recommended {s['recommended_cap']}{' (ceiling)' if s['at_ceiling'] else ''}; {conf_s}")
    if args.out:
        p = Path(os.path.expanduser(args.out))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[cap-ladder] wrote {p}")
    else:
        print(json.dumps(out, indent=2, sort_keys=True))


# --------------------------------------------------------------------------- #
# measure                                                                      #
# --------------------------------------------------------------------------- #


def _tokenizer_for(manifest: dict, override: str):
    from sisteer.models import load_tokenizer

    if override:
        return load_tokenizer(os.path.expanduser(override)), override
    ident = manifest.get("serving_identity") or {}
    model_id = ident.get("resolved_model_id") or manifest.get("model_id") \
        or (manifest.get("row_constants") or {}).get("model_id")
    if not model_id:
        _refuse("the manifest names no model id; pass --tokenizer")
    if str(model_id).startswith("/"):
        model_id = Path(model_id).name.replace("--", "/", 1)
    rev = ident.get("requested_revision") or manifest.get("model_revision")
    return load_tokenizer(model_id, revision=rev), f"{model_id}@{rev}"


def cmd_measure(args) -> None:
    from sisteer.capeval.tiers import cap_ladder as cl
    from sisteer.capeval.tiers import judged

    store = Path(os.path.expanduser(args.store))
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    tok, tok_name = _tokenizer_for(manifest, args.tokenizer)
    cap = int((manifest.get("resolved_config") or {}).get("gen", {}).get("max_new_tokens", 0))
    climbed = bool(args.climbed or manifest.get("carry_over"))
    try:
        hops = cl.parse_hop_timeouts(args.hop_timeout_s, in_process=args.in_process)
    except cl.CapLadderError as e:
        _refuse(str(e))
    statuses: Dict[str, Any] = {}
    for control, name in SPOOL_OF.items():
        p = store / "private" / name
        if not p.is_file():
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])[judged._HEADER_KEY]
        spool = judged.GenSpool(p, header, create=False)
        entries = spool.entries()
        hcap = int(header["gen_cfg"]["max_new_tokens"])
        source_caps = None
        sidecar = cl.sidecar_path(p)
        if sidecar.is_file() and not args.recompute_caps:
            try:
                source_caps, _src = cl.source_caps_for(entries, sidecar, hcap)
            except cl.CapLadderError as e:
                _refuse(str(e))
        status, records = cl.measure_rung(entries, tok, hcap, source_caps=source_caps,
                                          threshold=args.threshold, ceiling=args.ceiling,
                                          climbed=climbed, hop_timeouts_s=hops)
        cl.write_sidecar(sidecar, records, header={
            "cell_id": manifest["cell_id"], "control": control, "task": header["task"],
            "cap": hcap, "items_sha": header.get("items_sha"),
            "intervention_hash": header.get("intervention_hash"), "tokenizer": tok_name})
        statuses[control] = status
        print(cl.status_line("[cap-ladder]", control, status))
    if not statuses:
        _refuse(f"{store}: no judged spool under private/ ({sorted(SPOOL_OF.values())})")
    rec = cl.cell_record(cell_id=manifest["cell_id"], engine=manifest.get("engine", "?"),
                         cap=cap, statuses=statuses, store_dir=store,
                         threshold=args.threshold, ceiling=args.ceiling,
                         extra={"climbed": climbed, "measured_by": "scripts/cap_ladder.py measure",
                                "tokenizer": tok_name})
    path = cl.write_cell_record(store, rec)
    print(f"[cap-ladder] wrote {path}")


# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #


def collect_rungs(roots: List[Path]) -> List[Dict[str, Any]]:
    """Every ``cap_ladder.json`` under the roots -> one rung record per
    (cell, control) it carries."""
    from sisteer.capeval.tiers import cap_ladder as cl

    out: List[Dict[str, Any]] = []
    seen = set()
    for root in roots:
        for p in sorted(Path(os.path.expanduser(str(root))).rglob(cl.RECORD_NAME)):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec.get("schema") != cl.RECORD_SCHEMA:
                _refuse(f"{p}: schema {rec.get('schema')!r} != {cl.RECORD_SCHEMA!r}")
            for control, st in (rec.get("controls") or {}).items():
                out.append({"cell_id": rec["cell_id"], "control": control,
                            "cap": int(st["cap"]), "status": st, "store_dir": str(p.parent),
                            "engine": rec.get("engine")})
    return out


def cmd_report(args) -> None:
    from sisteer.capeval.tiers import cap_ladder as cl

    rungs = collect_rungs([Path(r) for r in args.roots])
    if not rungs:
        _refuse(f"no {cl.RECORD_NAME} under {args.roots}")
    try:
        report = cl.ladder_report(rungs)
    except cl.CapLadderError as e:
        _refuse(str(e))
    report["roots"] = [str(r) for r in args.roots]
    report["n_records"] = len(rungs)
    md = cl.render_report_md(report)
    if args.out:
        p = Path(os.path.expanduser(args.out))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[cap-ladder] wrote {p}")
    if args.md:
        Path(os.path.expanduser(args.md)).write_text(md, encoding="utf-8")
        print(f"[cap-ladder] wrote {args.md}")
    print(md)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    from sisteer.capeval.tiers import cap_ladder as cl

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("size", help="size the cap from the measured length distributions")
    s.add_argument("--family", required=True, help="model family key (cap__<family>__<task>.npy)")
    s.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                   help="comma list of measured task names (default: the judged controls)")
    s.add_argument("--raw-dir", default=str(cl.DEFAULT_RAW_DIR))
    s.add_argument("--measured-cap", type=int, default=None,
                   help="the cap the arrays were clamped at (default: the meta note's)")
    s.add_argument("--threshold", type=float, default=cl.AT_CAP_SHARE_THRESHOLD)
    s.add_argument("--ceiling", type=int, default=cl.CAP_CEILING)
    s.add_argument("--config", default="",
                   help="a harness config: the configured judged cap is read from it "
                        "(controls_gen_config) and compared with the recommendation")
    s.add_argument("--from-length-quantiles", default="", metavar="PATH",
                   help="size from the sampled lane's length_quantiles.json instead of the arrays: "
                        "each task's readout (MEASURED_TASK_OF) is sized per cell with the report's "
                        "residual classification, and the task takes the largest recommendation")
    s.add_argument("--lane", default="", help="the report's lane name when it differs from --family")
    s.add_argument("--cells", default="", help="comma list of cells to size (default: every cell present)")
    s.add_argument("--trial", default=None, help="size one trial (default: pool every trial)")
    s.add_argument("--out", default="", help="write the cap_sizing/1 JSON here")
    s.set_defaults(fn=cmd_size)

    m = sub.add_parser("measure", help="measure a cell store's judged spools under the rule")
    m.add_argument("--store", required=True, help="a cell store dir (manifest.json + private/)")
    m.add_argument("--tokenizer", default="", help="override the manifest's model id")
    m.add_argument("--threshold", type=float, default=cl.AT_CAP_SHARE_THRESHOLD)
    m.add_argument("--ceiling", type=int, default=cl.CAP_CEILING)
    m.add_argument("--climbed", action="store_true", default=False,
                   help="this store was reached by climbing (default: read off the "
                        "manifest's carry_over block)")
    m.add_argument("--hop-timeout-s", action="append", default=None, metavar="NAME=SECONDS",
                   help="a transport hop's total timeout (repeatable, e.g. driver=7200 "
                        "proxy_upstream=7200); the SMALLEST binds; omit -> escalation is "
                        "refused as undeterminable")
    m.add_argument("--in-process", action="store_true", default=False,
                   help="an in-process (HF) leg with no transport hop: the guard passes")
    m.add_argument("--recompute-caps", action="store_true", default=False,
                   help="ignore an existing sidecar's per-item source caps (every item "
                        "at the header cap)")
    m.set_defaults(fn=cmd_measure)

    r = sub.add_parser("report", help="the exit report across rungs")
    r.add_argument("roots", nargs="+", help="dirs to walk for cap_ladder.json records")
    r.add_argument("--out", default="", help="write the report JSON here")
    r.add_argument("--md", default="", help="write the markdown here")
    r.set_defaults(fn=cmd_report)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
