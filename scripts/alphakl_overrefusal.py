"""Behavioral over-refusal generation pass for the alphakl ladder objects.

User request 2026-08-19: measure the three json_format ladder objects'
behavioral over-refusal on XSTest (250 safe / 200 contrast-unsafe) and
OR-Bench-hard-1k (1,319 seemingly-toxic-but-benign). This script is the
GPU half: it generates responses for every (dataset x arm) cell and
writes them as JSONL on shared storage; grading happens off-cluster with
the Sonnet judge (scripts/judge_overrefusal.py), because these objects
push outputs toward JSON and prefix string-matching would misgrade
wrapped refusals.

Arms: baseline (unsteered, no SI), prompted (the SI as a system prompt),
and each requested object at eval-grid factors (1.0 = the KL-certified
operating point, 2.0 = the frontier's best-compliance point).

Datasets are read DIRECTLY from the shared processed cache's arrow files
(offline; with datasets>=5 the library's offline path on offline compute
nodes has burned this campaign four times).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

#: the two datasets' Arrow files in the Hugging Face cache ($HF_HOME through sisteer.site: HF_HOME, else SISTEER_HF_HOME, else
#: the cluster block's CLUSTER_HF_HOME), expanded where they are read
XSTEST_ARROW = ("$HF_HOME/datasets/"
                "natolambert___xstest-v2-copy/default/0.0.0/*/"
                "xstest-v2-copy-prompts.arrow")
ORBENCH_ARROW = ("$HF_HOME/datasets/"
                 "bench-llm___or-bench/or-bench-hard-1k/0.0.0/*/"
                 "or-bench-train.arrow")


def read_arrow(pattern: str):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no arrow file matches {pattern}")
    try:
        return ipc.open_file(paths[0]).read_all()
    except pa.lib.ArrowInvalid:
        with open(paths[0], "rb") as f:
            return ipc.open_stream(f).read_all()


def xstest_rows(table) -> list[dict]:
    """XSTest v2: `type` starting with contrast_ marks the UNSAFE half;
    everything else is the safe (over-refusal) half."""
    out = []
    for t, p in zip(table.column("type").to_pylist(),
                    table.column("prompt").to_pylist()):
        subset = "unsafe" if t.startswith("contrast") else "safe"
        out.append({"dataset": "xstest", "subset": subset, "kind": t,
                    "prompt": p})
    return out


def orbench_rows(table) -> list[dict]:
    return [{"dataset": "orbench_hard", "subset": "safe", "kind": c,
             "prompt": p}
            for p, c in zip(table.column("prompt").to_pylist(),
                            table.column("category").to_pylist())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--si", default="json_format")
    ap.add_argument("--arms", default="alphakl_null,alphakl_soft,alphakl_fisher")
    ap.add_argument("--scales", default="1.0,2.0")
    ap.add_argument("--out", default=None,
                    help="default: <out_root>/overrefusal/<si>")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    import yaml

    from sisteer.data.instructions import resolve_si
    from sisteer.generate import generate_with_object
    from sisteer.methods import load_object, object_artifact_name
    from sisteer.models import build_chat_prompt, load_model_and_tokenizer

    cfg = yaml.safe_load(open(args.config))
    # cfg["out_dir"] is the canonical (absolute) IFEval cell root and already
    # ends in "/ifeval" (stage2_run.py: cells live at out_dir/<si>/<arm>/,
    # sibling suites at out_dir.parent/<suite>). Match it exactly — a
    # relative guess resolves against the job's repo CWD, not the shared
    # out tree, and misses the objects (job 147913).
    ifeval_root = Path(cfg["out_dir"])
    out_dir = (Path(args.out) if args.out
               else ifeval_root.parent / "overrefusal" / args.si)
    out_dir.mkdir(parents=True, exist_ok=True)

    from sisteer import site as _site
    rows = xstest_rows(read_arrow(_site.expand(XSTEST_ARROW, key="XSTEST_ARROW"))) + orbench_rows(
        read_arrow(_site.expand(ORBENCH_ARROW, key="ORBENCH_ARROW")))
    print(f"[orb] {len(rows)} prompts "
          f"(xstest {sum(r['dataset'] == 'xstest' for r in rows)}, "
          f"orbench_hard {sum(r['dataset'] == 'orbench_hard' for r in rows)})")

    model, tok = load_model_and_tokenizer(
        cfg["model"], dtype=cfg.get("dtype"),
        revision=cfg.get("model_revision"))
    instruction = resolve_si(args.si)
    plain = [build_chat_prompt(tok, r["prompt"], system=None) for r in rows]
    with_si = [build_chat_prompt(tok, r["prompt"], system=instruction.text)
               for r in rows]

    def publish(name: str, responses: list[str], arm: str, scale: float):
        path = out_dir / f"{name}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for r, resp in zip(rows, responses):
                f.write(json.dumps({**r, "arm": arm, "scale": scale,
                                    "response": resp}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # existence == fully published
        print(f"[orb] wrote {path} ({len(responses)} rows)")

    gen = dict(batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)

    if not (out_dir / "baseline.jsonl").exists():   # resume unit = one cell
        publish("baseline",
                generate_with_object(model, tok, plain, obj=None, **gen),
                "baseline", 0.0)
    if not (out_dir / "prompted.jsonl").exists():
        publish("prompted",
                generate_with_object(model, tok, with_si, obj=None, **gen),
                "prompted", 0.0)

    for arm in args.arms.split(","):
        obj_path = ifeval_root / args.si / arm / object_artifact_name(arm)
        obj = load_object(obj_path, model, tok)
        for scale in (float(s) for s in args.scales.split(",")):
            name = f"{arm}_x{scale:g}"
            if (out_dir / f"{name}.jsonl").exists():
                print(f"[orb] {name} exists, skipping")
                continue
            publish(name,
                    generate_with_object(model, tok, plain, obj=obj,
                                         scale=scale, **gen),
                    arm, scale)
    print("[orb] done")


if __name__ == "__main__":
    main()
