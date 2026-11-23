"""Render the paper notebook's Petri figures under two Petri judges and compare them.

    PY=.venv/bin/python                                   # from the repository root, after `uv sync --frozen --extra cpu --group notebook`
    cd paper/notebooks
    # 1. the transcripts the Opus judge chain refused or scored by a fallback (from the re-judge's enumeration, read only)
    $PY petri_judge_compare.py exclusions --enum ENUM_DIR --scope SCOPE_DIR --out OUT/excl   # the re-judge's enumeration and scope directories
    # 2. one render per variant (the committed notebook's cells, executed in a kernel; the committed file is never written)
    $PY petri_judge_compare.py render --name opus_live --out OUT/renders                     # first: reads the live frames
    $PY petri_judge_compare.py render --name glm_live --judge glm-5.3 --reuse-cache --out OUT/renders
    # optional, for the comparison's sensitivity section (the names are the ones `compare` reads):
    #   opus_snapshot (--source snapshot), opus_rederived (--judge claude-opus-5), glm_matched (--judge glm-5.3 --exclude OUT/excl/refused.tsv),
    #   glm_opus5only / opus_opus5only (--judge glm-5.3 / primary, --exclude OUT/excl/refused_or_fallback.tsv), all with --reuse-cache
    # 3. the comparison: side-by-side figures, difference panels, the numbers the report quotes
    $PY petri_judge_compare.py compare --renders OUT/renders --figs OUT/figs --served OUT/excl/served.tsv

The judge is the loader's parameter (`registry_nb.select_petri_judge`, REGISTRY_NB_PETRI_JUDGE / REGISTRY_NB_PETRI_EXCLUDE); a render
sets the environment (REGISTRY_NB_JUDGES=primary, so the Petri judge is the only judge that varies) and executes the committed
notebook's cells unchanged, plus one appended cell that pickles the frames the Petri figures draw from. `--reuse-cache`
(NB_REUSE_CACHE=1) reads the frames the first live render cached, so every live variant is drawn at ONE registry head and the judge is
the only thing that differs between them. No model API is called; the registry is opened read only.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "registry_qwen_figures.ipynb")
PETRI_FIGURES = ("fig5_misalignment", "figA_misalignment_models", "figA_transfer_models")
PETRI_TABLES = ("tab:qwen-petri-seeds", "tab:allmodels-Petri", "tab:allmodels-absolute-petri", "tab:allmodels-transfer")
DUMP_CELL = r'''
import pickle as _pk
_keep = {}
for _n in ("d", "cm", "d_all", "cm_all", "tr_d", "tr_pe", "tr_cat", "tr_pairs", "HEAD"):
    if _n in globals():
        _keep[_n] = globals()[_n]
_keep["PETRI_JUDGE_REPORT"] = R.PETRI_JUDGE_REPORT
_keep["POOLED_CI_STATE"] = R.pooled_ci_state()
_keep["PETRI_JUDGE"] = R.PETRI_JUDGE
_keep["PETRI_EXCLUDE"] = R.PETRI_EXCLUDE
_keep["SOURCE"] = R.SOURCE
with open(os.environ["PETRI_COMPARE_DUMP"], "wb") as _f:
    _pk.dump(_keep, _f)
print("dumped", sorted(_keep))
'''


# ---- 1. exclusions ------------------------------------------------------------------------------------------------------------
def exclusions(args) -> int:
    """Per registry Petri item: who served the Opus chain's stored verdict (claude-opus-5, the claude-opus-4-8 / grok-4.6 refusal
    fallback, or nobody: a terminal refusal). Writes served.tsv (every item), refused.tsv and refused_or_fallback.tsv (exclusion files
    in REGISTRY_NB_PETRI_EXCLUDE's form)."""
    units = {}
    with gzip.open(os.path.join(args.scope, "units.tsv.gz"), "rt") as fh:
        cols = next(fh).rstrip("\n").split("\t")
        for line in fh:
            u = dict(zip(cols, line.rstrip("\n").split("\t")))
            units.setdefault(u["phys"], []).append(u)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    with gzip.open(os.path.join(args.enum, "inputs.jsonl.gz"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            cl = r.get("claude") or {}
            sc = cl.get("scores") or {}
            refused = (not sc) or sc.get(r["own_dimension"]) is None or bool(cl.get("refusal"))
            served = "refused" if refused else str(cl.get("served") or "unknown")
            for u in units.get(r["phys"], []):
                rows.append((u["run_id"], u["item_id"], u["trial_index"], served, r["behavior"], r["phys"]))
    hdr = "run_id\titem_id\ttrial_index\tserved\tbehavior\ttranscript\n"
    with open(os.path.join(args.out, "served.tsv"), "w") as f:
        f.write(hdr + "".join("\t".join(x) + "\n" for x in rows))
    for name, keep in (("refused", lambda s: s == "refused"), ("refused_or_fallback", lambda s: s == "refused" or "claude-opus-5" not in s)):
        with open(os.path.join(args.out, f"{name}.tsv"), "w") as f:
            f.write("run_id\titem_id\ttrial_index\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c, s, _, _ in rows if keep(s)))
    from collections import Counter
    print(json.dumps(dict(Counter(s for _, _, _, s, _, _ in rows)), indent=1))
    return 0


# ---- 2. render ----------------------------------------------------------------------------------------------------------------
#: inherited variables a render clears unless its own flag sets them (the comparison varies the Petri judge alone)
RENDER_CLEARED = ("REGISTRY_NB_SR_THRESHOLD", "NB_REUSE_CACHE")


def render_env(args, out: str, environ=None) -> tuple:
    """(the variables a render sets, the inherited ones it clears with their inherited values). Pure: reads `environ` (default
    os.environ) and changes nothing. REGISTRY_NB_JUDGES=primary: the Petri judge is the only judge this comparison varies, so every other
    reading stays on its filed judge whatever policy the shell exported (an inherited `rejudge` would draw GLM on the Opus side), and an
    inherited REGISTRY_NB_SR_THRESHOLD or NB_REUSE_CACHE is cleared (`--reuse-cache` sets the latter), as `judge_compare.render_env`
    clears them."""
    environ = os.environ if environ is None else environ
    env = {"REGISTRY_NB_SOURCE": args.source, "REGISTRY_NB_FIGS": os.path.join(out, "figures"), "REGISTRY_NB_JUDGES": "primary",
           "REGISTRY_NB_PETRI_JUDGE": args.judge, "REGISTRY_NB_PETRI_EXCLUDE": os.path.expanduser(args.exclude or ""),
           "PETRI_COMPARE_DUMP": os.path.join(out, "frames.pkl"), "MPLBACKEND": "Agg"}
    if getattr(args, "reuse_cache", False):
        env["NB_REUSE_CACHE"] = "1"
    cleared = {k: environ[k] for k in RENDER_CLEARED if k not in env and k in environ}
    return env, cleared


def render(args) -> int:
    import nbformat
    from nbclient import NotebookClient
    out = os.path.join(os.path.expanduser(args.out), args.name)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    env, cleared = render_env(args, out)
    for k in cleared:
        os.environ.pop(k, None)
    os.environ.update(env)
    nb = nbformat.read(NB, as_version=4)                      # the committed notebook's cells; this file is never written
    nb.cells.append(nbformat.v4.new_code_cell(DUMP_CELL))
    t = time.time()
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=True, resources={"metadata": {"path": HERE}})
    client.execute()
    nbformat.write(nb, os.path.join(out, "registry_qwen_figures.executed.ipynb"))
    errs = [(c.get("id"), o.get("ename"), o.get("evalue")) for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
            if o.get("output_type") == "error"]
    meta = dict(name=args.name, env=env, cleared=cleared, seconds=round(time.time() - t), errors=errs)
    json.dump(meta, open(os.path.join(out, "render.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    return 0 if not errs else 3


# ---- shared readers -----------------------------------------------------------------------------------------------------------
def cell_text(nb, cid: str) -> str:
    c = next((c for c in nb.cells if c.get("id") == cid), None)
    if c is None:
        return ""
    parts = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append(o.get("text", ""))
        elif "data" in o and "text/plain" in o["data"]:
            parts.append(o["data"]["text/plain"])
    return "".join(parts)


def latex_tables(text: str) -> dict:
    """{label: the LaTeX table} of every \\begin{table} ... \\end{table} block in a cell's text."""
    out = {}
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", text, flags=re.S):
        lab = re.search(r"\\label\{([^}]*)\}", m.group(0))
        if lab:
            out[lab.group(1)] = m.group(0)
    return out


def all_tables(nb) -> dict:
    out = {}
    for c in nb.cells:
        if c.cell_type == "code":
            out.update(latex_tables(cell_text(nb, c.get("id"))))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exclusions")
    e.add_argument("--enum", required=True)
    e.add_argument("--scope", required=True)
    e.add_argument("--out", required=True)
    r = sub.add_parser("render")
    r.add_argument("--name", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--source", default="live", choices=("live", "snapshot"))
    r.add_argument("--judge", default="primary")
    r.add_argument("--exclude", default="")
    r.add_argument("--reuse-cache", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--renders", required=True)
    c.add_argument("--figs", required=True)
    c.add_argument("--out-json", default="")
    c.add_argument("--served", default="", help="served.tsv from `exclusions`: adds the item-level comparison")
    a = ap.parse_args()
    if a.cmd == "compare":
        sys.path.insert(0, HERE)
        from petri_judge_analysis import compare
        sys.exit(compare(a))
    sys.exit({"exclusions": exclusions, "render": render}[a.cmd](a))
