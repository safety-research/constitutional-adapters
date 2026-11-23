"""Render the paper notebook under a judge policy (REGISTRY_NB_JUDGES) and compare the renders figure by figure.

    PY="$PWD/.venv/bin/python"     # from the repository root, after `uv sync --frozen --extra cpu --group notebook`
    OUT="$HOME/tmp/judge_compare"  # any scratch directory
    # the strength lanes' reads are the committed projection by default; REGISTRY_NB_STRENGTH_READS names other sidecars (README, "Choosing the judges")
    cd paper/notebooks
    "$PY" judge_compare.py render --name primary_live --out "$OUT/renders"                                  # first: reads the live frames
    "$PY" judge_compare.py render --name rejudge_live --judges rejudge --reuse-cache --out "$OUT/renders"   # the same frames, the re-judges' items
    "$PY" judge_compare.py render --name primary_snapshot --source snapshot --out "$OUT/renders"            # the filed judges on the committed frame
    "$PY" judge_compare.py render --name rejudge_t075 --judges rejudge --threshold 0.75 --reuse-cache --out "$OUT/renders"
    "$PY" judge_compare.py compare --renders "$OUT/renders" --base primary_live --other rejudge_live --figs "$OUT/figs"
    "$PY" judge_compare.py figures                                                                          # the derived figure map

`--judges` defaults to `primary`, the filed Claude judges, as a comparison tool's explicit baseline; the paper's own render, and the
notebook's default with REGISTRY_NB_JUDGES unset, is `rejudge` (registry_nb.PAPER_JUDGES), so pass `--judges rejudge` for it.
A render executes the committed notebook's cells unchanged in a kernel (the .ipynb on disk is never written), plus one appended cell
that pickles the frames the figures draw from (`frames.pkl`). The render sets REGISTRY_NB_JUDGES, REGISTRY_NB_SOURCE and
REGISTRY_NB_FIGS (its own figures directory), and clears REGISTRY_NB_SR_THRESHOLD, REGISTRY_NB_PETRI_JUDGE,
REGISTRY_NB_PETRI_EXCLUDE and NB_REUSE_CACHE from the inherited environment unless a flag passes them (`--threshold`,
`--petri-judge`, `--petri-exclude`, `--reuse-cache`), so a variable exported for another comparison never leaks into a render.
`render.json` records what the render set, what it cleared, the effective policy, threshold and Petri judge, NB_REUSE_CACHE and every
REGISTRY_NB_STRENGTH_READS file with its sha256, then the settings the kernel itself resolved.

`--reuse-cache` (NB_REUSE_CACHE=1) covers the FRAMES only: the view queries the first live render cached are read again, so every live
variant is drawn from ONE set of frames and the judge is the only thing that differs. The judges' item rows are always read at the live
head (`registry_nb.query(reuse=False)`), and compare.json records each render's frame heads and its item-row head per judge.

The figures compared come from the notebook's own figure map, never a list kept here: the paper number and caption from the
README's "Figure map" table, and the frames each figure draws from the call in `make_registry_notebook.py` that writes it (its
positional frame arguments, the frames its panel hooks read, and its layout's categories). A figure is LLM-judged when the rows it is
drawn from carry a judged rubric (`registry_nb.RUBRIC_FAMILY`), traced through the generator's assignments for a figure drawn from
derived points. A README figure the generator does not draw refuses rather than being skipped. `compare` reads two renders and writes,
under --figs, the side-by-side figures, the difference figures, and `compare.json` / `compare.md`: per figure the per-cell readings
under both judges, the ordering of objects within each model and category (Spearman, the best and worst object, the pairs that swap
with both intervals clear of each other), the intervals whose relation to zero changes and the contrasts whose sign flips, and the
level shift; then the mean-column verdict changes de-duplicated across figures, each with how far it clears zero. The registry is
opened read only and no model is called.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "registry_qwen_figures.ipynb")
README = os.path.join(HERE, "README.md")
GENERATOR = os.path.join(HERE, "make_registry_notebook.py")
HELPER = os.path.join(HERE, "registry_nb.py")

# the inherited variables a render clears unless a flag passes them, with the flag
RENDER_CLEARED = {"REGISTRY_NB_SR_THRESHOLD": "--threshold", "REGISTRY_NB_PETRI_JUDGE": "--petri-judge",
                  "REGISTRY_NB_PETRI_EXCLUDE": "--petri-exclude", "NB_REUSE_CACHE": "--reuse-cache"}
STRENGTH_READS_ENV = "REGISTRY_NB_STRENGTH_READS"
# what the dump keeps beside the figures' own frames: the comparison's other readers and the run's identity
DUMP_BASE = ("d", "cm", "d_all", "cm_all", "tr_d", "meta_mt", "meta_st", "pts_ab", "pts_ac", "pts_ad", "pts_ag", "_mt_clean", "op", "op_all",
             "HEAD", "OR_METRIC_OK")


class FigureMapError(RuntimeError):
    """The notebook's figure map names a figure the comparison cannot place."""


# ---- the figure map ---------------------------------------------------------------------------------------------------------------
def _read(path_or_text: str | None, default: str) -> str:
    if path_or_text is None:
        path_or_text = default
    if "\n" not in path_or_text and os.path.exists(os.path.expanduser(path_or_text)):
        with open(os.path.expanduser(path_or_text)) as f:
            return f.read()
    return path_or_text


NOTEBOOK_ONLY = "Notebook only"


def readme_figures(text: str) -> list:
    """The README's "Figure map" rows that name one figure file: [{paper, what, name}] in table order. A row is compared when its first
    cell is exactly a figure number ("Fig. 11", "Figure 11") or exactly `Notebook only` (a figure the notebook draws and the paper does
    not print); a row whose first cell says more ("Fig. 9, top (...)", "Notebook only (...)") is one the comparison leaves out, the
    figures drawn from committed data files with their own judge or none."""
    m = re.search(r"^##\s+Figure map\s*$(.*?)(?=^##\s|\Z)", text, flags=re.M | re.S)
    if not m:
        raise FigureMapError("the README has no '## Figure map' section")
    out = []
    for line in m.group(1).splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not (re.fullmatch(r"Fig(?:ure|\.)\s*\S+", cells[0]) or cells[0] == NOTEBOOK_ONLY):
            continue                                     # the header, the rule, a table row ("Tables 7 to 12", ...) or a row left out
        f = re.fullmatch(r"`([A-Za-z_][A-Za-z0-9_]*)`", cells[-1])
        if not f:
            raise FigureMapError(f"{cells[0]}: the README's File column ({cells[-1]!r}) names no single figure file")
        out.append(dict(paper=cells[0], what=cells[1], name=f.group(1)))
    if not out:
        raise FigureMapError("the README's figure map names no figure")
    return out


def _loads(node) -> list:
    """The names an expression reads, minus the names it binds itself (comprehension targets, lambda arguments)."""
    bound, loads = set(), []
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            (bound.add(n.id) if isinstance(n.ctx, ast.Store) else loads.append(n.id))
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
    return [x for x in dict.fromkeys(loads) if x not in bound]


def _targets(t) -> list:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        return [x for e in t.elts for x in _targets(e)]
    if isinstance(t, ast.Starred):
        return _targets(t.value)
    return []


def _dotted(f) -> str:
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f"{_dotted(f.value)}.{f.attr}"
    return ""


def _statements(body):
    """Every statement of a cell, into if / for / with / try blocks but never into a function or class body."""
    for s in body:
        yield s
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            yield from _statements(getattr(s, field, []) or [])
        for h in getattr(s, "handlers", []) or []:
            yield from _statements(h.body)


def parse_generator(text: str) -> dict:
    """The generator's code cells parsed: {cells: [(cell id, module)], parents: {name: [names its assignments read]},
    values: {name: the expression of its last single-name assignment}, calls: {name: [functions its assignments call]},
    functions: {name: FunctionDef}}. Only names the generator assigns count as parents (R, pd, np and builtins do not)."""
    cells = []
    for node in sorted((n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == "code" and len(n.args) >= 2), key=lambda n: (n.lineno, n.col_offset)):
        cid, src = node.args[0], node.args[1]
        if isinstance(src, ast.Constant) and isinstance(src.value, str):
            cells.append((cid.value if isinstance(cid, ast.Constant) else "?", ast.parse(src.value)))
    parents, values, calls, functions = {}, {}, {}, {}
    for _cid, mod in cells:
        for s in _statements(mod.body):
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[s.name] = s
                continue
            if isinstance(s, ast.Assign):
                targets, value = [x for t in s.targets for x in _targets(t)], s.value
                single = len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
            elif isinstance(s, ast.AnnAssign) and s.value is not None:
                targets, value, single = _targets(s.target), s.value, isinstance(s.target, ast.Name)
            elif isinstance(s, ast.AugAssign):
                targets, value, single = _targets(s.target), s.value, False
            else:
                continue
            reads = _loads(value) + (targets if isinstance(s, ast.AugAssign) else [])
            fns = [_dotted(c.func).rsplit(".", 1)[-1] for c in ast.walk(value) if isinstance(c, ast.Call)]
            for t in targets:
                parents.setdefault(t, [])
                parents[t] += [r for r in reads if r not in parents[t]]
                calls.setdefault(t, [])
                calls[t] += [f for f in fns if f not in calls[t]]
            if single:
                values[targets[0]] = value
    assigned = set(parents)
    parents = {k: [p for p in v if p in assigned and p != k] for k, v in parents.items()}
    return dict(cells=cells, parents=parents, values=values, calls=calls, functions=functions)


def _module_literals(text: str) -> dict:
    """{NAME: value} of the helper module's top-level assignments that are literals (NEGATIVE_DOSE_LAYOUT, ...)."""
    out = {}
    for s in ast.parse(text).body:
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
            try:
                out[s.targets[0].id] = ast.literal_eval(s.value)
            except (ValueError, TypeError, SyntaxError, RecursionError):
                pass
    return out


def _static(node, gen: dict, helper: dict, depth: int = 0):
    """A literal value of an argument: the literal itself, a generator name bound to one, or `R.<NAME>` of the helper module."""
    if node is None or depth > 4:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        pass
    if isinstance(node, ast.Name) and node.id in gen["values"]:
        return _static(gen["values"][node.id], gen, helper, depth + 1)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return helper.get(node.attr)
    return None


def _layout_categories(v) -> list | None:
    """The category keys of a panel layout, rows of panels ([["static", "padding"]], [["static"], ["multi-turn", "over-refusal"]]),
    or None when `v` is no layout (a flat list is an object or model roster)."""
    if isinstance(v, (list, tuple)) and v and all(isinstance(r, (list, tuple)) and r and all(isinstance(c, str) for c in r) for r in v):
        return list(dict.fromkeys(c for r in v for c in r))
    return None


#: the readers that build a gap table (one row per reading a panel lacks): the main figures' and the distillation-target battery's
GAP_TABLE_BUILDERS = ("gap_status", "sitarget_battery_gaps")


def _frame_args(node, gen: dict) -> list:
    """The frame variables an argument passes: a bare generator name, through list / tuple / dict literals and `dict(...)` (as the
    frontier passes its panels), never a constant (UPPER_CASE), a helper defined with `def` or a gap table (GAP_TABLE_BUILDERS), which
    marks what a panel lacks and is not a reading."""
    if isinstance(node, ast.Name):
        n = node.id
        if (n.isupper() or n in gen["functions"] or n not in gen["parents"]
                or any(b in gen["calls"].get(n, []) for b in GAP_TABLE_BUILDERS)):
            return []
        return [n]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [x for e in node.elts for x in _frame_args(e, gen)]
    if isinstance(node, ast.Dict):
        return [x for v in node.values for x in _frame_args(v, gen)]
    if isinstance(node, ast.Call) and _dotted(node.func) == "dict":
        return [x for k in node.keywords for x in _frame_args(k.value, gen)]
    return []


def _hook_frames(call, gen: dict) -> list:
    """The frames a figure's panel hooks read: every generator-assigned name a hook function (a `def` passed in a keyword) loads."""
    out = []
    for kw in call.keywords:
        vals = kw.value.values if isinstance(kw.value, ast.Dict) else [kw.value]
        for v in vals:
            if isinstance(v, ast.Name) and v.id in gen["functions"]:
                fn = gen["functions"][v.id]
                local = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
                local |= {n.id for s in fn.body for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                out += [n for s in fn.body for n in _loads(s) if n in gen["parents"] and n not in local and not n.isupper()]
    return list(dict.fromkeys(out))


def figure_map(readme: str | None = None, generator: str | None = None, helper: str | None = None) -> dict:
    """{figure file name: {paper, what, cell, function, frames, hook_frames, layout_categories}} for every figure of the README's map,
    in its order. `readme` / `generator` / `helper` are a path or the text itself (default: the files beside this one). Refuses
    (FigureMapError) when a README figure has no call in the generator, several, or a call that passes no frame."""
    rows = readme_figures(_read(readme, README))
    gen = parse_generator(_read(generator, GENERATOR))
    lits = _module_literals(_read(helper, HELPER))
    names = {r["name"] for r in rows}
    found = {}
    for cid, mod in gen["cells"]:
        for c in ast.walk(mod):
            if not isinstance(c, ast.Call):
                continue
            for a in list(c.args) + [k.value for k in c.keywords]:
                if isinstance(a, ast.Constant) and a.value in names:
                    found.setdefault(a.value, []).append((cid, c))
    out = {}
    for r in rows:
        calls = found.get(r["name"], [])
        if not calls:
            raise FigureMapError(f"{r['paper']} ({r['name']}): no call in the notebook generator draws it, so its frames are unknown")
        if len(calls) > 1:
            raise FigureMapError(f"{r['paper']} ({r['name']}): {len(calls)} calls in the generator name it "
                                 f"({', '.join(_dotted(c.func) for _, c in calls)}); the figure's frames are ambiguous")
        cid, call = calls[0]
        pos = [a for a in call.args if not (isinstance(a, ast.Constant) and a.value == r["name"])]
        frames = list(dict.fromkeys(x for a in pos for x in _frame_args(a, gen)))
        if not frames:
            raise FigureMapError(f"{r['paper']} ({r['name']}): its call ({_dotted(call.func)}) passes no frame the comparison can read")
        layout = next((lc for lc in (_layout_categories(_static(a, gen, lits)) for a in pos) if lc), None)
        out[r["name"]] = dict(paper=r["paper"], what=r["what"], cell=cid, function=_dotted(call.func), frames=frames,
                              hook_frames=[h for h in _hook_frames(call, gen) if h not in frames], layout_categories=layout)
    return out


def figures_not_in_map(figure_dirs, fmap: dict) -> list:
    """The figure files (PNG stems) a render wrote under any of `figure_dirs` that the README's figure map (`fmap`) does not name, so a
    comparison says which figures it left out rather than passing over them."""
    drawn = set()
    for d in figure_dirs:
        if d and os.path.isdir(d):
            drawn |= {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".png")}
    return sorted(drawn - set(fmap))


def lineage(generator: str | None = None) -> dict:
    """{name: [the generator names its assignments read]}: where a derived frame comes from."""
    return parse_generator(_read(generator, GENERATOR))["parents"]


def upstream(names, parents: dict) -> list:
    """Every name `names` are derived from, nearest first."""
    seen, out, todo = set(names), [], list(names)
    while todo:
        nxt = []
        for n in todo:
            for p in parents.get(n, []):
                if p not in seen:
                    seen.add(p)
                    out.append(p)
                    nxt.append(p)
        todo = nxt
    return out


def dump_cell(fmap: dict | None = None, parents: dict | None = None) -> str:
    """The cell appended to a render: pickles every frame the figure map draws from (plus DUMP_BASE), the judged rubrics of every frame
    upstream of them (FRAME_RUBRICS: None for a value with no rubric column, absent for a name the kernel does not hold), the judge
    reports and the settings the kernel resolved, which it also writes beside the pickle as settings.json."""
    fmap = figure_map() if fmap is None else fmap
    parents = lineage() if parents is None else parents
    drawn = list(dict.fromkeys(f for e in fmap.values() for f in e["frames"] + e["hook_frames"]))
    up = [n for n in upstream(drawn, parents) if n not in drawn]
    return f'''
import json as _js, pickle as _pk
_KEEP_TYPES = (pd.DataFrame, dict, list, tuple, str, int, float, bool, type(None), np.integer, np.floating, np.bool_)
_keep = {{}}
for _n in {json.dumps(list(dict.fromkeys(list(DUMP_BASE) + drawn)))}:
    if _n in globals() and isinstance(globals()[_n], _KEEP_TYPES):
        _keep[_n] = globals()[_n]
_rub = {{}}
for _n in {json.dumps(drawn + up)}:
    if _n in globals():
        _v = globals()[_n]
        _rub[_n] = (sorted(str(x) for x in _v["judge_rubric_name"].dropna().unique())
                    if isinstance(_v, pd.DataFrame) and "judge_rubric_name" in _v.columns else None)
_keep["FRAME_RUBRICS"] = _rub
_keep["JUDGE_REPORT"] = R.JUDGE_REPORT
_keep["PETRI_JUDGE_REPORT"] = R.PETRI_JUDGE_REPORT
_keep["POOLED_CI_STATE"] = R.pooled_ci_state()
_keep["JUDGES"] = R.JUDGES
_keep["SOURCE"] = R.SOURCE
_keep["SETTINGS"] = {{_k: getattr(R, _k, None) for _k in ("JUDGES", "SOURCE", "SR_THRESHOLD", "SR_PAPER_THRESHOLD", "PETRI_JUDGE", "PETRI_EXCLUDE")}}
_keep["SETTINGS"].update(HEAD=globals().get("HEAD"), NB_REUSE_CACHE=os.environ.get("NB_REUSE_CACHE"),
                         STRENGTH_READS=os.environ.get("{STRENGTH_READS_ENV}"))
with open(os.environ["JUDGE_COMPARE_DUMP"], "wb") as _f:
    _pk.dump(_keep, _f)
with open(os.path.join(os.path.dirname(os.environ["JUDGE_COMPARE_DUMP"]), "settings.json"), "w") as _f:
    _js.dump(_keep["SETTINGS"], _f, indent=1, default=str)
print("dumped", sorted(_keep))
'''


# ---- render -----------------------------------------------------------------------------------------------------------------------
def _sha256(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def render_env(args, out: str, environ=None) -> tuple:
    """(the variables a render sets, the inherited ones it clears with their inherited values, its provenance for render.json).
    Pure: reads `environ` (default os.environ) and changes nothing."""
    environ = os.environ if environ is None else environ
    env = {"REGISTRY_NB_SOURCE": args.source, "REGISTRY_NB_FIGS": os.path.join(out, "figures"), "REGISTRY_NB_JUDGES": args.judges,
           "JUDGE_COMPARE_DUMP": os.path.join(out, "frames.pkl"), "MPLBACKEND": "Agg"}
    flags = {"REGISTRY_NB_SR_THRESHOLD": None if getattr(args, "threshold", None) in (None, "") else str(float(args.threshold)),
             "REGISTRY_NB_PETRI_JUDGE": getattr(args, "petri_judge", None) or None,
             "REGISTRY_NB_PETRI_EXCLUDE": os.path.expanduser(args.petri_exclude) if getattr(args, "petri_exclude", None) else None,
             "NB_REUSE_CACHE": "1" if getattr(args, "reuse_cache", False) else None}
    env.update({k: v for k, v in flags.items() if v is not None})
    cleared = {k: environ[k] for k, v in flags.items() if v is None and k in environ}
    reads = environ.get(STRENGTH_READS_ENV)
    prov = dict(REGISTRY_NB_JUDGES=args.judges, REGISTRY_NB_SOURCE=args.source,
                REGISTRY_NB_SR_THRESHOLD=flags["REGISTRY_NB_SR_THRESHOLD"] or "unset: the paper threshold (registry_nb.SR_PAPER_THRESHOLD)",
                REGISTRY_NB_PETRI_JUDGE=flags["REGISTRY_NB_PETRI_JUDGE"] or "unset: the policy's Petri judge",
                REGISTRY_NB_PETRI_EXCLUDE=flags["REGISTRY_NB_PETRI_EXCLUDE"] or "unset: no exclusions",
                NB_REUSE_CACHE=bool(flags["NB_REUSE_CACHE"]),
                NB_CACHE=environ.get("NB_CACHE"),
                REGISTRY_NB_STRENGTH_READS=None if not reads else
                [dict(path=p, sha256=_sha256(os.path.expanduser(p))) for p in reads.split(os.pathsep) if p])
    return env, cleared, prov


def render(args) -> int:
    import nbformat
    from nbclient import NotebookClient
    out = os.path.join(os.path.expanduser(args.out), args.name)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    env, cleared, prov = render_env(args, out)
    for k in cleared:
        os.environ.pop(k, None)
    os.environ.update(env)
    nb = nbformat.read(NB, as_version=4)                      # the committed notebook's cells; this file is never written
    nb.cells.append(nbformat.v4.new_code_cell(dump_cell()))
    t = time.time()
    client = NotebookClient(nb, timeout=7200, kernel_name="python3", allow_errors=True, resources={"metadata": {"path": HERE}})
    client.execute()
    nbformat.write(nb, os.path.join(out, "registry_qwen_figures.executed.ipynb"))
    errs = [(c.get("id"), o.get("ename"), str(o.get("evalue"))[:400]) for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
            if o.get("output_type") == "error"]
    resolved = None
    if os.path.exists(os.path.join(out, "settings.json")):
        with open(os.path.join(out, "settings.json")) as f:
            resolved = json.load(f)
    meta = dict(name=args.name, env=env, cleared=cleared, provenance=prov, resolved=resolved, seconds=round(time.time() - t), errors=errs)
    with open(os.path.join(out, "render.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    return 0 if not errs else 3


# ---- shared readers -----------------------------------------------------------------------------------------------------------------
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


def print_figures(args) -> int:
    for name, e in figure_map(args.readme, args.generator).items():
        print(f"{e['paper']:<8} {name:<36} {e['function']:<26} frames {', '.join(e['frames'])}"
              + (f" · hooks {', '.join(e['hook_frames'])}" if e["hook_frames"] else "")
              + (f" · layout {', '.join(e['layout_categories'])}" if e["layout_categories"] else ""))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--name", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--source", default="live", choices=("live", "snapshot"))
    r.add_argument("--judges", default="primary")
    r.add_argument("--threshold", default=None, type=float, help="REGISTRY_NB_SR_THRESHOLD (default: cleared, the paper threshold)")
    r.add_argument("--petri-judge", default=None, help="REGISTRY_NB_PETRI_JUDGE (default: cleared, the policy's Petri judge)")
    r.add_argument("--petri-exclude", default=None, help="REGISTRY_NB_PETRI_EXCLUDE (default: cleared)")
    r.add_argument("--reuse-cache", action="store_true", help="NB_REUSE_CACHE=1: reuse the cached FRAMES (item rows are read live)")
    c = sub.add_parser("compare")
    c.add_argument("--renders", required=True)
    c.add_argument("--base", default="primary_live")
    c.add_argument("--other", default="rejudge_live")
    c.add_argument("--figs", required=True)
    c.add_argument("--readme", default=None, help="the figure map's README (default: the one beside this file)")
    c.add_argument("--generator", default=None, help="the notebook generator (default: the one beside this file)")
    f = sub.add_parser("figures")
    f.add_argument("--readme", default=None)
    f.add_argument("--generator", default=None)
    a = ap.parse_args()
    if a.cmd == "compare":
        sys.path.insert(0, HERE)
        from judge_analysis import compare
        sys.exit(compare(a))
    if a.cmd == "figures":
        sys.exit(print_figures(a))
    sys.exit(render(a))
