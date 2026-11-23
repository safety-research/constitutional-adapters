"""The committed render's battery cell states the committed battery sidecar's pending state, and nothing else.

The battery cell (`9d4a6f20` in paper/notebooks/registry_qwen_figures.ipynb) draws the distilled objects from the committed sidecar
paper/notebooks/data/qwen_si_target_battery_rows.json, the file paper/notebooks/si_target_battery_sidecar.py writes. While that file
lists the distilled objects' BigCodeBench and BFCL readings as `pending` (the pending-registration state), the cell prints a
`battery file state: pending registration` line, draws those readings as gap glyphs, names them on a `!! NOT DRAWN` line and rings
the capability means they leave partial. Once the file is final (no `pending` list) it prints none of these. The sync test holds the
notebook's sources and execution order to its generator, and the battery tests accept either state, so a final sidecar committed
without re-rendering the notebook, or a sidecar rewritten after the render, would pass both. This module holds the two committed files
to each other:

- the pending-state line, the `!! NOT DRAWN` line and the ringed-means line appear in the cell's output if and only if the sidecar
  lists pending registrations, and the cell's closing line names the same state
- the render counts and names exactly the sidecar's pending readings, and counts its drawn runs
- the sidecar is in a state the default render draws (final, or pending registration), and the render is not a provisional one

It reads both files as JSON and the state names from registry_nb.py's source, with no nbformat and no registry_nb import, so it runs
in the landing suite's environment. Its basename differs from every other test module's and does not start with `test_registry_`, the
prefix the registry-scoped scans of tests/ read. To fix a failure, regenerate the sidecar or re-render the notebook as
paper/notebooks/README.md says, so the committed render is the render of the committed sidecar."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "paper" / "notebooks"
NOTEBOOK = NOTEBOOKS / "registry_qwen_figures.ipynb"
SIDECAR = NOTEBOOKS / "data" / "qwen_si_target_battery_rows.json"
REGISTRY_NB = NOTEBOOKS / "registry_nb.py"
CELL = "9d4a6f20"
#: the line prefixes the battery cell prints (make_registry_notebook.py, the cell's source)
STATE_LINE = "battery file state: "
NOT_DRAWN = "!! NOT DRAWN: "
RINGED = "partial means (ringed): "
PROVISIONAL = "!! PROVISIONAL battery render"
#: the cell's closing line: drawn runs, objects, readings not yet in the data and the file's state
SUMMARY = re.compile(r"^the distilled objects: (\d+) battery runs of (\d+) objects from .*?; (\d+) reading\(s\) not yet in the data "
                     r"\(file state: ([^)]+)\); StrongREJECT threshold ")
#: the capability completeness check's last line: every pending capability reading by name, or `none`
PENDING_SET = re.compile(r"^capability pending set \((\d+)\): (.*)$")
STATE_COUNT = re.compile(r"the distilled objects' (\d+) ")
REBUILD = ("the committed render is not the render of the committed battery sidecar: regenerate the sidecar or re-render the "
           "notebook (paper/notebooks/README.md)")


def _constants(path: Path, names: tuple) -> dict:
    """The literal values of the module-level assignments `names` in `path`, read from its source (registry_nb is not imported)."""
    text = path.read_text(encoding="utf-8")
    out = {}
    for name in names:
        hits = re.findall(rf"^{name} = (.+)$", text, flags=re.M)
        assert len(hits) == 1, f"{path.name}: {len(hits)} module-level assignments of {name}"
        out[name] = ast.literal_eval(hits[0])
    return out


@pytest.fixture(scope="module")
def names() -> dict:
    return _constants(REGISTRY_NB, ("SITARGET_PENDING_REGISTRATION", "SITARGET_STATE_FINAL", "SITARGET_STATE_REGISTRATION",
                                    "SITARGET_STATE_PROVISIONAL"))


@pytest.fixture(scope="module")
def sidecar() -> dict:
    return json.loads(SIDECAR.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lines() -> list:
    """The battery cell's printed lines (its stdout streams, in order)."""
    cells = [c for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c.get("id") == CELL]
    assert len(cells) == 1, f"{NOTEBOOK.name}: {len(cells)} cells with id {CELL}"
    out = []
    for o in cells[0].get("outputs") or []:
        if o.get("output_type") == "stream" and o.get("name") == "stdout":
            text = o.get("text") or ""
            out += ("".join(text) if isinstance(text, list) else text).splitlines()
    return out


def _state(sidecar: dict, names: dict) -> str:
    """The sidecar's state as the default render reads it: final with no `pending` list, pending registration when the list is
    non-empty and holds readings of the pending-registration evals (BigCodeBench, BFCL) alone, else provisional. Whether the list is
    exactly every such reading is the sidecar contract's check and the render's (which printed its state); the pending set below ties
    the render's list to this file's."""
    if "pending" not in sidecar:
        return names["SITARGET_STATE_FINAL"]
    pend = sidecar["pending"]
    if isinstance(pend, list) and pend and all(isinstance(p, dict) and p.get("eval_id") in names["SITARGET_PENDING_REGISTRATION"]
                                               for p in pend):
        return names["SITARGET_STATE_REGISTRATION"]
    return names["SITARGET_STATE_PROVISIONAL"]


def _with(lines: list, prefix: str) -> list:
    return [line for line in lines if line.startswith(prefix)]


def _summary(lines: list):
    hits = [m for m in map(SUMMARY.match, lines) if m]
    assert len(hits) == 1, (f"cell {CELL} printed {len(hits)} closing lines ('the distilled objects: ...'); it did not draw the battery "
                            "(a RENDER POLICY render, an error, or a changed line)")
    return hits[0]


def test_the_sidecar_is_in_a_state_the_default_render_draws(sidecar, names):
    """A committed sidecar is final or pending registration: the default render refuses any other `pending` list."""
    state = _state(sidecar, names)
    assert state != names["SITARGET_STATE_PROVISIONAL"], (f"{SIDECAR.name} is {state}: its `pending` list is "
                                                          f"{sidecar.get('pending')!r:.300}")


def test_the_cell_drew_the_battery_in_the_default_render(lines):
    """The cell reached its closing line, and the committed render is not a provisional one."""
    _summary(lines)
    assert not _with(lines, PROVISIONAL), _with(lines, PROVISIONAL)


def test_the_pending_state_line_and_not_drawn_appear_iff_the_sidecar_lists_pending_registrations(lines, sidecar, names):
    pending = _state(sidecar, names) == names["SITARGET_STATE_REGISTRATION"]
    want = "lists" if pending else "does not list"
    state_lines = _with(lines, STATE_LINE + names["SITARGET_STATE_REGISTRATION"] + ":")
    assert bool(state_lines) == pending, (f"{SIDECAR.name} {want} pending registrations, but the cell printed {state_lines!r:.300}; "
                                          f"{REBUILD}")
    assert len(_with(lines, STATE_LINE)) == len(state_lines) <= 1, _with(lines, STATE_LINE)
    not_drawn = _with(lines, NOT_DRAWN)
    assert bool(not_drawn) == pending, f"{SIDECAR.name} {want} pending registrations, but the cell printed {not_drawn!r:.300}; {REBUILD}"
    ringed = _with(lines, RINGED)
    assert bool(ringed) == pending, f"{SIDECAR.name} {want} pending registrations, but the cell printed {ringed!r:.300}; {REBUILD}"
    state = _summary(lines).group(4)
    assert state == (names["SITARGET_STATE_REGISTRATION"] if pending else names["SITARGET_STATE_FINAL"]), (
        f"the cell's closing line names file state {state!r}; {SIDECAR.name} is {_state(sidecar, names)}; {REBUILD}")


def test_the_render_names_the_sidecars_pending_readings(lines, sidecar):
    """Every count of readings not yet in the data is the sidecar's, and the capability pending set is its `pending` list by name."""
    pend = list(sidecar.get("pending") or [])
    keys = {f"{p.get('slot')}/{p.get('seed')}/{p.get('eval_id')}" for p in pend}
    got = int(_summary(lines).group(3))
    assert got == len(pend), (f"the cell's closing line counts {got} reading(s) not yet in the data; {SIDECAR.name} lists {len(pend)}; "
                              f"{REBUILD}")
    for line in _with(lines, STATE_LINE):
        m = STATE_COUNT.search(line)
        assert m and int(m.group(1)) == len(pend), f"{line!r:.200}; {SIDECAR.name} lists {len(pend)}; {REBUILD}"
    hits = [m for m in map(PENDING_SET.match, lines) if m]
    assert len(hits) == 1, f"cell {CELL} printed {len(hits)} 'capability pending set' lines"
    n, named = int(hits[0].group(1)), hits[0].group(2)
    listed = set() if named == "none" else set(named.split(", "))
    assert n == len(listed) == len(keys) and listed == keys, (
        f"the render's pending set ({n}) and {SIDECAR.name}'s ({len(keys)}) differ: only in the render {sorted(listed - keys)[:6]}, "
        f"only in the file {sorted(keys - listed)[:6]}; {REBUILD}")


def test_the_render_counts_the_sidecars_drawn_runs(lines, sidecar):
    """The closing line's battery runs are the sidecar's distinct drawn runs (the cell anchors each drawn row once)."""
    drawn = {r.get("run_id") for r in sidecar.get("rows") or [] if r.get("role") == "drawn"}
    got = int(_summary(lines).group(1))
    assert got == len(drawn), f"the cell drew {got} battery runs; {SIDECAR.name} holds {len(drawn)} drawn runs; {REBUILD}"
