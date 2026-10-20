"""The s2rl Petri overflow re-run died at the frozen-enumeration check because scripts/petri_run.py's `_STEERED_AUDIT_RE`
(`_s[0-9]+r$`, suffix-anchored on the AUDIT id) was a copy of the steered-view grammar the S2RL widening (ece5402fd) missed:
`steered_wave_audit()` read `petri_full22_<key>_s2rl` as a u0 / legacy audit, so the guard demanded the complete baseline and
refused the proxy's explicit two-cell declaration. The same suffix-anchored copy sat in scripts/registry/emit_contrasts.py (`_U0_VIEW_RE`).

Pins: (1) the classification — True for the s2rl audit id, unchanged True for _s2r / _s3r, False for _u0; (2) the frozen enumeration of an
s2rl audit under an explicit two-cell declaration RESTRICTS to those two legs, no controls leg, no refusal (the real writer's records, as the
serving-path module's s2r case); (3) a SWEEP over every regex literal in the tree whose body is `s[0-9]+r` / `s\\d+r` in ANY anchoring: each
must accept s2rl wherever it accepts s2r — so the next copy fails this test, not a wave."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

import petri_run as prun  # noqa: E402
import test_petri_store_shape as shape  # noqa: E402
from test_petri_run_serving_path import (  # noqa: E402,F401  (the stage-1 world fixture rides along)
    A32, CELLS, LORA, NATIVE_IDENTITY, _args, _bf, _real_standup_records, _source, _stage1_world, _wire)

KEY = "synth_fd"          # a synthetic front-door key: the audit id's VIEW suffix is all the steered-wave grammar reads


# ------------------------------------------------------------------------------------------------- (1) the classification

def test_an_s2rl_audit_is_a_steered_wave_audit_like_s2r_and_s3r_and_u0_is_not():
    S = lambda audit: type("S", (), {"audit": audit})()
    assert prun.steered_wave_audit(S(f"petri_full22_{KEY}_s2rl"))
    assert prun.steered_wave_audit(S(f"petri_full22_{KEY}_s2r")) and prun.steered_wave_audit(S(f"petri_full22_{KEY}_s3r"))
    assert not prun.steered_wave_audit(S(f"petri_full22_{KEY}_u0")) and not prun.steered_wave_audit(S("petri_full22_synth"))
    # the s1 (LoRA) wave has always read u0 / legacy here (no `r` in its audit id; its passes declared the complete baseline), as
    # reproduce/lib/models.py STEERED_WAVE_VIEW_RE `^s[0-9]+rl?$` excludes s1 — UNCHANGED by this hotfix; s1's classification is an open item
    assert not prun.steered_wave_audit(S(f"petri_full22_{KEY}_s1"))
    assert not prun.steered_wave_audit(S(f"petri_full22_{KEY}_s2rlx"))                      # suffix-anchored: the view is the last segment
    assert prun._STEERED_AUDIT_RE.pattern == r"_s[0-9]+rl?$"


def test_the_contrast_emitter_strips_the_s2rl_suffix_to_its_u0_anchor_view():
    import importlib.util
    spec = importlib.util.spec_from_file_location("emit_contrasts_under_test", REPO / "scripts" / "registry" / "emit_contrasts.py")
    ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
    assert ec._U0_VIEW_RE.sub("_u0", f"{KEY}_s2rl") == f"{KEY}_u0" and ec._U0_VIEW_RE.sub("_u0", f"{KEY}_s2r") == f"{KEY}_u0"
    assert not ec._U0_VIEW_RE.search(f"{KEY}_u0")


# ------------------------------------------------------------------------------------------------- (2) the two-cell restriction

def test_an_s2rl_audit_with_an_explicit_two_cell_declaration_restricts_to_those_legs_with_no_controls_and_no_refusal(tmp_path, monkeypatch):
    source = _source(tmp_path / "bank")
    spec = shape._spec("ab" * 8, audit=f"petri_full22_{KEY}_s2rl", cells=list(CELLS))
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES)
    _wire(monkeypatch, source, identity=NATIVE_IDENTITY, calls=[])
    assert prun.steered_wave_audit(spec)
    two = ["prompted__safety", LORA]                                                        # the two overflowed legs' shape: no `none`
    rec, table, _ = _real_standup_records(tmp_path / "o", native, source, spec, requested=two, attempt=A32, monkeypatch=monkeypatch)
    assert prun.complete_baseline_problem(native.cell_set, two, spec, "p") is not None   # a u0 audit WOULD refuse this partial declaration …
    assert prun.cmd_print_cells(_args(tmp_path / "out", serving_path=prun.SERVING_PATH_LORA_NATIVE, standup=rec, table=table), spec) == 0   # … the s2rl audit restricts
    frozen = json.loads((tmp_path / "out" / prun.CELLS_JSON).read_text())
    assert [c["cell_id"] for c in frozen["canonical"]] == two and frozen["controls"] == [] and frozen[prun.REQUESTED_HARNESS_KEY] == two


# ------------------------------------------------------------------------------------------------- (3) the sweep

BODY = re.compile(r"s(?:\[0-9\]|\\d)\+r")
PROBES_S2RL = ("s2rl", "_s2rl", f"{KEY}_s2rl", f"petri_full22_{KEY}_s2rl", f"f22-petri-{KEY}_s2rl", "x/s2rl/y", "s2rl_sampled_t0")
PROBES_S2R = tuple(p.replace("s2rl", "s2r") for p in PROBES_S2RL)
OPTIONAL_L = ("l?", "l\\?", "l*", "[l]?", "(l)?", "l{0,1}", "l\\{0,1\\}")
SKIP_DIRS = {".git", ".venv", "worktrees", "__pycache__", "node_modules", ".pytest_cache", "site-packages"}
SUFFIXES = {".py", ".sh", ".sbatch", ".yaml", ".yml", ".json", ".patch"}        # code only: prose (.md / .txt) quotes grammars in backticks


def _literal_around(line: str, start: int, end: int):
    """The quoted literal enclosing [start, end) on `line`, else None (a quote-free line, or a body in prose)."""
    for q in ('"', "'"):
        a = line.rfind(q, 0, start); b = line.find(q, end)
        if a >= 0 and b >= 0:
            return line[a + 1:b]
    return None


def _tree_files(root: Path = REPO):
    """The TREE's files: `git ls-files` (tracked + the index), so a landing driver's scratch-root dotfiles (its `.patch.py` carries the OLD literals
    as edit anchors — chain 340 went red on exactly that) and other untracked residue are never swept; the rglob fallback skips dotted parts."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, check=True).stdout
        return [root / f for f in out.split("\0") if f]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in root.rglob("*") if not any(part.startswith(".") for part in p.relative_to(root).parts)]


def _offenders(files=None, root: Path = REPO):
    """The sweep over ``files`` (default: the tree's files under ``root``), reporting paths relative to ``root`` -- parameters so the
    dotted-path negative below is a real test over a synthetic root, not a description (S2RLFIX6-PETRI)."""
    bad = []; seen = 0
    for p in (list(files) if files is not None else _tree_files(root)):
        if not p.is_file() or p.suffix not in SUFFIXES or SKIP_DIRS & set(p.relative_to(root).parts) or p.resolve() == Path(__file__).resolve():
            continue
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue                                          # dotfiles / dot-dirs are never tree code
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not BODY.search(text):
            continue
        for ln, line in enumerate(text.splitlines(), 1):
            raw = line
            if p.suffix == ".patch":
                if line.startswith("-"):
                    continue                                      # a removed line of a vendored diff is the OLD text
                line = line[1:] if line[:1] == "+" else line
            if line.lstrip().startswith("#"):
                continue                                          # prose
            for m in BODY.finditer(line):
                seen += 1
                lit = _literal_around(line, m.start(), m.end())
                verdict = None
                if lit is not None:
                    try:
                        rx = re.compile(lit)
                    except re.error:
                        rx = None
                    if rx is not None:
                        if not any(rx.search(x) for x in PROBES_S2R):
                            continue                              # not a view grammar (admits no s2r form): out of this sweep's scope
                        verdict = any(rx.search(x) for x in PROBES_S2RL)
                if verdict is None:                               # no compilable literal: the text rule — an optional `l` right after the body
                    after = line[m.end():m.end() + 10]
                    verdict = any(after.startswith(o) for o in OPTIONAL_L) or after.startswith("?p?|s[0-9]+rl")
                if not verdict:
                    bad.append(f"{p.relative_to(root)}:{ln}: {raw.strip()[:140]}")
    return bad, seen


def test_the_sweep_never_reads_a_dotted_path_and_flags_a_stale_copy_in_tree_code(tmp_path):
    """The `.patch.py` negative, for real (S2RLFIX6-PETRI): a landing driver's edit script carries the OLD literal as an anchor and is a
    dotted path -- never swept (chain 340 went red on exactly that); the same literal in tree code is flagged; the widened copy passes."""
    stale = 'RX = re.compile(r"_s[0-9]+r$")'
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / ".patch.py").write_text(stale + "\n", encoding="utf-8")
    (tmp_path / "tests" / "stale_copy.py").write_text(stale + "\n", encoding="utf-8")
    (tmp_path / "tests" / "widened.py").write_text('RX = re.compile(r"_s[0-9]+rl?$")\n', encoding="utf-8")
    bad, seen = _offenders(files=sorted((tmp_path / "tests").iterdir()), root=tmp_path)
    assert bad == [f"tests/stale_copy.py:1: {stale}"] and seen == 2, (bad, seen)


def test_every_view_grammar_regex_in_the_tree_accepts_s2rl_wherever_it_accepts_s2r():
    bad, seen = _offenders()
    assert seen >= 10, f"the sweep saw only {seen} grammar bodies — the body regex or the file walk is wrong"
    assert bad == [], "view-grammar copies that admit s2r but not s2rl (widen each to `rl?`):\n  " + "\n  ".join(bad)
    # the sweep catches the shape this fix repairs: a suffix-anchored copy without the optional l
    assert not re.compile(r"_s[0-9]+r$").search(f"petri_full22_{KEY}_s2rl") and re.compile(r"_s[0-9]+rl?$").search(f"petri_full22_{KEY}_s2rl")
