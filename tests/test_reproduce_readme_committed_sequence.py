"""reproduce/README.md's worked example — the committed sequence on the paper's committed Qwen3.5-4B objects — replayed through the
stages it names, against the README TEXT.

The page gives the sequence twice, and this module holds the two to each other:
  - near the top, under "## The worked example", ONE fenced bash block holds every command in run order, with nothing but one-word stage
    labels (`# 03`, `# gate`) for comments;
  - "## The worked example, stage by stage" breaks it down: one section per label, in the block's order, each with one fenced block whose
    commands ARE the label's group (never retyped elsewhere, never a command the top block lacks).
Every command of the top block is extracted from the page (a backslash continuation joined), and so is every `reproduce/NN_*.sh --model …`
command the sections spell inline (the Petri overflow pass). Each stage line is filled as a reader would and run with `--dry-run` against a
site built from reproduce/site.example that follows the page's own setup step (`SISTEER_RUN_OVERRIDES` in config.env): the stage parses the
line's flags — an unknown argument, a `--step`, `--phase` or `--evals` value outside the stage's grammar, a cell that is not the view's,
refuse by name and exit non-zero before any dry run — prints its dry run, names the smoke recipe the config selected, and touches no state.
The lines that need what only a REAL run leaves — the gate call and the admission check (the frozen inventory 05's freeze writes),
`07 --phase mirror` and `07 --phase file` (the receipt, the ruled verdict) — are held to naming that prerequisite rather than to exit 0. The
page's other lines (the site copy, the registry import, the compute plan) write or read real state and are not replayed; the fixture's site
is the copy the first line makes.

The sequence's SHAPE is pinned beside the replay, against the stages' own behaviour:
  - the setup comes first: the site copy, then stage 00, then the registry import (05's pins read it);
  - the stages run in numeric order (00, 03, 05, 06, 07, 08, 09), and the one stage that comes back after a later one is 06, whose gate
    call reads the grades 08 files;
  - 05, 06's wave and 08 are one call each, through each stage's all-mode; 09 is one command run twice (the first call stops at `guard`
    until its `views` jobs have built the views); 05 with no `--step` and 09 with `--step all,views` run their six steps in the order their
    own dry runs print (09: views before guard before submit);
  - 03 is the probe's four calls (the probe, its parity pass, the re-verdict, the probe again to fetch), the compute plan after them;
  - 07 is called phase by phase, preflight before run: on the page's site its `--phase all` composes no preflight phase, and a real run
    refuses for want of the hashes the preflight derives;
  - the gate is ONE call, `--gate all --verdict`, which reads every family's headline from the registry (no number typed), and every family
    of the battery's unit plan is one `--gate all` reaches and one the committed direction table (lib/gate_headline.py DIRECTIONS) has a row
    for; the page's gate table IS that table (scale, expected direction, range), each range the table's own rule derives from the family's
    scale and expected direction, and stage 07's receipt bullet adds the table's 07 range;
  - the fit-your-own path continues with 06, 08 and 09 only: stage 07's bank stager refuses a key that imports no committed objects, so
    the page neither sends a key of one's own through 07 nor asks for a launcher row for it.
The page carried, at different times, a 09 line whose comment named steps the page never showed, a 07 line that skipped the preflight
the run refuses without, a wave with no overrides selection, and a sequence out of numeric order; this module holds the sequence to the
stages so such a line cannot ship un-run again."""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, VENV_PY, assert_no_tokens, run_stage, scrub_site  # noqa: E402

README = REPRO / "README.md"
MODEL = "qwen35_4b_demo"
TOP = "\n## The worked example\n"
STAGES = "\n## The worked example, stage by stage\n"
END = "\n## Fitting your own objects\n"
#: each top-block label and the heading of the section that breaks its group down, in the page's order
SECTIONS = [("setup", "### Setup"), ("03", "### 03, "), ("05", "### 05, "), ("06", "### 06, "), ("07", "### 07, "), ("08", "### 08, "),
            ("gate", "### 06 again, the gate"), ("09", "### 09, ")]
#: the page's placeholders, filled as a reader would: a forced admission names a unit and why
FILL = {"<logical work id>=<reason>": "X=smoke"}
PLACEHOLDER = re.compile(r"<[a-z][a-z _,-]{1,40}>")
#: the page's two home-relative paths (the admission line names them; a stage reads its config's), mapped onto the test's site
STATE_DEFAULT, REGISTRY_DEFAULT = "~/.local/state/sisteer-reproduce", "~/registry"
CONFIG_LINE = re.compile(r"In `config\.env`, select the smoke recipe: `(SISTEER_RUN_OVERRIDES=[^`\s]+)`")
FENCE = re.compile(r"^```bash\n(.*?)^```", flags=re.S | re.M)
INLINE = re.compile(r"`(reproduce/\d\d_\w+\.sh --model [^`]+)`")
LABEL = re.compile(r"^# (\S+)$")


def _between(start: str, end: str) -> str:
    text = README.read_text(encoding="utf-8")
    assert start in text, f"the README's {start.strip()!r} heading"
    i = text.index(start)
    j = text.find(end, i + len(start))
    assert j > i, f"the heading {end.strip()!r} after {start.strip()!r}"
    return text[i:j]


def _top() -> str:
    """The section that holds the one block: from its heading to the next top-level heading."""
    return _between(TOP, "\n## ")


def _stages() -> str:
    """The stage-by-stage breakdown: from its heading to the fit-your-own heading."""
    return _between(STAGES, END)


def _commands(block: str) -> list[str]:
    """The commands of one fenced bash block, continuations joined; a comment line is dropped, and a comment after a command refuses (the
    page's blocks carry none: the top block's only comments are its labels)."""
    out: list[str] = []
    cur = ""
    for raw in block.splitlines():
        line = raw.rstrip()
        if not cur and (not line.strip() or line.lstrip().startswith("#")):
            continue
        assert not re.search(r"\s#", line), f"a comment after a command: {line!r}"
        if line.endswith("\\"):
            cur += line[:-1].rstrip() + " "
            continue
        out.append((cur + line).strip())
        cur = ""
    assert not cur, "an unterminated continuation at a block's end"
    return out


def _top_block() -> str:
    blocks = FENCE.findall(_top())
    assert len(blocks) == 1, f"the worked example's top section holds ONE bash block, not {len(blocks)}"
    return blocks[0]


def _top_groups() -> list[tuple[str, list[str]]]:
    """The top block as (label, commands) groups, in order: every comment line is a one-word label, and every command sits under one."""
    groups: list[tuple[str, str]] = []
    for raw in _top_block().splitlines():
        if raw.lstrip().startswith("#"):
            m = LABEL.match(raw)
            assert m, f"the top block's only comments are one-word stage labels: {raw!r}"
            groups.append((m.group(1), ""))
        else:
            assert groups, f"a command before the first label: {raw!r}"
            groups[-1] = (groups[-1][0], groups[-1][1] + raw + "\n")
    return [(label, _commands(body)) for label, body in groups]


def _section_groups() -> list[tuple[str, list[str]]]:
    """The breakdown as (### heading, commands of its fenced blocks) groups, in page order."""
    parts = re.split(r"^(### .*)$", _stages(), flags=re.M)
    return [(parts[i], [c for b in FENCE.findall(parts[i + 1]) for c in _commands(b)]) for i in range(1, len(parts) - 1, 2)]


def _sequence() -> list[str]:
    """The steps a reader types, in run order: the top block's commands."""
    steps = [c for _label, cmds in _top_groups() for c in cmds]
    assert steps, "the worked example's commands"
    return steps


def _fill(line: str) -> str:
    for k, v in FILL.items():
        line = line.replace(k, v)
    assert not PLACEHOLDER.search(line), f"an unfilled placeholder in {line!r} — add it to FILL"
    return line


def _argv(line: str) -> list[str]:
    return shlex.split(_fill(line))


def _flag(line: str, name: str) -> str | None:
    argv = _argv(line)
    return argv[argv.index(name) + 1] if name in argv else None


def _num(line: str) -> int:
    return int(Path(_argv(line)[0]).name[:2])


def _short(line: str) -> str:
    return re.sub(r"^reproduce/(\d\d)_\w+\.sh --model \S+ ?", r"\1 ", line)[:60]


LINES = _sequence()
STAGE_LINES = [ln for ln in LINES if ln.startswith("reproduce/")]
#: every stage command the page spells, in the top block or inline in a stage's section (the rare-case Petri overflow pass), each once
DRY_LINES = list(dict.fromkeys(STAGE_LINES + [m.group(1) for m in INLINE.finditer(_stages())]))
GATE_LINES = [ln for ln in STAGE_LINES if "--gate" in _argv(ln)]
ADMISSION_LINES = [ln for ln in LINES if ln.startswith("python reproduce/lib/admission.py check")]
REPLAN_LINES = [ln for ln in LINES if ln.startswith("python reproduce/tools/replan_compute.py")]
IMPORT_LINES = [ln for ln in LINES if ln.startswith("python reproduce/tools/import_objects.py registry")]
COPY_LINES = [ln for ln in LINES if ln.startswith("cp ")]


def _by_stage(n: int) -> list[str]:
    return [ln for ln in STAGE_LINES if _num(ln) == n]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """reproduce/site.example (the copy the page's first line makes) with the test's own state and registry roots, plus the page's setup
    edit: the one config line that selects the smoke recipe (taken from the page's text, so the replay follows what the reader is told)."""
    root = tmp_path_factory.mktemp("readme_seq")
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    m = CONFIG_LINE.search(_stages())
    assert m, "the page's setup names the config.env line that selects the smoke recipe"
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n{m.group(1)}\n")
    return site


def _state_files(site: Path) -> set[str]:
    st = site.parent / "state"
    return {str(p) for p in st.rglob("*")} if st.exists() else set()


def _dry(site: Path, line: str) -> tuple[subprocess.CompletedProcess, str]:
    argv = _argv(line)
    r = run_stage(Path(argv[0]).name, *argv[1:], "--dry-run", site=site, timeout=600)
    return r, r.stdout + r.stderr


def _in_order(out: str, marks: list[tuple[str, str]], line: str) -> None:
    """Each step's marker is in the dry run, and the markers come in the steps' order."""
    missing = [step for step, mark in marks if mark not in out]
    assert not missing, (line, f"steps {missing} print nothing in one call", out[-2000:])
    pos = [out.index(mark) for _step, mark in marks]
    assert pos == sorted(pos), (line, [step for _p, (step, _m) in sorted(zip(pos, marks))])


def test_the_pages_config_line_selects_the_shipped_smoke_recipe_for_this_model():
    m = CONFIG_LINE.search(_stages())
    assert m, "the page's setup names the config.env line"
    path = m.group(1).split("=", 1)[1]
    doc = (REPO / path).read_text(encoding="utf-8")
    assert "OVERRIDE_ID=demo_smoke\n" in doc and f"OVERRIDE_MODEL={MODEL}\n" in doc and re.search(r"^OVERRIDE_SMOKE_GOALS=\d+-\d+/\d+", doc, flags=re.M)


def test_the_top_block_is_commands_under_one_word_stage_labels_and_each_section_breaks_one_label_down_in_order():
    """The page's shape: one block near the top with every command in run order and no comment but a one-word stage label,
    then the stage-by-stage sections, whose blocks together ARE that block — section k's commands are label k's group, in order."""
    top, secs = _top_groups(), _section_groups()
    assert [label for label, _c in top] == [label for label, _h in SECTIONS], [label for label, _c in top]
    assert len(secs) == len(SECTIONS) and all(h.startswith(want) for (h, _c), (_l, want) in zip(secs, SECTIONS)), [h for h, _c in secs]
    for (label, cmds), (heading, sec_cmds) in zip(top, secs):
        assert cmds, f"the label {label!r} heads no command"
        assert sec_cmds == cmds, (label, heading, sec_cmds, cmds)
    assert [c for _h, cs in secs for c in cs] == LINES, "the sections' blocks, concatenated, are the top block"
    # a numbered label names the stage of every stage line under it (`gate` is 06 again, `setup` is 00)
    for label, cmds in top:
        want = {"setup": 0, "gate": 6}.get(label, None if not label.isdigit() else int(label))
        assert want is not None, f"an unknown label {label!r}"
        assert all(_num(c) == want for c in cmds if c.startswith("reproduce/")), (label, cmds)


def test_setup_is_the_site_copy_then_stage_00_then_the_registry_import():
    [cp] = COPY_LINES
    [zero] = _by_stage(0)
    [imp] = IMPORT_LINES
    assert LINES[:3] == [cp, zero, imp], LINES[:3]
    assert shlex.split(cp)[:3] == ["cp", "-r", "reproduce/site.example"], "the copy the fixture's site makes"
    assert _flag(zero, "--what") == "all", "00 installs everything in one call"


def test_the_stages_run_in_numeric_order_and_only_06s_gate_comes_back_after_a_later_stage():
    assert {Path(_argv(ln)[0]).name for ln in STAGE_LINES} == {"00_setup.sh", "03_hpo.sh", "05_stage_roster.sh", "06_eval_waves.sh",
                                                               "07_bcb_bfcl.sh", "08_grade.sh", "09_dose_ladder.sh"}
    assert all(_flag(ln, "--model") == MODEL for ln in STAGE_LINES), "every stage line runs the worked example's key"
    runs = [n for n, _grp in itertools.groupby(_num(ln) for ln in STAGE_LINES)]
    assert runs == [0, 3, 5, 6, 7, 8, 6, 9], f"the stages in page order: {runs}"
    first8 = next(i for i, ln in enumerate(STAGE_LINES) if _num(ln) == 8)
    early, late = ([ln for ln in part if _num(ln) == 6] for part in (STAGE_LINES[:first8], STAGE_LINES[first8:]))
    assert early and not any("--gate" in _argv(ln) for ln in early), "06 submits the wave before 08"
    assert late and all("--gate" in _argv(ln) for ln in late), "06 comes back after 08 only to record the gate, which reads 08's grades"


def test_05_the_wave_and_08_are_one_call_each_and_09_one_command_run_twice_through_their_all_modes():
    [five] = _by_stage(5)
    assert _flag(five, "--step") is None, "05's default step is all"
    [wave] = [ln for ln in _by_stage(6) if "--gate" not in _argv(ln)]
    assert _flag(wave, "--view") == "battery" and _flag(wave, "--family") == "all"
    [eight] = _by_stage(8)
    assert "--daemon" in _argv(eight)
    nine, again = _by_stage(9)
    assert again == nine, "the second call is the first one again: it skips the views the first call's jobs built, then submits"
    assert _flag(nine, "--step") == "all,views" and _flag(nine, "--alpha") == "0.75"
    # the default `--evals` (static, multiturn, controls — the D.2 legs): the multiturn legs run as the dose view's multiturn family through
    # 06_eval_waves.sh, which starts their attacker pool with them, so the page selects nothing
    assert _flag(nine, "--evals") is None, "the D.2 legs by default; the multiturn legs go out through 06 with their pool"
    assert "=" in (_flag(nine, "--force-gate") or ""), "the recorded admission override, LWID=REASON"


def test_05s_one_call_runs_its_six_steps_in_order(site):
    [five] = _by_stage(5)
    r, out = _dry(site, five)
    assert r.returncode == 0, (five, out[-2500:])
    _in_order(out, [("stage", "stage_objects.py"), ("pins", "import_objects.py pins"), ("roster", "sdfbase_full22_roster.yaml carries"),
                    ("configs", "make_full22_configs.py --check"), ("bank-views", "make_bank_view.py"), ("freeze", "expected_units.json")], five)


def test_09s_one_command_runs_its_six_steps_views_before_guard_before_submit(site):
    nine = _by_stage(9)[0]
    r, out = _dry(site, nine)
    assert r.returncode == 0, (nine, out[-2500:])
    _in_order(out, [("build", "DRY-RUN: ledger row"), ("stage", "stage_objects.py"), ("configs", "make_full22_configs.py --check"),
                    ("views", "build_views.sbatch"), ("guard", "launch_guard.py"), ("submit", "admission.py check")], nine)


def test_03_is_the_probes_four_calls_and_the_compute_plan_follows_them():
    three = _by_stage(3)
    assert len(three) == 4 and all(_flag(ln, "--object-class") == "lora" and _flag(ln, "--phase") == "probe" for ln in three)
    assert "--parity-pass" in _argv(three[1]) and {"--reopen", "--reverdict-only"} <= set(_argv(three[2]))
    assert three[3] == three[0], "the first call again fetches the re-verdicted verdict"
    assert not ({"--parity-pass", "--reopen"} & set(_argv(three[0]))), "the first call is the plain probe"
    # the registry import before the probe (05's pins read it), the compute plan after the last probe call and before the wave (06 refuses
    # without it)
    [imp] = IMPORT_LINES
    [replan] = REPLAN_LINES
    fetch = max(i for i, ln in enumerate(LINES) if ln == three[3])
    first = {n: next(i for i, ln in enumerate(LINES) if ln.startswith(f"reproduce/{n:02d}_")) for n in (3, 5, 6)}
    assert LINES.index(imp) < first[3] and fetch < LINES.index(replan) < first[5] < first[6]
    assert "--serve-only" in shlex.split(replan) and f"{STATE_DEFAULT}/{MODEL}/03_hpo/lora/probe_verdict.json" in replan


def test_07_runs_phase_by_phase_preflight_before_run_because_its_all_mode_skips_the_preflight(site):
    seven = _by_stage(7)
    assert [_flag(ln, "--phase") for ln in seven] == ["stage", "preflight", "run", "receipt", "mirror", "file"]
    cells = {_flag(ln, "--phase"): _flag(ln, "--cells") for ln in seven}
    assert cells["stage"] is None and cells["preflight"] is None, "the bank and the preflight cover the whole view"
    assert cells["run"] == cells["receipt"] == cells["mirror"] == cells["file"] == "none,difference_lora", "one selection on run, receipt, mirror and file (file refuses another)"
    # the reason the page gives, held to the stage on the page's site: `--phase all` composes the stage and the run but no preflight phase,
    # and a real run refuses for want of the hashes only the preflight derives
    r = run_stage("07_bcb_bfcl.sh", "--model", MODEL, "--phase", "all", "--cells", cells["run"], "--dry-run", site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert "--upload-tree --stage" in out and "--pf --pass1" in out, "all = the stage and the run"
    assert "07_preflight_report.log" not in out, "all composes no preflight phase"
    assert "a real run refuses here" in out and "--phase preflight" in out, out[-2500:]
    assert "`--phase all` stages and runs but skips the preflight" in " ".join(_stages().split())


def _gate_table() -> dict[str, tuple[str, str, str]]:
    """The page's gate table: {family (or `07`): (scale, expected direction, range)}."""
    text = README.read_text(encoding="utf-8")
    gate = text[text.index("## The gate"):]
    gate = gate[:gate.index("\n## ", 1)]
    rows: dict[str, tuple[str, str, str]] = {}
    for line in gate.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or not cells[4].startswith("`") or cells[0] == "family":
            continue
        rng = cells[4].strip("`")
        fams = re.findall(r"`([a-z0-9]+)`", cells[0]) or (["07"] if "07" in cells[0] else [])
        assert fams, line
        for f in fams:
            rows[f] = (cells[2], cells[3], rng)
    return rows


def test_the_gate_table_derives_each_range_from_its_scale_and_expected_direction():
    table = _gate_table()
    assert set(table) == {"static", "padding", "bon", "multiturn", "petri", "agentic", "controls", "mmlu", "gsm8k", "07"}, sorted(table)
    for fam, (scale, direction, rng) in table.items():
        lo_s, hi_s = (float(x) for x in re.fullmatch(r"(\d+) to (\d+)", scale).groups())
        span = hi_s - lo_s
        want = {"lower it": (lo_s + 0.05 * span, hi_s), "raise it": (lo_s, lo_s + 0.95 * span)}[direction]
        got = tuple(float(x) for x in rng.split(","))
        assert got == pytest.approx(want), (fam, scale, direction, rng)


def test_the_gate_is_one_verdict_call_over_every_family_and_the_pages_table_is_the_committed_direction_table():
    [gate] = GATE_LINES
    argv = _argv(gate)
    assert _flag(gate, "--gate") == "all" and "--verdict" in argv and "--headline" not in argv and "--range" not in argv, gate
    order = re.search(r'^ALL_FAMILIES="([^"]+)"', (REPRO / "06_eval_waves.sh").read_text(encoding="utf-8"), flags=re.M).group(1).split()
    plan = subprocess.run([VENV_PY, str(LIB / "stage.py"), "plan", "--model", MODEL, "--view", "battery", "--families", ",".join(order),
                           "--nshards", "9", "--sha", "0" * 40, "--clusters", "cluster_a,cluster_b", "--models-table", str(REPRO / "models.yaml"),
                           "--json"], cwd=str(REPO), capture_output=True, text=True, timeout=300)
    assert plan.returncode == 0, plan.stderr[-1500:]
    planned = {row["family"] for row in json.loads(plan.stdout) if row["registers"]}
    spec = importlib.util.spec_from_file_location("_readme_sequence_gate_headline", LIB / "gate_headline.py")
    gh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gh)
    assert planned <= set(order) and planned <= set(gh.DIRECTIONS), (sorted(planned), sorted(gh.DIRECTIONS))   # `all` reaches each, and each has a row
    # the page's table IS the committed one: every row's scale, expected direction and range, and no row missing on either side
    table = {("bcb_bfcl" if fam == "07" else fam): row for fam, row in _gate_table().items()}
    assert set(table) == set(gh.DIRECTIONS), (sorted(table), sorted(gh.DIRECTIONS))
    for fam, d in gh.DIRECTIONS.items():
        scale, direction, rng = table[fam]
        assert scale == f"{gh.fmt(d.scale[0])} to {gh.fmt(d.scale[1])}", (fam, scale, d.scale)
        assert direction == {"decrease": "lower it", "increase": "raise it"}[d.expected], (fam, direction, d.expected)
        assert rng == gh.range_arg(fam), (fam, rng, d.range)
    # stage 07's verdict: the receipt's printed command, with the table's range added by the reader
    assert f"Add `--range {table['bcb_bfcl'][2]}` and run it" in " ".join(_stages().split())


@pytest.mark.parametrize("line", DRY_LINES, ids=[_short(ln) for ln in DRY_LINES])
def test_every_stage_line_of_the_worked_example_dry_runs_as_written(line, site):
    argv = _argv(line)
    stage = Path(argv[0]).name
    before = _state_files(site)
    r, out = _dry(site, line)
    scan = scrub_site(out, tmp=site.parent)          # the runner's own paths normalised (the scrub test_front_door_dry_run shares)
    assert_no_tokens(scan, line)
    assert "unknown argument" not in out, (line, out[-1500:])
    # (a stage's own `DRY-RUN:` notes carry `<…>` prose by design — the verdict command's `--range <lo,hi>`, the preflight's `<slug>` —
    # so the page's placeholders are held filled by `_fill`, never searched for in the stage's output)
    if "--gate" in argv:
        # the verdict pass reads the frozen inventory a REAL `05 --step freeze` writes; the dry run names it and nothing else stops it
        assert r.returncode != 0 and "no frozen inventory" in out and "05_stage_roster.sh --step freeze" in out, (line, out[-1500:])
    elif _flag(line, "--phase") == "file":
        # the filing reads the ruled verdict and the receipt only a REAL run and `07 --phase receipt` leave; the refusal names both steps
        assert r.returncode != 0 and "no verdict at" in out and "07 --phase receipt" in out and "nothing filed" in out, (line, out[-1500:])
    elif _flag(line, "--phase") == "mirror":
        # the mirror pulls the stores the receipt names, so its dry run reads the receipt only `07 --phase receipt` leaves; the refusal names that step
        assert r.returncode != 0 and "no receipt at" in out and "07 --phase receipt" in out and "stage07_mirror.py" in out, (line, out[-1500:])
    else:
        assert r.returncode == 0, (line, out[-2500:])
    if _flag(line, "--pass") == "overflow":
        # the rare-case pass on a site the daemon never wrote to: the dry run names the record a real call needs and composes nothing
        assert "DRY-RUN: --pass overflow reads the daemon's record" in out, (line, out[-1500:])
    if stage == "09_dose_ladder.sh":
        # the page's override, filled, reaches the admission call and the stage's own summary
        assert "--force-gate X=smoke" in out and "forced: X (smoke)" in out, (line, out[-1500:])
    if "overrides=" in out:
        assert "overrides=demo_smoke" in out, (line, "the config's one selection reaches the stage")
    assert _state_files(site) == before, (line, sorted(_state_files(site) - before)[:5])


def test_the_admission_line_reads_the_sites_own_state_and_names_the_frozen_inventory_it_needs(site):
    [line] = ADMISSION_LINES
    assert LINES.index(GATE_LINES[-1]) < LINES.index(line) < LINES.index(_by_stage(9)[0]), "the gate's verdicts, the admission check, then 09"
    assert f"--state-root {STATE_DEFAULT}/{MODEL}" in line and f"--registry {REGISTRY_DEFAULT}/registry.sqlite" in line, "the page's default roots"
    filled = line.replace(STATE_DEFAULT, str(site.parent / "state")).replace(REGISTRY_DEFAULT, str(site.parent / "registry"))
    assert "~" not in filled
    argv = shlex.split(filled)
    r = subprocess.run([VENV_PY, *argv[1:]], cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and "no frozen baseline inventory" in r.stderr and "05_stage_roster.sh" in r.stderr, r.stderr[-1500:]
    assert not (site.parent / "state" / MODEL / "gate").exists(), "a refused check records nothing"


FIT_HEADING, FIT_END = "## Fitting your own objects", "## Reading a demo run"
FRESH = "example"           # a key of one's own (the page's Adding a model): the worked example's row copied WITHOUT committed_objects_of


def _page_section(start_mark: str, end_mark: str) -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index(start_mark)
    return " ".join(text[start:text.index(end_mark, start + len(start_mark))].split())


def test_the_fit_your_own_path_leaves_out_stage_07_whose_bank_stager_refuses_a_key_with_no_committed_objects(tmp_path):
    """Stage 07's `stage` phase stages the paper's COMMITTED objects on Modal (tools/stage_demo_bank_modal.py over import_objects.Plan), so
    a key of one's own is refused before any Modal call, on the wave and on a dose view alike — and the launcher row the page once asked
    for cannot help, since the refusal comes first. The page once sent a fitted key through "06 to 09 as above": held here to the stager."""
    fit = _page_section(FIT_HEADING, FIT_END)
    cont = re.search(r"then ([0-9, and]+) as above with `--model <your key>`", fit)
    assert cont, "the fit-your-own continuation names its stages one by one (never a range that runs through 07)"
    assert re.findall(r"\d\d", cont.group(1)) == ["06", "08", "09"], cont.group(0)
    assert "Stage 07 does not run under a key of your own" in fit
    adding = _page_section("## Adding a model", "\n## Recovering")
    assert "contracts.py" not in adding and "needs no launcher row" in adding, adding[:800]
    t = yaml.safe_load((REPRO / "models.yaml").read_text(encoding="utf-8"))
    row = copy.deepcopy(t["models"][MODEL])
    row.pop("committed_objects_of", None)
    for o in (row.get("objects") or {}).values():
        if isinstance(o, dict):
            o.pop("committed_objects_of", None)
    row["view_legacy_prefix"] = FRESH
    row["fit"]["checkpoint"]["model_id"], row["serve"]["checkpoint"]["model_id"] = f"{FRESH}_base", f"{FRESH}_it"
    t["models"][FRESH] = row
    table = tmp_path / "models.yaml"
    table.write_text(yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
    for view in ("battery", "d0p75"):
        assert view in row["views"], (view, sorted(row["views"]))
        r = subprocess.run([VENV_PY, str(REPRO / "tools" / "stage_demo_bank_modal.py"), "--model", FRESH, "--view", view, "--table", str(table),
                            "--receipts-dir", str(tmp_path / "receipts"), "--stage-dir", str(tmp_path / "stage"), "--volume", "synthetic-volume",
                            "--environment", "synthetic-env", "--modal", "/bin/false"], cwd=str(REPO), capture_output=True, text=True, timeout=120)
        out = r.stdout + r.stderr
        assert r.returncode == 2 and "declares no `committed_objects_of`" in out and "no receipt written" in out, (view, out[-1500:])
    assert not (tmp_path / "receipts").exists() and not (tmp_path / "stage").exists(), "a refused stage writes nothing"
