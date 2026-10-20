"""scripts/modal/README.md's "BigCodeBench and BFCL" recipe, replayed through the launcher it documents.

The recipe on main before this test showed a `modal run` of scripts/modal/tinycap_served.py that did not run: no pass (the
launcher plans a configuration only behind `--pass1`; without one it plans nothing and exits 0 having evaluated nothing), no
`--pf` (every fitted cell launches only behind the CPU preflight), no `--modes` (the launcher's default plans an `on` leg a row
with no pinned reasoning parser refuses), no word on `--allow-extra` (more than contracts.MAX_GPU_CONTAINERS cells refuse) or
`--expect-hashes` (a fitted cell with no contracts.EXPECTED_INTERVENTION_HASHES entry is refused by the preflight), and it named
the paper models by slugs the harness cannot launch directly: the paper's rows are the roster's `model_row` lane rows
(contracts.BCBW_SLUGS), which the launcher serves only with the roster snapshot the bcbw wrapper renders — all found by
replaying stage 07's argv through the launcher (tests/test_front_door_dry_run.py), the technique this module reuses.

Pinned here, against the README TEXT (the command is extracted from the page, never retyped, so the page cannot drift from the
launcher again): the recipe's run command, with its placeholders filled for each direct-launch slug the page names, parses
through the launcher's own `cli_parser` and plans ONE GPU container per cell as its CPU dry run (Modal stubbed); the flags the
page adds are load-bearing exactly as it says (over the cap `--allow-extra`; a row serving `off` alone refuses without `--modes`;
without `--pass1` a real run plans nothing); the lane rows the page names refuse a direct launch by name, pointing at the
wrapper; and the export line names the launcher's app-name variable, which the launcher holds `--app-name` to."""
from __future__ import annotations

import contextlib
import inspect
import io
import re
import shlex
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "scripts/modal/README.md"
HEADING = "## BigCodeBench and BFCL"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"      # any committed sha: the dry run plans without the volume
APP = "tinycap-readme-recipe"                           # any app name, as the page says — the launcher is imported under it
#: the rows the page offers for a DIRECT launch (outside contracts.BCBW_SLUGS, a tool-call parser manifest for BFCL, a hash
#: table entry per fitted cell); the page is held to naming each, so the examples and this replay stay in step
DIRECT_SLUGS = ("gemma4e4b", "llama8b")
#: the paper's rows the page names as wrapper-only: the roster's model_row entries, every one a lane row
LANE_SLUGS = ("qwen35_4b-f22s1", "gemma4e4b-f22s1", "llama8b-f22s1", "nemotron3nano4b-f22s1")


def _section(text: str, heading: str) -> str:
    """The markdown under `heading` up to the next `## `."""
    start = text.index(heading + "\n")
    rest = text[start + len(heading) + 1:]
    m = re.search(r"^## ", rest, flags=re.M)
    return rest[:m.start()] if m else rest


def _bash_lines(section: str) -> list[str]:
    """The lines of the section's fenced bash block, backslash continuations joined into one command each."""
    m = re.search(r"```bash\n(.*?)```", section, flags=re.S)
    assert m, "the recipe's fenced bash block"
    out: list[str] = []
    for raw in m.group(1).splitlines():
        if out and out[-1].endswith("\\"):
            out[-1] = out[-1][:-1].rstrip() + " " + raw.strip()
        else:
            out.append(raw.rstrip())
    return [ln for ln in out if ln.strip()]


def recipe() -> tuple[str, str, str, str]:
    """(section text, the export line, the stage command, the run command) as the README spells them."""
    section = _section(README.read_text(encoding="utf-8"), HEADING)
    lines = _bash_lines(section)
    assert len(lines) == 3, lines
    export, stage, run = lines
    return section, export, stage, run


def _fill(command: str, *, slug: str, cells: list[str]) -> str:
    """The README's placeholders substituted: `$SHA`, `$APP`, `<slug>`, `<cells>`."""
    filled = command.replace("$SHA", SHA).replace("$APP", APP).replace("<slug>", slug).replace("<cells>", ",".join(cells))
    assert "<" not in filled and "$" not in filled, filled                  # every placeholder the page uses is one this test fills
    return filled


def _launcher_argv(command: str) -> list[str]:
    """The launcher's own argv out of a `modal run … scripts/modal/tinycap_served.py <argv>` command."""
    words = shlex.split(command)
    assert words[:2] == ["modal", "run"] and "scripts/modal/tinycap_served.py" in words, words
    return words[words.index("scripts/modal/tinycap_served.py") + 1:]


def _cells_for(slug: str) -> list[str]:
    """The paper roster's cell shape on a direct-launch slug's own bank: the unsteered model, the two prompted arms, then every
    fitted cell the hash table names under the slug (the README's rule for a fitted cell)."""
    fitted = sorted(C.EXPECTED_INTERVENTION_HASHES[slug])
    assert all(C.parse_launcher_cell(t).kind == C.CELL_KIND_FITTED for t in fitted)
    return ["none", "prompted__safety_refusal", "prompted__si_constitution", *fitted]


@pytest.fixture(scope="module")
def L():
    import test_tinycap_launcher as TL                                      # the modal stub + the by-path loader every launcher test uses
    return TL._load_launcher(TL._stub_modal(), app_name=APP, name="tinycap_served_readme_recipe")


def _replay(L, argv: list[str], **override) -> str:
    """The README's argv through the launcher's OWN argparse and `main` as its CPU dry run (test_front_door_dry_run's
    technique): `--pf` is a `modal run`-only local-entrypoint flag, stripped the same way; `override` reaches `main`."""
    a = L.cli_parser().parse_args([x for x in argv if x != "--pf"] + ["--dry-run"])
    kw = dict(sha=a.sha, pass1=a.pass1, pass2=a.pass2, cells=a.cells, modes=a.modes, controls=a.controls, model=a.model,
              app_name=a.app_name, serving_path=a.serving_path, allow_extra=a.allow_extra, expect_hashes=a.expect_hashes,
              sampling=a.sampling, dry_run=True)
    kw.update(override)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L.main(**kw)
    return buf.getvalue()


def test_the_recipe_spells_the_flags_a_run_needs_and_names_both_front_doors():
    """The page's command and table carry what a real run needs, and point at the front doors that compose it."""
    section, export, stage, run = recipe()
    assert export == f"export {C.MODAL_APP_NAME_ENV}=$APP", export                  # the launcher reads the variable at import
    stage_argv, run_argv = _launcher_argv(_fill(stage, slug="x", cells=[])), _launcher_argv(_fill(run, slug="x", cells=["none"]))
    assert "--upload-tree" in stage_argv and "--stage" in stage_argv and "--pass1" not in stage_argv and "--cells" not in stage_argv
    for flag in ("--pf", "--pass1", "--cells", "--modes", "--controls", "--sampling", "--serving-path"):
        assert flag in run_argv, (flag, run_argv)
    assert run_argv[run_argv.index("--modes") + 1] == C.THINKING_OFF
    # the paper's FULL sets (1,140 BigCodeBench ids, 3,981 BFCL ids) under the paper's protocol decoding, never a 200-item subset
    assert run_argv[run_argv.index("--controls") + 1] == f"{C.CONTROL_BIGCODEBENCH_FULL},{C.CONTROL_BFCL_FULL}" == "bigcodebench_full,bfcl_full"
    assert run_argv[run_argv.index("--sampling") + 1] == C.SAMPLING_PROTOCOL == "protocol"
    assert run_argv[run_argv.index("--serving-path") + 1] == "lora_native"
    for cmd in (stage, run):
        assert f"-n $APP" in cmd and "--app-name $APP" in cmd and "--sha $SHA" in cmd, cmd
    table = [ln for ln in section.splitlines() if ln.startswith("| `--")]
    flags = [ln.split("|")[1].strip() for ln in table]
    assert flags == ["`--upload-tree --stage`", "`--model`", "`--pf --pass1`", "`--cells`", "`--modes off`", "`--controls`",
                     "`--sampling protocol`", "`--serving-path lora_native`"], flags
    controls_row = next(ln for ln in table if ln.startswith("| `--controls`"))
    assert "bigcodebench_full,bfcl_full" in controls_row and "1,140" in controls_row and "3,981" in controls_row and "200" not in controls_row
    text = " ".join(table)
    assert "--allow-extra" in text and "contracts.MAX_GPU_CONTAINERS" in text and f"({C.MAX_GPU_CONTAINERS})" in text
    assert "--expect-hashes F" in text and "contracts.EXPECTED_INTERVENTION_HASHES" in text
    assert "contracts.BCBW_SLUGS" in text and "contracts.MODELS" in text and "thinking_modes" in text
    for slug in DIRECT_SLUGS + LANE_SLUGS:
        assert f"`{slug}`" in text, slug
    assert "scripts/bcb_bfcl_200_widen_launch.py --roster configs/capeval/bcb_bfcl_full_roster.yaml" in section
    assert "reproduce/07_bcb_bfcl.sh" in section and "qwen35_4b_demo" in section and Path(__file__).name in section


@pytest.mark.parametrize("slug", DIRECT_SLUGS)
def test_the_recipes_run_command_plans_one_container_per_cell_through_the_launcher(L, slug):
    """The README's run command, placeholders filled for a direct-launch slug the page names, through the launcher's own
    `cli_parser` + `main` as a dry run: every flag recognised, one GPU container per cell, every cell planned. `--pf` is a flag
    of the `modal run` entrypoint (`main`), not of the CPU dry run's argparse, and is stripped the way the front doors do."""
    _section_text, _export, _stage, run = recipe()
    assert L.APP_NAME == APP == L.app.name                                   # imported under the page's exported variable
    assert "pf" in inspect.signature(L.main).parameters                    # `--pf` reaches `main` through `modal run`
    cells = _cells_for(slug)
    assert len(cells) > C.MAX_GPU_CONTAINERS                                 # the paper's eight-arm shape on this bank exceeds the cap…
    within = cells[:C.MAX_GPU_CONTAINERS]                                   # …so the plain command is replayed within it
    argv = _launcher_argv(_fill(run, slug=slug, cells=within))
    plan = _replay(L, argv)
    assert f"[tinycap] dry run: {len(within)} GPU configuration container(s), pass 1" in plan, plan[-2000:]
    for tok in within:
        assert f'"cell": "{tok}"' in plan, (tok, plan[-2000:])
    assert f"[tinycap] app: {APP!r}" in plan and "serving path: lora_native" in plan
    # over the cap the page says `--allow-extra`: refused by name without it, every cell planned with it
    over = _launcher_argv(_fill(run, slug=slug, cells=cells))
    with pytest.raises(SystemExit, match="MAX_GPU_CONTAINERS"):
        _replay(L, over)
    plan = _replay(L, over + ["--allow-extra"])
    assert f"[tinycap] dry run: {len(cells)} GPU configuration container(s), pass 1" in plan, plan[-2000:]
    for tok in cells:
        assert f'"cell": "{tok}"' in plan, tok
    # the page's `--modes off` note: a row serving `off` alone refuses the default's ON leg; a row serving both plans it —
    # one container per cell AND per mode (twice the cells, so over the cap: `--allow-extra` rides along for the count)
    i = argv.index("--modes")
    without_modes = argv[:i] + argv[i + 2:]
    if C.THINKING_ON in C.model_row(slug).thinking_modes:
        plan = _replay(L, without_modes + ["--allow-extra"])
        assert f"[tinycap] dry run: {2 * len(within)} GPU configuration container(s), pass 1" in plan, plan[-2000:]
        assert plan.count('"mode": "on"') == len(within) and plan.count('"mode": "off"') == len(within), plan[-2000:]
    else:
        with pytest.raises(L.ThinkingUnservable, match="asks for thinking-ON"):
            _replay(L, without_modes)
    # the page's `--pass1` note: a REAL run (not the dry run, which plans regardless) without a pass plans nothing and returns
    j = argv.index("--pass1")
    out = _replay(L, argv[:j] + argv[j + 1:], dry_run=False)
    assert "GPU configuration container" not in out and '"cell":' not in out and "[tinycap] run id:" in out, out[-1500:]


def test_the_paper_rows_the_page_names_are_lane_rows_a_direct_launch_refuses_toward_the_wrapper(L):
    """The page says the paper's rows (the roster's model_row entries) are lane rows the harness serves only through the bcbw
    wrapper: each is in contracts.BCBW_SLUGS, and the README's command on one refuses by name, pointing at the wrapper."""
    import yaml
    roster = yaml.safe_load((REPO / "configs/capeval/bcb_bfcl_full_roster.yaml").read_text(encoding="utf-8"))
    rows = {w["model_row"] for w in roster["waves"].values()}
    assert set(LANE_SLUGS) <= rows and rows <= set(C.BCBW_SLUGS), (rows, C.BCBW_SLUGS)
    _section_text, _export, _stage, run = recipe()
    for slug in LANE_SLUGS:
        argv = _launcher_argv(_fill(run, slug=slug, cells=["none", "constv2-mix-full__lora_r4_ce"]))
        with pytest.raises(SystemExit, match="lane row .* launches only with the launch-time roster snapshot") as e:
            _replay(L, argv)
        assert "scripts/bcb_bfcl_200_widen_launch.py" in str(e.value), slug
    for slug in DIRECT_SLUGS:
        assert slug not in C.BCBW_SLUGS


def test_the_export_line_is_load_bearing(L):
    """The launcher names its Modal App at import from the exported variable and holds `--app-name` to it: another name on the
    flag refuses, which is why the page exports before both `modal run`s."""
    with pytest.raises(SystemExit, match="must be in the ENVIRONMENT"):
        L.check_app_name("tinycap-another-name")
    L.check_app_name(APP)
