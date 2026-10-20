"""The padded-static generator renders the FRONT DOOR's PER-VIEW steered families: one family per non-u0 phase view of a front-door key, named after the view, cut from the view's own wave pair, its cells
the production roster's (no cross-bank pin, no lane roster — the production roster names the view), so stage 06's
`target_static_padded_obj.sbatch <key> … all <view>` rows (lib/stage.py) have a config; a view the roster does not carry refuses by name;
qwen35_4b's `obj` / `obj_s<N>` families are byte-identical. The tree's front-door model runs the SINGLE-PHASE recipe (one wave `battery`, the row's declared wave, + two dose rungs, no u0), so EVERY one of its phase views is a per-view family — battery included — each cut from a wave
pair that carries the view's own `none` (the unsteered legs the phased recipe's u0 carried), and it has NO prompted-only family. The sbatch's family positional resolves a front-door family through the
generator (`--resolve-family`), the graders read the widened family table, and the post-standup geometry pin runs without a literal
width pin for such a family.

Synthetic fixtures: the test model's block as 05 --step roster prints it, the wave pairs as make_full22_configs renders them from that
block (placeholder pins), a scratch checkout for the resolver."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
LANE = REPO / "scripts" / "sdfbase" / "full22" / "static_padding"
sys.path.insert(0, str(Path(__file__).resolve().parent))
for p in (LANE, REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase" / "full22" / "sampled", REPO / "reproduce" / "lib", REPO / "src",
          REPO / "scripts" / "registry"):                                                     # grade_daemon: the daemon's table-aware padded-run split
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import f22_roster as fr  # noqa: E402
import make_full22_configs as wave  # noqa: E402
import make_padding_configs as gen  # noqa: E402
import stage as ST  # noqa: E402
from test_full22_front_door_targets import KEY, PY, VIEWS, front_door_roster  # noqa: E402

ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}
STEERED = ("battery", "d0p75", "d0p75_relu")   # qwen35_4b_demo (configured, not run): every phase view of the table (single-phase: no u0; the one wave is the row's DECLARED `battery`, demo-import M3), in its order — the table excludes none of them from padding


def _fresh(roster: Path):
    """The generator re-executed with the synthetic roster as its PRODUCTION roster (the import-time KEYS and GRADER_TARGETS read it)."""
    spec = importlib.util.spec_from_file_location("_gen_fd_views_fresh", LANE / "make_padding_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.PRODUCTION_ROSTER = roster
    mod.KEYS = (*mod.HAND_LISTED_KEYS, *(k for k in mod.PROMPTED_ONLY_KEYS if mod._roster_names(k, mod.FRONT_DOOR_SOURCE_VIEW)))
    mod.GRADER_TARGETS = mod.output_keys()      # the grader's table IS output_keys(); re-derived over the synthetic roster as KEYS is
    return mod


def _tree(tmp_path: Path, views=VIEWS):
    """A REPO-shaped tmp tree: the synthetic roster (production + the block) and the REAL generator's wave pair per view."""
    roster = front_door_roster(KEY, views)
    tree = tmp_path / "repo"
    (tree / "configs" / "capeval" / "manifests").mkdir(parents=True)
    (tree / "configs" / "sdfbase" / "static_padding").mkdir(parents=True)
    rpath = tree / "configs" / "sdfbase_full22_roster.yaml"
    rpath.write_text(yaml.safe_dump(roster, sort_keys=False), encoding="utf-8")
    pins = json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text(encoding="utf-8"))
    pins[KEY] = {sid: {"object_sha256": f"{i:064x}", "source": f"synthetic/{sid}"} for i, sid in enumerate(
        sorted({s for v in views for s in roster["targets"][KEY]["views"][v]["cells"].values() if s}), start=1)}
    for v in views:
        (tree / "configs" / f"sdfbase_full22_{KEY}_{v}.yaml").write_text(wave.render_config(roster, pins, KEY, v), encoding="utf-8")
        (tree / "configs" / "capeval" / "manifests" / f"petri_full22_{KEY}_{v}.yaml").write_text(wave.render_manifest(roster, KEY, v), encoding="utf-8")
    mod = _fresh(rpath)
    mod.REPO = tree; mod.CONFIG_DIR = tree / "configs" / "sdfbase" / "static_padding"; mod.MANIFEST_DIR = tree / "configs" / "capeval" / "manifests"
    return mod, roster, tree


def test_the_per_view_families_derive_from_the_table_and_the_hand_listed_table_is_untouched(tmp_path):
    assert gen.FAMILIES == (None, "obj", "obj_s2", "obj_s3") and gen.families_of("qwen35_4b") == gen.FAMILIES
    assert gen.FRONT_DOOR_VIEW_FAMILIES == {KEY: STEERED} and gen.front_door_view_families("qwen35_4b") == ()
    assert gen.ALL_FAMILY_NAMES[:len(gen.FAMILIES)] == gen.FAMILIES and set(gen.ALL_FAMILY_NAMES) == set(gen.FAMILIES) | set(STEERED)
    assert all(gen.front_door_view_excluded(KEY, v) is None for v in STEERED)                     # the table excludes none of this key's views
    # a view the table EXCLUDES from padding by ruling (`exclude_families: [padding]` + its reason) is no per-view family — omitted from
    # the derivation and named by its reason; no model of the table declares one today, so the ruling's shape rides a SYNTHETIC table
    t = tmp_path / "excluded.yaml"
    t.write_text(yaml.safe_dump({"schema": 1, "models": {"synth_fd": {"view_legacy_prefix": "synth_fd", "views": {
        "u0": {"cells": list(gen.CELLS)}, "s2": {"cells": ["none", "x"]},
        "s1": {"cells": ["none", "y"], "exclude_families": ["padding"], "exclude_reason": "ruling: the LoRA parents never run the ladder"}}}}}))   # a PHASED synthetic row
    assert gen.front_door_view_families("synth_fd", t) == ("s2",) and gen.front_door_view_excluded("synth_fd", "s2", t) is None
    assert gen.front_door_view_excluded("synth_fd", "s1", t) == "ruling: the LoRA parents never run the ladder"
    assert all(gen.is_front_door_family(f) for f in STEERED) and not any(gen.is_front_door_family(f) for f in gen.FAMILIES)
    assert not gen.is_front_door_family("u0")                                                    # u0 is the phased recipe's prompted-only source, never a per-view family
    for f in STEERED:
        assert gen.view_of(f) == f and gen.seed_of(f) is None and gen.pads_of(f) == gen.PADS and gen.seg_of(f) == f"_{f}"
        assert gen.padded_roster(KEY, f) == gen.PRODUCTION_ROSTER and gen.private_run(KEY, 0, f) == f"pad_static_{KEY}_{f}_p0"
        p = gen.padded_config(KEY, 8192, f)
        assert p.name == f"sdfbase_full22_{KEY}.padstatic_{f}_p8192.yaml" and (gen.key_of(p), gen.pad_of(p), gen.family_of(p)) == (KEY, 8192, f)
        assert gen.padded_manifest(KEY, f).name == f"petri_full22_{KEY}.padstatic_{f}.yaml"
    for pad in gen.OBJ_PADS:                                                                     # the hand-listed names parse as before
        p = gen.padded_config("qwen35_4b", pad, "obj_s2")
        assert (gen.key_of(p), gen.pad_of(p), gen.family_of(p)) == ("qwen35_4b", pad, "obj_s2")
    with pytest.raises(ValueError, match="does not know"):
        gen.family_of(Path(f"sdfbase_full22_{KEY}.padstatic_bogus_p0.yaml"))
    with pytest.raises(ValueError, match="pass the key"):
        gen.cells_of("battery")
    # today: the roster carries none of the views, so no per-view family renders and every one is pending by name; a single-phase key has no
    # prompted-only family (families_of lists its per-view families alone)
    carried = tuple(v for v in STEERED if gen._roster_names(KEY, v))
    assert gen.families_of(KEY) == carried and KEY in gen.SINGLE_PHASE_KEYS and KEY not in gen.PROMPTED_ONLY_KEYS
    assert set(gen.front_door_view_pending()) == {(k, v) for k, fams in gen.FRONT_DOOR_VIEW_FAMILIES.items() for v in fams if not gen._roster_names(k, v)}
    if "battery" not in carried:
        with pytest.raises(SystemExit, match="does not carry this view yet"):
            gen.render_config(KEY, 0, "battery")
    with pytest.raises(SystemExit, match="not a front-door key"):
        gen.render_config("qwen35_4b", 0, STEERED[0])                                           # a per-view family name, under a hand-listed key
    with pytest.raises(SystemExit, match="pads its steered views through its PER-VIEW families"):
        gen.render_config(KEY, 0, "obj")
    with pytest.raises(SystemExit, match="is not a steered phase view of this key"):
        gen._require_front_door_family(KEY, "obj")


def test_a_front_door_view_the_roster_names_renders_its_per_view_family_from_the_views_own_wave_pair(tmp_path, monkeypatch):
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")     # the documented test knob for a plan over views the PRODUCTION roster lacks
    mod, roster, tree = _tree(tmp_path)
    assert mod.KEYS == ("qwen35_4b",) and mod.families_of(KEY) == STEERED and mod.front_door_view_pending() == ()   # the demo never joins KEYS (no prompted-only stores)
    assert mod.output_keys() == mod.GRADER_TARGETS == ("qwen35_4b", KEY)                        # … but the grader admits it: the roster names its views, so it renders
    for fam in STEERED:
        want_cells = ["none", *roster["targets"][KEY]["views"][fam]["cells"]]        # the single-phase recipe: the view's own `none` + the roster's cells (the one universe)
        assert list(mod.cells_of(fam, KEY)) == want_cells
        for pad in mod.PADS:
            text = mod.render_config(KEY, pad, fam)
            cfg = yaml.safe_load(text)
            assert cfg["explicit_cells"] == want_cells and cfg["private_run"] == f"pad_static_{KEY}_{fam}_p{pad}" and cfg["out_dir"] == f"out/pad_static_{KEY}_{fam}_p{pad}"
            assert cfg["defenses_dir"] == f"out/full22/{KEY}/cells_view/{fam}" and cfg["attacks"]["static_max_new_tokens"] == gen.STATIC_CAP
            assert (cfg["model"], cfg["model_revision"]) == ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
            assert ("static_padding" in cfg["attacks"]) == (pad > 0) and (cfg["attacks"].get("static_padding") == gen.padding_block(pad) if pad else True)
            assert text.startswith(f"# sdfbase_full22_{KEY}.padstatic_{fam}_p{pad}.yaml") and f"(STEERED per-view family `{fam}`)" in text
            assert f"configs/sdfbase_full22_{KEY}_{fam}.yaml" in text and "no literal pin" in text and "GEOMETRY_PIN_S = 5" not in text
            assert ("arditi_allpos" in text) == (fam == "battery") and "padobj_roster" not in text and "no cross-bank pin" in text
        man_text = mod.render_manifest(KEY, fam)
        man = yaml.safe_load(man_text)["petri"]
        assert man["cells"] == mod.harness_cells(mod.cells_of(fam, KEY)) and f"OVERLAY of petri_full22_{KEY}_{fam}.yaml" in man_text
        assert man["chat_template_hash"] == "a4aee8afcf2e0711" and man["target_model"] == "Qwen/Qwen3.5-4B"
    # the outputs: six per phase view (the demo has NO prompted-only six — a single-phase key); no lane roster for a front-door key. The hand-listed
    # key's pair is not in the tmp tree (its 27 outputs are pinned by test_static_padding_wave), so KEYS is emptied of it
    mod.KEYS = ()
    names = [p.name for p, _ in mod.outputs()]
    fd = [n for n in names if KEY in n]
    assert names == fd and len(fd) == 6 * len(STEERED) and not any("roster" in n for n in fd) and not any(re.search(r"\.padstatic_p\d+\.yaml$|\.padstatic\.yaml$", n) for n in fd)
    assert fd[:6] == [f"sdfbase_full22_{KEY}.padstatic_{STEERED[0]}_p{p}.yaml" for p in mod.PADS] + [f"petri_full22_{KEY}.padstatic_{STEERED[0]}.yaml"]
    assert fd[6:12] == [f"sdfbase_full22_{KEY}.padstatic_d0p75_p{p}.yaml" for p in mod.PADS] + [f"petri_full22_{KEY}.padstatic_d0p75.yaml"]
    # the wave pair and the roster must agree: a stale pair (cells drifted) refuses, never pads
    stale = tree / "configs" / f"sdfbase_full22_{KEY}_battery.yaml"
    stale.write_text(re.sub(r"^- constv2ctl2-full__psr_ce_all\n", "", stale.read_text(), flags=re.M), encoding="utf-8")
    with pytest.raises(SystemExit, match="the wave pair is stale"):
        mod.render_config(KEY, 0, "battery")
    # stage.py's rows for the view name exactly this family and this config
    rows = ST.unit_plan(KEY, "battery", table=str(REPO / "reproduce" / "models.yaml"), families=["padding"], sha="a" * 40)
    assert [r["args"][-1] for r in rows] == ["battery"] * 5
    for r in rows:
        key, sha, window, pad, fams, shards, cells, fam = r["args"]
        assert f"configs/sdfbase/static_padding/sdfbase_full22_{key}.padstatic_{fam}_p{int(pad)}.yaml" == str(mod.padded_config(key, int(pad), fam).relative_to(tree))


def test_a_roster_carrying_some_views_renders_those_and_refuses_the_others_by_name(tmp_path):
    mod, roster, tree = _tree(tmp_path, views=("battery", "d0p75"))
    assert mod.families_of(KEY) == ("battery", "d0p75") and mod.front_door_view_pending() == ((KEY, "d0p75_relu"),)
    assert yaml.safe_load(mod.render_config(KEY, 0, "battery"))["private_run"] == f"pad_static_{KEY}_battery_p0"
    with pytest.raises(SystemExit, match="does not carry this view yet"):                        # pending by name: not rendered, not refused as a family
        mod.render_config(KEY, 0, "d0p75_relu")
    with pytest.raises(SystemExit, match="does not carry this view yet"):
        mod.cells_of("d0p75_relu", KEY)
    mod.KEYS = ()
    names = [p.name for p, _ in mod.outputs()]
    assert len(names) == 6 + 6 and f"petri_full22_{KEY}.padstatic_battery.yaml" in names and f"petri_full22_{KEY}.padstatic_d0p75.yaml" in names and not any("padstatic_d0p75_relu" in n for n in names)


def _scratch_checkout(tmp_path: Path, tree: Path) -> Path:
    """A checkout the sbatch's resolver can run in: the generator + its readers, the models table (+ a SYNTHETIC front-door key whose s1
    view the table EXCLUDES from padding — no model of the real table declares an exclusion today, and the resolver's
    refusal by name is pinned against one), the synthetic roster."""
    co = tmp_path / "checkout"
    for rel in ("scripts/sdfbase/full22/static_padding/make_padding_configs.py", "scripts/sdfbase/full22/f22_roster.py", "reproduce/lib/models.py"):
        (co / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, co / rel)
    table = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text(encoding="utf-8"))
    table["models"]["synth_fd"] = {"view_legacy_prefix": "synth_fd", "views": {"u0": {"cells": list(gen.CELLS)}, "s2": {"cells": ["none", "x"]},
                                   "s1": {"cells": ["none", "y"], "exclude_families": ["padding"], "exclude_reason": "ruling: the LoRA parents never run the ladder"}}}
    (co / "reproduce" / "models.yaml").write_text(yaml.safe_dump(table, sort_keys=False), encoding="utf-8")
    (co / "configs").mkdir(exist_ok=True)
    shutil.copy(tree / "configs" / "sdfbase_full22_roster.yaml", co / "configs" / "sdfbase_full22_roster.yaml")
    return co


def test_resolve_family_answers_the_sbatch_for_every_family_kind(tmp_path):
    mod, roster, tree = _tree(tmp_path, views=("battery", "d0p75_relu"))
    co = _scratch_checkout(tmp_path, tree)
    script = co / "scripts" / "sdfbase" / "full22" / "static_padding" / "make_padding_configs.py"

    def resolve(key, fam):
        return subprocess.run([PY, str(script), "--resolve-family", key, fam], capture_output=True, text=True, cwd=co, env=ENV)
    r = resolve(KEY, "battery"); assert r.returncode == 0 and r.stdout.split() == ["battery", "configs/sdfbase_full22_roster.yaml", "battery"], r.stderr   # the declared wave: a per-view family by the table, not by the grammar
    r = resolve(KEY, "d0p75_relu"); assert r.returncode == 0 and r.stdout.split() == ["d0p75_relu", "configs/sdfbase_full22_roster.yaml", "d0p75_relu"], r.stderr
    r = resolve(KEY, "d0p75"); assert r.returncode == 2 and "does not carry this view yet" in r.stderr                                   # a view the roster lacks: pending by name
    r = resolve("synth_fd", "s1"); assert r.returncode == 2 and "EXCLUDED for this view by ruling" in r.stderr and "never run the ladder" in r.stderr   # An excluded view pads nothing
    r = resolve(KEY, "obj"); assert r.returncode == 2 and "hand-listed key's steered family" in r.stderr
    r = resolve("qwen35_4b", "obj"); assert r.returncode == 0 and r.stdout.split() == ["padobj", "configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padobj_roster.yaml", "s1"]
    r = resolve("qwen35_4b", "obj_s3"); assert r.returncode == 0 and r.stdout.split()[0::2] == ["padobj_s3", "s3"]
    r = resolve("qwen35_4b", "s2"); assert r.returncode == 2 and "not a front-door key" in r.stderr
    r = resolve(KEY, "bogus"); assert r.returncode == 2 and "unknown padded-static config family" in r.stderr
    # the sbatch's family block, driven by that resolver: a front-door family selects the production view + roster, no literal width pin
    from test_static_padding_wave import _family_block
    harness = ('set -u; f22_fatal(){ echo "FATAL: $*"; exit 1; }; f22_driver_venv(){ :; }; f22_log(){ echo "$*"; }; '
               f'MPY={PY}; REPO_DIR={co}; PAD_DIR=scripts/sdfbase/full22/static_padding; export PYTHONPATH={REPO / "src"}; GEOMETRY_PIN_S=5; ')
    prog = harness + f'KEY={KEY}; CONFIG_FAMILY=battery\n' + _family_block() + '\necho "view=$VIEW roster=$FD_ROSTER reg=$FD_REGISTRY_VIEW pin=[$GEOMETRY_PIN_S]"'
    rc = subprocess.run(["bash", "-c", prog], capture_output=True, text=True)
    assert rc.returncode == 0 and rc.stdout.strip().splitlines()[-1] == "view=battery roster=configs/sdfbase_full22_roster.yaml reg=battery pin=[]", rc.stdout + rc.stderr
    prog = harness + 'KEY=synth_fd; CONFIG_FAMILY=s1\n' + _family_block()
    rc = subprocess.run(["bash", "-c", prog], capture_output=True, text=True)
    assert rc.returncode == 1 and "not a config family of this lane" in rc.stdout and "EXCLUDED for this view by ruling" in rc.stderr   # An excluded view pads nothing
    prog = harness + f'KEY={KEY}; CONFIG_FAMILY=d0p75\n' + _family_block()
    rc = subprocess.run(["bash", "-c", prog], capture_output=True, text=True)
    assert rc.returncode == 1 and "not a config family of this lane" in rc.stdout and "does not carry this view yet" in rc.stderr        # a view the roster lacks
    prog = harness + 'KEY=qwen35_4b; CONFIG_FAMILY=obj_s2\n' + _family_block() + '\necho "view=$VIEW pin=[$GEOMETRY_PIN_S]"'
    rc = subprocess.run(["bash", "-c", prog], capture_output=True, text=True)
    assert rc.returncode == 0 and rc.stdout.strip() == "view=padobj_s2 pin=[5]"                       # the hand-listed arms never call the resolver
    # the sbatch wires the resolver's answers: the roster override right after the pinned literal, the registry view default, the preflight flag
    src = (LANE / "target_static_padded_obj.sbatch").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert '[ -z "${FD_ROSTER:-}" ] || F22_ROSTER="$FD_ROSTER"' in code and 'PAD_REGISTRY_VIEW_DEFAULT="${FD_REGISTRY_VIEW:-main}"' in code
    assert '${GEOMETRY_PIN_S:+--expect-served-s "$GEOMETRY_PIN_S"}' in code and 'PIN_S = int(sys.argv[5]) if sys.argv[5] else None' in code
    assert code.index('F22_ROSTER="configs/sdfbase/static_padding/') < code.index('|| F22_ROSTER="$FD_ROSTER"') < code.index('f22_init "$WAVE_SHA"')


def test_the_family_block_resolves_a_front_door_family_before_f22_init_with_repo_dir_unset(tmp_path):
    """The family case runs BEFORE f22_init (the lane roster it selects must stand first), so REPO_DIR is not bound yet: under `set -u` the
    resolver's `cd "$REPO_DIR"` killed the process substitution, `read` saw EOF and the `|| f22_fatal` named the family as unknown —
    padobj-qwen35_4b_demo_battery-p0 (cluster A 981270) died in 54 s on a family the generator resolves. The block reads
    `${REPO_DIR:-$PWD}`, as the registry block already does: the job's working directory IS the checkout (the funnel submits
    `cd <checkout> && sbatch`, reproduce/lib/submit_twin.py). Rendered here as the job has it — REPO_DIR unset, the cwd the scratch
    checkout — and pinned statically over both padding sbatch scripts: no line before f22_init reads a variable f22_init binds without a
    default."""
    mod, roster, tree = _tree(tmp_path, views=("battery", "d0p75_relu"))
    co = _scratch_checkout(tmp_path, tree)
    from test_static_padding_wave import _family_block
    env = {k: v for k, v in ENV.items() if k != "REPO_DIR"}
    harness = ('set -u; f22_fatal(){ echo "FATAL: $*"; exit 1; }; f22_driver_venv(){ :; }; f22_log(){ echo "$*"; }; '
               f'MPY={PY}; PAD_DIR=scripts/sdfbase/full22/static_padding; GEOMETRY_PIN_S=5; ')
    prog = harness + f'KEY={KEY}; CONFIG_FAMILY=battery\n' + _family_block() + '\necho "view=$VIEW roster=$FD_ROSTER reg=$FD_REGISTRY_VIEW pin=[$GEOMETRY_PIN_S]"'
    rc = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, cwd=co, env=env)
    assert "unbound variable" not in rc.stderr, rc.stderr                                             # the shell never refuses what the generator answers
    assert rc.returncode == 0 and rc.stdout.strip().splitlines()[-1] == "view=battery roster=configs/sdfbase_full22_roster.yaml reg=battery pin=[]", rc.stdout + rc.stderr
    # a refusal stays the generator's, by name, with its reason on stderr — never the shell's (a view the roster lacks reaches the
    # resolver arm; `obj` would take the hand-listed literal arm and never ask)
    prog = harness + f'KEY={KEY}; CONFIG_FAMILY=d0p75\n' + _family_block()
    rc = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, cwd=co, env=env)
    assert rc.returncode == 1 and "not a config family of this lane" in rc.stdout and "does not carry this view yet" in rc.stderr, rc.stdout + rc.stderr
    assert "unbound variable" not in rc.stderr, rc.stderr
    for name in ("target_static_padded_obj.sbatch", "target_static_padded.sbatch"):
        code = [l for l in (LANE / name).read_text().splitlines() if not l.lstrip().startswith("#")]
        pre_init = "\n".join(code[:next(i for i, l in enumerate(code) if l.startswith('f22_init "'))])
        bare = re.findall(r'\$(?:REPO_DIR|SHARED|WAVE_ROOT|CLUSTER)\b|\$\{(?:REPO_DIR|SHARED|WAVE_ROOT|CLUSTER)\}', pre_init)
        assert not bare, (name, bare)


def test_the_geometry_pin_without_a_literal_holds_the_derivation_and_names_a_model_without_a_geometry_row(tmp_path):
    """The post-standup pin (the sbatch's heredoc, fed the same five arguments) with an EMPTY fifth argument — a front door's per-view
    family: no literal width pin — skips the literal checks only; the derivation, the engine, the zero adapter and the chain still hold;
    a model with no serving-geometry row refuses by name, never a KeyError."""
    from test_static_padding_wave import WINDOW, _pin_block
    good = {"engine": "lora_native", "id_name_map": {"1": "none__lora0", "2": "si_constv2_mix_full__psr_ce_all"},
            "geometry": {"adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": 7, "to_S": 5, "D": 4096, "ceiling": 524287}]}}

    def run(ep: dict, key: str, S: int, pin: str, L: int = WINDOW):
        ep_path = tmp_path / "endpoint.json"; ep_path.write_text(json.dumps(ep))
        r = subprocess.run([PY, "-", str(ep_path), key, str(L), str(S), pin], input=_pin_block(), capture_output=True, text=True, cwd=REPO,
                           env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
        return r.returncode, r.stdout + r.stderr
    rc, out = run(good, "qwen35_4b", 5, "5"); assert rc == 0 and "== the literal pin 5" in out, out
    rc, out = run(good, "qwen35_4b", 5, ""); assert rc == 0 and "no literal pin: a front door's per-view family" in out, out          # the derivation IS the pin
    rc, out = run(good, "qwen35_4b", 7, ""); assert rc == 1 and "attested max_num_seqs 7 != 5" in out and "pinned width" not in out, out   # the other checks still hold
    # the front-door model: the pin FIRST resolves the SELECTED module set from the models table's promoted pick (the one source)
    # — qwen35_4b_demo's pick is the paper's (`selected_by: paper`, o_proj -> wo), so the selection passes and the NEXT
    # gate refuses the UNMEASURED budget by name (a default row is not a measurement; tests/test_site_geometry_overlay.py runs this pin
    # against a folded overlay per set); a foreign key refuses at the resolver
    rc, out = run(good, KEY, 5, "")
    assert rc == 1 and "batched-token budget source 'default_llama_certified' (393,216) is not a measurement" in out and "Traceback" not in out, out
    assert "no promoted LoRA architecture pick" not in out and "no pinned LoRA max module dim" not in out, out      # past the selection, refused at the budget
    rc, out = run(good, "no_such_model", 5, ""); assert rc == 1 and "unknown model key 'no_such_model'" in out and "Traceback" not in out, out
    code = "\n".join(l for l in (LANE / "target_static_padded_obj.sbatch").read_text().splitlines() if not l.lstrip().startswith("#"))
    assert "g.BUDGET_BY_MODEL[" not in code and "g.budget_for(key, SET)" in code and "g.rows_for(key, SET)" in code    # the SELECTED set
    assert "pf.selected_module_set(key)" in code                                                                      # the one source


def test_the_preflight_and_the_sampled_planner_size_through_the_budget_resolver():
    """Every sizing site resolves through geometry.budget_for / rows_for — the ONE lookup that covers the
    paper-companion ONBOARDED keys — never BUDGET_BY_MODEL indexed directly. What the resolver holds today, precisely: qwen35_4b_demo (the front
    door's onboarded model, configured and not run) has NO row of its own — the Llama-certified default, source default_llama_certified, with
    the resolver's warning — and no pinned widest module in LORA_MAX_MODULE_DIM_BY_MODEL. The steered padding preflight's check 5 refuses it
    by name: it resolves the SELECTED MODULE SET first (the demo's pick is the paper's, `selected_by: paper`,
    o_proj -> wo, so the selection passes), and refuses at the BUDGET next: a provisional or default budget is
    not a measurement, and steered padding sizes only from one — until the probe's verdict folds the per-set rows into the site geometry
    overlay (tests/test_site_geometry_overlay.py covers the folded path)."""
    import padding_preflight as pf
    from sisteer.capeval.serving import geometry as g
    import submit_sampled as ss
    rb = g.budget_for(KEY)
    assert KEY in g.ONBOARDED_MODEL_KEYS and not rb.per_model and rb.source == g.DEFAULT_BUDGET_SOURCE and g.rows_for(KEY).budget is None
    assert KEY not in g.BUDGET_BY_MODEL and KEY not in g.LORA_MAX_MODULE_DIM_BY_MODEL
    # the preflight: the paper key's expectation is unchanged (qwen35_4b's `paper` pick on o_proj -> wo, its measured row, S 5 at the
    # window); the front-door key refuses BY NAME at the BUDGET — STEERED: its promoted pick (the paper's, wo) resolves and the selected
    # set has no measured row; UNSTEERED no pick is demanded — the zero adapter's class sizes it — and with no record the
    # resolver holds the default row, which every padded standup refuses
    from test_static_padding_wave import WINDOW
    geo = pf.geometry_expectation("qwen35_4b", WINDOW)
    assert geo["served_S"] == 5 and geo["module_set"] == "wo" and geo["budget_source"] == g.MEASURED_BUDGET_SOURCE and geo["budget_per_model"] and geo["batched_token_budget"] == g.budget_for("qwen35_4b").batched_token_budget
    assert pf.selected_module_set(KEY)[0] == "wo" and pf.selected_module_set(KEY)[1]["selected_by"] == "paper"
    with pytest.raises(ValueError, match=f"{KEY}: batched-token budget source is 'default_llama_certified' \\(393,216\\), not a measurement.*selected module set 'wo'"):
        pf.geometry_expectation(KEY, WINDOW, steered=True)
    with pytest.raises(ValueError, match=f"{KEY}: batched-token budget source is 'default_llama_certified' \\(393,216\\), not a measurement.*unsteered default"):
        pf.geometry_expectation(KEY, WINDOW)
    with pytest.raises(ValueError, match="batched-token budget source is 'default_llama_certified' \\(393,216\\), not a measurement.*selected module set 'wo'"):
        pf.geometry_expectation(KEY, WINDOW, module_set="wo")                    # a selection with no fold: the unmeasured-budget refusal
    rec, problems = pf.check_geometry(KEY, WINDOW, None)
    assert len(problems) == 1 and "not a measurement" in problems[0] and KEY in problems[0] and "no promoted LoRA architecture pick" not in problems[0]
    steered_problem = pf.check_geometry(KEY, WINDOW, None, steered=True)[1][0]        # the pick resolves; refused at the budget
    assert "not a measurement" in steered_problem and "selected module set 'wo'" in steered_problem and "no promoted LoRA architecture pick" not in steered_problem
    with pytest.raises(KeyError, match="unknown model key"):
        pf.geometry_expectation("no_such_model", WINDOW)
    src = (LANE / "padding_preflight.py").read_text()
    assert "g.BUDGET_BY_MODEL[" not in src and "g.budget_for(model_key, module_set)" in src      # the SELECTED set's row
    # submit_sampled's planner: an UNSTEERED view (a prompted-only wave: base + prompted only — the phased recipe's u0 shape, pasted under the
    # demo's key by hand since the single-phase demo declares none) sizes from the resolver's row (the default 393,216 at every phase window;
    # a HYBRID model, the HYBRID_SEQS_CAP applies) through the resolver's unsteered default and SAYS so; the STEERED view (the demo's one wave
    # battery) sizes from the SELECTED module set (the paper's pick, wo; tests/test_sampled_planner_module_set.py
    # covers the certified-record and pending-pick cases): with no record the planner holds the same default row for that set, and the padded
    # standups refuse it at their preflight (above), never the sampled planner, which only sizes
    roster = front_door_roster(KEY, ("battery",))
    roster["targets"][KEY]["views"]["u0"] = {"config": f"configs/sdfbase_full22_{KEY}_u0.yaml", "manifest": f"configs/capeval/manifests/petri_full22_{KEY}_u0.yaml",
                                             "cells": {"prompted__safety_refusal": None, "prompted__si_constitution": None}}   # the SYNTHETIC prompted-only block
    tag = f"{KEY}_u0"
    orig = ss.split_target
    ss.split_target = lambda t: (KEY, t[len(KEY) + 1:]) if t.startswith(KEY + "_") else orig(t)
    try:
        p = ss.plan(roster, [tag], {tag: "cluster_b"}, 9)
        p1 = ss.plan(roster, [f"{KEY}_battery"], {f"{KEY}_battery": "cluster_b"}, 9)
    finally:
        ss.split_target = orig
    served = {r["key"]: r for r in p["served"]}[tag]
    assert served["budget"] == rb.batched_token_budget == 393_216 and served["hybrid"] is True and served["geom"]["static_base"] == (4864, 393_216 // 4864)
    assert served["module_set"] == "wo" and p["skipped"] == []           # the promoted pick (the paper's, wo) is reported; the pending-pick note belongs to a table without one (test_sampled_planner_module_set)
    assert any("UNMEASURED" in j.est_h for j in p["jobs"] if j.key == tag)                   # no greedy anchor for a front-door model, said so
    steered_row = {r["key"]: r for r in p1["served"]}[f"{KEY}_battery"]
    assert steered_row["module_set"] == "wo" and steered_row["budget"] == 393_216 and steered_row["hybrid"] is True   # the selected set (the paper's pick), the resolver's default row
    with pytest.raises(SystemExit, match="unknown model key 'no_such_model'"):
        ss.split_target = lambda t: ("no_such_model", "battery") if t == "x_battery" else orig(t)      # the copied block's one wave (battery)
        try:
            roster["targets"]["no_such_model"] = roster["targets"][KEY]
            ss.plan(roster, ["x_battery"], {"x_battery": "cluster_b"}, 9)
        finally:
            ss.split_target = orig
    src = (REPO / "scripts" / "sdfbase" / "full22" / "sampled" / "submit_sampled.py").read_text()
    # no direct index; the resolver — asked for the SELECTED module set on an onboarded model (the pick's source is
    # padding_preflight.selected_module_set; tests/test_sampled_planner_module_set.py runs it)
    assert "BUDGET_BY_MODEL[" not in src and "g.budget_for(key, module_set)" in src and "g.rows_for(key, module_set)" in src and "pf.selected_module_set(key)" in src

def test_the_graders_read_the_widened_family_table():
    src = (LANE / "grade_padded_static.py").read_text()
    assert "CONFIG_FAMILIES = tuple(gen.ALL_FAMILY_NAMES)" in src and "gen.cells_of(family, key)" in src
    src = (LANE / "degeneracy_items.py").read_text()
    assert "CONFIG_FAMILIES = tuple(gen.ALL_FAMILY_NAMES)" in src and "gen.cells_of(config_family, key)" in src
    daemon = (REPO / "scripts" / "registry" / "grade_daemon.py").read_text()
    m = re.search(r'_PADDED_RUN_RE = re\.compile\(r"(.*)"\)', daemon)
    assert m and "u0|s[0-9]+r?p?|s[0-9]+rl|dm?[0-9]+(?:p[0-9]+)?(?:_relul?)?" in m.group(1)       # the phase-view grammar, restated (s2r / s2rl and _relul)
    R = re.compile(m.group(1))
    for run, key, fam in (("pad_static_qwen35_4b_obj_s2_p65536", "qwen35_4b", "obj_s2"), (f"pad_static_{KEY}_s2_p0", KEY, "s2"),
                          (f"pad_static_{KEY}_d0p75_relu_p8192", KEY, "d0p75_relu"), (f"pad_static_{KEY}_p0", KEY, None),
                          (f"pad_static_{KEY}_d0p75_relul_p8192", KEY, "d0p75_relul"), (f"pad_static_{KEY}_s2rl_p4096", KEY, "s2rl"),
                          (f"pad_static_{KEY}_s2r_p4096", KEY, "s2r")):   # PADDEDRELUL
        mm = R.match(run); assert mm and (mm.group("key"), mm.group("fam")) == (key, fam), run
    # every phase view the models table declares is a run name the daemon splits correctly: a grammar view by the regex alone; the row's
    # DECLARED wave (battery, outside every grammar — demo-import M3) by the daemon's split through the models table (_padded_key_family),
    # the regex leaving its `fam` group empty
    import grade_daemon as GD
    for k, fams in gen.FRONT_DOOR_VIEW_FAMILIES.items():
        for f in fams:
            run = f"pad_static_{k}_{f}_p16384"
            mm = R.match(run); assert mm, run
            if gen.PHASE_VIEW_RE.match(f):
                assert (mm.group("key"), mm.group("fam")) == (k, f), run
            else:
                assert mm.group("fam") is None and f == gen._models_reader().declared_wave(k), run
            assert GD._padded_key_family(run) == (k, f), run


def test_the_geometry_pin_selects_the_module_set_with_neither_torch_nor_transformers_importable(tmp_path):
    """The geometry pin's module-set selection (padding_preflight.selected_module_set) reads the vocabulary from its stdlib leaf,
    sisteer.extraction.lora_module_sets — lora_modules itself imports torch and sisteer.models (transformers), which the suite's
    CPU/stdlib venv lacks. Run the real pin block with BOTH blocked (``sys.modules[name] = None`` turns their import into an
    ImportError): the paper key still resolves its `wo` pick and passes, and the test model still refuses BY NAME at the
    selection."""
    from test_static_padding_wave import WINDOW, _pin_block
    good = {"engine": "lora_native", "id_name_map": {"1": "none__lora0", "2": "si_constv2_mix_full__psr_ce_all"},
            "geometry": {"adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": 7, "to_S": 5, "D": 4096, "ceiling": 524287}]}}
    ep_path = tmp_path / "endpoint.json"; ep_path.write_text(json.dumps(good))
    blocked = "import sys; sys.modules['torch'] = None; sys.modules['transformers'] = None; exec(sys.stdin.read())"

    def run(key: str, S: int, pin: str):
        r = subprocess.run([PY, "-c", blocked, str(ep_path), key, str(WINDOW), str(S), pin], input=_pin_block(), capture_output=True,
                           text=True, cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
        return r.returncode, r.stdout + r.stderr
    rc, out = run("qwen35_4b", 5, "")
    assert rc == 0 and "no literal pin: a front door's per-view family" in out and "Traceback" not in out, out
    rc, out = run(KEY, 5, "")
    assert rc == 1 and "not a measurement" in out and "no promoted LoRA architecture pick" not in out and "Traceback" not in out, out   # the pick (wo) resolves; refused at the unmeasured budget
    # the leaf itself, and padding_preflight's import, need nothing beyond the stdlib
    r = subprocess.run([PY, "-c", "import sys; sys.modules['torch'] = None; sys.modules['transformers'] = None\n"
                        "sys.path.insert(0, sys.argv[1]); import padding_preflight as pf\n"
                        "from sisteer.extraction import lora_module_sets as lms\n"
                        "assert pf.selected_module_set('qwen35_4b')[0] == 'wo' == lms.set_id_for(['o_proj']); print('stdlib-only OK')",
                        str(LANE)], capture_output=True, text=True, cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0 and "stdlib-only OK" in r.stdout, r.stdout + r.stderr



def test_the_daemons_padded_run_grammar_equals_the_canonical_phase_view_grammar():
    """The daemon's `_PADDED_RUN_RE` family alternation and the canonical PHASE_VIEW_RE (one of the seven byte-equal copies, read
    here from make_padding_configs) accept the SAME view spellings. Derived from the canonical regex, not a literal restatement:
    every candidate is judged by `gen.PHASE_VIEW_RE`, and the daemon must agree — a spelling the grammar accepts parses as that
    family under this key; one it rejects never parses as a family (the lazy key may absorb it, which makes it no view at all).
    The daemon's `obj(_sN)` alternative is the legacy per-object family outside the phase-view grammar and is not a candidate."""
    daemon = (REPO / "scripts" / "registry" / "grade_daemon.py").read_text()
    m = re.search(r'_PADDED_RUN_RE = re\.compile\(r"(.*)"\)', daemon); assert m, "the daemon's padded-run regex"
    R = re.compile(m.group(1)); canon = gen.PHASE_VIEW_RE
    stems = ("u0", "s1", "s2", "s3", "s12", "d0p75", "d1", "d0p5", "dm0p75", "dm1", "d0p75p8192", "u", "s", "d", "dm", "x1", "relu")
    tails = ("", "r", "p", "rp", "rl", "rlp", "l", "lp", "pr", "_relu", "_relul", "_relull", "_relup", "_rel", "x", "_")
    declared = {f for fams in gen.FRONT_DOOR_VIEW_FAMILIES.values() for f in fams if not f.startswith("obj")}
    # the table's per-view families: its GRAMMAR views, which both regexes must accept, and the row's DECLARED wave (battery — demo-import M3),
    # which NEITHER regex accepts by design: the daemon splits that run name through the models table (_padded_key_family), never by a literal
    waves = {gen._models_reader().declared_wave(k) for k in gen.FRONT_DOOR_VIEW_FAMILIES} - {None}
    grammar_declared, table_declared = declared - waves, declared & waves
    assert table_declared == {"battery"} and grammar_declared, sorted(declared)
    candidates = {st + t for st in stems for t in tails} | grammar_declared
    accepted = {v for v in candidates if canon.match(v)}
    assert {"s1", "s2", "s2r", "s2rl", "s2p", "s2rp", "d0p75", "d0p75_relu", "d0p75_relul", "dm0p75", "dm0p75_relul"} <= accepted
    assert {"s2rlp", "s2l", "d0p75_relull", "d0p75_relup", "u0r", "x1", "d0p75p8192"}.isdisjoint(accepted)
    assert grammar_declared <= accepted, sorted(grammar_declared - accepted)
    import grade_daemon as GD
    for w in table_declared:
        assert not canon.match(w) and R.match(f"pad_static_{KEY}_{w}_p4096").group("fam") is None and GD._padded_key_family(f"pad_static_{KEY}_{w}_p4096") == (KEY, w)
    for v in sorted(candidates):
        mm = R.match(f"pad_static_{KEY}_{v}_p4096")
        parsed = bool(mm) and mm.group("fam") == v and mm.group("key") == KEY
        assert parsed == (v in accepted), (v, v in accepted, mm and (mm.group("key"), mm.group("fam")))
    bad = R.match(f"pad_static_{KEY}_s2rlp_p4096")                        # the negative case: not a view, so never a family
    assert bad is None or bad.group("fam") is None
