"""The padded-static generator KNOWS the front door's keys (this fixes a case
where grade_padded_static's `run_identity` refused a front-door key because KEYS named the paper's model only):

* `FRONT_DOOR_KEYS` == every models-table key with a phase view (the reader's rule); a key that declares `u0` (the PHASED recipe) must list
  EXACTLY the three prompted-side arms there, and is a PROMPTED_ONLY key; a key with no u0 (the SINGLE-PHASE recipe — the tree's demo)
  is a SINGLE_PHASE key with no prompted-only family at all (source_view / campaign_config refuse by name; its padded
  families are its phase views, tests/test_padding_front_door_view_families.py); a u0 of another shape refuses; a hand-listed key with a
  phase view refuses; an absent table derives nothing;
* TODAY (the production roster carries no front-door view) `KEYS == HAND_LISTED_KEYS == ("qwen35_4b",)`, `--check` / `--list` are
  the 27 outputs byte for byte, and `main` names every pending view;
* the moment a production roster names `<key>/u0` of a PHASED key, the key joins KEYS and its PROMPTED-ONLY family renders from the u0
  wave pair (`sdfbase_full22_<key>_u0.yaml` / `petri_full22_<key>_u0.yaml`): five configs + one manifest, the header naming the u0 source,
  every output the u0 config as data except the named keys; the steered families stay the hand-listed keys' and refuse for a front-door key.
  No committed row declares u0, so that mechanism rides a SYNTHETIC phased row on a tmp copy of the table;
* the GRADER's targets (make_padding_configs.GRADER_TARGETS == grade_padded_static.TARGETS) are EXACTLY output_keys(): a PHASED key whose u0
  the production roster does not carry yet — out of KEYS, nothing rendered for it — is refused by both of the grader's gates, however truthy
  its families_of (the same synthetic row against the committed roster, and a stale config of the key)."""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
LANE = REPO / "scripts" / "sdfbase" / "full22" / "static_padding"
TABLE = REPO / "reproduce" / "models.yaml"
for p in (LANE, REPO / "scripts" / "sdfbase" / "full22", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import make_padding_configs as gen  # noqa: E402

ENV = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "HOME": str(REPO)}
KEY = "qwen35_4b_demo"                    # the tree's front-door model: SINGLE-PHASE (one wave battery + two dose rungs, no u0)
PHASED = "phased_demo"                    # a SYNTHETIC row that runs in PHASES (u0 = the three prompted-side arms + s1), on a tmp copy of the table


def _phased_table(path: Path) -> Path:
    """The models table plus PHASED: the demo's pins and objects under its own key and view_legacy_prefix, with a u0 of exactly CELLS and an s1."""
    import copy
    t = yaml.safe_load(TABLE.read_text(encoding="utf-8"))
    row = copy.deepcopy(t["models"][KEY]); row["view_legacy_prefix"] = PHASED
    row.pop("wave", None)                                                                      # a phased row declares no wave: its steered wave is a grammar name
    row["views"] = {"u0": {"cells": list(gen.CELLS)}, "s1": {"cells": ["control_lora", "constitution_lora", "difference_lora"]}}
    t["models"][PHASED] = row
    path.write_text(yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
    return path


def _fresh(table: Path | None = None, roster: Path | None = None):
    if table is not None:
        os.environ["SISTEER_MODELS_TABLE"] = str(table)
    else:
        os.environ.pop("SISTEER_MODELS_TABLE", None)
    try:
        spec = importlib.util.spec_from_file_location("_gen_fd_fresh", LANE / "make_padding_configs.py")
        mod = importlib.util.module_from_spec(spec)
        if roster is not None:
            # the import-time KEYS reads the production roster: point the fresh module at the synthetic one before exec
            spec.loader.exec_module(mod)
            mod.PRODUCTION_ROSTER = roster
            mod.KEYS = (*mod.HAND_LISTED_KEYS, *(k for k in mod.PROMPTED_ONLY_KEYS if mod._roster_names(k, mod.FRONT_DOOR_SOURCE_VIEW)))
            mod.GRADER_TARGETS = mod.output_keys()      # the grader's table IS output_keys(); re-derived over the synthetic roster as KEYS is
        else:
            spec.loader.exec_module(mod)
    finally:
        os.environ.pop("SISTEER_MODELS_TABLE", None)
    return mod


def test_front_door_keys_derive_from_the_models_table_and_the_hand_listed_tuple_is_untouched(tmp_path):
    table = yaml.safe_load(TABLE.read_text(encoding="utf-8"))["models"]
    expect = tuple(k for k, m in table.items() if any(gen.PHASE_VIEW_RE.match(str(v)) for v in (m.get("views") or {})))
    assert expect == (KEY,) and gen.FRONT_DOOR_KEYS == expect
    assert gen.SINGLE_PHASE_KEYS == (KEY,) and gen.PROMPTED_ONLY_KEYS == ()                  # the demo declares no u0: no prompted-only family on this tree
    assert gen.HAND_LISTED_KEYS == ("qwen35_4b",) and not (set(gen.FRONT_DOOR_KEYS) & set(gen.HAND_LISTED_KEYS))
    assert gen.source_view("qwen35_4b") == "main"
    assert gen.campaign_config("qwen35_4b").name == "sdfbase_full22_qwen35_4b.yaml" and gen.campaign_manifest("qwen35_4b").name == "petri_full22_qwen35_4b.yaml"
    for fn in (gen.source_view, gen.campaign_config, gen.campaign_manifest):
        with pytest.raises(ValueError, match="a single-phase front-door key has no prompted-only padding family"):
            fn(KEY)
    assert gen.campaign_config(KEY, "battery").name == f"sdfbase_full22_{KEY}_battery.yaml" and gen.campaign_manifest(KEY, "battery").name == f"petri_full22_{KEY}_battery.yaml"   # its per-view families' sources
    assert gen.front_door_keys(tmp_path / "absent.yaml") == () and gen.single_phase_keys(tmp_path / "absent.yaml") == ()
    # a SYNTHETIC phased row: a declared u0 of exactly the three prompted-side arms makes a PROMPTED-ONLY key (cut from its u0 pair)
    phased = _phased_table(tmp_path / "phased.yaml")
    assert gen.front_door_keys(phased) == (KEY, PHASED) and gen.single_phase_keys(phased) == (KEY,)
    assert gen.front_door_view_families(PHASED, phased) == ("s1",) and gen.front_door_view_families(KEY, phased) == ("battery", "d0p75", "d0p75_relu")
    bad = tmp_path / "shape.yaml"
    bad.write_text(yaml.safe_dump({"schema": 1, "models": {"m": {"view_legacy_prefix": "m", "views": {"u0": {"cells": ["none", "prompted__safety_refusal"]}}}}}))
    with pytest.raises(SystemExit, match="pads exactly those three arms"):
        gen.front_door_keys(bad)
    bad2 = tmp_path / "hand.yaml"
    bad2.write_text(yaml.safe_dump({"schema": 1, "models": {"qwen35_4b": {"view_legacy_prefix": "qwen35_4b", "views": {"u0": {"cells": list(gen.CELLS)}}}}}))
    with pytest.raises(SystemExit, match="one key, one source view"):
        gen.front_door_keys(bad2)


def test_keys_is_the_hand_listed_tuple_plus_exactly_the_front_door_keys_the_roster_carries():
    """ROSTER-CONDITIONAL (the fix-1 owner follow-up 10): before stage 05 lands the test model's block, KEYS is the hand-listed tuple and
    every front-door view is pending BY NAME (27 outputs); once the production roster carries a PHASED key's u0, the key joins KEYS with
    its prompted-only family (+ 6 outputs); per phase view the roster carries, a front-door key's per-view family (+ 6 each) — the
    single-phase demo's three views included, which never join KEYS (no prompted-only stores)."""
    roster = yaml.safe_load(gen.PRODUCTION_ROSTER.read_text(encoding="utf-8"))
    carried = tuple(k for k in gen.PROMPTED_ONLY_KEYS if gen._roster_names(k, gen.FRONT_DOOR_SOURCE_VIEW))
    assert gen.HAND_LISTED_KEYS == ("qwen35_4b",) and gen.KEYS == ("qwen35_4b", *carried) and set(gen.front_door_pending()) == set(gen.PROMPTED_ONLY_KEYS) - set(carried)
    assert KEY not in gen.KEYS                                                                 # a single-phase key: no prompted-only family, never a KEYS member
    n = 27 + sum(6 + 6 * (len(gen.families_of(k)) - 1) for k in carried) + sum(6 * len(gen.families_of(k)) for k in gen.SINGLE_PHASE_KEYS)
    r = subprocess.run([sys.executable, str(LANE / "make_padding_configs.py"), "--list"], capture_output=True, text=True, cwd=REPO, env=ENV)
    assert r.returncode == 0 and len(r.stdout.split()) == n, r.stdout
    r = subprocess.run([sys.executable, str(LANE / "make_padding_configs.py"), "--check"], capture_output=True, text=True, cwd=REPO, env=ENV)
    assert r.returncode == 0 and f"ok: {n} outputs byte-identical" in r.stdout, r.stdout + r.stderr
    for k in gen.PROMPTED_ONLY_KEYS:
        assert (f"skip {k}: the roster does not carry {k}/u0 yet" in r.stdout) == (k not in carried)
    assert f"skip {KEY}: the roster does not carry {KEY}/u0" not in r.stdout                 # the demo has no u0 to wait for
    for k, v in gen.front_door_view_pending():
        assert f"skip {k}/{v}: the roster does not carry this phase view yet" in r.stdout
    with pytest.raises(ValueError, match="a single-phase front-door key has no prompted-only padding family"):
        gen.render_config(KEY, 0)                                                             # the demo renders per view (test_padding_front_door_view_families), never a prompted-only config
    assert KEY not in roster["targets"] or set(gen.families_of(KEY)) <= set(roster["targets"][KEY]["views"])


def _u0_pair(tree: Path, key: str) -> None:
    """A phased key's u0 wave pair, cut from the real qwen35_4b pair: the three prompted-side arms as its explicit_cells / cells."""
    src_c = (REPO / "configs" / "sdfbase_full22_qwen35_4b.yaml").read_text(encoding="utf-8")
    src_c = re.sub(r"^explicit_cells:\n(?:- .*\n)+", "explicit_cells:\n" + "".join(f"- {c}\n" for c in gen.CELLS), src_c, flags=re.M)
    src_c = re.sub(r"^out_dir: .*$", f"out_dir: out/sdfbase_full22_{key}_u0", src_c, flags=re.M)
    src_c = re.sub(r"^private_run: .*$", f"private_run: sdfbase_full22_{key}_u0", src_c, flags=re.M)
    (tree / "configs").mkdir(parents=True, exist_ok=True)
    (tree / "configs" / f"sdfbase_full22_{key}_u0.yaml").write_text(src_c, encoding="utf-8")
    src_m = (REPO / "configs" / "capeval" / "manifests" / "petri_full22_qwen35_4b.yaml").read_text(encoding="utf-8")
    ids = gen.harness_cells(gen.CELLS)
    src_m = re.sub(r"^  cells:\n(?:    - .*\n)+", "  cells:\n" + "".join(f"    - {c}\n" for c in ids), src_m, flags=re.M)
    src_m = re.sub(r"^  audit: .*$", f"  audit: petri_full22_{key}_u0", src_m, flags=re.M)
    (tree / "configs" / "capeval" / "manifests").mkdir(parents=True, exist_ok=True)
    (tree / "configs" / "capeval" / "manifests" / f"petri_full22_{key}_u0.yaml").write_text(src_m, encoding="utf-8")


def test_a_phased_front_door_key_the_roster_names_joins_keys_and_renders_its_prompted_only_family_from_the_u0_pair(tmp_path):
    """The PHASED recipe's prompted-only family, on the SYNTHETIC phased row (no committed row declares u0)."""
    table = _phased_table(tmp_path / "models.yaml")
    roster = yaml.safe_load(gen.PRODUCTION_ROSTER.read_text(encoding="utf-8"))
    # the committed roster carries the demo's block (05 --step roster); this test's tmp tree holds the PHASED key's u0 pair alone, so the demo's
    # views are taken out of the synthetic roster (pending again, as before the block landed) and only the phased key renders
    assert KEY in roster["targets"], "the committed roster carries the worked example's block"
    del roster["targets"][KEY]
    roster["targets"][PHASED] = {"model": "Qwen/Qwen3.5-4B", "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "bank": "full22",
                                 "views": {"u0": {"config": f"configs/sdfbase_full22_{PHASED}_u0.yaml", "manifest": f"configs/capeval/manifests/petri_full22_{PHASED}_u0.yaml",
                                                  "cells": {"prompted__safety_refusal": None, "prompted__si_constitution": None}}}}
    rpath = tmp_path / "roster.yaml"; rpath.write_text(yaml.safe_dump(roster))
    mod = _fresh(table=table, roster=rpath)
    assert mod.FRONT_DOOR_KEYS == (KEY, PHASED) and mod.PROMPTED_ONLY_KEYS == (PHASED,) and mod.SINGLE_PHASE_KEYS == (KEY,)
    assert mod.KEYS == ("qwen35_4b", PHASED) and mod.front_door_pending() == () and mod.source_view(PHASED) == "u0"
    # every per-view family is pending here: the synthetic roster names the phased key's u0 alone and none of the demo's views
    assert mod.families_of(PHASED) == (None,) and set(mod.front_door_view_pending()) == {(PHASED, "s1"), *((KEY, v) for v in ("battery", "d0p75", "d0p75_relu"))}
    assert mod.families_of(KEY) == ()
    tree = tmp_path / "repo"; _u0_pair(tree, PHASED)
    mod.REPO = tree; mod.CONFIG_DIR = tree / "configs" / "sdfbase" / "static_padding"; mod.MANIFEST_DIR = tree / "configs" / "capeval" / "manifests"
    for pad in mod.PADS:
        text = mod.render_config(PHASED, pad)
        cfg = yaml.safe_load(text)
        assert cfg["explicit_cells"] == list(gen.CELLS) and cfg["private_run"] == f"pad_static_{PHASED}_p{pad}" and cfg["out_dir"] == f"out/pad_static_{PHASED}_p{pad}"
        assert cfg["attacks"]["static_max_new_tokens"] == gen.STATIC_CAP and (("static_padding" in cfg["attacks"]) == (pad > 0))
        if pad:
            assert cfg["attacks"]["static_padding"] == gen.padding_block(pad)
        assert f"# configs/sdfbase_full22_{PHASED}_u0.yaml (docs/static_padding.md)" in text          # the header names the u0 source
        assert mod.pad_of(mod.padded_config(PHASED, pad)) == pad and mod.key_of(mod.padded_config(PHASED, pad)) == PHASED and mod.family_of(mod.padded_config(PHASED, pad)) is None
    m_text = mod.render_manifest(PHASED)
    man = yaml.safe_load(m_text)
    assert man["petri"]["cells"] == gen.harness_cells(gen.CELLS) and f"OVERLAY of petri_full22_{PHASED}_u0.yaml" in m_text
    mod.KEYS = (PHASED,)                                                              # the front-door key's outputs alone (the tmp tree holds no qwen35_4b pair)
    names = [str(p.relative_to(tree)) for p, _t in mod.outputs()]
    assert names == [f"configs/sdfbase/static_padding/sdfbase_full22_{PHASED}.padstatic_p{p}.yaml" for p in gen.PADS] + [f"configs/capeval/manifests/petri_full22_{PHASED}.padstatic.yaml"]
    # the `obj` / `obj_s<N>` steered families are the hand-listed keys' — a front-door key refuses them by name (its steered views pad
    # through its PER-VIEW families; tests/test_padding_front_door_view_families.py)
    for fam in mod.FAMILIES[1:]:
        with pytest.raises(SystemExit, match="pads its steered views through its PER-VIEW families"):
            mod.render_config(PHASED, 0, fam)
        with pytest.raises(SystemExit, match="pads its steered views through its PER-VIEW families"):
            mod.render_manifest(PHASED, fam)


def test_a_phased_front_door_key_whose_u0_is_pending_is_outside_the_graders_targets_as_it_is_outside_the_generators_outputs(tmp_path):
    """GRADER_TARGETS once admitted every FRONT_DOOR_KEY with a truthy families_of — and a PHASED key's families_of is
    (None, …) whether or not the roster names its u0 — so a phased key added to the table BEFORE its u0 reaches the production roster
    (out of KEYS, out of output_keys(), nothing rendered) passed both of the grader's gates, and a stale config of it (rendered before a
    roster change dropped the key; --check rejects no extra file) could be graded against a view the roster no longer serves. Now
    GRADER_TARGETS IS output_keys(). The SYNTHETIC phased row on a tmp copy of the table against the COMMITTED roster (the demo's block,
    no PHASED block): the generator names the key pending and renders nothing for it (--list / --check), and the grader — a fresh process
    under SISTEER_MODELS_TABLE, the way the daemon reaches it — refuses it at argparse (rc 2, `invalid choice`) and at run_identity
    (Refuse, code 2) over exactly such a stale config, while the demo and the paper key parse as before; the moment a roster names the
    key's u0 it joins KEYS, output_keys() and the grader's table alike."""
    import json
    table = _phased_table(tmp_path / "models.yaml")
    assert not gen._roster_names(PHASED, gen.FRONT_DOOR_SOURCE_VIEW) and gen._roster_names(KEY, "battery")   # the committed roster: the demo's block, no PHASED block
    mod = _fresh(table=table)                                                                 # the committed roster stays the production roster
    assert mod.FRONT_DOOR_KEYS == (KEY, PHASED) and mod.PROMPTED_ONLY_KEYS == (PHASED,) and mod.SINGLE_PHASE_KEYS == (KEY,)
    assert mod.KEYS == ("qwen35_4b",) and mod.front_door_pending() == (PHASED,) and (PHASED, "s1") in mod.front_door_view_pending()
    assert mod.families_of(PHASED) == (None,) and bool(mod.families_of(PHASED))               # the trap: truthy while nothing of the key renders
    assert mod.output_keys() == ("qwen35_4b", KEY) == mod.GRADER_TARGETS and PHASED not in mod.GRADER_TARGETS
    env = {**ENV, "SISTEER_MODELS_TABLE": str(table)}
    r = subprocess.run([sys.executable, str(LANE / "make_padding_configs.py"), "--list"], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0 and PHASED not in r.stdout and len(r.stdout.split()) == 45, r.stdout            # the committed 45 and nothing of the pending key
    r = subprocess.run([sys.executable, str(LANE / "make_padding_configs.py"), "--check"], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0 and f"skip {PHASED}: the roster does not carry {PHASED}/u0 yet" in r.stdout and "ok: 45 outputs byte-identical" in r.stdout, r.stdout + r.stderr
    # a STALE config of the pending key: the shape 05 renders once a roster names the key's u0, left behind after a roster dropped it again
    stale = tmp_path / f"sdfbase_full22_{PHASED}.padstatic_p0.yaml"
    demo_cfg = REPO / "configs" / "sdfbase" / "static_padding" / f"sdfbase_full22_{KEY}.padstatic_battery_p0.yaml"
    assert demo_cfg.is_file(), demo_cfg
    stale.write_text(demo_cfg.read_text(encoding="utf-8"), encoding="utf-8")
    assert (mod.key_of(stale), mod.pad_of(stale), mod.family_of(stale)) == (PHASED, 0, None)  # by name it IS the key's prompted-only p0 config
    probe = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(LANE)!r})\n"
        "import yaml\n"
        "from pathlib import Path\n"
        "import make_padding_configs as gen, grade_padded_static as gps\n"
        "from sisteer.eval import static_grading as sg\n"
        "out = {'targets': list(gps.TARGETS), 'lane': list(gps.PaddedStaticLane.targets), 'keys': list(gps.KEYS),\n"
        "       'output_keys': list(gen.output_keys()), 'grader_targets': list(gen.GRADER_TARGETS), 'pending': list(gen.front_door_pending())}\n"
        "def parse(target, cfg):\n"
        "    try:\n"
        "        ns = sg.build_parser(gps.PaddedStaticLane()).parse_args(['--target', target, '--config', cfg, '--private-root', '/p', '--grades-root', '/g'])\n"
        "        return ['parsed', ns.target]\n"
        "    except SystemExit as e:\n"
        "        return ['exit', e.code]\n"
        "stale, demo = sys.argv[1], sys.argv[2]\n"
        f"out['parse_phased'] = parse({PHASED!r}, stale)\n"
        f"out['parse_demo'] = parse({KEY!r}, demo)\n"
        "out['parse_paper'] = parse('qwen35_4b', demo)\n"
        "try:\n"
        f"    gps.run_identity(yaml.safe_load(Path(stale).read_text()), {PHASED!r}, Path(stale)); out['identity'] = 'admitted'\n"
        "except gps.Refuse as e:\n"
        "    out['identity'] = ['refused', e.code, e.msg]\n"
        "print('PROBE ' + json.dumps(out))\n"
    )
    r = subprocess.run([sys.executable, "-c", probe, str(stale), str(demo_cfg)], capture_output=True, text=True, cwd=REPO,
                       env={**os.environ, "PYTHONPATH": str(REPO / "src"), "SISTEER_MODELS_TABLE": str(table)})
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("PROBE ")).removeprefix("PROBE "))
    assert out["pending"] == [PHASED] and out["keys"] == ["qwen35_4b"]
    assert out["output_keys"] == ["qwen35_4b", KEY] == out["grader_targets"] == out["targets"] == out["lane"]                  # one spelling, the generator's output set
    assert out["parse_phased"] == ["exit", 2] and f"invalid choice: '{PHASED}'" in r.stderr and f"'{KEY}'" in r.stderr, r.stderr[-800:]   # gate 1: argparse's own refusal
    assert out["parse_demo"] == ["parsed", KEY] and out["parse_paper"] == ["parsed", "qwen35_4b"]                                  # the demo and the paper key, as before
    assert out["identity"][:2] == ["refused", 2] and "is not a target of this grader" in out["identity"][2], out["identity"]        # gate 2: run_identity
    assert "GRADER_TARGETS" in out["identity"][2] and "output_keys" in out["identity"][2] and f"'{PHASED}'" in out["identity"][2], out["identity"]
    # the moment a roster names the key's u0, it joins KEYS, output_keys() and the grader's table alike (the phased mechanism above)
    roster = yaml.safe_load(gen.PRODUCTION_ROSTER.read_text(encoding="utf-8"))
    roster["targets"][PHASED] = {"model": "Qwen/Qwen3.5-4B", "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "bank": "full22",
                                 "views": {"u0": {"config": f"configs/sdfbase_full22_{PHASED}_u0.yaml", "manifest": f"configs/capeval/manifests/petri_full22_{PHASED}_u0.yaml",
                                                  "cells": {"prompted__safety_refusal": None, "prompted__si_constitution": None}}}}
    rpath = tmp_path / "roster.yaml"; rpath.write_text(yaml.safe_dump(roster))
    named = _fresh(table=table, roster=rpath)
    assert named.KEYS == ("qwen35_4b", PHASED) and named.front_door_pending() == () and named.output_keys() == named.GRADER_TARGETS == ("qwen35_4b", PHASED, KEY)


def test_a_stale_config_of_a_family_the_generator_does_not_render_for_an_admitted_key_is_refused_by_name(tmp_path, monkeypatch):
    """GRADER_TARGETS admits the KEY, and run_identity's remaining gates — pads_of / cells_of / private_run — know
    nothing of the key for the hand-listed families (cells_of(None, key) is CELLS for every key), so a stale `<demo>.padstatic_p0.yaml` (the
    prompted-only family) or `..._obj_p0.yaml` with that family's cells passed every gate for the single-phase demo key, whose generator
    renders its three per-view families ALONE (families_of): a previously rendered prompted-only config + store left behind after a model
    moved from the phased to the single-phase recipe would be dispatched by the daemon (its discovery is the private root's store names;
    --check rejects no extra file) and judgments filed for an instrument the table no longer serves. Now run_identity holds the config's
    (key, family) PAIR to make_padding_configs.families_of(key) right after the key's admission: the three stale shapes are refused by name
    (code 2, the family, the key and the key's rendered families in the message; never the target refusal), the committed configs of every
    rendered family — the demo's three per-view p0s, the paper key's prompted-only and obj p0s — are admitted as before, and a per-view
    config of a view the production roster has since DROPPED (rendered once, stale now) is refused by the same gate as a Refuse, never
    cells_of's SystemExit, while the key stays a target through its other views."""
    import importlib.util as _ilu
    for _p in (REPO / "src", REPO / "scripts", LANE):                         # jailbreak_run / sisteer: run_identity's decode resolution
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    monkeypatch.delenv("SISTEER_MODELS_TABLE", raising=False)
    spec = _ilu.spec_from_file_location("_gps_fd_pair", LANE / "grade_padded_static.py")
    gps = _ilu.module_from_spec(spec); spec.loader.exec_module(gps)
    from sisteer.eval import static_grading as sg
    cfg_dir = REPO / "configs" / "sdfbase" / "static_padding"
    demo_cfg = cfg_dir / gen.padded_config(KEY, 0, "battery").name
    assert demo_cfg.is_file(), demo_cfg
    assert gen.families_of(KEY) == ("battery", "d0p75", "d0p75_relu") and gen.families_of("qwen35_4b") == gen.FAMILIES == (None, "obj", "obj_s2", "obj_s3")
    assert KEY in gps.TARGETS and KEY in gps.PaddedStaticLane.targets and gps.TARGETS == gen.GRADER_TARGETS   # the KEY is admitted: round 1's gate is not what refuses below
    # the three stale shapes: by name AND content exactly what a rendered p0 of that family carries for a key that has the family
    for fam, cells in ((None, gen.CELLS), ("obj", gen.OBJ_CELLS), ("obj_s2", gen.OBJ_SEED_CELLS)):
        assert fam not in gen.families_of(KEY) and fam in gps.CONFIG_FAMILIES                # a family the grader serves (for the paper key), never this key's
        cfg = yaml.safe_load(demo_cfg.read_text(encoding="utf-8"))
        cfg["private_run"] = gen.private_run(KEY, 0, fam); cfg["explicit_cells"] = list(cells)
        stale = tmp_path / gen.padded_config(KEY, 0, fam).name
        stale.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        assert (gen.key_of(stale), gen.pad_of(stale), gen.family_of(stale)) == (KEY, 0, fam) and cfg["explicit_cells"] == list(gen.cells_of(fam, KEY))
        ns = sg.build_parser(gps.PaddedStaticLane()).parse_args(["--target", KEY, "--config", str(stale), "--private-root", "/p", "--grades-root", "/g"])
        assert ns.target == KEY                                                              # gate 1 (argparse choices=): the key IS a target
        with pytest.raises(gps.Refuse) as ei:                                                # the PAIR is refused, by name
            gps.run_identity(cfg, KEY, stale)
        msg = ei.value.msg
        assert ei.value.code == 2 and f"{stale.name}: the {fam or 'prompted-only'} family is not one the generator renders for '{KEY}'" in msg, msg
        assert "families_of" in msg and "['battery', 'd0p75', 'd0p75_relu']" in msg and "is not a target of this grader" not in msg, msg
    # the controls: every family the generator renders is admitted as before — the demo's per-view p0s, the paper key's prompted-only + obj
    for fam in gen.families_of(KEY):
        p = cfg_dir / gen.padded_config(KEY, 0, fam).name
        assert gps.run_identity(yaml.safe_load(p.read_text(encoding="utf-8")), KEY, p)["config_family"] == fam, p
    for fam in (None, "obj"):
        p = cfg_dir / gen.padded_config("qwen35_4b", 0, fam).name
        assert gps.run_identity(yaml.safe_load(p.read_text(encoding="utf-8")), "qwen35_4b", p)["config_family"] == fam, p
    # a per-view config of a view the production roster has since DROPPED: the committed d0p75 p0 against a roster without the view —
    # the grader holds the very module whose families_of / _roster_names read PRODUCTION_ROSTER at call time
    d0 = cfg_dir / gen.padded_config(KEY, 0, "d0p75").name
    roster = yaml.safe_load(gen.PRODUCTION_ROSTER.read_text(encoding="utf-8"))
    del roster["targets"][KEY]["views"]["d0p75"]
    rpath = tmp_path / "roster_without_d0p75.yaml"
    rpath.write_text(yaml.safe_dump(roster, sort_keys=False), encoding="utf-8")        # roster ORDER is the cell order cells_of reads: never re-sorted
    monkeypatch.setattr(gen, "PRODUCTION_ROSTER", rpath)
    assert gen.families_of(KEY) == ("battery", "d0p75_relu") and gen.output_keys() == ("qwen35_4b", KEY)   # the key still renders (two views): a target still
    with pytest.raises(gps.Refuse) as ei:
        gps.run_identity(yaml.safe_load(d0.read_text(encoding="utf-8")), KEY, d0)
    assert ei.value.code == 2 and f"{d0.name}: the d0p75 family is not one the generator renders for '{KEY}'" in ei.value.msg, ei.value.msg
    assert "['battery', 'd0p75_relu']" in ei.value.msg and "no longer carries" in ei.value.msg, ei.value.msg
    battery = cfg_dir / gen.padded_config(KEY, 0, "battery").name                                    # the view the roster still names: admitted
    assert gps.run_identity(yaml.safe_load(battery.read_text(encoding="utf-8")), KEY, battery)["config_family"] == "battery"
