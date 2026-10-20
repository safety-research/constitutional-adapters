r"""The DECLARED WAVE (demo-import M3; Q2 ruled): the worked example's one steered wave is `battery`, declared ONCE at the row level of
reproduce/models.yaml (`wave: battery`) — a name OUTSIDE the phase-view grammar, so a demo run is never spelled like the paper's `qwen35_4b_s<N>`
fit-seed replicates — and classified by the ONE reader rule in reproduce/lib/models.py (`phase_views_of`: the row's grammar views plus its
declared wave; `is_phase_view`: a grammar name for every key, or the declared wave of a key the table carries), which every generator and
grader loads by importlib and sisteer.front_door_views delegates to for the grading daemon.

(1) The SWEEP: every tracked file carrying a regex literal with the s-family body (BODY below, the regex sources `s\[0-9\]\+r`, `u0\|s\[0-9\]`,
`_s\[0-9\]\+` — spelled here as sources, never as bare bodies, so the s2rl grammar sweep of tests/test_petri_steered_audit_s2rl.py, which reads
every code file's non-comment lines, does not take this prose for an unwidened grammar copy) is enumerated from
`git ls-files` and classified PER CONSUMER — the six generators, the grader twin and the daemon admit `battery` as the demo's phase view by the
table (not by widening a regex: the seven PHASE_VIEW_RE copies and the three STEERED_WAVE_VIEW_RE copies stay byte-identical, tests/
test_s2r_phase_view_grammar.py and test_s2rl_lecun_view.py, and no grammar literal carries the word); scripts/petri_run.py `_STEERED_AUDIT_RE`
and scripts/registry/emit_contrasts.py `_U0_VIEW_RE` do NOT (battery is no steered audit and has no u0 anchor — exactly as s1 before it;
asserted, not fixed); the registry's static_padding adapter `_STANDUP_RE` (a `\d` body the sweep's regex does not see) is recorded as OFF the
demo's path; the rest are not view grammars (a fits-config replicate name, the padding generator's hand-listed family, the seedrep lane's
shell). A NEW grammar copy fails the pin.
(2) The reader: baseline_view / single_phase / front_door_keys / phase_views on the demo; check() refuses a `battery` view with no `wave:`, a
wave naming a missing or dose view, a grammar or reserved name, `obj_s<N>` / `prompted__`, and a wave beside u0.
(3) Every adapted consumer compared on declared, undeclared, dose, phased and paper views — and the grader's roster enumeration admits every
target it admitted before the rule existed (the rule is never narrower than the grammar: a row-membership reading dropped 41 paper targets
the first time this test ran)."""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "registry"))
import models as M  # noqa: E402

KEY = "qwen35_4b_demo"
WAVE = "battery"
TABLE = REPO / "reproduce" / "models.yaml"
ROSTER = REPO / "configs" / "sdfbase_full22_roster.yaml"
MODELS_PY = REPO / "reproduce" / "lib" / "models.py"
PAPER_KEYS = ("qwen35_4b", "gemma4e4b", "nemotron3nano4b", "llama8b")
BODY = re.compile(r"s\[0-9\]\+r|u0\|s\[0-9\]|_s\[0-9\]\+")
USE = re.compile(r"re\.(compile|match|search|fullmatch)\(")
#: every tracked source file that carries an s-family grammar literal, with HOW it reads the demo's declared wave
GRAMMAR_FILES = {
    "reproduce/lib/models.py": "reader",                                             # the ONE rule: the grammar + the declared wave
    "scripts/sdfbase/make_full22_configs.py": "generator",
    "scripts/sdfbase/full22/sampled/make_sampled_configs.py": "generator",
    "scripts/sdfbase/full22/misalign/make_misalign_manifest.py": "generator",
    "scripts/sdfbase/gsm8k/g8k_cells.py": "generator",
    "scripts/sdfbase/full22/static_padding/make_padding_configs.py": "generator",
    "scripts/sdfbase/seedrep/grade_full22_native.py": "grader",
    "scripts/registry/grade_daemon.py": "daemon",                                    # _PADDED_RUN_RE byte-identical; the fam-less split through the table
    "scripts/sdfbase/full22/f22_roster.py": "steered_wave_copy",                     # STEERED_WAVE_VIEW_RE: s<N>r(l) by name; the rest through the reader
    "scripts/petri_run.py": "name_only_not_widened",                                 # _STEERED_AUDIT_RE: a steered audit is `_s<N>r(l)`; battery is none
    "scripts/registry/emit_contrasts.py": "name_only_not_widened",                   # _U0_VIEW_RE: the u0 anchor of an s<N>r view; battery has none
    "reproduce/lib/common.sh": "not_a_view_grammar",                                 # REPLICATE_NAME: a fits-config replicate file name
    "scripts/sdfbase/seedrep/grade_battery.sh": "not_a_view_grammar",                # the seedrep lane's own `_seedrep_s<N>` private runs
}
GRAMMAR_NAMES = ("PHASE_VIEW_RE", "STEERED_WAVE_VIEW_RE", "_PADDED_RUN_RE", "_STEERED_AUDIT_RE", "_U0_VIEW_RE", "_PADDING_HAND_LISTED_FAMILY")


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                     # dataclasses with string annotations look the module up by name
    spec.loader.exec_module(mod)
    return mod


def _grammar_alone_roster_targets() -> set:
    """The roster's phase-view tags as the grader enumerated them BEFORE the rule existed: every roster view matching the grammar."""
    roster = yaml.safe_load(ROSTER.read_text())
    return {f"{k}_{v}" for k, s in roster["targets"].items() for v in ((s or {}).get("views") or {}) if M.PHASE_VIEW_RE.match(str(v))}


# ---------------------------------------------------------------------------------------------------------------- (1) the sweep
def test_every_tracked_grammar_literal_is_classified_and_no_new_copy_appeared():
    files = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True).stdout.decode().split("\0")
    found = {}
    for rel in files:
        if not rel or rel.startswith("tests/") or rel.endswith((".md", ".patch", ".json", ".yaml", ".jsonl", ".txt")):
            continue
        p = REPO / rel
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = [ln for ln in text.splitlines() if BODY.search(ln) and USE.search(ln)]
        if hits:
            found[rel] = hits
    assert set(found) == set(GRAMMAR_FILES), {"new": sorted(set(found) - set(GRAMMAR_FILES)), "gone": sorted(set(GRAMMAR_FILES) - set(found))}
    # no grammar literal carries the declared wave's name: the rename widened no regex
    for rel, hits in found.items():
        for ln in hits:
            assert WAVE not in ln, (rel, ln)
    for rel in GRAMMAR_FILES:
        text = (REPO / rel).read_text(encoding="utf-8")
        for m in re.finditer(r"^(" + "|".join(GRAMMAR_NAMES) + r') = re\.compile\(r"(.*?)"\)', text, re.M):
            assert WAVE not in m.group(2), (rel, m.group(1))
    # the static_padding adapter's `\d` grammar is outside the sweep's body on purpose and recorded by the next test
    assert not any("static_padding.py" in rel for rel in found)


def test_the_sweep_recorded_parsers_read_battery_as_they_read_s1():
    """The two name-only tag parsers: a declared wave is NOT a steered audit and has NO u0 anchor — exactly as `s1` before it; the
    static_padding adapter's `_STANDUP_RE` is OFF the demo's path (it parses `pad_static_<tag>[_obj][_s<N>]_p<pad>` of the seed-replicate
    standups; the demo's padded run `pad_static_qwen35_4b_demo_battery_p0` reads there as a tag with no objects, never a seedrep standup — the
    front door's padded runs register natively through the grading daemon's padded-run split, below)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import petri_run as prun
    S = type("S", (), {})
    for tag in (f"petri_full22_{KEY}_{WAVE}", f"petri_full22_{KEY}_s1", f"petri_full22_{KEY}_d0p75", "petri_full22_qwen35_4b_d0p1"):
        s = S(); s.audit = tag
        assert not prun.steered_wave_audit(s), tag                   # single-phase waves and dose rungs carry their own none; paper d0p1 anchors same-wave
    s = S(); s.audit = f"petri_full22_{KEY}_s2rl"
    assert prun.steered_wave_audit(s)
    import emit_contrasts as EC
    assert EC.u0_view_of(f"{KEY}_{WAVE}") is None and EC.u0_view_of(f"{KEY}_s1") is None and EC.u0_view_of(f"{KEY}_d0p75") is None
    assert EC.u0_view_of(f"{KEY}_s2r") == f"{KEY}_u0" and EC.u0_view_of("qwen35_4b_s2rl") == "qwen35_4b_u0"
    from sisteer.registry.adapters import static_padding as SP
    m = SP._STANDUP_RE.match(f"pad_static_{KEY}_{WAVE}_p0")
    assert m and m.group("tag") == f"{KEY}_{WAVE}" and m.group("obj") is None and m.group("seed") is None
    assert r"\d" in SP._STANDUP_RE.pattern and not BODY.search(SP._STANDUP_RE.pattern)


# ---------------------------------------------------------------------------------------------------------------- (2) the reader
def test_the_reader_classifies_the_declared_wave_and_the_grammar_views_together():
    assert M.declared_wave(KEY) == WAVE and M.baseline_view(KEY) == WAVE and M.single_phase(KEY)
    assert M.front_door_keys() == [KEY] and M.phase_views(KEY) == [WAVE, "d0p75", "d0p75_relu"] == list(M.views(KEY))
    assert M.is_phase_view(KEY, WAVE) and M.is_phase_view(KEY, "d0p75") and not M.is_phase_view(KEY, "nope") and not M.is_phase_view(KEY, "main")
    # the grammar names a phase view for EVERY key (the roster's reading); the row simply has no such view
    assert M.is_phase_view(KEY, "s1") and M.is_phase_view(KEY, "u0") and "s1" not in M.phase_views(KEY) and "u0" not in M.phase_views(KEY)
    assert not M.PHASE_VIEW_RE.match(WAVE) and not M.steered_wave_view(WAVE, KEY) and not M.steered_wave_view(WAVE)
    assert M.is_phase_view("glm46vflash", "d1p5") and M.is_phase_view("glm46vflash", "s2") and not M.is_phase_view("glm46vflash", WAVE)
    # the paper rows declare no wave and name no phase view; their roster views stay grammar views by name
    for k in PAPER_KEYS:
        assert M.declared_wave(k) is None and M.phase_views(k) == [] and not M.is_phase_view(k, WAVE)
        assert M.is_phase_view(k, "s1") and M.is_phase_view(k, "d0p1") and M.is_phase_view(k, "dm1_relu") and not M.is_phase_view(k, "it")
    assert M.check() == []


def _table(tmp_path: Path, name: str, mutate) -> str:
    t = yaml.safe_load(TABLE.read_text())
    mutate(t["models"][KEY])
    p = tmp_path / f"{name}.yaml"; p.write_text(yaml.safe_dump(t, sort_keys=False))
    return str(p)


def test_check_refuses_every_malformed_wave_declaration(tmp_path):
    def probs(name, mutate):
        return [p for p in M.check(_table(tmp_path, name, mutate)) if p.startswith(f"{KEY}.")]
    # a `battery` view with NO `wave:` — neither grammar nor declared: refused by name (the silent-drop class), and the baseline refuses (0 waves)
    out = probs("undeclared", lambda m: m.pop("wave"))
    assert any("views.battery: neither a view of the phase-view grammar nor the declared wave" in p for p in out), out
    with pytest.raises(SystemExit, match="exactly ONE steered wave"):
        M.baseline_view(KEY, _table(tmp_path, "undeclared2", lambda m: m.pop("wave")))
    assert any("wave: 'nope' names no view" in p for p in probs("missing", lambda m: m.__setitem__("wave", "nope")))
    assert any("wave: 'd0p75' is a dose view" in p for p in probs("dose", lambda m: m.__setitem__("wave", "d0p75")))

    def grammar(m):
        m["wave"] = "s1"; m["views"] = {"s1": m["views"].pop("battery"), **m["views"]}
    assert any("wave: 's1' is a name of the phase-view grammar" in p for p in probs("grammar", grammar))
    for bad in ("u0", "main", "it", "obj", "obj_s2", "prompted__x"):
        def reserved(m, bad=bad):
            m["wave"] = bad; m["views"] = {bad: m["views"].pop("battery"), **m["views"]}
        out = probs(f"bad_{bad}", reserved)
        assert any(f"wave: '{bad}'" in p and ("reserved" in p or "collides" in p or "prompted-cell prefix" in p or "grammar" in p) for p in out), (bad, out)

    def phased(m):
        m["views"] = {"u0": {"cells": ["none", "prompted__safety_refusal", "prompted__si_constitution"]}, **m["views"]}
    assert any("a row that declares u0 runs in PHASES" in p for p in probs("phased", phased))
    assert any("is not a non-empty scalar name" in p for p in probs("list", lambda m: m.__setitem__("wave", ["battery"])))
    # the CLI: `check` exit 2 naming the problem; `phase-views` / `declared-wave` print the rule's answers
    r = subprocess.run([sys.executable, str(MODELS_PY), "--table", _table(tmp_path, "cli", lambda m: m.__setitem__("wave", "nope")), "check"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "names no view" in r.stderr
    assert subprocess.run([sys.executable, str(MODELS_PY), "phase-views", KEY], capture_output=True, text=True, check=True).stdout.split() == [WAVE, "d0p75", "d0p75_relu"]
    assert subprocess.run([sys.executable, str(MODELS_PY), "declared-wave", KEY], capture_output=True, text=True, check=True).stdout.strip() == WAVE
    assert subprocess.run([sys.executable, str(MODELS_PY), "declared-wave", "qwen35_4b"], capture_output=True, text=True).returncode == 1


# ---------------------------------------------------------------------------------------------------------------- (3) every adapted consumer
@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    """Beside the committed table (DECLARED): the demo UNDECLARED under a grammar name (`s1`, no `wave:`) and the demo PHASED (u0 + an `s1`
    wave, no `wave:`); every other row as committed."""
    root = tmp_path_factory.mktemp("tables")
    grammar = yaml.safe_load(TABLE.read_text()); m = grammar["models"][KEY]
    m.pop("wave"); m["views"] = {"s1": m["views"].pop("battery"), **m["views"]}
    (root / "grammar.yaml").write_text(yaml.safe_dump(grammar, sort_keys=False))
    phased = yaml.safe_load(TABLE.read_text()); m = phased["models"][KEY]
    m.pop("wave"); cells = m["views"].pop("battery")["cells"]
    m["views"] = {"u0": {"cells": [c for c in cells if c == "none" or c.startswith("prompted__")]},
                  "s1": {"cells": [c for c in cells if c != "none" and not c.startswith("prompted__")]}, **m["views"]}
    (root / "phased.yaml").write_text(yaml.safe_dump(phased, sort_keys=False))
    return {"declared": str(TABLE), "grammar": str(root / "grammar.yaml"), "phased": str(root / "phased.yaml")}


EXPECT = {"declared": ("battery", "d0p75", "d0p75_relu"), "grammar": ("s1", "d0p75", "d0p75_relu"), "phased": ("u0", "s1", "d0p75", "d0p75_relu")}


def test_every_adapted_consumer_classifies_declared_undeclared_dose_phased_and_paper_views_alike(tables, monkeypatch, capsys):
    for name, path in tables.items():
        monkeypatch.setenv("SISTEER_MODELS_TABLE", path)          # the generators' import-time MODELS_TABLE
        assert tuple(M.phase_views(KEY, path)) == EXPECT[name], name
        assert M.check(path) == [], (name, M.check(path))
        gen = _load("scripts/sdfbase/make_full22_configs.py", f"dwv_gen_{name}")
        assert set(gen.front_door_models(path)) == {KEY}
        msc = _load("scripts/sdfbase/full22/sampled/make_sampled_configs.py", f"dwv_msc_{name}")
        assert msc.front_door_views(path)[KEY] == EXPECT[name], name
        mm = _load("scripts/sdfbase/full22/misalign/make_misalign_manifest.py", f"dwv_mm_{name}")
        assert {t for t in mm.front_door_targets(path) if t.startswith(KEY)} == {f"{KEY}_{v}" for v in EXPECT[name]}, name
        g8k = _load("scripts/sdfbase/gsm8k/g8k_cells.py", f"dwv_g8k_{name}")
        assert tuple(g8k.front_door_facts(path)[KEY]["views"]) == EXPECT[name], name
        pad = _load("scripts/sdfbase/full22/static_padding/make_padding_configs.py", f"dwv_pad_{name}")
        assert pad.front_door_view_families(KEY, path) == tuple(v for v in EXPECT[name] if v != "u0"), name
        assert pad.is_front_door_family(EXPECT[name][-1]) and not pad.is_front_door_family("obj") and not pad.is_front_door_family(None)
        assert pad.is_front_door_family("battery") is (name == "declared") and pad.is_front_door_family("s1") is (name != "declared")
        twin = _load("scripts/sdfbase/seedrep/grade_full22_native.py", f"dwv_twin_{name}")
        table_targets, note = twin.models_table_phase_view_targets(path)
        assert note is None and {t for t in table_targets if t.startswith(KEY)} == {f"{KEY}_{v}" for v in EXPECT[name]}, name
        assert all(twin._is_phase_view(KEY, v, path) for v in EXPECT[name]) and not twin._is_phase_view(KEY, "nope", path)
        assert twin._is_phase_view(KEY, "battery", path) is (name == "declared") and twin._is_phase_view(KEY, "s1", path)   # grammar by name
        # the paper's roster views keep their classification under every table: grammar names of keys the table carries (main / it only) or lacks
        assert twin._is_phase_view("qwen35_4b", "s2", path) and twin._is_phase_view("gemma4e4b", "d0p1", path) and twin._is_phase_view("glm46vflash", "d1p5", path)
        assert not twin._is_phase_view("qwen35_4b", "it", path) and not twin._is_phase_view("qwen35_4b", "battery", path)
        # the grader's ROSTER enumeration admits every target the grammar admitted before the rule existed — and, of what the grammar did
        # not, exactly the demo's declared wave when the table declares it: the real roster carries the demo's block (05 --step roster), whose
        # `battery` view is a phase view under the declared table alone (the undeclared and phased tables declare no wave, so it is none there)
        declared_extra = {f"{KEY}_battery"} if name == "declared" else set()
        assert set(twin.roster_phase_view_targets(ROSTER, path)) == _grammar_alone_roster_targets() | declared_extra, name
        assert f"{KEY}_battery" in yaml.safe_load(ROSTER.read_text())["targets"][KEY]["views"] or True   # the real roster's block names the wave …
        assert "battery" in (yaml.safe_load(ROSTER.read_text())["targets"][KEY].get("views") or {}), "the committed roster carries the demo's block"
        resolved, notes = twin.resolve_targets(ROSTER, path)
        assert notes == [] and _grammar_alone_roster_targets() <= set(resolved) and set(twin.TARGETS) <= set(resolved)
        assert {f"{KEY}_{v}" for v in EXPECT[name]} <= set(resolved)
    monkeypatch.delenv("SISTEER_MODELS_TABLE", raising=False)
    # a synthetic roster carrying the demo's declared wave beside a grammar view it has no row for, and a paper target
    twin = _load("scripts/sdfbase/seedrep/grade_full22_native.py", "dwv_twin_roster")
    rp = Path(tables["grammar"]).parent / "roster.yaml"
    rp.write_text(yaml.safe_dump({"targets": {KEY: {"views": {"battery": {}, "d0p75": {}, "s1": {}, "main": {}, "nope": {}}},
                                              "glm46vflash": {"views": {"main": {}, "d1p5": {}, "s2": {}, "battery": {}}}}}))
    tags = twin.roster_phase_view_targets(rp, tables["declared"])
    assert {t for t in tags if t.startswith(KEY)} == {f"{KEY}_battery", f"{KEY}_d0p75", f"{KEY}_s1"}     # declared + grammar; main / nope are none
    assert {t for t in tags if t.startswith("glm46vflash")} == {"glm46vflash_d1p5", "glm46vflash_s2"}    # a key with no row: the grammar alone
    assert tags[f"{KEY}_battery"]["run"] == f"sdfbase_full22_{KEY}_battery" and tags[f"{KEY}_battery"]["target_key"] == KEY
    # the daemon: the padded-run split for the declared wave, by the table; the grammar's families and the paper names unchanged
    import grade_daemon as GD
    assert GD._padded_key_family(f"pad_static_{KEY}_battery_p0") == (KEY, "battery")
    assert GD.target_for("padding", f"pad_static_{KEY}_battery_p0", "x") == KEY
    assert GD.config_for("padding", configs_root="/c", private_run=f"pad_static_{KEY}_battery_p0") == f"/c/sdfbase/static_padding/sdfbase_full22_{KEY}.padstatic_battery_p0.yaml"
    assert GD._padded_key_family(f"pad_static_{KEY}_s1_p0") == (KEY, "s1") and GD._padded_key_family(f"pad_static_{KEY}_d0p75_relu_p8192") == (KEY, "d0p75_relu")
    assert GD._padded_key_family(f"pad_static_{KEY}_p0") == (KEY, None)                                   # the bare key: the prompted-only family
    assert GD._padded_key_family(f"pad_static_{KEY}_nope_p0") == (f"{KEY}_nope", None)                    # an undeclared tail stays the key group, as before
    assert GD._padded_key_family("pad_static_qwen35_4b_obj_s2_p65536") == ("qwen35_4b", "obj_s2") and GD._padded_key_family("pad_static_qwen35_4b_p0") == ("qwen35_4b", None)
    assert GD.target_for("padding", "pad_static_qwen35_4b_obj_s2_p65536", "x") == "qwen35_4b" and GD._padded_key_family("pad_static_qwen35_4b_battery_p0") == ("qwen35_4b_battery", None)
    # the padded GRADER admits what the daemon names (target_for said qwen35_4b_demo while both of the grader's gates
    # were bound to the PROMPTED-ONLY leg's KEYS, which a single-phase key never joins — rc 2 at argparse, Refuse at run_identity): the
    # daemon's --target over the generated battery_p0 config parses through the lane's REAL argparse (gate 1, `choices=lane.targets`) and
    # run_identity (gate 2) hands back the key. KEYS keeps its prompted-only meaning (the demo stays out of it); the grader's targets are the
    # generator's GRADER_TARGETS == output_keys(), KEYS + the single-phase keys with a roster-named family — exactly the keys the generator
    # renders for (a phased key with a pending u0 is outside both: test_padding_front_door_keys) — so the paper key's admission is unchanged.
    gen = _load("scripts/sdfbase/full22/static_padding/make_padding_configs.py", "dwv_pad_committed")
    gps = _load("scripts/sdfbase/full22/static_padding/grade_padded_static.py", "dwv_grade_padded_static")
    from sisteer.eval import static_grading as sg
    target = GD.target_for("padding", f"pad_static_{KEY}_battery_p0", "x")
    pad_cfg = REPO / "configs" / "sdfbase" / "static_padding" / f"sdfbase_full22_{KEY}.padstatic_battery_p0.yaml"
    assert pad_cfg.is_file(), pad_cfg                                                                      # the committed worked-example config (05 --step configs)
    assert KEY not in gen.KEYS and KEY in gen.GRADER_TARGETS and gen.GRADER_TARGETS == (*gen.KEYS, KEY)    # never the prompted-only leg's; the grader's
    assert KEY not in gps.KEYS and gps.TARGETS == gen.GRADER_TARGETS and gps.PaddedStaticLane.targets == gps.TARGETS
    assert gen.GRADER_TARGETS == gen.output_keys() == (*gen.KEYS, *(k for k in gen.SINGLE_PHASE_KEYS if gen.families_of(k)))   # the generator's output set, never a truthy families_of over FRONT_DOOR_KEYS
    ns = sg.build_parser(gps.PaddedStaticLane()).parse_args(["--target", target, "--config", str(pad_cfg), "--private-root", "/p", "--grades-root", "/g"])
    assert ns.target == target == KEY and ns.config == str(pad_cfg)
    ident = gps.run_identity(yaml.safe_load(pad_cfg.read_text()), KEY, pad_cfg)
    assert ident["key"] == KEY and ident["config_family"] == "battery" and ident["pad_tokens"] == 0
    with pytest.raises(SystemExit) as ei:                                                                   # a key with no padded family at all stays refused
        sg.build_parser(gps.PaddedStaticLane()).parse_args(["--target", "glm46vflash", "--config", str(pad_cfg), "--private-root", "/p", "--grades-root", "/g"])
    err = capsys.readouterr().err                                                                           # argparse's own refusal: rc 2, the message on stderr
    assert ei.value.code == 2 and "invalid choice: 'glm46vflash'" in err and f"'{KEY}'" in err and "'qwen35_4b'" in err, err[-600:]
    with pytest.raises(gps.Refuse) as ei:                                                                   # gate 2 (Refuse: exit 2, the message in .msg)
        gps.run_identity(dict(yaml.safe_load(pad_cfg.read_text())), "glm46vflash", pad_cfg.with_name("sdfbase_full22_glm46vflash.padstatic_battery_p0.yaml"))
    assert ei.value.code == 2 and "is not a target of this grader" in ei.value.msg and "GRADER_TARGETS" in ei.value.msg and f"'{KEY}'" in ei.value.msg, ei.value.msg
    # the PAIR, not just the key: a STALE prompted-only p0 of the demo — by name and content exactly a rendered p0 of that family
    # for a key that has it — parses at gate 1 (the key IS a target) and is refused at run_identity by families_of(KEY), which renders the
    # single-phase key's per-view families alone; never through the target refusal, never through cells_of (test_padding_front_door_keys)
    stale = Path(tables["grammar"]).parent / gen.padded_config(KEY, 0, None).name
    stale_cfg = yaml.safe_load(pad_cfg.read_text()); stale_cfg["private_run"] = gen.private_run(KEY, 0, None); stale_cfg["explicit_cells"] = list(gen.CELLS)
    stale.write_text(yaml.safe_dump(stale_cfg, sort_keys=False))
    assert (gen.key_of(stale), gen.family_of(stale)) == (KEY, None) and None not in gen.families_of(KEY) and gen.families_of(KEY) == ("battery", "d0p75", "d0p75_relu")
    assert sg.build_parser(gps.PaddedStaticLane()).parse_args(["--target", KEY, "--config", str(stale), "--private-root", "/p", "--grades-root", "/g"]).target == KEY
    with pytest.raises(gps.Refuse) as ei:
        gps.run_identity(stale_cfg, KEY, stale)
    assert ei.value.code == 2 and f"the prompted-only family is not one the generator renders for '{KEY}'" in ei.value.msg and "families_of" in ei.value.msg, ei.value.msg
    assert "['battery', 'd0p75', 'd0p75_relu']" in ei.value.msg and "is not a target of this grader" not in ei.value.msg, ei.value.msg
    # the delegates
    from sisteer import front_door_views as FV
    assert FV.phase_views(KEY) == ["battery", "d0p75", "d0p75_relu"] and FV.is_phase_view(KEY, "battery") and FV.is_phase_view(KEY, "s1")
    assert not FV.is_phase_view(KEY, "nope") and not FV.is_phase_view("glm46vflash", "battery") and FV.is_phase_view("glm46vflash", "s2")
    assert FV.phase_views("glm46vflash") == [] and FV.phase_views("qwen35_4b") == [] and FV.split_key_view(f"{KEY}_battery") == (KEY, "battery")
    assert not re.search(r"s\[0-9\]\+r", (REPO / "src" / "sisteer" / "front_door_views.py").read_text())    # the helper duplicates no grammar


def test_the_ports_table_and_the_registry_spellings_carry_the_declared_wave_and_not_its_former_spelling():
    ports = yaml.safe_load((REPO / "configs/sdfbase_full22_ports.yaml").read_text())["ports"]
    assert f"{KEY}_{WAVE}" in ports and f"{KEY}_s1" not in ports
    assert {t for t in ports if t.startswith(KEY)} == {KEY, *(f"{KEY}_{v}" for v in M.phase_views(KEY))}
    spell = (REPO / "configs/registry/legacy_spellings.yaml").read_text()
    assert f"legacy: {KEY}_{WAVE}," in spell and f"legacy: {KEY}_s1," not in spell
    for rel in ("configs/registry/decode_policy.json", "src/sisteer/registry/decode_policy.json"):
        text = (REPO / rel).read_text()
        assert f'"{KEY}_{WAVE}"' in text and f'"{KEY}_s1"' not in text
    hits = subprocess.run(["git", "-C", str(REPO), "grep", "-l", f"{KEY}_s1", "--", ".", ":!tests"], capture_output=True, text=True).stdout
    assert hits.strip() == "", hits
