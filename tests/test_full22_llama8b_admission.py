"""The sdfbase-llama8b wave's CPU generation-and-admission test (the wave tree, 2026-09-10).

Three views over ONE served target — `main` (base-lane objects transferred), `it` (instruct-direct
objects) and `fullconst` (the FULL constitution as a prompted baseline: static families +
best-of-N + capability only) — must generate from the roster, load as Petri specs, pass the standup's
manifest <-> config gate, enumerate against a bank (the fullconst view: `none` + the prompted cell,
its prefix hash the registry's bytes), name their SI-variant behavior dirs, and plan exactly the legs
their roster entry says. Every serving geometry derives from budget_for("llama8b") — the PROVISIONAL
229,376 ceiling today — so a measured row landing changes every S in one place, and this test pins the
DERIVATION rather than the literals.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22",
          REPO / "scripts" / "sdfbase" / "full22" / "sampled"):
    sys.path.insert(0, str(p))

import f22_roster as fr
import make_full22_configs as gen
import make_sampled_configs as msc
import submit_newmodels as sn
import submit_sampled as ss

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.petri import scenarios as psc
from sisteer.capeval.serving import geometry as sg
from sisteer.data.instructions import BY_ID

KEY = "llama8b"
VIEWS = ("main", "it", "fullconst", "shortconst")
#: the SHORT constitution-SI prompted view (the paper's constitution-SI arm): every leg, PER-LEG windows
SHORTCONST_WINDOWS = {"static": 10496, "controls": 6656, "rungs": {2048: 8064, 8192: 14592, 16384: 22784}}
SHORTCONST_MT_PAIR = 18688
SHORTCONST_SI_SHA256 = "21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f"
ROSTER = gen.load_roster()
PINS = gen.load_pins()
LEGS_ALL = fr.LEGS
FULLCONST_LEGS = ("static", "bon", "capability")
FULLCONST_OMITTED = ("controls", "attacks", "petri")
SHA_PLACEHOLDER = "a" * 40
#: the lane's eval-time TOKEN CAPS (roster targets.llama8b.caps) — read, then every consumer
#: is asserted against them; the literals below are what the roster carries today and fail loudly if either side moves.
#: The multi-turn cap is PER FAMILY; every family runs at 4,096 per reply: Crescendo, PAIR and ActorAttack (kept);
#: BoN is the NATIVE 4,096 pass (bon.max_new_tokens WRITTEN)
CAPS = fr.target_caps(ROSTER, KEY)
MT_CAPS = {"crescendo": 4096, "pair": 4096, "actorattack": 4096}
BON_CAP = 4096
#: the lane's PINNED multi-turn windows PER FAMILY, widening the derived ones one step for Crescendo / PAIR and one
#: for ActorAttack (roster targets.llama8b.caps.windows) — first pass and over-window re-run; every consumer is
#: asserted to honour them over the derivation
MT_WINDOWS = fr.target_windows(ROSTER, KEY)
#: (first pass L, S, re-run L, S) per family at the measured 475,136 budget — the served numbers every consumer must agree on
#: Crescendo 49,152 x 9 (the canonical measured window) / 65,536 x 7; PAIR 16,384 x 29 (canonical) / 32,768 x 14;
#: ActorAttack 49,152 x 9 / 65,536 x 7; the re-run pins are our typed over-window safety net
MT_GEOM = {"crescendo": (49152, 9, 65536, 7), "pair": (16384, 29, 32768, 14), "actorattack": (49152, 9, 65536, 7)}
#: the derived minimums per family (p99 prompt 7,496 + cap; measured max call 12,657 + cap; on the 128 grid) the pins must clear: the same
#: for every family now that all three run at 4,096
MT_DERIVED = {"crescendo": (11648, 16768), "pair": (11648, 16768), "actorattack": (11648, 16768)}
#: the REAL auditor / judge prompt bytes (configs/capeval/petri/prompts/<role>.md, what petri_run.load_prompt_files
#: reads): the manifests declare prompt_provenance override@<hash>, and the harness hash refuses any other bytes (H10)
PROMPT_FILES = {role: (REPO / "configs" / "capeval" / "petri" / "prompts" / f"{role}.md").read_bytes()
                for role in ("auditor", "judge")}


def _render_all(tmp_path: Path, roster=ROSTER, pins=PINS):
    """Generate the three views' config + manifest into a tmp root; return {view: (cfg dict, spec)}."""
    out = {}
    for view in VIEWS:
        v = roster["targets"][KEY]["views"][view]
        cfg_text = gen.render_config(roster, pins, KEY, view)
        man_text = gen.render_manifest(roster, KEY, view)
        cpath = tmp_path / Path(v["config"]).name
        mpath = tmp_path / Path(v["manifest"]).name
        cpath.write_text(cfg_text); mpath.write_text(man_text)
        out[view] = (yaml.safe_load(cfg_text), cm.load_petri_spec(mpath), cfg_text, man_text)
    return out


def test_the_committed_configs_and_manifests_are_the_generators_output():
    for view in VIEWS:
        v = ROSTER["targets"][KEY]["views"][view]
        assert (REPO / v["config"]).read_text() == gen.render_config(ROSTER, PINS, KEY, view), view
        assert (REPO / v["manifest"]).read_text() == gen.render_manifest(ROSTER, KEY, view), view


def test_three_views_generate_load_and_pass_the_standups_gate(tmp_path):
    import mt_serving_up as m
    rendered = _render_all(tmp_path)
    for view, (cfg, spec, _, _) in rendered.items():
        assert (spec.target_model, spec.target_revision) == (cfg["model"], cfg["model_revision"]) == (
            "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77"), view
        tt_of = m.assert_manifest_matches_config(spec, cfg)          # the standup's gate, factored
        assert list(tt_of.values()) == cfg["explicit_cells"], view
        assert spec.chat_template_hash == "b48c47f644389271" and spec.tool_parser == "llama3_json", view
    main_cfg, it_cfg, fc_cfg, sc_cfg = (rendered[v][0] for v in VIEWS)
    # -- staged-state pin (written by the lane's staging launcher): explicit_cells BEGIN --
    # the staged objects (objects first, the prompted baseline LAST — the campaign's order); a partial move leaves the rest pending
    # prompted__si_constitution is DECLARED on main / it and dropped from them at staging
    # (stage_tools DROP_SETS): this lane serves the arm through the shortconst view only; the on-branch configs list the declaration
    assert main_cfg["explicit_cells"] == ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal", "prompted__si_constitution"]
    assert it_cfg["explicit_cells"] == ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal", "prompted__si_constitution"]
    # -- staged-state pin: explicit_cells END --
    assert fc_cfg["explicit_cells"] == ["none", "prompted__si_constitution_full"]
    assert rendered["fullconst"][1].cells == ["none", "prompted__constitution_full"]
    # the SHORT constitution-SI arm: its own prompted-only view, the sampled lane's cell / harness ids, distinct from fullconst
    assert sc_cfg["explicit_cells"] == ["none", "prompted__si_constitution"] and rendered["shortconst"][1].cells == ["none", "prompted__constitution"]
    assert sc_cfg["private_run"] == "sdfbase_full22_llama8b_shortconst" and sc_cfg["defenses_dir"].endswith("/llama8b/cells_view/shortconst")
    assert sc_cfg["out_dir"] != fc_cfg["out_dir"] and sc_cfg["private_run"] != fc_cfg["private_run"]
    # the two lane views are separate configs with distinct stores (cell ids collide by construction)
    assert main_cfg["out_dir"] != it_cfg["out_dir"] != fc_cfg["out_dir"]
    assert main_cfg["defenses_dir"].endswith("/llama8b/cells_view/main") and it_cfg["defenses_dir"].endswith("/cells_view/it")
    # a mismatched pair REFUSES (the gate is the standup's)
    bad = dict(main_cfg); bad["explicit_cells"] = ["none", "prompted__safety_refusal", "constv2-mix-full__psr_ce_all"]
    with pytest.raises(SystemExit, match="manifest/config cell mismatch"):
        m.assert_manifest_matches_config(rendered["main"][1], bad)


def test_the_fullconst_manifest_carries_the_registry_hash_and_the_lane_manifests_do_not(tmp_path):
    rendered = _render_all(tmp_path)
    fc = rendered["fullconst"][1]
    text = BY_ID["si_constitution_full"].text
    assert fc.prompted_prefix_hashes["constitution_full"] == cm.sha256_hex16(text.encode("utf-8")) == "82f88c5532e4939e"
    for view in ("main", "it", "shortconst"):
        assert "constitution_full" not in rendered[view][1].prompted_prefix_hashes, view   # H10: never re-key another view
        assert set(rendered[view][1].prompted_prefix_hashes) == set(gen.PREFIX_HASHES_ALWAYS)
    # the SHORT constitution SI's hash is one of the historical three and is the registry bytes' (the fourth measurement, 2026-09-12)
    short = BY_ID["si_constitution"].text.encode("utf-8")
    assert rendered["shortconst"][1].prompted_prefix_hashes["constitution"] == cm.sha256_hex16(short) == "21e06f482a6559b3"
    assert hashlib.sha256(short).hexdigest() == SHORTCONST_SI_SHA256 and len(short) == 31222 and len(short.decode().splitlines()) == 121
    # the six SI-variant behavior dirs the fullconst spec implies exist with the exact bytes
    keys = cm.petri_expected_behavior_keys(fc)
    want = {f"{b}/PROMPTED_CONSTITUTION_FULL" for b in fc.behaviors}
    assert want <= set(keys) and len(keys) == 2 * len(fc.behaviors)
    root = REPO / "configs" / "capeval" / "petri"
    for k in want:
        front, _ = psc.parse_behavior_md((root / k / "BEHAVIOR.md").read_bytes())
        assert front["target_sysprompt_prefix"] == text, k


def _fake_bank():
    text = BY_ID["si_constitution_full"].text
    safety = BY_ID["safety_refusal"].text if "safety_refusal" in BY_ID else None
    entries = {
        "none": dc.RegistryEntry(cell_id="none", kind=cm.NONE_KIND),
        "prompted__constitution_full": dc.RegistryEntry(cell_id="prompted__constitution_full", kind=cm.PETRI_PROMPTED_KIND,
                                                        si_id="constitution_full", si_prefix=text.encode("utf-8")),
    }
    if safety is not None:
        entries["prompted__safety"] = dc.RegistryEntry(cell_id="prompted__safety", kind=cm.PETRI_PROMPTED_KIND,
                                                       si_id="safety", si_prefix=safety.encode("utf-8"))
    return dc.MappingCellSource(entries)


def test_enumerate_admits_the_full_constitution_cell_with_the_registry_prefix(tmp_path):
    rendered = _render_all(tmp_path)
    fc = rendered["fullconst"][1]
    root = REPO / "configs" / "capeval" / "petri"
    # behavior_files() reads EVERY variant on disk; the harness hash wants exactly the in-play inventory
    # (petri_expected_behavior_keys) — the runner prunes the same way (tiers/petri_runner.py)
    expected = set(cm.petri_expected_behavior_keys(fc))
    bf = {k: v for k, v in psc.behavior_files(root, fc.behaviors).items() if k in expected}
    assert set(bf) == expected
    cells = dc.enumerate_cells(fc, _fake_bank(), behavior_files=bf, prompt_files=PROMPT_FILES)
    ids = [c.cell_id for c in cells.canonical]
    assert ids == ["none", "prompted__constitution_full"] and not cells.absent
    prompted = next(c for c in cells.canonical if c.cell_id == "prompted__constitution_full")
    assert prompted.si_prefix_hash == fc.prompted_prefix_hashes["constitution_full"] == "82f88c5532e4939e"
    # a prefix that is NOT the registry's bytes refuses at enumeration (H8 is a check, not a label)
    bad = dc.MappingCellSource({
        "none": dc.RegistryEntry(cell_id="none", kind=cm.NONE_KIND),
        "prompted__constitution_full": dc.RegistryEntry(cell_id="prompted__constitution_full", kind=cm.PETRI_PROMPTED_KIND,
                                                        si_id="constitution_full",
                                                        si_prefix=BY_ID["si_constitution"].text.encode("utf-8"))})   # the EXCERPT
    with pytest.raises(dc.DefenseCellError):
        dc.enumerate_cells(fc, bad, behavior_files=bf, prompt_files=PROMPT_FILES)


def test_the_shortconst_view_is_a_full_battery_prompted_only_view_with_per_leg_windows():
    """The paper's constitution-SI arm: prompted-only, EVERY leg, per-leg windows max-prompt-sized for each leg's cap on the 128
    grid (measured with the pinned tokenizer), PAIR's first pass widened over the target's pin for the ~5.9k-token prefix; the readers
    the sbatch scripts (roster_view.py) and the printers share answer the same numbers; the grammar refuses a half block."""
    assert fr.view_legs(ROSTER, KEY, "shortconst") == LEGS_ALL and fr.view_legs_omitted(ROSTER, KEY, "shortconst") == {}
    assert fr.view_prompted_only(ROSTER, KEY, "shortconst") and fr.served_cells(ROSTER, KEY, "shortconst") == {"prompted__si_constitution": None}
    assert fr.view_controls(ROSTER, KEY, "shortconst") == ("mmlu", "xstest", "or_bench_hard_1k")
    vw = fr.view_windows(ROSTER, KEY, "shortconst")
    assert vw["kind"] == "per_leg" and {k: vw[k] for k in ("static", "controls", "rungs")} == SHORTCONST_WINDOWS and "short constitution SI as a prompted-only view" in vw["source"]
    assert fr.view_windows(ROSTER, KEY, "fullconst") == {"static": 45056, "controls": 45056, "controls_gsm8k": None, "rungs": {2048: 45056, 8192: 45056, 16384: 45056}, "source": None, "kind": "one_window"}
    assert fr.view_windows(ROSTER, KEY, "main")["kind"] == "table" and fr.view_windows(ROSTER, KEY, "main")["rungs"] == {}
    # every window is on the 128 grid and S = budget // L
    rb = sg.budget_for(KEY)
    for L, S in ((10496, 45), (6656, 71), (8064, 58), (14592, 32), (22784, 20), (SHORTCONST_MT_PAIR, 25)):
        assert L % sg.WINDOW_ALIGN == 0 and rb.entry.seqs_at(L) == S == rb.batched_token_budget // L
    # the multi-turn pins: the target's for every view, PAIR widened on shortconst only (and still below its re-run pin)
    mt = fr.view_multiturn_windows(ROSTER, KEY, "shortconst")
    assert {f: mt[f]["multiturn"] for f in mt} == {"crescendo": 49152, "pair": SHORTCONST_MT_PAIR, "actorattack": 49152}
    assert {f: mt[f]["multiturn_rerun"] for f in mt} == {"crescendo": 65536, "pair": 32768, "actorattack": 65536}
    assert fr.view_multiturn_windows(ROSTER, KEY, "main") == fr.view_multiturn_windows(ROSTER, KEY, "it") == MT_WINDOWS
    assert sg.multiturn_window(4096, SHORTCONST_MT_PAIR) == SHORTCONST_MT_PAIR   # geometry admits the pin (>= the derived minimum, on the grid)
    # the config the jobs read carries the view's pins; every other view the target's
    sc_cfg = yaml.safe_load((REPO / ROSTER["targets"][KEY]["views"]["shortconst"]["config"]).read_text())
    assert sc_cfg["attacks"]["multiturn_max_model_len"] == {"crescendo": 49152, "pair": SHORTCONST_MT_PAIR, "actorattack": 49152}
    assert sc_cfg["attacks"]["multiturn_rerun_max_model_len"] == {"crescendo": 65536, "pair": 32768, "actorattack": 65536}
    for view in ("main", "it", "fullconst"):
        cfg = yaml.safe_load((REPO / ROSTER["targets"][KEY]["views"][view]["config"]).read_text())
        assert cfg["attacks"]["multiturn_max_model_len"]["pair"] == 16384, view
    # roster_view.py prints the same facts for the sbatch side
    def run(*args):
        r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/full22/roster_view.py"), *args], capture_output=True, text=True)
        return r.returncode, r.stdout.strip()
    assert run(KEY, "shortconst", "window", "static") == (0, "10496") == run(KEY, "shortconst", "window", "bon")
    assert run(KEY, "shortconst", "window", "controls") == (0, "6656") == run(KEY, "shortconst", "window", "capability")
    assert [run(KEY, "shortconst", "rung_window", str(r))[1] for r in (2048, 8192, 16384)] == ["8064", "14592", "22784"]
    assert run(KEY, "shortconst", "max_model_len") == (0, "") and run(KEY, "shortconst", "prompted_only") == (0, "1")
    assert run(KEY, "fullconst", "window", "controls") == (0, "45056") and run(KEY, "fullconst", "rung_window", "16384") == (0, "45056")
    assert run(KEY, "main", "window", "static") == (0, "") and run(KEY, "main", "rung_window", "8192") == (0, "")
    assert run(KEY, "shortconst", "window", "petri")[0] == 2 and run(KEY, "shortconst", "rung_window", "4096")[0] == 2
    # the grammar: both shapes never together; a per-leg block is complete or refused; a view override may only widen and must stay below the re-run pin
    for mutate, msg in ((lambda v: v["serving"].__setitem__("max_model_len", 10496), "BOTH max_model_len and windows"),
                        (lambda v: v["serving"]["windows"].pop("controls"), r"missing \['controls'\]"),
                        (lambda v: v["serving"]["windows"]["rungs"].pop(2048), "exactly the rungs"),
                        (lambda v: v["serving"]["windows"].__setitem__("static", 10500), "multiple of the vLLM block size"),
                        (lambda v: v["serving"]["windows"].__setitem__("bogus", 1), "unknown keys"),
                        (lambda v: v["serving"]["windows"].pop("source"), "source must name"),
                        (lambda v: v["serving"]["windows"]["multiturn"].__setitem__("bogus", 16384), "unknown families")):
        bad = copy.deepcopy(ROSTER); mutate(bad["targets"][KEY]["views"]["shortconst"])
        with pytest.raises(ValueError, match=msg):
            fr.view_windows(bad, KEY, "shortconst")
    for pin, msg in ((16256, "below the target's pin"), (32768, "not below the family's re-run pin")):
        bad = copy.deepcopy(ROSTER); bad["targets"][KEY]["views"]["shortconst"]["serving"]["windows"]["multiturn"]["pair"] = pin
        with pytest.raises(ValueError, match=msg):
            fr.view_multiturn_windows(bad, KEY, "shortconst")
    with pytest.raises(ValueError, match="only max_model_len .* or windows"):
        bad = copy.deepcopy(ROSTER); bad["targets"][KEY]["views"]["fullconst"]["serving"]["bogus"] = 1
        fr.view_serving(bad, KEY, "fullconst")


def test_the_view_legs_and_the_prompted_only_kind_read_from_the_roster():
    assert fr.view_legs(ROSTER, KEY, "main") == LEGS_ALL == fr.view_legs(ROSTER, KEY, "it")
    assert fr.view_legs(ROSTER, KEY, "fullconst") == FULLCONST_LEGS
    om = fr.view_legs_omitted(ROSTER, KEY, "fullconst")
    assert tuple(om) == FULLCONST_OMITTED and all(why.strip() for why in om.values()) and "cheap legs only" in om["controls"]
    assert fr.view_serving(ROSTER, KEY, "fullconst") == {"max_model_len": 45056}
    assert fr.view_serving(ROSTER, KEY, "main") == {} and fr.view_prompted_only(ROSTER, KEY, "fullconst")
    assert not fr.view_prompted_only(ROSTER, KEY, "main")
    # the capability leg is MMLU ONLY on every view (gsm8k removed; roster capability_omitted)
    assert fr.view_controls(ROSTER, KEY, "fullconst") == ("mmlu",)            # capability only, no over-refusal
    assert fr.view_controls(ROSTER, KEY, "main") == ("mmlu", "xstest", "or_bench_hard_1k")
    assert fr.target_capability_controls(ROSTER, KEY) == ("mmlu",) and "owned by the evaluation lane" in fr.target_capability_omitted(ROSTER, KEY)["gsm8k"]
    # the grammar refuses an omission without a reason and a leg both run and omitted
    bad = copy.deepcopy(ROSTER)
    bad["targets"][KEY]["views"]["fullconst"]["legs_omitted"].pop("petri")
    with pytest.raises(ValueError, match="omitted without a reason"):
        fr.view_legs(bad, KEY, "fullconst")
    bad = copy.deepcopy(ROSTER)
    bad["targets"][KEY]["views"]["fullconst"]["legs"] = list(FULLCONST_LEGS) + ["petri"]
    with pytest.raises(ValueError, match="names a leg the view runs"):
        fr.view_legs(bad, KEY, "fullconst")
    # roster_view.py prints the same facts for the sbatch side
    import contextlib
    import io

    import roster_view as rv
    def run(*a):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rv.main(list(a))
        return rc, buf.getvalue().strip()
    assert run(KEY, "fullconst", "legs") == (0, "static,bon,capability")
    assert run(KEY, "fullconst", "controls") == (0, "mmlu") and run(KEY, "main", "controls") == (0, "mmlu,xstest,or_bench_hard_1k")
    assert run(KEY, "fullconst", "max_model_len") == (0, "45056") and run(KEY, "main", "max_model_len") == (0, "")
    assert run(KEY, "fullconst", "prompted_only") == (0, "1") and run(KEY, "main", "prompted_only") == (0, "0")
    assert run(KEY, "fullconst", "has_leg", "static")[0] == 0 and run(KEY, "fullconst", "has_leg", "petri")[0] == 3


def test_the_submit_plan_gives_shortconst_every_leg_at_its_per_leg_windows():
    """The greedy printer on the shortconst view (2026-09-12): every leg planned (ONE combined multi-turn job at 2 cells, the static+BoN
    job, the controls job, the Petri proxy), the static / controls geometry the view's per-leg windows, Petri the certified table window,
    the multi-turn geometry per family with PAIR at the view's widened pin; the sampled printer plans every phase at its window."""
    targets = ["llama8b", "llama8b/shortconst"]
    p = sn.plan(ROSTER, targets, {t: "cluster_b" for t in targets}, "cluster_b")
    by = {s["key"]: s for s in p["served"]}
    sc = by["llama8b/shortconst"]
    assert sc["legs"] == LEGS_ALL and sc["max_model_len"] is None and sc["leg_windows"]["kind"] == "per_leg" and p["omitted"]["llama8b/shortconst"] == {}
    assert sc["geom"]["static"] == (10496, 45) and sc["geom"]["controls"] == (6656, 71) and sc["geom"]["petri"] == (16384, 29)
    assert sc["geom"]["attacks"] == (49152, 9) and sc["rerun_geom"] == (65536, 7)     # the MAX over the families (the -all job's standup)
    g = sc["mt_geom"]
    assert {f: (g[f]["window"], g[f]["S"], g[f]["rerun_window"], g[f]["rerun_S"]) for f in g} == {"crescendo": (49152, 9, 65536, 7), "pair": (SHORTCONST_MT_PAIR, 25, 32768, 14), "actorattack": (49152, 9, 65536, 7)}
    assert all(g[f]["window_source"] == "pinned" for f in g) and "short constitution SI as a prompted-only view" in g["pair"]["pin_source"] and "short constitution SI" not in g["crescendo"]["pin_source"]
    assert sc["windows"]["pair"]["multiturn"] == SHORTCONST_MT_PAIR and by["llama8b"]["windows"] == MT_WINDOWS   # the lane view's pins untouched
    assert [j[0] for j in p["att"] if j[1] == "llama8b/shortconst"] == ["f22-att-llama8b-shortconst-all"]   # 2 cells: one job over the three families
    assert [j[2] for j in p["att"] if j[1] == "llama8b/shortconst"] == ["crescendo,pair,actorattack"]
    assert [j[0] for j in p["stat"] if j[1] == "llama8b/shortconst"] == ["f22-stat-llama8b-shortconst-all"]
    assert [j[0] for j in p["ctl"] if j[1] == "llama8b/shortconst"] == ["f22-ctl-llama8b-shortconst"] and [j[0] for j in p["petri"] if j[1] == "llama8b/shortconst"] == ["f22-petri-llama8b-shortconst"]
    assert next(j for j in p["petri"] if j[1] == "llama8b/shortconst")[2] == 3     # none + none__lens0 + the prompted cell
    sp = ss.plan(ROSTER, ["llama8b_shortconst"], {"llama8b_shortconst": "cluster_b"}, 9)
    geom = sp["served"][0]["geom"]
    assert geom["static_base"] == (10496, 45) and geom["static_esc"] == (14592, 32) and geom["static_esc16"] == (22784, 20)
    assert geom["controls_512"] == (6656, 71) and geom["controls_2048"] == (8064, 58) and geom["controls_8192"] == (14592, 32) and geom["controls_16384"] == (22784, 20)
    names = [j.name for j in sp["jobs"] + sp["esc"] + sp["esc16"]]
    assert names == ["f22s-llama8b_shortconst-sb0", "f22s-llama8b_shortconst-sb1", "f22s-llama8b_shortconst-ctl0", "f22s-llama8b_shortconst-ctl1",
                     "f22s-llama8b_shortconst-se0", "f22s-llama8b_shortconst-se1", "f22s-llama8b_shortconst-s16k0", "f22s-llama8b_shortconst-s16k1"]
    assert all(j.args.endswith(' 9 "" shortconst') for j in sp["jobs"] + sp["esc"] + sp["esc16"])
    assert ss.split_target("llama8b_shortconst") == ("llama8b", "shortconst") and "llama8b_shortconst" in ss.LLAMA_TAGS


def test_the_submit_plan_gives_fullconst_exactly_static_bon_capability_and_the_lane_views_every_leg():
    targets = ["llama8b", "llama8b/it", "llama8b/fullconst"]
    p = sn.plan(ROSTER, targets, {t: "cluster_a" for t in targets}, "cluster_a")
    by = {s["key"]: s for s in p["served"]}
    assert set(by) == set(targets) and not p["skipped"]
    # fullconst: static families + BoN (one -all static job: n < 5) + ONE controls job (gsm8k,mmlu); NO attacks, NO petri
    fc_stat = [j for j in p["stat"] if j[1] == "llama8b/fullconst"]
    assert [j[2] for j in fc_stat] == ["pasttense,codeattack,renellm,grok420,bon"]
    assert [j[2] for j in p["ctl"] if j[1] == "llama8b/fullconst"] == ["controls"]
    assert not [j for j in p["att"] if j[1] == "llama8b/fullconst"] and not [j for j in p["petri"] if j[1] == "llama8b/fullconst"]
    assert tuple(p["omitted"]["llama8b/fullconst"]) == FULLCONST_OMITTED
    assert all(why.strip() for why in p["omitted"]["llama8b/fullconst"].values()) and "cheap legs only" in p["omitted"]["llama8b/fullconst"]["controls"]
    # main / it: every leg (attacks, static + bon, controls, petri); nothing omitted
    for t in ("llama8b", "llama8b/it"):
        assert p["omitted"][t] == {} and by[t]["legs"] == LEGS_ALL
        assert [j for j in p["att"] if j[1] == t] and [j for j in p["stat"] if j[1] == t]
        assert [j for j in p["ctl"] if j[1] == t] and [j for j in p["petri"] if j[1] == t]
    # every fullconst geometry is the view's window x the budget's S there; the lane views the table's
    fc = by["llama8b/fullconst"]
    assert fc["max_model_len"] == 45056 and all(g == (45056, 10) for g in fc["geom"].values())
    # the multi-turn windows are PER FAMILY (roster caps.multiturn_target[family] -> geometry.multiturn_window at THAT cap, the
    # roster's PIN over it — caps.windows[family]): Crescendo / PAIR 49,152 x 9 (re-run 65,536 x 7),
    # ActorAttack 32,768 x 14 (re-run 49,152 x 9) on both lane views — never WINDOWS' 8,704; geom["attacks"] / rerun_geom are the
    # MAX over the families (what a combined standup would serve), the split jobs serve each family's own (mt_geom)
    assert {f: sg.multiturn_window(CAPS["multiturn_target"][f], MT_WINDOWS[f]["multiturn"]) for f in fr.MULTITURN_FAMILIES} == {f: g[0] for f, g in MT_GEOM.items()}
    assert by["llama8b"]["geom"]["static"] == (4864, 97) and by["llama8b"]["geom"]["attacks"] == (49152, 475_136 // 49152) == (49152, 9)
    assert by["llama8b"]["caps"] == CAPS and by["llama8b/it"]["geom"]["attacks"] == (49152, 9)
    assert by["llama8b"]["windows"] == by["llama8b/it"]["windows"] == MT_WINDOWS
    for t in ("llama8b", "llama8b/it"):
        g = by[t]["mt_geom"]
        assert {f: (g[f]["window"], g[f]["S"], g[f]["rerun_window"], g[f]["rerun_S"]) for f in g} == MT_GEOM
        assert {f: g[f]["cap"] for f in g} == MT_CAPS and all(g[f]["window_source"] == g[f]["rerun_window_source"] == "pinned" for f in g)
        assert all(g[f]["S"] == 475_136 // g[f]["window"] and g[f]["rerun_S"] == 475_136 // g[f]["rerun_window"] for f in g)
    assert by["llama8b"]["rerun_geom"] == by["llama8b/it"]["rerun_geom"] == (65536, 7)
    assert by["llama8b/fullconst"]["rerun_geom"] is None and by["llama8b/fullconst"]["mt_geom"] is None    # no multi-turn leg on fullconst: its geometry is the view's window
    assert by["llama8b"]["budget_source"] == sg.MEASURED_BUDGET_SOURCE


def test_the_sampled_plan_and_configs_follow_the_view_legs():
    tags = ["llama8b", "llama8b_it", "llama8b_fullconst"]
    p = ss.plan(ROSTER, tags, {t: "cluster_a" for t in tags}, 9)
    names = [j.name for j in p["jobs"] + p["esc"] + p["esc16"]]
    assert "f22s-llama8b_fullconst-ctl0" not in names and "f22s-llama8b_fullconst-ctl1" not in names    # no controls ladder
    assert {"f22s-llama8b_fullconst-sb0", "f22s-llama8b_fullconst-sb1", "f22s-llama8b_fullconst-se0", "f22s-llama8b_fullconst-se1",
            "f22s-llama8b_fullconst-s16k0", "f22s-llama8b_fullconst-s16k1"} <= set(names)   # the 16,384 rung too (round 2 P1)
    for t in ("llama8b", "llama8b_it"):
        assert {f"f22s-{t}-ctl0", f"f22s-{t}-ctl1", f"f22s-{t}-sb0", f"f22s-{t}-se0", f"f22s-{t}-s16k0"} <= set(names)
    # gsm8k is REMOVED from this lane: no gsm phase on any view; the sampled configs record the omission
    assert not [n for n in names if "-gsm" in n]
    fc = next(s for s in p["served"] if s["key"] == "llama8b_fullconst")
    # every phase at the view's 45,056 window EXCEPT the 16,384 rungs, which stand up at their own 51,200 (45,056 carries
    # 4,096 and 8,192 but not 16,384; S = 475,136 // 51,200 = 9). The controls_16384 row is informational here: no controls leg
    RUNGS_16K = {"static_esc16", "controls_16384"}
    assert all(g == (45056, 10) for ph, g in fc["geom"].items() if ph not in RUNGS_16K)
    assert all(fc["geom"][ph] == (51200, 9) for ph in RUNGS_16K)
    # the sampled configs: the fullconst ones exist for base + both rungs; readouts.gsm8k thinking OFF for this model
    for tag in tags:
        for trial in (0, 1):
            for esc in (False, 8192, 16384):
                cfg = yaml.safe_load(msc.output_path(tag, trial, esc).read_text())
                sd = cfg["sampled_decode"]
                assert (sd["temperature"], sd["top_p"]) == (0.6, 0.9) and "top_k" not in sd
                assert sd["readouts"]["gsm8k"] == {"omitted": True, "reason": fr.target_capability_omitted(ROSTER, KEY)["gsm8k"]}
                assert sd["readouts"]["mmlu"]["instrument"] == "letter_logprob_argmax"
                # the caps ride the wave config the sampled one embeds: the multi-turn cap on its own key, the controls' cap unchanged
                # (the multi-turn cap on its own per-family key beside main's wave-wide target_max_new_tokens 4,096; the controls'
                # cap on ITS key, attacks.controls_max_new_tokens = the caps table's controls_ladder — the 2026-09-11 reconciliation)
                assert cfg["attacks"]["multiturn_target_max_new_tokens"] == CAPS["multiturn_target"] and cfg["attacks"]["target_max_new_tokens"] == ROSTER["target_max_new_tokens"] == 4096
                assert cfg["attacks"]["controls_max_new_tokens"] == CAPS["controls_ladder"] == ROSTER["controls_max_new_tokens"]


#: the expected S per leg at the MEASURED 475,136 (the 2026-09-10 serving gate) — derived, then asserted.
#: {eval: (explicit window or None, L, S)}: the controls leg's 2,048 is the sbatch's EXPLICIT --max-model-len
#: (target_static.sbatch `capability:short 2048`); the geometry table's own capability:short entry is 1,536
#: (S 149) — the wave has always served the controls at 2,048 by that override, never at the table's window
S_LEGS = {"multiturn_attacks": (None, 8704, 54), "families_served:short": (None, 4864, 97), "bon": (None, 7424, 64),
          "capability:short": (2048, 2048, 232), "petri": (None, 16384, 29)}


def test_every_geometry_derives_from_budget_for_llama8b_and_a_measured_row_moves_them_all(monkeypatch):
    rb = sg.budget_for(KEY)
    assert rb.batched_token_budget == 475_136 and rb.measured and rb.per_model and not rb.provisional
    assert rb.entry.basis.first_refused == 507_904 and rb.entry.basis.measured_on == "2026-09-10"
    for ev, (explicit, L, S) in S_LEGS.items():
        r = sg.resolve_geometry(ev, max_model_len=explicit, model_key=KEY)
        assert (r.max_model_len, r.max_num_seqs) == (L, S) == (L, rb.batched_token_budget // L), ev
        assert r.is_identity and r.budget_source == sg.MEASURED_BUDGET_SOURCE
    assert sg.geometry_for("capability", task="short").max_model_len == 1536     # the table's own controls window
    assert sg.resolve_geometry("capability:short", model_key=KEY).max_num_seqs == 475_136 // 1536 == 309
    fc = sg.resolve_geometry("families_served:short", max_model_len=45056, model_key=KEY)
    assert fc.max_num_seqs == 10 == 475_136 // 45056 and fc.is_identity and not fc.over_budget
    assert "model llama8b: measured]" in fc.describe()
    # ANOTHER MEASURED ROW LANDS (a re-measurement through the production lens standup): replace the entry in
    # the ONE table and every S follows — resolve_geometry, the printers, the load check's S (endpoint.json)
    measured = sg.ModelBudget(
        model_key=KEY, batched_token_budget=458_752, certified=False, source=sg.MEASURED_BUDGET_SOURCE,
        basis=sg.BudgetBasis(method="test: a lens-stack measurement", max_model_len=16384, max_num_seqs=28,
                             gpu_memory_utilization=0.9, kv_tokens_reported=458_752, probe_job_id="t",
                             measured_on="2026-09-11", source="test", first_refused=524_288))
    monkeypatch.setitem(sg.BUDGET_BY_MODEL, KEY, measured)
    rb2 = sg.budget_for(KEY)
    assert rb2.measured and rb2.batched_token_budget == 458_752
    for ev, (explicit, L, _S) in S_LEGS.items():
        assert sg.resolve_geometry(ev, max_model_len=explicit, model_key=KEY).max_num_seqs == 458_752 // L, ev
    assert sg.resolve_geometry("families_served:short", max_model_len=45056, model_key=KEY).max_num_seqs == 10
    targets = ["llama8b", "llama8b/fullconst"]
    p = sn.plan(ROSTER, targets, {t: "cluster_a" for t in targets}, "cluster_a")
    by = {s["key"]: s for s in p["served"]}
    L_mt = sg.multiturn_window(CAPS["multiturn_target"]["crescendo"], MT_WINDOWS["crescendo"]["multiturn"])
    assert by["llama8b"]["geom"]["static"] == (4864, 94) and by["llama8b"]["geom"]["attacks"] == (L_mt, 458_752 // L_mt) == (49152, 9)
    # a smaller budget moves S per family, never L: the pins are windows (458,752 // 49,152 is still 9; // 65,536 = 7; // 16,384 = 28; // 32,768 = 14)
    assert {f: (g["window"], g["S"]) for f, g in by["llama8b"]["mt_geom"].items()} == {"crescendo": (49152, 9), "pair": (16384, 458_752 // 16384), "actorattack": (49152, 9)}
    assert by["llama8b"]["mt_geom"]["pair"]["S"] == 28 and by["llama8b"]["mt_geom"]["pair"]["rerun_S"] == 458_752 // 32768 == 14 and by["llama8b"]["mt_geom"]["crescendo"]["rerun_S"] == 458_752 // 65536 == 7
    assert by["llama8b/fullconst"]["geom"]["static"] == (45056, 10) and by["llama8b"]["budget_source"] == "measured"
    sp = ss.plan(ROSTER, ["llama8b_fullconst"], {"llama8b_fullconst": "cluster_a"}, 9)
    geom = sp["served"][0]["geom"]
    assert all(g == (45056, 10) for ph, g in geom.items() if ph not in ("static_esc16", "controls_16384"))
    # the 16,384 rung's own window (round 2 P1), S from the row IN FORCE — the moved 458,752 here, so 8 (9 at the measured 475,136)
    assert geom["static_esc16"] == (51200, 458_752 // 51200) == (51200, 8) == geom["controls_16384"]


def test_the_caps_table_is_the_one_source_and_every_window_derives_from_it():
    """The caps table: the multi-turn cap FLAT on its own config key — PER FAMILY (Crescendo / PAIR 4,096,
    ActorAttack at its filed 512), the controls' 512 untouched, each family's multi-turn serving window and
    over-window re-run window DERIVED from ITS cap through the geometry table (p99 rendered
    prompt 7,496 / measured max call 12,657 + cap, on the table's 128-token grid; S = the measured budget // L) — never a
    literal in a launcher; the roster grammar refuses a pending or missing cap and a family map missing a family; the
    driver and the load check resolve the family they run and refuse one the config does not name."""
    assert CAPS == {"source": CAPS["source"], "multiturn_target": MT_CAPS, "multiturn_target_source": CAPS["multiturn_target_source"],
                    "attacker": 1024, "static": 4096, "bon": BON_CAP, "controls_ladder": 512, "petri": 1024}
    assert "the canonical battery of 2026-09-11" in CAPS["source"]
    assert "the canonical battery of 2026-09-11" in CAPS["multiturn_target_source"] and "4,096" in CAPS["multiturn_target_source"] and "ActorAttack KEPT" in CAPS["multiturn_target_source"]
    assert fr.target_multiturn_families(ROSTER, KEY) == (fr.MULTITURN_FAMILIES, {})   # ActorAttack KEPT, all three families run
    for view in VIEWS:
        cfg = yaml.safe_load((REPO / ROSTER["targets"][KEY]["views"][view]["config"]).read_text())
        a = cfg["attacks"]
        assert a["multiturn_target_max_new_tokens"] == MT_CAPS and a["target_max_new_tokens"] == 4096 and a["attacker_max_new_tokens"] == 1024
        assert a["controls_max_new_tokens"] == 512   # the controls ladder's root rides its own key (cell_controls.controls_cap)
        # bon.max_new_tokens is WRITTEN from the caps table (the NATIVE 4,096 pass; the body's 512 is the campaign's); every other bon key the body's
        assert a["static_max_new_tokens"] == 4096 and cfg["bon"]["max_new_tokens"] == BON_CAP == CAPS["bon"] and cfg["bon"]["n_max"] == 75
    # the driver reads the multi-turn cap from its OWN key — the family's entry of a per-family map (a missing family REFUSES; a
    # map with no family named refuses), a flat int for every family — and falls back to the shared cap without the key
    sys.path.insert(0, str(REPO / "scripts"))
    import jailbreak_run as jr
    fam_cfg = {"attacks": {"target_max_new_tokens": 512, "multiturn_target_max_new_tokens": dict(MT_CAPS)}}
    assert [jr._multiturn_target_max_new_tokens(fam_cfg, f) for f in ("crescendo", "pair", "actorattack")] == [4096, 4096, 4096]
    with pytest.raises(SystemExit, match="names no cap for 'bon'"):
        jr._multiturn_target_max_new_tokens(fam_cfg, "bon")
    with pytest.raises(SystemExit, match="needs the attack family"):
        jr._multiturn_target_max_new_tokens(fam_cfg)
    short = {"attacks": {"target_max_new_tokens": 512, "multiturn_target_max_new_tokens": {"crescendo": 4096, "pair": 4096}}}
    with pytest.raises(SystemExit, match="names no cap for 'actorattack'"):
        jr._multiturn_target_max_new_tokens(short, "actorattack")
    assert jr._multiturn_target_max_new_tokens({"attacks": {"target_max_new_tokens": 512, "multiturn_target_max_new_tokens": 4096}}, "pair") == 4096
    assert jr._multiturn_target_max_new_tokens({"attacks": {"target_max_new_tokens": 512}}, "crescendo") == 512
    assert jr._target_max_new_tokens(fam_cfg) == 512
    # the load check's probe cap follows the same key, for the family(ies) the standup serves: one family -> its cap; a comma list
    # (a combined job) -> the MAX over them; a family the map does not name, or no family, refuses (never a default)
    import load_check as lc
    assert lc.cap_for("multiturn_attacks", fam_cfg, None, attack="crescendo") == (4096, "config:attacks.multiturn_target_max_new_tokens[crescendo]")
    assert lc.cap_for("multiturn_attacks", fam_cfg, None, attack="actorattack") == (4096, "config:attacks.multiturn_target_max_new_tokens[actorattack]")
    # every family at 4,096 (the ruling): the max over a combined job is the FIRST family named (a tie), and the source says so
    assert lc.cap_for("multiturn_attacks", fam_cfg, None, attack="actorattack,pair") == (4096, "config:attacks.multiturn_target_max_new_tokens[actorattack] (max over actorattack,pair)")
    with pytest.raises(SystemExit, match="pass --attack"):
        lc.cap_for("multiturn_attacks", fam_cfg, None)
    with pytest.raises(SystemExit, match=r"names no cap for \['bogus'\]"):
        lc.cap_for("multiturn_attacks", fam_cfg, None, attack="bogus")
    assert lc.cap_for("multiturn_attacks", {"attacks": {"target_max_new_tokens": 512, "multiturn_target_max_new_tokens": 4096}}, None) == (4096, "config:attacks.multiturn_target_max_new_tokens")
    assert lc.cap_for("multiturn_attacks", {"attacks": {"target_max_new_tokens": 512}}, None) == (512, "config:attacks.target_max_new_tokens")
    assert lc.cap_for("capability", fam_cfg, None) == (512, "config:attacks.target_max_new_tokens")
    assert lc.cap_for("multiturn_attacks", fam_cfg, 8192, attack="pair") == (8192, "flag:--cap")
    # the derived windows: p99_prompt 7,496 + 4,096 = 11,592 -> next block 11,600 -> grid 11,648; S = 475,136 // 11,648 = 40
    b = sg.geometry_for("multiturn_attacks").basis
    assert (b.p99_prompt, b.max_total) == (7496, 12657) and sg.WINDOW_ALIGN == 128
    L_mt = sg.multiturn_window(4096)
    assert L_mt == sg.align_window(sg.next_block(7496 + 4096)) == 11648 and L_mt % 16 == 0 and L_mt > 8704
    r = sg.resolve_geometry("multiturn_attacks", max_model_len=L_mt, model_key=KEY, strict_identity=True)
    assert (r.max_model_len, r.max_num_seqs) == (11648, 40) and r.is_identity and not r.over_budget
    # the over-window RE-RUN window: the measured MAX call 12,657 + 4,096 = 16,753 does NOT fit 16,384; derived 16,768, S 28
    L_re = sg.multiturn_rerun_window(4096)
    assert 12657 + 4096 > 16384 and L_re == 16768 and sg.resolve_geometry("multiturn_attacks", max_model_len=L_re, model_key=KEY, strict_identity=True).max_num_seqs == 28
    # ActorAttack's cap 512: the derivation gives 8,064 (7,496 + 512 -> 8,016 -> grid 8,064) and a re-run of 13,184 (12,657 + 512),
    # both BELOW the table's 8,704 / 16,384 — the table's 8,704 stands for a config WITHOUT the key (no other target moves); the
    # lane PINS ActorAttack at 16,384 / 32,768 over these, see the pinned-windows test
    assert sg.multiturn_window(512) == 8064 < 8704 and sg.multiturn_rerun_window(512) == 13184
    with pytest.raises(ValueError):
        sg.derived_window("multiturn_attacks", 0)
    # the Petri per-turn cap is the composer's window // 16 = the table's petri cap
    pr = pytest.importorskip("sisteer.capeval.tiers.petri_runner")
    assert pr.target_response_cap() == CAPS["petri"] == 1024
    # the roster grammar refuses a pending / missing cap, a family map missing a family / with an unknown family / without a
    # source, and an omission without a reason; a flat int is every family at that cap (the block's source covers it)
    for mutate, msg in ((lambda c: c.__setitem__("multiturn_target", "pending"), "positive int"),
                        (lambda c: c["multiturn_target"].pop("actorattack"), r"missing the families \['actorattack'\]"),
                        (lambda c: c["multiturn_target"].__setitem__("bon", 512), r"unknown families \['bon'\]"),
                        (lambda c: c["multiturn_target"].__setitem__("pair", "pending"), "positive int"),
                        (lambda c: c["multiturn_target"].pop("source"), "multiturn_target.source"),
                        (lambda c: c.pop("petri"), "missing"),
                        (lambda c: c.__setitem__("bogus", 1), "unknown"),
                        (lambda c: c.pop("source"), "source")):
        bad = copy.deepcopy(ROSTER); mutate(bad["targets"][KEY]["caps"])
        with pytest.raises(ValueError, match=msg):
            fr.target_caps(bad, KEY)
    flat = copy.deepcopy(ROSTER); flat["targets"][KEY]["caps"]["multiturn_target"] = 2048
    fc = fr.target_caps(flat, KEY)
    assert fc["multiturn_target"] == {"crescendo": 2048, "pair": 2048, "actorattack": 2048} and fc["multiturn_target_source"] == CAPS["source"]
    bad = copy.deepcopy(ROSTER); bad["targets"][KEY]["capability_omitted"].pop("gsm8k")
    with pytest.raises(ValueError, match="without a reason"):
        fr.target_capability_controls(bad, KEY)
    bad = copy.deepcopy(ROSTER); bad["targets"][KEY]["capability_controls"] = ["gsm8k", "mmlu"]
    with pytest.raises(ValueError, match="names a control the target runs"):
        fr.target_capability_controls(bad, KEY)
    # a caps value that disagrees with the body's leg value REFUSES generation (never a silent second value) ...
    bad = copy.deepcopy(ROSTER); bad["targets"][KEY]["caps"]["attacker"] = 2048
    with pytest.raises(SystemExit, match="caps.attacker=2048"):
        gen.render_config(bad, PINS, KEY, "main")
    # ... EXCEPT bon.max_new_tokens, the ONE cap the table WRITES (the native best-of-75 pass at the table's
    # cap; the body's 512 is the campaign's): the generated config carries the table's value, the rest of the bon block the body's
    moved = copy.deepcopy(ROSTER); moved["targets"][KEY]["caps"]["bon"] = 2048
    got = yaml.safe_load(gen.render_config(moved, PINS, KEY, "main"))["bon"]
    body_bon = yaml.safe_load((REPO / ROSTER["targets"][KEY]["body_config"]).read_text())["bon"]
    # (main's wave-wide climb keys — bon.climb_cap / climb_decode, the canonical 512-store climb every config carries since
    # 2026-09-11 (climb_decode `as_drawn` since 2026-09-17) — are the generator's additions, not the body's; this lane's native
    # pass has nothing to climb)
    assert got["max_new_tokens"] == 2048 and {k: v for k, v in got.items() if k not in ("max_new_tokens", "climb_cap", "climb_decode")} == {k: v for k, v in body_bon.items() if k != "max_new_tokens"}
    assert got["climb_decode"] == "as_drawn" and "climb_temperature" not in got
    assert body_bon["max_new_tokens"] == 512
    # the keep-or-drop switch: a dropped family is a recorded omission with a reason; the cap map still names it; the
    # printer plans no job for it and prints the omission; refusals for an unknown family, a pending reason, or dropping every family
    drop = copy.deepcopy(ROSTER); drop["targets"][KEY]["caps"]["multiturn_families_omitted"] = {"actorattack": "the user's cost call (test)"}
    assert fr.target_multiturn_families(drop, KEY) == (("crescendo", "pair"), {"actorattack": "the user's cost call (test)"})
    assert fr.target_caps(drop, KEY) == CAPS                                      # the switch is not a cap: target_caps is unchanged by it
    pd_ = sn.plan(drop, ["llama8b"], {"llama8b": "cluster_a"}, "cluster_a")
    assert [j[2] for j in pd_["att"]] == ["crescendo", "pair"] and pd_["served"][0]["families_omitted"] == {"actorattack": "the user's cost call (test)"}
    assert "multi-turn family OMITTED actorattack" in gen.render_config(drop, PINS, KEY, "main")
    assert yaml.safe_load(gen.render_config(drop, PINS, KEY, "main"))["attacks"]["multiturn_target_max_new_tokens"] == MT_CAPS
    for mutate, msg in ((lambda c: c.__setitem__("multiturn_families_omitted", {"bon": "x"}), "unknown families"),
                        (lambda c: c.__setitem__("multiturn_families_omitted", {"pair": "pending the user"}), "final reason"),
                        (lambda c: c.__setitem__("multiturn_families_omitted", {f: "x" for f in fr.MULTITURN_FAMILIES}), "drops every multi-turn family"),
                        (lambda c: c.__setitem__("multiturn_families_omitted", ["actorattack"]), "must be a mapping")):
        bad = copy.deepcopy(ROSTER); mutate(bad["targets"][KEY]["caps"])
        with pytest.raises(ValueError, match=msg):
            fr.target_caps(bad, KEY)


def test_the_pinned_windows_widen_the_derived_ones_and_never_shrink_them():
    """The lane PINS its multi-turn windows, widening the derived ones one step per family: the 32,768 minimum was
    RETRACTED — GLM's realized Crescendo full-conversation context at a 4,096 target cap reached 34,361 and is still
    climbing (8,731 -> 16,428 -> 34,361, each read from an incomplete store, each a floor on the max), past the table's
    p99 prompt 7,496 + cap the derivation sizes to; multi-turn is FLAT, so the lane PINS the Crescendo / PAIR first-pass
    window at 49,152 and the over-window re-run at 65,536, both through the measured budget (S 9 and 7). ActorAttack
    stays at its filed cap 512 but runs 23-26 target forwards per goal, and its earlier 16,384 / 32,768 were sized to the
    table's measured max 12,657 — itself a floor — so it is PINNED at 32,768 (S 14) with the re-run at 49,152 (S 9).
    65,536 is within the served model's 131,072 context and the tree has no lower ceiling. Every pin is honoured over the
    family's derived value by every consumer (the roster grammar, the configs, the printer, the wave manifest), a pin
    below the derived minimum or off the grid is refused, a re-run window not wider than the first pass is refused, and a
    target without pins derives exactly as before."""
    assert set(MT_WINDOWS) == set(fr.MULTITURN_FAMILIES)
    assert {f: (w["multiturn"], w["multiturn_rerun"]) for f, w in MT_WINDOWS.items()} == {f: (g[0], g[2]) for f, g in MT_GEOM.items()}
    assert all("2026-09-11" in MT_WINDOWS[f]["source"] for f in fr.MULTITURN_FAMILIES)   # the canonical battery's values
    assert "the canonical battery" in MT_WINDOWS["crescendo"]["source"] and "canonical" in MT_WINDOWS["pair"]["source"] and "ActorAttack KEPT at 4,096" in MT_WINDOWS["actorattack"]["source"]
    assert "34,361" in MT_WINDOWS["crescendo"]["source"] and "14:23" in MT_WINDOWS["actorattack"]["source"]
    for f in fr.MULTITURN_FAMILIES:
        cap_f, w = CAPS["multiturn_target"][f], MT_WINDOWS[f]
        g = sg.multiturn_geometry(cap_f, model_key=KEY, window=w["multiturn"], rerun_window=w["multiturn_rerun"])
        assert (g["window"], g["S"], g["rerun_window"], g["rerun_S"]) == MT_GEOM[f], f
        assert (g["derived_window"], g["derived_rerun_window"]) == MT_DERIVED[f] and g["window_source"] == g["rerun_window_source"] == "pinned"
        assert g["budget"] == 475_136 and g["S"] == 475_136 // g["window"] and g["rerun_S"] == 475_136 // g["rerun_window"]
        for L, S in ((g["window"], g["S"]), (g["rerun_window"], g["rerun_S"])):
            r = sg.resolve_geometry("multiturn_attacks", max_model_len=L, model_key=KEY, strict_identity=True)
            assert (r.max_model_len, r.max_num_seqs) == (L, S) and r.is_identity and not r.over_budget
        # the derived path is untouched: no pin -> the family's derived minimums; the record says so
        d = sg.multiturn_geometry(cap_f, model_key=KEY)
        assert (d["window"], d["rerun_window"]) == MT_DERIVED[f] and d["window_source"] == d["rerun_window_source"] == "derived"
    cap = CAPS["multiturn_target"]["crescendo"]
    d = sg.multiturn_geometry(cap, model_key=KEY)
    assert (d["window"], d["S"], d["rerun_window"], d["rerun_S"]) == (11648, 40, 16768, 28)
    assert sg.multiturn_window(cap) == 11648 and sg.multiturn_rerun_window(cap) == 16768
    assert (sg.multiturn_geometry(512, model_key=KEY)["window"], sg.multiturn_geometry(512, model_key=KEY)["S"]) == (8064, 58)
    # a pin may only WIDEN: below the derived minimum, off the grid, not an int, or a re-run not wider than the first pass -> refused
    with pytest.raises(ValueError, match="below the derived minimum"):
        sg.multiturn_window(cap, 11520)
    with pytest.raises(ValueError, match="below the derived minimum"):
        sg.multiturn_rerun_window(cap, 16384)
    with pytest.raises(ValueError, match="grid"):
        sg.multiturn_window(cap, 32768 + 64)
    with pytest.raises(ValueError, match="positive int"):
        sg.multiturn_window(cap, "32768")
    with pytest.raises(ValueError, match="not wider than the first pass"):
        sg.multiturn_rerun_window(cap, 32768, window=32768)
    with pytest.raises(ValueError, match="not wider than the first pass"):
        sg.multiturn_geometry(cap, model_key=KEY, window=32768)          # the derived re-run (16,768) is below a pinned first pass
    with pytest.raises(ValueError, match="exceeds the llama8b batched-token budget"):
        sg.multiturn_geometry(cap, model_key=KEY, window=475_136 + 128, rerun_window=475_136 + 256)
    assert sg.multiturn_window(cap, 11648) == 11648      # a pin AT the derived minimum is legal (it widens nothing)
    # ActorAttack's pins are judged against ITS derived minimums (8,064 / 13,184 at cap 512), never Crescendo's: 16,384 first pass and
    # 32,768 re-run are legal at cap 512; the same 16,384 as a RE-RUN pin at cap 4,096 is refused (16,384 < 16,768), and 7,936 as a
    # first pass at cap 512 is refused (below 8,064)
    assert sg.multiturn_window(512, 16384) == 16384 and sg.multiturn_rerun_window(512, 32768, window=16384) == 32768
    with pytest.raises(ValueError, match="below the derived minimum"):
        sg.multiturn_rerun_window(4096, 16384)
    with pytest.raises(ValueError, match="below the derived minimum"):
        sg.multiturn_window(512, 7936)
    # every view config carries the per-family pins on the multi-turn keys; each family's re-run key is wider than its first pass — the
    # target's pins on every view but shortconst, whose PAIR first pass is the VIEW's widened pin (f22_roster.view_multiturn_windows)
    for view in VIEWS:
        a = yaml.safe_load((REPO / ROSTER["targets"][KEY]["views"][view]["config"]).read_text())["attacks"]
        want_first = {f: (SHORTCONST_MT_PAIR if view == "shortconst" and f == "pair" else g[0]) for f, g in MT_GEOM.items()}
        assert a["multiturn_max_model_len"] == want_first and a["multiturn_rerun_max_model_len"] == {f: g[2] for f, g in MT_GEOM.items()}, view
        assert a["multiturn_target_max_new_tokens"] == MT_CAPS
        assert all(a["multiturn_rerun_max_model_len"][f] > a["multiturn_max_model_len"][f] for f in fr.MULTITURN_FAMILIES)
    # the roster grammar: an unknown family, unknown keys, a missing source, a bad value, an empty block refuse; a flat block pins
    # every family alike; a target without the block derives
    for mutate, msg in ((lambda w: w.__setitem__("bon", {"multiturn": 32768, "source": "x"}), "unknown families"),
                        (lambda w: w["pair"].__setitem__("bogus", 1), "unknown keys"),
                        (lambda w: w["pair"].pop("source"), "source"),
                        (lambda w: w["actorattack"].__setitem__("multiturn", 16376), "multiple of 16"),
                        (lambda w: w["crescendo"].__setitem__("multiturn_rerun", "pending"), "positive int"),
                        (lambda w: [w["pair"].pop(k) for k in ("multiturn", "multiturn_rerun")], "names no window")):
        bad = copy.deepcopy(ROSTER); mutate(bad["targets"][KEY]["caps"]["windows"])
        with pytest.raises(ValueError, match=msg):
            fr.target_windows(bad, KEY)
    flatw = copy.deepcopy(ROSTER); flatw["targets"][KEY]["caps"]["windows"] = {"multiturn": 32768, "multiturn_rerun": 49152, "source": "one block"}
    assert fr.target_windows(flatw, KEY) == {f: {"multiturn": 32768, "multiturn_rerun": 49152, "source": "one block"} for f in fr.MULTITURN_FAMILIES}
    assert fr.target_windows(ROSTER, "gemma4e4b") == {} and "windows" not in CAPS
    # a pin below the family's derived minimum refuses GENERATION (never a config that would refuse at a standup): 8,704 is below
    # the 11,648 every family derives at cap 4,096; a pin AT the minimum is legal and judged per family (the other pins untouched)
    bad = copy.deepcopy(ROSTER); bad["targets"][KEY]["caps"]["windows"]["crescendo"]["multiturn"] = 8704
    with pytest.raises(ValueError, match="below the derived minimum"):
        gen.render_config(bad, PINS, KEY, "main")
    bad = copy.deepcopy(ROSTER); bad["targets"][KEY]["caps"]["windows"]["actorattack"]["multiturn"] = 8704
    with pytest.raises(ValueError, match="below the derived minimum"):
        gen.render_config(bad, PINS, KEY, "main")
    ok = copy.deepcopy(ROSTER); ok["targets"][KEY]["caps"]["windows"]["actorattack"]["multiturn"] = 11648
    a = yaml.safe_load(gen.render_config(ok, PINS, KEY, "main"))["attacks"]
    assert a["multiturn_max_model_len"]["actorattack"] == 11648 and a["multiturn_max_model_len"]["crescendo"] == 49152 and a["multiturn_max_model_len"]["pair"] == 16384
    # a target whose roster drops ONE family's pins renders that family's window entries absent (it derives) and the others pinned;
    # dropping every pin renders the multi-turn caps WITHOUT the window keys (the derived path, byte for byte)
    one = copy.deepcopy(ROSTER); one["targets"][KEY]["caps"]["windows"].pop("actorattack")
    a = yaml.safe_load(gen.render_config(one, PINS, KEY, "main"))["attacks"]
    assert set(a["multiturn_max_model_len"]) == set(a["multiturn_rerun_max_model_len"]) == {"crescendo", "pair"} and a["multiturn_target_max_new_tokens"] == MT_CAPS
    nopin = copy.deepcopy(ROSTER); nopin["targets"][KEY]["caps"].pop("windows")
    a = yaml.safe_load(gen.render_config(nopin, PINS, KEY, "main"))["attacks"]
    assert "multiturn_max_model_len" not in a and "multiturn_rerun_max_model_len" not in a and a["multiturn_target_max_new_tokens"] == MT_CAPS
    # (a generated config can never lack a family's cap — the roster grammar requires every family; a hand-edited one is refused by
    # the readers, never defaulted — the driver / load-check checks above)
    # the printer plans the per-family pinned windows for both lane views (and the derived ones when the roster carries no pin)
    p = sn.plan(ROSTER, ["llama8b", "llama8b/it"], {"llama8b": "cluster_a", "llama8b/it": "cluster_a"}, "cluster_a")
    for s in p["served"]:
        assert s["geom"]["attacks"] == (49152, 9) and s["rerun_geom"] == (65536, 7) and s["windows"] == MT_WINDOWS
        assert {f: (g["window"], g["S"], g["rerun_window"], g["rerun_S"]) for f, g in s["mt_geom"].items()} == MT_GEOM
        assert s["mt_geom"]["actorattack"]["pin_source"] == MT_WINDOWS["actorattack"]["source"]
    # a view with its own serving window (fullconst, 45,056) keeps it for every standup: the pin never overrides a view window, and no
    # re-run geometry is planned there (no multi-turn leg) — the printer labels that line VIEW, never PINNED / DERIVED
    fc = sn.plan(ROSTER, ["llama8b/fullconst"], {"llama8b/fullconst": "cluster_a"}, "cluster_a")["served"][0]
    assert fc["geom"]["attacks"] == (45056, 10) and fc["rerun_geom"] is None and fc["mt_geom"] is None and fc["windows"] == MT_WINDOWS and fc["max_model_len"] == 45056
    p = sn.plan(nopin, ["llama8b"], {"llama8b": "cluster_a"}, "cluster_a")
    s = p["served"][0]
    assert s["geom"]["attacks"] == (11648, 40) and s["rerun_geom"] == (16768, 28) and s["windows"] == {}
    assert {f: (g["window"], g["S"], g["rerun_window"], g["rerun_S"], g["window_source"]) for f, g in s["mt_geom"].items()} == {
        "crescendo": (11648, 40, 16768, 28, "derived"), "pair": (11648, 40, 16768, 28, "derived"), "actorattack": (11648, 40, 16768, 28, "derived")}
    p = sn.plan(one, ["llama8b"], {"llama8b": "cluster_a"}, "cluster_a")
    g = p["served"][0]["mt_geom"]
    assert (g["actorattack"]["window"], g["actorattack"]["window_source"], g["crescendo"]["window"], g["crescendo"]["window_source"]) == (11648, "derived", 49152, "pinned")


def test_pins_from_an_empty_staging_manifest_renders_unpinned_configs(tmp_path):
    staging = tmp_path / "staging_manifest.json"
    staging.write_text(json.dumps([]))                       # nothing staged yet: no pin is invented
    pins = gen.pins_from_staging_manifest(staging)
    assert pins == {}
    # -- staged-state pin (written by the lane's staging launcher): the pins flow BEGIN --
    rendered = _render_all(tmp_path, pins=pins)
    # the staged objects are in cells (the lane's staging launcher moved them): an EMPTY manifest renders every object cell
    # UNPINNED — no pin is invented — while the committed configs (rendered with the clusters' pins) carry none; what a partial
    # move left pending is named in the header
    for view in ("main", "it"):
        assert "UNPINNED" in rendered[view][2]
        assert "pending_fit      (none)" in rendered["main"][2]
        assert "pending_fit      (none)" in rendered["it"][2]
    for view in ("fullconst", "shortconst"):   # the two prompted-only views serve no object: nothing to pin
        assert "UNPINNED" not in rendered[view][2]
    for view in VIEWS:
        assert "UNPINNED" not in (REPO / ROSTER["targets"][KEY]["views"][view]["config"]).read_text()
    # -- staged-state pin: the pins flow END --
    # a staging manifest that lands one object: the pin is copied, never typed (the example is a ROSTERED cell — the
    # constitution LoRA; the PSR objects stay staged and pinned but are no longer in the roster)
    staging.write_text(json.dumps([{"target": KEY, "staged_id": "base__constv2-mix-full__lora_r4_ce",
                                    "object_sha256": "ab" * 32, "volume_path": "cluster_a:/x",
                                    "cluster_a_verified": True, "cluster_b_verified": True}]))
    pins = gen.pins_from_staging_manifest(staging)
    assert pins[KEY]["base__constv2-mix-full__lora_r4_ce"]["object_sha256"] == "ab" * 32
    # the fits are named by the roster (every_staged_id) and enter `cells` only through this flow
    ids = fr.every_staged_id(ROSTER, KEY)
    assert "base__constv2-mix-full__lora_r4_ce" in ids and "itdoc__constv2ctl2-full__lora_r4_ce" in ids
    # exactly once, up to the ONE admitted re-serve (b64-capability-agentic-queue, 2026-09-23): the all-models one-seed
    # view `s1` RE-SERVES main's three base-lane objects under the seed grammar (one seed per family; the s1 bank is a
    # volume-internal copy of the main bank) and the paired view `s1p` re-serves s1's r=4 companion (the twin
    # marker) — four repeats, each a view cell of the same config cell id; anything else is the collision the rule guards
    from collections import Counter
    repeats = {sid: n for sid, n in Counter(ids).items() if n > 1}
    assert repeats == {"base__constv2-mix-full__lora_r4_ce": 3, "base__constv2ctl2-full__lora_r4_ce": 2,
                       "derived__constv2-mix-full__lora_r4_ce_minus_ctl": 2}, repeats
    assert len(ids) == len(set(ids)) + 4


def test_xstest_over_refusal_is_read_on_the_safe_subset_and_the_replicate_floor_pair_is_recorded():
    from sisteer.capeval.tiers import judged
    assert judged.XSTEST_N_SAFE == 250 and judged.XSTEST_N_UNSAFE == 200
    import pandas as pd
    df = pd.DataFrame({"prompt": [f"p{i}" for i in range(450)],
                       "type": ["safe_x"] * 250 + ["contrast_y"] * 200})
    rows = judged.xstest_rows_from_frame(df)
    assert sum(r["subset"] == "safe" for r in rows) == 250 and sum(r["subset"] == "unsafe" for r in rows) == 200
    src = (REPO / "src/sisteer/capeval/tiers/judged.py").read_text()
    # the judge iterates the halves PRESENT in the spec (safe required; unsafe only when the spec has it)
    assert 'for subset in summary["subsets"]' in src and "xst_{subset}_{lab}_frac_aggregate" in src
    assert 'if "safe" not in present:' in src
    man = json.loads((REPO / "scripts/sdfbase/full22/wave_manifest.json").read_text())["llama8b_lane"]
    # the caps table rides the wave manifest with its source; the gsm8k omission is recorded there like the other omissions
    assert man["caps"]["values"] == {k: v for k, v in CAPS.items() if k not in ("source", "multiturn_target_source")} and man["caps"]["source"] == CAPS["source"]
    assert man["caps"]["values"]["multiturn_target"] == MT_CAPS and man["caps"]["multiturn_target_source"] == CAPS["multiturn_target_source"]
    assert "owned by the evaluation lane" in man["capability_omitted"]["gsm8k"] and man["capability_controls"] == ["mmlu"]
    sbl = man["budget"]["S_by_leg"]
    for f in fr.MULTITURN_FAMILIES:
        L_f = sg.multiturn_window(CAPS["multiturn_target"][f], MT_WINDOWS[f]["multiturn"])
        L_rf = sg.multiturn_rerun_window(CAPS["multiturn_target"][f], MT_WINDOWS[f]["multiturn_rerun"], window=L_f)
        assert (L_f, L_rf) == (MT_GEOM[f][0], MT_GEOM[f][2])
        assert sbl[f"multiturn_attacks:{f}@{L_f}"] == 475_136 // L_f == MT_GEOM[f][1] and sbl[f"multiturn_rerun:{f}@{L_rf}"] == 475_136 // L_rf == MT_GEOM[f][3]
    assert not [k for k in sbl if k.startswith("multiturn_attacks@") or k.startswith("multiturn_rerun@")]   # the stale one-window keys are gone
    mt = man["multiturn"]["by_family"]
    assert {f: (mt[f]["cap"], mt[f]["window"], mt[f]["S"], mt[f]["rerun_window"], mt[f]["rerun_S"]) for f in mt} == {f: (MT_CAPS[f], *MT_GEOM[f]) for f in MT_GEOM}
    assert all(mt[f]["window_source"] == mt[f]["rerun_window_source"] == "pinned" for f in mt)
    assert {f: (mt[f]["derived_window"], mt[f]["derived_rerun_window"]) for f in mt} == MT_DERIVED
    assert mt["crescendo"]["pin_source"] == MT_WINDOWS["crescendo"]["source"] and "34,361" in mt["crescendo"]["pin_source"] and "ActorAttack KEPT at 4,096" in mt["actorattack"]["pin_source"]
    assert "the canonical battery of 2026-09-11" in man["multiturn"]["cap_source"] and "per family" in man["multiturn"]["window_rule"]
    # the settings' records: the families run, the decodes, the native BoN route + derived window,
    # the ladders with the 1% rule and the 16,384 rung's outcome slot, the IFEval slot (typed refusal while pending), the record-only notes
    assert man["multiturn"]["families_run"] == list(fr.MULTITURN_FAMILIES) and "the canonical battery of 2026-09-11" in man["settings_source"]
    assert man["decode"]["bon"]["declared"] == "harness_default" and "generation_config" in man["decode"]["bon"]["on_the_wire"] and "seed" in man["decode"]["bon"]["on_the_wire"]
    assert man["decode"]["multiturn_target"]["declared"] == "vendor" and man["decode"]["multiturn_target_seed"]["declared"] == "none"
    assert man["decode"]["petri"]["declared"] == "harness" and "max_tokens 1,024" in man["decode"]["petri"]["on_the_wire"] and man["decode"]["mmlu"]["declared"] == "letter_logprob"
    assert man["bon"]["cap"] == BON_CAP == CAPS["bon"] and man["bon"]["route"] == "native_4096" and "same estimand" in man["bon"]["provenance"]
    assert man["bon"]["derived_window"]["window"] == 4480 and man["bon"]["derived_window"]["S"] == 475_136 // 4480 == 106 and man["bon"]["derived_window"]["fits_lane_standup"] is True
    assert man["bon"]["standup"]["lane_views"]["window"] == 4864 and man["bon"]["standup"]["fullconst"]["window"] == 45056 and man["budget"]["S_by_leg"]["bon:native_4096@4864"] == 97
    assert man["ladders"]["static"] == [4096, 8192, 16384] and man["ladders"]["controls"] == [512, 2048, 8192, 16384] and "1%" in man["ladders"]["rule"]
    assert man["ladders"]["tree_cross_check"]["agrees"] == {"static": True, "controls": True} and "AVAILABLE" in man["ladders"]["rung_16384"]["policy"]
    assert man["ifeval"]["task"] == "pending" and "REFUSES" in man["ifeval"]["status"] and man["ifeval"]["ladder"] == [2048, 4096, 8192] and man["ifeval"]["seeds"] == 2
    assert man["ifeval"]["window_estimate"]["by_rung"]["8192"] == {"window": 8576, "S": 55, "basis": 370 + 8192}
    assert "EXCLUDED" in man["not_in_this_wave"]["math500"] and "T 1.0" in man["not_in_this_wave"]["agentic_misalignment"]
    assert "SAFE subset" in man["over_refusal_readout"] and "250" in man["over_refusal_readout"]
    assert man["replicate_floor_pair"]["pair"] == ["main", "it"]
    assert ROSTER["targets"][KEY]["replicate_floor"]["pair"] == ["main", "it"]
    assert man["views"]["fullconst"]["legs"] == list(FULLCONST_LEGS)
    assert set(man["views"]["fullconst"]["legs_omitted"]) == set(FULLCONST_OMITTED)
    assert man["budget"]["batched_token_budget"] == sg.budget_for(KEY).batched_token_budget


# --------------------------------------------------------------------------- the multi-turn target's SEED policy

def test_the_multiturn_target_sends_no_seed_when_declared_and_refuses_an_undeclared_unseeded_preset():
    """The canonical multi-turn battery: the target's replies are drawn under the generation_config preset (0.6 / 0.9) with NO
    per-request seed. jailbreak_run's `--multiturn-target-seed none` declares it: the served target's request then carries the
    preset's keys and no `seed`; without the declaration an unseeded preset is REFUSED; the adapters pass sampling_seed None when
    seed_base is None; the seed policy is part of the store identity; the wave's sbatch scripts pass the flag from
    F22_MT_TARGET_SEED (default per_turn: every earlier run byte-identical)."""
    from sisteer.eval.attacks import served_target as stg
    from sisteer.eval.attacks.pyrit_adapters import multiturn_seed
    sys.path.insert(0, str(REPO / "scripts"))
    import jailbreak_run as jr
    assert jr.MULTITURN_TARGET_SEED_MODES == ("per_turn", "none")
    src = (REPO / "scripts/jailbreak_run.py").read_text()
    assert '"--multiturn-target-seed", choices=MULTITURN_TARGET_SEED_MODES, default="per_turn"' in src
    assert 'defense.allow_unseeded_decode = bool(mt_decode is not None and mt_decode.get("seed_base") is None)' in src
    d = object.__new__(stg.ServedChatTarget)
    d.system = None; d.virtual_model = "vm"; d.allow_unseeded_decode = False; d.decode_note = None   # every attribute _payload reads
    preset = {"temperature": 0.6, "top_p": 0.9}
    msgs = [{"role": "user", "content": "hi"}]
    with pytest.raises(stg.ServedTargetError, match="unseeded"):
        d._payload(msgs, max_new_tokens=4096, do_sample=True, temperature=0.6, sampling_seed=None, decode=preset)
    d.allow_unseeded_decode = True
    req = d._payload(msgs, max_new_tokens=4096, do_sample=True, temperature=0.6, sampling_seed=None, decode=preset)
    assert "seed" not in req and req["temperature"] == 0.6 and req["top_p"] == 0.9 and req["max_tokens"] == 4096 and "top_k" not in req
    req2 = d._payload(msgs, max_new_tokens=4096, do_sample=True, temperature=0.6, sampling_seed=7, decode=preset)
    assert req2["seed"] == 7                                                        # a seeded call is unchanged by the flag
    assert "multiturn_target_seed" in stg.TARGET_DECODE_IDENTITY_KEYS and "seed_rule" in stg.TARGET_DECODE_IDENTITY_KEYS
    seeded = {"do_sample": True, **preset, "cap": 4096, "arm": "vendor_recommended", "seed_base": 20260909, "trial": 0,
              "seed_rule": "sha256(...)", "multiturn_target_decode": "vendor", "multiturn_target_seed": "per_turn"}
    unseeded = {**seeded, "seed_base": None, "seed_rule": "none (unseeded vendor-preset draws; --multiturn-target-seed none)", "multiturn_target_seed": "none"}
    assert stg._decode_identity(seeded) != stg._decode_identity(unseeded)          # a seeded and an unseeded store never mix
    with pytest.raises(ValueError):
        multiturn_seed(None, 0, "pair", "g1", 0)                                    # the seed helper itself still refuses seed_base None: the adapters pass None instead
    mt = (REPO / "src/sisteer/eval/attacks/multiturn.py").read_text(); pa = (REPO / "src/sisteer/eval/attacks/pyrit_adapters.py").read_text()
    assert "None if seed_base is None else" in mt and "None if self._seed_fn is None else" in pa and "target_decode is None or seed_base is None" in pa
    att = (REPO / "scripts/sdfbase/full22/target_attacks.sbatch").read_text()
    assert '--multiturn-target-seed "${F22_MT_TARGET_SEED:-per_turn}"' in att


# --------------------------------------------------------------------------- the XSTest control set

def test_the_xstest_control_set_is_exactly_the_250_safe_prompts_and_zero_contrast(monkeypatch):
    """The lane's over-refusal control is the 250 SAFE XSTest prompts: the wave config selects the half, both
    controls drivers build the spec over exactly those rows (0 contrast), and the frozen historical configs keep
    the 450-row set. The selection precedes the seeded subset / item hashes / ladder inputs (the loader)."""
    import pandas as pd
    sys.path.insert(0, str(REPO / "scripts"))
    import cell_controls_run as ccr
    import served_capability_run as srun
    from sisteer.capeval.tiers import judged
    df = pd.DataFrame({"prompt": [f"safe {i}" for i in range(250)] + [f"contrast {i}" for i in range(200)],
                       "type": ["homonyms_synth"] * 250 + ["contrast_synth"] * 200})
    monkeypatch.setattr(judged, "xstest_rows", lambda pins=None, *, subset=None: judged.xstest_rows_from_frame(df, subset=subset))
    for view in ("main", "it"):
        cfg = yaml.safe_load((REPO / f"configs/sdfbase_full22_llama8b{'' if view == 'main' else '_' + view}.yaml").read_text())
        assert cfg["xstest_subset"] == "safe"
        for spec in (ccr._judged_specs(cfg, ["xstest"], {"t": 1}, None, 0, "judge")["xstest"],
                     srun._judged_specs(cfg, ["xstest"], {"t": 1}, None, 0, "judge", None, 0)["xstest"]):
            assert len(spec.rows) == 250, view
            assert sum(r["subset"] == "safe" for r in spec.rows) == 250 and sum(r["subset"] == "unsafe" for r in spec.rows) == 0
            assert not any(r["type"].startswith("contrast_") for r in spec.rows)
    hist = yaml.safe_load((REPO / "configs/sdfbase_full22_gemma4e4b.yaml").read_text())
    assert "xstest_subset" not in hist
    assert len(ccr._judged_specs(hist, ["xstest"], {"t": 1}, None, 0, "judge")["xstest"].rows) == 450
