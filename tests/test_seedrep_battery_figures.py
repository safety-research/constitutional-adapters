"""battery_figures.py: every figure is drawn from
battery_summary.json ONLY; the render records assert WHAT was drawn against the summary (admitted points, the
readable-but-not-admitted hollow markers, the MISSING seed, the absent aggregate, identical axes across a family's legs,
the at-cap states, the ratio strip, the bar verdicts), that a PARTIAL summary and a REJECTED leg carry their banner
and that the title never asserts an unaudited decode contract, the manifest lists exactly the files written, and a dry run
never touches W&B or the mirror."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
FIX = REPO / "tests" / "fixtures" / "seedrep_battery"
for p in (SEEDREP, FIX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bt = _load("battery_tables", SEEDREP / "battery_tables.py")
bf = _load("battery_figures", SEEDREP / "battery_figures.py")
fx = _load("seedrep_battery_fixture", FIX / "build_fixture.py")
DIFF = "constv2-mix-full__lora_r4_ce_minus_ctl"
MIXPSR = "constv2-mix-full__psr_ce_all"
CTLPSR = "constv2ctl2-full__psr_ce_all"
CTLLORA = "constv2ctl2-full__lora_r4_ce"
PSR = "prompted__safety_refusal"
ARDITI = "ar_c0p3__arditi_allpos"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("figs")
    fx.build(root)
    bt.run(bt.load_inputs(root), root / "tables")
    summary_path = root / "tables" / "battery_summary.json"
    summary = bf.load_summary(summary_path)
    res = bf.build_all(summary, root / "figs", summary_path=summary_path)
    return {"root": root, "summary": summary, "res": res, "figs": root / "figs"}


def test_every_figure_written_and_manifest_lists_exactly_them(built):
    m = built["res"]["manifest"]
    figs = built["figs"]
    expect_png = {f"{l}_arms.png" for l in built["summary"]["legs"]} | {f"summary_{f}.png" for f in built["summary"]["families"]} | \
                 {f"diag_{d}.png" for d in bf.DIAG_FIGURES}
    on_disk = {p.name for p in figs.glob("*.png")}
    assert on_disk == expect_png
    assert {p.name for p in figs.glob("*.pdf")} == {f"{l}_arms.pdf" for l in built["summary"]["legs"]}
    listed = {Path(v["path"]).name for v in m["figures"].values()}
    assert listed == on_disk | {p.name for p in figs.glob("*.pdf")}
    for v in m["figures"].values():
        assert Path(v["path"]).stat().st_size > 5_000 and len(v["sha256"]) == 64
    assert m["partial"] is False and m["table_status"] == "official" and m["legs_filter"] is None and m["summary_sha256"] and len(m["summary_sha256"]) == 64
    assert m["rejected_legs"] == [] and m["evidence"]["n_admissible"] == 211
    assert json.loads((figs / "figures_manifest.json").read_text())["figures"].keys() == m["figures"].keys()
    assert set(m["figures"]) >= {"pasttense/arms", "pasttense/arms_pdf", "summary/jailbreak", "diag/utilisation", "diag/at_cap", "diag/determinism", "diag/h200h"}


def test_arms_figure_draws_exactly_the_admitted_summary(built):
    s = built["summary"]
    rec = built["res"]["records"]["gsm8k/arms"]
    rows = {(r["arm"], r["seed"]): r for r in s["legs"]["gsm8k"]["rows"]}
    assert len(rec["points"]) == 26 and rec["missing"] == [[DIFF, 3]] and rec["not_admitted"] == []
    for arm, seed, v in rec["points"]:
        assert v == rows[(arm, seed)]["value"] and rows[(arm, seed)]["admissible"]
    assert DIFF not in rec["aggregates"] and rec["no_aggregate"] == {DIFF: "2/3"}       # no tick over the survivors
    assert set(rec["aggregates"]) == {a for a in s["arm_order"] if a != DIFF}
    for arm, agg in rec["aggregates"].items():
        a = s["legs"]["gsm8k"]["arms"][arm]
        assert agg == {"mean": a["mean"], "min": a["min"], "max": a["max"]}
    assert rec["at_cap_strip"]["none"] == {"state": "measured", "text": "≤0.4 %"}
    assert rec["banner"] is None and rec["provisional"] is False
    # a readable value that was NOT admitted is drawn as its own hollow state — never as a point, never in an aggregate
    p = built["res"]["records"]["pasttense/arms"]
    assert len(p["points"]) == 26 and p["not_admitted"] == [[PSR, 2, 9.0, "ladder: MISSING (no at-cap row)"]] and p["missing"] == []
    assert PSR not in p["aggregates"] and p["no_aggregate"] == {PSR: "2/3"}
    assert p["at_cap_strip"][PSR] == {"state": "partial", "text": "≤0.0 % (2/3)"} and p["at_cap_strip"][MIXPSR]["text"] == "≤0.8 %"
    x = built["res"]["records"]["xstest/arms"]
    assert x["not_admitted"][0][:3] == [CTLLORA, 2, pytest.approx(5.2)] and x["not_admitted"][0][3].startswith("coverage: missing(audit REFUSED")
    b = built["res"]["records"]["bon/arms"]
    assert all(v["state"] == "measured" for v in b["at_cap_strip"].values()) and len(b["points"]) == 26
    assert b["not_admitted"] == [[CTLLORA, 1, 80.0, "audit: unaudited"]]
    mis = built["res"]["records"]["misalign_leaking/arms"]
    assert mis["at_cap_strip"][CTLPSR] == {"state": "not_final", "text": "NOT FINAL"} and mis["not_admitted"][0][:2] == [CTLPSR, 2]
    assert all(v["state"] == "not_applicable" and v["text"] == "" for v in built["res"]["records"]["mmlu/arms"]["at_cap_strip"].values())
    c = built["res"]["records"]["codeattack/arms"]
    assert c["points"] == [] and len(c["missing"]) == 27 and c["no_aggregate"] == {a: "0/3" for a in s["arm_order"]}
    assert c["leg_status"].startswith("unverified: engine identity MISSING")


def test_title_asserts_the_decode_contract_only_when_every_arm_run_agrees(built):
    r = built["res"]["records"]
    # fg2: every non-GSM8K leg is reconciled ledger-less -- the title states the qualification beside the AGREES, never a bare one
    assert r["crescendo/arms"]["evidence"] == ("27/27 arm-runs admitted (27 readable) · vendor preset, native serving (audit AGREES on every arm-run; "
                                               f"D106: 27 ledger-less under {bf.LEDGERLESS_CONTRACT}, body digest not available) · engine identity PASS")
    assert r["bon/arms"]["evidence"] == ("26/27 arm-runs admitted (27 readable) · decode contract UNVERIFIED on 1/27 arm-runs (audit AGREES 26/27; "
                                         f"D106: 26 ledger-less under {bf.LEDGERLESS_CONTRACT}) · engine identity PASS")
    assert r["codeattack/arms"]["evidence"].startswith("0/27 arm-runs admitted (0 readable) · decode contract UNVERIFIED on 27/27 arm-runs")
    assert "vendor preset" not in r["bon/arms"]["evidence"] and "vendor preset" not in r["codeattack/arms"]["evidence"]


def test_axes_identical_across_the_legs_of_a_family_and_fixed_for_petri(built):
    recs = built["res"]["records"]
    s = built["summary"]
    for fam, body in s["families"].items():
        yl = {tuple(recs[f"{l}/arms"]["ylim"]) for l in body["legs"]}
        assert len(yl) == 1, (fam, yl)
    assert recs["petri/arms"]["ylim"] == [1.0, 10.0]
    # the family range is a DISPLAY range over every READABLE value of the family: jailbreak includes BoN at 90 pp (+ SE 3.0, × 1.08 → 100)
    assert recs["pasttense/arms"]["ylim"] == [0.0, 100.0] and recs["bon/arms"]["ylim"] == [0.0, 100.0]
    assert bf.family_axis_range(s, "over_refusal") == (0.0, 40.0)   # xstest max 28 pp + SE 2.84 → 33.3 → 40
    assert bf.family_axis_range(s, "misalign") == (0.0, 30.0)       # 22 pp + SE 1.85 → 25.8 → 30
    assert bf.family_axis_range(s, "capability") == (0.0, 100.0)    # 89 pp + SE → capped at 100


def test_summary_heatmap_cells_are_the_admitted_means_and_missing_is_hatched(built):
    rec = built["res"]["records"]["summary/capability"]
    s = built["summary"]
    assert rec["columns"] == ["mmlu", "gsm8k"] and rec["range"] == [0.0, 100.0]
    assert rec["cells"][f"gsm8k|{DIFF}"] == {"value": None, "why": "2/3 admitted"}
    assert rec["ratio_cells"][f"gsm8k|{DIFF}"] == {"value": None, "why": "2/3 admitted"}
    assert rec["cells"]["gsm8k|none"]["value"] == s["legs"]["gsm8k"]["arms"]["none"]["mean"] == 88.5
    assert rec["cells"][f"gsm8k|{ARDITI}"]["value"] == 82.5                               # the Arditi VALUE is drawn (its contrast is withheld elsewhere)
    assert rec["ratio_cells"]["mmlu|none"]["value"] == s["legs"]["mmlu"]["arms"]["none"]["seed_spread"]["ratio"]
    pe = built["res"]["records"]["summary/petri"]
    assert pe["columns"][0] == "petri" and set(pe["columns"][1:]) == set(fx.PETRI_BEHAVIOURS)
    assert pe["ratio_cells"][f"petri|{PSR}"] == {"value": None, "why": "no SE"}
    assert pe["cells"][f"petri|{PSR}"]["value"] == pytest.approx(3.75)   # the mean is drawn; only the ratio is n/a
    jb = built["res"]["records"]["summary/jailbreak"]
    assert all(c["value"] is None and c["why"] == "0/3 admitted" for k, c in jb["cells"].items() if k.startswith("codeattack|"))
    assert jb["cells"][f"pasttense|{PSR}"] == {"value": None, "why": "2/3 admitted"}      # the not-admitted seed voids the triplet
    assert jb["banner"] is None and jb["rejected_legs"] == []


def test_diagnostics_records_use_ws3g_verdicts(built):
    u = built["res"]["records"]["diag/utilisation"]
    assert u["n_standups"] == 40 and u["n_with_reading"] == 40 and u["unmeasured"] == []
    assert u["panels"] == ["running_ratio", "median_sm_decode", "tok_s_ratio_s5", "h200_h_ratio"]       # the four bars, incl. hours
    assert u["below_bar"] == [["att_crescendo_arditi_base_shard00", "running_ratio", pytest.approx(0.6)], ["att_crescendo_arditi_base_shard00", "median_sm_decode", 70.0]]
    assert [p for p in bf.UTILISATION_PANELS if p[0] == "tok_s_ratio_s5"][0][2].startswith("AGGREGATE decode tok/s") and bf.UTILISATION_PANELS[3][1] == "hours_bar"
    a = built["res"]["records"]["diag/at_cap"]
    assert "mmlu" not in a["legs"] and set(a["legs"]) >= {"pasttense", "xstest", "gsm8k", "petri", "bon", "crescendo", "misalign_leaking", "codeattack"}
    assert a["legs"]["pasttense"] == {"measured": 26, "missing": 1, "not_final": 0, "n": 27, "max_share": 0.008}
    assert a["legs"]["misalign_leaking"] == {"measured": 26, "missing": 0, "not_final": 1, "n": 27, "max_share": 0.008}
    assert a["legs"]["bon"] == {"measured": 27, "missing": 0, "not_final": 0, "n": 27, "max_share": 0.05}
    assert a["legs"]["codeattack"] == {"measured": 0, "missing": 27, "not_final": 0, "n": 27, "max_share": None}
    d = built["res"]["records"]["diag/determinism"]
    assert d["legs"]["pasttense"] == {"n": 20, "identical": 20} and d["legs"]["gsm8k"] == {"n": 20, "identical": 17} and d["legs"]["bon"] is None
    h = built["res"]["records"]["diag/h200h"]
    hs = built["summary"]["h200_hours"]
    assert h["legs"]["pasttense"] == {"actual": hs["pasttense"]["actual"], "expected": hs["pasttense"]["expected"], "complete": True, "status": "ok", "n_standups": 7, "ratio": hs["pasttense"]["ratio"]}
    assert h["legs"]["pasttense"]["actual"] == pytest.approx(8.4) and h["legs"]["controls"]["actual"] == pytest.approx(8.4)   # one row for the shared standup
    assert h["legs"]["bon"]["n_standups"] == 4 and "xstest" not in h["legs"]
    assert h["legs"]["codeattack"] == {"actual": None, "expected": None, "complete": False, "status": "MISSING (no accepted attempt (no identity row for the container leg))",
                                       "n_standups": 0, "n_with_actual": 0}


def test_unmet_bar_verdict_is_drawn_from_ws3g_not_recomputed(built):
    """A reading above the numeric bar with an input verdict of `false` (or `unmeasured(...)`) is drawn as below / unmeasured —
    the figure never re-judges the bar."""
    s = json.loads(json.dumps(built["summary"]))
    st = next(x for x in s["standups"] if x["attempt_id"] == "att_pasttense_s1_base_shard00")
    st["utilisation"]["running_bar"] = "False"                                  # 0.9 ≥ 0.8 numerically, but the verdict says no
    st2 = next(x for x in s["standups"] if x["attempt_id"] == "att_pasttense_s2_base_shard00")
    st2["utilisation"]["running_bar"] = "unmeasured(in-flight never reached S)"
    st3 = next(x for x in s["standups"] if x["attempt_id"] == "att_pasttense_s3_base_shard00")
    st3["utilisation"]["hours_bar"] = "False"                                   # 0.8 ≤ 1.5 numerically, but the writer's hours verdict says no
    rec = bf.diag_utilisation_figure(s, built["root"] / "figs_bar")
    assert ["att_pasttense_s1_base_shard00", "running_ratio", pytest.approx(0.9)] in rec["below_bar"]
    assert ["att_pasttense_s3_base_shard00", "h200_h_ratio", pytest.approx(0.8)] in rec["below_bar"]
    assert rec["unmeasured"] == ["att_pasttense_s2_base_shard00"]


def test_partial_summary_puts_the_banner_on_every_figure(tmp_path):
    root = fx.build(tmp_path / "partial", ws3=False)
    bt.run(bt.load_inputs(root), root / "tables", partial=True, legs=(bt.LEG_BY_ID["pasttense"], bt.LEG_BY_ID["bon"]))
    summary = bf.load_summary(root / "tables/battery_summary.json")
    res = bf.build_all(summary, root / "figs", summary_path=root / "tables/battery_summary.json")
    m = res["manifest"]
    assert m["partial"] is True and m["table_status"].startswith("PARTIAL (absent inputs: coverage.csv")
    for key, rec in res["records"].items():
        assert rec["provisional"] is True, key
        assert rec["banner"].startswith("PARTIAL — PROVISIONAL: absent inputs: coverage.csv, at_cap.csv, engine_identity.csv, engine_identity_legs.json, "
                                        "utilisation.csv, request_log_summary.csv; 0/54 arm-runs admitted"), key
    arms = res["records"]["pasttense/arms"]
    assert arms["points"] == [] and len(arms["not_admitted"]) == 27 and arms["aggregates"] == {}      # nothing official on a provisional figure
    assert arms["not_admitted"][0][3] == ("coverage: MISSING (no coverage.csv); audit: MISSING (no request_log_summary.csv); identity: MISSING "
                                          "(no engine_identity_legs.json); ladder: MISSING (no at_cap.csv); standup: MISSING (no engine_identity.csv); "
                                          "engine: MISSING (no engine_identity.csv)")
    # P2-3: a provisional manifest publishes under its OWN namespace, never the official one
    pub = m["publication"]
    assert pub["official"] is False and pub["evidence_hash"] and len(pub["evidence_hash"]) == 12
    assert pub["run_name"] == f"seedrep-native-battery-PARTIAL-{pub['evidence_hash']}" and pub["artifact"] == f"figures-seedrep-native-battery-PARTIAL-{pub['evidence_hash']}"
    assert Path(pub["mirror_dest"]).name == f"qwen-seedrep-native-battery-PARTIAL-{pub['evidence_hash']}" and Path(pub["mirror_dest"]).parent == bf.DEFAULT_MIRROR.parent
    assert arms["evidence"].startswith("0/27 arm-runs admitted (27 readable) · decode contract UNVERIFIED on 27/27 arm-runs")
    assert all(v["state"] == "missing" for v in arms["at_cap_strip"].values())
    assert res["records"]["summary/jailbreak"]["cells"]["pasttense|none"] == {"value": None, "why": "0/3 admitted"}
    assert Path(m["figures"]["pasttense/arms"]["path"]).stat().st_size > 20_000


def test_rejected_leg_carries_its_banner(tmp_path):
    root = fx.build(tmp_path / "rej")
    legs = fx.engine_identity_legs()
    legs["pasttense"] = {**legs["pasttense"], "verdict": "FAIL", "problems": ["Arditi absent"]}
    (root / "tables/engine_identity_legs.json").write_text(json.dumps(legs))
    bt.run(bt.load_inputs(root), root / "tables", legs=(bt.LEG_BY_ID["pasttense"], bt.LEG_BY_ID["bon"]))
    summary = bf.load_summary(root / "tables/battery_summary.json")
    res = bf.build_all(summary, root / "figs", summary_path=root / "tables/battery_summary.json")
    rec = res["records"]["pasttense/arms"]
    assert rec["banner"] == "LEG REJECTED — engine identity FAIL — Arditi absent" and rec["provisional"] is False
    assert rec["points"] == [] and len(rec["not_admitted"]) == 27 and rec["aggregates"] == {}
    assert res["records"]["bon/arms"]["banner"] is None
    jb = res["records"]["summary/jailbreak"]
    assert jb["rejected_legs"] == ["pasttense"] and jb["banner"] == "LEG(S) REJECTED by the engine-identity verdict: pasttense"
    assert res["manifest"]["rejected_legs"] == ["pasttense"]


def test_legs_filter_changes_panels_not_values(built):
    s = built["summary"]
    res = bf.build_all(s, built["root"] / "figs_filtered", legs=["pasttense", "gsm8k"], summary_path=built["root"] / "tables" / "battery_summary.json")
    m = res["manifest"]
    assert m["legs"] == ["pasttense", "gsm8k"] and m["legs_filter"] == ["pasttense", "gsm8k"]
    assert {k for k in m["figures"] if k.endswith("/arms")} == {"pasttense/arms", "gsm8k/arms"}
    assert {k for k in m["figures"] if k.startswith("summary/")} == {"summary/jailbreak", "summary/capability"}
    # the values drawn are the summary's, identical to the unfiltered run; the axis range is the FAMILY's, not the shown subset's
    assert res["records"]["pasttense/arms"]["aggregates"] == built["res"]["records"]["pasttense/arms"]["aggregates"]
    assert res["records"]["pasttense/arms"]["ylim"] == built["res"]["records"]["pasttense/arms"]["ylim"]
    assert res["records"]["summary/jailbreak"]["cells"]["pasttense|none"] == built["res"]["records"]["summary/jailbreak"]["cells"]["pasttense|none"]
    assert res["records"]["diag/utilisation"]["n_standups"] == 7 + 7      # the two container legs' standups only


def test_dry_run_cli_never_imports_wandb_or_rsyncs(built, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(bf, "mirror", lambda *a, **k: calls.append("mirror"))
    monkeypatch.setattr(bf, "log_wandb", lambda *a, **k: calls.append("wandb"))
    out = built["root"] / "figs_cli"
    base = ["--summary", str(built["root"] / "tables" / "battery_summary.json"), "--out", str(out), "--legs", "pasttense"]
    rc = bf.main(base)
    assert rc == 0 and calls == [] and "dry run" in capsys.readouterr().out
    rc = bf.main(base + ["--wandb", "--mirror", "--dry-run"])
    assert rc == 0 and calls == []
    rc = bf.main(base + ["--mirror", "--mirror-dest", str(built["root"] / "mirror")])
    assert rc == 0 and calls == ["mirror"]
    m = json.loads((out / "figures_manifest.json").read_text())
    assert m["legs"] == ["pasttense"] and {k for k in m["figures"] if k.startswith("summary/")} == {"summary/jailbreak"}


def test_log_wandb_refuses_without_a_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY is not in the environment"):
        bf.log_wandb({"figures": {}, "legs": [], "partial": False, "summary_sha256": None}, None)


def test_mirror_copies_exactly_the_manifest_files(built, tmp_path):
    if shutil.which("rsync") is None:
        pytest.skip("rsync not available on this host (battery_figures.mirror shells out to it)")   # main's test_sampled_grading idiom
    m = built["res"]["manifest"]
    (built["figs"] / "stale_extra.png").write_bytes(b"not in the manifest")
    try:
        bf.mirror(m, built["figs"], tmp_path / "dest")
    finally:
        (built["figs"] / "stale_extra.png").unlink()
    got = {p.name for p in (tmp_path / "dest").iterdir()}
    assert "stale_extra.png" not in got and "figures_manifest.json" in got
    assert got == {Path(v["path"]).name for v in m["figures"].values()} | {"figures_manifest.json"}


# ---------------------------------------------------------------- a provisional publication never reuses the official namespace
def test_partial_publication_uses_a_distinct_namespace_and_the_official_one_is_refused(built, tmp_path, monkeypatch):
    if shutil.which("rsync") is None:
        pytest.skip("rsync not available on this host (battery_figures.mirror shells out to it)")   # main's test_sampled_grading idiom
    official = built["res"]["manifest"]
    names = bf.publication_names(official)
    assert names == {"partial": False, "evidence_hash": None, "run_name": "seedrep-native-battery", "artifact": "figures-seedrep-native-battery",
                     "mirror_dest": bf.DEFAULT_MIRROR, "official_mirror": bf.DEFAULT_MIRROR, "official_artifact": "figures-seedrep-native-battery",
                     "official_run_name": "seedrep-native-battery"}
    assert official["publication"] == {"run_name": "seedrep-native-battery", "artifact": "figures-seedrep-native-battery", "mirror_dest": str(bf.DEFAULT_MIRROR),
                                       "evidence_hash": None, "official": True}
    # a provisional manifest (absent input tables) → every name carries PARTIAL + the evidence hash; the hash is a pure function of the evidence
    root = fx.build(tmp_path / "p", ws3=False)
    bt.run(bt.load_inputs(root), root / "tables", partial=True, legs=(bt.LEG_BY_ID["pasttense"],))
    s = bf.load_summary(root / "tables/battery_summary.json")
    m = bf.build_all(s, root / "figs", summary_path=root / "tables/battery_summary.json")["manifest"]
    n = bf.publication_names(m)
    h = bf.evidence_hash(m)
    assert n["partial"] is True and n["evidence_hash"] == h and len(h) == 12 and int(h, 16) >= 0
    assert n["run_name"] == f"seedrep-native-battery-PARTIAL-{h}" and n["artifact"] == f"figures-seedrep-native-battery-PARTIAL-{h}"
    assert n["mirror_dest"] == bf.DEFAULT_MIRROR.with_name(f"qwen-seedrep-native-battery-PARTIAL-{h}")
    assert bf.evidence_hash(m) == bf.evidence_hash(json.loads(json.dumps(m)))                         # deterministic
    # a DIFFERENT evidence state (a pool-provenance gap instead of absent inputs) → a different namespace
    root2 = fx.build(tmp_path / "p2")
    idx = root2 / "grades/seedrep_final_rung_static.json"
    d = json.loads(idx.read_text())
    d["tags"]["qwen35_4b_s1"]["static"]["pasttense|none"].update(judge_pools={"lowprio": 498, "None": 2}, n_items_without_pool_entry=2)
    idx.write_text(json.dumps(d, sort_keys=True))
    bt.run(bt.load_inputs(root2), root2 / "tables", legs=(bt.LEG_BY_ID["pasttense"],))
    s2 = bf.load_summary(root2 / "tables/battery_summary.json")
    assert s2["partial"] is True and s2["provisional_reasons"] == ["judge-pool provenance missing on 1 judged arm-run(s): pasttense ×1"]
    m2 = bf.build_all(s2, root2 / "figs", summary_path=root2 / "tables/battery_summary.json")["manifest"]
    assert bf.evidence_hash(m2) != h and m2["publication"]["run_name"] == f"seedrep-native-battery-PARTIAL-{bf.evidence_hash(m2)}"
    assert m2["publication"]["official"] is False
    # the banner names the provenance gap (not an absent input)
    rec = bf.arms_figure(s2, "pasttense", root2 / "figs_b")
    assert rec["banner"].startswith("PARTIAL — PROVISIONAL: judge-pool provenance missing on 1 judged arm-run(s): pasttense ×1; 26/27 arm-runs admitted")
    # mirror: the default destination is the derived one; the OFFICIAL directory is refused for a provisional manifest (and only for one)
    monkeypatch.setattr(bf, "DEFAULT_MIRROR", tmp_path / "figs_root" / "qwen-seedrep-native-battery")      # never the real mirror root
    dest_default = bf.publication_names(m)["mirror_dest"]
    assert dest_default == tmp_path / "figs_root" / f"qwen-seedrep-native-battery-PARTIAL-{h}"
    with pytest.raises(RuntimeError, match="a PARTIAL \\(provisional\\) output may not be published under the official namespace: mirror"):
        bf.mirror(m, root / "figs", bf.DEFAULT_MIRROR)
    assert not bf.DEFAULT_MIRROR.exists()                                                              # nothing was written there
    bf.mirror(m, root / "figs", dest_default)
    assert (dest_default / "figures_manifest.json").is_file() and (dest_default / "pasttense_arms.png").is_file()
    bf.mirror(official, built["figs"], bf.DEFAULT_MIRROR)                                            # the official manifest may use the official directory
    assert (bf.DEFAULT_MIRROR / "figures_manifest.json").is_file()
    # W&B: an explicit official run / artifact name for a provisional manifest is refused BEFORE anything is logged
    monkeypatch.setenv("WANDB_MODE", "offline")
    with pytest.raises(RuntimeError, match="official namespace: artifact 'figures-seedrep-native-battery'"):
        bf.log_wandb(m, None, artifact="figures-seedrep-native-battery")
    with pytest.raises(RuntimeError, match="official namespace: run 'seedrep-native-battery'"):
        bf.log_wandb(m, None, name="seedrep-native-battery")
    bf.refuse_official_namespace(official, mirror_dest=bf.DEFAULT_MIRROR, artifact="figures-seedrep-native-battery", run_name="seedrep-native-battery")   # official: allowed
    # the CLI: --mirror without --mirror-dest goes to the derived directory; an explicit official directory exits 2
    calls = []
    monkeypatch.setattr(bf, "mirror", lambda man, out, dest=None: calls.append(Path(dest)))
    rc = bf.main(["--summary", str(root / "tables/battery_summary.json"), "--out", str(root / "figs_cli"), "--legs", "pasttense", "--mirror"])
    assert rc == 0 and calls == [Path(bf.publication_names(m)["mirror_dest"])]



# ---------------------------------------------------------------- the H200-hour renderer draws the tables' totals and never reduces rows
def test_h200h_renderer_draws_the_summarys_totals_and_a_withheld_total_stays_withheld(built, tmp_path):
    s = json.loads(json.dumps(built["summary"]))
    # the renderer reads h200_hours ONLY: standups with hours but a withheld leg total → hollow + the status, never a sum of the rest
    s["h200_hours"]["pasttense"].update(complete=False, actual=None, expected=None, ratio=None, n_with_actual=6,
                                         missing_actual=["att_pasttense_s2_base_shard00"],
                                         status="MISSING (1 of 7 standup(s) without H200-hours: att_pasttense_s2_base_shard00)")
    rec = bf.diag_h200h_figure(s, tmp_path / "h")
    assert rec["legs"]["pasttense"] == {"actual": None, "expected": None, "complete": False, "n_standups": 7, "n_with_actual": 6,
                                        "status": "MISSING (1 of 7 standup(s) without H200-hours: att_pasttense_s2_base_shard00)"}
    assert rec["legs"]["bon"]["complete"] is True and rec["legs"]["bon"]["actual"] == s["h200_hours"]["bon"]["actual"]
    # a summary without the field at all → every leg withheld with a MISSING status (never recomputed from standups)
    s.pop("h200_hours")
    rec = bf.diag_h200h_figure(s, tmp_path / "h2")
    assert all(v["complete"] is False and v["status"] == "MISSING (no h200_hours record)" for v in rec["legs"].values())
    # end to end: one utilisation row without hours → the figure's record for that leg is the tables' withheld total
    root = fx.build(tmp_path / "e2e")
    fx._write_csv(root / "tables/utilisation.csv", [dict(r, h200_h_actual="unmeasured") if r["attempt_id"] == "att_bon_s1_base_shard00" else r for r in fx.utilisation_rows()])
    bt.run(bt.load_inputs(root), root / "tables", legs=(bt.LEG_BY_ID["bon"], bt.LEG_BY_ID["pasttense"]))
    summ = bf.load_summary(root / "tables/battery_summary.json")
    rec = bf.diag_h200h_figure(summ, tmp_path / "h3")
    assert rec["legs"]["bon"]["complete"] is False and rec["legs"]["bon"]["status"] == "MISSING (1 of 4 standup(s) without H200-hours: att_bon_s1_base_shard00)"
    assert rec["legs"]["pasttense"]["complete"] is True and rec["legs"]["pasttense"]["actual"] == pytest.approx(8.4)
