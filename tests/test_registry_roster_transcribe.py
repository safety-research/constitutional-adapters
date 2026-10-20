"""`registry roster transcribe` against a SYNTHETIC mini paperlib: the tool reads the pinned config in a
fresh subprocess, refuses a checkout that is not at the named sha, regenerates a roster whose expansion is the
declared cross product, is byte-for-byte repeatable, carries the previous version's disagreement prose so the
disagreements are listed first, and keeps written rulings the config cannot see.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import yaml

from sisteer.registry import roster_transcribe as T
from sisteer.registry.adapters import roster as RO

MINI_CONFIG = textwrap.dedent('''
    """A synthetic paperlib.config: the symbols the transcription reads, invented values."""
    from dataclasses import dataclass
    HOME = "/synthetic"
    R2_PARQUET = HOME + "/r2/results.parquet"
    R2_SOURCES = HOME + "/r2/sources.json"
    R2_TRANSFER_ITEMS = HOME + "/r2/items.parquet"
    R2_TRANSFER_STATS = HOME + "/r2/stats.csv"
    MODELS = {
        "gemma4e4b": dict(label="G", served="synthetic/gemma-it", lane_kind="instruct-direct", view="gemma4e4b_it", transfer_view="gemma4e4b"),
        "nemotron3nano4b": dict(label="N", served="synthetic/nemotron", lane_kind="chat-ckpt", view="nemotron3nano4b", transfer_view=None),
    }
    MODEL_ORDER = list(MODELS)
    PAPER_LANE = {m: d["lane_kind"] for m, d in MODELS.items()}
    INSTRUCT_DIRECT_VIEW = {"gemma4e4b": "gemma4e4b_it"}
    TRANSFER_VIEW = {"gemma4e4b": "gemma4e4b"}
    @dataclass(frozen=True)
    class Arm:
        key: str; label: str; family: str; object_type: str; cell_id: object; owner: str; in_fig2: bool = True; note: str = ""
    ARMS = [Arm("none", "Unsteered", "none", "none", "none", "S"), Arm("si_safety", "Safety SI", "prompted", "prompt", "prompted__safety_refusal", "S"),
            Arm("arditi", "Arditi", "baseline", "vector", None, "A"), Arm("alphasteer_rebake", "AlphaSteer", "baseline", "matrix", None, "A"),
            Arm("ctl_lora", "Control LoRA", "control", "lora", "ctl__lora", "S"), Arm("const_lora", "Constitution LoRA", "constitution", "lora", "const__lora", "S"),
            Arm("diff_lora", "Difference LoRA", "difference", "lora", "diff__lora", "S"), Arm("ctl_psr", "Control ReLU", "control", "vector", "ctl__psr", "S", in_fig2=False)]
    FIG2_ARMS = ["none", "si_safety", "arditi", "alphasteer_rebake", "ctl_lora", "const_lora", "diff_lora"]
    FIG4_ARMS = ["ctl_lora", "const_lora", "diff_lora"]
    FRONTIER_ARMS = ["diff_lora", "const_lora", "ctl_lora", "alphasteer_rebake", "arditi"]
    SI_POINT_ARMS = ["si_safety"]
    NEVER_DRAWN_ARMS = ("ctl_psr",)
    PSR_ARMS = ["ctl_psr"]
    FIG_ARM_ORDER = ["si_safety", "arditi", "alphasteer_rebake", "ctl_lora", "const_lora", "diff_lora"]
    ARM_ORDER_ALL = FIG_ARM_ORDER
    EVALS = {"static": ["pasttense", "codeattack"], "bon": ["bon"], "multi-turn": ["pair"], "over-refusal": ["xstest"], "capability": ["mmlu", "gsm8k"],
             "petri": ["petri_pooled"], "misalignment": ["agentic_murder"], "em_organism": ["em_risky_financial"]}
    EVAL_COMPONENT = {e: ("misalignment" if comp == "em_organism" else comp) for comp, es in EVALS.items() for e in es}
    EVAL_LABEL = {e: e for es in EVALS.values() for e in es}
    JAILBREAK_EVALS = EVALS["static"] + EVALS["bon"] + EVALS["multi-turn"]
    CATEGORIES = {"single-turn + best-of-N": dict(evals=EVALS["static"] + EVALS["bon"]), "multi-turn": dict(evals=EVALS["multi-turn"]),
                  "over-refusal": dict(evals=EVALS["over-refusal"]), "Petri": dict(evals=["petri_pooled"]), "agentic misalignment": dict(evals=EVALS["misalignment"]),
                  "capability": dict(evals=EVALS["capability"], mean_over=["mmlu", "gsm8k"])}
    FIG2_CATEGORIES = ["single-turn + best-of-N", "multi-turn", "over-refusal"]
    FIG3_CATEGORIES = ["Petri", "agentic misalignment", "capability"]
    FIG4_CATEGORIES = ["single-turn + best-of-N", "multi-turn", "over-refusal", "Petri", "agentic misalignment", "capability"]
    FIG4_ORGANISM = dict(evals=["em_risky_financial"], dropped=[])
    HEADLINE = {"static": dict(metric="asr", threshold=0.5, bon_n=None), "bon": dict(metric="asr", threshold=0.5, bon_n=75), "multi-turn": dict(metric="asr", threshold=0.5, bon_n=None),
                "over-refusal": dict(metric="over_refusal_graded", threshold=None, bon_n=None), "capability": dict(metric="acc", threshold=None, bon_n=None),
                "petri": dict(metric="petri_score", threshold=None, bon_n=None), "misalignment": dict(metric="misaligned_rate", threshold=None, bon_n=None)}
    FIGURES = {"fig1_headline": dict(arms=FIG2_ARMS, evals=JAILBREAK_EVALS + ["mmlu", "gsm8k"], doses=[1.0]),
               "fig2_jailbreak": dict(arms=FIG2_ARMS, evals=JAILBREAK_EVALS + EVALS["over-refusal"], doses=[1.0]),
               "fig3_misalign": dict(arms=FIG2_ARMS, evals=EVALS["petri"] + EVALS["misalignment"] + EVALS["capability"], doses=[1.0]),
               "fig4_transfer": dict(arms=FIG4_ARMS, evals=JAILBREAK_EVALS + EVALS["over-refusal"] + ["mmlu", "gsm8k", "petri_pooled"], doses=[1.0],
                                     models=["gemma4e4b"], lanes=["base→instruct", "instruct-direct"]),
               "fig5_frontier": dict(arms=FRONTIER_ARMS, evals=JAILBREAK_EVALS + EVALS["over-refusal"] + ["mmlu", "gsm8k"], doses=[0.5, 1.0])}
    FIGQ_FIGURES = {}
    FIGQ_MODEL = "qwen35_4b"
    FIGQ_MODELS = []
    FIGQ_ARMS = []
    FIGQ_FRONTIER_ARMS = []
    FIGQ_FIG4_ARMS = []
    FIGQ_PANEL_COLUMNS = {}
    CANONICAL = {"static": dict(decode_contains="sampled", caps=(4096, 8192), rung="max", why="sampled"), "bon": dict(decode_contains="sampled", caps=(512, 4096), rung="max", why="climb"),
                 "multi-turn": dict(decode_contains="sampled", caps=(4096,), rung=None, why="4096"), "over-refusal": dict(decode_contains="sampled", caps=(512, 2048), rung="max", why="ladder"),
                 "petri": dict(decode_contains=None, caps=None, rung=None, why="as run"), "misalignment": dict(decode_contains=None, caps=None, rung=None, why="as run"),
                 "capability": dict(decode_contains=None, caps=None, rung=None, per_eval={"gsm8k": dict(decode_contains="thinking", caps=(8192,), rung="max", why="paired")}, why="per eval")}
    SUPERSEDED_SERVED_BASELINE_CELLS = [
        dict(model="gemma4e4b", arm="alphasteer_rebake", source="em/agentic/gemma4e4b_bl__lam3-superseded", components=("misalignment",),
             served=dict(rung=3, lambda_=0.2, object_sha256="7" * 64), input=dict(path=HOME + "/em/summary.csv", sha256_prefix="4083cf8c"), evidence="served lam3", owner="EM", fix="re-pin"),
        dict(model="gemma4e4b", arm="arditi", source="r2_export", components=("multi-turn",), served_id="ar_r0p1__arditi", served=dict(label="ar_r0p1", value=0.1, object_sha256="0" * 64),
             expires_when=dict(lane_file=HOME + "/run-state/transfer_stats.json", json_path="models.gemma4e4b_it.cells.arditi", root="grades_old"), evidence="served 0.1", owner="Scale", fix="re-run"),
    ]
    EXCLUDED_MEASUREMENTS = [dict(model="gemma4e4b", arm="diff_lora", dose=1.5, eval_id="xstest", source=None, ruling="D208a", note="dropped whole")]
    SECOND_PASS_ONLY = {("strength_lane", 1.0): "the ladder's re-serve"}
    SECOND_PASS_ROWS = []
    R2_VIEW_TO_PAPER_LANE = {}
    DECLINED = [dict(what="constitution-SI multi-turn wave", since="2026-09-13", cost="x", consequence="no multi-turn cell for the constitution SI")]
    NOT_RUN_BY_DESIGN = [dict(models=["nemotron3nano4b"], evals=["agentic_murder"], owner="EM", ruling="D2", note="no headroom")]
    DROPPED_EVALS = ("tiny_ll",)
    ALPHASTEER_OPERATING_RUNG = {"gemma4e4b": 2, "nemotron3nano4b": 2}
    ALPHASTEER_OPERATING_RUNG_SUPERSEDED = {"gemma4e4b": 1}
    ARDITI_OPERATING_ALPHA_OVERRIDE = {}
    @dataclass(frozen=True)
    class StrengthRecord:
        path: str; source: str; arms: tuple; lane: object = None
    STRENGTH_RECORDS = [StrengthRecord(HOME + "/strength/lora/battery_curves.json", "strength_lane", ("const_lora", "diff_lora")),
                        StrengthRecord(HOME + "/strength/lora_b2i/battery_curves.json", "strength_lane_base2it_native", ("const_lora", "ctl_lora", "diff_lora"), "base→instruct")]
    FRONTIER_SOURCE = {("static", 1.0): ["r2_export"]}
    FRONTIER_SOURCE_MODEL_OVERRIDE = {("gemma4e4b", "instruct-direct"): ["strength_lane"]}
    FRONTIER_SOURCE_DEFAULT = ["strength_lane", "strength_lane_base2it_native"]
    BASELINE_LANE_SOURCES = ["arditi_strength", "alphasteer_rebake"]
    CAMPAIGN_SOURCES = ["r2_export"]
    BATCH64_MIXING_RULING = "batch-8 constitution + batch-64 control/difference"
    FRONTIER_CONST_LORA_LINEAGE_CAVEAT = "batch-64 constitution on the frontier"
    DECODE_MATCHING_RULING = "top_p 1.0 where it exists"
    BATCH64_PAIR_ASYMMETRY = {"gemma4e4b_it_doc": dict(constitution=dict(sha256="8" * 64))}
    LADDER_1X_IS_THE_REFIT_PARENT = ("gemma4e4b",)
    PETRI_NOT_IN_ROSTER = set()
    DESCOPED_LABELS = {}
    BASELINE_CAPABILITY_SUPERSEDED_BY_RUNG_CHANGE = set()
''')

MINI_FIGS = "FRONTIER_DEFENCE_OUT = ('bon', 'multi-turn')\nFRONTIER_DEFENCE_EVALS = ['pasttense', 'codeattack']\n"
MINI_LOAD = "COLUMNS = ['source', 'model', 'lane_kind', 'arm', 'eval_id', 'metric', 'value']\nSECONDARY = []\n"


def _git(cwd, *args):
    return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", cwd] + list(args), text=True, stderr=subprocess.STDOUT).strip()


@pytest.fixture
def checkout(tmp_path):
    co = tmp_path / "paper-figures-pin-synthetic"
    lib = co / "paper" / "paperlib"
    lib.mkdir(parents=True)
    (lib / "__init__.py").write_text("")
    (lib / "config.py").write_text(MINI_CONFIG)
    (lib / "figs.py").write_text(MINI_FIGS)
    (lib / "load.py").write_text(MINI_LOAD)
    _git(str(co), "init", "-q")
    _git(str(co), "add", "-A")
    _git(str(co), "commit", "-q", "-m", "synthetic pin")
    return str(co), _git(str(co), "rev-parse", "HEAD")


def test_transcribe_is_repeatable_and_expands_to_the_cross_product(checkout, tmp_path):
    co, sha = checkout
    out1 = T.transcribe(co, sha, str(tmp_path / "o1"), read_utc="2026-01-01T00:00:00.000Z")
    out2 = T.transcribe(co, sha[:8], str(tmp_path / "o2"), read_utc="2026-01-01T00:00:00.000Z")
    assert open(out1["roster"], "rb").read() == open(out2["roster"], "rb").read()
    assert open(out1["rulings"], "rb").read() == open(out2["rulings"], "rb").read()
    roster = yaml.safe_load(open(out1["roster"]))
    assert roster["meta"]["schema"] == RO.ROSTER_SCHEMA and roster["meta"]["paper_round"]["paper_build_commit"] == sha
    assert roster["meta"]["columns_at_sha"] == ["source", "model", "lane_kind", "arm", "eval_id", "metric", "value"]
    load = RO.load(out1["roster"], out1["rulings"])
    # fig1: 2 models x 7 objects x 6 evals x 1 rule = 84; fig2 companion: gemma b2i + nemotron carried = 2 x 7 x 5 = 70
    assert load.counts_by_comparison["fig1_headline"] == 84 and load.counts_by_comparison["fig2_jailbreak/base_to_instruct"] == 70
    assert load.counts_by_comparison["fig4_transfer"] == 2 * 3 * (8 + 1)      # 8 declared + agentic_murder drawn via FIG4_CATEGORIES (inferred)
    assert roster["meta"]["expanded_counts_by_comparison"]["total"] == len(load.entries)
    # every entry cites the pinned symbol at the sha; the fig4 inferred eval carries its reason
    assert all(f"@ {sha[:8]}" in e["source_citation"] for e in load.entries)
    inferred = [e for e in load.entries if e["comparison_id"] == "fig4_transfer" and e["eval_id"] == "agentic_murder"]
    assert inferred and all(e["inferred"] == 1 and "FIG4_CATEGORIES" in e["extra"]["inferred_reason"] for e in inferred)
    # the AlphaSteer ladder rule, the Arditi rungs from the ladder grammar, the descoped family
    assert any(e["strength_rule"] == "ladder:alphasteer_lambda_ladder" for e in load.entries)
    assert {e["strength_rule"] for e in load.entries if e["comparison_id"] == "fig5_frontier/instruct_direct" and (e["object_selector_json"] or {}).get("paper_label") == "arditi"} >= {"fixed:-2", "fixed:2", "fixed:4", "fixed:8", "fixed:0.5", "operating_point"}
    assert roster["models"]["glm46vflash"]["descoped"] is True and roster["models"]["qwen35_4b_it"]["descoped"] is True
    # rulings regenerated from the symbols, with their expiry semantics
    ids = {r["ruling_id"]: r for r in load.rulings}
    mt = ids["r2v1-supersede-gemma-arditi-r2-multiturn-r0p1"]
    assert mt["expires_when_json"]["kind"] == "any_of" and [c["kind"] for c in mt["expires_when_json"]["conditions"]] == ["served_id_absent", "grades_root_absent"]
    em = ids["r2v1-supersede-gemma-alphasteer-em-bl-lam3"]
    assert em["expires_when_json"]["conditions"] == [] and em["expires_when_json"]["guards"][0]["sha256_prefix"] == "4083cf8c"
    assert ids["r2v1-exclude-gemma-difference_lora-1p5-xstest"]["scope_json"]["match"] == {"model_ids": ["gemma4e4b_it"], "paper_labels": ["difference_lora"], "strength_nominals": [1.5], "eval_ids": ["xstest"]}
    assert ids["r2v1-descope-model-glm46vflash"]["action"] == "descope" and ids["r2v1-drop_eval-tiny_ll"]["action"] == "drop_eval"
    assert ids["r2v1-exclude-fig5-defence-axis-bon-multiturn"]["scope_kind"] == "comparison"
    assert "r2v1-exclude-not_run_by_design-nemotron-agentic_murder" in ids and "r2v1-exclude-declined-constitution_si-multiturn" in ids
    # the not-run-by-design cells get dispositions; the declined constitution-SI cells would too, had the mini config an si_constitution arm
    assert load.dispositions and {d["state"] for d in load.dispositions} == {"not_applicable"}


def test_transcribe_refuses_a_checkout_not_at_the_sha(checkout, tmp_path):
    co, sha = checkout
    with pytest.raises(T.TranscriptionRefusal, match="not"):
        T.transcribe(co, "0" * 40, str(tmp_path / "o"))
    with pytest.raises(T.TranscriptionRefusal):
        T.transcribe(str(tmp_path / "nowhere"), sha, str(tmp_path / "o"))


def test_previous_version_provenance_and_disagreements_are_carried_and_listed_first(checkout, tmp_path):
    co, sha = checkout
    first = T.transcribe(co, sha, str(tmp_path / "v1"), read_utc="2026-01-01T00:00:00.000Z")
    # a reviewer edits the first version: a disagreement exception and a written ruling the config cannot see
    prev = yaml.safe_load(open(first["roster"]))
    prev["sets"]["common_exceptions"].append({"kind": "annotate", "model_id": "gemma4e4b_it", "object": "alphasteer_rebake", "inferred": True,
                                             "disagreement": "D-1: the user ruled the object obsolete; the build still reads it"})
    prev_roster = tmp_path / "prev_roster.yaml"
    prev_roster.write_text(yaml.safe_dump(prev, sort_keys=False))
    header, rulings = RO.read_rulings_jsonl(first["rulings"])
    written = {"ruling_id": "r2v1-exclude-alphasteer-objects-out", "round_id": "r2", "action": "exclude", "scope_kind": "object",
               "scope_json": {"object_kind": "alphasteer", "match": {"object_kinds": ["alphasteer"]}}, "target_json": {}, "expires_when_json": None,
               "reason": "the user: AlphaSteer is out", "ruled_utc": "2026-01-05T00:20:00Z", "ruled_by": "the user via DataWrangler", "source_ref": "decisions.md D29", "extra_json": {}}
    em = next(r for r in rulings if r["ruling_id"] == "r2v1-supersede-gemma-alphasteer-em-bl-lam3")
    em["ruled_by"] = "Orchestrator (paper owner); the move by the user via AlphaKL"
    em["ruled_utc"] = "2026-09-14T18:25:00Z"
    em.pop("ruled_utc_precision", None)      # the reviewer moved the stamp to a time of day: a stale `date` precision would be refused by name
    prev_rulings = tmp_path / "prev_rulings.jsonl"
    prev_rulings.write_text(json.dumps(header) + "\n" + "".join(json.dumps(r) + "\n" for r in rulings + [written]))
    second = T.transcribe(co, sha, str(tmp_path / "v2"), previous_roster=str(prev_roster), previous_rulings=str(prev_rulings), read_utc="2026-01-02T00:00:00.000Z")
    load = RO.load(second["roster"], second["rulings"])
    assert load.disagreements and all("D-1" in e["extra"]["disagreement"] for e in load.disagreements)
    assert RO.show_text(load).splitlines()[1].startswith("DISAGREEMENTS FIRST") and "D-1" in RO.show_text(load).splitlines()[2]
    ids = {r["ruling_id"]: r for r in load.rulings}
    assert ids["r2v1-supersede-gemma-alphasteer-em-bl-lam3"]["ruled_by"].startswith("Orchestrator") and ids["r2v1-supersede-gemma-alphasteer-em-bl-lam3"]["ruled_utc"] == "2026-09-14T18:25:00.000Z"
    assert ids["r2v1-supersede-gemma-alphasteer-em-bl-lam3"]["ruled_utc_precision"] == "datetime"
    assert "r2v1-exclude-alphasteer-objects-out" in ids and ids["r2v1-exclude-alphasteer-objects-out"]["ruled_by"] == "the user via DataWrangler"
    # the diff between the two versions is on inferred / citation only, never on the entry set
    a = RO.load(first["roster"], first["rulings"])
    d = RO.diff_entries(a.entries, load.entries)
    assert not d["only_a"] and not d["only_b"] and d["coarse_equal"] and all(set(c["diffs"]) <= {"inferred"} for c in d["changed"])


def test_transcribe_cli_writes_and_reports(checkout, tmp_path, monkeypatch, capsys):
    from sisteer.registry import cli
    co, sha = checkout
    monkeypatch.setenv("REGISTRY_ROOT", str(tmp_path / "reg"))
    rc = cli.main(["roster", "transcribe", "--paper-checkout", co, "--sha", sha, "--out", str(tmp_path / "cli_out")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert os.path.exists(out["roster"]) and os.path.exists(out["rulings"]) and out["entries"] > 0
    assert cli.main(["roster", "transcribe", "--paper-checkout", co, "--sha", "1" * 40, "--out", str(tmp_path / "cli_out2")]) == 1
    assert "refused" in capsys.readouterr().err
