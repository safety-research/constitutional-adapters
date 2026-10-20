"""ifeval-canonical-battery — the frozen 40-row matrix through the IFEval row builder.

Pinned: ``make_ifeval_matrix.py --check`` is byte-identical and resolves every IFEVAL_CONTRASTS leg; the counts (40 rows /
30 launched / 2 held / 8 deferred / 13 fitted launched / 5 constitution-prompted / 0 header-located / 7 TAGs / 7 slugs,
gemma 7 / Qwen 7 / Nemotron 6 / Llama 10 launched); NO PSR cell and no ``safety_refusal__lora_r4`` anywhere (the
exclusions); ``prompted__si_constitution`` on exactly the five (model, view)s that carry a prompted cell, right after it,
with the constitution SI text's sha256 ASSERTED and recorded per row, its OWN geometry entries and prompt class, and the
``_full`` spelling refused by the cell grammar; the two held rows are the it-view constitution LoRA cells (``status: held``
+ ``held_reason``) and get no configuration / argv (P41); the eight DEFERRED slots (AlphaSteer / Arditi x four models, LAST)
carry no object sha, a per-model record, the selection rule, and are refused by every production consumer; every row
carries a ``pin`` (the roster pin / a stated null / the deferred record); ``if_views`` imports from the GSM8K builder ONLY
the neutral enumeration (a source-text pin, P31); every row is ``declared False``, ``standup_cell == cell``, thinking OFF,
``vendor_preset`` with the model's row by import, the OFF stamp triple; the GSM8K frozen matrix is untouched;
matrix-to-driver argv for a Llama object cell (pass 1 and a rung) with ``--attempt-id`` LAST.

FROZEN INPUTS (the roster-shadow remedy): the matrix regenerates from the shared roster / pins AS COMMITTED AT
THE PIN 59cedcca (byte-identical copies under ``configs/sdfbase/ifeval/frozen/59cedcca/``, named by ``views.yaml``
``frozen_inputs:`` and sha-asserted on every load through the GSM8K lane's own ``frozen_input_paths``), never from the live
registry, which gained a ``llama8b`` target and block on 2026-09-12 (the Llama wave) and a roster-served constitution
cell, and refuses this lane's inputs by the adds-never-shadows and not-an-extra rules. Pinned here: the ``--check``
byte-identity against the REAL configs at the repo tip (the four strict xfails came off), the copies' shas = the matrix's
recorded registry shas, the live registry moved and the live path refuses on both files, a tree with NO live registry
reproduces the matrix, a tree whose LIVE registry is MUTATED (targets added / a cell dropped / a pin re-pointed) reproduces
the matrix byte for byte while the live path reads the mutation (the known-bad twin), an edited or missing copy refuses by
name, the adds-never-shadows rule intact (a frozen copy that carries the target refuses), and the ``--views`` form reading a
peer's LIVE registry with ``frozen_inputs: null`` recorded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

REPO = Path(__file__).resolve().parents[1]
IFE = REPO / "scripts" / "sdfbase" / "ifeval"
for p in (REPO / "scripts" / "sdfbase" / "gsm8k", REPO / "scripts" / "sdfbase" / "full22", IFE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load("_if_views_under_test", "scripts/sdfbase/ifeval/if_views.py")
G8K = _load("_g8k_views_for_the_ifeval_tests", "scripts/sdfbase/gsm8k/g8k_views.py")   # the GSM8K matrix's own reader (P16 frozen inputs)
IM = _load("_if_matrix_under_test", "scripts/sdfbase/ifeval/if_matrix.py")
GEN = _load("_make_ifeval_matrix_under_test", "scripts/sdfbase/ifeval/make_ifeval_matrix.py")
DOC = IM.load_matrix()
ROWS = DOC["rows"]
COUNTS = IM.counts(ROWS)
HELD_TRIPLES = {("gemma4e4b", "it", "constv2-mix-full__lora_r4_ce"), ("qwen35_4b", "it", "constv2-mix-full__lora_r4_ce")}
CONSTITUTION = "prompted__si_constitution"
CONSTITUTION_SHA = "21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f"
DEFERRED_SLOTS = {(arm, key, view) for arm in ("alphasteer", "arditi")
                  for key, view in (("gemma4e4b", "it"), ("qwen35_4b", "it"), ("nemotron3nano4b", "main"), ("llama8b", "main"))}
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}
SPEC = V.load_views_yaml(V.views_path(REPO))
FROZEN_PIN = "59cedcca"
FROZEN_DIR = REPO / "configs/sdfbase/ifeval/frozen" / FROZEN_PIN
FROZEN_ROSTER_SHA = "945f97f5b447d2a01bfa8ac966b4e44324224351fb405cb7524dd05d2869ff6a"
FROZEN_PINS_SHA = "a57146e774b4f5b64781162ff3b8189ab149e58dc858792cefd15a2ceee09e9b"
LIVE_ROSTER_REL, LIVE_PINS_REL = "configs/sdfbase_full22_roster.yaml", "configs/sdfbase_full22_pins.json"
LIVE_ROSTER, LIVE_PINS = REPO / LIVE_ROSTER_REL, REPO / LIVE_PINS_REL
LANE_FILES = ("configs/sdfbase/ifeval/views.yaml", "configs/sdfbase/ifeval/llama8b_roster.yaml", "configs/sdfbase/ifeval/llama8b_pins.json",
              f"configs/sdfbase/ifeval/frozen/{FROZEN_PIN}/sdfbase_full22_roster.yaml", f"configs/sdfbase/ifeval/frozen/{FROZEN_PIN}/sdfbase_full22_pins.json")


def _load_inputs():
    """The generator's inputs the way the generator reads them: the FROZEN copies views.yaml names + the overlays."""
    return V.load_inputs(REPO, SPEC)


def _lane_tree(root: Path, *, frozen_roster: bytes | None = None, frozen_pins: bytes | None = None,
               live_roster: bytes | None = None, live_pins: bytes | None = None) -> Path:
    """A configs root carrying this lane's files (views.yaml, the two overlays, the frozen copies) and — only when given —
    a LIVE registry; a frozen copy optionally replaced (its declared sha in views.yaml re-pinned to the replacement)."""
    views = root / LANE_FILES[0]
    for rel in LANE_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((REPO / rel).read_bytes())
    text = views.read_text(encoding="utf-8")
    if frozen_roster is not None:
        (root / SPEC.frozen_inputs["roster"]).write_bytes(frozen_roster)
        text = text.replace(FROZEN_ROSTER_SHA, hashlib.sha256(frozen_roster).hexdigest())
    if frozen_pins is not None:
        (root / SPEC.frozen_inputs["pins"]).write_bytes(frozen_pins)
        text = text.replace(FROZEN_PINS_SHA, hashlib.sha256(frozen_pins).hexdigest())
    views.write_text(text, encoding="utf-8")
    for rel, blob in ((LIVE_ROSTER_REL, live_roster), (LIVE_PINS_REL, live_pins)):
        if blob is not None:
            (root / rel).write_bytes(blob)
        else:
            assert not (root / rel).exists()
    return root


def _row(key, view, cell):
    return IM.row_for(key, view, cell, ROWS)


# --------------------------------------------------------------------------- the frozen matrix
def test_matrix_regenerates_byte_identically_resolves_every_contrast_and_prints_the_held_and_deferred_lines():
    """Against the REAL configs at the repo tip — the generator reads the FROZEN registry copies, so this is the
    check that refused at the live tip on 2026-09-13 and 2026-09-17; the strict xfail came off when the inputs were frozen."""
    r = subprocess.run([sys.executable, str(IFE / "make_ifeval_matrix.py"), "--check"], capture_output=True, text=True,
                       cwd=REPO, env=ENV, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout and "23/23 cell_minus_none legs resolve" in r.stdout
    assert r.stdout.count("HELD:") == 2 and all(f"index={i}" in r.stdout for i in (3, 11))
    assert r.stdout.count("DEFERRED:") == 8 and all(f"DEFERRED: index={i}" in r.stdout for i in range(32, 40))
    text, doc = GEN.build(configs_root=REPO, views_yaml=None, views_spec=None)
    assert text == (REPO / IC.IFEVAL_MATRIX_REL).read_text(encoding="utf-8")
    assert doc["schema"] == IC.MATRIX_SCHEMA == "ifeval_matrix/1"
    assert doc["task"] == "inspect/ifeval" and doc["samples_per_item"] == 2 and doc["thinking"] == "off"
    assert doc["caps"] == [2048, 4096, 8192] and doc["trigger_threshold"] == 0.01 and doc["seed_rule"] == IC.SEED_RULE_IFEVAL
    assert doc["trigger_provenance"] == IC.IFEVAL_TRIGGER_PROVENANCE
    src = doc["source"]
    assert src["roster_overlay"] == "configs/sdfbase/ifeval/llama8b_roster.yaml" and src["pins_overlay"] == "configs/sdfbase/ifeval/llama8b_pins.json"
    assert all(re.fullmatch(r"[0-9a-f]{64}", src[k]) for k in ("roster_sha256", "pins_sha256", "roster_overlay_sha256", "pins_overlay_sha256"))
    assert src["frozen_inputs"] == SPEC.frozen_inputs and src["roster_sha256"] == FROZEN_ROSTER_SHA and src["pins_sha256"] == FROZEN_PINS_SHA
    assert len(doc["slugs"]) == 7
    # the exclusion policy is recorded in the document, and the prompt classes name both entry families
    pol = doc["policy"]
    assert [e.get("arms_matching") or e.get("cells") for e in pol["exclude"]] == ["psr", ["safety_refusal__lora_r4"]]
    assert [e["cell"] for e in pol["prompted_extra"]] == [CONSTITUTION] and pol["prompted_extra"][0]["text_sha256"] == CONSTITUTION_SHA
    assert pol["prompted_extra"][0]["placement"] == "views_with_prompted_cell" and pol["prompted_extra"][0]["text_chars"] == 30952
    assert [d["arm"] for d in pol["deferred"]] == ["alphasteer", "arditi"]
    assert doc["prompt_classes"] == {"short": {"2048": "capability:ifeval_2048", "4096": "capability:ifeval_4096", "8192": "capability:ifeval_8192"},
                                     "long_si": {"2048": "capability:ifeval_longsi_2048", "4096": "capability:ifeval_longsi_4096",
                                                 "8192": "capability:ifeval_longsi_8192"}}


def test_counts_are_40_rows_30_launch_2_held_8_deferred_and_the_per_model_x_view_split():
    assert [r["index"] for r in ROWS] == list(range(40))
    assert {k: COUNTS[k] for k in ("rows", "held", "launched", "deferred", "fitted", "fitted_launched", "in_support",
                                    "prompted_constitution", "prompted_safety", "header_located", "declared",
                                    "contrasts_designed", "n_tags", "n_slugs", "unpinned_deferred")} == {
        "rows": 40, "held": 2, "launched": 30, "deferred": 8, "fitted": 15, "fitted_launched": 13, "in_support": 17,
        "prompted_constitution": 5, "prompted_safety": 5, "header_located": 0, "declared": 0, "contrasts_designed": 23,
        "n_tags": 7, "n_slugs": 7, "unpinned_deferred": 8}
    assert COUNTS["by_model"] == {"gemma4e4b": {"launched": 7, "held": 1, "deferred": 2}, "qwen35_4b": {"launched": 7, "held": 1, "deferred": 2},
                                  "nemotron3nano4b": {"launched": 6, "held": 0, "deferred": 2}, "llama8b": {"launched": 10, "held": 0, "deferred": 2}}
    assert COUNTS["by_model_view"] == {
        "gemma4e4b/it": {"launched": 3, "held": 1, "deferred": 2}, "gemma4e4b/main": {"launched": 4, "held": 0, "deferred": 0},
        "qwen35_4b/it": {"launched": 3, "held": 1, "deferred": 2}, "qwen35_4b/main": {"launched": 4, "held": 0, "deferred": 0},
        "nemotron3nano4b/main": {"launched": 6, "held": 0, "deferred": 2},
        "llama8b/main": {"launched": 5, "held": 0, "deferred": 2}, "llama8b/it": {"launched": 5, "held": 0, "deferred": 0}}
    assert COUNTS["tags"] == ["gemma4e4b_it", "gemma4e4b", "qwen35_4b_it", "qwen35_4b", "nemotron3nano4b", "llama8b", "llama8b_it"]
    assert COUNTS["by_tag"]["gemma4e4b"] == {"rows": 4, "held": 0, "launched": 4, "deferred": 0}
    assert COUNTS["by_tag"]["gemma4e4b_it"] == {"rows": 6, "held": 1, "launched": 3, "deferred": 2}
    assert COUNTS["distinct_paths"] == COUNTS["n_paths"] and COUNTS["distinct_out_roots"] == 40
    assert COUNTS["pins"] == {"gemma4e4b": 4, "qwen35_4b": 4, "nemotron3nano4b": 3, "llama8b": 4}
    assert COUNTS["families"] == ["gemma4", "llama", "nemotron_h", "qwen3_5"] and COUNTS["images"] == ["default", "nemotron"]
    assert COUNTS["two_view_models"] == ["gemma4e4b", "llama8b", "qwen35_4b"]
    assert len(IM.launch_rows(ROWS)) == 30 and len(IM.held_rows(ROWS)) == 2 and len(IM.deferred_rows(ROWS)) == 8
    assert len({r["job_id"] for r in ROWS}) == 40
    # the launch set is the paper's eight arms minus what is deferred / held, per model
    arms = {m: sorted({r["cell"] for r in ROWS if r["model_key"] == m and r["status"] != IC.STATUS_DEFERRED}) for m in COUNTS["models"]}
    assert arms["gemma4e4b"] == arms["qwen35_4b"] == sorted(["none", "prompted__safety_refusal", CONSTITUTION, "constv2-mix-full__lora_r4_ce",
                                                              "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"])
    assert arms["nemotron3nano4b"] == arms["gemma4e4b"]
    assert arms["llama8b"] == sorted(["none", "prompted__safety_refusal", CONSTITUTION, "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce"])


def test_the_d23_exclusions_leave_no_psr_and_no_safety_lora_row_at_all():
    cells = [r["cell"] for r in ROWS]
    assert not any("psr" in c for c in cells) and "safety_refusal__lora_r4" not in cells
    assert not any(r["header_located"] for r in ROWS) and COUNTS["header_located"] == 0
    pol = V.load_ifeval_policy(V.views_path(REPO))
    for cell in ("constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all_minus_ctl_distilled",
                 "safety_refusal__psr_kl_all"):
        assert pol.exclusion_for(cell) and "PSR is OUT" in pol.exclusion_for(cell)
    assert pol.exclusion_for("safety_refusal__lora_r4") and "not one of the paper's eight arms" in pol.exclusion_for("safety_refusal__lora_r4")
    for cell in ("none", "prompted__safety_refusal", CONSTITUTION, "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "alphasteer"):
        assert pol.exclusion_for(cell) is None
    # without the policy the neutral enumeration still yields the old 46 (the exclusions are the matrix's, not the roster's)
    spec = SPEC
    roster, pins, _ = _load_inputs()
    assert len(V.select_rows(roster, pins, spec.views, holds=spec.holds)) == 46


def test_the_constitution_cell_is_on_the_five_prompted_views_with_its_sha_asserted_and_its_own_geometry():
    con = [r for r in ROWS if r["cell"] == CONSTITUTION]
    assert [(r["model_key"], r["view"]) for r in con] == [("gemma4e4b", "it"), ("qwen35_4b", "it"), ("nemotron3nano4b", "main"),
                                                          ("llama8b", "main"), ("llama8b", "it")]
    assert {(r["model_key"], r["view"]) for r in con} == {(r["model_key"], r["view"]) for r in ROWS if r["cell"] == IC.CELL_PROMPTED}
    for r in con:
        assert r["status"] == "launch" and r["fitted"] is False and r["object_sha256"] is None and r["route_kind"] == "prompted"
        assert r["si_text_sha256"] == CONSTITUTION_SHA == C.IFEVAL_CONSTITUTION_SI_SHA256 and r["prompt_class"] == "long_si"
        assert r["geometry_eval_by_cap"] == {"2048": "capability:ifeval_longsi_2048", "4096": "capability:ifeval_longsi_4096",
                                             "8192": "capability:ifeval_longsi_8192"}
        assert r["budget_class"] == CONSTITUTION and r["harness_cell"] == r["standup_harness_cell"] == "prompted__constitution"
        assert r["standup_cell"] == CONSTITUTION and r["pin"] == {"kind": "prompted", "object_sha256": None}
        assert r["job_id"].endswith("-constitution") and r["expected_stamps"] == ["cell-stable", "not-consulted", "thinking_off"]
        # right after the view's prompted safety cell
        prev = ROWS[r["index"] - 1]
        assert (prev["model_key"], prev["view"], prev["cell"]) == (r["model_key"], r["view"], IC.CELL_PROMPTED)
    # the live SI text IS the pinned one (the generator asserts it; a moved text refuses)
    from sisteer.eval.jailbreak import resolve_instruction
    text = str(resolve_instruction("si_constitution").text)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == CONSTITUTION_SHA and len(text) == 30952 == C.IFEVAL_CONSTITUTION_SI_CHARS
    assert V.si_text_sha256("si_constitution") == (CONSTITUTION_SHA, 30952)
    with pytest.raises(SystemExit, match="sha256 is .* not the pinned"):
        V.assert_prompted_extra({"cell": CONSTITUTION, "si": "si_constitution", "text_sha256": "0" * 64, "text_chars": 30952}, where="t")
    with pytest.raises(SystemExit, match="is 30952 chars, not the pinned"):
        V.assert_prompted_extra({"cell": CONSTITUTION, "si": "si_constitution", "text_sha256": CONSTITUTION_SHA, "text_chars": 1}, where="t")
    # the safety-prompted rows carry their SI sha too; object / bare rows none
    for r in ROWS:
        if r["cell"] == IC.CELL_PROMPTED:
            assert r["si_text_sha256"] == hashlib.sha256(str(resolve_instruction("safety_refusal").text).encode()).hexdigest()
            assert r["prompt_class"] == "short" and r["geometry_eval_by_cap"]["2048"] == "capability:ifeval_2048"
        elif r["cell"] != CONSTITUTION:
            assert r["si_text_sha256"] is None and r["prompt_class"] == "short"


def test_the_full_constitution_spelling_is_refused_everywhere():
    """NEVER prompted__si_constitution_full: the cell grammar (parse_cell_id / parse_launcher_cell), the harness translation
    and the views.yaml policy loader all refuse it by name."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra
    bad = "prompted__si_constitution_full"
    with pytest.raises(dc.DefenseCellError, match="malformed prompted cell id"):
        dc.parse_cell_id(bad)
    # the harness spelling `prompted__constitution_full` is admitted by the grammar (the Llama fullconst view) and
    # the `si_`-prefixed spelling translates to it — so the IFEval design's refusal is a policy one: no row and no
    # views.yaml entry names either spelling, and the loader refuses a `_full` prompted extra by name (below)
    assert dc.parse_cell_id("prompted__constitution_full").si_id == "constitution_full"
    assert ra.harness_cell_id(bad) == "prompted__constitution_full"
    assert not any(r["cell"].endswith("_full") or str(r.get("harness_cell", "")).endswith("_full") for r in ROWS)
    policy_text = "\n".join(l.split("#", 1)[0] for l in (REPO / "configs/sdfbase/ifeval/views.yaml").read_text(encoding="utf-8").splitlines())
    # comments aside; the LoRA SI names and the frozen registry copies' file names (sdfbase_full22_*) carry `full` legitimately
    assert "_full" not in policy_text.replace("constv2-mix-full", "").replace("constv2ctl2-full", "").replace("sdfbase_full22", "")
    # the launcher grammar admits the TrainTest spelling too since the Llama fullconst view (it translates to the harness
    # spelling above); the IFEval policy loader is what refuses it as an extra — pinned in
    # test_the_policy_loader_refuses_ill_formed_entries
    assert C.parse_launcher_cell(bad).kind == "prompted"
    with pytest.raises(C.IfevalPromptClassUnmeasured):
        C.ifeval_prompt_class_of_cell("prompted__si_irrelevant")
    assert C.ifeval_prompt_class_of_cell(CONSTITUTION) == "long_si" and dc.parse_cell_id("prompted__constitution").si_id == "constitution"
    assert not any(r["cell"].endswith("_full") and r["cell"].startswith("prompted__") for r in ROWS)


def test_the_two_held_rows_are_the_it_view_constitution_loras_and_a_hard_exclusion_everywhere():
    held = IM.held_rows(ROWS)
    assert {(r["model_key"], r["view"], r["cell"]) for r in held} == HELD_TRIPLES
    assert [r["index"] for r in held] == [3, 11]
    g8k = json.loads((REPO / "configs/sdfbase/gsm8k/matrix.json").read_text(encoding="utf-8"))
    g8k_held = {(r["model_key"], r["view"], r["cell"]) for r in g8k["rows"] if r["status"] == "held"}
    assert HELD_TRIPLES < g8k_held                  # GSM8K still holds the PSR twins; here they left the matrix
    for r in held:
        assert r["status"] == "held" and "P41" in r["held_reason"] and "batch 64" in r["held_reason"]
        assert r["fitted"] and r["object_sha256"] and r["pin"]["kind"] == "roster" and r["pin"]["object_sha256"] == r["object_sha256"]
    with pytest.raises(SystemExit, match="HELD"):
        IM.refuse_held([held[0]], flag="--cells")
    with pytest.raises(SystemExit, match="HELD"):
        V.driver_argv_for_row(held[0], rung=2048, python="py", script="s", proxy_url="u", routing_table="t", ledger_url="l",
                              defenses_dir="d", out_root="o", budget="b", concurrency=8, vllm_log="v", attempt_id="a1",
                              judge_model="judge")
    assert all(r["status"] == "launch" for r in IM.launch_rows(ROWS))
    assert len(IM.rows_of_slug("gemma4e4b-sdfbase-it", ROWS)) == 6      # 3 launch + 1 held + 2 deferred


def test_the_eight_deferred_slots_carry_no_sha_a_record_and_are_refused_by_every_consumer():
    deferred = IM.deferred_rows(ROWS)
    assert {(r["cell"], r["model_key"], r["view"]) for r in deferred} == DEFERRED_SLOTS
    assert [r["index"] for r in deferred] == list(range(32, 40)) and [r["cell"] for r in deferred] == ["alphasteer"] * 4 + ["arditi"] * 4
    for r in deferred:
        assert r["status"] == "deferred" and r["object_sha256"] is None and r["staged_id"] is None and r["fitted"] is True
        assert r["route_kind"] == "lens" and r["held_reason"] is None and r["served_positions_expected"] == "all"
        d = r["deferred"]
        assert d["arm"] == r["cell"] and d["record"] == f"configs/sdfbase/ifeval/deferred/{r['cell']}__{r['model_key']}.json"
        assert d["record_fields"] == ["object_sha256", "view", "rung", "selection_rule"] == list(V.DEFERRED_RECORD_FIELDS) and d["filled"] is False
        assert "minimal over-refusal strength" in d["selection_rule"] and "20 pp" in d["selection_rule"] and d["view_provisional"] is True
        assert r["pin"] == {"kind": "deferred", "object_sha256": None, "record": d["record"], "record_fields": d["record_fields"], "filled": False}
        assert f"record {d['record']}" in d["pending"]
        assert r["geometry_eval_by_cap"]["2048"] == "capability:ifeval_2048" and r["prompt_class"] == "short"
        assert r["job_id"] == f"ife-{r['model_key']}-{r['view']}-{r['cell']}"
        # both Class B ids are in this tree's harness vocabulary (`arditi` joined `SERVED_ONLY_CLASS_B_IDS` after the
        # freeze), so the harness spelling resolves and the ONLY pending item is the record (the frozen inputs freeze the
        # registry, never the code: a vocabulary gap the freeze recorded closes when the code closes it)
        assert r["harness_cell"] == r["standup_harness_cell"] == r["standup_cell"] == r["cell"]
        assert d["pending"] == [f"record {d['record']}"]
        assert not any(p.startswith("harness vocabulary:") for p in d["pending"])
    # every production consumer refuses a deferred slot by name
    with pytest.raises(SystemExit, match="DEFERRED"):
        IM.refuse_held([deferred[0]], flag="--rows")
    with pytest.raises(SystemExit, match="DEFERRED"):
        V.driver_argv_for_row(deferred[0], rung=2048, python="py", script="s", proxy_url="u", routing_table="t", ledger_url="l",
                              defenses_dir="d", out_root="o", budget="b", concurrency=8, vllm_log="v", attempt_id="a1", judge_model="judge")
    assert not any(r["status"] == "deferred" for r in IM.launch_rows(ROWS))
    # a deferred slot is never a contrast leg, but its view must carry a launched none anchor (the fill must resolve)
    res = V.contrast_resolution(ROWS)
    assert len(res["deferred_legs"]) == 8 and all(d["anchor"] for d in res["deferred_legs"])
    assert not any(leg["cell"] in ("alphasteer", "arditi") for leg in res["cell_minus_none"]["legs"])
    orphan = [r for r in ROWS if not (r["model_key"] == "llama8b" and r["view"] == "main" and r["cell"] == "none")]
    with pytest.raises(SystemExit, match="cell_minus_none.*llama8b/main"):
        V.check_contrasts(orphan)
    # the slots' provisional views: it on gemma / qwen (the itdoc parents), main on nemotron / llama
    assert {(r["model_key"], r["view"]) for r in deferred} == {("gemma4e4b", "it"), ("qwen35_4b", "it"), ("nemotron3nano4b", "main"), ("llama8b", "main")}


def test_every_row_carries_a_pin_that_says_where_its_object_identity_comes_from():
    shared = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text(encoding="utf-8"))
    overlay = json.loads((REPO / "configs/sdfbase/ifeval/llama8b_pins.json").read_text(encoding="utf-8"))
    for r in ROWS:
        pin = r["pin"]
        assert pin["object_sha256"] == r["object_sha256"]
        if r["status"] == "deferred":
            assert pin["kind"] == "deferred"
        elif r["fitted"]:
            assert pin["kind"] == "roster" and pin["staged_id"] == r["staged_id"]
            table, rel, key = ((overlay, "configs/sdfbase/ifeval/llama8b_pins.json", "pins_overlay_sha256") if r["model_key"] == "llama8b"
                               else (shared, "configs/sdfbase_full22_pins.json", "pins_sha256"))
            # the pin's source sha is the FREEZE-TIME sha the matrix recorded (main's shared pins moved under the frozen
            # matrix — the llama8b block joined them with identical object shas); the live file's sha is a separate fact
            assert pin["source"] == rel and pin["source_sha256"] == DOC["source"][key]
            assert table[r["model_key"]][r["staged_id"]]["object_sha256"] == r["object_sha256"]
            # ... and main's shared pins, which now carry every model, agree on the object
            assert shared[r["model_key"]][r["staged_id"]]["object_sha256"] == r["object_sha256"]
        else:
            assert pin == {"kind": "prompted" if r["cell"].startswith("prompted__") else "none", "object_sha256": None}
    assert {r["pin"]["kind"] for r in ROWS} == {"roster", "none", "prompted", "deferred"}


def test_unholding_is_one_views_yaml_edit_with_no_other_diff():
    spec = SPEC
    pol = V.load_ifeval_policy(V.views_path(REPO))
    roster, pins, pins_source = _load_inputs()
    base = V.select_rows(roster, pins, spec.views, holds=spec.holds, policy=pol, pins_source=pins_source)
    assert base == ROWS
    unheld = V.select_rows(roster, pins, spec.views, holds=[h for h in spec.holds if h["key"] != "gemma4e4b"], policy=pol, pins_source=pins_source)
    assert len(unheld) == 40
    for a, b in zip(base, unheld):
        if a["index"] == 3:
            assert (a["status"], b["status"], b["held_reason"]) == ("held", "launch", None)
            assert {k: v for k, v in a.items() if k not in ("status", "held_reason")} == {k: v for k, v in b.items() if k not in ("status", "held_reason")}
        else:
            assert a == b
    c = IM.counts(unheld)
    assert (c["launched"], c["held"], c["deferred"], c["fitted_launched"]) == (31, 1, 8, 14)


def test_if_views_reuses_only_the_gsm8k_lanes_neutral_enumeration():
    # (no longer xfail: the generator reads the FROZEN registry copies, which carry no llama8b, so the overlay ADDS the
    # target under the one rule)
    src = (IFE / "if_views.py").read_text(encoding="utf-8")
    m = re.search(r"^from g8k_views import (.+?)\s*#", src, re.MULTILINE)
    assert m, "if_views must import the neutral enumeration from g8k_views"
    names = {n.strip() for n in m.group(1).split(",")}
    # + the frozen-copy sha assertion: an INPUT helper, not row logic — the GSM8K lane's P16 pattern reused, not copied
    assert names == {"ViewsSpec", "cells_of", "frozen_input_paths", "load_views_yaml", "parse_views_spec", "pin_for"}
    assert "def select_rows(" in src and src.count("from g8k_views import") == 1 and "import g8k_cells" not in src
    assert "standup_spellings" not in src and "g8k_expected_stamps" not in src and "tuple_name" not in src
    assert "@thinking" not in json.dumps(ROWS)
    # the neutral rule, through the merged roster, yields the Llama views the FROZEN shared roster lacks (PSR cells
    # included — the exclusion is the matrix's policy, applied AFTER the enumeration)
    roster, _pins, _ = _load_inputs()
    assert [c for c, _ in V.cells_of(roster, "llama8b", "it")] == ["none", "prompted__safety_refusal", "constv2-mix-full__psr_ce_all",
                                                                    "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all",
                                                                    "constv2ctl2-full__lora_r4_ce"]
    with pytest.raises(SystemExit, match="no target"):
        V.cells_of(json.loads(json.dumps({"targets": {k: v for k, v in roster["targets"].items() if k != "llama8b"}, "lane_flags": roster["lane_flags"]})), "llama8b", "it")


def test_every_row_is_an_off_undeclared_vendor_preset_row_with_its_pin_slug_and_stamps():
    from sisteer.capeval.petri import registry_adapter as ra
    shared = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text(encoding="utf-8"))
    overlay = json.loads((REPO / "configs/sdfbase/ifeval/llama8b_pins.json").read_text(encoding="utf-8"))
    for r in ROWS:
        assert r["declared"] is False and r["thinking"] == "off" and r["standup_cell"] == r["cell"]
        if r["harness_cell"] is not None:
            assert r["standup_harness_cell"] == r["harness_cell"] == ra.harness_cell_id(r["cell"]) == C.cell_spellings(r["cell"], declared=False).harness
        assert r["sampling_mode"] == "vendor_preset" and r["sampling"] == IC.sampling_row_dict(IC.vendor_preset_row(r["model_key"]))
        assert r["samples_per_item"] == 2 and r["caps"] == [2048, 4096, 8192] and r["cap"] == 2048
        assert r["slug"] == IC.IFEVAL_SLUGS[(r["model_key"], r["view"])] and r["tag"] == IC.ifeval_tag(r["model_key"], r["view"])
        assert r["route_kind"] == IC.ifeval_route_kind(r["cell"])
        assert r["fitted"] is ((r["staged_id"] is not None) or r["status"] == "deferred")
        assert r["header_located"] is (r["fitted"] and r["cell"] in IC.IFEVAL_HEADER_LOCATED_CELLS)
        assert r["served_positions_expected"] == (("response" if r["header_located"] else "all") if r["fitted"] else None)
        assert r["expected_stamps"] == list(IC.ifeval_expected_stamps(fitted=r["fitted"], header_located=r["header_located"]))
        assert r["family"] == C.FAMILY_BY_MODEL_KEY[r["model_key"]]
        assert r["image"] == ("nemotron" if r["family"] == "nemotron_h" else "default")
        assert (r["hf"], r["rev"]) == IC.IFEVAL_MODEL_PINS[r["model_key"]]
        assert r["geometry_eval_by_cap"] == {str(c): IC.IFEVAL_GEOMETRY_ENTRY_BY_CLASS[r["prompt_class"]][c] for c in (2048, 4096, 8192)}
        assert r["trigger_threshold"] == 0.01 and r["trigger_provenance"] == IC.IFEVAL_TRIGGER_PROVENANCE
        assert r["config"] == f"configs/sdfbase/ifeval/sdfbase_full22_{r['tag']}.ifeval.yaml"
        assert r["manifest"] == f"configs/capeval/manifests/petri_full22_{r['tag']}.ifeval.yaml"
        assert r["budget_class"] == (r["cell"] if r["cell"].startswith("prompted__") else "none")
        if r["staged_id"] is not None:
            table = overlay if r["model_key"] == "llama8b" else shared
            assert r["object_sha256"] == table[r["model_key"]][r["staged_id"]]["object_sha256"]
            assert re.fullmatch(r"[0-9a-f]{64}", r["object_sha256"])
        else:
            assert r["object_sha256"] is None and r["route_kind"] in ("base", "prompted", "lens")
    qwen = _row("qwen35_4b", "main", "none")
    assert qwen["sampling"]["presence_penalty"] == 1.5 and "top_k" not in _row("llama8b", "main", "none")["sampling"]


def test_llamas_twelve_rows_carry_both_views_the_pin_and_the_it_view_anchors():
    llama = [r for r in ROWS if r["model_key"] == "llama8b"]
    assert len(llama) == 12 and sum(r["status"] == "launch" for r in llama) == 10 and sum(r["status"] == "deferred" for r in llama) == 2
    assert all(r["rev"] == "d10aef7999a2b5ba950ab3974312feeedbfe0b77" and r["hf"] == "NousResearch/Meta-Llama-3.1-8B-Instruct" for r in llama)
    by_view = {v: [r["cell"] for r in llama if r["view"] == v and r["status"] == "launch"] for v in ("main", "it")}
    objects = ["constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce"]
    assert by_view["main"] == ["none", "prompted__safety_refusal", CONSTITUTION, *objects] == by_view["it"]
    assert {r["staged_id"] for r in llama if r["view"] == "main" and r["staged_id"]} == {f"base__{c}" for c in objects}
    assert {r["staged_id"] for r in llama if r["view"] == "it" and r["staged_id"]} == {f"itdoc__{c}" for c in objects}
    assert len({r["object_sha256"] for r in llama if r["object_sha256"]}) == 4
    assert _row("llama8b", "it", "none")["slug"] == "llama8b-sdfbase-it" and _row("llama8b", "main", "none")["slug"] == "llama8b-sdfbase-main"


def test_every_promised_contrast_resolves_and_a_matrix_missing_the_llama_it_none_fails_the_check():
    res = V.contrast_resolution(ROWS)
    assert res["unresolved"] == [] and res["contrasts"] == list(IC.IFEVAL_CONTRASTS)
    assert res["cell_minus_none"]["n"] == res["cell_minus_none"]["n_resolved"] == 23
    assert res["none_it_minus_none_main"]["models_with_floor"] == ["gemma4e4b", "qwen35_4b", "llama8b"]
    assert res["none_it_minus_none_main"]["models_without_floor"] == ["nemotron3nano4b"]
    assert DOC["contrasts"] == res
    without = [r for r in ROWS if not (r["model_key"] == "llama8b" and r["view"] == "it" and r["cell"] == "none")]
    with pytest.raises(SystemExit, match="none_it_minus_none_main|cell_minus_none.*llama8b/it"):
        V.check_contrasts(without)
    assert len(V.contrast_resolution(without)["unresolved"]) == 4
    # a matrix where a whole view is held would still resolve (held rows are not legs)
    assert V.contrast_resolution([r for r in ROWS if r["model_key"] != "nemotron3nano4b"])["unresolved"] == []


def test_the_gsm8k_frozen_matrix_is_untouched_by_the_ifeval_lane():
    """The Llama target lives in an OVERLAY: the inputs the GSM8K matrix pins by sha256 are not edited. Since llama-paper-rows
    (P16) the published GSM8K document regenerates from FROZEN roster / pins copies its views.yaml names (`frozen_inputs:`,
    under configs/sdfbase/gsm8k/frozen/), so the shas it records are the frozen copies' — not the live shared registry's,
    which moves (a llama8b block has since landed) without the frozen document moving. The GSM8K generator asserts the
    copies' shas itself; this test reads them through the same loader."""
    g8k = json.loads((REPO / "configs/sdfbase/gsm8k/matrix.json").read_text(encoding="utf-8"))
    spec = G8K.load_views_yaml(REPO / G8K.VIEWS_REL)
    frozen = G8K.frozen_input_paths(spec, REPO)
    assert frozen is not None and spec.frozen_inputs is not None
    assert g8k["source"]["roster_sha256"] == V.sha256_of(frozen[0]) == spec.frozen_inputs["roster_sha256"]
    assert g8k["source"]["pins_sha256"] == V.sha256_of(frozen[1]) == spec.frozen_inputs["pins_sha256"]
    # the frozen copies live under the GSM8K lane's own tree, which the IFEval lane does not write
    assert all(REPO / "configs/sdfbase/gsm8k/frozen" in f.parents for f in frozen)
    # the shared pins carry the llama8b block too (the Llama wave landed); the lane's overlay agrees
    # with it on every staged id both name — the frozen matrix's llama8b object shas are main's
    shared_llama = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text(encoding="utf-8"))["llama8b"]
    overlay_llama = json.loads((REPO / "configs/sdfbase/ifeval/llama8b_pins.json").read_text(encoding="utf-8"))["llama8b"]
    assert set(overlay_llama) <= set(shared_llama)
    assert all(overlay_llama[k]["object_sha256"] == shared_llama[k]["object_sha256"] for k in overlay_llama)
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/gsm8k/make_gsm8k_matrix.py"), "--check"], capture_output=True,
                       text=True, cwd=REPO, env=ENV, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_policy_loader_refuses_ill_formed_entries(tmp_path):
    base = (REPO / "configs/sdfbase/ifeval/views.yaml").read_text(encoding="utf-8")
    def write(text):
        p = tmp_path / "views.yaml"
        p.write_text(text, encoding="utf-8")
        return p
    assert V.load_ifeval_policy(None) == V.IfevalPolicy()
    with pytest.raises(SystemExit, match="every `exclude:` entry needs"):
        V.load_ifeval_policy(write("views: {gemma4e4b: [it]}\nexclude:\n  - reason: x\n"))
    with pytest.raises(C.CellNameError):
        V.load_ifeval_policy(write("views: {gemma4e4b: [it]}\nprompted_extra:\n  - {cell: prompted__si_constitution_full, si: si_constitution_full, "
                                   "text_sha256: a, placement: views_with_prompted_cell, reason: r}\n"))
    with pytest.raises(SystemExit, match="is not prompted__"):
        V.load_ifeval_policy(write("views: {gemma4e4b: [it]}\nprompted_extra:\n  - {cell: prompted__si_constitution, si: safety_refusal, "
                                   "text_sha256: a, placement: views_with_prompted_cell, reason: r}\n"))
    with pytest.raises(SystemExit, match="`placement` must be one of"):
        V.load_ifeval_policy(write("views: {gemma4e4b: [it]}\nprompted_extra:\n  - {cell: prompted__si_constitution, si: si_constitution, "
                                   "text_sha256: a, placement: everywhere, reason: r}\n"))
    with pytest.raises(SystemExit, match="is not an IFEval \\(model, view\\)"):
        V.load_ifeval_policy(write("views: {gemma4e4b: [it]}\ndeferred:\n  - {arm: a, cell: alphasteer, views: {glm46vflash: main}, "
                                   "record: 'x/{model_key}.json', selection_rule: s, reason: r}\n"))
    with pytest.raises(SystemExit, match="must carry {model_key}"):
        V.load_ifeval_policy(write("views: {gemma4e4b: [it]}\ndeferred:\n  - {arm: a, cell: alphasteer, views: {gemma4e4b: it}, "
                                   "record: x.json, selection_rule: s, reason: r}\n"))
    assert V.load_ifeval_policy(write(base)).as_record() == DOC["policy"]


# --------------------------------------------------------------------------- matrix -> driver argv (plan Appendix A)
def _argv(row, rung, subset=None):
    return V.driver_argv_for_row(row, rung=rung, python="/opt/client/bin/python", script="scripts/served_capability_run.py",
                                 proxy_url="http://127.0.0.1:9", routing_table="/tmp/t.json", ledger_url="http://127.0.0.1:10",
                                 defenses_dir="/vol/objects/sdfbase/x/cells_view/main", out_root="/vol/out", budget="/vol/b.json",
                                 concurrency=64, vllm_log="/tmp/v.log", attempt_id="a1", judge_model="judge",
                                 subset_path=subset)


def test_matrix_to_driver_argv_for_a_gemma_object_cell_and_the_constitution_cell():
    row = _row("gemma4e4b", "main", "constv2-mix-full__lora_r4_ce")
    assert row["served_positions_expected"] == "all" and not row["header_located"] and row["declared"] is False
    argv = _argv(row, 2048)
    s = " ".join(argv)
    assert "--cell constv2-mix-full__lora_r4_ce " in s + " " and "@thinking" not in s
    # contracts.driver_argv's shape: the legacy head (--controls / --thinking / --samples ...) then the IFEval tail
    assert "--controls ifeval --thinking off --samples 2 " in s and "--sampling vendor_preset --tinycap-identity-version 2" in s
    assert argv[:3] == ["/opt/client/bin/python", "scripts/served_capability_run.py", "generate"] and argv[-2:] == ["--attempt-id", "a1"]
    assert "--max-tokens-by-task ifeval=2048" in s and "--tinycap-identity-version 2" in s and "--bank-view main" in s
    assert f"--expect-object-sha256 {row['object_sha256']}" in s and "--oos-prompt-budget /vol/b.json" in s
    assert "--off-cap-by-task" not in s and "--item-subset" not in s
    assert "--config configs/sdfbase/ifeval/sdfbase_full22_gemma4e4b.ifeval.yaml" in s
    con = " ".join(_argv(_row("nemotron3nano4b", "main", CONSTITUTION), 2048))
    assert f"--cell {CONSTITUTION} " in con + " " and "--expect-object-sha256" not in con and "--bank-view main" in con
    assert "--config configs/sdfbase/ifeval/sdfbase_full22_nemotron3nano4b.ifeval.yaml" in con


def test_matrix_to_driver_argv_for_a_llama_it_view_object_cell_at_pass_1_and_a_rung():
    row = _row("llama8b", "it", "constv2-mix-full__lora_r4_ce")
    assert row["rev"] == "d10aef7999a2b5ba950ab3974312feeedbfe0b77" and row["geometry_eval_by_cap"]["2048"] == "capability:ifeval_2048"
    s1 = " ".join(_argv(row, 2048))
    assert "--bank-view it" in s1 and f"--expect-object-sha256 {row['object_sha256']}" in s1 and "--max-tokens-by-task ifeval=2048" in s1
    subset = row["subset_paths"]["4096"]
    s4 = " ".join(_argv(row, 4096, subset=f"/vol/{subset}"))
    assert "--max-tokens-by-task ifeval=4096" in s4 and "--off-cap-by-task ifeval=4096" in s4 and f"--item-subset /vol/{subset}" in s4
    assert s4.endswith("--bank-view it --attempt-id a1")
    with pytest.raises(SystemExit, match="draw-subset path"):
        _argv(row, 4096)
    with pytest.raises(SystemExit, match="rung must be one of"):
        _argv(row, 16384)
    bare = " ".join(_argv(_row("llama8b", "it", "none"), 2048))
    assert "--expect-object-sha256" not in bare and "--bank-view it" in bare


def test_the_stdlib_reader_carries_the_same_ladder_and_statuses_and_refuses_a_foreign_matrix(tmp_path):
    assert IM.CAP_LADDER == IC.IFEVAL_CAP_LADDER and IM.CAP == IC.IFEVAL_PASS1_CAP and IM.MATRIX_SCHEMA == IC.MATRIX_SCHEMA
    assert IM.STATUS_HELD == IC.STATUS_HELD and IM.STATUS_DEFERRED == IC.STATUS_DEFERRED == C.IFEVAL_STATUS_DEFERRED == "deferred"
    assert IM.STATUSES == IC.ROW_STATUSES == C.IFEVAL_ROW_STATUSES == ("launch", "held", "deferred")
    assert IM.VIEW_IT == IC.VIEW_IT and IM.CELL_PROMPTED == IC.CELL_PROMPTED and IM.CELL_PROMPTED_CONSTITUTION == IC.CELL_PROMPTED_CONSTITUTION
    src = (IFE / "if_matrix.py").read_text(encoding="utf-8")
    assert "import yaml" not in src and "sisteer" not in src.replace("sisteer-", "")
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps({"schema": "g8k_matrix/1", "rows": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a ifeval_matrix/1"):
        IM.load_matrix(bad)
    odd = tmp_path / "s.json"
    odd.write_text(json.dumps({"schema": "ifeval_matrix/1", "rows": [{**ROWS[0], "status": "maybe"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not launch \\| held \\| deferred"):
        IM.load_matrix(odd)
    with pytest.raises(SystemExit, match="not a matrix row"):
        IM.row_for("gemma4e4b", "main", "prompted__safety_refusal", ROWS)


# --------------------------------------------------------------------------- the frozen inputs (the roster-shadow remedy)
def test_the_matrix_regenerates_from_frozen_registry_copies_whose_shas_are_the_recorded_ones_and_the_live_registry_has_moved():
    """The copies under configs/sdfbase/ifeval/frozen/<pin>/ ARE the bytes whose sha256 the matrix records as its registry
    inputs (main's roster / pins at 59cedcca); they carry no llama8b (the overlay ADDS it) and no roster-served constitution
    cell (the prompted extra IS an extra). The LIVE registry has moved past the pin — a llama8b target + block since the
    Llama wave, the constitution cell served — so the live path REFUSES by name on the roster AND on the pins: the frozen
    copies are load-bearing."""
    frozen = V.frozen_inputs_of(SPEC, REPO)
    assert frozen is not None
    assert frozen["roster_path"] == FROZEN_DIR / "sdfbase_full22_roster.yaml" and frozen["pins_path"] == FROZEN_DIR / "sdfbase_full22_pins.json"
    src = DOC["source"]
    assert src["roster_sha256"] == V.sha256_of(frozen["roster_path"]) == SPEC.frozen_inputs["roster_sha256"] == FROZEN_ROSTER_SHA
    assert src["pins_sha256"] == V.sha256_of(frozen["pins_path"]) == SPEC.frozen_inputs["pins_sha256"] == FROZEN_PINS_SHA
    assert src["frozen_inputs"] == frozen["record"] == SPEC.frozen_inputs
    assert src["roster"] == LIVE_ROSTER_REL and src["pins"] == LIVE_PINS_REL      # what the copies are copies OF (the GSM8K precedent)
    froster = yaml.safe_load(frozen["roster_path"].read_text(encoding="utf-8"))
    fpins = json.loads(frozen["pins_path"].read_text(encoding="utf-8"))
    assert "llama8b" not in froster["targets"] and "llama8b" not in fpins
    assert CONSTITUTION not in frozen["roster_path"].read_text(encoding="utf-8")
    # the merged inputs: the overlay's llama8b block, the frozen pins + the overlay's llama8b pins; the ONE source record
    roster, pins, source = _load_inputs()
    assert "llama8b" in roster["targets"] and sorted(roster["targets"]["llama8b"]["views"]) == ["it", "main"] and "llama8b" in pins
    assert {k: v for k, v in src.items() if k != "views_yaml"} == source                # the document adds only views_yaml
    assert list(source) == ["roster", "roster_sha256", "roster_overlay", "roster_overlay_sha256", "pins", "pins_sha256",
                            "pins_overlay", "pins_overlay_sha256", "frozen_inputs"]
    # every fitted row's pin still names the REGISTRY file (the frozen bytes' sha): the copies changed no row's provenance
    assert all(r["pin"]["source"] in (LIVE_PINS_REL, "configs/sdfbase/ifeval/llama8b_pins.json") for r in ROWS if r["pin"]["kind"] == "roster")
    # the live registry moved (the Llama wave + the constitution cell): the live path refuses BY NAME, roster and pins
    assert V.sha256_of(LIVE_ROSTER) != FROZEN_ROSTER_SHA and V.sha256_of(LIVE_PINS) != FROZEN_PINS_SHA
    live_roster = yaml.safe_load(LIVE_ROSTER.read_text(encoding="utf-8"))
    assert "llama8b" in live_roster["targets"] and CONSTITUTION in LIVE_ROSTER.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="'llama8b' is already in the shared roster"):
        V.load_roster(REPO)
    with pytest.raises(SystemExit, match="'llama8b' is already pinned in the shared pins"):
        V.load_pins(REPO)
    # ... and a live regeneration would not even reach the rows: the FIRST refusal is the roster's
    with pytest.raises(SystemExit, match="'llama8b' is already in the shared roster"):
        V.load_inputs(REPO, V.ViewsSpec(views=dict(SPEC.views), holds=list(SPEC.holds)))   # the same views, NO frozen_inputs


def test_a_tree_with_no_live_registry_reproduces_the_matrix_from_the_frozen_copies(tmp_path):
    """The generator reads ONLY the frozen copies (+ the overlays and views.yaml): a configs root carrying the lane's files
    and NO shared roster / pins reproduces the committed matrix byte for byte — so no shared-roster commit can move this lane."""
    root = _lane_tree(tmp_path / "tree")
    text, doc = GEN.build(configs_root=root, views_yaml=None, views_spec=None)
    assert text == (REPO / IC.IFEVAL_MATRIX_REL).read_text(encoding="utf-8")
    assert doc["source"]["frozen_inputs"] == SPEC.frozen_inputs
    with pytest.raises(SystemExit, match="no roster at"):                     # the live path has nothing to read here
        V.load_roster(root)


def test_a_mutated_live_registry_does_not_move_the_matrix_the_frozen_input_generator_never_reads_it(tmp_path):
    """The KNOWN-BAD twin: a configs root whose LIVE roster and pins are the pin's bytes MUTATED — a target added, a served cell
    dropped from an expressed view, a pinned object sha re-pointed, a model key added to the pins — reproduces the committed
    matrix byte for byte, while the live path at the SAME root reads every mutation. Then the mutation the live tip actually
    carries (a llama8b target + block): the live path refuses, the frozen path is unmoved."""
    roster = yaml.safe_load((FROZEN_DIR / "sdfbase_full22_roster.yaml").read_text(encoding="utf-8"))
    pins = json.loads((FROZEN_DIR / "sdfbase_full22_pins.json").read_text(encoding="utf-8"))
    gemma_main = roster["targets"]["gemma4e4b"]["views"]["main"]["cells"]
    dropped = "constv2-mix-full__lora_r4_ce"
    assert dropped in gemma_main and _row("gemma4e4b", "main", dropped)["status"] == "launch"
    del gemma_main[dropped]                                                             # a matrix row's cell, gone from the live view
    roster["targets"]["zz_mutant"] = dict(roster["targets"]["gemma4e4b"], model="mutant/model", model_revision="0" * 40)
    staged = _row("qwen35_4b", "main", "constv2ctl2-full__lora_r4_ce")["staged_id"]
    pins["qwen35_4b"][staged]["object_sha256"] = "f" * 64                               # a matrix row's object, re-pointed
    pins["zz_mutant"] = {"base__x": {"object_sha256": "e" * 64}}
    mutated_roster, mutated_pins = yaml.safe_dump(roster, sort_keys=False).encode(), json.dumps(pins, indent=1).encode()
    assert hashlib.sha256(mutated_roster).hexdigest() != FROZEN_ROSTER_SHA and hashlib.sha256(mutated_pins).hexdigest() != FROZEN_PINS_SHA
    root = _lane_tree(tmp_path / "mutated", live_roster=mutated_roster, live_pins=mutated_pins)
    # the live path READS the mutations (they landed where a live-registry generator would look) ...
    live_r, live_r_src = V.load_roster(root)
    live_p, live_p_src = V.load_pins(root)
    assert "zz_mutant" in live_r["targets"] and dropped not in live_r["targets"]["gemma4e4b"]["views"]["main"]["cells"]
    assert live_p["qwen35_4b"][staged]["object_sha256"] == "f" * 64 and "zz_mutant" in live_p
    assert live_r_src["roster_sha256"] != FROZEN_ROSTER_SHA and live_p_src["pins_sha256"] != FROZEN_PINS_SHA
    # ... and the generator does not: byte-identical, every row and the recorded shas the pin's
    text, doc = GEN.build(configs_root=root, views_yaml=None, views_spec=None)
    assert text == (REPO / IC.IFEVAL_MATRIX_REL).read_text(encoding="utf-8")
    assert doc["source"]["roster_sha256"] == FROZEN_ROSTER_SHA and doc["source"]["pins_sha256"] == FROZEN_PINS_SHA
    assert IM.row_for("gemma4e4b", "main", dropped, doc["rows"])["status"] == "launch"
    assert IM.row_for("qwen35_4b", "main", "constv2ctl2-full__lora_r4_ce", doc["rows"])["object_sha256"] != "f" * 64
    # the mutation the live tip carries: the live path refuses by name, the frozen path is unmoved
    root2 = _lane_tree(tmp_path / "shadowed", live_roster=LIVE_ROSTER.read_bytes(), live_pins=LIVE_PINS.read_bytes())
    with pytest.raises(SystemExit, match="'llama8b' is already in the shared roster"):
        V.load_roster(root2)
    text2, _ = GEN.build(configs_root=root2, views_yaml=None, views_spec=None)
    assert text2 == text


def test_an_edited_or_missing_frozen_copy_refuses_by_name_never_falling_back_to_a_live_registry(tmp_path):
    """The copies are IMMUTABLE: one appended byte on either refuses with the declared and the actual sha (restore it with
    git show <pin>:<path>); a missing copy refuses by name. There is no live registry in these trees, so a fallback would
    show up as a different refusal ("no roster at …") — it does not."""
    edited = (FROZEN_DIR / "sdfbase_full22_roster.yaml").read_bytes() + b"\n# one edited byte\n"
    root = _lane_tree(tmp_path / "roster")
    (root / SPEC.frozen_inputs["roster"]).write_bytes(edited)            # the declared sha stays the pin's
    with pytest.raises(SystemExit, match=r"frozen roster .* hashes to [0-9a-f]{64}, not the declared " + FROZEN_ROSTER_SHA + r" .* IMMUTABLE"):
        GEN.build(configs_root=root, views_yaml=None, views_spec=None)
    root = _lane_tree(tmp_path / "pins")
    (root / SPEC.frozen_inputs["pins"]).write_bytes((FROZEN_DIR / "sdfbase_full22_pins.json").read_bytes() + b"\n")
    with pytest.raises(SystemExit, match=r"frozen pins .* hashes to [0-9a-f]{64}, not the declared " + FROZEN_PINS_SHA):
        GEN.build(configs_root=root, views_yaml=None, views_spec=None)
    root = _lane_tree(tmp_path / "missing")
    (root / SPEC.frozen_inputs["roster"]).unlink()
    with pytest.raises(SystemExit, match="no frozen roster at"):
        GEN.build(configs_root=root, views_yaml=None, views_spec=None)


def test_the_overlay_adds_and_never_shadows_a_frozen_copy_that_carries_the_target_refuses(tmp_path):
    """The adds-never-shadows rule is INTACT under frozen inputs — the remedy is the pin's bytes, not a relaxed rule: freezing
    the LIVE registry (which names llama8b with the SAME model + revision since the Llama wave) in place of the pin's copy,
    with its sha declared, makes the roster overlay refuse exactly as at the live tip; the pins twin (same object shas)
    likewise."""
    live_roster = yaml.safe_load(LIVE_ROSTER.read_text(encoding="utf-8"))["targets"]["llama8b"]
    overlay = yaml.safe_load((REPO / LANE_FILES[1]).read_text(encoding="utf-8"))["targets"]["llama8b"]
    assert (live_roster["model"], live_roster["model_revision"]) == (overlay["model"], overlay["model_revision"])   # SAME identity — still refused
    root = _lane_tree(tmp_path / "roster", frozen_roster=LIVE_ROSTER.read_bytes())
    with pytest.raises(SystemExit, match="'llama8b' is already in the shared roster — the overlay adds a target, never shadows one"):
        GEN.build(configs_root=root, views_yaml=None, views_spec=None)
    root = _lane_tree(tmp_path / "pins", frozen_pins=LIVE_PINS.read_bytes())
    with pytest.raises(SystemExit, match="'llama8b' is already pinned in the shared pins"):
        GEN.build(configs_root=root, views_yaml=None, views_spec=None)


def test_a_spec_without_frozen_inputs_reads_the_live_registry_and_records_a_null(tmp_path):
    """The --views CLI form (a peer's tree, no views.yaml) reads THAT tree's live registry, as before; the source block
    records `frozen_inputs: null` and the live file's sha."""
    root = tmp_path / "peer"
    for rel, src in ((LIVE_ROSTER_REL, FROZEN_DIR / "sdfbase_full22_roster.yaml"), (LIVE_PINS_REL, FROZEN_DIR / "sdfbase_full22_pins.json"),
                     (LANE_FILES[1], REPO / LANE_FILES[1]), (LANE_FILES[2], REPO / LANE_FILES[2])):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(src.read_bytes())
    assert V.frozen_inputs_of(V.ViewsSpec(views={"gemma4e4b": ["main"]}), root) is None
    text, doc = GEN.build(configs_root=root, views_yaml=None, views_spec="gemma4e4b=main")
    assert doc["source"]["frozen_inputs"] is None and doc["source"]["views_yaml"] is None
    assert doc["source"]["roster_sha256"] == FROZEN_ROSTER_SHA and doc["source"]["roster"] == LIVE_ROSTER_REL
    assert [r["model_key"] for r in doc["rows"]] == ["gemma4e4b"] * len(doc["rows"]) and doc["counts"]["rows"] > 0


# --------------------------------------------------------------------------- the seed sibling document
SEED_VIEWS_REL = IC.IFEVAL_VIEWS_QWEN_SEEDREP_REL
SEED_MATRIX_REL = IC.IFEVAL_MATRIX_QWEN_SEEDREP_REL
SEED_SPEC = V.load_views_yaml(REPO / SEED_VIEWS_REL)
SEED_POLICY = V.load_ifeval_policy(REPO / SEED_VIEWS_REL)
SEED_DOC = IM.load_matrix(REPO / SEED_MATRIX_REL)
SEED_ROWS = SEED_DOC["rows"]
SEED_PIN = "7b71cb2d"
ARDITI_PINS_REL = "configs/arditi_allpos_pins.json"
COMPOSED = "constv2-mix-full__psr_ce_all_minus_ctl"
EIGHT_ARMS_ROSTER_ORDER = ["none", "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all",
                           "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal",
                           CONSTITUTION]


def test_the_seed_sibling_document_regenerates_byte_identically_with_30_launch_rows_and_every_contrast_resolved():
    """24 seed rows + 3 composed + 3 Arditi = 30 rows, 0 held, 0 deferred, 21 fitted, native serving; the
    ReLU arms IN (`exclude: []`); the paired views' composed rows anchor on s<s>/none through `anchor_view`; the frozen 40-row
    document and its overlays are byte-identical beside it (the two `--check`s above and here)."""
    r = subprocess.run([sys.executable, str(IFE / "make_ifeval_matrix.py"), "--views-yaml", str(REPO / SEED_VIEWS_REL),
                        "--out", str(REPO / SEED_MATRIX_REL), "--check", "--table"], capture_output=True, text=True, cwd=REPO, env=ENV, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout and "30 rows / 0 held / 30 launched / 0 deferred" in r.stdout and "27/27 cell_minus_none legs resolve" in r.stdout
    assert "HELD:" not in r.stdout and "DEFERRED:" not in r.stdout
    # the (view, cell) table the launch-set derivation pins by value: per seed view the roster's eight arms in roster order,
    # then the three composed rows, then the three Arditi rows (30 lines)
    table = [tuple(line.split("\t")[1:]) for line in r.stdout.splitlines() if line.startswith("TABLE\t")]
    want = [(v, c) for v in ("s1", "s2", "s3") for c in EIGHT_ARMS_ROSTER_ORDER] + [(f"s{i}p", COMPOSED) for i in (1, 2, 3)] \
        + [(f"s{i}", "arditi_allpos") for i in (1, 2, 3)]
    assert table == want and len(table) == 30
    text, doc = GEN.build(configs_root=REPO, views_yaml=REPO / SEED_VIEWS_REL, views_spec=None)
    assert text == (REPO / SEED_MATRIX_REL).read_text(encoding="utf-8")
    c = doc["counts"]
    assert (c["rows"], c["held"], c["launched"], c["deferred"], c["fitted"], c["fitted_launched"], c["in_support"]) == (30, 0, 30, 0, 21, 21, 9)
    assert c["prompted_constitution"] == 3 and c["prompted_safety"] == 3 and c["header_located"] == 0 and c["declared"] == 0
    assert c["tags"] == [f"qwen35_4b_{v}" for v in IC.IFEVAL_LANE_VIEWS] and c["n_slugs"] == 6 and c["distinct_out_roots"] == 30
    assert c["by_model_view"] == {f"qwen35_4b/{v}": {"launched": (8 + 1 if v in ("s1", "s2", "s3") else 1), "held": 0, "deferred": 0}
                                  for v in IC.IFEVAL_LANE_VIEWS}
    assert doc["slugs"] == {f"qwen35_4b-sdfbase-{v}": {"model_key": "qwen35_4b", "view": v} for v in IC.IFEVAL_LANE_VIEWS}
    assert doc["serving_path"] == "lora_native" and doc["policy"]["serving_path"] == "lora_native"
    assert doc["policy"]["exclude"] == [] and doc["policy"]["prompted_extra"] == [] and doc["policy"]["deferred"] == []
    assert doc["policy"]["anchor_view"] == {"s1p": "s1", "s2p": "s2", "s3p": "s3"}
    assert doc["policy"]["view_cells"] == {f"s{i}p": [COMPOSED] for i in (1, 2, 3)}
    assert [e["view"] for e in doc["policy"]["lane_extra"]] == ["s1", "s2", "s3"] and {e["cell"] for e in doc["policy"]["lane_extra"]} == {"arditi_allpos"}
    assert doc["holds"] == [] and doc["views"] == {"qwen35_4b": list(IC.IFEVAL_LANE_VIEWS)}
    # the same instrument as the frozen document: caps, trigger, seed rule, sampling, prompt classes
    for k in ("task", "control", "n_items", "samples_per_item", "thinking", "sampling_mode", "caps", "trigger_threshold",
              "trigger_rule", "trigger_provenance", "seed_rule", "prompt_classes"):
        assert doc[k] == DOC[k], k
    # the contrasts: every composed leg resolves against its SEED view's none (recorded on the leg); no instrument floor
    legs = doc["contrasts"][IC.CONTRAST_VS_NONE]["legs"]
    assert [l for l in legs if l.get("anchor_view")] == [{"model_key": "qwen35_4b", "view": f"s{i}p", "cell": COMPOSED,
                                                          "anchor_view": f"s{i}", "anchor": True} for i in (1, 2, 3)]
    assert doc["contrasts"][IC.CONTRAST_NOISE] == {"models_with_floor": [], "models_without_floor": ["qwen35_4b"]}
    # the source: the frozen copies at the seed base, NO Llama overlays (the document expresses no llama8b)
    src = doc["source"]
    assert src["frozen_inputs"] == SEED_SPEC.frozen_inputs and src["roster_overlay"] is None and src["pins_overlay"] is None
    assert src["frozen_inputs"]["roster"] == f"configs/sdfbase/ifeval/frozen/{SEED_PIN}/sdfbase_full22_roster.yaml"
    assert src["roster_sha256"] == V.sha256_of(REPO / src["frozen_inputs"]["roster"]) == SEED_SPEC.frozen_inputs["roster_sha256"]
    assert src["pins_sha256"] == V.sha256_of(REPO / src["frozen_inputs"]["pins"]) == SEED_SPEC.frozen_inputs["pins_sha256"]
    assert src["roster_sha256"] != FROZEN_ROSTER_SHA and src["pins_sha256"] != FROZEN_PINS_SHA      # a later pin than the 40-row document's
    # the frozen 40-row document is untouched beside it
    assert (REPO / IC.IFEVAL_MATRIX_REL).read_text(encoding="utf-8") == GEN.build(configs_root=REPO, views_yaml=None, views_spec=None)[0]


def test_every_seed_row_is_native_pinned_from_its_own_pins_file_and_carries_the_lane_grammar():
    frozen_pins = json.loads((REPO / SEED_SPEC.frozen_inputs["pins"]).read_text(encoding="utf-8"))["qwen35_4b"]
    arditi_pins = json.loads((REPO / ARDITI_PINS_REL).read_text(encoding="utf-8"))["qwen35_4b"]
    for r in SEED_ROWS:
        assert r["serving_path"] == "lora_native" and r["status"] == "launch" and r["held_reason"] is None and r["deferred"] is None
        assert r["model_key"] == "qwen35_4b" and r["view"] in IC.IFEVAL_LANE_VIEWS and r["tag"] == f"qwen35_4b_{r['view']}"
        assert r["slug"] == IC.IFEVAL_SLUGS[("qwen35_4b", r["view"])] == C.model_row(r["slug"]).slug
        assert r["declared"] is False and r["thinking"] == "off" and r["sampling_mode"] == "vendor_preset" and r["samples_per_item"] == 2
        assert r["config"] == f"configs/sdfbase/ifeval/sdfbase_full22_{r['tag']}.ifeval.yaml" and (REPO / r["config"]).is_file()
        assert (REPO / r["manifest"]).is_file() and r["hf"] == "Qwen/Qwen3.5-4B" and r["family"] == "qwen3_5" and r["image"] == "default"
        if r["cell"] == "arditi_allpos":
            assert r["pin"]["kind"] == "lane_extra" and r["pin"]["source"] == ARDITI_PINS_REL
            assert r["pin"]["source_sha256"] == V.sha256_of(REPO / ARDITI_PINS_REL)
            assert r["staged_id"] == "ar_c0p3__arditi_allpos" and r["object_sha256"] == arditi_pins["ar_c0p3__arditi_allpos"]["object_sha256"]
            assert r["object_sha256"].startswith("ba2212f0") and r["object_source"] == "external" and r["bank_view"] == "ar_c0p3"
            assert r["lane_extra"]["sidecar"] == "fit_provenance.json" and r["lane_extra"]["bank"] == arditi_pins["ar_c0p3__arditi_allpos"]["view"]
            assert r["fitted"] and r["route_kind"] == "lens" and r["served_positions_expected"] == "all" and r["prompt_class"] == "short"
            assert r["view"] in ("s1", "s2", "s3") and r["index"] >= 27
        elif r["fitted"]:
            assert r["pin"]["kind"] == "roster" and r["pin"]["source"] == "configs/sdfbase_full22_pins.json"
            assert r["pin"]["source_sha256"] == SEED_SPEC.frozen_inputs["pins_sha256"]
            assert r["object_sha256"] == frozen_pins[r["staged_id"]]["object_sha256"] == r["pin"]["object_sha256"]
            assert "object_source" not in r and "lane_extra" not in r
            seed = r["view"][1]
            assert seed in r["staged_id"], (r["staged_id"], r["view"])      # base64s<s>__ / derived_base64s<s>__ / paired_eb64s<s>__
        else:
            assert r["pin"]["kind"] in ("none", "prompted") and r["object_sha256"] is None and r["staged_id"] is None
    # the paired views: the composed cell alone, no own none, the paired object's pin
    for i in (1, 2, 3):
        prs = [r for r in SEED_ROWS if r["view"] == f"s{i}p"]
        assert [r["cell"] for r in prs] == [COMPOSED] and prs[0]["staged_id"] == f"paired_eb64s{i}__{COMPOSED}"
    # 21 fitted rows carry 21 pins that all sit in the contracts' intervention-hash inventory
    fitted = [r for r in SEED_ROWS if r["fitted"]]
    assert len(fitted) == 21
    for r in fitted:
        assert C.expected_intervention_hash(r["slug"], r["cell"]) is not None, (r["slug"], r["cell"])
    # every row's pin is in the frozen pins copy or the Arditi pins file; the frozen 40-row document's Qwen pins are not these
    frozen40_qwen = {r["object_sha256"] for r in ROWS if r["model_key"] == "qwen35_4b" and r["object_sha256"]}
    assert not ({r["object_sha256"] for r in fitted} & frozen40_qwen)
    # the stdlib reader reads it: the registered documents are exactly the two
    assert IM.MATRIX_PATHS == (IM.MATRIX_PATH, IM.MATRIX_QWEN_SEEDREP_PATH)
    assert [str(p.relative_to(REPO)) for p in IM.MATRIX_PATHS] == list(C.IFEVAL_MATRIX_DOCUMENTS)
    assert len(IM.matrix_rows(IM.MATRIX_QWEN_SEEDREP_PATH)) == 30 and len(IM.matrix_rows()) == 40 and IM.counts(SEED_ROWS)["rows"] == 30
    IM.refuse_held(SEED_ROWS, flag="--rows")                                            # nothing held, nothing deferred
    merged = C.load_ifeval_matrices(REPO)
    assert len(merged["rows"]) == 70 and merged["documents"] == list(C.IFEVAL_MATRIX_DOCUMENTS)
    assert C.ifeval_hold_problem(merged, slug="qwen35_4b-sdfbase-s1", cell="arditi_allpos") is None
    assert C.ifeval_matrix_row_status(merged, slug="qwen35_4b-sdfbase-s1p", cell=COMPOSED) == ("launch", None)
    assert C.ifeval_matrix_row_status(merged, slug="qwen35_4b-sdfbase-s1p", cell="none") == (None, None)   # no own none: no row


def test_the_seed_document_grammar_is_validated_and_the_llama_overlays_are_not_merged(tmp_path):
    """The four blocks refuse ill-formed entries by name; the driver argv for a seed row carries --bank-view <seed view>
    (the served bank dir) and the pin; a document without llama8b reads its frozen copies WITHOUT the Llama overlays."""
    def _doc(**over):
        d = yaml.safe_load((REPO / SEED_VIEWS_REL).read_text(encoding="utf-8"))
        d.update(over)
        pth = tmp_path / "v.yaml"
        pth.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
        return pth
    with pytest.raises(SystemExit, match="anchor_view"):
        V.load_ifeval_policy(_doc(anchor_view={"s1p": "s1p"}))
    with pytest.raises(SystemExit, match="anchor_view"):
        V.load_ifeval_policy(_doc(anchor_view={"s1p": "ar_c0p3"}))
    with pytest.raises(SystemExit, match="view_cells"):
        V.load_ifeval_policy(_doc(view_cells={"s1p": []}))
    with pytest.raises(SystemExit, match="serving_path"):
        V.load_ifeval_policy(_doc(serving_path="vllm"))
    bad = [dict(SEED_POLICY.lane_extra[0])]; del bad[0]["bank"]
    with pytest.raises(SystemExit, match="lane_extra"):
        V.load_ifeval_policy(_doc(lane_extra=bad))
    with pytest.raises(SystemExit, match="bare fitted Class B"):
        V.load_ifeval_policy(_doc(lane_extra=[{**SEED_POLICY.lane_extra[0], "cell": "constv2-mix-full__lora_r4_ce"}]))
    # a view_cells cell outside the enumeration, a lane_extra on an unexpressed view, a duplicate row: refused at select time
    roster, pins, source = V.load_inputs(REPO, SEED_SPEC)
    assert source["roster_overlay"] is None and "llama8b" in roster["targets"]        # the base copy carries llama8b; no overlay merged
    pol = V.load_ifeval_policy(_doc(view_cells={"s1p": ["constv2ctl2-full__lora_r4_ce"]}))
    with pytest.raises(SystemExit, match="not in the view's enumeration"):
        V.select_rows(roster, pins, SEED_SPEC.views, policy=pol, pins_source=source, configs_root=REPO)
    pol = V.load_ifeval_policy(_doc(lane_extra=[{**SEED_POLICY.lane_extra[0], "view": "it"}]))
    with pytest.raises(SystemExit, match="not a view this document expresses"):
        V.select_rows(roster, pins, SEED_SPEC.views, policy=pol, pins_source=source, configs_root=REPO)
    pol = V.load_ifeval_policy(_doc(lane_extra=[SEED_POLICY.lane_extra[0], SEED_POLICY.lane_extra[0]]))
    with pytest.raises(SystemExit, match="already a row"):
        V.select_rows(roster, pins, SEED_SPEC.views, policy=pol, pins_source=source, configs_root=REPO)
    pol = V.load_ifeval_policy(_doc(lane_extra=[{**SEED_POLICY.lane_extra[0], "bank": "ar_c0p5"}]))
    with pytest.raises(SystemExit, match="records view='ar_c0p3'"):
        V.select_rows(roster, pins, SEED_SPEC.views, policy=pol, pins_source=source, configs_root=REPO)
    with pytest.raises(SystemExit, match="needs the configs root"):
        V.select_rows(roster, pins, SEED_SPEC.views, policy=SEED_POLICY, pins_source=source)
    # a composed row without its anchor view's none fails the contrast check by name
    rows = V.select_rows(roster, pins, SEED_SPEC.views, policy=SEED_POLICY, pins_source=source, configs_root=REPO)
    no_s1_none = [r for r in rows if not (r["view"] == "s1" and r["cell"] == "none")]
    with pytest.raises(SystemExit, match="s1p/constv2-mix-full__psr_ce_all_minus_ctl has no launched none anchor \\(anchor view s1\\)"):
        V.check_contrasts(no_s1_none, anchor_view=SEED_POLICY.anchor_view)
    with pytest.raises(SystemExit, match="has no launched none anchor"):
        V.check_contrasts(rows)                                                         # without the map the paired rows are orphans
    # the driver argv of a seed row: the seed view as --bank-view, the pin, no @thinking, --attempt-id last
    row = IM.row_for("qwen35_4b", "s2", "arditi_allpos", SEED_ROWS)
    argv = V.driver_argv_for_row(row, rung=2048, python="py", script="drv", proxy_url="u", routing_table="rt", ledger_url="l",
                                 defenses_dir="d", out_root="o", budget="b", concurrency=4, vllm_log="v", attempt_id="a1",
                                 judge_model="j", top_logprobs=1)
    assert argv[argv.index("--bank-view") + 1] == "s2" and argv[argv.index("--expect-object-sha256") + 1] == row["object_sha256"]
    assert argv[-2:] == ["--attempt-id", "a1"] and "--inspect-seed-policy" not in argv and "@thinking" not in " ".join(argv)
    # the frozen document's policy record is unchanged by the new grammar (no new keys when nothing is declared)
    assert set(V.load_ifeval_policy(V.views_path(REPO)).as_record()) == {"exclude", "prompted_extra", "deferred"}

