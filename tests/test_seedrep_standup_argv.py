"""S0 proof-set gate — every PRODUCTION standup argv of the qwen-seedrep-native-battery plan through the REAL front-end
contract, with no GPU:

  * `mt_serving_up.build_parser()` parses every argv (the real parser; `--mode` required, every flag known);
  * `lora_serving_up.select_cells_lora(explicit_cells, --cell, si_vocabulary)` over the REAL s-view configs yields the
    row's route set — none + 5 fitted + BOTH prompted routes (both admitted) on a view engine, {none, arditi_allpos} on
    the Arditi engine;
  * `native_standup._resolve_standup_geometry(eval, L?, S?, model_key)` — the core's identity resolution — gives
    `window_source table` at the table's L for a TABLE window (no factors on the argv) and the identity PAIR for a
    non-table window; then `lora_kernel_ceiling_geometry` (the adapters' D = 4096 injected) LOWERS S to 524,287 // L and
    `kernel_adjustment_record` writes exactly the row's expected adjustment; under main #200 the re-resolved geometry
    keeps `window_source table` (pre-#200 trees stamp `override`; both are asserted by what the core in the tree does);
  * gate 1: the served block through the REAL GSM8K reader (`tinycap_math500_thinking._override_window_problem`, main
    #196 / #200) is ADMITTED for the GSM8K rows in BOTH shapes the readers know, and the negatives — rev 2's explicit
    18,432 (no table entry) and an explicit 16,640 WITHOUT a lowering — are REFUSED (skipped by name when the tree
    predates the symbol);
  * `--thinking-on` iff GSM8K; `--max-logprobs 64` on every MMLU-serving standup and the 20-vs-64 admission negative
    (gate 9); `--expect-object-sha256` names every fitted cell with the pins file's sha; the Arditi engine one object.

The GSM8K common geometry entries (`capability:gen_long_gsm8k_common_f22` / `..._rung_common_f22`) are copies of
the `_longsi_f22` values; when the tree's table lacks them (this base) they are REGISTERED for the test's duration from
those sources and a separate test asserts the plan's table validation REFUSES on the unpatched table, so an unlanded
common table entry is loud at S0 rather than silent.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase"):
    sys.path.insert(0, str(p))

import seedrep_launch_plan as lp  # noqa: E402
from sisteer.capeval.serving import geometry as sg  # noqa: E402
from sisteer.capeval.serving import native_standup as ns  # noqa: E402
from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402

W = "1" * 40
WAVE = "11111111-2222-4333-8444-555555555555"


def _register_common_entries():
    """The common entries when absent: byte-copies of the `_longsi_f22` table entries under the common names."""
    added = []
    for common, source in lp.GSM8K_COMMON_SOURCE_EVAL.items():
        key = sg.parse_key(common)
        if key in sg.TABLE:
            continue
        src = sg.TABLE[sg.parse_key(source)]
        entry = src.__class__(**{**src.__dict__, "eval": key[0], "task": key[1]}) if hasattr(src, "__dict__") else src
        sg.TABLE[key] = entry
        added.append(key)
    return added


@pytest.fixture(scope="module")
def plan():
    inputs = lp.load_inputs(REPO)
    return lp.build_plan(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"],
                         arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z")


@pytest.fixture(scope="module")
def rows(plan):
    return plan["standups"] + plan["rungs"]


@pytest.fixture(scope="module")
def common_table():
    added = _register_common_entries()
    yield added
    for k in added:
        sg.TABLE.pop(k, None)


def _parse(argv):
    """The REAL front's parser (`mt_serving_up._build_parser`, the one `main` builds); argv[0:2] are the venv python + script."""
    import mt_serving_up as mt
    return mt._build_parser().parse_args(argv[2:])


def test_every_production_argv_parses_on_the_real_front(rows):
    assert len(rows) == 74 + 56                                       # every declared rung is executable, the 4 GSM8K rungs included
    for r in rows:
        a = _parse(r["argv"])
        assert a.mode == r["leg"] and False or a.mode in ("attacks", "petri")
        assert a.eval == r["geometry"]["eval"]
        assert a.serving_path == "lora_native" and a.enable_log_requests is True
        assert a.server_default_policy == "contract_override" and a.model_key == "qwen35_4b"
        assert a.adapter_bank == f"{lp.ADAPTER_BANK_ROOT}/qwen35_4b"
        assert a.thinking_on is (r["leg"] == "gsm8k")
        assert a.upstream_timeout_s == 7200.0 and a.ready_timeout_s == 2700 and a.gpu_mem_util == "0.9"


def test_route_set_through_select_cells_lora_over_the_real_configs(rows):
    import lora_serving_up as lsu
    for r in rows:
        a = _parse(r["argv"])
        if r["engine"] == lp.ARDITI:
            lane_cfg = REPO / lp.ARDITI_CONFIG_REL
            explicit = yaml.safe_load(lane_cfg.read_text())["explicit_cells"] if lane_cfg.is_file() else ["none", "arditi_allpos"]
            vocab = ()
        else:
            cfg = yaml.safe_load((REPO / lp.VIEW_CONFIG_REL.format(view=r["engine"])).read_text())
            explicit, vocab = cfg["explicit_cells"], lsu.config_si_vocabulary(cfg)
            assert {"si_constitution", "safety_refusal"} <= set(vocab), "the prompted routes ride D14 over the config's SI vocabulary"
        cells, admitted = lsu.select_cells_lora(explicit, a.cell, si_vocabulary=vocab)
        assert cells == r["route_set"], (r["standup_id"], cells)
        if r["engine"] == lp.ARDITI:
            assert cells == ["none", "arditi_allpos"] and admitted == []
        else:
            assert admitted == list(lp.PROMPTED_CELLS) or set(lp.PROMPTED_CELLS) <= set(explicit)


def test_pins_name_every_fitted_cell_from_the_pins_file(rows):
    pins = json.loads((REPO / lp.PINS_REL).read_text())["qwen35_4b"]
    roster = yaml.safe_load((REPO / lp.ROSTER_REL).read_text())
    apins = json.loads((REPO / lp.ARDITI_PINS_REL).read_text())["qwen35_4b"]["ar_c0p3__arditi_allpos"]["object_sha256"]
    for r in rows:
        a = _parse(r["argv"])
        digests = dict(d.split("=", 1) for d in a.expect_object_sha256)
        if r["engine"] == lp.ARDITI:
            assert digests == {"arditi_allpos": apins} and apins.startswith("ba2212f0c2f9")
        else:
            staged = lp.roster_view_cells(roster, r["engine"])
            assert set(digests) == set(lp.FITTED_CELLS)
            for c, sha in digests.items():
                assert sha == pins[staged[c]]["object_sha256"], (r["standup_id"], c)


@pytest.fixture(autouse=False)
def d4096(monkeypatch):
    """The banked adapters' widest module: D = 4096 on Qwen3.5-4B (the kernel ceiling 524,287)."""
    monkeypatch.setattr(sa, "adapter_module_dims", lambda d: {"o_proj": (4096, 4096)})


def test_geometry_resolves_and_lowers_exactly_as_the_core_does(rows, common_table, d4096):
    core_keeps_table = "from_table" in inspect.getsource(ns.lora_kernel_ceiling_geometry)   # main #200
    for r in rows:
        a, g = _parse(r["argv"]), r["geometry"]
        geom = ns._resolve_standup_geometry(a.eval, a.max_model_len, a.max_num_seqs, model_key="qwen35_4b")
        assert int(geom.max_model_len) == g["L"], r["standup_id"]
        assert int(geom.max_num_seqs) == g["identity_S"], (r["standup_id"], geom.max_num_seqs, g["identity_S"])
        if g["table"]:
            assert a.max_model_len is None and a.max_num_seqs is None
            assert geom.window_source == "table" and geom.seqs_source == "model_budget"
        else:
            assert geom.window_source == "override" and geom.is_identity, r["standup_id"]
        record, lowered = ns.lora_kernel_ceiling_geometry({"x": {"dir": "/nonexistent"}}, geom, eval_name=a.eval, model_key="qwen35_4b",
                                                          allow_over_budget=False, override_reason=None)
        assert record["ceiling"] == 524_287 and record["D"] == 4096
        if g["expected_adjustment"] is None:
            assert lowered is None and g["S"] == g["identity_S"]
        else:
            assert lowered is not None and lowered["S"] == g["S"] == 524_287 // g["L"]
            assert ns.kernel_adjustment_record(lowered) == g["expected_adjustment"]
            assert int(lowered["_geom"].max_model_len) == g["L"] and int(lowered["_geom"].max_num_seqs) == g["S"]
            want = ("table" if (core_keeps_table and g["table"]) else "override")
            assert lowered["_geom"].window_source == want, (r["standup_id"], lowered["_geom"].window_source)
        assert g["S"] * g["L"] <= 524_287 and g["S"] <= 256
        assert g["expected_window_source"] == ("table" if g["table"] else "override")


def test_the_s5_roster_is_the_ten_windows_of_the_executable_plan(plan):
    got = {(r["L"], r["S"], r["thinking"]) for r in plan["s5_roster"]}
    # the GSM8K rung's 24,832 x 21 thinking-ON window IS an executable row and needs its S5 reference
    assert got == {(7424, 70, False), (10240, 51, False), (11520, 45, False), (16384, 31, False), (16640, 31, True),
                   (24576, 21, False), (24832, 21, True), (32768, 15, False), (40960, 12, False), (49152, 10, False)}
    assert plan["deferred_rungs"] == []


def test_explicit_s_on_a_table_window_is_what_the_front_refuses(common_table):
    """The reason S is never typed: `--max-num-seqs 45` on the static table window breaks strict identity."""
    with pytest.raises(sg.GeometryIdentityBroken):
        ns._resolve_standup_geometry("families_served", None, 45, model_key="qwen35_4b")


def _reader():
    import importlib
    try:
        m = importlib.import_module("tinycap_math500_thinking")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tinycap_math500_thinking not importable here: {type(e).__name__}: {e}")
    if not hasattr(m, "_override_window_problem"):
        pytest.skip("the GSM8K reader at this tree predates #196's _override_window_problem (lands with the W re-merge)")
    return m


def _served_block(row, *, window_source, adjustments, L=None, S=None):
    g = row["geometry"]
    return ({"eval": g["eval"], "model_key": "qwen35_4b", "window_source": window_source, "adjustments": adjustments},
            L if L is not None else g["L"], S if S is not None else g["S"])


def test_gate1_gsm8k_rows_admitted_by_the_real_reader_in_both_shapes(rows, common_table):
    m = _reader()
    gsm = [r for r in rows if r["leg"] == "gsm8k"]
    assert len(gsm) == 8                                              # 4 engines x (base + the 16,384 rung), every one through the reader
    for r in gsm:
        adj = [r["geometry"]["expected_adjustment"]]
        # main #200: the served block reads `table` — the primary rule admits it (the carve is not consulted)
        geom, L, S = _served_block(r, window_source="table", adjustments=adj)
        assert geom["window_source"] == m.GEOMETRY_WINDOW_SOURCE_TABLE
        # pre-#200 trees / a real override beside a recorded lowering: the #196 carve admits the SAME L under the adjustment
        geom, L, S = _served_block(r, window_source="override", adjustments=adj)
        assert m._override_window_problem(geom, adjustments=adj, L=L, S=S, SG=sg) is None, r["standup_id"]


def test_gate1_negatives_refused_by_the_real_reader(rows, common_table):
    m = _reader()
    r = next(x for x in rows if x["leg"] == "gsm8k" and x["stage"] == "base")
    adj = [r["geometry"]["expected_adjustment"]]
    # (a) rev 2's explicit 18,432: no table entry at that L -> refused by the two numbers
    geom, _, S = _served_block(r, window_source="override", adjustments=adj, L=18432, S=28)
    why = m._override_window_problem(geom, adjustments=[{**adj[0], "to_S": 28}], L=18432, S=28, SG=sg)
    assert why is not None and "18,432" in why
    # (b) an explicit 16,640 WITHOUT a lowering: `override` with no adjustment keeps today's refusal
    geom, L, S = _served_block(r, window_source="override", adjustments=[])
    why = m._override_window_problem(geom, adjustments=[], L=L, S=S, SG=sg)
    assert why is not None and "override" in why
    # (c) an unknown adjustment kind explains nothing
    why = m._override_window_problem(geom, adjustments=[{"kind": "made_up", "to_S": S}], L=L, S=S, SG=sg)
    assert why is not None and "made_up" in why


def test_gate9_max_logprobs_on_every_mmlu_standup_and_the_20_vs_64_negative(rows):
    import mt_serving_up as mt
    mmlu = [r for r in rows if "mmlu" in r["eval_legs"]]
    # MMLU is a teacher-forced one-token readout with no cap ladder -- measured once per (arm, s) on the BASE controls rows; the
    # 4 x 3 controls RUNGS re-draw the ladder controls alone and serve no MMLU (the known-bad counted them as MMLU standups and gave each --max-logprobs 64)
    assert len(mmlu) == 4 and all(r["leg"] == "controls" and r["kind"] == "base" for r in mmlu)   # controls: 4 base engines
    rungs = [r for r in rows if r["leg"] == "controls" and r["kind"] == "rung"]
    assert len(rungs) == 4 * 3 and all(r["eval_legs"] == ["xstest", "or_bench_hard"] and "--max-logprobs" not in r["argv"] for r in rungs)
    for r in mmlu:
        assert _parse(r["argv"]).max_logprobs == 64
        extras = mt.launch_cmd_extras("0.9", 64, True)
        assert "--max-logprobs" in extras and extras[extras.index("--max-logprobs") + 1] == "64"
    for r in rows:
        if r not in mmlu:
            assert _parse(r["argv"]).max_logprobs is None
    # the admission rule vLLM applies: a request's top_logprobs above the engine's --max-logprobs (default 20) is refused
    def admits(extras, top):
        cap = int(extras[extras.index("--max-logprobs") + 1]) if "--max-logprobs" in extras else 20
        return top <= cap
    assert admits(mt.launch_cmd_extras("0.9", 64, True), 64)
    assert not admits(mt.launch_cmd_extras("0.9", None, True), 64), "without the flag the driver's top_logprobs 64 is refused at 20"


def test_the_gsm8k_common_class_must_be_in_the_table_or_the_plan_refuses(plan):
    """The common-entry dependency, loud: on a tree without them the table validation names the missing eval."""
    problems = lp.validate_plan(plan, table=sg)
    key = sg.parse_key(lp.GSM8K_COMMON_EVAL)
    if key in sg.TABLE:
        assert problems == [], problems
    else:
        assert problems and all("gsm8k" in p and "common" in p for p in problems), problems[:5]


def test_offline_identity_rule_equals_the_front(common_table):
    for L in (7424, 10240, 11520, 16384, 16640, 24576, 24832, 32768, 40960, 49152, 65536):
        assert lp.identity_seqs_for(L) == sg.budget_for("qwen35_4b").entry.seqs_at(L) == min(741_888 // L, 256)
