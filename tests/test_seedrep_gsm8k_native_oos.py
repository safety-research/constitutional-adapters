"""The native battery's GSM8K rows run under an authorised out-of-support tuple of their OWN,
`gsm8k_paired200_qwen35_4b_native_<cap>`, beside the untouched lens-era pair.

The defect (every fitted GSM8K child of the native standups, S3 / S5 gsm8k s1): the driver's identity for a standup
on the native multi-LoRA / residual-slot engine (`mt_serving_up --eval lora_native`) records `reduction_mode =
not_applicable` (its own log line: "the served identity records reduction_mode='not_applicable' (no lens hook;
SISTEER_LENS_STACKED_REDUCTIONS not_consulted)"), and the ONLY registry tuples for the lane cells pinned the lens-era
`stacked` -- so `check_out_of_support_thinking_authorised` refused `reduction_mode differs -- requested
'not_applicable', authorised 'stacked'`, and the child exited OOS_UNAUTHORISED before a store existed.

The identity those children ran (their recorded argv, fields only -- no ids): model qwen35_4b (the per-cell overlay
config `sdfbase_full22_qwen35_4b_s<s>.gsm8k_<cell>.yaml`), one of the five fitted lane cells, `--controls gsm8k`,
`--thinking on`, `--sampling thinking_card`, `--samples 1`, `--max-tokens-by-task gsm8k_paired200=8192` (the plan's
BASE row; the rung states 16384), `--allow-out-of-support-thinking`, `--expect-object-sha256 <pin>`, and the standup's
served reduction_mode `not_applicable`. Known-bad before the fix: the checker refused that identity by the message
above; green now: it resolves `gsm8k_paired200_qwen35_4b_native_8192`, and the lens-era tuple STILL refuses it with the
same words (pinned here, so the fix is a second door, never a widened one).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "modal"):
    sys.path.insert(0, str(p))

import seedrep_launch_plan as lp  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

W = "2" * 40
WAVE = "22222222-3333-4444-8444-555555555555"
MODEL = "qwen35_4b"
LENS_ERA = "gsm8k_paired200_qwen35_4b_{cap}"
NATIVE = "gsm8k_paired200_qwen35_4b_native_{cap}"
PAIRED = "gsm8k_paired200_qwen35_4b_native_paired_{cap}"          # the composed paired-write arm's own pair
PAIRED_CELL = "constv2-mix-full__psr_ce_all_minus_ctl"
#: the standup's served identity and the exact refusal every fitted child logged
W19_REDUCTION_MODE = "not_applicable"
W19_REFUSAL = "reduction_mode differs — requested 'not_applicable', authorised 'stacked'"
W19_BASE_CAP, RUNG_CAP = 8192, 16384


@pytest.fixture(scope="module")
def inputs():
    return lp.load_inputs(REPO)


@pytest.fixture(scope="module")
def draft(inputs):
    return lp.build_plan(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"],
                         arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z")


def _w19(cell: str, cap: int = W19_BASE_CAP, **over):
    """The fitted child's identity as the driver hands it to the checker (8 fields + the stated cap)."""
    req = dict(model=MODEL, cell=cell, controls=["gsm8k"], mode="on", sampling_mode="thinking_card", samples=1,
               reduction_mode=W19_REDUCTION_MODE, max_tokens_by_task={"gsm8k_paired200": cap})
    req.update(over)
    return req


def _gsm8k_rows(draft):
    return [r for coll in ("standups", "rungs") for r in draft[coll] if r["leg"] == "gsm8k"]


# --------------------------------------------------------------------------- the registry (contracts.py)


def test_the_native_battery_has_its_own_gsm8k_pair_and_the_lens_era_pair_did_not_move():
    names = [C.g8k_native_tuple_name(MODEL, cap) for cap in C.G8K_CAPS]
    assert names == [NATIVE.format(cap=8192), NATIVE.format(cap=16384)]
    assert C.G8K_NATIVE_MODELS == (MODEL,) == (lp.MODEL_KEY,) and C.G8K_NATIVE_TUPLE_INFIX == "_native"
    # the five fitted lane cells, exactly the plan's -- five of the six all-position cells; the distilled one is NOT in the lane
    assert C.G8K_NATIVE_CELLS == lp.FITTED_CELLS and len(C.G8K_NATIVE_CELLS) == 5
    assert set(C.G8K_NATIVE_CELLS) < set(C.G8K_ALL_POSITION_CELLS)
    assert set(C.G8K_ALL_POSITION_CELLS) - set(C.G8K_NATIVE_CELLS) == {"constv2-mix-full__psr_ce_all_minus_ctl_distilled"}
    for name, cap in zip(names, C.G8K_CAPS):
        assert name in C.AUTHORISED_OOS_EXPERIMENTS and name in C.G8K_EXPERIMENT_NAMES
        exp = C.AUTHORISED_OOS_EXPERIMENTS[name]
        # the battery's identity, field by field: native serving, thinking ON, the vendor thinking_card row, k = 1, pass 1, the cap
        assert exp.model == MODEL and exp.cells == C.G8K_NATIVE_CELLS and exp.controls == (C.CONTROL_GSM8K,)
        assert exp.modes == (C.THINKING_ON,) and exp.sampling_mode == C.SAMPLING_THINKING_CARD and exp.samples == 1
        assert exp.reduction_mode == C.REDUCTION_MODE_NOT_APPLICABLE == W19_REDUCTION_MODE and exp.pass_no == 1
        assert exp.max_tokens_on == {C.GSM8K_PAIRED_TASK_NAME: cap}
        assert exp.header_located is False and exp.declares is False
        assert exp.prompt_opens_think_on == C.FAMILY_RENDER_FACTS[C.FAMILY_BY_MODEL_KEY[MODEL]].prompt_opens_think_on
        assert "D101" in exp.what and "not_applicable" in exp.what and "lora_native" in exp.what and str(cap) in exp.what
        # byte-equal to the lens-era twin except the three fields that ARE the native identity
        lens = C.AUTHORISED_OOS_EXPERIMENTS[LENS_ERA.format(cap=cap)]
        assert {k for k in exp._asdict() if exp._asdict()[k] != lens._asdict()[k]} == {"cells", "reduction_mode", "what"}
        # the lens-era tuple is untouched: stacked, the six cells, its own prose
        assert lens.reduction_mode == C.REDUCTION_MODE_STACKED and lens.cells == C.G8K_ALL_POSITION_CELLS
        assert "gsm8k-paired200-paper-objects" in lens.what and "D101" not in lens.what
    C.check_authorised_render_facts()
    # the registry's shape invariant the other tests pin still holds with the pair inside it
    assert set(C.AUTHORISED_OOS_EXPERIMENTS) == {C.AUTHORISED_OOS_LEGACY_NAME, *C.M5T_EXPERIMENT_NAMES, *C.G8K_EXPERIMENT_NAMES}


def test_the_w19_identity_resolves_the_native_door_and_the_lens_era_tuple_still_refuses_it_by_name():
    for cell in C.G8K_NATIVE_CELLS:
        # green: the base row's identity (8,192) and the rung's (16,384) each resolve THEIR native door
        assert C.check_out_of_support_thinking_authorised(**_w19(cell)) == NATIVE.format(cap=8192)
        assert C.check_out_of_support_thinking_authorised(**_w19(cell, RUNG_CAP)) == NATIVE.format(cap=16384)
        assert C.check_out_of_support_thinking_authorised(**_w19(cell, pass_no=1)) == NATIVE.format(cap=8192)
        # the lens identity keeps its own door -- a lens-served leg never resolves the native tuple
        assert C.check_out_of_support_thinking_authorised(**_w19(cell, reduction_mode="stacked")) == LENS_ERA.format(cap=8192)
        assert C.check_out_of_support_thinking_authorised(**_w19(cell, RUNG_CAP, reduction_mode="stacked")) == LENS_ERA.format(cap=16384)
        # the lens-era tuple against the native identity: the EXACT original refusal, unchanged (a second door, never a widened one)
        for cap in C.G8K_CAPS:
            lens_name = LENS_ERA.format(cap=cap)
            req = {k: v for k, v in _w19(cell, cap).items() if k != "max_tokens_by_task"}
            with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e:
                C._check_authorised_experiment(C.authorised_experiment_named(lens_name), flag=C.FLAG_ALLOW_OOS_THINKING,
                                               exc=C.OutOfSupportThinkingUnauthorised, tuple_name=C._authorised_tuple_name(lens_name),
                                               where="driver run", pass_no=None, **req)
            assert W19_REFUSAL in str(e.value) and f"AUTHORISED_OOS_EXPERIMENTS[{lens_name!r}]" in str(e.value)
    cell = C.G8K_NATIVE_CELLS[0]
    # a native run whose identity differs in ANY compared field refuses, naming the field
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="max_tokens_on differs"):
        C.check_out_of_support_thinking_authorised(**_w19(cell, 4096))
    with pytest.raises(C.OutOfSupportAuthorisationAmbiguous, match="differ ONLY in max_tokens_on"):
        C.check_out_of_support_thinking_authorised(**_w19(cell, max_tokens_by_task=None))   # unstated: the pair is two doors
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="sampling_mode differs"):
        C.check_out_of_support_thinking_authorised(**_w19(cell, sampling_mode="card"))       # not the vendor thinking row
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="mode differs"):
        C.check_out_of_support_thinking_authorised(**_w19(cell, mode="off"))                 # thinking OFF
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="samples differs"):
        C.check_out_of_support_thinking_authorised(**_w19(cell, samples=2))
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="pass_no differs"):
        C.check_out_of_support_thinking_authorised(**_w19(cell, pass_no=2))
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="reduction_mode differs"):
        C.check_out_of_support_thinking_authorised(**_w19(cell, reduction_mode="per_row"))   # neither native nor the lens-era shape
    # under `math500` the SAME five lane cells on the native identity now resolve the MATH-500 seed wave's OWN native tuple (a door of
    # its own, beside this pair; the GSM8K pair is untouched) — a WIDER control set is still no experiment's (the controls must be a
    # subset of one tuple's)
    assert C.check_out_of_support_thinking_authorised(**_w19(cell, controls=["math500"])) == "math500_thinking_on_full22_qwen35_4b_native"
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="controls"):
        C.check_out_of_support_thinking_authorised(**_w19(cell, controls=["gsm8k", "math500"]))
    # a cell outside the lane (the sixth all-position cell) under the native identity has NO door
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="is not it"):
        C.check_out_of_support_thinking_authorised(**_w19("constv2-mix-full__psr_ce_all_minus_ctl_distilled"))
    # the Arditi all-positions arm keeps ITS OWN native tuples (already not_applicable; untouched)
    assert C.check_out_of_support_thinking_authorised(**_w19(C.ALLPOS_CELL)) == C.g8k_allpos_tuple_name(MODEL, 8192)


# --------------------------------------------------------------------------- the plan (seedrep_launch_plan.py)


def test_every_fitted_lane_gsm8k_child_of_the_draft_records_the_native_tuple_and_validate_plan_refuses_the_lens_era_one(draft):
    assert lp.GSM8K_NATIVE_OOS_TUPLE.format(model=lp.MODEL_KEY, cap=8192) == NATIVE.format(cap=8192) == C.g8k_native_tuple_name(lp.MODEL_KEY, 8192)
    assert lp.GSM8K_LENS_ERA_OOS_TUPLE.format(model=lp.MODEL_KEY, cap=8192) == LENS_ERA.format(cap=8192) == C.g8k_tuple_name(lp.MODEL_KEY, 8192)
    for cap in (8192, 16384):
        for cell in lp.FITTED_CELLS:
            assert lp.gsm8k_oos_tuple(cell, cap, engine="s1") == NATIVE.format(cap=cap)
        assert lp.gsm8k_oos_tuple(lp.ARDITI_CELL, cap, engine=lp.ARDITI) == C.g8k_allpos_tuple_name(lp.MODEL_KEY, cap)
        assert lp.gsm8k_oos_tuple("none", cap, engine="s1") is None and lp.gsm8k_oos_tuple(lp.PROMPTED_CELLS[0], cap, engine="s1") is None
    seen = {"base": 0, "rung": 0}
    for r in _gsm8k_rows(draft):
        for k in r["children"]:
            fitted_lane = r["engine"] != lp.ARDITI and k["cell"] in lp.FITTED_CELLS
            if fitted_lane:
                want_cap = 8192 if r["kind"] == "base" else 16384
                assert k["cap"] == want_cap and k["fitted"] is True and "--allow-out-of-support-thinking" in k["argv"]
                assert k["oos_tuple"] == NATIVE.format(cap=want_cap), (r["standup_id"], k["arm"])
                assert C.AUTHORISED_OOS_EXPERIMENTS[k["oos_tuple"]].reduction_mode == W19_REDUCTION_MODE
                assert k["argv"][k["argv"].index("--max-tokens-by-task") + 1] == f"gsm8k_paired200={want_cap}"
                seen[r["kind"]] += 1
            elif r["engine"] == lp.ARDITI:
                assert k["oos_tuple"] == C.g8k_allpos_tuple_name(lp.MODEL_KEY, k["cap"])
            else:
                assert k["oos_tuple"] is None and "--allow-out-of-support-thinking" not in k["argv"]
    assert seen == {"base": 3 * len(lp.FITTED_CELLS), "rung": 3 * len(lp.FITTED_CELLS)} == {"base": 15, "rung": 15}
    assert lp.validate_plan(draft) == []
    # KNOWN-BAD: a lane row recorded under the lens-era tuple refuses BY NAME, naming the native tuple
    import copy
    bad = copy.deepcopy(draft)
    row = next(r for r in bad["standups"] if r["leg"] == "gsm8k" and r["engine"] == "s1")
    kid = next(k for k in row["children"] if k["cell"] in lp.FITTED_CELLS)
    kid["oos_tuple"] = LENS_ERA.format(cap=kid["cap"])
    probs = [p for p in lp.validate_plan(bad) if "LENS-ERA" in p]
    assert len(probs) == 1 and row["standup_id"] in probs[0] and LENS_ERA.format(cap=kid["cap"]) in probs[0]
    assert "D101" in probs[0] and NATIVE.format(cap=kid["cap"]) in probs[0] and "not_applicable" in probs[0]
    # the lens-era spelling at the OTHER cap is refused just the same (a tuple of another rung is never a door either)
    kid["oos_tuple"] = LENS_ERA.format(cap=16384 if kid["cap"] == 8192 else 8192)
    assert any("LENS-ERA" in p for p in lp.validate_plan(bad))


# --------------------------------------------------------------------------- the driver's first door (served_capability_run.py)


def test_the_drivers_first_door_states_the_served_reduction_mode_read_off_the_standup_record():
    """The early door -- whose return the manifest records -- reads `endpoint.json` beside --routing-table through the
    SAME reader the served identity uses (`_served_reduction_mode`) and passes it as `reduction_mode=`; an absent record
    leaves it unstated (the later `_standup_endpoint` read refuses in its own words). The second / third doors are
    unchanged."""
    src = (REPO / "scripts/served_capability_run.py").read_text()
    door = src.index("oos_experiment = tc.check_out_of_support_thinking_authorised(")
    read = src.index("first_door_record = Path(args.routing_table).resolve().parent / STANDUP_ENDPOINT_FILENAME")
    assert read < door < src.index("standup_endpoint = _standup_endpoint(args.routing_table, cell=args.cell,")
    call = src[door:src.index('where="driver run")', door)]
    assert "reduction_mode=first_door_reduction_mode," in call and "max_tokens_by_task=(dict(max_tokens_stmt) if max_tokens_stmt else None)," in call
    assert ('first_door_reduction_mode = (\n            _served_reduction_mode(json.loads(first_door_record.read_text()), source="standup record")\n'
            "            if first_door_record.is_file() else None)") in src
    # the reader itself, on the two record shapes: native -> not_applicable; lens with an attested stacked mode -> stacked
    sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as scr
    assert scr._served_reduction_mode({"engine": "lora_native"}, source="t") == "not_applicable" == C.REDUCTION_MODE_NOT_APPLICABLE
    assert scr._served_reduction_mode({"lens_reduction_mode": "stacked"}, source="t") == "stacked" == C.REDUCTION_MODE_STACKED
    # so a LENS-ERA standup's first door stays unambiguous with the native pair registered: it resolves the lens-era door
    assert C.check_out_of_support_thinking_authorised(**_w19(C.G8K_NATIVE_CELLS[0], reduction_mode="stacked")) == LENS_ERA.format(cap=8192)


# --------------------------------------------------------------------------- the REAL chain: plan argv -> driver parser -> registry


_CHAIN = r'''
import json, sys
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1] + "/scripts"]
import served_capability_run as scr
from sisteer.capeval.tinycap import contracts as tc
out = []
for rec in json.load(sys.stdin):
    a = scr.build_parser().parse_args(rec["argv"][2:])          # argv[0:2] = the client venv python + the driver script
    assert a.allow_out_of_support_thinking is True and a.expect_object_sha256
    stmt = tc.parse_max_tokens_by_task(a.max_tokens_by_task)
    # the eight fields exactly as the driver hands them (served_capability_run: sampling.model_key from the per-cell config,
    # args.cell, the parsed controls, args.thinking, sampling.mode, sampling.samples, the standup's served reduction_mode)
    req = dict(model=rec["model_key"], cell=a.cell, controls=[c for c in str(a.controls).split(",") if c], mode=a.thinking,
               sampling_mode=a.sampling, samples=a.samples, max_tokens_by_task=dict(stmt), where="driver run")
    row = {"cell": a.cell, "stmt": stmt}
    # KNOWN-BAD (the W19 driver's first door): the eight fields + the stated cap with NO reduction_mode -- with the native pair
    # in the registry that door is AMBIGUOUS for every lane cell (two tuples differ only in the field it leaves unstated)
    try:
        row["first_door_unstated"] = tc.check_out_of_support_thinking_authorised(**req)
    except tc.OutOfSupportThinkingUnauthorised as e:
        row["first_door_unstated"] = {"refused": type(e).__name__, "message": str(e)[:400]}
    # the driver's FIRST door as it runs now (D101 / W20 fc2): the served reduction_mode is a pure read of the standup record
    # beside --routing-table through scr._served_reduction_mode -- here a synthetic NATIVE record in the shape the W19
    # standup wrote (engine lora_native, no lens_reduction_mode) -- and its return is the name the manifest records
    import pathlib
    record = pathlib.Path(a.routing_table).resolve().parent / scr.STANDUP_ENDPOINT_FILENAME
    assert record.is_file(), record
    first_door_reduction_mode = scr._served_reduction_mode(json.loads(record.read_text()), source="standup record")
    assert first_door_reduction_mode == rec["served_reduction_mode"] == "not_applicable"
    row["first_door"] = tc.check_out_of_support_thinking_authorised(**req, reduction_mode=first_door_reduction_mode)
    # the second / third doors (:6205, :6255): the same request WITH the standup's served reduction_mode; return value discarded by the driver
    row["native"] = tc.check_out_of_support_thinking_authorised(**req, reduction_mode=rec["served_reduction_mode"])
    row["stacked"] = tc.check_out_of_support_thinking_authorised(**req, reduction_mode="stacked")
    lens_name = "gsm8k_paired200_" + rec["model_key"] + "_" + str(stmt["gsm8k_paired200"])
    try:
        tc._check_authorised_experiment(tc.authorised_experiment_named(lens_name), flag=tc.FLAG_ALLOW_OOS_THINKING,
                                        exc=tc.OutOfSupportThinkingUnauthorised, tuple_name=tc._authorised_tuple_name(lens_name),
                                        reduction_mode=rec["served_reduction_mode"], pass_no=None,
                                        **{k: v for k, v in req.items() if k != "max_tokens_by_task"})
        row["lens_era_refusal"] = None
    except tc.OutOfSupportThinkingUnauthorised as e:
        row["lens_era_refusal"] = str(e)
    out.append(row)
json.dump(out, sys.stdout)
'''


def _ctx(root: str, child, *, S: int = 31):
    subs = {lp.LEDGER_URL_PLACEHOLDER: "http://127.0.0.1:0/v1", lp.CHILD_ROUTING_TABLE_PLACEHOLDER: f"{root}/{child['ledger']['routing_table_rel']}",
            lp.GSM8K_ATTEMPT_ID_PLACEHOLDER: "a1"}
    return lp.SpawnContext(attempt_root=root, attempt_id="ta-1", standup_id="x", plans_root="/vol/p", proxy_url="http://127.0.0.1:0/v1",
                           routing_table="/t.json", driver_python="/py", client_python="/cpy", S=S, ncells=8, bank_view="/b", base_env={},
                           child_subs={lp.child_key(child): subs})


def test_the_composed_gsm8k_child_argv_through_the_drivers_parser_resolves_the_native_tuple_in_a_fresh_interpreter(draft, tmp_path):
    """Known-bad before the fix: the same argv, parsed by the same parser, handed to the same checker with the standup's
    served reduction_mode refused `reduction_mode differs -- requested 'not_applicable', authorised 'stacked'`; with the
    native pair registered but the former driver's UNSTATED first door, `ambiguous authorisation` (pinned below as
    `first_door_unstated`). Fixed: the first door reads the record's served identity and resolves the native name."""
    # the child's standup record, where the driver reads it: `endpoint.json` beside the child's --routing-table copy --
    # a synthetic NATIVE record (the fields the standup's record carried, no ids)
    attempt_root = tmp_path / "attempt"
    picks = []
    for r in _gsm8k_rows(draft):
        if r["engine"] != "s1":
            continue
        for k in r["children"]:
            if k["cell"] == lp.FITTED_CELLS[0] and r["kind"] in ("base", "rung"):
                argv, _env = lp.resolve_spawn(k, _ctx(str(attempt_root), k))
                assert "{" not in " ".join(argv) and argv[2] == "generate", argv
                table = Path(argv[argv.index("--routing-table") + 1])
                table.parent.mkdir(parents=True, exist_ok=True)
                (table.parent / "endpoint.json").write_text(json.dumps({
                    "engine": "lora_native", "serving_engine": "lora_native", "serving_path": "lora_native",
                    "model": "Qwen/Qwen3.5-4B", "model_revision": "0" * 40, "routing_table": str(table)}))
                picks.append({"argv": argv, "model_key": lp.MODEL_KEY, "served_reduction_mode": W19_REDUCTION_MODE, "want": k["oos_tuple"],
                              "kind": r["kind"], "cap": k["cap"]})
    assert sorted(p["kind"] for p in picks) == ["base", "rung"]
    proc = subprocess.run([sys.executable, "-c", _CHAIN, str(REPO)], input=json.dumps(picks), capture_output=True, text=True,
                          timeout=600, cwd=str(REPO))
    assert proc.returncode == 0, proc.stderr[-4000:]
    rows = json.loads(proc.stdout)
    assert len(rows) == 2
    for pick, row in zip(picks, rows):
        assert row["cell"] == lp.FITTED_CELLS[0] and row["stmt"] == {"gsm8k_paired200": pick["cap"]}
        assert row["native"] == pick["want"] == NATIVE.format(cap=pick["cap"])           # the doors that STATE the served identity resolve the native tuple
        assert row["first_door"] == pick["want"], row["first_door"]                     # the door the driver RECORDS resolves the same name
        # the known-bad this fix exists for: an UNSTATED first door is ambiguous between the lens-era and the native pair
        assert isinstance(row["first_door_unstated"], dict) and row["first_door_unstated"]["refused"] == "OutOfSupportAuthorisationAmbiguous"
        assert LENS_ERA.format(cap=pick["cap"]) in row["first_door_unstated"]["message"] and NATIVE.format(cap=pick["cap"]) in row["first_door_unstated"]["message"]
        assert row["stacked"] == LENS_ERA.format(cap=pick["cap"])                          # the lens identity keeps its own
        assert row["lens_era_refusal"] and W19_REFUSAL in row["lens_era_refusal"]         # the original message, verbatim, from the lens-era tuple


# --------------------------------------------------------------------------- the composed paired-write arm's pair


def test_the_composed_paired_write_arm_has_its_own_gsm8k_pair_beside_the_native_one_and_neither_door_opens_for_the_other():
    """2026-09-21, 9865993c7: the paired views' fitted rows refused `OOS_UNAUTHORISED: … cell differs — requested
    'constv2-mix-full__psr_ce_all_minus_ctl'` — the composed arm was in NO tuple. It gets its own pair (never a widening of the
    native pair, whose 13 finished stores record their tuple by name); the driver's identity for a paired-view row resolves it
    at either cap, the native cells never resolve it, the composed arm never resolves the native pair or the lens-era one, and
    the `_distilled` sibling stays unauthorised."""
    names = [C.g8k_native_paired_tuple_name(MODEL, cap) for cap in C.G8K_CAPS]
    assert names == [PAIRED.format(cap=8192), PAIRED.format(cap=16384)]
    assert C.G8K_NATIVE_PAIRED_TUPLE_INFIX == "_native_paired" and C.G8K_NATIVE_PAIRED_CELLS == (PAIRED_CELL,)
    assert PAIRED_CELL not in C.G8K_NATIVE_CELLS and PAIRED_CELL not in C.G8K_ALL_POSITION_CELLS       # a NEW arm, not a lane cell
    for cap, name in zip(C.G8K_CAPS, names):
        assert name in C.AUTHORISED_OOS_EXPERIMENTS and name in C.G8K_EXPERIMENT_NAMES
        exp = C.AUTHORISED_OOS_EXPERIMENTS[name]; native = C.AUTHORISED_OOS_EXPERIMENTS[NATIVE.format(cap=cap)]
        assert exp.cells == (PAIRED_CELL,) and exp.reduction_mode == C.REDUCTION_MODE_NOT_APPLICABLE
        assert exp.max_tokens_on == {"gsm8k_paired200": cap} == native.max_tokens_on
        # byte-equal to the native tuple but for cells + what
        assert {k: v for k, v in exp.as_dict().items() if k not in ("cells", "what")} == {k: v for k, v in native.as_dict().items() if k not in ("cells", "what")}
        assert "PR #300" in exp.what and "psr_paired_write" in exp.what and "never a widening" in exp.what
    # the paired-view row's identity (the driver's 8 fields + the stated cap) resolves the paired door at each cap
    assert C.check_out_of_support_thinking_authorised(**_w19(PAIRED_CELL)) == PAIRED.format(cap=8192)
    assert C.check_out_of_support_thinking_authorised(**_w19(PAIRED_CELL, RUNG_CAP)) == PAIRED.format(cap=16384)
    # the native cells never resolve the paired door; the composed arm never resolves the native one — each refuses by name
    for cell in C.G8K_NATIVE_CELLS:
        assert C.check_out_of_support_thinking_authorised(**_w19(cell)) == NATIVE.format(cap=8192)
        with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="cell differs"):
            C._check_authorised_experiment(C.authorised_experiment_named(names[0]), flag=C.FLAG_ALLOW_OOS_THINKING,
                                           exc=C.OutOfSupportThinkingUnauthorised, tuple_name=C._authorised_tuple_name(names[0]),
                                           where="driver run", pass_no=None, **{k: v for k, v in _w19(cell).items() if k != "max_tokens_by_task"})
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="cell differs"):
        C._check_authorised_experiment(C.authorised_experiment_named(NATIVE.format(cap=8192)), flag=C.FLAG_ALLOW_OOS_THINKING,
                                       exc=C.OutOfSupportThinkingUnauthorised, tuple_name=C._authorised_tuple_name(NATIVE.format(cap=8192)),
                                       where="driver run", pass_no=None, **{k: v for k, v in _w19(PAIRED_CELL).items() if k != "max_tokens_by_task"})
    # no lens-era door for the composed arm (a stacked identity finds no tuple), and the distilled sibling is nowhere
    with pytest.raises(C.OutOfSupportThinkingUnauthorised):
        C.check_out_of_support_thinking_authorised(**_w19(PAIRED_CELL, reduction_mode="stacked"))
    with pytest.raises(C.OutOfSupportThinkingUnauthorised):          # the resolver names the NEAREST tuple's first differing field
        C.check_out_of_support_thinking_authorised(**_w19(PAIRED_CELL + "_distilled"))
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="cell differs"):   # and the paired door itself refuses it by cell
        C._check_authorised_experiment(C.authorised_experiment_named(names[0]), flag=C.FLAG_ALLOW_OOS_THINKING,
                                       exc=C.OutOfSupportThinkingUnauthorised, tuple_name=C._authorised_tuple_name(names[0]),
                                       where="driver run", pass_no=None, **{k: v for k, v in _w19(PAIRED_CELL + "_distilled").items() if k != "max_tokens_by_task"})
    # the registry's shape invariant holds with the pair inside it
    assert set(C.AUTHORISED_OOS_EXPERIMENTS) == {C.AUTHORISED_OOS_LEGACY_NAME, *C.M5T_EXPERIMENT_NAMES, *C.G8K_EXPERIMENT_NAMES}
