"""F-AF -- the plan composer's controls children must turn ON the driver's per-draw event stream.

THE DEFECT. Every controls row (S2 controls-arditi, S3 controls s1, S5 controls s1 FULL) ended
``LADDER_INVENTORY_INCOMPLETE rc=1`` with ALL children rc 0 and real draws (xstest 450, or_bench_hard_1k 1319 per child; MMLU acc
beside them -- F-T fixed). `require_ladder_inventory(..., require_rows=True)` (the app; F-H) refuses an EMPTY inventory over ZERO
streams, and the row's `ladder_enqueued.json` read `{streams: [], entries: [], n: 0, complete: true}`: the app's
`ControlsStreamAdapter.streams()` globs `exp/sampled/controls_cap*/**/controls_events.jsonl`, but `served_capability_run.py` writes that
stream ONLY under its opt-in flag (`cev.CONTROLS_EVENTS_FLAG`, dest `controls_events`, default off), and the composed controls argv never
passed it -- the stores held `rows.jsonl`, `cap_ladder.json`, `gen_stats.json`, `manifest.json`, `evidence.json` and no
`controls_events.jsonl`. The static ladder producer (S5 pasttense FULL: 182 entries / 48 streams) works; this is controls-only.

THE FIX (this module's pins). (1) `_controls_children` appends `lp.CONTROLS_EVENTS_FLAG` to every child that carries a ladder control
(the base row's three-control set, a rung's two, a dose row's `xstest`); the literal equals the driver's `cev.CONTROLS_EVENTS_FLAG`
(pinned here: the plan module cannot import `sisteer`). (2) `validate_plan` (S0's PLAN gate) refuses a controls child without the flag
by name, and refuses the flag beside a set with NO ladder control (the driver refuses that by name: the stream rides a SAMPLED
judged-controls run). (3) The composed argv drives the REAL `served_capability_run.build_parser()` in a fresh interpreter and parses
`controls_events True`; the known-bad argv (verbatim shape, synthetic paths) parses `controls_events False`. (4) The app's
adapter finds a store laid out the driver's way once the file exists, and does NOT over the known-bad store; the inventory refusal is the
exact one the defect hit. (5) The ladder semantics: the driver's own `cap-ladder ... ESCALATE -> 1024` line is the tiers check's DOUBLING
ladder (`capeval.tiers.cap_ladder.CAP_LADDER`, 512 -> 1024 -> ...), a report line nothing in the seedrep chain consumes; the queue the
tailer feeds is `row.ladder.cap_next` = `seedrep_ladder.LADDERS["controls"]` = `sampled_controls.CONTROLS_CAP_LADDER` (512 -> 2,048 ->
8,192 -> 16,384), and an event carries no `next_cap` -- the adapter files every at-cap, error-free event into the plan's 2,048 queue.
"""
from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts" / "modal", REPO / "tests"):
    sys.path.insert(0, str(_p))
import seedrep_launch_plan as lp  # noqa: E402
import seedrep_ladder as SL  # noqa: E402  (the protocol module; stdlib only -- `L` is RF's fixture-loaded copy, used by the tailer harness)
import test_seedrep_review_fixes as RF  # noqa: E402
from test_seedrep_review_fixes import L, draft, inputs, srv  # noqa: E402,F401  (module fixtures register by import; `srv` = the app under a modal stub)
from sisteer.eval import controls_events as cev  # noqa: E402
from sisteer.eval.sampled_controls import CONTROLS_CAP_LADDER  # noqa: E402

_row = RF._row

BASE_CONTROLS_CHILDREN = 27           # 3 views x 8 arm-runs + Arditi x 3 seed bases
RUNG_CONTROLS_CHILDREN = 81           # the same 27 per rung x 3 rungs (2,048 / 8,192 / 16,384)


def _controls_children(plan, kind):
    return [(r, k) for r in plan["standups"] + plan["rungs"] if r["leg"] == "controls" and r["kind"] == kind and not r.get("dose_standup")
            for k in r["children"]]


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


# --------------------------------------------------------------------------- the constant IS the driver's ------------------------------------ #

def test_the_plans_flag_literal_is_the_drivers_and_the_stream_file_is_the_adapters_glob_target():
    assert lp.CONTROLS_EVENTS_FLAG == cev.CONTROLS_EVENTS_FLAG == "--controls-events"
    assert cev.CONTROLS_EVENTS_FILENAME == "controls_events.jsonl"
    # an event names the draw and its outcome, never a next rung: the adapter cannot be reading the driver's ESCALATE suggestion
    assert "next_cap" not in cev.CONTROLS_EVENT_FIELDS and "at_cap" in cev.CONTROLS_EVENT_FIELDS and "error" in cev.CONTROLS_EVENT_FIELDS
    assert lp.controls_ride_the_ladder(lp.CONTROLS_ALL) and lp.controls_ride_the_ladder(lp.CONTROLS_LADDER) and lp.controls_ride_the_ladder(("xstest",))
    assert not lp.controls_ride_the_ladder(lp.CONTROLS_FIXED_CAP) and not lp.controls_ride_the_ladder(())


# --------------------------------------------------------------------------- the composed shape --------------------------------------------- #

def test_every_controls_child_carries_the_flag_once(draft):
    base, rung = _controls_children(draft, "base"), _controls_children(draft, "rung")
    assert len(base) == BASE_CONTROLS_CHILDREN and len(rung) == RUNG_CONTROLS_CHILDREN
    for r, k in base + rung:
        a = k["argv"]
        assert a.count(lp.CONTROLS_EVENTS_FLAG) == 1, (r["standup_id"], k["arm"], k["s"], a)
        assert lp.controls_ride_the_ladder(_flag(a, "--controls").split(",")), (r["standup_id"], k["arm"])
        assert k["out_rel"] == f"exp/sampled/controls_cap{k['cap']}__s{k['s']}"        # the root the adapter's glob walks
    # a dose row's controls child (xstest alone, sampled) rides the ladder too -- the stream is on
    for r in draft["standups"]:
        if r["leg"] == "controls" and r.get("dose_standup"):
            for k in r["children"]:
                assert k["argv"].count(lp.CONTROLS_EVENTS_FLAG) == 1 and _flag(k["argv"], "--controls") == "xstest"
    # unit: a fixed-cap-only set never carries the flag (the driver refuses it by name on such a run)
    kids = lp._controls_children("s1", "2" * 40, lp.CONTROLS_BASE_CAP, runs=[{"arm": "none", "cell": "none", "s": 1, "object": "none"}],
                                 controls=("mmlu",), base_cap=lp.CONTROLS_BASE_CAP)
    assert len(kids) == 1 and lp.CONTROLS_EVENTS_FLAG not in kids[0]["argv"] and _flag(kids[0]["argv"], "--controls") == "mmlu"


# --------------------------------------------------------------------------- the driver's REAL parser ----------------------------------------- #

def _driver_parse(argvs):
    """Every argv (the tokens after `python script`) through the REAL driver in a fresh interpreter (it binds `sampled_controls` at import):
    `build_parser().parse_args` -> `controls_events`, the parsed controls, the driver's own stream refusal condition (a SAMPLED judged
    control among the legacy controls; the config's decode class is not readable here, so only the controls half), and its ladder constants
    beside the tiers check's."""
    code = (
        "import json, sys, io, contextlib\n"
        "sys.path.insert(0, sys.argv[1] + '/src'); sys.path.insert(0, sys.argv[1] + '/scripts')\n"
        "import served_capability_run as R\n"
        "from sisteer.capeval.tiers import cap_ladder as cl\n"
        "P = R.build_parser(); out = []\n"
        "for argv in json.load(sys.stdin):\n"
        "    buf = io.StringIO()\n"
        "    try:\n"
        "        with contextlib.redirect_stderr(buf):\n"
        "            ns = P.parse_args(argv)\n"
        "            controls = R.parse_controls(ns.controls)\n"
        "        legacy = [c for c in controls if c in R.SERVED_CONTROLS]\n"
        "        out.append({'ok': True, 'controls_events': bool(ns.controls_events), 'controls': controls, 'controls_cap': ns.controls_cap,\n"
        "                    'has_sampled_control': any(c in R.SAMPLED_CONTROLS for c in legacy)})\n"
        "    except SystemExit as e:\n"
        "        out.append({'ok': False, 'error': (buf.getvalue().strip()[-600:] or str(e))})\n"
        "out.append({'SAMPLED_CONTROLS': list(R.SAMPLED_CONTROLS), 'CONTROLS_CAP_LADDER': list(R.CONTROLS_CAP_LADDER),\n"
        "            'CHECK_CAP_LADDER': list(cl.CAP_LADDER), 'CHECK_NEXT_OF_512': cl.next_rung(512, cl.CAP_LADDER, cl.CAP_CEILING),\n"
        "            'FLAG': R.cev.CONTROLS_EVENTS_FLAG, 'FILENAME': R.cev.CONTROLS_EVENTS_FILENAME})\n"
        "json.dump(out, sys.stdout)\n")
    r = subprocess.run([sys.executable, "-c", code, str(REPO)], input=json.dumps([list(a) for a in argvs]), capture_output=True, text=True,
                       timeout=600, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


# the known-bad composed shape, from the children's `*.argv.json` records, with synthetic host paths -- no stream flag
W19_BASE_ARGV = ["generate", "--proxy-url", "http://127.0.0.1:9/v1", "--routing-table", "/workspace/attempt/exp/serve/routing_table.json",
                 "--cell", "none", "--config", "configs/sdfbase_full22_qwen35_4b_s1_sampled_t0.yaml", "--defenses-dir", "/b",
                 "--out-root", "/workspace/attempt/exp/sampled/controls_cap512__s1", "--controls", "xstest,or_bench_hard_1k,mmlu",
                 "--concurrency", "35", "--top-logprobs", "64", "--trial", "0"]


def test_every_composed_controls_child_parses_controls_events_true_and_w19s_argv_does_not(draft):
    kids = _controls_children(draft, "base") + _controls_children(draft, "rung")
    ctx = lp.SpawnContext(attempt_root="/workspace/attempt", attempt_id="ta-1", standup_id="x", plans_root="/vol/p", proxy_url="http://127.0.0.1:9/v1",
                          routing_table="/t.json", driver_python="/py", client_python="/cpy", S=52, ncells=8, bank_view="/b", base_env={})
    resolved = [lp.resolve_spawn(k, ctx)[0][2:] for _, k in kids]
    res = _driver_parse(resolved + [W19_BASE_ARGV])
    consts, res = res[-1], res[:-1]
    assert consts["FLAG"] == lp.CONTROLS_EVENTS_FLAG and consts["FILENAME"] == cev.CONTROLS_EVENTS_FILENAME
    for (r, k), v in zip(kids, res):
        assert v["ok"], (r["standup_id"], k["arm"], v)
        assert v["controls_events"] is True, (r["standup_id"], k["arm"], k["s"], v)
        assert v["has_sampled_control"], (r["standup_id"], k["arm"], v)         # the driver's own condition for accepting the flag
        assert v["controls"] == k["controls"]
    w19 = res[len(kids)]
    assert w19["ok"] and w19["controls_events"] is False and w19["has_sampled_control"], w19   # parses fine; writes no stream -> F-AF
    # THE LADDER SEMANTICS: the queue ladder is the plan's / the driver's SAMPLED ladder; the "ESCALATE -> 1024" line is the
    # tiers check's doubling ladder, which the seedrep chain never consumes
    assert consts["CONTROLS_CAP_LADDER"] == list(CONTROLS_CAP_LADDER) == list(SL.LADDERS["controls"]) == [512, 2048, 8192, 16384]
    assert consts["CHECK_NEXT_OF_512"] == 1024 and 1024 in consts["CHECK_CAP_LADDER"] and 1024 not in consts["CONTROLS_CAP_LADDER"]
    assert consts["SAMPLED_CONTROLS"] == list(lp.CONTROLS_LADDER)


def test_the_queue_the_tailer_feeds_is_the_plans_cap_next_not_the_drivers_check_line(draft):
    rows = [r for r in draft["standups"] if r["leg"] == "controls" and r["kind"] == "base" and not r.get("dose_standup")]
    assert len(rows) == 4
    for r in rows:
        assert r["ladder"]["kind"] == "controls" and int(r["ladder"]["cap_next"]) == SL.next_cap("controls", lp.CONTROLS_BASE_CAP) == 2048
    stages = sorted({int(r["stage"]) for r in draft["rungs"] if r["leg"] == "controls"})
    assert stages == list(SL.LADDERS["controls"][1:]) == [2048, 8192, 16384]
    for r in draft["rungs"]:
        if r["leg"] == "controls":
            lad = r["ladder"]
            assert int(lad["consumes"]) == int(r["stage"]) and lad["cap_next"] == SL.next_cap("controls", int(r["stage"]))
    # the queue record built from an at-cap event binds the PLAN's base cap (the spec's), reading nothing like a next cap off the event
    spec = SL.QueueSpec.build("2222222", RF.WAVE, "controls", "s1", "controls", 2048, ["0"])
    assert spec.base_cap == 512 and spec.next_spec(["0"]).cap_next == 8192
    with pytest.raises(SL.LadderRefusal, match="not at cap"):
        SL.controls_queue_record(spec, arm="none", s=1, control="xstest", event={"item_id": "i1", "at_cap": False, "sampling_seed": 7, "text_sha256": "0" * 64},
                                prompt="p", system=None, base_store="exp/sampled/controls_cap512__s1/cells/x", producer=spec.producers[0],
                                producer_attempt_id="a", vmodel="m", sampling={"temperature": 0.7})


# --------------------------------------------------------------------------- the app side: adapter glob + the exact refusal ----------------- #

def _store(root: Path, s: int, *names: str) -> Path:
    cell = root / f"exp/sampled/controls_cap512__s{s}/cells/none__controls__t0__a0__{'a' * 12}__{'b' * 12}"
    cell.mkdir(parents=True, exist_ok=True)
    for n in names:
        (cell / n).write_text("")
    return cell


def test_the_adapter_finds_the_drivers_stream_and_not_w19s_store(srv, tmp_path):
    adapter = srv.ControlsStreamAdapter(L, None, None, "ta-1", container_root=tmp_path, attempt_root_vol="/vol/seedrep/x",
                                        seed_of_store=lambda p: 1, vmodel_of=lambda arm: arm)
    _store(tmp_path, 1, "rows.jsonl", "cap_ladder.json", "gen_stats.json", "manifest.json", "evidence.json")     # the known-bad store: no stream
    assert adapter.streams(tmp_path) == []
    cell2 = _store(tmp_path, 2, "rows.jsonl", cev.CONTROLS_EVENTS_FILENAME)                                       # the flagged store
    assert adapter.streams(tmp_path) == [cell2 / cev.CONTROLS_EVENTS_FILENAME]
    assert adapter.at_cap({"at_cap": True, "error": None}) and not adapter.at_cap({"at_cap": False, "error": None})
    assert not adapter.at_cap({"at_cap": True, "error": "timeout"})


def test_the_inventory_refusal_w19_hit_and_the_one_a_flagged_row_passes(srv, tmp_path):
    def write(doc):
        (tmp_path / srv.LADDER_ENQUEUED_JSON).write_text(json.dumps({"schema": "seedrep_ladder_enqueued/1", "attempt_id": "ta-1", "complete": True,
                                                                     "problems": [], **doc}))
    write({"streams": [], "entries": [], "n": 0})                                     # zero streams, nothing drawn as far as the tailer saw
    with pytest.raises(srv.LadderInventoryIncomplete, match="an empty inventory over 0 stream\\(s\\) that carried no rows"):
        srv.require_ladder_inventory(tmp_path, require_rows=True)
    write({"streams": [{"path": "s", "rows_read": 450, "at_cap": 0, "enqueued": 0, "bad_lines": 0}], "entries": [], "n": 0})
    assert srv.require_ladder_inventory(tmp_path, require_rows=True)["n"] == 0       # a stream that carried rows, nothing at cap: stands


# --------------------------------------------------------------------------- the PLAN gate ----------------------------------------------------- #

def test_plan_gate_refuses_a_controls_child_without_the_flag_and_the_flag_without_a_ladder_control(draft):
    assert lp.validate_plan(draft) == []
    for engine, stage in (("s1", "base"), ("arditi", "2048")):
        bad = copy.deepcopy(draft)
        k = _row(bad, "controls", engine, stage=stage)["children"][0]
        k["argv"].remove(lp.CONTROLS_EVENTS_FLAG)                                      # the known-bad argv
        probs = lp.validate_plan(bad)
        assert any(f"carries no {lp.CONTROLS_EVENTS_FLAG}" in p and "LADDER_INVENTORY_INCOMPLETE" in p and k["arm"] in p for p in probs), (stage, probs)
    bad = copy.deepcopy(draft)
    k = _row(bad, "controls", "s2")["children"][0]
    k["argv"][k["argv"].index("--controls") + 1] = "mmlu"
    k["task"] = {"mmlu": lp.CONTROLS_TASKS["mmlu"]}; k["controls"] = ["mmlu"]
    probs = lp.validate_plan(bad)
    assert any(f"carries {lp.CONTROLS_EVENTS_FLAG} with no ladder control" in p for p in probs), probs


def test_ledger_and_control_keys_unchanged_by_the_flag(draft):
    led = lp.capacity_ledger(draft)
    assert (led["unstaged_peak"]["containers"], led["unstaged_peak"]["gpus"]) == (134, 134)      # one H200 per pool
    rows = lp._plan_rows_the_app_may_execute(draft)
    assert len({r["control_key"] for r in rows}) == len(rows) == 130


# --------------------------------------------------------------------------- the full seam: config -> helper -> adapter -> enqueue -> drain ------ #
# With the stream ON, the first at-cap event reaches `ControlsStreamAdapter.records()`, whose sampling resolver is the helper's
# `controls-sampling` (`controls_gen_config`): that row carries the store's bookkeeping (`arm, do_sample, max_new_tokens, seed_base, trial`)
# beside the six frozen sampler keys, and `Producer.enqueue` -> `check_redraw_sampling` refuses ANY key beyond `sampler_keys()`, so the drain
# raised `LadderUnavailable` and the row would have failed LADDER_TAILER_ERROR one step past F-AF. The helper now writes the projection
# (`static_common.sampler_preset_of`, the same `bon_served.SAMPLER_KEYS` the validator reads); the validator is untouched.

PINNED_CONTROLS_CONFIG = "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s1_sampled_t0.yaml"    # the S2 controls-arditi s1 child's --config
SIX = ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"]


def _load_helper():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("seedrep_ladder_helpers_fa2", REPO / "scripts/modal/seedrep_ladder_helpers.py")
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _at_cap_event(seed: int = 17) -> dict:
    return {"v": 1, "kind": "draw", "seq": 1, "ts_utc": "2026-09-19T00:00:00Z", "cell_id": "arditi_allpos", "control": "xstest", "cap": 512,
            "escalated_from_cap": None, "item_id": "xst-1", "sampling_seed": seed, "finish_reason": "length", "at_cap": True, "completion_tokens": 512,
            "prompt_tokens": 40, "text_sha256": "a" * 64, "request_id": "r", "completion_id": "c", "error": None}


class TestSamplingRowReachesTheQueue:
    def _world(self, srv, L, draft, tmp_path):
        row = _row(draft, "controls", "arditi")
        volroot, root, attempt_id, arv, _vol, prod = RF.TestP1_5_ladder_tailer()._setup(srv, L, tmp_path, row)
        ev_dir = root / f"exp/sampled/controls_cap512__s1/cells/arditi_allpos__controls__t0__a0__{'a' * 12}__{'b' * 12}"    # the driver's store layout
        ev_dir.mkdir(parents=True)
        (ev_dir / cev.CONTROLS_EVENTS_FILENAME).write_text(json.dumps(_at_cap_event()) + "\n")
        return row, volroot, root, attempt_id, arv, prod

    @staticmethod
    def _adapter(srv, L, prod, attempt_id, root, arv, sampling_of):
        return srv.ControlsStreamAdapter(L, prod.spec, prod.ref, attempt_id, container_root=root, attempt_root_vol=arv, seed_of_store=srv.controls_seed_of_store,
                                         vmodel_of=lambda a: "petri__arditi_allpos", prompt_of=lambda c, i, a: (f"prompt-{i}", None), sampling_of=sampling_of)

    def test_known_bad_the_unprojected_gen_block_is_refused_at_enqueue_and_the_drain_fails(self, srv, L, draft, tmp_path):
        import yaml
        from sisteer.eval.cell_controls import controls_gen_config
        gen = controls_gen_config(yaml.safe_load((REPO / PINNED_CONTROLS_CONFIG).read_text(encoding="utf-8")))
        assert {"arm", "do_sample", "max_new_tokens", "seed_base", "trial"} <= set(gen) and set(SIX) <= set(gen)   # the row the helper used to write
        row, volroot, root, attempt_id, arv, prod = self._world(srv, L, draft, tmp_path)
        adapter = self._adapter(srv, L, prod, attempt_id, root, arv, lambda arm, s_: dict(gen))
        with pytest.raises(srv.LadderUnavailable, match=r"exactly the frozen sampler keys .*extra \['arm', 'do_sample', 'max_new_tokens', 'seed_base', 'trial'\]"):
            srv.StreamTailer(root, adapter, prod, poll_s=10).drain()
        assert prod.entries == []

    def test_the_helper_projects_the_row_and_the_at_cap_event_lands_in_the_2048_queue(self, srv, L, draft, tmp_path):
        H = _load_helper()
        out = tmp_path / "sampling.json"
        assert H.main(["controls-sampling", "--config", str(REPO / PINNED_CONTROLS_CONFIG), "--out", str(out)]) == 0   # the app's subcommand, the REAL resolver
        sampling = json.loads(out.read_text())
        assert list(sampling) == list(L.sampler_keys()) == SIX
        assert sampling == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
        row, volroot, root, attempt_id, arv, prod = self._world(srv, L, draft, tmp_path)
        asked = []
        adapter = self._adapter(srv, L, prod, attempt_id, root, arv, lambda arm, s_: asked.append(arm) or dict(sampling))
        summary = srv.StreamTailer(root, adapter, prod, poll_s=10).drain()
        assert summary["enqueued"] == 1 and summary["complete"] and summary["problems"] == [] and asked == ["arditi_allpos"]
        entries = prod.seal()
        assert len(entries) == 1
        rec = json.loads((volroot / entries[0]["path"]).read_text())
        assert rec["sampling_seed"] == 17 and int(rec["cap_next"]) == 2048 == int(row["ladder"]["cap_next"]) and int(rec["base_cap"]) == 512
        assert rec["kind"] == "controls" and rec["arm"] == "arditi_allpos" and rec["s"] == 1 and rec["redraw"]["sampling"] == sampling
        assert L.check_redraw_sampling(rec, where="fa2") == sampling

    def test_the_helper_refuses_a_row_missing_a_frozen_key_by_name(self, tmp_path, capsys):
        H = _load_helper()
        cfgp = tmp_path / "c.yaml"
        cfgp.write_text("attacks: {target_max_new_tokens: 512}\n")
        out = tmp_path / "s.json"
        assert H.controls_sampling(cfgp, out, gen_of=lambda cfg: {"temperature": 0.7, "top_p": 0.8, "max_new_tokens": 512}) == 3
        assert "lacks frozen sampler key(s) ['top_k', 'min_p', 'presence_penalty', 'repetition_penalty']" in capsys.readouterr().err and not out.exists()
        assert H.controls_sampling(cfgp, out, gen_of=lambda cfg: {**dict(zip(SIX, (0.7, 0.8, 20, 0.0, 1.5, 1.0))), "arm": "x", "trial": 0}) == 0
        assert list(json.loads(out.read_text())) == SIX


# --------------------------------------------------------------------------- F-AH (fa2c): the misalign-messages helper iterated load_items' TUPLE ---- #
# The S3 misalign s1 row ran its 5 children rc 0 and died LADDER_TAILER_ERROR: every store's `[tailer] REFUSED .../responses.jsonl:
# LadderUnavailable: misalign-messages helper rc 1 ... AttributeError: 'list' object has no attribute 'get'` -- the helper iterated
# `mr.load_items(family, behaviors)`, which returns `(items, suite_revision)` (the runner unpacks it; the helper never did). The helper now
# unpacks the tuple and logs the suite revision to stderr (the queue record has no revision slot: `misalign_queue_record`'s identity is the
# row's own keys and `redraw` is {vmodel, messages, sampling}); its output contract `{cell|behavior|item: messages}` is unchanged.

MISALIGN_CELL_H = "si_constv2_mix_full__lora_r4_ce"           # a fitted cell's HARNESS id (cells.json's namespace), si_id None
IDENT = "proxy_vllm+passthrough+interleaved"


def _cells_json(path: Path) -> Path:
    """cells.json in the app's shape (`{"cells": [...], "serving": {...}}`, `seedrep_served.py` at standup): a bare cell, a fitted cell, a prompted one."""
    path.write_text(json.dumps({"cells": [
        {"cell_id": "none", "virtual_model": "petri__base", "serving_identity": IDENT, "intervention_hash": "0" * 16, "si_id": None, "kind": "bare"},
        {"cell_id": MISALIGN_CELL_H, "virtual_model": f"petri__{MISALIGN_CELL_H}", "serving_identity": IDENT, "intervention_hash": "b" * 16, "si_id": None, "kind": "fitted"},
        {"cell_id": "prompted__safety", "virtual_model": "petri__prompted__safety", "serving_identity": IDENT, "intervention_hash": "a" * 16, "si_id": "safety", "kind": "prompted"},
    ], "serving": {"substrate": "modal", "attempt_id": "ta-1-abc", "standup_id": "smoke-S3__misalign__s1__base__shard00"}}))
    return path


class TestMisalignMessagesHelper:
    def test_load_items_is_a_tuple_and_the_helper_resolves_messages_through_the_real_runner(self, tmp_path, capsys):
        from sisteer.capeval.misalign import runner as mr
        loaded = mr.load_items("agentic", ["leaking"])
        assert isinstance(loaded, tuple) and len(loaded) == 2 and isinstance(loaded[0], list) and isinstance(loaded[1], str) and loaded[0]
        items, revision = loaded
        with pytest.raises(AttributeError, match="'list' object has no attribute 'get'"):        # the broken helper line, verbatim: the F-AH face
            {(str(it.get("behavior")), str(it["item_id"])): it for it in mr.load_items("agentic", ["leaking"])}
        by_id = {it["item_id"]: it for it in items}
        i1, i2 = items[0]["item_id"], items[1]["item_id"]
        H = _load_helper()
        cells, req, out = _cells_json(tmp_path / "cells.json"), tmp_path / "req.json", tmp_path / "msgs.json"
        req.write_text(json.dumps([{"cell_id": "none", "behavior": "leaking", "item_id": i1}, {"cell_id": MISALIGN_CELL_H, "behavior": "leaking", "item_id": i2},
                                   {"cell_id": "prompted__safety", "behavior": "leaking", "item_id": i1}]))
        assert H.main(["misalign-messages", "--family", "agentic", "--cells", str(cells), "--requests", str(req), "--out", str(out)]) == 0
        got = json.loads(out.read_text())
        assert sorted(got) == sorted([f"none|leaking|{i1}", f"{MISALIGN_CELL_H}|leaking|{i2}", f"prompted__safety|leaking|{i1}"])
        assert got[f"none|leaking|{i1}"] == mr.build_messages("agentic", by_id[i1], None)                  # byte-identical to the runner's composition
        assert got[f"{MISALIGN_CELL_H}|leaking|{i2}"] == mr.build_messages("agentic", by_id[i2], None)
        assert got[f"prompted__safety|leaking|{i1}"] == mr.build_messages("agentic", by_id[i1], mr.si_text_for("safety"))
        assert got[f"prompted__safety|leaking|{i1}"][0]["role"] == "system" and got[f"none|leaking|{i1}"][-1] == {"role": "user", "content": by_id[i1]["user_prompt"]}
        err = capsys.readouterr().err
        assert f"suite {revision}" in err and revision.startswith("agentic.v2@")                             # the provenance the container log keeps
        # an item no scenario carries: the resolved ones are written, the hole is named, rc 2 (the tailer refuses on rc != 0)
        req.write_text(json.dumps([{"cell_id": "none", "behavior": "leaking", "item_id": i1}, {"cell_id": "none", "behavior": "leaking", "item_id": "no-such-item"}]))
        assert H.main(["misalign-messages", "--family", "agentic", "--cells", str(cells), "--requests", str(req), "--out", str(out)]) == 2
        assert list(json.loads(out.read_text())) == [f"none|leaking|{i1}"] and "none|leaking|no-such-item" in capsys.readouterr().err

    def test_the_chain_misalign_store_to_the_queue_through_the_apps_real_helper_wrapper(self, srv, L, draft, tmp_path):
        """store -> MisalignStreamAdapter -> `misalign_messages_via_helper` (the app's wrapper: the helper as a SUBPROCESS, as in the container)
        -> enqueue -> drain: one queue entry whose redraw.messages are the runner's own composition for the at-cap item."""
        from sisteer.capeval.misalign import runner as mr
        items, _rev = mr.load_items("agentic", ["leaking"])
        item = items[0]
        row = _row(draft, "misalign", "s2")
        volroot, root, attempt_id, arv, _vol, prod = RF.TestP1_5_ladder_tailer()._setup(srv, L, tmp_path, row)
        d = root / "exp/misalign/agentic__s2__leaking" / MISALIGN_CELL_H
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}}))
        rng = random.Random(11)
        prose = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ,.") for _ in range(900))                 # incompressible: at cap and NOT a loop
        assert srv.zlib_ratio(prose) >= srv.MISALIGN_MIN_ZLIB
        (d / "responses.jsonl").write_text(json.dumps({"family": "agentic", "cell_id": MISALIGN_CELL_H, "virtual_model": f"petri__{MISALIGN_CELL_H}", "item_id": item["item_id"],
                                                       "sample_idx": 0, "behavior": "leaking", "prompt_sha": "p", "response": prose, "finish_reason": "length",
                                                       "usage": {}, "model_reported": "m", "seed": 7}) + "\n")
        cells = _cells_json(root / "exp/serve/cells.json") if (root / "exp/serve").mkdir(parents=True, exist_ok=True) is None else None
        env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
        adapter = srv.MisalignStreamAdapter(L, prod.spec, prod.ref, attempt_id, container_root=root, attempt_root_vol=arv,
                                            messages_of=lambda reqs: srv.misalign_messages_via_helper(sys.executable, REPO, cells, reqs, env))
        summary = srv.StreamTailer(root, adapter, prod, poll_s=10).drain()
        assert summary["enqueued"] == 1 and summary["complete"] and summary["problems"] == []
        rec = json.loads((volroot / prod.seal()[0]["path"]).read_text())
        assert rec["kind"] == "misalign" and rec["s"] == 2 and rec["sampling_seed"] == 7 and int(rec["cap_next"]) == int(row["ladder"]["cap_next"])
        assert rec["arm"] == lp.traintest_id(MISALIGN_CELL_H) and rec["redraw"]["vmodel"] == f"petri__{MISALIGN_CELL_H}"
        assert rec["redraw"]["messages"] == mr.build_messages("agentic", item, None) and len(rec["redraw"]["messages"]) == 2
        assert L.check_redraw_sampling(rec, where="fa2c")


# --------------------------------------------------------------------------------- F-AK (fa2d): the Arditi controls store's owner is looked up by LABEL -- #
# S2 controls-arditi: 3 children rc 0, then `[tailer] REFUSED .../controls_cap512__s<k>/cells/arditi_allpos__controls__t0__a0__92b66dce__<h>/
# controls_events.jsonl: LadderUnavailable: controls: no child of this row names a config for arm arditi_allpos -- its sampling row cannot be
# resolved` (x3). `ladder_adapter_for_row` indexed the row's children by their arm LABEL (`ar_c0p3__arditi_allpos`, the coverage table's arm)
# and looked the store up by the arm its cell id names (`arditi_allpos`, the driver's --cell) -- the F-C identity seam, in the tailer. The
# index is now keyed on `lp.ladder_arm_of_cell(child.cell)` + the child's seed base (the seed-owned root; the Arditi row's three children
# carry three configs), the SAME translation `ControlsStreamAdapter.records()` applies to the event's cell_id; no owner refuses by name.

W20_ARDITI_STORE_DIR = "arditi_allpos__controls__t0__a0__92b66dce__6a1274f3"          # the s2 store's cell dir, as the driver named it


class TestArditiControlsSamplingSeam:
    @staticmethod
    def _store(root: Path, s: int, cell: str, cell_dir: str, seed: int = 17) -> Path:
        d = root / f"exp/sampled/controls_cap512__s{s}/cells/{cell_dir}"
        d.mkdir(parents=True)
        (d / cev.CONTROLS_EVENTS_FILENAME).write_text(json.dumps({**_at_cap_event(seed), "cell_id": cell}) + "\n")
        return d / cev.CONTROLS_EVENTS_FILENAME

    @staticmethod
    def _wire(srv, L, row, prod, root, arv, attempt_id, vmodels):
        """The app's REAL wiring (`ladder_adapter_for_row`): the sampling row through `controls_sampling_via_helper` (the helper as a subprocess on
        the child's real config, as in the container); only the item prompt is injected (`controls_prompt_resolver`: the corpora are not loaded)."""
        env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
        ctx = {"controls_prompt_resolver": lambda c, i, a: (f"prompt-{i}", None)}
        return srv.ladder_adapter_for_row(L, row, ctx, producer=prod, root=root, attempt_root_vol=arv, attempt_id=attempt_id, vmodel_by_harness=vmodels,
                                          si_by_harness={}, driver_python=sys.executable, tree=REPO, env=env)

    def test_ladder_arm_of_cell_is_the_one_translation_and_indexes_every_controls_child(self, draft):
        assert lp.ladder_arm_of_cell(lp.ARDITI_CELL) == "arditi_allpos" == lp.ARDITI_CELL != lp.ARDITI_ARM_ID           # the seam: cell, never the label
        assert lp.ladder_arm_of_cell("none") == "none" and lp.ladder_arm_of_cell("si_constv2_mix_full__lora_r4_ce") == "constv2-mix-full__lora_r4_ce"
        assert lp.ladder_arm_of_cell("constv2-mix-full__lora_r4_ce") == "constv2-mix-full__lora_r4_ce"
        rows = [r for r in draft["standups"] if r["leg"] == "controls" and r["kind"] == "base" and not r.get("dose_standup")]
        assert len(rows) == 4
        for r in rows:
            keys = [(lp.ladder_arm_of_cell(k["cell"]), int(k["s"])) for k in r["children"]]
            assert len(set(keys)) == len(keys) == len(r["children"])                       # one owner per (arm, seed base): the seed-owned root
            if r["engine"] == lp.ARDITI:
                assert keys == [("arditi_allpos", 1), ("arditi_allpos", 2), ("arditi_allpos", 3)] and {k["arm"] for k in r["children"]} == {lp.ARDITI_ARM_ID}
                assert len({k["config"] for k in r["children"]}) == 3                            # three configs: the index MUST carry the seed
            else:
                assert keys == [(k["arm"], int(k["s"])) for k in r["children"]]                   # a view row: arm == cell already

    def test_the_arditi_store_feeds_the_2048_queue_through_the_real_wiring(self, srv, L, draft, tmp_path):
        row = _row(draft, "controls", "arditi")
        child = next(k for k in row["children"] if int(k["s"]) == 2)
        assert child["arm"] == lp.ARDITI_ARM_ID and child["cell"] == lp.ARDITI_CELL and (REPO / child["config"]).is_file(), child
        volroot, root, attempt_id, arv, _vol, prod = RF.TestP1_5_ladder_tailer()._setup(srv, L, tmp_path, row)
        self._store(root, 2, lp.ARDITI_CELL, W20_ARDITI_STORE_DIR)
        adapter = self._wire(srv, L, row, prod, root, arv, attempt_id, {"arditi_allpos": "petri__arditi_allpos", "none": "petri__base"})
        summary = srv.StreamTailer(root, adapter, prod, poll_s=10).drain()
        assert summary["enqueued"] == 1 and summary["complete"] and summary["problems"] == [], summary
        rec = json.loads((volroot / prod.seal()[0]["path"]).read_text())
        assert rec["arm"] == "arditi_allpos" and rec["s"] == 2 and rec["sampling_seed"] == 17 and int(rec["cap_next"]) == 2048 == int(row["ladder"]["cap_next"])
        assert rec["redraw"]["vmodel"] == "petri__arditi_allpos" and sorted(rec["redraw"]["sampling"]) == sorted(SIX)      # canonical JSON: sorted keys
        assert rec["redraw"]["sampling"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
        assert rec["arm_s"] == L.arm_s("arditi_allpos", 2)                                        # the slot the rung consumer filters by the driver's defense id

    def test_a_store_no_child_owns_refuses_by_name(self, srv, L, draft, tmp_path):
        row = _row(draft, "controls", "arditi")
        volroot, root, attempt_id, arv, _vol, prod = RF.TestP1_5_ladder_tailer()._setup(srv, L, tmp_path, row)
        self._store(root, 2, lp.ARDITI_CELL, W20_ARDITI_STORE_DIR)
        # the s2 store with a row whose children cover seed bases 1 and 3 only: the owner is named, not guessed from another seed's config
        orphan = {**row, "children": [k for k in row["children"] if int(k["s"]) != 2]}
        adapter = self._wire(srv, L, orphan, prod, root, arv, attempt_id, {"arditi_allpos": "petri__arditi_allpos"})
        with pytest.raises(srv.LadderUnavailable, match=r"no child of this row owns the store for arm arditi_allpos at seed base 2 .*\('arditi_allpos', 1\).*\('arditi_allpos', 3\)"):
            srv.StreamTailer(root, adapter, prod, poll_s=10).drain()
        assert prod.entries == []
        # an unknown cell in the store: refused by name too (an arm the plan never composed)
        _volroot2, root2, attempt_id2, arv2, _vol2, prod2 = RF.TestP1_5_ladder_tailer()._setup(srv, L, tmp_path / "u", row)
        self._store(root2, 2, "arditi_somewhere_else", "arditi_somewhere_else__controls__t0__a0__92b66dce__6a1274f3")
        adapter2 = self._wire(srv, L, row, prod2, root2, arv2, attempt_id2, {"arditi_allpos": "petri__arditi_allpos"})
        with pytest.raises(srv.LadderUnavailable, match="no child of this row owns the store for arm arditi_somewhere_else at seed base 2"):
            srv.StreamTailer(root2, adapter2, prod2, poll_s=10).drain()

    def test_the_view_row_is_unchanged_through_the_real_wiring(self, srv, L, draft, tmp_path):
        row = _row(draft, "controls", "s1")
        child = next(k for k in row["children"] if k["cell"] == "none")
        assert child["arm"] == "none" and int(child["s"]) == 1 and (REPO / child["config"]).is_file(), child
        volroot, root, attempt_id, arv, _vol, prod = RF.TestP1_5_ladder_tailer()._setup(srv, L, tmp_path, row)
        self._store(root, 1, "none", f"none__controls__t0__a0__{'c' * 8}__{'d' * 8}", seed=5)
        adapter = self._wire(srv, L, row, prod, root, arv, attempt_id, {"none": "petri__base"})
        summary = srv.StreamTailer(root, adapter, prod, poll_s=10).drain()
        assert summary["enqueued"] == 1 and summary["problems"] == []
        rec = json.loads((volroot / prod.seal()[0]["path"]).read_text())
        assert rec["arm"] == "none" and rec["s"] == 1 and rec["sampling_seed"] == 5 and sorted(rec["redraw"]["sampling"]) == sorted(SIX) and rec["redraw"]["vmodel"] == "petri__base"
