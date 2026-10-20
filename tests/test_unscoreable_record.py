"""unscoreable_items.json — the store SAYS why it has no aggregate and no gen_stats.json.

A served MMLU store whose driver refused a partial aggregate (one answer letter below the server's
top-K logprobs) carries item rows, no ``__aggregate__`` row and no gen_stats.json — and so does a
store whose job was killed. These tests pin the record that tells them apart in the store dir
(``sisteer.capeval.unscoreable``), the reader that tells "complete" from "refused with N of M
scored" from "undeterminable items" (the driver's own ``topk_undeterminable`` rows: an attempted run,
never "never ran") from "partial, no record" (a run cut short; the store does not say why) from
"never ran" (no MMLU evidence at all), and the cell_facts.json fragment the cell sbatch carries, per
MMLU leg (``mmlu/cells`` and ``mmlu_full/cells`` are two store roots, reported separately) — including
its DECLARATION of whether the capability driver on this tree writes the record at all, derived from
the driver's own source rather than typed. All fixtures synthetic.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval import unscoreable as unsc  # noqa: E402

AR_CELL = REPO / "scripts" / "arditi" / "sbatch" / "ar_cell.sbatch"
DRIVER = REPO / "scripts" / "served_capability_run.py"

IDENTITY = {"cell_id": "arditi", "harness_cell_id": "arditi",
            "intervention_hash": "ab" * 8, "harness_config_hash": "cd" * 8,
            "model_id": "test-org/test-model", "model_revision": "11" * 20, "engine": "vllm"}
STORE_KEY = "arditi__controls__t0__a0__abababab__cdcdcdcd"
NONE_KEY = "none__controls__t0__a0__efefefef__cdcdcdcd"
JOB = "424242"
GAP = 19            # the unscored position
N_REQ, N_CORRECT = 200, 142


def _item_id(pos: int) -> str:
    return f"m{pos:04d}-{pos:08x}"


VICTIM = _item_id(GAP)


def _row(pos, item_id, value):
    return {"task": unsc.MMLU_TASK, "metric": "acc", "item_id": item_id, "value": float(value),
            "position_order": pos, "engine": "vllm", "arm_name": "arditi",
            "extra_json": json.dumps({"pick": "A", "gold": "A" if value else "B",
                                      "nlls": [0.1, 2.0, 3.0, 4.0], "top_logprobs": 64})}


def _urow(pos, item_id):
    """The driver's row for an item whose EVERY letter fell outside the top-K (served_capability_run.py
    `_run_mmlu_served`): metric topk_undeterminable, value 1.0, the letters / ids / window in extra."""
    return {"task": unsc.MMLU_TASK, "metric": unsc.UNDETERMINABLE_METRIC, "item_id": item_id, "value": 1.0,
            "position_order": pos, "engine": "vllm", "arm_name": "arditi",
            "extra_json": json.dumps({"letter_read": "all_absent", "gold": "C", "letters": ["A", "B", "C", "D"],
                                      "letter_ids": [362, 426, 356, 422], "present_letter_logprobs": {},
                                      "top_logprobs": 64, "min_returned_logprob": -14.2})}


def _store(root: Path, *, key=STORE_KEY, arm="arditi", n_requested=N_REQ, gaps=(GAP,),
           n_correct=N_CORRECT, rows=True, aggregate=False, gen_stats=False, leg="mmlu",
           undeterminable=(), undeterminable_then_scored=()) -> Path:
    """A synthetic MMLU cell store in the driver's layout under ``<root>/<leg>/cells/<key>``: manifest.json +
    evidence.json always; rows.jsonl with an acc row for every position but ``gaps`` and ``undeterminable``
    (those positions get the driver's topk_undeterminable row instead; ``undeterminable_then_scored``
    positions get BOTH, the resumed run's acc row after the earlier attempt's undeterminable row);
    optionally the aggregate row and gen_stats.json."""
    d = root / leg / "cells" / key
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(
        {**IDENTITY, "cell_id": arm, "harness_cell_id": arm,
         "row_constants": {"arm_name": arm, "slurm_job_id": JOB}}, indent=1))
    (d / "evidence.json").write_text(json.dumps({"driver": "served_capability_run_v2"}))
    if rows:
        scored = [p for p in range(n_requested) if p not in gaps and p not in undeterminable]
        lines = [_urow(p, _item_id(p)) for p in sorted(set(undeterminable) | set(undeterminable_then_scored))]
        lines += [_row(p, _item_id(p), 1.0 if i < n_correct else 0.0)
                  for i, p in enumerate(scored)]
        if aggregate:
            lines.append({"task": unsc.MMLU_TASK, "metric": "acc_aggregate",
                          "item_id": unsc.AGGREGATE_ITEM_ID,
                          "value": n_correct / len(scored), "position_order": None})
        (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    if gen_stats:
        (d / "gen_stats.json").write_text(json.dumps(
            {"mmlu": {"acc": n_correct / n_requested, "n": n_requested, "top_logprobs": 64,
                      "response_prefill": True, "wall_clock_s": 1.0}}))
    return d


def _block(*, n_scored=N_REQ - 1, n_correct=N_CORRECT, item_id=VICTIM, pos=GAP):
    return unsc.mmlu_failure_block(
        top_logprobs=64, n_requested=N_REQ, n_scored=n_scored, n_correct_scored=n_correct,
        items=[unsc.letter_item(item_id, position_order=pos, missing_letter_token_ids=[622],
                                missing_letters=["D"], gold="C",
                                present_letter_logprobs={"A": -9.1, "B": -8.4, "C": -0.01},
                                message="letter token id(s) [622] absent from the top-64")])


# --------------------------------------------------------------------------- #
# the record                                                                   #
# --------------------------------------------------------------------------- #


def test_the_task_constants_match_their_owners():
    from sisteer.capeval.tiers import parity

    assert unsc.MMLU_TASK == parity.PARITY_TASK_MMLU
    assert unsc.AGGREGATE_ITEM_ID == "__aggregate__"
    # the driver's undeterminable-row metric, read off its source (the driver imports vllm-side modules; this reader stays stdlib-only)
    import re
    m = re.search(r'^MMLU_UNDETERMINABLE_METRIC = "([^"]+)"$', DRIVER.read_text(encoding="utf-8"), re.M)
    assert m and m.group(1) == unsc.UNDETERMINABLE_METRIC == "topk_undeterminable", m
    assert unsc.STATUSES == ("complete", "refused_unscoreable", "undeterminable_items", "partial_no_record", "never_ran", "inconsistent")
    assert unsc.MMLU_LEGS == ("mmlu", "mmlu_full")


def test_the_failure_block_carries_exact_bounds_and_the_why_names_gen_stats():
    blk = _block()
    assert (blk["n_requested"], blk["n_scored"], blk["n_unscored"]) == (200, 199, 1)
    assert blk["top_logprobs"] == 64
    assert blk["acc_over_scored"] == pytest.approx(142 / 199)
    assert blk["acc_bounds_over_requested"] == [142 / 200, 143 / 200] == [0.71, 0.715]
    assert blk["kinds"] == {unsc.KIND_LETTER_OUTSIDE_TOP_K: 1}
    assert blk["unscoreable_item_ids"] == [VICTIM]
    [item] = blk["items"]
    assert item["missing_letter_token_ids"] == [622] and item["missing_letters"] == ["D"]
    assert item["position_order"] == GAP and item["gold"] == "C"
    why = unsc.why_no_aggregate("mmlu", blk)
    for needle in ("gen_stats.json", "__aggregate__", "1 of 200", "top_logprobs=64",
                   "199 scored items are in rows.jsonl"):
        assert needle in why, (needle, why)


@pytest.mark.parametrize("kw, match", [
    (dict(n_scored=200), "nothing unscored"),                 # a complete leg files its aggregate
    (dict(n_scored=198), "names 1 failed item.*but 2"),       # every unscored item must be named
    (dict(n_correct=200), "n_correct_scored 200 outside"),   # more correct than scored
])
def test_the_block_refuses_counts_that_do_not_add_up(kw, match):
    with pytest.raises(unsc.UnscoreableError, match=match):
        _block(**kw)


def test_write_record_keeps_history_instead_of_rewriting(tmp_path):
    d = _store(tmp_path)
    p1 = unsc.write_record(d, written_by=unsc.WRITTEN_BY_DRIVER, identity=IDENTITY,
                           controls={"mmlu": _block()})
    assert p1 == d / unsc.RECORD_NAME
    first = json.loads(p1.read_text())
    assert "previous_attempts" not in first
    unsc.write_record(d, written_by=unsc.WRITTEN_BY_DRIVER, identity=IDENTITY,
                      controls={"mmlu": _block(n_correct=141)})
    second = json.loads(p1.read_text())
    assert second["controls"]["mmlu"]["n_correct_scored"] == 141
    assert [a["controls"]["mmlu"]["n_correct_scored"] for a in second["previous_attempts"]] == [142]
    assert "previous_attempts" not in second["previous_attempts"][0]
    unsc.write_record(d, written_by=unsc.WRITTEN_BY_DRIVER, identity=IDENTITY,
                      controls={"mmlu": _block(n_correct=140)})
    third = json.loads(p1.read_text())
    assert [a["controls"]["mmlu"]["n_correct_scored"] for a in third["previous_attempts"]] == [142, 141]
    # only the record was ever written into the store dir
    assert sorted(p.name for p in d.iterdir()) == ["evidence.json", "manifest.json", "rows.jsonl",
                                                    unsc.RECORD_NAME]


# --------------------------------------------------------------------------- #
# the reader                                                                   #
# --------------------------------------------------------------------------- #


def test_a_reader_tells_refused_with_n_of_m_from_never_ran_and_from_partial(tmp_path):
    """THE distinction the record exists for: three stores with no gen_stats.json, three
    different answers, none of them inferred from the absent file."""
    refused = _store(tmp_path / "refused")
    unsc.write_record(refused, written_by=unsc.WRITTEN_BY_DRIVER, identity=IDENTITY,
                      controls={"mmlu": _block()})
    never = _store(tmp_path / "never", rows=False)          # manifest + evidence only
    partial = _store(tmp_path / "partial")                   # rows, nothing else

    r = unsc.mmlu_store_status(refused)
    assert r["status"] == unsc.STATUS_REFUSED
    assert (r["n_scored"], r["n_requested"], r["n_correct_scored"]) == (199, 200, 142)
    assert r["acc"] == pytest.approx(142 / 199)
    assert r["acc_bounds_over_requested"] == [0.71, 0.715]
    assert r["top_logprobs"] == 64 and r["unscoreable_item_ids"] == [VICTIM]
    assert r["record_written_by"] == unsc.WRITTEN_BY_DRIVER and r["gen_stats"] is None
    assert "gen_stats.json" in r["why"] and "1 of 200" in r["why"]

    n = unsc.mmlu_store_status(never)
    assert n["status"] == unsc.STATUS_NEVER_RAN
    assert (n["n_scored"], n["record"], n["gen_stats"]) == (0, None, None)
    assert "never scored or attempted an item" in n["why"] and "neither acc nor topk_undeterminable" in n["why"]
    assert (n["n_undeterminable"], n["undeterminable_item_ids"]) == (0, [])

    p = unsc.mmlu_store_status(partial)
    assert p["status"] == unsc.STATUS_PARTIAL_NO_RECORD
    assert (p["n_scored"], p["n_requested"], p["record"]) == (199, None, None)
    assert unsc.RECORD_NAME in p["why"]
    # an unrecorded partial run's cause is UNKNOWN and is reported as such: the store does not say, and the reader does not guess
    assert "does not record why" in p["why"] and "other than" not in p["why"], p["why"]
    assert (p["n_undeterminable"], p["undeterminable_item_ids"]) == (0, [])
    assert unsc.mmlu_store_status(partial, n_requested=200)["n_requested"] == 200

    assert len({r["status"], n["status"], p["status"]}) == 3


def test_undeterminable_rows_are_mmlu_evidence_of_an_attempted_run_and_establish_why_the_aggregate_is_absent(tmp_path):
    """The driver files a `topk_undeterminable` row for an item no letter of which the top-K carried, and no aggregate while one exists.
    A store carrying such rows was RUN: with no acc row at all it is not `never_ran`; with some acc rows its missing aggregate is not
    "cut short by something other than an unscoreable item" — the rows establish the cause, and the reader names the items."""
    # every item undeterminable: zero acc rows, N undeterminable rows -> an attempted run, the cause established, never `never_ran`
    all_undet = _store(tmp_path / "all", gaps=(), undeterminable=tuple(range(N_REQ)))
    a = unsc.mmlu_store_status(all_undet)
    assert a["status"] == unsc.STATUS_UNDETERMINABLE_ITEMS and a["status"] != unsc.STATUS_NEVER_RAN
    assert (a["n_scored"], a["n_undeterminable"], a["record"], a["gen_stats"]) == (0, N_REQ, None, None)
    assert a["undeterminable_item_ids"][:3] == [_item_id(0), _item_id(1), _item_id(2)] and len(a["undeterminable_item_ids"]) == N_REQ   # sorted, every one named
    assert "established by the rows" in a["why"] and "every answer letter outside" in a["why"] and "not recorded" in a["why"], a["why"]
    assert "never" not in a["why"]
    # some acc rows and two undeterminable rows: the same status, the two items named, the scored count intact
    some = _store(tmp_path / "some", gaps=(), undeterminable=(GAP, GAP + 1))
    s = unsc.mmlu_store_status(some)
    assert s["status"] == unsc.STATUS_UNDETERMINABLE_ITEMS
    assert (s["n_scored"], s["n_undeterminable"], s["undeterminable_item_ids"]) == (N_REQ - 2, 2, [_item_id(GAP), _item_id(GAP + 1)])
    assert f"2 {unsc.UNDETERMINABLE_METRIC} rows" in s["why"] and _item_id(GAP) in s["why"] and "other than" not in s["why"], s["why"]
    assert s["n_requested"] is None and unsc.mmlu_store_status(some, n_requested=N_REQ)["n_requested"] == N_REQ
    # an item undeterminable on a FIRST attempt and scored on the resumed run: it is a scored item (its old row is history), and with
    # no unscored undeterminable item left and no aggregate the store is a partial run of unrecorded cause, never an undeterminable one
    resumed = _store(tmp_path / "resumed", gaps=(GAP,), undeterminable_then_scored=(GAP + 1,))
    r = unsc.mmlu_store_status(resumed)
    assert r["status"] == unsc.STATUS_PARTIAL_NO_RECORD and (r["n_scored"], r["n_undeterminable"]) == (N_REQ - 1, 0), r
    # the rows reader sorts the kinds
    rows = unsc.mmlu_rows(some)
    assert (len(rows["acc"]), len(rows["undeterminable"]), rows["has_aggregate"]) == (N_REQ - 2, 2, False)
    assert unsc.mmlu_item_rows(some) == (rows["acc"], False)
    # a COMPLETE store that carries an undeterminable row from an earlier attempt (the resumed run scored it) stays complete
    done = _store(tmp_path / "done", gaps=(), aggregate=True, gen_stats=True, undeterminable_then_scored=(3,))
    c = unsc.mmlu_store_status(done)
    assert c["status"] == unsc.STATUS_COMPLETE and c["n_undeterminable"] == 0
    # a record present beside undeterminable rows: the record governs (refused_unscoreable), the row count is reported beside it
    rec = _store(tmp_path / "rec", gaps=(), undeterminable=(GAP,))
    unsc.write_record(rec, written_by=unsc.WRITTEN_BY_DRIVER, identity=IDENTITY, controls={"mmlu": _block()})
    g = unsc.mmlu_store_status(rec)
    assert g["status"] == unsc.STATUS_REFUSED and g["n_undeterminable"] == 1 and g["undeterminable_item_ids"] == [VICTIM]


def test_a_complete_store_reads_complete_and_an_old_record_is_history(tmp_path):
    d = _store(tmp_path, gaps=(), aggregate=True, gen_stats=True)
    c = unsc.mmlu_store_status(d)
    assert c["status"] == unsc.STATUS_COMPLETE
    assert (c["n_scored"], c["n_requested"], c["acc"]) == (200, 200, pytest.approx(0.71))
    assert c["record"] is None
    # a record left by an earlier failed attempt does not demote a completed store
    unsc.write_record(d, written_by=unsc.WRITTEN_BY_DRIVER, identity=IDENTITY,
                      controls={"mmlu": _block()})
    c2 = unsc.mmlu_store_status(d)
    assert c2["status"] == unsc.STATUS_COMPLETE and "history" in c2["why"]


def test_files_that_disagree_read_inconsistent_not_a_guess(tmp_path):
    # the record says 199 scored, rows.jsonl carries 198
    d = _store(tmp_path / "a", gaps=(GAP, GAP + 1))
    unsc.write_record(d, written_by=unsc.WRITTEN_BY_DRIVER, identity=IDENTITY,
                      controls={"mmlu": _block()})
    s = unsc.mmlu_store_status(d)
    assert s["status"] == unsc.STATUS_INCONSISTENT and "198" in s["why"]
    # an aggregate row with no gen_stats.json
    d2 = _store(tmp_path / "b", gaps=(), aggregate=True)
    assert unsc.mmlu_store_status(d2)["status"] == unsc.STATUS_INCONSISTENT


# --------------------------------------------------------------------------- #
# cell_facts.json                                                              #
# --------------------------------------------------------------------------- #


def test_the_cell_facts_fragment_states_every_store_per_leg(tmp_path):
    root = tmp_path / "controls" / "TESTTAG_t0"
    ard = _store(root)
    unsc.write_record(ard, written_by=unsc.WRITTEN_BY_DRIVER, identity=IDENTITY,
                      controls={"mmlu": _block()})
    _store(root, key=NONE_KEY, arm="none", gaps=(), aggregate=True, gen_stats=True)
    facts = unsc.mmlu_stores_facts(root)
    assert facts["record"] == unsc.RECORD_NAME and set(facts["legs"]) == {"mmlu", "mmlu_full"}
    mm = facts["legs"]["mmlu"]
    assert mm["present"] is True and mm["root"] == "mmlu/cells" and set(mm["stores"]) == {STORE_KEY, NONE_KEY}
    assert mm["stores"][STORE_KEY]["arm"] == "arditi"
    assert mm["stores"][STORE_KEY]["status"] == unsc.STATUS_REFUSED
    assert mm["stores"][STORE_KEY]["n_scored"] == 199
    assert mm["stores"][NONE_KEY]["arm"] == "none"
    assert mm["stores"][NONE_KEY]["status"] == unsc.STATUS_COMPLETE
    # the whole-pool leg did not run under this root: said so, separately, never folded into the 200-item leg's stores
    full = facts["legs"]["mmlu_full"]
    assert full["present"] is False and full["root"] == "mmlu_full/cells" and full["stores"] == {} and "mmlu_full leg did not run" in full["why"]
    assert facts["statuses"] == list(unsc.STATUSES) and unsc.STATUS_UNDETERMINABLE_ITEMS in facts["statuses"]
    # the fragment is JSON-serialisable as cell_facts.json needs it
    json.dumps(facts)
    empty = unsc.mmlu_stores_facts(tmp_path / "controls" / "no-mmlu-leg_t0")
    assert all(empty["legs"][leg]["present"] is False and empty["legs"][leg]["stores"] == {} for leg in unsc.MMLU_LEGS)
    # the full-pool leg alone (legs=mmlu_full writes under <root>/mmlu_full/cells): its stores appear under ITS leg, with the 1,531-item
    # requested count the caller supplies for that leg only, and the 200-item leg reads absent
    froot = tmp_path / "controls" / "FULLTAG_t0"
    _store(froot, leg="mmlu_full", key=STORE_KEY, n_requested=1531, gaps=(7,), n_correct=1000)
    _store(froot, leg="mmlu_full", key=NONE_KEY, arm="none", n_requested=1531, gaps=(), n_correct=1100, aggregate=True, gen_stats=True)
    ff = unsc.mmlu_stores_facts(froot, n_requested_by_leg={"mmlu_full": 1531})
    assert ff["legs"]["mmlu"]["present"] is False and ff["legs"]["mmlu"]["stores"] == {}
    fl = ff["legs"]["mmlu_full"]
    assert fl["present"] is True and set(fl["stores"]) == {STORE_KEY, NONE_KEY}
    assert fl["stores"][STORE_KEY]["status"] == unsc.STATUS_PARTIAL_NO_RECORD and (fl["stores"][STORE_KEY]["n_scored"], fl["stores"][STORE_KEY]["n_requested"]) == (1530, 1531)
    assert fl["stores"][NONE_KEY]["status"] == unsc.STATUS_COMPLETE and fl["stores"][NONE_KEY]["n_requested"] == 1531 and fl["stores"][NONE_KEY]["acc"] == pytest.approx(1100 / 1531)
    # both legs under one root: two store sets, neither merged into the other
    _store(froot, leg="mmlu", key=STORE_KEY, gaps=())
    both = unsc.mmlu_stores_facts(froot)
    assert set(both["legs"]["mmlu"]["stores"]) == {STORE_KEY} and set(both["legs"]["mmlu_full"]["stores"]) == {STORE_KEY, NONE_KEY}
    assert both["legs"]["mmlu"]["stores"][STORE_KEY]["n_scored"] == 200 and both["legs"]["mmlu_full"]["stores"][STORE_KEY]["n_scored"] == 1530


def test_the_cell_sbatch_writes_the_fragment_into_cell_facts_beside_the_existing_fields():
    text = AR_CELL.read_text()
    # the reader is imported from the module; the line is matched by its module and the names it
    # must bind (the record name and the refused status feed the record_writer declaration beside
    # it), never as one exact spelling
    import re
    m = re.search(r"^\s*from sisteer\.capeval\.unscoreable import (.+)$", text, re.M)
    assert m, "ar_cell.sbatch does not import from sisteer.capeval.unscoreable"
    names = [n.strip() for n in m.group(1).split(",")]
    assert {"mmlu_stores_facts", "RECORD_NAME", "STATUS_REFUSED"} <= set(names), m.group(0)
    assert "mmlu_stores = mmlu_stores_facts(ctl)" in text
    assert '"mmlu_stores": mmlu_stores,' in text
    # the writer declaration reads the driver's source, never a typed boolean
    assert '_drv_rel = "scripts/served_capability_run.py"' in text
    assert '"unscoreable" in open(_drv_rel, encoding="utf-8").read()' in text
    assert 'mmlu_stores["record_writer"] = {' in text
    # the existing fields are untouched
    for existing in ('"resolved_sampler": resolved_sampler,', '"declared_deviations": {',
                     '"static_driver_cap": int(driver_cap),', '"sampler_departure": _sampler_departure(cfg),'):
        assert existing in text, existing


# --------------------------------------------------------------------------- #
# the cell sbatch's cell_facts block, RUN over synthetic stores                 #
# --------------------------------------------------------------------------- #

sys.path.insert(0, str(REPO / "scripts" / "arditi"))
import arditi_strength as st  # noqa: E402

UNITS = (st.DOSE_UNIT, st.UNIT_RAW_COEFFICIENT)


def _cell_facts_body() -> str:
    """The python heredoc ar_cell.sbatch hands to ``"$MPY" - "$REC_DIR/cell_facts.json" …``."""
    src = AR_CELL.read_text(encoding="utf-8")
    start = src.index('"$MPY" - "$REC_DIR/cell_facts.json"')
    start = src.index("<<'PY'\n", start) + len("<<'PY'\n")
    end = src.index("\nPY\n", start)
    return src[start:end]


def _cell_facts(tmp_path: Path, unit: str, ctl: Path, *, legs: str = "all") -> dict:
    """Run the block exactly as the sbatch does (cwd = the repo root, the module on PYTHONPATH),
    with a synthetic sampled config, ``ctl`` as the controls root and ``legs`` as the legs run."""
    out = tmp_path / f"cf_{unit}_{legs}.json"
    cfgp = tmp_path / "sampled_t0.yaml"
    cfgp.write_text(yaml.safe_dump({
        "model": "test-org/test-model", "model_revision": "11" * 20,
        "sampled_decode": {"arm": "vendor_recommended", "temperature": 0.7, "top_p": 0.8,
                           "source": "synthetic vendor row"}}))
    argv = [str(out), "testtarget_ar_r0p1", "0", "0.1", "12", str(cfgp), "man.yaml", "/synthetic/view", "prun",
            str(ctl), "0" * 40, "impl", "148", "4864", "", "", "TESTHOST", legs, "/nonexistent/rs.json", "96", unit]
    r = subprocess.run([sys.executable, "-", *argv], input=_cell_facts_body(), capture_output=True, text=True,
                       cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0, (r.stdout, r.stderr)
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("unit", UNITS)
def test_the_cell_record_states_whether_the_unscoreable_WRITER_exists_on_this_tree(tmp_path, unit):
    """The reader is on the tree, so ``mmlu_stores`` is never an import-error string; and the
    record additionally DECLARES whether the driver that wrote the stores can write
    ``unscoreable_items.json`` at all — derived by reading the driver's source, so the declaration
    follows the tree instead of a comment. Both halves are asserted against an INDEPENDENT read of
    the same driver here, so the pin bites whichever way the tree moves."""
    writes = "unscoreable" in DRIVER.read_text(encoding="utf-8")
    ctl = tmp_path / "ctl"; ctl.mkdir()
    rec = _cell_facts(tmp_path, unit, ctl)
    stores = rec["mmlu_stores"]
    assert "error" not in stores, stores
    assert stores["reader"] == "sisteer.capeval.unscoreable.mmlu_store_status"
    assert set(stores["legs"]) == {"mmlu", "mmlu_full"}                  # neither MMLU leg under this root, each said so on its own
    assert all(stores["legs"][leg]["present"] is False and stores["legs"][leg]["stores"] == {} for leg in ("mmlu", "mmlu_full"))
    rw = stores["record_writer"]
    assert rw["driver"] == "scripts/served_capability_run.py"
    assert rw["record"] == "unscoreable_items.json"
    assert rw["writes_record"] is writes
    # a consequence is stated exactly when the writer is absent, and names the two statuses
    if writes:
        assert rw["consequence"] is None
    else:
        assert "refused_unscoreable" in rw["consequence"] and "partial_no_record" in rw["consequence"]


def test_the_cell_record_distinguishes_complete_partial_undeterminable_never_ran_and_refused_stores(tmp_path):
    """The cell record tells the five store states apart, each read off the store itself — the
    reason the per-store reader is worth carrying over a single "not read" field."""
    ctl = tmp_path / "ctl"
    complete = "none__controls__t0__a0__efefefef__cdcdcdcd"
    partial = "arditi__controls__t0__a0__abababab__cdcdcdcd"
    never = "arditi__controls__t0__a1__abababab__cdcdcdcd"
    refused = "arditi__controls__t0__a2__abababab__cdcdcdcd"
    undet = "arditi__controls__t0__a3__abababab__cdcdcdcd"
    _store(ctl, key=complete, arm="none", gaps=(), aggregate=True, gen_stats=True)
    _store(ctl, key=partial)
    _store(ctl, key=never, rows=False)
    _store(ctl, key=undet, gaps=(), undeterminable=(GAP,))
    unsc.write_record(_store(ctl, key=refused), written_by=unsc.WRITTEN_BY_DRIVER, identity=IDENTITY,
                      controls={"mmlu": _block()})
    rec = _cell_facts(tmp_path, st.DOSE_UNIT, ctl)
    stores = rec["mmlu_stores"]
    mm = stores["legs"]["mmlu"]
    assert mm["present"] is True and stores["legs"]["mmlu_full"]["present"] is False
    assert {k: v["status"] for k, v in mm["stores"].items()} == {
        complete: unsc.STATUS_COMPLETE, partial: unsc.STATUS_PARTIAL_NO_RECORD,
        never: unsc.STATUS_NEVER_RAN, refused: unsc.STATUS_REFUSED, undet: unsc.STATUS_UNDETERMINABLE_ITEMS}
    assert {k: v["arm"] for k, v in mm["stores"].items()} == {
        complete: "none", partial: "arditi", never: "arditi", refused: "arditi", undet: "arditi"}
    assert mm["stores"][complete]["n_requested"] == 200 and mm["stores"][complete]["acc"] == pytest.approx(0.71)
    assert (mm["stores"][partial]["n_scored"], mm["stores"][partial]["n_requested"]) == (199, None)
    assert mm["stores"][never]["n_scored"] == 0
    assert (mm["stores"][refused]["n_scored"], mm["stores"][refused]["n_requested"]) == (199, 200)
    assert mm["stores"][refused]["unscoreable_item_ids"] == [VICTIM]
    assert (mm["stores"][undet]["n_scored"], mm["stores"][undet]["n_undeterminable"], mm["stores"][undet]["undeterminable_item_ids"]) == (199, 1, [VICTIM])
    # the writer declaration rides beside the statuses, in the same fragment
    assert stores["record_writer"]["record"] == unsc.RECORD_NAME
    # the record's other fields are where they were
    assert rec["controls_root"] == str(ctl) and rec["legs_run"] == "all"


def test_the_cell_record_of_a_full_pool_run_states_its_stores_under_the_mmlu_full_leg(tmp_path):
    """`legs=mmlu_full` writes its stores under `$CTL_ROOT/mmlu_full/cells` (ar_cell.sbatch leg 1F), never under `$CTL_ROOT/mmlu`: the cell
    record reads that root and reports the leg's stores under `legs.mmlu_full`, with the 200-item leg absent — before, the fragment read only
    `mmlu/cells` and said MMLU was absent after the whole-pool leg had run."""
    ctl = tmp_path / "ctl"
    partial = "arditi__controls__t0__a0__abababab__cdcdcdcd"
    complete = "none__controls__t0__a0__efefefef__cdcdcdcd"
    _store(ctl, leg="mmlu_full", key=partial, n_requested=1531, gaps=(7,), n_correct=1000)
    _store(ctl, leg="mmlu_full", key=complete, arm="none", n_requested=1531, gaps=(), n_correct=1100, aggregate=True, gen_stats=True)
    rec = _cell_facts(tmp_path, st.DOSE_UNIT, ctl, legs="mmlu_full")
    stores = rec["mmlu_stores"]
    assert "error" not in stores, stores
    assert stores["legs"]["mmlu"]["present"] is False and stores["legs"]["mmlu"]["stores"] == {}
    full = stores["legs"]["mmlu_full"]
    assert full["present"] is True and full["root"] == "mmlu_full/cells"
    assert {k: v["status"] for k, v in full["stores"].items()} == {partial: unsc.STATUS_PARTIAL_NO_RECORD, complete: unsc.STATUS_COMPLETE}
    assert full["stores"][partial]["n_scored"] == 1530 and full["stores"][complete]["n_requested"] == 1531
    assert rec["legs_run"] == "mmlu_full" and rec["geometry"]["standup_1"]["legs"] == ["mmlu_full"]
