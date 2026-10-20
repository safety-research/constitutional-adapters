"""The H200-hour meter (``scripts/bcb_bfcl_200_meter.py``): reservations keyed
``(wave, suffix, unit)`` with the stage KIND apart — two waves' ladder probes never
replace each other's hours, ``--record`` replaces one identity's forecast only; the
admission transaction under ``fcntl.flock`` (two processes held on the lock, then
released: exactly one of two near-ceiling admissions lands); the 25 h envelope
with NO implicit ladder allowance — a raise is a recorded decision naming a fired
leg (refused without one, refused beyond 1.5 h per leg, a leg credited once;
lowering / restating free) OR an explicit ruling (``--user-ruling``: a raise
admitted without a leg and without the per-leg cap, the ruling recorded verbatim and
rendered by the report, an empty citation refused, a ruling and a leg never one line,
no leg credited; absent the flag, byte-for-byte the old rule); attempts and supersession persisted on disk; an
admission is a CLAIM: a second admission under an identity whose
launch has not recorded its actual is REFUSED — sequentially and from two processes
at once — until the claim is recorded or RELEASED (``--release``: an actual of 0.0 h
noted as released, which hands a scoring claim's superseded probes back); a
partially specified identity refuses; the ladder forecast from the MODEL's
LadderSpec (``rung_probe.ladder_spec``): a probe reservation per fitted cell at the
worst-case rung with a full walk PLUS one re-probe standup and no scoring, the
scoring admission at the common S over per-cell certificates forecasting the
SCORING ONLY (probe and scoring spend are independent records — a
common-rung re-probe certificate or a scoring actual, in either order, never makes
incurred probe spend vanish; writer-produced certificates), ONE cell's scoring
(``score:<cell>``) as its own claim, the probe reservations' state read UNDER the
admission lock, a tree without ``ladder_spec`` refusing by name; a legacy stage-only
ledger still reads; the report grouped by wave. Pure: a JSONL ledger under tmp_path."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple, Tuple

import pytest

import test_tinycap_ladder as F  # noqa: E402  - the shared certificate builder (schema 2)
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import rung_probe as RP

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bcb_bfcl_200_meter.py"


def _load():
    spec = importlib.util.spec_from_file_location("bcb_bfcl_200_meter", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bcb_bfcl_200_meter"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


M = _load()
BCB, BFCL = C.inspect_task("bigcodebench_200"), C.inspect_task("bfcl_200")
NONE, LORA, PSR = "none", "constv2-mix-full__lora_r4_ce", "constv2-mix-full__psr_ce_all"
CELLS = [NONE, LORA, PSR]
PIN_IDS = {BCB: list(C.load_bcb_pin(REPO)["ids"]), BFCL: list(C.load_bfcl_pin(REPO)["ids"])}
W, SFX = "gemma4e4b_it", "bcbw-gemma4e4b_it-abc1234"


class _Spec(NamedTuple):
    """The `rung_probe.LadderSpec` interface: model key, budget, window, rung list, identity S."""
    model_key: str
    budget: object
    window: int
    seqs: Tuple[int, ...]
    identity_S: int


def _stub_spec(model_key: str) -> _Spec:
    if model_key == "gemma4e4b":
        return _Spec("gemma4e4b", RP.ladder_budget(), 32_768, (18, 16, 14, 12), 18)
    if model_key == "nemotron3nano4b":
        return _Spec("nemotron3nano4b", None, 32_768, (24, 22, 20, 18), 24)
    raise ValueError(f"{model_key}: no MEASURED budget row (provisional)")


@pytest.fixture()
def spec(monkeypatch):
    """The real `ladder_spec` when Slice B has landed it; the plan-shaped stub otherwise."""
    if not hasattr(RP, "ladder_spec"):
        monkeypatch.setattr(RP, "ladder_spec", _stub_spec, raising=False)
    return RP.ladder_spec


def _selection_doc(counts: dict, *, cap_to: int = 16_384, max_prompt: int = 1200, cells=CELLS) -> dict:
    selection = {}
    for t, n in counts.items():
        ids = PIN_IDS[t][:n]
        selection[t] = {"ids": ids, "n": n, "n_selected": n, "ids_sha256": C.ids_sha256(ids), "cap_from": 4096, "cap_to": cap_to,
                        "max_prompt_tokens_rung1": max_prompt}
    return {"schema": 1, "cells": list(cells), "detector": {}, "sources": [], "selection": selection, "selection_sha256": RP.selection_digest(selection)}


def _selection(path: Path, counts: dict, **kw) -> Path:
    path.write_text(json.dumps(_selection_doc(counts, **kw)))
    return path


_KV_AFTER = {18: 528_000, 16: 500_000, 14: 400_000}


def _certificate(path: Path, *, seqs: int, walked_n: int, selection: dict, probed_cell: str = LORA, standup_s: float = 330.0, **overrides) -> Path:
    """A VALID schema-2 `rung_probe.<cell>.json` (`rung_probe.certificate_is_valid`) over ``selection``: a legal walk of
    ``walked_n`` rungs ending at ``seqs``, the model's ladder block, ``probed_cell`` = the fitted cell it certifies."""
    rungs = list(RP.LADDER_SEQS)
    walked = []
    for k in range(walked_n - 1):
        S = rungs[k]
        walked.append({"S": S, "outcome": "refused", "kv_tokens": _KV_AFTER[S],
                       "refusal_tail": f"KV cache holds {_KV_AFTER[S]} tokens < the {S * RP.LADDER_WINDOW}-token budget (= {S} x {RP.LADDER_WINDOW})"})
    walked.append({"S": seqs, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None})
    identity = seqs == rungs[0]
    F.make_certificate(path, selection, accepted=(RP.LADDER_WINDOW, seqs), rung_source=("identity" if identity else "probe"),
                       seqs_source=("model_budget" if identity else "override"), suffix=SFX + "-cap16k", walked=walked,
                       object_digests={LORA: "b" * 64, PSR: "c" * 64})
    cert = json.loads(path.read_text())
    cert.update(probed_cell=probed_cell, standup_s=standup_s, **overrides)
    path.write_text(json.dumps(cert))
    return path


def _res(unit: str, hours: float, *, kind: str = "wave", wave: str = W, suffix: str = SFX, entry_kind: str = "forecast") -> dict:
    return {"kind": entry_kind, "wave": wave, "suffix": suffix, "unit": unit, "stage_kind": kind, "hours": hours}


def _ledger_with(tmp_path, *entries):
    p = tmp_path / "gpu_ledger.jsonl"
    for e in entries:
        M.append_entry(p, e)
    return p


# --------------------------------------------------------------------------- #
# reservation identity                                                         #
# --------------------------------------------------------------------------- #


def test_reservations_are_keyed_by_wave_suffix_unit_with_the_kind_apart(tmp_path):
    p = _ledger_with(tmp_path, _res(f"probe:{LORA}", 0.4, kind="ladder-probe"),
                     _res(f"probe:{LORA}", 0.3, kind="ladder-probe", wave="gemma4e4b_main", suffix="bcbw-gemma4e4b_main-abc1234"),
                     _res(NONE, 0.18), _res("smoke", 0.55, kind="smoke"))
    table = M.stage_table(M.read_ledger(p))
    a, b = f"{W}/{SFX}/probe:{LORA}", f"gemma4e4b_main/bcbw-gemma4e4b_main-abc1234/probe:{LORA}"
    assert a in table and b in table and table[a]["counted"] == 0.4 and table[b]["counted"] == 0.3         # the second wave's probe did NOT overwrite the first's
    assert table[a]["stage_kind"] == "ladder-probe" and table[a]["wave"] == W and table[a]["unit"] == f"probe:{LORA}"
    assert math.isclose(M.committed_hours(table), 0.4 + 0.3 + 0.18 + 0.55) and math.isclose(M.committed_hours(table, wave=W), 0.4 + 0.18 + 0.55)
    assert M.hours_by_wave(table) == {W: pytest.approx(1.13), "gemma4e4b_main": pytest.approx(0.3)}
    # --record replaces ONE identity's forecast
    assert M.main(["--ledger", str(p), "--record", "--wave", W, "--suffix", SFX, "--unit", f"probe:{LORA}", "--actual-h", "0.25"]) == 0
    table = M.stage_table(M.read_ledger(p))
    assert table[a]["actual"] == 0.25 and table[a]["counted"] == 0.25 and table[b]["counted"] == 0.3
    with pytest.raises(M.MeterError, match="without '/'"):
        M.reservation_id("a/b", SFX, NONE)
    with pytest.raises(M.MeterError, match="stage_kind"):
        M.append_entry(p, {**_res("x", 0.1), "stage_kind": "guess"})
    with pytest.raises(M.MeterError, match="needs --wave, --suffix and --unit"):
        M.main(["--ledger", str(p), "--stage-kind", "wave", "--forecast-h", "0.1"])


def test_a_legacy_stage_only_ledger_still_reads(tmp_path):
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps({"at": "x", "kind": "forecast", "stage": "run2x2", "attempt": 1, "hours": 1.5}) + "\n"
                 + json.dumps({"at": "y", "kind": "actual", "stage": "run2x2", "attempt": 1, "hours": 0.6}) + "\n")
    table = M.stage_table(M.read_ledger(p))
    assert table["run2x2"]["counted"] == 0.6 and table["run2x2"]["wave"] is None and M.hours_by_wave(table) == {"(legacy / no wave)": 0.6}
    assert "| — | — | run2x2 |" in M.report_text(M.read_ledger(p))


# --------------------------------------------------------------------------- #
# the ceiling and the fired-leg decision                                       #
# --------------------------------------------------------------------------- #


def test_the_ceiling_is_25h_with_no_implicit_ladder_allowance(tmp_path):
    p = _ledger_with(tmp_path, _res(NONE, 24.5))
    entries = M.read_ledger(p)
    assert M.ceiling_hours(entries) == {"ceiling_h": 25.0, "source": "spec envelope (25.0 h)", "note": None, "at": None}
    assert M.check(entries, stage=f"{W}/{SFX}/x", forecast_h=0.4)["ok"] is True
    c = M.check(entries, stage=f"{W}/{SFX}/x", forecast_h=0.6)
    assert c["ok"] is False and math.isclose(c["total_h"], 25.1) and "spec envelope" in c["ceiling_source"]
    assert "(i) raise the ceiling" in M.refusal_text(c) and "fired truncation leg" in M.refusal_text(c) and "(iii) stop the lane" in M.refusal_text(c)


def test_a_raise_needs_a_fired_leg_and_stays_within_1_5h_per_leg(tmp_path, capsys):
    ledger = tmp_path / "l.jsonl"
    assert M.main(["--ledger", str(ledger), "--decision", "spec envelope", "--ceiling-h", "25.0"]) == 0        # restating: no leg needed
    with pytest.raises(M.MeterError, match="ONLY for a fired truncation leg, none named"):
        M.main(["--ledger", str(ledger), "--decision", "more budget please", "--ceiling-h", "26.0"])
    with pytest.raises(M.MeterError, match="exceeds 1.5 h"):
        M.main(["--ledger", str(ledger), "--decision", f"ladder fired: {W}/bigcodebench_200", "--ceiling-h", "27.0"])
    with pytest.raises(M.MeterError, match="not <wave>/<leg>"):
        M.main(["--ledger", str(ledger), "--decision", "ladder fired: gemma4e4b_it/mmlu", "--ceiling-h", "26.0"])
    assert M.main(["--ledger", str(ledger), "--decision", f"ladder fired: {W}/bigcodebench_200", "--ceiling-h", "26.5"]) == 0
    out = capsys.readouterr().out
    assert "25.00 → 26.50" in out and f"['{W}/bigcodebench_200']" in out
    entries = M.read_ledger(ledger)
    assert M.ceiling_hours(entries)["ceiling_h"] == 26.5 and M.credited_legs(entries) == [f"{W}/bigcodebench_200"]
    # the same leg is credited ONCE; a second leg (as a flag) is fine; two new legs allow 3.0 h
    with pytest.raises(M.MeterError, match="already credited"):
        M.main(["--ledger", str(ledger), "--decision", f"ladder fired: {W}/bigcodebench_200", "--ceiling-h", "28.0"])
    assert M.main(["--ledger", str(ledger), "--decision", "ladder fired again", "--fired-leg", f"{W}/inspect/bfcl_200", "--fired-leg", "gemma4e4b_main/bigcodebench_200",
                   "--ceiling-h", "29.5"]) == 0
    assert M.ceiling_hours(M.read_ledger(ledger))["ceiling_h"] == 29.5
    # lowering is always allowed
    assert M.main(["--ledger", str(ledger), "--decision", "cut back", "--ceiling-h", "26.0"]) == 0
    assert M.ceiling_hours(M.read_ledger(ledger))["ceiling_h"] == 26.0
    with pytest.raises(M.MeterError, match="go together"):
        M.main(["--ledger", str(ledger), "--ceiling-h", "7.0"])


# --------------------------------------------------------------------------- #
# the rule: the other door of a ceiling raise                                   #
# --------------------------------------------------------------------------- #

#: the shape of the citation (who, when, what) — invented text, the tinycap-shared-standup 2026-09-29 raise's shape
RULING = "the user, 2026-09-29, reading the tinycap-shared-standup review artifact: run landing 2's GPU canaries (16.0 → 21.0 H200-h)"


def _lane_ledger(tmp_path, *, ceiling_h: float = 16.0, committed_h: float = 12.92) -> Path:
    """A lane ledger opened at ``ceiling_h`` with one row's forecast committed — the tinycap-shared-standup shape
    at the moment of the ruling (16.0 in force, 12.92 h committed, ≈ 4.35 h of canaries asking)."""
    ledger = tmp_path / "gpu_ledger.jsonl"
    assert M.main(["--ledger", str(ledger), "--decision", "lane ceiling", "--ceiling-h", f"{ceiling_h:.1f}"]) == 0
    M.append_entry(ledger, _res("row:a", committed_h))
    return ledger


def test_a_raise_without_a_fired_leg_or_a_user_ruling_is_still_refused_and_the_refusal_names_the_door(tmp_path, capsys):
    ledger = _lane_ledger(tmp_path)
    before = ledger.read_bytes()
    with pytest.raises(M.MeterError, match=r"a ceiling raise \(16\.00 → 21\.00 h\) is admitted ONLY for a fired truncation leg, none named") as ei:
        M.main(["--ledger", str(ledger), "--decision", "landing 2's canaries run", "--ceiling-h", "21.0"])
    assert "--user-ruling" in str(ei.value)                         # the guard names its other door
    assert ledger.read_bytes() == before                             # nothing appended; the entries written without the flag are byte-identical
    entries = M.read_ledger(ledger)
    assert M.ceiling_hours(entries)["ceiling_h"] == 16.0 and all("user_ruling" not in e for e in entries)
    # the pure function's default is the old rule, and the launch refusal's option (i) names both doors
    with pytest.raises(M.MeterError, match="ONLY for a fired truncation leg, none named"):
        M.check_decision(entries, ceiling_h=21.0, note="landing 2's canaries run")
    c = M.check(entries, stage=f"{W}/{SFX}/row:b", forecast_h=4.35)
    assert c["ok"] is False and math.isclose(c["total_h"], 17.27)
    assert "(i) raise the ceiling" in M.refusal_text(c) and "fired truncation leg" in M.refusal_text(c) and "--user-ruling" in M.refusal_text(c)


def test_a_user_ruling_admits_the_raise_without_a_leg_records_it_verbatim_and_the_next_forecast_reads_it(tmp_path, capsys):
    ledger = _lane_ledger(tmp_path)
    assert M.check(M.read_ledger(ledger), stage=f"{W}/{SFX}/row:b", forecast_h=4.35)["ok"] is False      # refused at 16.0
    assert M.main(["--ledger", str(ledger), "--decision", "landing 2's canaries run", "--ceiling-h", "21.0", "--user-ruling", RULING, "--json"]) == 0
    out = capsys.readouterr().out
    assert "[meter] decision recorded: ceiling 21.00 H200-h (16.00 → 21.00; raised by the user's ruling: " + RULING + ") — landing 2's canaries run" in out
    entries = M.read_ledger(ledger)
    d = entries[-1]
    assert d["kind"] == "decision" and d["ceiling_h"] == 21.0 and d["note"] == "landing 2's canaries run"
    assert d["user_ruling"] == RULING and d["fired_legs"] is None
    assert json.loads(out.splitlines()[-1])["user_ruling"] == RULING                       # --json prints the entry as written
    assert M.ceiling_hours(entries)["ceiling_h"] == 21.0 and M.credited_legs(entries) == []   # a ruling credits no leg
    c = M.check(entries, stage=f"{W}/{SFX}/row:b", forecast_h=4.35)                      # the following forecast reads 21.0
    assert c["ok"] is True and c["ceiling_h"] == 21.0 and c["ceiling_source"] == "decision" and math.isclose(c["remaining_h"], 21.0 - 12.92 - 4.35)
    assert M.main(["--ledger", str(ledger), "--wave", W, "--suffix", SFX, "--unit", "row:b", "--stage-kind", "wave", "--forecast-h", "4.35"]) == 0
    assert "Σ 17.27 of 21.00 H200-h (decision)" in capsys.readouterr().out
    # forecasts and actuals are untouched by the ruling: the committed sum is the rows', the entry carries no hours
    assert d["hours"] is None and math.isclose(M.committed_hours(M.stage_table(M.read_ledger(ledger))), 17.27)
    # the ruling never lowers the cap on fired-leg raises: a later fired leg still gets its full 1.5 h, and no more
    with pytest.raises(M.MeterError, match="exceeds 1.5 h"):
        M.main(["--ledger", str(ledger), "--decision", f"ladder fired: {W}/bigcodebench_200", "--ceiling-h", "23.0"])
    assert M.main(["--ledger", str(ledger), "--decision", f"ladder fired: {W}/bigcodebench_200", "--ceiling-h", "22.5"]) == 0
    assert "raised by the user's ruling" not in capsys.readouterr().out
    assert "user_ruling" not in M.read_ledger(ledger)[-1]                                  # an entry without the flag carries no key
    # the report renders the ruling next to its decision's note; the greppable ceiling line keeps its prefix
    assert M.main(["--ledger", str(ledger), "--report"]) == 0
    rep = capsys.readouterr().out
    assert f"ceiling 22.50 H200-h (decision: ladder fired: {W}/bigcodebench_200); remaining" in rep
    assert f"ceiling_h 21.0 — landing 2's canaries run (the user's ruling: {RULING})" in rep
    assert "ceiling_h 22.5 — ladder fired" in rep and rep.count("the user's ruling") == 1
    # while a ruling's decision is the one in force, the ceiling line carries it too
    assert M.main(["--ledger", str(ledger), "--decision", "back to the ruling's number", "--ceiling-h", "21.0", "--user-ruling", RULING]) == 0
    assert M.main(["--ledger", str(ledger), "--report"]) == 0
    rep = capsys.readouterr().out
    assert f"ceiling 21.00 H200-h (decision: back to the ruling's number; the user's ruling: {RULING}); remaining" in rep


@pytest.mark.parametrize("ruling", ["", "   ", "\n\t "])
def test_an_empty_or_blank_user_ruling_cites_nothing_and_is_refused(tmp_path, ruling):
    ledger = _lane_ledger(tmp_path)
    before = ledger.read_bytes()
    with pytest.raises(M.MeterError, match="--user-ruling must cite the ruling"):
        M.main(["--ledger", str(ledger), "--decision", "landing 2's canaries run", "--ceiling-h", "21.0", "--user-ruling", ruling])
    with pytest.raises(M.MeterError, match="--user-ruling must cite the ruling"):          # a lowering with a blank citation too
        M.main(["--ledger", str(ledger), "--decision", "cut back", "--ceiling-h", "15.0", "--user-ruling", ruling])
    with pytest.raises(M.MeterError, match="--user-ruling must cite the ruling"):
        M.check_decision(M.read_ledger(ledger), ceiling_h=21.0, note="x", user_ruling=ruling)
    assert ledger.read_bytes() == before and M.ceiling_hours(M.read_ledger(ledger))["ceiling_h"] == 16.0
    # the flag alone is a decision flag: it needs --decision and --ceiling-h like the others
    with pytest.raises(M.MeterError, match="go together"):
        M.main(["--ledger", str(ledger), "--user-ruling", RULING])


def test_lowering_or_restating_with_a_user_ruling_is_admitted_and_records_it(tmp_path, capsys):
    ledger = _lane_ledger(tmp_path, ceiling_h=21.0)
    lower = "the user, 2026-09-30: landing 2 is done, back to 18"
    assert M.main(["--ledger", str(ledger), "--decision", "cut back", "--ceiling-h", "18.0", "--user-ruling", lower]) == 0
    assert f"ceiling 18.00 H200-h (21.00 → 18.00; lowered by the user's ruling: {lower}) — cut back" in capsys.readouterr().out
    entries = M.read_ledger(ledger)
    assert entries[-1]["user_ruling"] == lower and entries[-1]["fired_legs"] is None and entries[-1]["ceiling_h"] == 18.0
    assert M.ceiling_hours(entries)["ceiling_h"] == 18.0
    assert M.main(["--ledger", str(ledger), "--decision", "restated", "--ceiling-h", "18.0", "--user-ruling", lower]) == 0
    assert f"(18.00 → 18.00; restated by the user's ruling: {lower})" in capsys.readouterr().out
    assert M.read_ledger(ledger)[-1]["user_ruling"] == lower
    # a pure lowering without the flag is as free as ever, and carries no key
    assert M.main(["--ledger", str(ledger), "--decision", "cut back again", "--ceiling-h", "17.0"]) == 0
    assert "user_ruling" not in M.read_ledger(ledger)[-1]


def test_a_user_ruling_and_a_fired_leg_are_two_decisions_never_one(tmp_path):
    ledger = _lane_ledger(tmp_path)
    for argv in (["--decision", f"ladder fired: {W}/bigcodebench_200", "--ceiling-h", "21.0", "--user-ruling", RULING],
                 ["--decision", "both", "--fired-leg", f"{W}/bfcl_200", "--ceiling-h", "17.0", "--user-ruling", RULING]):
        with pytest.raises(M.MeterError, match="ONE authority"):
            M.main(["--ledger", str(ledger), *argv])
    assert len(M.read_ledger(ledger)) == 2 and M.credited_legs(M.read_ledger(ledger)) == []
    # two lines: the ruling's raise, then the leg's — each admitted on its own authority
    assert M.main(["--ledger", str(ledger), "--decision", "landing 2's canaries run", "--ceiling-h", "21.0", "--user-ruling", RULING]) == 0
    assert M.main(["--ledger", str(ledger), "--decision", f"ladder fired: {W}/bigcodebench_200", "--ceiling-h", "22.5"]) == 0
    entries = M.read_ledger(ledger)
    assert M.ceiling_hours(entries)["ceiling_h"] == 22.5 and M.credited_legs(entries) == [f"{W}/bigcodebench_200"]
    assert entries[-2]["user_ruling"] == RULING and entries[-2]["fired_legs"] is None and "user_ruling" not in entries[-1]


# --------------------------------------------------------------------------- #
# attempts, supersession, the lock                                             #
# --------------------------------------------------------------------------- #


def test_an_actual_replaces_only_its_attempt_and_a_retry_stays_reserved(tmp_path, capsys):
    ledger = tmp_path / "l.jsonl"
    ident = ["--wave", W, "--suffix", SFX, "--unit", NONE]
    assert M.main(["--ledger", str(ledger), *ident, "--stage-kind", "wave", "--forecast-h", "0.3"]) == 0
    assert M.main(["--ledger", str(ledger), "--record", *ident, "--actual-h", "1.0"]) == 0
    assert M.main(["--ledger", str(ledger), *ident, "--stage-kind", "wave", "--forecast-h", "0.8"]) == 0            # a retry: attempt 2
    out = capsys.readouterr().out
    assert f"{W}/{SFX}/{NONE} attempt 2: forecast 0.80 h admitted (the claim of this launch) — Σ 1.80 of 25.00" in out
    table = M.stage_table(M.read_ledger(ledger))
    assert table[f"{W}/{SFX}/{NONE}"]["attempts"] == {1: {"forecast": 0.3, "actual": 1.0, "superseded_by": None, "released": False},
                                                       2: {"forecast": 0.8, "actual": None, "superseded_by": None, "released": False}}
    assert math.isclose(M.committed_hours(table), 1.8)
    rc = M.main(["--ledger", str(ledger), "--wave", W, "--suffix", SFX, "--unit", "another", "--stage-kind", "wave", "--forecast-h", "23.5"])
    assert rc == 2 and "= 25.30 H200-h > ceiling 25.00 h" in capsys.readouterr().err
    assert M.main(["--ledger", str(ledger), "--record", *ident, "--actual-h", "0.6"]) == 0
    table = M.stage_table(M.read_ledger(ledger))
    assert table[f"{W}/{SFX}/{NONE}"]["attempts"][2]["actual"] == 0.6 and math.isclose(table[f"{W}/{SFX}/{NONE}"]["counted"], 1.6)


def test_an_admission_is_a_claim_refused_again_until_recorded_or_released(tmp_path, capsys):
    """The attempt number is the launch's claim token. A second admission under an identity whose
    launch has not recorded its actual is REFUSED (exit 2, nothing appended) — it never rides the first's
    reservation; recording the actual or RELEASING the claim (the launch never ran) opens the next attempt."""
    ledger = tmp_path / "l.jsonl"
    ident = ["--wave", W, "--suffix", SFX, "--unit", NONE]
    assert M.main(["--ledger", str(ledger), *ident, "--stage-kind", "wave", "--forecast-h", "0.3"]) == 0
    assert "attempt 1: forecast 0.30 h admitted (the claim of this launch)" in capsys.readouterr().out
    rc = M.main(["--ledger", str(ledger), *ident, "--stage-kind", "wave", "--forecast-h", "0.3"])
    err = capsys.readouterr().err
    assert rc == 2 and "attempt 1 is CLAIMED" in err and "--release" in err and "--record" in err
    assert len(M.read_ledger(ledger)) == 1, "the refused admission appended nothing"
    c = M.check(M.read_ledger(ledger), stage=f"{W}/{SFX}/{NONE}", forecast_h=0.3)
    assert c["ok"] is False and c["claimed"] is True and c["attempt"] == 1 and math.isclose(c["committed_h"], 0.3)
    # the launch ran: its actual closes the claim; the next admission opens attempt 2
    assert M.main(["--ledger", str(ledger), "--record", *ident, "--actual-h", "0.2"]) == 0
    assert M.main(["--ledger", str(ledger), *ident, "--stage-kind", "wave", "--forecast-h", "0.4"]) == 0
    assert "attempt 2: forecast 0.40 h admitted" in capsys.readouterr().out
    assert M.main(["--ledger", str(ledger), *ident, "--stage-kind", "wave", "--forecast-h", "0.4"]) == 2
    capsys.readouterr()
    # the launch never ran: the claim is RELEASED (an actual of 0.0 h, noted), its forecast no longer counts
    assert M.main(["--ledger", str(ledger), "--release", "--wave", W, "--suffix", SFX, "--unit", NONE]) == 0
    out = capsys.readouterr().out
    assert "attempt 2: claim RELEASED (the claimed launch did not run); its forecast 0.40 h no longer counts; Σ counted 0.20 h" in out
    last = M.read_ledger(ledger)[-1]
    assert last["kind"] == "actual" and last["hours"] == 0.0 and last["attempt"] == 2 and last["note"].startswith("released:")
    table = M.stage_table(M.read_ledger(ledger))
    row = table[f"{W}/{SFX}/{NONE}"]
    assert row["attempts"][2] == {"forecast": 0.4, "actual": 0.0, "superseded_by": None, "released": True} and math.isclose(row["counted"], 0.2)
    assert "released (never ran)" in M.report_text(M.read_ledger(ledger))
    assert M.main(["--ledger", str(ledger), *ident, "--stage-kind", "wave", "--forecast-h", "0.5"]) == 0
    assert "attempt 3: forecast 0.50 h admitted" in capsys.readouterr().out
    with pytest.raises(M.MeterError, match="nothing to release"):
        M.main(["--ledger", str(ledger), "--release", "why not", "--wave", W, "--suffix", SFX, "--unit", "never-reserved"])
    with pytest.raises(M.MeterError, match="on its own"):
        M.main(["--ledger", str(ledger), "--release", *ident, "--record", "--actual-h", "0.1"])
    # a released entry with hours is not a release
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"kind": "forecast", "stage": "x", "attempt": 1, "hours": 1.0}) + "\n"
                 + json.dumps({"kind": "actual", "stage": "x", "attempt": 1, "hours": 0.5, "note": "released: but spent"}) + "\n")
    with pytest.raises(M.MeterError, match="released claim records 0.0 h"):
        M.stage_table(M.read_ledger(p))


def test_two_simultaneous_admissions_of_the_same_identity_admit_exactly_one(tmp_path):
    """Two processes launch the SAME identity at once, well under the ceiling: the lock serialises them and the
    second reads the first's open claim — exactly one is admitted, the other refused as CLAIMED (the ceiling
    never doubled up: 24.3 + 0.6, not 25.5)."""
    ledger = tmp_path / "gpu_ledger.jsonl"
    M.append_entry(ledger, _res("spent", 24.3, entry_kind="actual"))
    lock = ledger.with_name(ledger.name + ".lock")
    argv = [sys.executable, str(SCRIPT), "--ledger", str(ledger), "--wave", W, "--suffix", SFX, "--unit", NONE, "--stage-kind", "wave",
            "--forecast-h", "0.6"]
    with open(lock, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        procs = [subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
                 for _ in range(2)]
        time.sleep(1.5)
        assert len(M.read_ledger(ledger)) == 1
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    results = [(p.wait(timeout=60), p.communicate()) for p in procs]
    assert sorted(rc for rc, _ in results) == [0, 2], results
    refused = next(out_err for rc, out_err in results if rc == 2)
    assert "is CLAIMED" in refused[1] and "never rides an earlier launch's reservation" in refused[1]
    entries = M.read_ledger(ledger)
    assert len(entries) == 2 and math.isclose(M.committed_hours(M.stage_table(entries)), 24.9)


def test_a_partially_specified_identity_refuses(tmp_path):
    p = tmp_path / "l.jsonl"
    with pytest.raises(M.MeterError, match=r"partially specified reservation identity .*\['wave'\] missing"):
        M.append_entry(p, {"kind": "forecast", "suffix": SFX, "unit": NONE, "stage_kind": "wave", "hours": 0.1})
    with pytest.raises(M.MeterError, match="partially specified"):
        M.append_entry(p, {"kind": "actual", "wave": W, "unit": NONE, "hours": 0.1})
    assert not p.exists()
    M.append_entry(p, {"kind": "forecast", "stage": "legacy-only", "hours": 0.1})           # a legacy stage name alone still writes
    assert M.stage_table(M.read_ledger(p))["legacy-only"]["wave"] is None


def test_supersession_is_persisted_and_honoured_on_reload(tmp_path):
    ledger = tmp_path / "l.jsonl"
    M.append_entry(ledger, _res(NONE, 1.5, entry_kind="actual"))
    M.append_entry(ledger, _res(f"probe:{LORA}", 2.0, kind="ladder-probe"))
    probe = f"{W}/{SFX}/probe:{LORA}"
    c = M.admit(ledger, wave=W, suffix=SFX, unit="score", stage_kind="ladder-score", forecast_h=1.5, supersedes=(probe,))
    assert c["ok"] and math.isclose(c["total_h"], 3.0) and c["excluded"] == [(probe, 1)]
    entries = M.read_ledger(ledger)
    assert entries[-1]["supersedes"] == [{"stage": probe, "attempt": 1}] and entries[-1]["stage_kind"] == "ladder-score"
    table = M.stage_table(entries)
    assert table[probe]["attempts"][1]["superseded_by"] == {"stage": f"{W}/{SFX}/score", "attempt": 1} and math.isclose(M.committed_hours(table), 3.0)
    assert "superseded (by" in M.report_text(entries)
    M.append_entry(ledger, _res(f"probe:{LORA}", 0.3, kind="ladder-probe", entry_kind="actual"))                   # a measured probe is spent hours
    table = M.stage_table(M.read_ledger(ledger))
    assert table[probe]["attempts"][1]["superseded_by"] is None and math.isclose(M.committed_hours(table), 3.3)
    assert M.next_attempt(table, probe) == 2


def test_two_simultaneous_near_ceiling_admissions_admit_exactly_one(tmp_path):
    """Two processes race for the last 0.8 h under a 25 h ceiling with 24.3 h committed. The test HOLDS the ledger
    lock while both start (neither may append while held — proof they take the lock), then releases: the second
    reads the first's append and is refused."""
    ledger = tmp_path / "gpu_ledger.jsonl"
    M.append_entry(ledger, _res(NONE, 24.3, entry_kind="actual"))
    lock = ledger.with_name(ledger.name + ".lock")
    argv = lambda unit: [sys.executable, str(SCRIPT), "--ledger", str(ledger), "--wave", W, "--suffix", SFX, "--unit", unit, "--stage-kind", "wave",
                         "--forecast-h", "0.6"]
    with open(lock, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        procs = [subprocess.Popen(argv(u), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
                 for u in ("cellA", "cellB")]
        time.sleep(1.5)
        assert len(M.read_ledger(ledger)) == 1, "an admission appended while the lock was held — the meter did not take the lock"
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    results = [(p.wait(timeout=60), p.communicate()) for p in procs]
    codes = sorted(rc for rc, _ in results)
    assert codes == [0, 2], results
    entries = M.read_ledger(ledger)
    assert len(entries) == 2 and math.isclose(M.committed_hours(M.stage_table(entries)), 24.9)
    refused = next(out_err for rc, out_err in results if rc == 2)
    assert "REFUSED launch" in refused[1] and "> ceiling 25.00 h" in refused[1]


# --------------------------------------------------------------------------- #
# the model-aware ladder forecast                                              #
# --------------------------------------------------------------------------- #


def _expected_scoring(n_by_task, seqs, *, cells=CELLS, cap_to=16_384):
    total = 0.0
    for cell in cells:
        tok = M.TOK_S_UNSTEERED if cell == NONE else M.TOK_S_STEERED
        total += M.STANDUP_S + sum(math.ceil(n / seqs) * cap_to / tok for n in n_by_task.values())
    return total


def test_ladder_forecast_counts_a_probe_per_fitted_cell_and_reprobes(tmp_path):
    counts = M.load_selection_counts(_selection(tmp_path / "sel.json", {BCB: 20}))
    fc = M.ladder_forecast(counts, seqs=12, fitted_cells=[LORA, PSR], probe_rungs_by_cell={LORA: 4, PSR: 4})
    assert fc["probe_seconds"] == 2 * 4 * M.STANDUP_S and fc["reprobe_seconds"] == 0 and set(fc["probes"]) == {LORA, PSR}
    assert math.isclose(fc["hours"], (fc["probe_seconds"] + _expected_scoring({BCB: 20}, 12)) / 3600.0)
    fc2 = M.ladder_forecast(counts, seqs=16, fitted_cells=[LORA, PSR], probe_seconds_by_cell={LORA: 0.0, PSR: 660.0}, probe_rungs_by_cell={LORA: 1, PSR: 2},
                            reprobe_cells=[LORA])
    assert fc2["probe_seconds"] == 660.0 and fc2["reprobe_seconds"] == M.STANDUP_S and fc2["probes"][LORA]["measured"] is True
    with pytest.raises(M.MeterError, match="object-free and needs no probe"):
        M.ladder_forecast(counts, seqs=16, fitted_cells=[NONE])
    with pytest.raises(M.MeterError, match="not this ladder's cells"):
        M.ladder_forecast(counts, seqs=16, fitted_cells=["ghost"])
    with pytest.raises(M.MeterError, match="fired no leg"):
        M.ladder_forecast({"tasks": {}, "cells": CELLS}, seqs=16, fitted_cells=[LORA])


def test_ladder_spec_is_required_and_comes_from_rung_probe(monkeypatch, tmp_path):
    sel = _selection(tmp_path / "sel.json", {BCB: 20})
    if hasattr(RP, "ladder_spec"):
        monkeypatch.delattr(RP, "ladder_spec")
    with pytest.raises(M.MeterError, match="rung_probe.ladder_spec is not on this tree"):
        M.main(["--ledger", str(tmp_path / "l.jsonl"), "--wave", W, "--suffix", SFX, "--unit", f"probe:{LORA}", "--stage-kind", "ladder-probe",
                "--model", "gemma4e4b", "--selection", str(sel)])
    monkeypatch.setattr(RP, "ladder_spec", _stub_spec, raising=False)
    with pytest.raises(M.MeterError, match="refused: "):
        M.ladder_spec("qwen35_4b")                                                                  # a provisional model refuses through B's check
    with pytest.raises(M.MeterError, match="needs --model"):
        M.main(["--ledger", str(tmp_path / "l.jsonl"), "--wave", W, "--suffix", SFX, "--unit", f"probe:{LORA}", "--stage-kind", "ladder-probe", "--selection", str(sel)])


def test_probe_reservation_uses_the_models_worst_rung_and_score_the_common_certified_s(spec, tmp_path, capsys):
    ledger = tmp_path / "l.jsonl"
    sel = _selection(tmp_path / "sel.json", {BCB: 20})
    selection = json.loads(sel.read_text())
    base = ["--ledger", str(ledger), "--wave", W, "--suffix", SFX, "--model", "gemma4e4b", "--selection", str(sel)]
    for cell in (LORA, PSR):
        assert M.main([*base, "--unit", f"probe:{cell}", "--stage-kind", "ladder-probe"]) == 0
    out = capsys.readouterr().out
    assert "S=12" in out and "worst case S = 12 of [18, 16, 14, 12]" in out
    e = [x for x in M.read_ledger(ledger) if x["unit"] == f"probe:{LORA}"][-1]
    fc = e["inputs"]["forecast"]
    assert fc["probes"][LORA]["rungs"] == 4 and fc["ladder"] == {"model_key": "gemma4e4b", "seqs": [18, 16, 14, 12], "window": 32_768}
    # the probe reservation: a full walk + ONE re-probe standup, and NO scoring (the scoring is its own claim)
    assert fc["reprobe_cells"] == [LORA] and fc["scoring_cells"] == [] and fc["cells"] == {} and fc["scoring_hours"] == 0.0
    assert math.isclose(fc["hours"], (4 * M.STANDUP_S + M.STANDUP_S) / 3600.0)
    # nemotron's ladder is its own (24, 22, 20, 18)
    assert M.main(["--ledger", str(ledger), "--wave", "nemotron3nano4b_main", "--suffix", "bcbw-nemo-abc1234", "--model", "nemotron3nano4b", "--selection", str(sel),
                   "--unit", f"probe:{LORA}", "--stage-kind", "ladder-probe"]) == 0
    out = capsys.readouterr().out
    assert "S=18" in out and "worst case S = 18 of [24, 22, 20, 18]" in out
    # the scoring admission: LoRA accepted 18 (one rung), PSR 16 (two rungs) → common 16; SCORING ONLY — the probe reservations
    # are independent records this admission neither carries nor supersedes
    c18 = _certificate(tmp_path / "c_lora.json", seqs=18, walked_n=1, selection=selection, probed_cell=LORA)
    c16 = _certificate(tmp_path / "c_psr.json", seqs=16, walked_n=2, selection=selection, probed_cell=PSR, standup_s=660.0)
    with pytest.raises(M.MeterError, match="needs --certificate"):
        M.main([*base, "--unit", "score", "--stage-kind", "ladder-score"])
    assert M.main([*base, "--unit", "score", "--stage-kind", "ladder-score", "--certificate", str(c18), "--certificate", str(c16)]) == 0
    out = capsys.readouterr().out
    assert "S=16" in out and "scoring only" in out
    e = [x for x in M.read_ledger(ledger) if x["unit"] == "score"][-1]
    fc = e["inputs"]["forecast"]
    assert fc["accepted_by_cell"] == {LORA: 18, PSR: 16} and fc["reprobe_cells"] == [] and fc["probe_seconds"] == 0.0 and fc["probes"] == {}
    assert fc["probe_reservations"] == {LORA: "open", PSR: "open"} and e["supersedes"] is None
    assert math.isclose(fc["hours"], _expected_scoring({BCB: 20}, 16) / 3600.0)
    table = M.stage_table(M.read_ledger(ledger))
    assert table[f"{W}/{SFX}/probe:{LORA}"]["attempts"][1]["superseded_by"] is None
    assert math.isclose(M.committed_hours(table, wave=W), 2 * (5 * M.STANDUP_S) / 3600.0 + fc["hours"])
    # the scoring claim is OPEN: a second scoring admission under it is refused (C-P1) until it is recorded or released
    assert M.main([*base, "--unit", "score", "--stage-kind", "ladder-score", "--certificate", str(c18), "--certificate", str(c16)]) == 2
    assert "is CLAIMED" in capsys.readouterr().err
    # the scoring launch never ran: RELEASED — the probe reservations stand untouched (they were never carried by it)
    assert M.main(["--ledger", str(ledger), "--release", "the scoring launch failed to start", "--wave", W, "--suffix", SFX, "--unit", "score"]) == 0
    table = M.stage_table(M.read_ledger(ledger))
    assert table[f"{W}/{SFX}/score"]["attempts"][1]["released"] and math.isclose(M.committed_hours(table, wave=W), 2 * (5 * M.STANDUP_S) / 3600.0)
    # a probe recorded as an ACTUAL is spent hours, counted through its actual; the successor scoring claim (attempt 2) records that
    assert M.main(["--ledger", str(ledger), "--record", "--wave", W, "--suffix", SFX, "--unit", f"probe:{PSR}", "--actual-h", "0.2"]) == 0
    assert M.main([*base, "--unit", "score", "--stage-kind", "ladder-score", "--certificate", str(c18), "--certificate", str(c16)]) == 0
    e = [x for x in M.read_ledger(ledger) if x["unit"] == "score"][-1]
    fc = e["inputs"]["forecast"]
    assert e["attempt"] == 2 and fc["probe_reservations"] == {LORA: "open", PSR: "measured"} and e["supersedes"] is None
    with pytest.raises(M.MeterError, match="--reprobe-cell belongs to the ladder-probe reservation"):
        M.main([*base, "--unit", "score", "--stage-kind", "ladder-score", "--certificate", str(c18), "--certificate", str(c16), "--reprobe-cell", LORA])
    other = _certificate(tmp_path / "other.json", seqs=16, walked_n=2, selection=_selection_doc({BCB: 25}), probed_cell=PSR)
    with pytest.raises(M.MeterError, match="selection"):
        M.main([*base, "--unit", "score", "--stage-kind", "ladder-score", "--certificate", str(c18), "--certificate", str(other)])
    with pytest.raises(M.MeterError, match="do not also pass --forecast-h"):
        M.main([*base, "--unit", "score", "--stage-kind", "ladder-score", "--certificate", str(c18), "--forecast-h", "1.0"])
    with pytest.raises(M.MeterError, match="belongs to --stage-kind ladder-score"):
        M.main(["--ledger", str(ledger), "--wave", W, "--suffix", SFX, "--unit", "x", "--stage-kind", "wave", "--forecast-h", "0.1", "--certificate", str(c18)])


def test_per_cell_scoring_units_are_separate_claims_and_probes_count_once(spec, tmp_path):
    """The wrapper scores one cell per launch: each launch claims ``score:<cell>`` (its own identity, its own
    forecast — that cell's scoring only); the probe standups and re-probes are carried by the FIRST scoring
    claim, which supersedes the open probe reservations, and counted ZERO by the later ones."""
    ledger = tmp_path / "l.jsonl"
    sel = _selection(tmp_path / "sel.json", {BCB: 20})
    selection = json.loads(sel.read_text())
    base = ["--ledger", str(ledger), "--wave", W, "--suffix", SFX, "--model", "gemma4e4b", "--selection", str(sel)]
    for cell in (LORA, PSR):
        assert M.main([*base, "--unit", f"probe:{cell}", "--stage-kind", "ladder-probe"]) == 0
    c18 = _certificate(tmp_path / "c_lora.json", seqs=18, walked_n=1, selection=selection, probed_cell=LORA)
    c16 = _certificate(tmp_path / "c_psr.json", seqs=16, walked_n=2, selection=selection, probed_cell=PSR, standup_s=660.0)
    certs = ["--certificate", str(c18), "--certificate", str(c16)]
    assert M.main([*base, "--unit", f"score:{LORA}", "--stage-kind", "ladder-score", *certs]) == 0
    e1 = [x for x in M.read_ledger(ledger) if x["unit"] == f"score:{LORA}"][-1]
    fc = e1["inputs"]["forecast"]
    assert fc["scoring_cells"] == [LORA] and list(fc["cells"]) == [LORA] and fc["probe_seconds"] == 0.0 and fc["reprobe_cells"] == []
    assert e1["supersedes"] is None and fc["probe_reservations"] == {LORA: "open", PSR: "open"}
    assert math.isclose(fc["hours"], _expected_scoring({BCB: 20}, 16, cells=[LORA]) / 3600.0)
    # the second cell's scoring is ANOTHER claim, admitted, scoring its own cell only; the probe reservations stay their own records
    assert M.main([*base, "--unit", f"score:{PSR}", "--stage-kind", "ladder-score", *certs]) == 0
    e2 = [x for x in M.read_ledger(ledger) if x["unit"] == f"score:{PSR}"][-1]
    fc2 = e2["inputs"]["forecast"]
    assert fc2["scoring_cells"] == [PSR] and fc2["probe_seconds"] == 0.0 and fc2["reprobe_cells"] == [] and e2["supersedes"] is None
    assert math.isclose(fc2["hours"], _expected_scoring({BCB: 20}, 16, cells=[PSR]) / 3600.0)
    table = M.stage_table(M.read_ledger(ledger))
    probes_h = 2 * (5 * M.STANDUP_S) / 3600.0                              # two probe reservations: a full walk + one re-probe standup each
    assert math.isclose(M.committed_hours(table), probes_h + fc["hours"] + fc2["hours"])
    assert all(table[f"{W}/{SFX}/probe:{c}"]["attempts"][1]["superseded_by"] is None for c in (LORA, PSR))
    with pytest.raises(M.MeterError, match="not this ladder's cells"):
        M.main([*base, "--unit", "score:ghost", "--stage-kind", "ladder-score", *certs])
    with pytest.raises(M.MeterError, match="unit is 'score'"):
        M.main([*base, "--unit", "scoring", "--stage-kind", "ladder-score", *certs])


def test_ladder_inputs_are_read_under_the_admission_lock(spec, tmp_path):
    """The supersession decisions of a scoring admission are made on the ledger read UNDER the lock: the test
    holds the lock, starts the admission, then records the PSR probe's actual before releasing — the admitted
    forecast must count that probe as measured and supersede only the LoRA probe."""
    ledger = tmp_path / "l.jsonl"
    sel = _selection(tmp_path / "sel.json", {BCB: 20})
    selection = json.loads(sel.read_text())
    base = ["--ledger", str(ledger), "--wave", W, "--suffix", SFX, "--model", "gemma4e4b", "--selection", str(sel)]
    for cell in (LORA, PSR):
        assert M.main([*base, "--unit", f"probe:{cell}", "--stage-kind", "ladder-probe"]) == 0
    c18 = _certificate(tmp_path / "c_lora.json", seqs=18, walked_n=1, selection=selection, probed_cell=LORA)
    c16 = _certificate(tmp_path / "c_psr.json", seqs=16, walked_n=2, selection=selection, probed_cell=PSR, standup_s=660.0)
    argv = [sys.executable, str(SCRIPT), *base, "--unit", "score", "--stage-kind", "ladder-score", "--certificate", str(c18), "--certificate", str(c16)]
    lock = ledger.with_name(ledger.name + ".lock")
    with open(lock, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
        time.sleep(3.0)                                            # the process is up and blocked on the lock
        assert proc.poll() is None and len([e for e in M.read_ledger(ledger) if e["unit"] == "score"]) == 0
        M.append_entry(ledger, _res(f"probe:{PSR}", 0.2, kind="ladder-probe", entry_kind="actual"))
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    rc, (out, err) = proc.wait(timeout=90), proc.communicate()
    assert rc == 0, (out, err)
    e = [x for x in M.read_ledger(ledger) if x["unit"] == "score"][-1]
    fc = e["inputs"]["forecast"]
    assert fc["probe_reservations"] == {LORA: "open", PSR: "measured"} and e["supersedes"] is None       # the state read under the lock


def _writer_certificate(path: Path, *, selection: dict, probed_cell: str, walked, standup_s: float, reprobed_from=None) -> dict:
    """A certificate through the PRODUCER's writer (`rung_probe.build_certificate` over a `Provenance`, written by
    `write_certificate`): what `ladder_admit`'s probe / re-probe functions file on the volume."""
    spec_ = RP.ladder_spec("gemma4e4b")
    budget = RP.ladder_budget(spec_)
    prov = RP.Provenance(tree_sha=F.TREE, model_slug="gemma4e4b-sdfbase", hf=F.HF, rev=F.REV, cfg_path="configs/two_si_gemma4e4b_sdfbase.yaml",
                         cfg_sha256="4" * 64, object_digests={LORA: "b" * 64, PSR: "c" * 64},
                         selection={t: dict(b) for t, b in selection["selection"].items()},
                         cap_overrides_by_task={t: int(b["cap_to"]) for t, b in selection["selection"].items()},
                         batched_token_budget=int(budget.batched_token_budget), budget_source=str(budget.source),
                         suffix=SFX + "-cap16k", probed_cell=probed_cell, ladder=spec_)
    S = walked[-1]["S"]
    cert = RP.build_certificate(accepted=RP.accepted_block(S, spec=spec_), walked=walked, provenance=prov, probe_run_id="abc1234-00000001",
                                standup_s=standup_s, reprobed_from=reprobed_from)
    RP.write_certificate(path, cert)
    return cert


def _refused(S: int) -> dict:
    return RP.walked_entry(S, RP.WALK_REFUSED, f"KV cache holds {_KV_AFTER[S]} tokens < the {S * RP.LADDER_WINDOW}-token budget (= {S} x {RP.LADDER_WINDOW})")


@pytest.mark.parametrize("scoring_actual_first", [True, False])
def test_probe_spend_is_an_independent_record_that_scoring_never_carries_or_replaces(spec, tmp_path, scoring_actual_first):
    """The repro: 330 s probe + 330 s re-probe + 660 s second walk carried 990 s, then 0 after the scoring
    actual. With WRITER-produced final certificates — LoRA's is a common-rung RE-PROBE whose `standup_s`
    covers only the re-probe (its first walk lives in `reprobed_from`), PSR's a two-rung walk — and every final
    certificate at the common S (the producer normalises the accepted S away): the scoring admission forecasts scoring
    ONLY; the probe reservations keep their own forecasts until their actuals land; recording the scoring actual and
    the probe actuals in EITHER order leaves every second of probe spend on the ledger."""
    ledger = tmp_path / "l.jsonl"
    sel = _selection(tmp_path / "sel.json", {BCB: 20})
    selection = json.loads(sel.read_text())
    base = ["--ledger", str(ledger), "--wave", W, "--suffix", SFX, "--model", "gemma4e4b", "--selection", str(sel)]
    for cell in (LORA, PSR):
        assert M.main([*base, "--unit", f"probe:{cell}", "--stage-kind", "ladder-probe"]) == 0
    first = _writer_certificate(tmp_path / "lora_first.json", selection=selection, probed_cell=LORA,
                                walked=[RP.walked_entry(18, RP.WALK_ACCEPTED)], standup_s=330.0)                  # accepted at identity 18
    lora = _writer_certificate(tmp_path / RP.certificate_filename(LORA), selection=selection, probed_cell=LORA,
                               walked=[RP.walked_entry(16, RP.WALK_ACCEPTED)], standup_s=330.0,                    # the RE-PROBE at the common S
                               reprobed_from={"certificate_sha256": RP.certificate_sha256(first), "accepted_S": 18})
    psr = _writer_certificate(tmp_path / RP.certificate_filename(PSR), selection=selection, probed_cell=PSR,
                              walked=[_refused(18), RP.walked_entry(16, RP.WALK_ACCEPTED)], standup_s=660.0)        # a two-rung walk
    assert lora["accepted"] == psr["accepted"] and lora["reprobed_from"]["accepted_S"] == 18 and psr["reprobed_from"] is None
    certs = ["--certificate", str(tmp_path / RP.certificate_filename(LORA)), "--certificate", str(tmp_path / RP.certificate_filename(PSR))]
    assert M.main([*base, "--unit", "score", "--stage-kind", "ladder-score", *certs]) == 0
    e = [x for x in M.read_ledger(ledger) if x["unit"] == "score"][-1]
    fc = e["inputs"]["forecast"]
    assert fc["probe_seconds"] == 0.0 and fc["reprobe_cells"] == [] and e["supersedes"] is None
    assert fc["reprobed_from_S_by_cell"] == {LORA: 18, PSR: None} and fc["accepted_by_cell"] == {LORA: 16, PSR: 16}
    probe_forecast_h = 5 * M.STANDUP_S / 3600.0
    scoring_h = fc["hours"]
    table = M.stage_table(M.read_ledger(ledger))
    assert math.isclose(M.committed_hours(table), 2 * probe_forecast_h + scoring_h)
    # the incurred probe spend: LoRA 330 (first walk) + 330 (re-probe) = 660 s; PSR 660 s (two standups)
    probe_actuals = [("--record", "--wave", W, "--suffix", SFX, "--unit", f"probe:{LORA}", "--actual-h", repr(660 / 3600)),
                     ("--record", "--wave", W, "--suffix", SFX, "--unit", f"probe:{PSR}", "--actual-h", repr(660 / 3600))]
    scoring_actual = ("--record", "--wave", W, "--suffix", SFX, "--unit", "score", "--actual-h", "0.5")
    order = ([scoring_actual, *probe_actuals] if scoring_actual_first else [*probe_actuals, scoring_actual])
    for argv in order:
        assert M.main(["--ledger", str(ledger), *argv]) == 0
    table = M.stage_table(M.read_ledger(ledger))
    assert math.isclose(M.committed_hours(table), 1320.0 / 3600.0 + 0.5), M.committed_hours(table)      # 990 s + 330 s of probes never vanish
    assert all(table[f"{W}/{SFX}/probe:{c}"]["attempts"][1]["superseded_by"] is None for c in (LORA, PSR))
    assert math.isclose(table[f"{W}/{SFX}/probe:{LORA}"]["counted"], 660 / 3600) and math.isclose(table[f"{W}/{SFX}/score"]["counted"], 0.5)
    # a second scoring admission after the actual (a re-run) is a NEW attempt and still carries no probe spend
    assert M.main([*base, "--unit", "score", "--stage-kind", "ladder-score", *certs]) == 0
    e2 = [x for x in M.read_ledger(ledger) if x["unit"] == "score"][-1]
    assert e2["attempt"] == 2 and e2["inputs"]["forecast"]["probe_reservations"] == {LORA: "measured", PSR: "measured"}
    assert math.isclose(M.committed_hours(M.stage_table(M.read_ledger(ledger))), 1320.0 / 3600.0 + 0.5 + scoring_h)


def test_selection_inputs_are_recomputed_never_trusted(spec, tmp_path):
    good = _selection_doc({BCB: 20})
    ledger = tmp_path / "l.jsonl"
    for mutate, why in ((lambda d: d["selection"][BCB].update(n=19, n_selected=19), "n / n_selected / len"),
                        (lambda d: d["selection"][BCB]["ids"].__setitem__(0, "BigCodeBench/999999"), "ids"),
                        (lambda d: d.update(selection_sha256="f" * 64), "selection_sha256"),
                        (lambda d: d["selection"][BCB].update(cap_to=8192), "cap_to")):
        d = json.loads(json.dumps(good)); mutate(d)
        bad = tmp_path / "bad.json"; bad.write_text(json.dumps(d))
        with pytest.raises(M.MeterError, match=why):
            M.main(["--ledger", str(ledger), "--wave", W, "--suffix", SFX, "--unit", f"probe:{LORA}", "--stage-kind", "ladder-probe", "--model", "gemma4e4b",
                    "--selection", str(bad)])
    assert not ledger.exists()


def test_report_groups_by_wave_and_the_cli_help_paths(tmp_path, capsys):
    ledger = _ledger_with(tmp_path, {"kind": "decision", "ceiling_h": 25.0, "note": "spec envelope"}, _res("smoke", 0.5, kind="smoke"),
                          _res(NONE, 0.18), _res(NONE, 0.2, wave="gemma4e4b_main", suffix="bcbw-gemma4e4b_main-abc1234"))
    assert M.main(["--ledger", str(ledger), "--report", "--json"]) == 0
    out = capsys.readouterr().out
    assert f"| {W} | {SFX} | smoke | smoke | 1 | 0.50 | — | 0.50 |" in out and "by wave: gemma4e4b_it 0.68 h; gemma4e4b_main 0.20 h" in out
    assert "ceiling 25.00 H200-h (decision: spec envelope)" in out and '"by_wave"' in out
    assert M.main(["--ledger", str(ledger)]) == 2
    p = tmp_path / "bad.jsonl"
    p.write_text('{"kind": "guess", "stage": "x", "hours": 1}\n')
    with pytest.raises(M.MeterError, match="kind"):
        M.read_ledger(p)
    p.write_text('{"kind": "forecast", "stage": "x", "hours": -1}\n')
    with pytest.raises(M.MeterError, match="non-negative"):
        M.stage_table(M.read_ledger(p))


def _native_certificate(path: Path, *, seqs: int, walked: list, selection: dict, probed_cell: str) -> Path:
    """A VALID native gemma certificate: the ENGINE-SERVED ladder block (15, 14, 12) and an accepted block read against the
    served identity 15 (`rung_probe.accepted_block(..., spec=native)`)."""
    native = RP.native_ladder_spec(RP.ladder_spec("gemma4e4b"))
    p = _certificate(path, seqs=18, walked_n=1, selection=selection, probed_cell=probed_cell)
    cert = json.loads(p.read_text())
    cert.update(ladder=native.block(), walked=walked, accepted=RP.accepted_block(seqs, spec=native))
    p.write_text(json.dumps(cert))
    return p


def test_the_meter_sizes_a_native_ladder_from_the_engine_served_rungs(spec, tmp_path, capsys):
    """The meter sized every ladder from the LENS rung list, so a native wave's scoring admission refused its own certificates ("the
    common S 15 is not on gemma4e4b's ladder [18, 16, 14, 12]"). Now `--serving-engine lora_native` sizes the pre-probe forecast from
    the ENGINE-SERVED rungs (15, 14, 12: three standups per walk, worst case 12) and records the engine and the mapping; a scoring
    admission READS the engine off the certificates' ladder blocks (their authority), sizes from that engine's ladder, records it, and
    refuses a flag that disagrees or certificates walked on two engines. The lens path is byte-identical (the lens tests above)."""
    ledger = tmp_path / "l.jsonl"
    sel = _selection(tmp_path / "sel.json", {BCB: 20})
    selection = json.loads(sel.read_text())
    base = ["--ledger", str(ledger), "--wave", W, "--suffix", SFX, "--model", "gemma4e4b", "--selection", str(sel)]
    assert M.main([*base, "--unit", f"probe:{LORA}", "--stage-kind", "ladder-probe", "--serving-engine", "lora_native"]) == 0
    out = capsys.readouterr().out
    assert "S=12" in out and "worst case S = 12 of [15, 14, 12]" in out
    e = [x for x in M.read_ledger(ledger) if x["unit"] == f"probe:{LORA}"][-1]
    fc = e["inputs"]["forecast"]
    assert fc["probes"][LORA]["rungs"] == 3 and math.isclose(fc["hours"], (3 * M.STANDUP_S + M.STANDUP_S) / 3600.0)
    assert fc["ladder"] == {"model_key": "gemma4e4b", "seqs": [15, 14, 12], "window": 32_768, "serving_engine": "lora_native",
                            "requested_seqs": [18, 16, 14, 12],
                            "served_rungs": [{"served": 15, "requested": [18, 16]}, {"served": 14, "requested": [14]}, {"served": 12, "requested": [12]}]}
    # the scoring admission: LoRA accepted the served identity 15, PSR 14 after a refusal at 15 -> common 14, the engine read
    # off the certificates (no flag needed), the served ladder recorded
    c15 = _native_certificate(tmp_path / "n_lora.json", seqs=15, selection=selection, probed_cell=LORA,
                              walked=[{"S": 15, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None}])
    c14 = _native_certificate(tmp_path / "n_psr.json", seqs=14, selection=selection, probed_cell=PSR,
                              walked=[{"S": 15, "outcome": "refused", "kv_tokens": 480_000,
                                       "refusal_tail": "KV cache holds 480000 tokens < the 491520-token budget (= 15 x 32768)"},
                                      {"S": 14, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None}])
    assert M.main([*base, "--unit", "score", "--stage-kind", "ladder-score", "--certificate", str(c15), "--certificate", str(c14)]) == 0
    out = capsys.readouterr().out
    assert "S=14" in out and "scoring only" in out
    e = [x for x in M.read_ledger(ledger) if x["unit"] == "score"][-1]
    fc = e["inputs"]["forecast"]
    assert fc["accepted_by_cell"] == {LORA: 15, PSR: 14}
    assert fc["ladder"] == {"model_key": "gemma4e4b", "seqs": [15, 14, 12], "serving_engine": "lora_native", "requested_seqs": [18, 16, 14, 12],
                            "served_rungs": [{"served": 15, "requested": [18, 16]}, {"served": 14, "requested": [14]}, {"served": 12, "requested": [12]}]}
    assert math.isclose(fc["hours"], _expected_scoring({BCB: 20}, 14) / 3600.0)
    assert M.main(["--ledger", str(ledger), "--release", "never ran", "--wave", W, "--suffix", SFX, "--unit", "score"]) == 0
    # the flag is held to the certificates; two engines are not one ladder
    with pytest.raises(M.MeterError, match="--serving-engine lens but the certificates were walked on the lora_native engine"):
        M.main([*base, "--unit", "score", "--stage-kind", "ladder-score", "--serving-engine", "lens", "--certificate", str(c15), "--certificate", str(c14)])
    lens18 = _certificate(tmp_path / "l_lora.json", seqs=18, walked_n=1, selection=selection, probed_cell=LORA)
    with pytest.raises(M.MeterError, match="several serving engines"):
        M.main([*base, "--unit", "score", "--stage-kind", "ladder-score", "--certificate", str(lens18), "--certificate", str(c14)])
    # the engine-aware spec: the same function the readers use
    assert tuple(M.ladder_spec("gemma4e4b", "lora_native").seqs) == (15, 14, 12) and tuple(M.ladder_spec("gemma4e4b").seqs) == (18, 16, 14, 12)
    assert tuple(M.ladder_spec("gemma4e4b", "lens").seqs) == (18, 16, 14, 12)
    with pytest.raises(M.MeterError, match="is not one of"):
        M.ladder_spec("gemma4e4b", "hf")
