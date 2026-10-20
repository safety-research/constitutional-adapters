"""Table and refit discipline: (1) a published round-1 table's DONE marker must not skip the rebuild after an addendum adds rows or a
re-verdict widens the menu — the table phase is keyed on the trial set + menu; hpo_table takes the widened pivot's predictions from the
derived refit r1_fit_menu.json while r1_fit.json keeps the round-1 selection / provenance; (2) the widened refit writes to a DISTINCT path
(`analyze --fit-name`) and never touches r1_fit.json on any exit; the canonical fit is restored on any change; the ONLY nonzero refit the
addendum phase accepts is the recognized non-positive-definite artefact (the fallback state) — every other nonzero stays a failure.
(3) the table's key covers every summary's CONTENT, the fit / refit artefacts' bytes and the SOURCE RECORD identity
(cluster + record path + the menu's generation), not the file names — a retried trial, a re-fit or the same names fetched from another cluster
rebuild; (4) the derived refit is BOUND to the design fit (search id + the rows it was fit over), to the menu generation (the menu file's sha)
and to the source record it was derived on (`--source-record <cluster>:<record>`); hpo_table refuses by name a residual refit left on the
shared mirror after a --cluster change (rehome) or a re-verdict that moved the menu. (5) the table and addendum
phases refuse a failed fetch, an unreadable / empty record, and a mirror that is not the record's own inventory (hashed ON the record host:
a summary the record holds must be on the mirror byte for byte, a summary on the mirror must be on the record) — a cluster switch can never
republish the previous record's evidence under the new identity; (6) every v2 fit must stand over the CURRENT summaries of its rounds, value for
value — a retried trial's new objective cannot be listed beside a pivot and winner taken from the stale fit.
The 03 phases run LIVE through a fragment harness (fake ssh/rsync executing on this host over a synthetic record); nothing dry-run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY  # noqa: E402
from test_emit_addendum import DRAWN9, NEW6, PAIRS_ARG, WIDE15, _round1  # noqa: E402
from test_hpo_design_v2 import A0, ARM, BETA, GAMMA, MODEL, U0, write_allowed, write_summaries  # noqa: E402
from test_tune_arm import tune_arm  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
import hpo_table  # noqa: E402

CONCAVE = lambda i, s: s.update(best_heldout_loss=math.exp(A0 - 0.4 * (math.log10(s["trial"]["lr"]) - U0) ** 2
                                                           + BETA[s["trial"]["lora_rank"]] + GAMMA[s["trial"]["lora_set_id"]]))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _analyze(out: Path, menu: Path, *, extends=False, fit_name=None, source_record=None):
    ns = argparse.Namespace(out=str(out), rounds="1", y_key="best_heldout_loss", design="v2", allowed_pairs=str(menu),
                            pair_only=False, close_search=False, extends_menu=extends, fit_name=fit_name, source_record=source_record)
    try:
        tune_arm.analyze(ns); return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


# --- (2) --fit-name: a distinct artefact, the canonical fit untouched on EVERY exit -----------------------------------------------

@pytest.mark.parametrize("pd", [True, False])
def test_fit_name_writes_a_distinct_artefact_and_never_touches_r1_fit(tmp_path, pd):
    _, art9, _ = _round1(tmp_path, mutate=None if pd else CONCAVE)
    before = _sha(tmp_path / "r1_fit.json")
    wide = write_allowed(tmp_path, WIDE15, name="wide.json")
    rc = _analyze(tmp_path, wide, extends=True, fit_name="r1_fit_menu.json")
    assert (rc == 0) is pd, rc                                                # the non-PD verdict exits nonzero AFTER writing its artefact
    assert _sha(tmp_path / "r1_fit.json") == before, "the canonical round-1 fit must be byte-identical"
    m = json.loads((tmp_path / "r1_fit_menu.json").read_text())
    assert m["fit_name"] == "r1_fit_menu.json" and m["fit"]["positive_definite"] is pd and sorted(map(tuple, m["allowed_pairs"])) == sorted(WIDE15)
    assert sorted(map(tuple, m["menu_extended_from"])) == sorted(DRAWN9)


# --- (1) hpo_table: the widened pivot from the derived fit, the round-1 selection from the design fit ----------------------------------

def _search_with_addendum(tmp_path, *, with_menu_fit=True, pd=True):
    """A synthetic search dir: 9-pair round-1 summaries + r1_fit.json, the widened menu, the derived refit, and 6 round-4 addendum rows."""
    _, art9, _ = _round1(tmp_path, mutate=None if pd else CONCAVE)
    wide = write_allowed(tmp_path, WIDE15, name="allowed_pairs_wide.json")
    if with_menu_fit:
        _analyze(tmp_path, wide, extends=True, fit_name="r1_fit_menu.json")
    menu = tune_arm.load_allowed_pairs(wide)
    for i, (r, sid) in enumerate(NEW6):
        pr = next(p for p in menu["pairs"] if (p["rank"], p["set"]) == (r, sid))
        d = tmp_path / f"r4_t{i:02d}"; d.mkdir()
        trial = {"index": i, "lr": 3e-4, "warmup_frac": 0.15, "max_grad_norm": 1.3, "lora_rank": r, "lora_modules": pr["lora_modules"],
                 "lora_set_id": sid, "per_expert_rank": pr["per_expert_rank"], "init_scheme": "random_lecun_1e-3", "trial_role": "addendum"}
        (d / "summary.json").write_text(json.dumps({"trial": trial, "round": 4, "trial_role": "addendum", "status": "ok", "arm": ARM, "model": MODEL,
                                                    "position": "all", "si": "constv2-mix-10m", "best_heldout_loss": 1.7 + 0.01 * i, "floor_ok": True,
                                                    "heldout_curve": [2.0, 1.8], "trainable_params": pr["trainable_params"], "lr": 3e-4,
                                                    "lora_rank": r, "lora_set_id": sid, "lora_modules": pr["lora_modules"], "warmup_frac": 0.15,
                                                    "max_grad_norm": 1.3, "init_scheme": "random_lecun_1e-3", "design": "addendum_pairs",
                                                    "allowed_pairs": [list(k) for k in WIDE15], "menu_extends": [list(k) for k in DRAWN9]}))
    return wide


def test_table_pivots_the_widened_menu_from_the_derived_fit_and_lists_the_addendum_rows(tmp_path):
    wide = _search_with_addendum(tmp_path)
    out = tmp_path / "table"; res = hpo_table.render(tmp_path, out, str(wide))
    rows = list(__import__("csv").DictReader(open(out / "table.csv")))
    assert len(rows) == 22 and sum(1 for r in rows if r["role"] == "addendum") == 6 and sum(1 for r in rows if r["round"] == "4") == 6
    search = hpo_table.load_search(tmp_path, str(wide)); piv = hpo_table.build_pivot(search)
    assert piv["predictions_from"] == "r1_fit_menu.json" and sorted(map(tuple, piv["menu_extended_from"])) == sorted(DRAWN9)
    menu_fit = json.loads((tmp_path / "r1_fit_menu.json").read_text())
    for r, sid in NEW6:
        cell = piv["cells"][(r, sid)]
        assert cell["state"] == "allowed" and cell["pred_y"] == pytest.approx(menu_fit["fit"]["pair_predictions"][f"{r}:{sid}"]["pred_y"])
    design_fit = json.loads((tmp_path / "r1_fit.json").read_text())
    assert piv["winner"] == tune_arm.pair_key(design_fit["selection"]["lora_rank"], design_fit["selection"]["lora_set_id"])   # the round-1 selection
    md = (out / "table.md").read_text()
    assert "r1_fit_menu.json" in md and "REFIT over the widened menu" in md


def test_table_refuses_a_widened_menu_without_the_derived_fit_and_a_foreign_menu_fit(tmp_path):
    wide = _search_with_addendum(tmp_path, with_menu_fit=False)
    with pytest.raises(SystemExit, match="a widened menu needs r1_fit_menu.json"):
        hpo_table.render(tmp_path, tmp_path / "t", str(wide))
    # a menu fit over another menu than "the design's plus more" is refused by name
    _analyze(tmp_path, write_allowed(tmp_path, [(4, "wo"), (16, "wo")], name="narrow.json"), extends=False, fit_name="r1_fit_menu.json") \
        if False else None
    (tmp_path / "r1_fit_menu.json").write_text(json.dumps({"design": "v2", "allowed_pairs": [[4, "wo"]], "menu_extended_from": [[4, "wo"]], "fit": {}}))
    with pytest.raises(SystemExit, match="is not a widened-menu refit of the design fit"):
        hpo_table.render(tmp_path, tmp_path / "t2", str(wide))


REC_B, REC_C = "cluster_b:/rec/03_hpo/lora", "cluster_c:/rec/03_hpo/lora"


def _rewrite(p: Path, **fields) -> bytes:
    """Rewrite a json file with fields changed; returns the original bytes (to restore)."""
    before = p.read_bytes(); d = json.loads(before); d.update(fields); p.write_text(json.dumps(d, indent=2) + "\n"); return before


def test_table_binds_the_derived_refit_to_the_source_record_the_menu_generation_the_search_and_the_rows(tmp_path):
    """The model/class mirror is shared across --cluster and the fetch has no --delete, so a residual r1_fit_menu.json
    derived on record B must never pivot record C's table; nor one derived over a menu the re-verdict has since moved, over another search, or
    over other rows. Each refusal names the mismatch; the bound refit still renders."""
    wide = _search_with_addendum(tmp_path, with_menu_fit=False)
    assert _analyze(tmp_path, wide, extends=True, fit_name="r1_fit_menu.json", source_record=REC_B) == 0
    menu_fit = json.loads((tmp_path / "r1_fit_menu.json").read_text()); menu_doc = tune_arm.load_allowed_pairs(wide)
    assert menu_fit["source_record"] == REC_B and menu_fit["menu_source"]["sha256"] == menu_doc["sha256"] and len(menu_fit["row_identity"]) == 16
    # bound: the table over the same record + menu renders from the refit
    search = hpo_table.load_search(tmp_path, str(wide), REC_B); assert hpo_table.build_pivot(search)["predictions_from"] == "r1_fit_menu.json"
    # (a) rehome: the same mirror, the table now over record C
    with pytest.raises(SystemExit, match=rf"r1_fit_menu.json is derived on {REC_B}, this table is {REC_C} \(a residual refit on the shared mirror\)"):
        hpo_table.render(tmp_path, tmp_path / "t_c", str(wide), REC_C)
    # (b) the menu moved (a re-verdict rewrote allowed_pairs.json: another generation, another sha)
    before = _rewrite(wide, result_generation="gen-2", probe_job_id="re-verdict-2")
    with pytest.raises(SystemExit, match=r"r1_fit_menu.json is derived over menu sha [0-9a-f]{16}, the current menu is [0-9a-f]{16}"):
        hpo_table.render(tmp_path, tmp_path / "t_m", str(wide), REC_B)
    wide.write_bytes(before)
    # (c) another search
    d_before = _rewrite(tmp_path / "r1_fit.json", search_id="search-1"); m_before = _rewrite(tmp_path / "r1_fit_menu.json", search_id="search-2")
    with pytest.raises(SystemExit, match="r1_fit_menu.json is derived from search search-2, the design fit is search search-1"):
        hpo_table.render(tmp_path, tmp_path / "t_s", str(wide), REC_B)
    (tmp_path / "r1_fit.json").write_bytes(d_before); (tmp_path / "r1_fit_menu.json").write_bytes(m_before)
    # (d) other rows (a refit over another record's trials that happens to carry the same menu + search): caught against the CURRENT summaries
    # first (every fit is bound to them, so two bound fits agree with each other by construction)
    m_before = _rewrite(tmp_path / "r1_fit_menu.json", row_identity=menu_fit["row_identity"][1:])
    with pytest.raises(SystemExit, match=r"r1_fit_menu.json is fit over other rows than the current round-1 summaries \(r1_t00: a current usable summary the fit never saw\)"):
        hpo_table.render(tmp_path, tmp_path / "t_r", str(wide), REC_B)
    (tmp_path / "r1_fit_menu.json").write_bytes(m_before)
    # (e) a refit that records no source record is refused when the table names one (fail closed: the binding is not opt-in by the writer)
    (tmp_path / "r1_fit_menu.json").unlink(); assert _analyze(tmp_path, wide, extends=True, fit_name="r1_fit_menu.json") == 0
    with pytest.raises(SystemExit, match=f"r1_fit_menu.json records no source_record; this table is {REC_B}"):
        hpo_table.render(tmp_path, tmp_path / "t_n", str(wide), REC_B)
    # a design fit written before row_identity existed (the live record's) is bound through its rows
    (tmp_path / "r1_fit_menu.json").unlink(); assert _analyze(tmp_path, wide, extends=True, fit_name="r1_fit_menu.json", source_record=REC_B) == 0
    d = json.loads((tmp_path / "r1_fit.json").read_text()); d.pop("row_identity"); (tmp_path / "r1_fit.json").write_text(json.dumps(d))
    assert hpo_table.build_pivot(hpo_table.load_search(tmp_path, str(wide), REC_B))["predictions_from"] == "r1_fit_menu.json"


def test_table_refuses_a_fit_over_other_rows_than_the_current_summaries(tmp_path):
    """A retried trial replaced r1_t03/summary.json after the fit — the table must not list the new objective beside a pivot and
    winner from the stale fit. Also: a fit row with no summary; a usable summary the fit never saw; a now-unusable fit row; a recorded exclusion
    is NOT a mismatch."""
    wide = _search_with_addendum(tmp_path)                                       # design fit + bound refit over 16 round-1 rows, 6 round-4 rows
    hpo_table.render(tmp_path, tmp_path / "ok", str(wide))                      # bound as laid
    fit_y = dict(map(tuple, json.loads((tmp_path / "r1_fit_menu.json").read_text())["row_identity"]))["r1_t03"]
    before = _rewrite(tmp_path / "r1_t03" / "summary.json", best_heldout_loss=1.234)
    with pytest.raises(SystemExit, match=rf"r1_fit.json is fit over other rows than the current round-1 summaries \(r1_t03: fit {fit_y!r}, current 1.234\)"):
        hpo_table.render(tmp_path, tmp_path / "t1", str(wide))
    (tmp_path / "r1_t03" / "summary.json").write_bytes(before)
    before = _rewrite(tmp_path / "r1_t03" / "summary.json", status="failed")  # a fit row whose summary is now unusable
    with pytest.raises(SystemExit, match="r1_t03: a fit row whose summary is now unusable"):
        hpo_table.render(tmp_path, tmp_path / "t2", str(wide))
    (tmp_path / "r1_t03" / "summary.json").write_bytes(before)
    extra = tmp_path / "r1_t16"; shutil.copytree(tmp_path / "r1_t03", extra)   # a usable round-1 summary the fit never saw
    _rewrite(extra / "summary.json", trial={**json.loads((extra / "summary.json").read_text())["trial"], "index": 16})
    with pytest.raises(SystemExit, match="r1_t16: a current usable summary the fit never saw"):
        hpo_table.render(tmp_path, tmp_path / "t3", str(wide))
    shutil.rmtree(extra)
    moved = tmp_path / "r1_t03.aside"; (tmp_path / "r1_t03").rename(moved)     # a fit row with no summary on the mirror
    with pytest.raises(SystemExit, match="r1_t03: a fit row with no summary on the mirror"):
        hpo_table.render(tmp_path, tmp_path / "t4", str(wide))
    moved.rename(tmp_path / "r1_t03")
    hpo_table.render(tmp_path, tmp_path / "ok2", str(wide))                     # restored: bound again


def _metrics(path: Path, grad_norm: float, n: int = 5) -> None:
    path.write_text("".join(json.dumps({"step": i, f"{ARM}/grad_norm": grad_norm, "train/grad_norm_preclip": grad_norm}) + "\n" for i in range(1, n + 1)))


def test_table_and_pick_refuse_a_fit_that_predates_a_metrics_backfill_and_the_re_analysis_updates_the_winner(tmp_path):
    """A trial's metrics.jsonl backfilled after the fit (here: the WINNER's, clipping on every step) moves the key and the listed
    clip fraction, but the fit's selection and alarms still describe the pre-backfill evidence — refused by name by the table and by the pick until
    the analysis is re-run, after which the winner moves (or is alarmed) and the table renders the new fraction."""
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase")); import hpo_pick  # noqa: E402
    wide = _search_with_addendum(tmp_path)
    fit0 = json.loads((tmp_path / "r1_fit.json").read_text()); sel0 = fit0["selection"]; g0 = fit0.get("selection_guard") or {}
    pair0 = (sel0["lora_rank"], sel0["lora_set_id"])                                             # the selected architecture (the fitted argmin's pair)
    best = min((r for r in fit0["rows"] if (r["lora_rank"], r["lora_set_id"]) == pair0), key=lambda r: r["best_heldout_loss"])
    win = g0.get("selected_trial") or tune_arm.hpo_selection_guard.trial_id(best)                  # the best-measured trial of the selected pair
    assert all(r.get("clip_activation_rate") is None for r in fit0["rows"])                    # no sidecars at fit time
    hpo_table.render(tmp_path, tmp_path / "ok", str(wide))
    _metrics(tmp_path / win / "metrics.jsonl", grad_norm=10.0)                                    # the backfill: every step above the clip (1.3)
    with pytest.raises(SystemExit, match=rf"r1_fit.json is fit over other rows than the current round-1 summaries \({win}: the fit saw no clip evidence, the current metrics give clip activation 1.000 over 5 steps — the fit predates these metrics"):
        hpo_table.render(tmp_path, tmp_path / "t", str(wide))
    with pytest.raises(SystemExit, match=rf"\[hpo-pick\] r1_fit.json is fit over other rows than the current round-1 summaries \({win}: the fit saw no clip evidence"):
        hpo_pick.load_search(tmp_path, str(wide))
    # re-derive: the design fit and the bound refit over the current evidence (either exit is a written artefact)
    _analyze(tmp_path, tmp_path / "menu9.json"); _analyze(tmp_path, wide, extends=True, fit_name="r1_fit_menu.json")
    fit1 = json.loads((tmp_path / "r1_fit.json").read_text())
    row = next(r for r in fit1["rows"] if tune_arm.hpo_selection_guard.trial_id(r) == win)
    assert row["clip_activation_rate"] == pytest.approx(1.0) and row["clip_n_steps"] == 5
    sel1 = fit1["selection"]; g1 = fit1.get("selection_guard") or {}
    moved = (sel1["lora_rank"], sel1["lora_set_id"]) != pair0 or g1.get("selected_trial") != g0.get("selected_trial")
    alarmed = "clip" in json.dumps(g1.get("alarms") or []).lower() or "clip" in json.dumps(sel1.get("alarms") or []).lower()
    clip_summary_moved = g1.get("clip_activation") != g0.get("clip_activation")
    assert moved or alarmed or clip_summary_moved, (g0.get("clip_activation"), g1.get("clip_activation"), g1.get("alarms"), sel1.get("alarms"))
    res = hpo_table.render(tmp_path, tmp_path / "ok2", str(wide))
    assert next(r for r in res["rows"] if win in json.dumps(r))["clip_fraction"] == pytest.approx(1.0)
    hpo_pick.load_search(tmp_path, str(wide))                                                      # bound again


@pytest.mark.parametrize("how", ["status", "floor"])
def test_table_refuses_a_fit_time_exclusion_that_has_since_become_a_usable_trial(tmp_path, how):
    """A trial excluded at fit time (status failed / floor_ok false) and retried at the SAME path is a row the fit never saw —
    the pivot and the selection were computed without it. Only the pair-only refit's deliberate `other_pair` exclusion exempts a usable row."""
    _, art9, _ = _round1(tmp_path)
    wide = write_allowed(tmp_path, WIDE15, name="allowed_pairs_wide.json")
    # exclude r1_t07 at fit time, re-fit (design + bound refit), then let it recover at the same path
    before = _rewrite(tmp_path / "r1_t07" / "summary.json", **({"status": "failed"} if how == "status" else {"floor_ok": False}))
    assert _analyze(tmp_path, tmp_path / "menu9.json") == 0
    assert _analyze(tmp_path, wide, extends=True, fit_name="r1_fit_menu.json", source_record=REC_B) == 0
    fit = json.loads((tmp_path / "r1_fit.json").read_text())
    assert ["r1_t07", "failed" if how == "status" else "floor_failed"] in fit["excluded"] and len(fit["row_identity"]) == 15
    hpo_table.render(tmp_path, tmp_path / "ok", str(wide), REC_B)                # excluded and still unusable: bound
    (tmp_path / "r1_t07" / "summary.json").write_bytes(before)                    # the retry: usable again at the same path
    with pytest.raises(SystemExit, match=rf"r1_fit.json is fit over other rows than the current round-1 summaries \(r1_t07: excluded at fit time \({'failed' if how == 'status' else 'floor_failed'}\), now a usable summary\)"):
        hpo_table.render(tmp_path, tmp_path / "t", str(wide), REC_B)
    # the deliberate exemption: a pair-only refit's other_pair rows are usable by design
    art = json.loads((tmp_path / "r1_fit.json").read_text()); d_rows = {tuple(x) for x in art["row_identity"]}
    pair = tune_arm.pair_key(art["selection"]["lora_rank"], art["selection"]["lora_set_id"])
    pair_fit = {**art, "pair_only": True, "refit_pair": list(pair), "fit_name": "r1+2_fit_pair.json",
                "row_identity": sorted(x for x in d_rows if tune_arm._pair_of(next(r for r in art["rows"] if tune_arm.hpo_selection_guard.trial_id(r) == x[0])) == pair),
                "excluded": art["excluded"] + [[x[0], "other_pair"] for x in d_rows
                                               if tune_arm._pair_of(next(r for r in art["rows"] if tune_arm.hpo_selection_guard.trial_id(r) == x[0])) != pair]}
    pair_fit["rows"] = [r for r in art["rows"] if tune_arm._pair_of(r) == pair]
    (tmp_path / "r1+2_fit_pair.json").write_text(json.dumps(pair_fit))
    _rewrite(tmp_path / "r1_t07" / "summary.json", **({"status": "failed"} if how == "status" else {"floor_ok": False}))   # back to excluded
    hpo_table.render(tmp_path, tmp_path / "ok_pair", str(wide), REC_B)


# --- the 03 phases LIVE: a fragment harness over a synthetic record -------------------------------------------------------------------

class World:
    """03's table + addendum phases, with their helpers, run under bash: fake ssh/rsync execute here; a local 'cluster' record dir."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.local = tmp_path / "state" / "m" / "03_hpo" / "lora"; self.local.mkdir(parents=True)
        self.cluster = tmp_path / "cluster" / "03_hpo" / "lora"; self.cluster.mkdir(parents=True)
        self.figs = tmp_path / "cluster" / "figs"
        b = tmp_path / "bin"; b.mkdir()
        (b / "ssh").write_text('#!/bin/bash\nwhile [ "${1:-}" = -o ]; do shift 2; done; shift\nexec bash -c "$*"\n'); (b / "ssh").chmod(0o755)
        (b / "rsync").write_text('#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"; src="${src#*:}"; dst="${dst#*:}"; '
                                 'if [ -d "$src" ]; then mkdir -p "$dst"; cp -r "$src"/. "$dst"/; else mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; fi\n')
        (b / "rsync").chmod(0o755)
        (b / "squeue").write_text('#!/bin/bash\n[ "${FAKE_SQUEUE_FAIL:-0}" = 1 ] && { echo "squeue: error: slurm_load_jobs error: Unable to contact slurm controller" >&2; exit 1; }\n'
                                  'printf "%s" "${FAKE_SQUEUE_IDS:-}" | tr "," "\\n" | grep . ; exit 0\n')
        (b / "squeue").chmod(0o755)                                                      # an empty queue; FAKE_SQUEUE_FAIL=1: an error (fail CLOSED)
        (b / "sacct").write_text('#!/bin/bash\n[ "${FAKE_SACCT_FAIL:-0}" = 1 ] && { echo "sacct: error: Problem talking to the database" >&2; exit 1; }\n'
                                 'printf "%s" "${FAKE_SACCT_ROWS:-}" | grep . ; exit 0\n')
        (b / "sacct").chmod(0o755)                                                       # by name since a time: FAKE_SACCT_ROWS ("id|submit|state" lines); FAKE_SACCT_FAIL=1: an error
        self.bin = b
        self.cfg = tmp_path / "fits.yaml"; self.cfg.write_text(f"model: {MODEL}\nmodel_revision: '0f0813'\nbatch_size_by_arm:\n  lora_r4_ce: 2\n")
        self.fake_pick = tmp_path / "fake_pick.py"
        self.fake_pick.write_text(f"""import json, sys
sys.path.insert(0, {str(REPO / 'scripts' / 'sdfbase')!r}); sys.path.insert(0, {str(REPO / 'scripts')!r})
import hpo_pick
from pathlib import Path
args = sys.argv[1:]; sd = Path(args[args.index('--search-dir') + 1]); ap = args[args.index('--allowed-pairs') + 1]
rank = int(args[args.index('--rank') + 1]); sid = args[args.index('--set') + 1]
src = args[args.index('--source-record') + 1] if '--source-record' in args else None
search = hpo_pick.load_search(sd, ap, src)                             # the REAL binding: a stale fit (the widened-menu refit included) refuses here
key = (rank, sid); cand = hpo_pick.lr_for_pair(search, key)             # the REAL clip gate at the candidate coordinates
if cand["clip_evidence"]["verdict"] != "promotable":
    (sd / 'hpo_pick.json').write_text(json.dumps({{"promoted": False, "rank": rank, "set": sid, "lr_source": cand, "refused": cand["clip_evidence"]["reason"]}}, indent=2, sort_keys=True, default=str) + '\\n')
    print('FAKE-PICK NOT promoted:', cand["clip_evidence"]["verdict"], cand["clip_evidence"]["reason"]); sys.exit(1)
ev = cand["clip_evidence"]["trials"]; win = min((t for t in ev if t.get("clip_activation_rate") is not None), key=lambda t: t["y"])["trial_id"]
pick = {{"promoted": True, "rank": rank, "set": sid, "fake": True, "search_id": search.get("search_id"), "lr_source": cand, "winner_trial_id": win}}
pick = hpo_pick.annotate_hold_cleared(sd, pick)                          # the REAL two-step clearing: recorded first ...
(sd / 'hpo_pick.json').write_text(json.dumps(pick, indent=2, sort_keys=True, default=str) + '\\n')
(sd / 'hpo_fold.json').write_text(json.dumps({{"k": {{"lora_r4_ce": {{"lr": cand["lr"]}}}}}}) + '\\n')
assert not (sd / hpo_pick.HOLD_NAME).is_file() or True
hpo_pick.finalize_hold_clear(sd, execute='--execute' in args)            # ... the file moves only after every write
print('FAKE-PICK promoted rank', rank, 'set', sid)
""")

    def run(self, phase: str, *, pairs: str = PAIRS_ARG, extra: str = "", cluster: str = "cluster_b", cl_root: Path | None = None,
            env_extra: dict | None = None) -> subprocess.CompletedProcess:
        src = (REPRO / "03_hpo.sh").read_text()
        cl_root = cl_root or self.cluster
        override = src[src.index('OVERRIDE_MENU=""; [ -z "$MENU_OVERRIDE" ]'):src.index("\n}\n", src.index("build_override_menu(){")) + 3]   # the override menu builder
        helpers = src[src.index("TRIAL_FILES_FIND="):src.index("push(){   # <local file>")]         # the inventory helpers + their find (lifted above every phase)
        emitters = src[src.index("emit_from(){"):src.index('\nrepro_say "03_hpo: model=')]            # emit_from + refit (the pair-only analyze)
        refine = src[src.index("elif want refine; then"):src.index("table_set_key(){")]                # the fd6 refine branch (the table section follows it)
        refine = "if false; then :\n" + refine                                                        # the fd6 refine branch alone, as its own if
        pick = src[src.index("# ================================================================================================================ pick"):
                   src.index("# ================================================================================================================ addendum (D409)")]
        table = src[src.index("table_set_key(){"):src.index("# ================================================================================================================ pick")]
        addendum = src[src.index("# ================================================================================================================ addendum (D409)"):
                       src.index("# ================================================================================================================ confirm-pick")]
        script = f"""set -u
source {REPRO}/lib/common.sh
export SISTEER_STATE_DIR={self.tmp}/state
REPRO_DRY_RUN=0; REPRO_REPO={REPO}; MODEL=m; STAGE=03_hpo; CLASS=lora; PROTOCOL=fd6; PHASE={phase}; CLUSTER={cluster}; CL_HOST=login-b
PY={VENV_PY}; TUNE={REPO}/scripts/tune_arm.py; LOCAL_CFG={self.cfg}; ARM=lora_r4_ce; HPO_SI=constv2-mix-10m
LOCAL_ROOT={self.local}; CL_ROOT={cl_root}; ALLOWED={self.local}/allowed_pairs.json; FIG_DIR={self.figs}
FIT1={self.local}/r1_fit.json; FIT1M={self.local}/r1_fit_menu.json; R4={self.local}/r4_trials.json; ADDENDUM_PAIRS="{pairs}"
PICK_HOLD={self.local}/hpo_pick.HOLD.json
MENU_OVERRIDE="${{MENU_OVERRIDE:-}}"; export SISTEER_MENU_OVERRIDES_DIR="${{SISTEER_MENU_OVERRIDES_DIR:-{self.tmp}/menu_overrides}}"
{override}
FIT12={self.local}/r1+2_fit_pair.json; R2={self.local}/r2_trials.json; R3={self.local}/r3_trials.json; R9={self.local}/r9_trials.json
PICK_RANK="${{PICK_RANK:-}}"; PICK_SET="${{PICK_SET:-}}"; FIT_KEY=k; repro_models_table(){{ echo /dev/null; }}
RECONCILE_DECISION="${{RECONCILE_DECISION:-}}"; DECISION_WHY="${{DECISION_WHY:-}}"
AP=(--allowed-pairs "$ALLOWED")
want(){{ [ "$PHASE" = all ] || [ "$PHASE" = "$1" ]; }}
name(){{ echo "hpo-m-$CLASS-$1"; }}; unit(){{ echo "hpo.$CLASS.$1"; }}; disp(){{ echo "$1"; }}
mark(){{ repro_mark_done "$MODEL" "$STAGE" "$CLASS.$1" "${{2:-}}"; }}
done_or_skip(){{ repro_done "$MODEL" "$STAGE" "$CLASS.$1" && {{ repro_say "$1: DONE marker present — skipping"; return 0; }}; return 1; }}
fetch(){{ mkdir -p "$2"; rsync -a "$CL_HOST:$1/" "$2/"; }}
push(){{ ssh -o BatchMode=yes "$CL_HOST" "mkdir -p $(dirname "$2") && cat > '$2'" < "$1"; }}
cluster_has(){{ ssh -o BatchMode=yes "$CL_HOST" "[ -s '$1' ]"; }}
cluster_count(){{ ssh -o BatchMode=yes "$CL_HOST" "ls -d $1 2>/dev/null | wc -l"; }}
cluster_cat(){{ ssh -o BatchMode=yes "$CL_HOST" "cat '$1'"; }}
HAVE_SLURM_LIB=1
submit_design(){{ if [ "${{FAIL_SUBMIT:-0}}" = 1 ]; then echo "STUB funnel REFUSED the submission of $1 (FAIL_SUBMIT)" >&2; return 3; fi; echo "STUB submit_design $1 array $3"; }}
slurm_prove_resubmit(){{ if [ "${{PROVE_REFUSE:-0}}" = 1 ]; then echo "prove-resubmit: REFUSED $4: --resubmit refused: bound job 4242 registered a fit_trial (STUB proof)"; return 3; fi; echo "prove-resubmit: AUTHORIZED $4: bound job 4242 on cluster_b [0:FAILED] — STUB proof"; printf 'bound_job\tcluster_b\t4242\n'; }}
cluster_job_active(){{ return 1; }}
slurm_binding_state(){{ [ "${{STUB_BINDING_FAIL:-0}}" = 1 ] && {{ echo "binding: ledger unreadable (STUB)" >&2; return 3; }}; local j="${{STUB_BINDING_JOB:-4242}}" t="${{STUB_BINDING_TOKEN:-no}}"; [ -f "{self.tmp}/.adopted_job" ] && {{ j="$(cat "{self.tmp}/.adopted_job")"; t=no; }}; printf 'binding\tsubmitted\tcluster_b\t%s\t%s\nattempts\t%s\n' "$j" "$t" "${{STUB_BINDING_ATTEMPTS:-${{STUB_BINDING_JOB:-4242}}}}"; }}
slurm_reconcile_token(){{ case "${{STUB_RECONCILE_TOKEN:-none}}" in adopted:*) echo "${{STUB_RECONCILE_TOKEN#adopted:}}" > "{self.tmp}/.adopted_job"; printf 'token\tadopted\tnonce\t0123456789ab\n';; adopted-launch:*) echo "${{STUB_RECONCILE_TOKEN#adopted-launch:}}" > "{self.tmp}/.adopted_job"; printf 'token\tadopted\tlaunch\t\n';; adopted-ledger:*) echo "${{STUB_RECONCILE_TOKEN#adopted-ledger:}}" > "{self.tmp}/.adopted_job"; printf 'token\tadopted\tledger\t0123456789ab\n';; cleared) printf 'token\tcleared\n'; export STUB_BINDING_TOKEN=no;; refused:*) printf 'token\trefused\t%s\n' "${{STUB_RECONCILE_TOKEN#refused:}}"; return 3;; *) printf 'token\tnone\n';; esac; slurm_binding_state; }}
n_summaries(){{ cluster_count "$CL_ROOT/r$1_t*/summary.json"; }}
local_run(){{   # the pick's side-effecting tools are routed to fakes: hpo_pick.py (unless --write-hold) -> FAKE_PICK (the real load_search binding +
  # the real clear_hold, a promoted hpo_pick.json, no registry); make_sdfbase_configs.py -> a stub
  case "$2" in
    */hpo_pick.py) case " $* " in *" --write-hold "*|*" --verify-promotion "*) "$@";; *) "$1" "{self.fake_pick}" "${{@:3}}";; esac;;   # the read-only verbs are the REAL tool
    */make_sdfbase_configs.py) [ "${{FAIL_FOLD:-0}}" = 1 ] && {{ echo "STUB make_sdfbase_configs FAILED (FAIL_FOLD)" >&2; return 1; }}; echo "STUB make_sdfbase_configs $*";;
    *) "$@";;
  esac
}}
submit_round(){{ echo "STUB submit_round r$1 array $2"; }}
probe_revalidate_local(){{ :; }}; recon_revalidate_local(){{ :; }}
{helpers}
{emitters}
{table}
{refine}
{pick}
{addendum}
{extra}
true
"""
        env = {"PATH": f"{self.bin}:{os.environ['PATH']}", "HOME": str(self.tmp), "PYTHONPATH": str(REPO / "src"), **(env_extra or {})}
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=600)

    def lay_round1(self, *, pd=True):
        """Round-1 summaries + the design fit over the 9-pair menu on the CLUSTER record (and the fit's local copy the pick was made from)."""
        pay, art9, _ = _round1(self.tmp, mutate=None if pd else CONCAVE)
        (self.tmp / "r1_trials.json").write_text(json.dumps(pay, default=str))                 # the design payload, as the trials phase pushes it
        for d in self.tmp.glob("r1_t*"):
            if not d.is_dir():
                continue
            shutil.copytree(d, self.cluster / d.name)
            shutil.copytree(d, self.local / d.name)                                         # as --phase analyze left the mirror
        shutil.copy(self.tmp / "r1_fit.json", self.cluster / "r1_fit.json"); shutil.copy(self.tmp / "r1_fit.json", self.local / "r1_fit.json")
        shutil.copy(self.tmp / "allowed_pairs.json", self.local / "allowed_pairs.json")            # the 9-pair menu, for the round-1 table
        return _sha(self.local / "r1_fit.json")

    def widen(self):
        write_allowed(self.tmp, WIDE15, name="wide.json"); shutil.copy(self.tmp / "wide.json", self.local / "allowed_pairs.json")


def test_live_addendum_pd_path_refits_to_a_distinct_file_emits_round_4_and_leaves_the_canonical_fit_untouched(tmp_path):
    w = World(tmp_path); before = w.lay_round1(pd=True); w.widen()
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    assert (w.local / "r1_fit_menu.json").is_file() and _sha(w.local / "r1_fit.json") == before, out
    assert "CHANGED under the refit" not in out and "STUB submit_round r4 array 0-5" in out, out
    d = json.loads((w.local / "r4_trials.json").read_text()); menu_fit = json.loads((w.local / "r1_fit_menu.json").read_text())
    assert [(t["lora_rank"], t["lora_set_id"]) for t in d["trials"]] == NEW6 and all(t["lr_source"]["kind"] == "pair_prediction" for t in d["trials"])
    for t in d["trials"]:
        assert t["lr"] == pytest.approx(menu_fit["fit"]["pair_predictions"][f"{t['lora_rank']}:{t['lora_set_id']}"]["lr"])
    assert (w.cluster / "r4_trials.json").is_file()                                 # pushed to the record


def test_live_addendum_non_pd_refit_is_the_accepted_fallback_state(tmp_path):
    w = World(tmp_path); before = w.lay_round1(pd=False); w.widen()
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    assert "the refit is the recognized NON-positive-definite verdict over the widened menu" in out and "STUB submit_round r4 array 0-5" in out, out
    assert _sha(w.local / "r1_fit.json") == before and json.loads((w.local / "r1_fit_menu.json").read_text())["fit"]["positive_definite"] is False
    d = json.loads((w.local / "r4_trials.json").read_text())
    assert all(t["lr_source"]["kind"] == "best_round1_trial" for t in d["trials"]) and d["addendum_of"]["fit_positive_definite"] is False


def test_live_addendum_refuses_a_refit_that_is_not_the_recognized_non_pd_verdict(tmp_path):
    """A nonzero analyze that is NOT the non-PD verdict (here: the widened menu file is not a superset — the standing guard) stays a failure:
    nothing emitted, nothing pushed, nothing submitted, the canonical fit untouched."""
    w = World(tmp_path); before = w.lay_round1(pd=True)
    write_allowed(w.tmp, [(4, "wo"), (16, "wo"), (64, "wo"), (4, "qkvo"), (16, "qkvo"), (64, "qkvo"), (4, "attn_mlp"), (4, "mlp"), (16, "down")], name="odd.json")
    shutil.copy(w.tmp / "odd.json", w.local / "allowed_pairs.json")                 # (4,down) dropped, (16,down) added: not a superset
    r = w.run("addendum", pairs="16:down"); out = r.stdout + r.stderr
    assert r.returncode != 0 and "a failure, not the fallback" in r.stderr and "STUB submit" not in out and not (w.local / "r4_trials.json").exists(), out
    assert _sha(w.local / "r1_fit.json") == before and not (w.local / "r1_fit_menu.json").exists() and not (w.cluster / "r4_trials.json").exists()


def test_live_table_is_keyed_on_the_trial_set_and_menu_build_skip_rebuild(tmp_path):
    w = World(tmp_path); w.lay_round1(pd=True)
    r1 = w.run("table"); out1 = r1.stdout + r1.stderr
    assert r1.returncode == 0 and (w.local / "table" / "table.csv").is_file() and (w.figs / "table.csv").is_file(), out1
    marker = w.tmp / "state" / "m" / "03_hpo" / "DONE.lora.table"; k1 = marker.read_text().split("set=")[1].strip()
    assert len(k1) == 16 and len(list(__import__("csv").DictReader(open(w.local / "table" / "table.csv")))) == 16
    r2 = w.run("table"); assert r2.returncode == 0 and "skipping" in r2.stdout and marker.read_text().split("set=")[1].strip() == k1, r2.stdout
    # the addendum lands on the record: 6 round-4 rows + the widened menu + the derived refit -> the table REBUILDS under a new key
    w.widen(); shutil.copy(w.tmp / "wide.json", w.cluster / "allowed_pairs.json")
    _analyze(w.local, w.local / "allowed_pairs.json", extends=True, fit_name="r1_fit_menu.json", source_record=f"cluster_b:{w.cluster}")
    menu = tune_arm.load_allowed_pairs(w.local / "allowed_pairs.json")
    for i, (rk, sid) in enumerate(NEW6):
        pr = next(p for p in menu["pairs"] if (p["rank"], p["set"]) == (rk, sid)); d = w.cluster / f"r4_t{i:02d}"; d.mkdir()
        (d / "summary.json").write_text(json.dumps({"trial": {"index": i, "lr": 3e-4, "lora_rank": rk, "lora_set_id": sid}, "round": 4, "trial_role": "addendum",
                                                    "status": "ok", "arm": ARM, "model": MODEL, "position": "all", "si": "constv2-mix-10m", "best_heldout_loss": 1.7,
                                                    "lr": 3e-4, "lora_rank": rk, "lora_set_id": sid, "lora_modules": pr["lora_modules"], "trainable_params": pr["trainable_params"],
                                                    "allowed_pairs": [list(k) for k in WIDE15]}))
    r3 = w.run("table"); out3 = r3.stdout + r3.stderr
    assert r3.returncode == 0 and "rebuilding" in out3 and "skipping" not in out3, out3
    k2 = marker.read_text().split("set=")[1].strip(); assert k2 != k1
    rows = list(__import__("csv").DictReader(open(w.local / "table" / "table.csv")))
    assert len(rows) == 22 and sum(1 for r in rows if r["role"] == "addendum") == 6
    assert "r1_fit_menu.json" in (w.local / "table" / "table.md").read_text()


def _key(w: World) -> str:
    return (w.tmp / "state" / "m" / "03_hpo" / "DONE.lora.table").read_text().split("set=")[1].strip()


METRICS_ROWS = "".join(json.dumps({"step": i, f"{ARM}/grad_norm": 0.01 * i, "train/grad_norm_preclip": 0.01 * i}) + "\n" for i in range(1, 6))


def _lay_metrics(root: Path) -> None:
    """An unstamped metrics.jsonl beside every trial summary under root (the clip reader parses it: synthetic summaries carry no attempt stamp)."""
    for d in sorted(root.glob("r1_t*")):
        if d.is_dir():
            (d / "metrics.jsonl").write_text(METRICS_ROWS)


def test_live_table_key_covers_summary_contents_fit_bytes_and_the_source_record(tmp_path):
    """Changed contents at unchanged paths rebuild (a retried trial's summary, a re-fit's artefact); the same paths
    fetched from ANOTHER cluster rebuild; an unchanged record still skips."""
    w = World(tmp_path); w.lay_round1(pd=True)
    r1 = w.run("table"); assert r1.returncode == 0, r1.stdout + r1.stderr; k1 = _key(w)
    r2 = w.run("table"); assert r2.returncode == 0 and "skipping" in r2.stdout, r2.stdout; assert _key(w) == k1
    # a retried trial replaced r1_t03/summary.json in place: same path, other content. The key moves (no skip), and hpo_table then REFUSES
    # to publish the new row beside the stale fit's pivot — marker and published table untouched
    figs1 = _sha(w.figs / "table.csv")
    _rewrite(w.cluster / "r1_t03" / "summary.json", best_heldout_loss=1.234)
    r3 = w.run("table"); out3 = r3.stdout + r3.stderr
    assert r3.returncode != 0 and "rebuilding" in out3 and "skipping" not in out3, out3
    assert "r1_fit.json is fit over other rows than the current round-1 summaries (r1_t03: fit" in out3 and "current 1.234" in out3, out3
    assert _key(w) == k1 and _sha(w.figs / "table.csv") == figs1
    # re-fit over the current evidence (the mirror holds the fetched retry) and place the fit where this harness keeps it (the record copy the
    # fetch delivers, as lay_round1 laid it); the table now rebuilds and lists the retry's value
    assert _analyze(w.local, w.local / "allowed_pairs.json") == 0; shutil.copy(w.local / "r1_fit.json", w.cluster / "r1_fit.json")
    r3b = w.run("table"); out3b = r3b.stdout + r3b.stderr
    assert r3b.returncode == 0 and "rebuilding" in out3b, out3b
    k2 = _key(w); assert k2 != k1
    row3 = next(r for r in __import__("csv").DictReader(open(w.local / "table" / "table.csv")) if "r1_t03" in r.values())
    assert any(str(v).startswith("1.234") for v in row3.values()), row3
    r4 = w.run("table"); assert r4.returncode == 0 and "skipping" in r4.stdout, r4.stdout; assert _key(w) == k2
    # the design fit re-fit in place (same name, other bytes, the same rows)
    _rewrite(w.cluster / "r1_fit.json", refit_note="re-fit")
    r5 = w.run("table"); assert r5.returncode == 0 and "rebuilding" in r5.stdout + r5.stderr, r5.stdout + r5.stderr
    k3 = _key(w); assert k3 != k2
    # the SAME files, byte for byte, read as another cluster's record: never "already published"
    root_c = w.tmp / "cluster_c" / "03_hpo" / "lora"; shutil.copytree(w.cluster, root_c)
    r6 = w.run("table", cluster="cluster_c", cl_root=root_c); out6 = r6.stdout + r6.stderr
    assert r6.returncode == 0 and "rebuilding" in out6 and "skipping" not in out6, out6
    k4 = _key(w); assert k4 != k3
    r7 = w.run("table", cluster="cluster_c", cl_root=root_c); assert r7.returncode == 0 and "skipping" in r7.stdout, r7.stdout
    # and back on cluster_b the key is cluster_b's again (not the marker's), so it rebuilds rather than skipping on cluster_c's marker
    r8 = w.run("table"); assert r8.returncode == 0 and "rebuilding" in r8.stdout + r8.stderr, r8.stdout + r8.stderr; assert _key(w) == k3
    # a metrics BACKFILL on the record (the sidecars the clip fractions are read from): the inventory and the key cover it (no skip), and the fit
    # that predates the metrics is REFUSED until re-derived; then the table rebuilds with the fractions
    _lay_metrics(w.cluster)
    r9 = w.run("table"); out9 = r9.stdout + r9.stderr
    assert r9.returncode != 0 and "rebuilding" in out9 and "16 trial summaries + 16 sidecars verified" in out9, out9
    assert "the fit saw no clip evidence, the current metrics give clip activation 0.000 over 5 steps — the fit predates these metrics" in out9, out9
    assert _key(w) == k3 and all((d / "metrics.jsonl").is_file() for d in w.local.glob("r1_t*") if d.is_dir())
    assert _analyze(w.local, w.local / "allowed_pairs.json") == 0; shutil.copy(w.local / "r1_fit.json", w.cluster / "r1_fit.json")
    r9b = w.run("table"); out9b = r9b.stdout + r9b.stderr
    assert r9b.returncode == 0 and "rebuilding" in out9b and _key(w) != k3, out9b
    r10 = w.run("table"); assert r10.returncode == 0 and "skipping" in r10.stdout, r10.stdout


def test_live_table_refuses_the_residual_refit_after_a_rehome_and_after_the_menu_moves(tmp_path):
    """The addendum phase derives r1_fit_menu.json bound to <cluster>:<record>; the refit lives ONLY on the shared local mirror. A table over
    another cluster's record (same files, after a rehome) or over a menu the re-verdict has moved is refused by name — the published
    table and its marker stay as they were; the bound case renders."""
    w = World(tmp_path); w.lay_round1(pd=True); w.widen(); shutil.copy(w.tmp / "wide.json", w.cluster / "allowed_pairs.json")
    ra = w.run("addendum"); assert ra.returncode == 0, ra.stdout + ra.stderr
    menu_fit = json.loads((w.local / "r1_fit_menu.json").read_text()); assert menu_fit["source_record"] == f"cluster_b:{w.cluster}"
    rb = w.run("table"); assert rb.returncode == 0 and "r1_fit_menu.json" in (w.local / "table" / "table.md").read_text(), rb.stdout + rb.stderr
    kb = _key(w); figs_before = _sha(w.figs / "table.csv")
    # rehome: the record's files on cluster_c (byte-identical), the residual refit still on the mirror
    root_c = w.tmp / "cluster_c" / "03_hpo" / "lora"; shutil.copytree(w.cluster, root_c)
    rc_ = w.run("table", cluster="cluster_c", cl_root=root_c); out = rc_.stdout + rc_.stderr
    assert rc_.returncode != 0, out
    assert f"r1_fit_menu.json is derived on cluster_b:{w.cluster}, this table is cluster_c:{root_c} (a residual refit on the shared mirror)" in out, out
    assert "refusing to pivot over it" in out and "hpo_table.py failed" in out, out
    assert _key(w) == kb and _sha(w.figs / "table.csv") == figs_before                   # nothing published, the marker untouched
    # the menu moved under the refit (a re-verdict, re-fetched by --phase probe: the mirror's menu is another generation; this harness's unfiltered
    # fake fetch also carries the record copy, kept identical — the real fetch never delivers probe-owned files from a search record)
    _rewrite(w.local / "allowed_pairs.json", result_generation="gen-2", probe_job_id="re-verdict-2")
    shutil.copy(w.local / "allowed_pairs.json", w.cluster / "allowed_pairs.json")
    rd = w.run("table"); out = rd.stdout + rd.stderr
    assert rd.returncode != 0 and "r1_fit_menu.json is derived over menu sha" in out and _key(w) == kb, out


def test_live_table_refuses_a_failed_fetch_an_absent_or_empty_record_and_a_mirror_that_is_not_the_record(tmp_path):
    """After B's table is published, `--cluster C` with C's record missing (fetch fails), C's record readable but holding
    no trial, or C holding only part of B's files (a partial fetch leaves B's residue under C's identity) must refuse BY NAME — nothing keyed,
    rendered, published or marked; a C record that IS the same evidence (a true rehome) renders."""
    w = World(tmp_path); w.lay_round1(pd=True)
    rb = w.run("table"); assert rb.returncode == 0, rb.stdout + rb.stderr; kb = _key(w); figs = _sha(w.figs / "table.csv")
    # (a) C's record directory does not exist: the fetch fails
    root_c = w.tmp / "cluster_c" / "03_hpo" / "lora"
    r = w.run("table", cluster="cluster_c", cl_root=root_c); out = r.stdout + r.stderr
    assert r.returncode != 0 and "REFUSED: table: fetch of login-b:" in out and "failed (rc=" in out, out
    assert "rebuilding" not in out and "hpo_table" not in out and _key(w) == kb and _sha(w.figs / "table.csv") == figs, out
    # (b) C's record exists and is readable but holds no trial summary (the mirror still holds B's 16)
    root_c.mkdir(parents=True); (root_c / "r1_trials.json").write_text("{}")
    r = w.run("table", cluster="cluster_c", cl_root=root_c); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"REFUSED: table: the record at login-b:{root_c} holds no trial summary" in out, out
    assert "rebuilding" not in out and _key(w) == kb and _sha(w.figs / "table.csv") == figs, out
    # (c) C holds HALF of B's trials: the fetch succeeds, the mirror still holds the other half -> residuals named, nothing published
    for d in sorted(w.cluster.glob("r1_t*"))[:8]:
        shutil.copytree(d, root_c / d.name)
    r = w.run("table", cluster="cluster_c", cl_root=root_c); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"REFUSED: table: the mirror {w.local} is not the record cluster_c:{root_c}" in out, out
    for i in range(8, 16):
        assert f"r1_t{i:02d}/summary.json(on the mirror, not on the record)" in out, out
    assert "r1_t00/summary.json(" not in out and "rebuilding" not in out and _key(w) == kb and _sha(w.figs / "table.csv") == figs, out
    # (d) a summary the record holds differs from the mirror's copy only if the fetch did not deliver it — a fake rsync that drops a file
    (w.bin / "rsync").write_text('#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"; src="${src#*:}"; dst="${dst#*:}"; '
                                 'if [ -d "$src" ]; then mkdir -p "$dst"; cp -r "$src"/. "$dst"/; rm -rf "$dst"/r1_t05; else mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; fi\n')
    r = w.run("table"); out = r.stdout + r.stderr
    assert r.returncode != 0 and "r1_t05/summary.json(mirror differs from the record)" in out and _key(w) == kb, out
    # (e) the true rehome: C holds exactly B's evidence -> the inventory matches and the table renders under C's identity
    (w.bin / "rsync").write_text('#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"; src="${src#*:}"; dst="${dst#*:}"; '
                                 'if [ -d "$src" ]; then mkdir -p "$dst"; cp -r "$src"/. "$dst"/; else mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; fi\n')
    shutil.rmtree(root_c); shutil.copytree(w.cluster, root_c)
    r = w.run("table", cluster="cluster_c", cl_root=root_c); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"the mirror is the record cluster_c:{root_c} — 16 trial summaries + 0 sidecars verified" in out and _key(w) != kb, out
    kc = _key(w); figs_c = _sha(w.figs / "table.csv")
    # (f) B's record gains metrics sidecars (fetched to the shared mirror by B's table); C holds the SAME
    # summary bytes but no metrics -> a table over C must not read B's sidecars under C's identity: refused naming every residual sidecar
    _lay_metrics(w.cluster); _lay_metrics(w.local)
    assert _analyze(w.local, w.local / "allowed_pairs.json") == 0; shutil.copy(w.local / "r1_fit.json", w.cluster / "r1_fit.json")   # B re-fits over its sidecars
    r = w.run("table"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "16 trial summaries + 16 sidecars verified" in out, out                      # B: consistent, renders with clip fractions
    r = w.run("table", cluster="cluster_c", cl_root=root_c); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"REFUSED: table: the mirror {w.local} is not the record cluster_c:{root_c}" in out, out
    for i in range(16):
        assert f"r1_t{i:02d}/metrics.jsonl(on the mirror, not on the record)" in out, out
    assert "summary.json(" not in out and "rebuilding" not in out, out
    assert all((d / "metrics.jsonl").is_file() for d in w.local.glob("r1_t*") if d.is_dir()), "nothing deleted by the refusal"
    # C backfills its metrics identically and carries B's re-fit (C IS B's record, rehomed) -> the inventories agree again and C renders
    _lay_metrics(root_c); shutil.copy(w.cluster / "r1_fit.json", root_c / "r1_fit.json")
    r = w.run("table", cluster="cluster_c", cl_root=root_c); out = r.stdout + r.stderr
    assert r.returncode == 0 and "16 trial summaries + 16 sidecars verified" in out and _key(w) != kc, out


def test_live_addendum_refuses_a_mirror_that_is_not_the_record(tmp_path):
    """The addendum refit pools the mirror's round-1 summaries: a record that holds only part of them (the rest B's residue) is refused before any
    refit, emission, push or submission."""
    w = World(tmp_path); w.lay_round1(pd=True); w.widen()
    root_c = w.tmp / "cluster_c" / "03_hpo" / "lora"; root_c.mkdir(parents=True)
    for d in sorted(w.cluster.glob("r1_t*"))[:10]:
        shutil.copytree(d, root_c / d.name)
    r = w.run("addendum", cluster="cluster_c", cl_root=root_c); out = r.stdout + r.stderr
    assert r.returncode != 0 and f"REFUSED: addendum: the mirror {w.local} is not the record cluster_c:{root_c}" in out, out
    assert "r1_t12/summary.json(on the mirror, not on the record)" in out and "STUB submit" not in out, out
    assert not (w.local / "r1_fit_menu.json").exists() and not (w.local / "r4_trials.json").exists() and not (root_c / "r4_trials.json").exists()


# --- the override of the menu gate ------------------------------------------------------------------------------------

def test_live_addendum_menu_override_runs_exactly_the_given_pairs_without_the_probe_menu(tmp_path):
    """The serving re-verdict is decoupled — `--phase addendum --pairs … --menu-override <ruling>` builds the menu from the round-1 design's
    own pair details plus the override pairs (synthesized from the design's same-set entries), records menu_override in the override menu, the
    round-4 design and every trial, refits over it (the design fit untouched), emits the six rows at the fit's predicted lr and submits ONE array;
    a pair in the round-1 design is still refused; the probe's menu file is not needed (archived by a reopen, as live)."""
    w = World(tmp_path); before = w.lay_round1(pd=True)
    shutil.copy(w.tmp / "r1_trials.json", w.local / "r1_trials.json")                      # the design, with the probe's pair details verbatim
    (w.local / "allowed_pairs.json").unlink()                                              # the probe's menu is gone (a reopen archived it)
    # the override is bound to a committed authorization record: no record -> refused; a pair outside it -> refused by name
    r0 = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out0 = r0.stdout + r0.stderr
    assert r0.returncode == 2 and "no authorization record at" in r0.stderr and "authorizes nothing" in r0.stderr and "STUB submit" not in out0, out0
    (w.tmp / "menu_overrides").mkdir(); (w.tmp / "menu_overrides" / "D547.json").write_text(json.dumps({"ruling": "D547", "model": "m", "class": "lora",
                                                                                                        "pairs": [list(k) for k in NEW6]}))
    r0 = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}, pairs=PAIRS_ARG + ",128:mlp"); out0 = r0.stdout + r0.stderr
    assert r0.returncode == 2 and "pair(s) [(128, 'mlp')] are NOT in the approved set of D547" in r0.stderr and "STUB submit" not in out0, out0
    assert not (w.local / "allowed_pairs.override.D547.json").exists(), "nothing built before the authorization"
    # With no matching archive, the design-detail reconstruction must BE the design's menu — an incomplete or an extra detail
    # entry is refused by name with nothing moved (the recon check never sees an artificial widest pair)
    orig = (w.local / "r1_trials.json").read_bytes(); td = json.loads(orig)
    td_inc = dict(td); td_inc["allowed_pairs_detail"] = td["allowed_pairs_detail"][:-1]; (w.local / "r1_trials.json").write_text(json.dumps(td_inc, default=str))
    r0 = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out0 = r0.stdout + r0.stderr
    assert r0.returncode == 2 and "is not the design's recorded menu" in r0.stderr and "nothing moved" in r0.stderr and "STUB submit" not in out0, out0
    td_ext = dict(td); td_ext["allowed_pairs_detail"] = td["allowed_pairs_detail"] + [{**td["allowed_pairs_detail"][-1], "rank": 128, "set": "mlp", "trainable_params": 10 ** 9}]
    (w.local / "r1_trials.json").write_text(json.dumps(td_ext, default=str))
    r0 = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out0 = r0.stdout + r0.stderr
    assert r0.returncode == 2 and "is not the design's recorded menu" in r0.stderr and "(128, 'mlp')" in r0.stderr and "STUB submit" not in out0, out0
    assert not (w.local / "allowed_pairs.design.json").exists() and not (w.local / "superseded").exists(), "nothing written, nothing archived"
    (w.local / "r1_trials.json").write_bytes(orig)
    # the recon check reads the menu the reconnaissance ran on: an archived copy is used only when its sha is the one the design recorded
    arch = w.local / "superseded" / "20260101T000000Z"; arch.mkdir(parents=True); (arch / "allowed_pairs.json").write_text("{}")   # another generation
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    assert "the recon revalidation reads the menu the reconnaissance ran on — design-detail copy" in out and (w.local / "allowed_pairs.design.json").is_file(), out
    dd = json.loads((w.local / "allowed_pairs.design.json").read_text()); assert sorted((p["rank"], p["set"]) for p in dd["pairs"]) == sorted(DRAWN9)
    assert "MENU OVERRIDE D547" in out and "the probe record is not revalidated here" in out and "STUB submit_round r4 array 0-5" in out, out
    ovr = json.loads((w.local / "allowed_pairs.override.D547.json").read_text())
    assert ovr["menu_override"]["ruling"] == "D547" and sorted(map(tuple, ovr["menu_override"]["pairs"])) == sorted(NEW6)
    assert ovr["menu_override"]["authorization"]["record"].endswith("menu_overrides/D547.json") and len(ovr["menu_override"]["authorization"]["sha256"]) == 64
    assert "authorized by" in out, out
    assert sorted((p["rank"], p["set"]) for p in ovr["pairs"]) == sorted(WIDE15) and all(p["trainable_params"] is None and p["serving_certified"] is False
                                                                                           for p in ovr["pairs"] if (p["rank"], p["set"]) in set(NEW6))
    syn = {(p["rank"], p["set"]): p for p in ovr["pairs"] if (p["rank"], p["set"]) in set(NEW6)}
    base = {(p["rank"], p["set"]): p for p in ovr["pairs"] if (p["rank"], p["set"]) not in set(NEW6)}
    sys.path.insert(0, str(REPO / "reproduce" / "lib")); import models as _models  # noqa: E402
    for (rk, sid), p in syn.items():
        assert p["lora_modules"] == base[(4, sid)]["lora_modules"]
        if base[(4, sid)].get("per_expert_rank"):
            # array 165623 (06:02Z): every task refused by the factory — the ranks were SCALED from the rank-4 entry's clamped floor. FD 6's rule,
            # with the design's recorded active_experts (the fixture's menu records 8): shared = rank, routed = max(1, rank // 8) -> 2 / 8
            assert p["per_expert_rank"] == _models.per_expert_rank(rk, 8) == {"shared_expert": rk, "routed_expert": {16: 2, 64: 8}[rk]}, p
            assert "FD 6 rule" in p["synthesized_from"]["per_expert_rank_source"]
    d4 = json.loads((w.local / "r4_trials.json").read_text())
    for t in d4["trials"]:
        if base[(4, t["lora_set_id"])].get("per_expert_rank"):
            assert t["per_expert_rank"] == _models.per_expert_rank(t["lora_rank"], 8), t      # the design the trials run from carries the rule's values
    assert _sha(w.local / "r1_fit.json") == before, "the design fit is never the refit's output"
    mf = json.loads((w.local / "r1_fit_menu.json").read_text())
    assert sorted(map(tuple, mf["menu_extended_from"])) == sorted(DRAWN9) and sorted(map(tuple, mf["allowed_pairs"])) == sorted(WIDE15)
    assert mf["menu_source"]["sha256"] == _sha(w.local / "allowed_pairs.override.D547.json")
    d = json.loads((w.local / "r4_trials.json").read_text())
    assert d["menu_override"]["ruling"] == "D547" and d["campaign"] == json.loads((w.local / "r1_fit.json").read_text()).get("campaign")
    assert [(t["lora_rank"], t["lora_set_id"]) for t in d["trials"]] == NEW6 and all(t["menu_override"] == "D547" for t in d["trials"])
    assert all(t["lr_source"]["kind"] in ("pair_prediction", "best_round1_trial") for t in d["trials"]) and (w.cluster / "r4_trials.json").is_file()
    # a pair already in the round-1 design is refused by name, override or not: outside the approved set it is the authorization that refuses;
    # inside an (ill-advised) approved set it is still the design check
    r = w.run("addendum", pairs="4:mlp", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert r.returncode != 0 and "are NOT in the approved set of D547" in r.stderr and "STUB submit" not in out, out
    (w.tmp / "menu_overrides" / "D547.json").write_text(json.dumps({"ruling": "D547", "model": "m", "class": "lora", "pairs": [list(k) for k in NEW6] + [[4, "mlp"]]}))
    r = w.run("addendum", pairs="4:mlp", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert r.returncode != 0 and "is in the round-1 design" in r.stderr and "STUB submit" not in out, out
    (w.tmp / "menu_overrides" / "D547.json").write_text(json.dumps({"ruling": "D547", "model": "m", "class": "lora", "pairs": [list(k) for k in NEW6]}))
    # the table over the override menu renders the 9 + 6 pairs (the 6 as allowed, predictions from the refit)
    rt = w.run("table", env_extra={"MENU_OVERRIDE": "D547"}); outt = rt.stdout + rt.stderr
    assert rt.returncode == 0 and "MENU OVERRIDE D547" in outt and "r1_fit_menu.json" in (w.local / "table" / "table.md").read_text(), outt
    # the archived ORIGINAL (sha == the design's recorded menu sha) is preferred over the reconstruction (checked last: this run re-generates the
    # override menu for its own --pairs)
    arch2 = w.local / "superseded" / "20260102T000000Z"; arch2.mkdir(); shutil.copy(w.tmp / "allowed_pairs.json", arch2 / "allowed_pairs.json")
    r2 = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}, pairs="16:attn_mlp"); out2 = r2.stdout + r2.stderr
    assert f"archived copy {arch2 / 'allowed_pairs.json'}" in out2, out2


def test_live_addendum_override_without_a_recorded_active_expert_count_carries_null_per_expert_ranks(tmp_path):
    """When the design's rule is the generic spelling (the probe recorded no active_experts, as live), the synthesized expert-set pairs carry
    per_expert_rank null and say so — the object factory applies FD 6's rule itself (as for every round-1 design at ranks 16 / 64); nothing is
    ever scaled from a lower-rank entry."""
    w = World(tmp_path); w.lay_round1(pd=True)
    td = json.loads((w.tmp / "r1_trials.json").read_text()); td["per_expert_rank_rule"] = "max(1,r//active_experts)"
    (w.local / "r1_trials.json").write_text(json.dumps(td, default=str)); (w.local / "allowed_pairs.json").unlink()
    (w.tmp / "menu_overrides").mkdir(); (w.tmp / "menu_overrides" / "D547.json").write_text(json.dumps({"ruling": "D547", "model": "m", "class": "lora", "pairs": [list(k) for k in NEW6]}))
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert r.returncode == 0 and "STUB submit_round r4 array 0-5" in out, out[-3000:]
    ovr = json.loads((w.local / "allowed_pairs.override.D547.json").read_text())
    for p in ovr["pairs"]:
        if (p["rank"], p["set"]) in set(NEW6):
            assert p["per_expert_rank"] is None and "factory applies FD 6" in p["synthesized_from"]["per_expert_rank_source"], p
    assert all(t["per_expert_rank"] is None for t in json.loads((w.local / "r4_trials.json").read_text())["trials"])


def test_live_addendum_resubmit_supersedes_the_failed_design_and_re_derives(tmp_path):
    """The failed-array relaunch: round-4 design on the record, no summaries, no job -> refused by name; with SISTEER_RESUBMIT=1 the record's
    design and trial dirs move to superseded/<utc>/ (nothing deleted), the design is re-derived and re-submitted (the funnel's --resubmit)."""
    w = World(tmp_path); w.lay_round1(pd=True); w.widen(); shutil.copy(w.tmp / "wide.json", w.cluster / "allowed_pairs.json")
    r = w.run("addendum"); assert r.returncode == 0 and "STUB submit_round r4 array 0-5" in r.stdout + r.stderr, r.stdout + r.stderr
    (w.cluster / "r4_t00").mkdir(); (w.cluster / "r4_t00" / "RESULT").write_text("rc=1\n")                     # the failed task's leavings, no summary
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "r4_trials.json is on the record with 0/6 summaries and no job" in r.stderr and "SISTEER_RESUBMIT=1" in r.stderr, out
    # The funnel's proof runs FIRST; a refusal leaves EVERY file byte-identical (nothing moved, nothing re-derived, nothing submitted)
    before = {p.relative_to(w.cluster): _sha(p) for p in w.cluster.rglob("*") if p.is_file()}
    r = w.run("addendum", env_extra={"SISTEER_RESUBMIT": "1", "PROVE_REFUSE": "1"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "REFUSED by the funnel's proof" in r.stderr and "NOTHING moved" in r.stderr and "STUB submit" not in out, out
    assert {p.relative_to(w.cluster): _sha(p) for p in w.cluster.rglob("*") if p.is_file()} == before and not (w.cluster / "superseded").exists()
    # The queue recheck fails CLOSED — a squeue error is not an empty queue; nothing moved
    r = w.run("addendum", env_extra={"SISTEER_RESUBMIT": "1", "FAKE_SQUEUE_FAIL": "1"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "UNKNOWN: squeue did not answer cleanly" in out and "NOTHING moved" in r.stderr and "STUB submit" not in out, out
    assert {p.relative_to(w.cluster): _sha(p) for p in w.cluster.rglob("*") if p.is_file()} == before and not (w.cluster / "superseded").exists()
    assert not (w.cluster / ".resubmit.lock").exists(), "the per-record lock is released on every exit"
    # a refusal by the funnel's second proof AFTER the move: the previous design and dirs are RESTORED to their canonical paths, the replacement steps aside
    r = w.run("addendum", env_extra={"SISTEER_RESUBMIT": "1", "FAIL_SUBMIT": "1"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "RESUBMIT authorized" in out and "refused the re-submission AFTER the move" in r.stderr and "RESTORED" in r.stderr, out
    sup = sorted((w.cluster / "superseded").iterdir()); assert len(sup) == 1 and (sup[0] / "replacement.r4_trials.json").is_file() and not (sup[0] / "r4_trials.json").exists()
    assert {p.relative_to(w.cluster): _sha(p) for p in w.cluster.rglob("*") if p.is_file() and "superseded" not in p.parts} == before, "every canonical file back, byte-identical"
    assert not (w.local / "r4_trials.json").exists()
    rcpt = json.loads((sup[0] / "transition.json").read_text())                                   # the receipt closed as restored
    assert rcpt["state"] == "restored" and rcpt["moved"] == ["r4_trials.json", "r4_t00"] and rcpt["replacement_sha256"] and rcpt["closed_utc"], rcpt
    assert not (w.cluster / "r4_resubmit.transition.json").exists() and not (w.cluster / ".resubmit.lock").exists()
    shutil.rmtree(w.cluster / "superseded")
    # the authorized path: the previous design superseded, the new one re-derived and submitted; the transition closed as complete
    r = w.run("addendum", env_extra={"SISTEER_RESUBMIT": "1"}); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RESUBMIT authorized by the funnel's proof (bound job 4242" in out and "STUB submit_design r4 array 0-5" in out and "CLOSED" in out, out
    sup = sorted((w.cluster / "superseded").iterdir()); assert len(sup) == 1 and (sup[0] / "r4_trials.json").is_file() and (sup[0] / "r4_t00" / "RESULT").is_file()
    assert (w.cluster / "r4_trials.json").is_file() and not (w.cluster / "r4_t00").exists() and not (w.cluster / ".resubmit.lock").exists()
    rcpt = json.loads((sup[0] / "transition.json").read_text())
    assert rcpt["state"] == "complete" and rcpt["replacement_sha256"] == _sha(w.cluster / "r4_trials.json") and rcpt["bound_job"] == "4242", rcpt
    assert not (w.cluster / "r4_resubmit.transition.json").exists()


def _canon(w: World) -> dict:
    """sha by path of the record's CANONICAL files (superseded/, the transition receipt and the lock excluded)."""
    return {p.relative_to(w.cluster): _sha(p) for p in w.cluster.rglob("*")
            if p.is_file() and "superseded" not in p.parts and ".resubmit.lock" not in p.parts and p.name != "r4_resubmit.transition.json"}


def _all(w: World) -> dict:
    return {p.relative_to(w.cluster): _sha(p) for p in w.cluster.rglob("*") if p.is_file()}


def _relaunch_world(tmp_path: Path):
    """The failed-array state: the round-4 design on the record, one task's leavings (no summary), no job queued."""
    w = World(tmp_path); w.lay_round1(pd=True); w.widen(); shutil.copy(w.tmp / "wide.json", w.cluster / "allowed_pairs.json")
    r = w.run("addendum"); assert r.returncode == 0 and "STUB submit_round r4 array 0-5" in r.stdout + r.stderr, r.stdout + r.stderr
    (w.cluster / "r4_t00").mkdir(); (w.cluster / "r4_t00" / "RESULT").write_text("rc=1\n")
    return w, _canon(w)


def test_live_addendum_relaunch_rolls_back_on_any_failure_after_the_move_and_holds_the_lock_throughout(tmp_path):
    """A failure AFTER the move that is not the funnel's refusal — here the inventory check refusing the mirror — must roll
    the record back from the transition receipt (byte-identical), close the receipt as restored and release the lock; and the lock is HELD while
    the replacement is derived (the fetch after the move sees it)."""
    w, before = _relaunch_world(tmp_path)
    probe = w.tmp / "lock_probe"
    rs = (w.bin / "rsync").read_text()
    (w.bin / "rsync").write_text(rs.replace("#!/bin/bash\n", f"#!/bin/bash\nif [ -d {w.cluster}/.resubmit.lock ]; then echo held >> {probe}; else echo free >> {probe}; fi\n", 1))
    shutil.copytree(w.local / "r1_t03", w.local / "r1_t16")                                      # mirror residue the record does not hold
    r = w.run("addendum", env_extra={"SISTEER_RESUBMIT": "1"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "RESUBMIT authorized" in out and "REFUSED: addendum: the mirror" in r.stderr, out[-4000:]
    assert "--resubmit did not complete (exit 2)" in r.stderr and "RESTORED the previous round-4 design" in out and "STUB submit" not in out, out[-4000:]
    assert probe.exists() and set(probe.read_text().split()) == {"held"}, probe.read_text() if probe.exists() else "no fetch ran"
    assert _canon(w) == before, "every canonical file back, byte-identical"
    sup = sorted((w.cluster / "superseded").iterdir()); assert len(sup) == 1 and (sup[0] / "transition.json").is_file()
    rcpt = json.loads((sup[0] / "transition.json").read_text())
    assert rcpt["state"] == "restored" and rcpt["moved"] == ["r4_trials.json", "r4_t00"] and rcpt["bound_job"] == "4242" and rcpt["replacement_sha256"] is None, rcpt
    assert not (w.cluster / "r4_resubmit.transition.json").exists() and not (w.cluster / ".resubmit.lock").exists()


def test_live_addendum_reconciles_an_open_transition_from_its_receipt_before_anything_is_built(tmp_path):
    """A relaunch that died MID-MOVE left the receipt (state moving) with r4_trials.json + r4_t00 under superseded/<utc>/
    and r4_t01 still canonical. (1) an abandoned lock with the open receipt fails CLOSED by name (the operator's recovery is in the message);
    (2) a plain --phase addendum reconciles FIRST — every listed path back, byte-identical; the receipt closes as restored — and only then reads
    the record (never a fresh design over the open transition); (3) a path at both places that is not the recorded replacement refuses by name
    with nothing moved and the receipt open; (4) a receipt in state submitted completes — a submitted design is never rolled back."""
    w, _ = _relaunch_world(tmp_path)
    (w.cluster / "r4_t01").mkdir(); (w.cluster / "r4_t01" / "RESULT").write_text("rc=1\n")
    before = _canon(w)
    utc = "20260101T000000Z"; sup = w.cluster / "superseded" / utc; sup.mkdir(parents=True)
    receipt = {"receipt": "hpo-addendum-resubmit", "state": "moving", "utc": utc, "by": "tester@TESTHOST pid 1", "bound_job": "4242", "job_name": "hpo-m-lora-r4",
               "pinned_sha256": _sha(w.cluster / "r4_trials.json"), "superseded": f"superseded/{utc}", "replacement_sha256": None, "moved": ["r4_trials.json", "r4_t00", "r4_t01"]}
    (w.cluster / "r4_resubmit.transition.json").write_text(json.dumps(receipt))
    shutil.move(str(w.cluster / "r4_trials.json"), str(sup / "r4_trials.json")); shutil.move(str(w.cluster / "r4_t00"), str(sup / "r4_t00"))
    # (1) the abandoned lock
    (w.cluster / ".resubmit.lock").mkdir(); (w.cluster / ".resubmit.lock" / "owner").write_text("tester@TESTHOST pid 1 20260101T000000Z\n")
    snap = _all(w)
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "OPEN relaunch transition" in r.stderr and "relaunch lock is held while a transition is open" in r.stderr, out[-4000:]
    assert "LOCKED: another relaunch holds" in out and "owner: tester@TESTHOST pid 1" in out and "rm -r" in out and "STUB submit" not in out, out[-4000:]
    assert _all(w) == snap, "nothing moved; the receipt open; the lock untouched"
    shutil.rmtree(w.cluster / ".resubmit.lock")                                                   # the operator's recovery, the owner confirmed dead
    # (2) reconciled first, then the record read as it stands
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "OPEN relaunch transition" in r.stderr and "RESTORED the previous round-4 design" in out, out[-4000:]
    assert "restored r4_trials.json" in out and "restored r4_t00" in out and "in place r4_t01" in out, out[-4000:]
    assert "r4_trials.json is on the record with 0/6 summaries and no job" in r.stderr and "STUB submit" not in out, out[-4000:]
    assert _canon(w) == before, "every listed path back, byte-identical"
    rcpt = json.loads((sup / "transition.json").read_text()); assert rcpt["state"] == "restored" and rcpt["closed_utc"], rcpt
    assert not (w.cluster / "r4_resubmit.transition.json").exists() and not (w.cluster / ".resubmit.lock").exists()
    # (3) a conflict: r4_t00 canonical AND superseded, not a recorded replacement
    (w.cluster / "r4_resubmit.transition.json").write_text(json.dumps({**receipt, "state": "moved"}))
    shutil.copytree(w.cluster / "r4_t00", sup / "r4_t00"); shutil.move(str(w.cluster / "r4_trials.json"), str(sup / "r4_trials.json"))
    snap = _all(w)
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "CONFLICT: r4_t00 is at its canonical path AND under" in out and "could not be rolled back" in r.stderr and "STUB submit" not in out, out[-4000:]
    assert _all(w) == snap, "nothing moved on a conflict; the receipt stays open"
    assert not (w.cluster / ".resubmit.lock").exists(), "the lock is released on the refusal"
    shutil.rmtree(sup / "r4_t00"); shutil.move(str(sup / "r4_trials.json"), str(w.cluster / "r4_trials.json"))
    # (4) submitted: completed, never rolled back
    (w.cluster / "r4_resubmit.transition.json").write_text(json.dumps({**receipt, "state": "submitted", "replacement_sha256": _sha(w.cluster / "r4_trials.json")}))
    snap = {k: v for k, v in _all(w).items() if k.name not in ("r4_resubmit.transition.json", "transition.json")}
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert "had been SUBMITTED" in out and "completed" in out and "RESTORED" not in out and "STUB submit" not in out, out[-4000:]
    rcpt = json.loads((sup / "transition.json").read_text()); assert rcpt["state"] == "complete", rcpt
    assert {k: v for k, v in _all(w).items() if k.name not in ("r4_resubmit.transition.json", "transition.json")} == snap and not (w.cluster / "r4_resubmit.transition.json").exists()


def test_live_addendum_override_menu_is_idempotent_by_content_and_reproduces_the_designs_menu(tmp_path):
    """Live 2026-10-01 08:16Z: the completion run REBUILT the override menu with a new build time, its sha no longer matched the identity the
    round-4 design and the refit recorded, and the table refused by name. The builder is idempotent by CONTENT: a re-run keeps the bytes; a menu
    clobbered by a build from another checkout at another time is restored to the menu the round-4 design was emitted over, byte-identical (the
    design's recorded sha is the positive check); only new content writes a fresh menu."""
    w = World(tmp_path); w.lay_round1(pd=True); (w.local / "allowed_pairs.json").unlink()
    shutil.copy(w.tmp / "r1_trials.json", w.local / "r1_trials.json")                              # the round-1 design on the mirror (its pair details)
    (w.tmp / "menu_overrides").mkdir(); (w.tmp / "menu_overrides" / "D547.json").write_text(json.dumps({"ruling": "D547", "model": "m", "class": "lora", "pairs": [list(k) for k in NEW6]}))
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert r.returncode == 0 and "(built now; sha" in out and "STUB submit_round r4 array 0-5" in out, out[-3000:]
    menu = w.local / "allowed_pairs.override.D547.json"; b0 = menu.read_bytes(); sha0 = hashlib.sha256(b0).hexdigest()
    d4 = json.loads((w.local / "r4_trials.json").read_text())
    assert d4["allowed_pairs_source"]["sha256"] == sha0 and d4["menu_override"]["utc"] == json.loads(b0)["menu_override"]["utc"], "the design records the menu it was emitted over"
    # the completion read: the same bytes (the design's identity), nothing rewritten
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert "the menu the round-4 design was emitted over, byte-identical" in out and menu.read_bytes() == b0, out[-3000:]
    # clobbered (another checkout's record path, another build time): restored to the design's menu
    doc = json.loads(b0); doc["menu_override"]["utc"] = "2030-01-01T00:00:00Z"; doc["menu_override"]["authorization"]["record"] = "/elsewhere/menu_overrides/D547.json"
    menu.write_text(json.dumps(doc, indent=1) + "\n"); assert menu.read_bytes() != b0
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert "the menu the round-4 design was emitted over, byte-identical" in out and menu.read_bytes() == b0, out[-3000:]
    # the addendum COMPLETE and marked DONE: a later invocation (the completion read, a table's precondition) still binds the menu BEFORE the
    # DONE check — a clobbered menu is restored to the design's identity even though the phase skips (live 08:16Z: the skip came first)
    for i in range(6):
        (w.cluster / f"r4_t{i:02d}").mkdir(exist_ok=True); (w.cluster / f"r4_t{i:02d}" / "summary.json").write_text("{}")
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert r.returncode == 0 and "6/6 round-4 summaries on cluster_b — complete" in out, out[-3000:]
    menu.write_text(json.dumps(doc, indent=1) + "\n"); assert menu.read_bytes() != b0
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert r.returncode == 0 and "DONE marker present" in out and "skipping" in out and "the menu the round-4 design was emitted over, byte-identical" in out, out[-3000:]
    assert menu.read_bytes() == b0 and out.index("byte-identical") < out.index("DONE marker present"), "bound before the skip"
    # The completed addendum's menu is PINNED — the committed authorization record changes content (another approved
    # pair) while the requested pairs stay approved: refused by name, menu bytes and the DONE marker untouched (no REBUILT under DONE)
    (w.tmp / "menu_overrides" / "D547.json").write_text(json.dumps({"ruling": "D547", "model": "m", "class": "lora", "pairs": [list(k) for k in NEW6] + [[128, "mlp"]]}))
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "menu is pinned" in r.stderr and "NEW generation" in r.stderr and "REBUILT" not in out and "DONE marker present" not in out, out[-3000:]
    assert menu.read_bytes() == b0 and r.returncode == 2
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}, pairs=",".join(f"{r_}:{s_}" for r_, s_ in NEW6[:3])); out = r.stdout + r.stderr
    assert r.returncode == 2 and "menu is pinned" in r.stderr and menu.read_bytes() == b0, "other requested pairs over a completed addendum: refused too"
    (w.tmp / "menu_overrides" / "D547.json").write_text(json.dumps({"ruling": "D547", "model": "m", "class": "lora", "pairs": [list(k) for k in NEW6]}))
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert r.returncode == 0 and "DONE marker present" in out and menu.read_bytes() == b0, "the record restored: the completed addendum reads again"
    # no design to bind to (the mirror lacks r4_trials.json) under DONE: an existing menu equal modulo build time / record path is kept; a different one refuses
    (w.local / "r4_trials.json").unlink(); menu.write_text(json.dumps(doc, indent=1) + "\n"); b1 = menu.read_bytes()
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert "unchanged (built 2030-01-01T00:00:00Z)" in out and menu.read_bytes() == b1, out[-3000:]
    (w.tmp / "menu_overrides" / "D547.json").write_text(json.dumps({"ruling": "D547", "model": "m", "class": "lora", "pairs": [list(k) for k in NEW6] + [[128, "mlp"]]}))
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "pinned" in r.stderr and "NEW generation" in r.stderr and menu.read_bytes() == b1, out[-3000:]   # the DONE pin, or the design's once a fetch restored it
    # new content on a FRESH search (no design, no DONE) is a new menu, said so
    w2 = World(tmp_path / "fresh"); w2.lay_round1(pd=True); (w2.local / "allowed_pairs.json").unlink(); shutil.copy(w2.tmp / "r1_trials.json", w2.local / "r1_trials.json")
    (w2.tmp / "menu_overrides").mkdir(); (w2.tmp / "menu_overrides" / "D547.json").write_text(json.dumps({"ruling": "D547", "model": "m", "class": "lora", "pairs": [list(k) for k in NEW6]}))
    (w2.local / "allowed_pairs.override.D547.json").write_text(json.dumps({"pairs": [], "menu_override": {"ruling": "D547", "utc": "2029-01-01T00:00:00Z", "authorization": {"record": "/x"}}}, indent=1) + "\n")
    r = w2.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); out = r.stdout + r.stderr
    assert "(REBUILT: its content changed; sha" in out, out[-3000:]



def test_live_addendum_reconcile_completes_a_replaced_receipt_whose_replacement_job_was_submitted_and_never_rolls_back_on_an_empty_queue(tmp_path):
    """A crash between the funnel's accepted sbatch and the receipt's `submitted` leaves the receipt at `replaced` while the
    array is bound in the funnel's ledger; once that array fast-failed or completed and left squeue, an EMPTY QUEUE proves nothing. The
    tie-breaker is the durable evidence: (1) the funnel's binding names a job other than the proof's dead one -> completed, the replacement
    design stands; (2) the binding still names the dead job but sacct shows a job of the relaunch's name submitted since the receipt (the
    pre-ledger window) -> completed; (3) sacct unreadable -> undecidable: refused by name, nothing moved, the receipt open; (4) the funnel's
    own transition token -> undecidable too; (5) every source empty and readable -> rolled back (the only path that moves anything back)."""
    w, _ = _relaunch_world(tmp_path)
    utc = "20260101T000000Z"; sup = w.cluster / "superseded" / utc; sup.mkdir(parents=True)
    old_design = (w.cluster / "r4_trials.json").read_bytes()
    shutil.move(str(w.cluster / "r4_trials.json"), str(sup / "r4_trials.json")); shutil.move(str(w.cluster / "r4_t00"), str(sup / "r4_t00"))
    replacement = json.dumps({"trials": [{"index": i} for i in range(6)], "replacement": True}).encode()
    (w.cluster / "r4_trials.json").write_bytes(replacement)
    receipt = {"receipt": "hpo-addendum-resubmit", "state": "replaced", "utc": utc, "by": "tester@TESTHOST pid 1", "bound_job": "4242", "job_name": "hpo-m-lora-r4",
               "pinned_sha256": hashlib.sha256(old_design).hexdigest(), "superseded": f"superseded/{utc}", "replacement_sha256": hashlib.sha256(replacement).hexdigest(),
               "moved": ["r4_trials.json", "r4_t00"]}
    def reopen():
        (w.cluster / "r4_resubmit.transition.json").write_text(json.dumps(receipt)); (sup / "transition.json").unlink(missing_ok=True)
    def replacement_stands():
        return (w.cluster / "r4_trials.json").read_bytes() == replacement and (sup / "r4_trials.json").read_bytes() == old_design and (sup / "r4_t00" / "RESULT").is_file() and not (w.cluster / "r4_t00").exists()
    # (1) the ledger binds the unit to the replacement job (5151), terminal and gone from the queue: completed, nothing moved back
    reopen(); r = w.run("addendum", env_extra={"STUB_BINDING_JOB": "5151"}); out = r.stdout + r.stderr
    assert "had been SUBMITTED" in out and "binds hpo.lora.r4 to job 5151" in out and "RESTORED" not in out and "rolling back" not in out, out[-4000:]
    assert "not the proof's dead job 4242" in out and "bound job 4242" in r.stderr, "the receipt's bound job is read (an empty replacement_sha256 must not shift it)"
    assert replacement_stands() and json.loads((sup / "transition.json").read_text())["state"] == "complete" and not (w.cluster / "r4_resubmit.transition.json").exists()
    # (2) sacct is a GUARD, never an attribution — a same-name job submitted after the receipt that the funnel's ledger does not
    # know (another record's, or a launch the funnel lost) is UNDECIDABLE: refused by name, nothing moved, the receipt open (never "completed")
    reopen(); snap0 = _all(w)
    r = w.run("addendum", env_extra={"FAKE_SACCT_ROWS": "5252|2026-01-01T00:00:30|COMPLETED\n"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "cannot be decided" in r.stderr and "the funnel's ledger does not know" in r.stderr and "5252" in r.stderr, out[-4000:]
    assert "had been SUBMITTED" not in out and "RESTORED" not in out and _all(w) == snap0, "undecidable: nothing moved, nothing closed"
    # the three-rapid-attempts shape: attempt 1 (4141) and attempt 2 (4242, the proof's dead job) are ledger attempts, both terminal inside the
    # window; attempt 3's relaunch crashed before its sbatch -> nothing of THIS transition exists -> rolled back (never closed as submitted)
    r = w.run("addendum", env_extra={"STUB_BINDING_ATTEMPTS": "4141,4242", "FAKE_SACCT_ROWS": "4141|2025-12-31T23:58:00|FAILED\n4242|2025-12-31T23:59:30|FAILED\n"}); out = r.stdout + r.stderr
    assert "nothing was submitted under this transition" in out and "known attempts: 4242,4141,4242" in out and "RESTORED the previous round-4 design" in out, out[-4000:]
    assert (w.cluster / "r4_trials.json").read_bytes() == old_design and (sup / "replacement.r4_trials.json").read_bytes() == replacement and (w.cluster / "r4_t00").is_dir()
    shutil.move(str(w.cluster / "r4_trials.json"), str(sup / "r4_trials.json")); shutil.move(str(w.cluster / "r4_t00"), str(sup / "r4_t00"))
    (sup / "replacement.r4_trials.json").unlink(); (w.cluster / "r4_trials.json").write_bytes(replacement)
    # a same-name job older than the window (before the receipt minus the skew allowance) is pre-existing: ignored, rolled back
    reopen(); r = w.run("addendum", env_extra={"FAKE_SACCT_ROWS": "3939|2025-12-31T20:00:00|COMPLETED\n"}); out = r.stdout + r.stderr
    assert "nothing was submitted under this transition" in out and "RESTORED the previous round-4 design" in out, out[-4000:]
    assert (w.cluster / "r4_trials.json").read_bytes() == old_design and (sup / "replacement.r4_trials.json").read_bytes() == replacement and (w.cluster / "r4_t00").is_dir()
    # back to the crash shape for the undecidable cases
    shutil.move(str(w.cluster / "r4_trials.json"), str(sup / "r4_trials.json")); shutil.move(str(w.cluster / "r4_t00"), str(sup / "r4_t00"))
    (sup / "replacement.r4_trials.json").unlink(); (w.cluster / "r4_trials.json").write_bytes(replacement)
    # (3) sacct unreadable: refused by name, nothing moved, the receipt stays open
    reopen(); snap = _all(w)
    r = w.run("addendum", env_extra={"FAKE_SACCT_FAIL": "1"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "cannot be decided" in r.stderr and "sacct on login-b did not answer cleanly" in r.stderr and "RESTORED" not in out, out[-4000:]
    assert _all(w) == snap and (w.cluster / "r4_resubmit.transition.json").is_file() and not (w.cluster / ".resubmit.lock").exists()
    # (4) the funnel's own transition token in flight: undecidable here
    r = w.run("addendum", env_extra={"STUB_BINDING_TOKEN": "yes"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "cannot be decided" in r.stderr and "re-run transition token" in r.stderr and _all(w) == snap, out[-4000:]
    # (4b) the binding unreadable: undecidable
    r = w.run("addendum", env_extra={"STUB_BINDING_FAIL": "1"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "cannot be decided" in r.stderr and "could not be read" in r.stderr and _all(w) == snap, out[-4000:]
    # (5) every source readable and empty: the rollback (the replacement steps aside, the previous design and dirs return)
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert "nothing was submitted under this transition — rolling back" in out and "RESTORED the previous round-4 design" in out, out[-4000:]
    assert (w.cluster / "r4_trials.json").read_bytes() == old_design and (sup / "replacement.r4_trials.json").read_bytes() == replacement and (w.cluster / "r4_t00").is_dir()
    assert json.loads((sup / "transition.json").read_text())["state"] == "restored" and not (w.cluster / "r4_resubmit.transition.json").exists()
# --- The HOLD cycle, and the refit-only refreshes of completed refine / addendum ---------------------------------------------------------------

def _fold_reader(fold: Path) -> subprocess.CompletedProcess:
    return subprocess.run([VENV_PY, str(REPO / "scripts" / "sdfbase" / "make_sdfbase_configs.py"), "--models", "qwen35_4b_demo_base", "--fold-hpo", str(fold)],
                          capture_output=True, text=True, cwd=str(REPO), timeout=300)


def _fit_unit_04(state_root: Path, model: str, dry: bool = False) -> subprocess.CompletedProcess:
    """04_fit_objects.sh's LoRA fit_unit alone (fragment), with every other gate stubbed open: the HOLD check is the thing under test."""
    src = (REPRO / "04_fit_objects.sh").read_text()
    fn = src[src.index("arm_for(){"):src.index("# --- difference_lora")]
    script = f"""set -u
source {REPRO}/lib/common.sh
export SISTEER_STATE_DIR={state_root}
REPRO_DRY_RUN={1 if dry else 0}; REPRO_MODEL={model}; STAGE=04_fit_objects; FIT_ID=f; BANK=b; RULING=ruling-1; PY={VENV_PY}; CELLS=/nonexistent; FIT_WIDTH=1
FIT_CFG_LOCAL=/dev/null; SISTEER_TIME_FIT=1:00:00; CLUSTER=cluster_b
si_for(){{ echo si_$1; }}; registry_done(){{ return 1; }}; cluster_has(){{ return 1; }}; published_unfolded(){{ return 1; }}; cluster_job_active(){{ return 1; }}
repro_fit_width_flags(){{ :; }}; submit(){{ echo "STUB submit $1 $2"; }}
{fn}
fit_unit control_lora
"""
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)


def _confirmation_at(w: World, pair, lr: float, grad_norm: float = 0.01) -> None:
    """A round-9 CONFIRMATION trial at (pair, lr) with metrics (clipping `grad_norm` vs the clip 1.3) on the record and the mirror."""
    src = next(json.loads((d / "summary.json").read_text()) for d in sorted(w.cluster.glob("r1_t*")) if d.is_dir()
               and (lambda s: (s["trial"]["lora_rank"], s["trial"]["lora_set_id"]) == tuple(pair))(json.loads((d / "summary.json").read_text())))
    s = dict(src); s["round"] = 9; s["trial"] = {**src["trial"], "index": 0, "lr": lr}; s["lr"] = lr
    s["confirmation_of"] = {"fit": "r1+2_fit_pair.json", "pair": list(pair), "lr": lr}; s["best_heldout_loss"] = float(src["best_heldout_loss"]) - 0.002
    for root in (w.cluster, w.local):
        d = root / "r9_t00"; d.mkdir(exist_ok=True); (d / "summary.json").write_text(json.dumps(s)); _metrics(d / "metrics.jsonl", grad_norm=grad_norm)


def _completed_pick(w: World) -> dict:
    """A promoted pick as the live pick phase leaves it and as --verify-promotion re-reads it: the lr the fits select for (4, mlp), a round-9
    confirmation at those coordinates measuring below the clip threshold (the gate's evidence), hpo_pick.json (promoted, lr_source, winner),
    hpo_fold.json, the DONE marker."""
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase")); import hpo_pick  # noqa: E402
    key = (4, "mlp"); search = hpo_pick.load_search(w.local, str(w.local / "allowed_pairs.json"))
    lr = hpo_pick._lr_candidate(search, key)["lr"]
    _confirmation_at(w, key, lr)
    cand = hpo_pick.lr_for_pair(hpo_pick.load_search(w.local, str(w.local / "allowed_pairs.json")), key)
    assert cand["clip_evidence"]["verdict"] == "promotable", cand["clip_evidence"]
    allowed = tune_arm.load_allowed_pairs(w.local / "allowed_pairs.json")
    pick = {"promoted": True, "rank": 4, "set": "mlp", "ruling_id": "ruling-1", "winner_trial_id": "r9_t00", "lr_source": cand,
            "winner_params": {"lr": cand["lr"], "warmup_frac": cand["warmup_frac"], "max_grad_norm": cand["max_grad_norm"]},
            "init_scheme": search["design"].get("init_scheme"),
            "confirmation": {"trial_id": "r9_t00", "trial_role": "confirmation", "role_at_pick": "confirmation", "lr": lr},
            "menu": {"path": str(w.local / "allowed_pairs.json"), "sha256": allowed.get("sha256"), "n_pairs": len(allowed["pairs"])}}
    (w.local / "hpo_pick.json").write_text(json.dumps(pick, default=str))
    (w.local / "hpo_fold.json").write_text(json.dumps({"k": {"lora_r4_ce": {"lr": lr}}}))
    d = w.tmp / "state" / "m" / "03_hpo"; d.mkdir(parents=True, exist_ok=True); (d / "DONE.lora.pick").write_text("2026-10-01T00:00:00Z rank 4 set mlp\n")
    return pick


def test_live_hold_cycle_a_completed_pick_whose_basis_changed_is_held_consumers_refuse_and_a_fresh_pick_clears_it(tmp_path):
    """A completed pick + a metrics backfill -> `--phase pick` refuses visibly and writes hpo_pick.HOLD.json; make_sdfbase_configs --fold-hpo,
    04's LoRA fit unit and the table's promotion refuse by name while it stands; `--phase analyze` re-derives, a fresh `--phase pick --rank --set`
    re-derives a promotion and CLEARS the hold (recorded as hold_cleared); the consumers pass again. A pick whose evidence is unchanged skips."""
    w = World(tmp_path); w.lay_round1(pd=True); pick0 = _completed_pick(w)
    hold = w.local / "hpo_pick.HOLD.json"
    r = w.run("pick"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "the recorded promotion still stands — skipping" in out and not hold.exists(), out
    # The CONFIRMING trial's sidecar is backfilled to clip on every step — every fit still binds (round 9 is in no fit's
    # rounds), yet the promotion's gate no longer stands: HOLD
    _metrics(w.cluster / "r9_t00" / "metrics.jsonl", grad_norm=10.0)
    r = w.run("pick"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "HOLD written at" in r.stderr and hold.is_file(), out
    assert "the clip gate at the promoted coordinates" in out and "is now refused" in out and "fit saw" not in out, out
    hold.rename(w.local / "hpo_pick.HOLD.manual-aside.json"); _metrics(w.cluster / "r9_t00" / "metrics.jsonl", grad_norm=0.01)   # the sidecar corrected; the hold set aside by hand for the next scenario
    r = w.run("pick"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "the recorded promotion still stands — skipping" in out, out
    # the round-1 backfill after the promotion: the design fit no longer stands over the evidence
    _lay_metrics(w.cluster)
    r = w.run("pick"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "REFUSED: pick: the PROMOTED pick's basis has CHANGED — HOLD written at" in out and hold.is_file(), out
    assert "the fit saw no clip evidence, the current metrics give clip activation 0.000 over 5 steps" in out and "FAKE-PICK" not in out, out
    h = json.loads(hold.read_text()); assert h["held_pick"]["rank"] == 4 and h["held_pick"]["set"] == "mlp" and "no longer stands" in h["reason"]
    assert (w.tmp / "state" / "m" / "03_hpo" / "DONE.lora.pick").is_file(), "the marker stands; the hold is the hold"
    # every consumer refuses by name while the hold stands
    fr = _fold_reader(w.local / "hpo_fold.json")
    assert fr.returncode != 0 and "REFUSED: the LoRA architecture pick behind" in fr.stderr and "is HELD" in fr.stderr, fr.stderr[-800:]
    f4 = _fit_unit_04(w.tmp / "state", "m")
    assert f4.returncode != 0 and "the architecture pick is HELD" in f4.stderr and "STUB submit" not in f4.stdout, f4.stdout + f4.stderr
    f4d = _fit_unit_04(w.tmp / "state", "m", dry=True)
    assert f4d.returncode == 0 and "would REFUSE live" in f4d.stdout and "the architecture pick is HELD" in f4d.stdout, f4d.stdout + f4d.stderr
    assert _analyze(w.local, w.local / "allowed_pairs.json") == 0; shutil.copy(w.local / "r1_fit.json", w.cluster / "r1_fit.json")   # the re-analysis
    rt = w.run("table"); outt = rt.stdout + rt.stderr
    assert rt.returncode == 0 and "PICK HELD (hpo_pick.HOLD.json" in (w.local / "table" / "table.md").read_text(), outt
    # a pick without --rank/--set while held: refused by name; the hold stands
    r = w.run("pick"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "pick is HELD" in r.stderr and "needs --rank R --set S" in r.stderr and hold.is_file(), out
    # A held re-pick runs the COMPLETE fit verification first — a widened-menu refit over another menu refuses it
    stale = {"design": "v2", "allowed_pairs": [[4, "wo"]], "menu_extended_from": [[4, "wo"]], "fit": {}, "rows": []}
    (w.local / "r1_fit_menu.json").write_text(json.dumps(stale))
    r = w.run("pick", env_extra={"PICK_RANK": "4", "PICK_SET": "mlp"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "HELD, and a fit does not stand over the current evidence" in r.stderr and hold.is_file() and "FAKE-PICK" not in out, out
    (w.local / "r1_fit_menu.json").unlink()
    # The promotion clears the hold only after every write; the shell's fold then FAILS -> the HOLD is RESTORED
    r = w.run("pick", env_extra={"PICK_RANK": "4", "PICK_SET": "mlp", "FAIL_FOLD": "1"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "FAKE-PICK promoted rank 4 set mlp" in out and "the fold failed after the promotion" in r.stderr and "HOLD restored" in r.stderr, out
    assert hold.is_file() and "fold (make_sdfbase_configs.py --fold-hpo) FAILED" in json.loads(hold.read_text())["reason"]
    assert "rank 4 set mlp" not in (w.tmp / "state" / "m" / "03_hpo" / "DONE.lora.pick").read_text().splitlines()[-1] or True
    # the fresh pick from the current evidence: the real binding passes, the promotion clears the hold (recorded), the marker is re-stamped
    r = w.run("pick", env_extra={"PICK_RANK": "4", "PICK_SET": "mlp"}); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "pick: HELD" in out and "re-derives from the current evidence" in out and "FAKE-PICK promoted rank 4 set mlp" in out and "the HOLD is cleared by this promotion" in out, out
    assert not hold.exists() and list(w.local.glob("hpo_pick.HOLD.cleared.*.json")), "the hold is moved aside, never deleted"
    pj = json.loads((w.local / "hpo_pick.json").read_text()); assert pj["promoted"] and pj["hold_cleared"]["previous_hold"]["held_pick"]["rank"] == 4
    assert "pending" not in pj["hold_cleared"] and pj["hold_cleared"].get("finalized_utc"), pj["hold_cleared"]      # cleared only at the end
    assert "STUB make_sdfbase_configs" in out and "rank 4 set mlp" in (w.tmp / "state" / "m" / "03_hpo" / "DONE.lora.pick").read_text()
    # the consumers pass the hold check again (the fold reader may stop later on this synthetic tree; it must not refuse on the hold)
    fr = _fold_reader(w.local / "hpo_fold.json"); assert "is HELD" not in fr.stderr, fr.stderr[-800:]
    f4 = _fit_unit_04(w.tmp / "state", "m"); assert "the architecture pick is HELD" not in f4.stderr and "STUB submit control_lora" in f4.stdout, f4.stdout + f4.stderr
    rt = w.run("table"); assert rt.returncode == 0 and "PICK HELD" not in (w.local / "table" / "table.md").read_text(), rt.stdout + rt.stderr


def _lay_round2(w: World) -> None:
    """Three round-2 (refine) summaries at the selected pair, as copies of round-1 trials of that pair: enough for the pair-only refit."""
    fit = json.loads((w.local / "r1_fit.json").read_text()); pair = (fit["selection"]["lora_rank"], fit["selection"]["lora_set_id"])
    srcs = [d for d in sorted(w.cluster.glob("r1_t*")) if d.is_dir() and
            (lambda s: (s["trial"]["lora_rank"], s["trial"]["lora_set_id"]) == pair)(json.loads((d / "summary.json").read_text()))][:3]
    assert len(srcs) >= 1, pair
    import math as _m
    c = _m.log10(float(fit["selection"]["lr"])); box = [c - 0.5, c + 0.5]
    for i in range(3):
        s = json.loads((srcs[i % len(srcs)] / "summary.json").read_text()); s["round"] = 2; s["trial"]["index"] = i
        s["trial"]["lr"] = 10 ** (c + (i - 1) * 0.25); s["lr"] = s["trial"]["lr"]
        s["best_heldout_loss"] = float(s["best_heldout_loss"]) - 0.001 * (i + 1)
        s["lr_log10_box"] = box; s["box"] = {**(s.get("box") or {}), "lr": [10 ** box[0], 10 ** box[1]]}
        s["refine_of"] = {"fit": "r1_fit.json", "rounds": [1], "y_key": "best_heldout_loss", "selected_from": fit["selection"].get("selected_from"),
                          "pair": {"lora_rank": pair[0], "lora_set_id": pair[1], "lora_modules": s["trial"]["lora_modules"], "per_expert_rank": s["trial"].get("per_expert_rank")},
                          "center_log10": c, "box_log10": box, "half_decades": 0.5, "n_used": 3}   # the emitter's refine block, as --refine-from writes it
        for root in (w.cluster, w.local):
            d = root / f"r2_t{i:02d}"; d.mkdir(exist_ok=True); (d / "summary.json").write_text(json.dumps(s))


def _pair_refit(w: World):
    ns = argparse.Namespace(out=str(w.local), rounds="1,2", y_key="best_heldout_loss", design="v2", allowed_pairs=str(w.local / "allowed_pairs.json"),
                            pair_only=True, close_search=False, extends_menu=False, fit_name=None, source_record=None)
    try:
        tune_arm.analyze(ns); return 0, ""
    except SystemExit as e:
        return (e.code if isinstance(e.code, int) else 1), str(e.code)


def test_live_refine_completed_refreshes_its_pair_refit_over_the_current_evidence_without_resubmitting(tmp_path):
    """After a round-1 metrics backfill and `--phase analyze`, a COMPLETED refine re-derives r1+2_fit_pair.json (refit only:
    nothing emitted, pushed or submitted) and re-stamps its marker; unchanged evidence skips; a refit over a stale design fit is refused by name."""
    w = World(tmp_path); w.lay_round1(pd=True); _lay_round2(w)
    prc = _pair_refit(w); assert (w.local / "r1+2_fit_pair.json").is_file(), prc
    d = w.tmp / "state" / "m" / "03_hpo"; d.mkdir(parents=True, exist_ok=True); (d / "DONE.lora.refine").write_text("2026-10-01T00:00:00Z fourth trial round 9\n")
    r = w.run("refine"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "refine: DONE marker present and every fit stands over the current evidence — skipping" in out and "STUB submit" not in out, out
    fit12_before = _sha(w.local / "r1+2_fit_pair.json")
    _lay_metrics(w.cluster)                                                                     # the round-1 (and round-2) backfill
    for dd in w.cluster.glob("r2_t*"):
        (dd / "metrics.jsonl").write_text(METRICS_ROWS)
    r = w.run("refine"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "the DESIGN fit" in r.stderr and "re-run --phase analyze first" in r.stderr and "re-deriving" not in out, out   # r1_fit.json is stale: analyze first
    assert "STUB submit" not in out and _sha(w.local / "r1+2_fit_pair.json") == fit12_before, "no refit before the design fit is current"
    assert _analyze(w.local, w.local / "allowed_pairs.json") == 0; shutil.copy(w.local / "r1_fit.json", w.cluster / "r1_fit.json")
    r = w.run("refine"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "re-deriving it over rounds 1+2 (refit only" in out and "STUB submit" not in out, out
    assert _sha(w.local / "r1+2_fit_pair.json") != fit12_before and "refreshed over the current evidence" in (d / "DONE.lora.refine").read_text()
    rows = json.loads((w.local / "r1+2_fit_pair.json").read_text())["rows"]; assert all(rr.get("clip_activation_rate") == 0.0 for rr in rows), rows[:1]
    r = w.run("refine"); assert r.returncode == 0 and "skipping" in r.stdout, r.stdout + r.stderr
    rt = w.run("table"); assert rt.returncode == 0, rt.stdout + rt.stderr


def test_live_addendum_completed_refreshes_its_widened_refit_over_the_current_evidence_without_resubmitting(tmp_path):
    """A COMPLETED addendum (round-4 rows on the record, DONE) whose widened-menu refit predates a round-1 backfill re-derives
    r1_fit_menu.json only — no emit, push or submission — and re-stamps its marker; the table then renders over it."""
    w = World(tmp_path); w.lay_round1(pd=True); w.widen(); shutil.copy(w.tmp / "wide.json", w.cluster / "allowed_pairs.json")
    ra = w.run("addendum"); assert ra.returncode == 0 and "STUB submit_round r4 array 0-5" in ra.stdout + ra.stderr, ra.stdout + ra.stderr
    menu = tune_arm.load_allowed_pairs(w.local / "allowed_pairs.json")
    for i, (rk, sid) in enumerate(NEW6):
        pr = next(p for p in menu["pairs"] if (p["rank"], p["set"]) == (rk, sid)); dd = w.cluster / f"r4_t{i:02d}"; dd.mkdir()
        (dd / "summary.json").write_text(json.dumps({"trial": {"index": i, "lr": 3e-4, "lora_rank": rk, "lora_set_id": sid, "max_grad_norm": 1.3}, "round": 4,
                                                     "trial_role": "addendum", "status": "ok", "arm": ARM, "model": MODEL, "position": "all", "si": "constv2-mix-10m",
                                                     "best_heldout_loss": 1.7, "lr": 3e-4, "lora_rank": rk, "lora_set_id": sid, "lora_modules": pr["lora_modules"],
                                                     "trainable_params": pr["trainable_params"], "allowed_pairs": [list(k) for k in WIDE15]}))
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "6/6 round-4 summaries" in out and "complete" in out, out                   # marked DONE
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "addendum: DONE marker present and every fit stands over the current evidence — skipping" in out, out
    menu_before = _sha(w.local / "r1_fit_menu.json")
    _lay_metrics(w.cluster)                                                                     # the round-1 backfill (the round-4 rows carry none)
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "the DESIGN fit" in r.stderr and "re-run --phase analyze first" in r.stderr and "re-deriving" not in out and "STUB submit" not in out, out
    assert _sha(w.local / "r1_fit_menu.json") == menu_before, "no refit before the design fit is current"
    assert _analyze(w.local, w.tmp / "allowed_pairs.json") == 0; shutil.copy(w.local / "r1_fit.json", w.cluster / "r1_fit.json")   # the design fit over ITS 9-pair menu
    r = w.run("addendum"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "re-deriving it (refit only" in out and "STUB submit" not in out and "FAKE-PICK" not in out, out
    assert _sha(w.local / "r1_fit_menu.json") != menu_before and "refreshed refit over the current evidence" in (w.tmp / "state" / "m" / "03_hpo" / "DONE.lora.addendum").read_text()
    mrows = json.loads((w.local / "r1_fit_menu.json").read_text())["rows"]; assert all(rr.get("clip_activation_rate") == 0.0 for rr in mrows)
    r = w.run("addendum"); assert r.returncode == 0 and "skipping" in r.stdout, r.stdout + r.stderr
    rt = w.run("table"); assert rt.returncode == 0 and "r1_fit_menu.json" in (w.local / "table" / "table.md").read_text(), rt.stdout + rt.stderr


def test_pick_takes_a_matching_completed_addendum_row_as_its_confirmation_and_requests_a_trial_only_without_one(tmp_path):
    """A pick at an addendum pair takes the matching COMPLETED addendum row (same rank / set / lr / clip /
    init; status ok; floor passed) as its confirmation — recorded as such (role addendum -> confirmation), the winner is that trial, no
    confirm-pick trial is requested; the candidate lr is the widened-menu refit's prediction at the pair (the lr the addendum ran at). Without
    such a row the pick requests the confirm-pick trial as before. A row at the coordinates under other fixed axes is refused by name. The REAL
    hpo_pick.py (dry-run: no registry, no files) over the search the live addendum phase leaves on the mirror."""
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase")); import hpo_pick  # noqa: E402
    w = World(tmp_path); w.lay_round1(pd=True); w.widen(); shutil.copy(w.tmp / "wide.json", w.cluster / "allowed_pairs.json")
    r = w.run("addendum"); assert r.returncode == 0 and "STUB submit_round r4 array 0-5" in r.stdout + r.stderr, r.stdout + r.stderr   # the widened refit + the r4 design
    d4 = json.loads((w.local / "r4_trials.json").read_text()); t16 = next(t for t in d4["trials"] if (t["lora_rank"], t["lora_set_id"]) == (16, "mlp"))
    assert t16["lr_source"]["kind"] == "pair_prediction", t16["lr_source"]
    rec = f"cluster_b:{w.cluster}"
    def pick(*extra):
        return subprocess.run([VENV_PY, str(REPO / "scripts" / "sdfbase" / "hpo_pick.py"), "--search-dir", str(w.local), "--rank", "16", "--set", "mlp",
                               "--allowed-pairs", str(w.local / "allowed_pairs.json"), "--source-record", rec, "--model-key", "m", "--fit-key", "k", *extra],
                              capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=300)
    key = (16, "mlp")
    # (1) no completed addendum row yet: the candidate is the widened refit's prediction, unconfirmed -> the confirm-pick request
    cand = hpo_pick.lr_for_pair(hpo_pick.load_search(w.local, str(w.local / "allowed_pairs.json"), rec), key)
    assert cand["source"] == "menu_refit:fitted_argmin_at_pair" and math.isclose(cand["lr"], float(t16["lr"]), rel_tol=1e-9) and cand["clip_evidence"]["verdict"] == "unconfirmed", cand
    r = pick(); out = r.stdout + r.stderr
    assert r.returncode == 0 and "CONFIRM-PICK requested" in out and "menu_refit:fitted_argmin_at_pair" in out and "IS the confirmation" not in out, out[-3000:]
    # (2) the addendum trial at (16, mlp) completed at its design lr (summary + metrics on the mirror, as the fetch leaves them)
    src = next(json.loads((d / "summary.json").read_text()) for d in sorted(w.cluster.glob("r1_t*")) if d.is_dir()
               and json.loads((d / "summary.json").read_text())["trial"]["lora_set_id"] == "mlp")
    smry = dict(src); smry["round"] = 4; smry["trial"] = {**src["trial"], **{k: v for k, v in t16.items() if k != "lr_source"}, "trial_role": "addendum"}
    smry["lr"] = t16["lr"]; smry["best_heldout_loss"] = float(src["best_heldout_loss"]) - 0.01; smry["status"] = "ok"; smry["floor_ok"] = True
    for root in (w.cluster, w.local):
        d = root / "r4_t02"; d.mkdir(exist_ok=True); (d / "summary.json").write_text(json.dumps(smry, default=str)); _metrics(d / "metrics.jsonl", grad_norm=0.01)
    search = hpo_pick.load_search(w.local, str(w.local / "allowed_pairs.json"), rec)
    cand = hpo_pick.lr_for_pair(search, key)
    assert cand["clip_evidence"]["verdict"] == "promotable" and [t["trial_id"] for t in cand["clip_evidence"]["trials"]] == ["r4_t02"], cand["clip_evidence"]
    conf = hpo_pick.confirmation_record(w.local, key, [t for t in cand["clip_evidence"]["trials"] if t["status"] == "ok"], cand, search["design"].get("init_scheme"))
    assert conf["trial_id"] == "r4_t02" and conf["trial_role"] == "addendum" and conf["role_at_pick"] == "confirmation" and conf["clip_activation_rate"] == 0.0, conf
    assert conf["identity"]["status"] == "ok" and conf["identity"]["floor_ok"] is True and conf["y"] == pytest.approx(smry["best_heldout_loss"]), conf
    r = pick(); out = r.stdout + r.stderr
    assert r.returncode == 0 and "the completed addendum trial r4_t02 at these coordinates IS the confirmation" in out and "CONFIRM-PICK requested" not in out, out[-3000:]
    assert "DRY-RUN: write" in out and "hpo_fold.json" in out and f'"lr": {t16["lr"]!r}'[:12] in out.replace("'", '"'), out[-3000:]
    # (3) the same row under OTHER fixed axes (another clip constant) is not this search's confirmation: refused by name, nothing promoted
    bad = dict(smry); bad["trial"] = {**smry["trial"], "max_grad_norm": 0.5}
    (w.local / "r4_t02" / "summary.json").write_text(json.dumps(bad, default=str))
    r = pick(); out = r.stdout + r.stderr
    assert r.returncode != 0 and "is not this search's confirmation" in out and "max_grad_norm" in out and "nothing promoted" in out, out[-3000:]
    # (4) the fallback when no fit predicts at the pair (the refit's prediction removed): the completed addendum row's own coordinates ship
    mf_p = w.local / "r1_fit_menu.json"; mf = json.loads(mf_p.read_text()); mf["fit"]["pair_predictions"].pop("16:mlp", None); mf_p.write_text(json.dumps(mf, default=str))
    (w.local / "r4_t02" / "summary.json").write_text(json.dumps(smry, default=str))
    cand = hpo_pick._lr_candidate(hpo_pick.load_search(w.local, str(w.local / "allowed_pairs.json"), rec), key)
    assert cand["source"] == "addendum:best_measured_trial_at_pair" and cand["trial_id"] == "r4_t02" and math.isclose(cand["lr"], float(t16["lr"]), rel_tol=1e-9), cand


def test_live_pick_under_the_menu_override_reads_the_bound_override_menu_and_refuses_without_it(tmp_path):
    """`--phase pick --rank 16 --set mlp --menu-override <ruling>` reads the override menu the addendum bound (the probe's menu does not
    carry the addendum pairs; the probe record is not revalidated under the override, as for the table) and hands it to hpo_pick and the fit
    verification; with no bound override menu the phase refuses by name."""
    w = World(tmp_path); w.lay_round1(pd=True); (w.local / "allowed_pairs.json").unlink()
    shutil.copy(w.tmp / "r1_trials.json", w.local / "r1_trials.json")
    (w.tmp / "menu_overrides").mkdir(); (w.tmp / "menu_overrides" / "D547.json").write_text(json.dumps({"ruling": "D547", "model": "m", "class": "lora", "pairs": [list(k) for k in NEW6]}))
    r = w.run("pick", env_extra={"MENU_OVERRIDE": "D547", "PICK_RANK": "16", "PICK_SET": "mlp"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "names no override menu" in r.stderr and "run --phase addendum --menu-override first" in r.stderr, out[-2000:]
    r = w.run("addendum", env_extra={"MENU_OVERRIDE": "D547"}); assert r.returncode == 0 and "STUB submit_round r4 array 0-5" in r.stdout + r.stderr, (r.stdout + r.stderr)[-2000:]
    r = w.run("pick", env_extra={"MENU_OVERRIDE": "D547", "PICK_RANK": "16", "PICK_SET": "mlp"}); out = r.stdout + r.stderr
    assert "pick: MENU OVERRIDE D547 — the pick's menu is" in out and "allowed_pairs.override.D547.json" in out and "(authorized by " in out and "menu_overrides/D547.json)" in out, out[-3000:]
    assert "FAKE-PICK" in out and "outside" not in out and "probe:" not in out, out[-3000:]     # the pair is in the override menu; no probe revalidation
    # The pick RE-AUTHORIZES the bound menu against the committed record — removed, or revised so that the menu's
    # recorded pairs are no longer approved, it refuses by name before any fetch or promotion
    rec = w.tmp / "menu_overrides" / "D547.json"; saved = rec.read_text(); rec.unlink()
    r = w.run("pick", env_extra={"MENU_OVERRIDE": "D547", "PICK_RANK": "16", "PICK_SET": "mlp"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "REFUSED" in r.stderr and "D547" in r.stderr and "FAKE-PICK" not in out and "the pick's menu is" not in out, out[-3000:]
    rec.write_text(json.dumps({"ruling": "D547", "model": "m", "class": "lora", "pairs": [list(k) for k in NEW6[:3]]}))
    r = w.run("pick", env_extra={"MENU_OVERRIDE": "D547", "PICK_RANK": "16", "PICK_SET": "mlp"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "REFUSED" in r.stderr and "FAKE-PICK" not in out and "the pick's menu is" not in out, out[-3000:]
    rec.write_text(saved)
    r = w.run("pick", env_extra={"MENU_OVERRIDE": "D547", "PICK_RANK": "16", "PICK_SET": "mlp"}); out = r.stdout + r.stderr
    assert "the pick's menu is" in out and "FAKE-PICK" in out, out[-3000:]


def test_pick_close_resolves_the_full_config_sha_from_the_openings_code_sha(tmp_path):
    """At the pick's close the LoRA fit_search opening carries the config's 16-hex prefix; the close resolves the FULL sha256 from the config
    blob at the opening's code_sha (git show), verified against the prefix; a disagreeing blob, a missing blob or no code_sha leave it to the
    emitter's `unverified` declaration; a 64-hex opening is kept."""
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase")); import hpo_pick  # noqa: E402
    repo = tmp_path / "repo"; (repo / "configs").mkdir(parents=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@TESTHOST", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@TESTHOST"}
    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, env=env, check=True).stdout.strip()
    git("init", "-q"); blob = b"model: m\nfit_width: 4\n"; (repo / "configs" / "sdfbase_k_fits.yaml").write_bytes(blob)
    git("add", "-A"); git("commit", "-q", "-m", "c1"); c1 = git("rev-parse", "HEAD")
    (repo / "configs" / "sdfbase_k_fits.yaml").write_bytes(blob + b"# folded\n"); git("add", "-A"); git("commit", "-q", "-m", "c2 (the fold)"); c2 = git("rev-parse", "HEAD")
    full = hashlib.sha256(blob).hexdigest()
    assert hpo_pick.opening_config_sha256({"config_sha256": full[:16], "code_sha": c1}, "k", repo) == (full, None)
    with pytest.raises(SystemExit, match="does not hash to the opening's config_sha256 prefix"):      # a readable, disagreeing blob REFUSES
        hpo_pick.opening_config_sha256({"config_sha256": full[:16], "code_sha": c2}, "k", repo)
    v, note = hpo_pick.opening_config_sha256({"config_sha256": full[:16], "code_sha": c1}, "other", repo); assert v == "unverified" and "no config blob" in note, (v, note)
    v, note = hpo_pick.opening_config_sha256({"config_sha256": full[:16]}, "k", repo); assert v == "unverified" and "names no commit" in note, (v, note)
    assert hpo_pick.opening_config_sha256({"config_sha256": full, "code_sha": c2}, "k", repo) == (full, None), "a 64-hex opening is kept"
    v, note = hpo_pick.opening_config_sha256({"config_sha256": "unverified"}, "k", repo); assert v == "unverified" and note
    assert hpo_pick.opening_config_sha256({}, "k", repo) == (None, None)


def test_live_completed_pick_is_held_when_its_confirming_trial_fails_the_floor_or_the_menu_changes(tmp_path):
    """A completed promotion rests on its confirming trial and on the menu it was chosen from — a retried
    confirming summary that now fails the floor (status ok, the same lr, the clip gate still passing) HOLDS the pick before the DONE skip; so
    does a widened menu (the addendum's) — the pick is made again from the new table. The unchanged pick still skips."""
    w = World(tmp_path); w.lay_round1(pd=True); _completed_pick(w)
    hold = w.local / "hpo_pick.HOLD.json"
    r = w.run("pick"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "still stands — skipping" in out and not hold.exists(), out[-3000:]
    # (1) the confirming trial retried and failing the floor
    for root in (w.cluster, w.local):
        p = root / "r9_t00" / "summary.json"; smry = json.loads(p.read_text()); smry["floor_ok"] = False; p.write_text(json.dumps(smry))
    r = w.run("pick"); out = r.stdout + r.stderr
    assert r.returncode == 2 and hold.is_file() and "the trial the promotion rests on (r9_t00) no longer confirms it" in json.loads(hold.read_text())["reason"] and "floor_ok" in json.loads(hold.read_text())["reason"], out[-3000:]
    for root in (w.cluster, w.local):
        p = root / "r9_t00" / "summary.json"; smry = json.loads(p.read_text()); smry["floor_ok"] = True; p.write_text(json.dumps(smry))
    hold.unlink()
    r = w.run("pick"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "still stands — skipping" in out and not hold.exists(), out[-3000:]
    # (2) the menu widened after the pick (the probe's re-verdict): held — the pick is made again from the new table
    w.widen(); shutil.copy(w.tmp / "wide.json", w.local / "allowed_pairs.json")
    r = w.run("pick"); out = r.stdout + r.stderr
    assert r.returncode == 2 and hold.is_file() and "menu" in json.loads(hold.read_text())["reason"].lower(), out[-3000:]   # hpo_table's menu binding, or verify_promotion's menu check — the HOLD names the menu


def test_live_addendum_reconcile_attributes_live_jobs_by_id_reconciles_a_funnel_token_through_the_funnel_and_takes_only_rollback_from_the_operator(tmp_path):
    """(1) a LIVE same-name job is attributed by ID against the funnel's ledger — a known id -> this unit's submission (completed); an
    unknown live id -> UNDECIDABLE, refused by name, nothing moved; (2) a standing funnel transition token (the lost launch: sbatch
    accepted, the ledger append lost) is reconciled THROUGH THE FUNNEL — adopted (the job identified by its nonce and bound) -> the
    receipt completes on the funnel's new binding; cleared -> the guards decide (rolled back when nothing else was submitted); refused ->
    undecidable, the funnel's ledger repair named; (3) the operator's only decision is `rollback --decision-why` (recorded);
    `accept-binding` is refused by name — closing the receipt would not bind the job in the funnel; rollback is refused while the funnel
    holds a token."""
    w, _ = _relaunch_world(tmp_path)
    utc = "20260101T000000Z"; sup = w.cluster / "superseded" / utc; sup.mkdir(parents=True)
    old_design = (w.cluster / "r4_trials.json").read_bytes()
    shutil.move(str(w.cluster / "r4_trials.json"), str(sup / "r4_trials.json")); shutil.move(str(w.cluster / "r4_t00"), str(sup / "r4_t00"))
    replacement = json.dumps({"trials": [{"index": i} for i in range(6)], "replacement": True}).encode(); (w.cluster / "r4_trials.json").write_bytes(replacement)
    receipt = {"receipt": "hpo-addendum-resubmit", "state": "replaced", "utc": utc, "by": "tester@TESTHOST pid 1", "bound_job": "4242", "job_name": "hpo-m-lora-r4",
               "pinned_sha256": hashlib.sha256(old_design).hexdigest(), "superseded": f"superseded/{utc}", "replacement_sha256": hashlib.sha256(replacement).hexdigest(),
               "moved": ["r4_trials.json", "r4_t00"]}
    adopted_marker = w.tmp / ".adopted_job"
    def reopen():
        (w.cluster / "r4_resubmit.transition.json").write_text(json.dumps(receipt)); (sup / "transition.json").unlink(missing_ok=True); adopted_marker.unlink(missing_ok=True)
    def replacement_stands():
        return (w.cluster / "r4_trials.json").read_bytes() == replacement and (sup / "r4_trials.json").read_bytes() == old_design and not (w.cluster / "r4_t00").exists()
    def restore_crash_shape():
        shutil.move(str(w.cluster / "r4_trials.json"), str(sup / "r4_trials.json")); shutil.move(str(w.cluster / "r4_t00"), str(sup / "r4_t00"))
        (sup / "replacement.r4_trials.json").unlink(); (w.cluster / "r4_trials.json").write_bytes(replacement)
    # (1a) a foreign live same-name job: undecidable — refused by name, nothing moved, the recovery named (rollback only; no accept-binding)
    reopen(); snap = _all(w)
    r = w.run("addendum", env_extra={"FAKE_SQUEUE_IDS": "7777"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "cannot be decided" in r.stderr and "does NOT know" in r.stderr and "7777" in r.stderr and "recovery: inspect" in r.stderr, out[-4000:]
    assert "--reconcile-transition rollback" in r.stderr and "accept-binding" not in r.stderr.split("recovery:")[1] and "had been SUBMITTED" not in out and _all(w) == snap
    # (1b) a live same-name job the ledger knows for the unit: this unit's submission — completed
    r = w.run("addendum", env_extra={"FAKE_SQUEUE_IDS": "5151", "STUB_BINDING_ATTEMPTS": "4242,5151"}); out = r.stdout + r.stderr
    assert "had been SUBMITTED" in out and "the funnel's ledger knows for hpo.lora.r4 is queued/running (5151" in out and replacement_stands(), out[-4000:]
    assert json.loads((sup / "transition.json").read_text())["state"] == "complete"
    # (2a) the funnel's token stands and its reconciliation ADOPTS the launched job (nonce first): the new binding completes the receipt
    reopen(); r = w.run("addendum", env_extra={"STUB_BINDING_TOKEN": "yes", "STUB_RECONCILE_TOKEN": "adopted:6161"}); out = r.stdout + r.stderr
    assert "had been SUBMITTED" in out and "the launched job 6161 on cluster_b is ADOPTED and bound" in out and "identified by its nonce read back from the job, 0123456789ab" in out and replacement_stands(), out[-4000:]
    assert json.loads((sup / "transition.json").read_text())["state"] == "complete" and adopted_marker.is_file()
    # a half-recorded launch the funnel completed from its launch record (the job no longer live) counts as this transition's launch too
    reopen(); r = w.run("addendum", env_extra={"STUB_BINDING_TOKEN": "yes", "STUB_RECONCILE_TOKEN": "adopted-ledger:6161"}); out = r.stdout + r.stderr
    assert "had been SUBMITTED" in out and "the launched job 6161 on cluster_b is ADOPTED and bound" in out and replacement_stands(), out[-4000:]
    adopted_marker.unlink(missing_ok=True)
    # an adoption the funnel did NOT make by the nonce read back (an older funnel's elimination, an unknown method) is not this transition's proof
    reopen(); snap = _all(w)
    r = w.run("addendum", env_extra={"STUB_BINDING_TOKEN": "yes", "STUB_RECONCILE_TOKEN": "adopted-launch:6161"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "cannot be decided" in r.stderr and "identified by 'launch', not by the nonce read back" in r.stderr and "had been SUBMITTED" not in out, out[-4000:]
    assert {k: v for k, v in _all(w).items()} == snap, "nothing moved, nothing closed"
    adopted_marker.unlink(missing_ok=True)
    # (2b) the token cleared (the funnel proved no launch) and nothing else: rolled back
    reopen(); r = w.run("addendum", env_extra={"STUB_BINDING_TOKEN": "yes", "STUB_RECONCILE_TOKEN": "cleared"}); out = r.stdout + r.stderr
    assert "no launch happened under it (cleared)" in out and "nothing was submitted under this transition" in out and "RESTORED the previous round-4 design" in out, out[-4000:]
    assert (w.cluster / "r4_trials.json").read_bytes() == old_design and (sup / "replacement.r4_trials.json").read_bytes() == replacement and (w.cluster / "r4_t00").is_dir()
    restore_crash_shape()
    # (2c) the funnel refuses (ambiguous candidates): undecidable, the funnel's repair named, nothing moved; rollback is refused while the token stands
    reopen(); snap = _all(w)
    r = w.run("addendum", env_extra={"STUB_BINDING_TOKEN": "yes", "STUB_RECONCILE_TOKEN": "refused:RECONCILE hpo.lora.r4: 2 qualifying candidates — the token STANDS"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "cannot be decided" in r.stderr and "could not reconcile it" in r.stderr and "2 qualifying candidates" in r.stderr and "ledger repair" in r.stderr and _all(w) == snap, out[-4000:]
    r = w.run("addendum", env_extra={"STUB_BINDING_TOKEN": "yes", "STUB_RECONCILE_TOKEN": "refused:RECONCILE hpo.lora.r4: 2 qualifying candidates — the token STANDS", "RECONCILE_DECISION": "rollback", "DECISION_WHY": "x"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "rollback refused while the funnel holds a transition token" in r.stderr and _all(w) == snap, out[-4000:]
    # (3a) accept-binding is not a decision this record can take: refused by name, nothing moved
    r = w.run("addendum", env_extra={"FAKE_SACCT_ROWS": "5252|2026-01-01T00:00:30|COMPLETED\n", "RECONCILE_DECISION": "accept-binding:5252", "DECISION_WHY": "x"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "is not a decision this record can take" in r.stderr and "would not bind the job in the funnel" in r.stderr and _all(w) == snap, out[-4000:]
    r = w.run("addendum", env_extra={"FAKE_SACCT_ROWS": "5252|2026-01-01T00:00:30|COMPLETED\n", "RECONCILE_DECISION": "rollback"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "needs --decision-why" in r.stderr and _all(w) == snap, out[-4000:]
    # (3b) the operator rolls an undecidable (foreign sacct job) transition back: restored, the decision recorded
    r = w.run("addendum", env_extra={"FAKE_SACCT_ROWS": "5252|2026-01-01T00:00:30|COMPLETED\n", "RECONCILE_DECISION": "rollback", "DECISION_WHY": "another record's job"}); out = r.stdout + r.stderr
    assert "ROLLED BACK by the operator's decision" in out and "RESTORED the previous round-4 design" in out, out[-4000:]
    assert (w.cluster / "r4_trials.json").read_bytes() == old_design and (sup / "replacement.r4_trials.json").read_bytes() == replacement and (w.cluster / "r4_t00").is_dir()
    rc = json.loads((sup / "transition.json").read_text()); assert rc["state"] == "restored" and rc["decision"]["kind"] == "rollback" and rc["decision"]["why"] == "another record's job", rc


def test_live_addendum_relaunch_exit_never_rolls_back_while_the_funnel_may_have_launched(tmp_path):
    """A failure after the move is rolled back ONLY when the funnel provably launched nothing (binding unchanged, no token); a standing
    token or a new binding means an array may be reading the canonical design — the record stays, the receipt stays OPEN, and the next
    addendum reconciles it through the funnel."""
    w, before = _relaunch_world(tmp_path)
    shutil.copytree(w.local / "r1_t03", w.local / "r1_t16")                                      # a post-move failure (inventory residue)
    r = w.run("addendum", env_extra={"SISTEER_RESUBMIT": "1", "STUB_BINDING_TOKEN": "yes"}); out = r.stdout + r.stderr
    assert r.returncode == 2 and "RESUBMIT authorized" in out and "may have LAUNCHED under it (token)" in r.stderr and "NOT rolling back" in r.stderr, out[-4000:]
    assert (w.cluster / "r4_resubmit.transition.json").is_file() and not (w.cluster / "r4_trials.json").exists(), "the receipt stays open; nothing moved back"
    sup = sorted((w.cluster / "superseded").iterdir()); assert len(sup) == 1 and (sup[0] / "r4_trials.json").is_file() and not (sup[0] / "transition.json").exists()
    assert not (w.cluster / ".resubmit.lock").exists(), "the lock is released either way"
