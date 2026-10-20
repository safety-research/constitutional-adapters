"""THE CANONICAL BEST-OF-N: the 512 store PLUS the 4,096 CLIMB of its
cap-coherent at-cap draws — and a NATIVE 4,096 pass for a cell that has NO 512 store.

The dangerous failure this file exists to prevent is the SILENT one: a cell with no 512 store that
takes the climb path finds nothing to escalate and reports zero escalations as "clean" rather than
"not applicable". Every assertion below is about that branch being REAL — in the library, in the
planner, in the driver's refusals and in the sbatch — plus the two rulings that travel with the
instrument (loops are never climbed; the base cap stays 512).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import re
import yaml

from sisteer.eval.attacks import bon_climb as bc

from full22_wave_configs import wave_configs  # the wave configs, minus the tables sharing their prefix

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts/sdfbase/full22"
ROSTER = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())


def _store(tmp_path: Path, rows, *, complete: bool = True, units: dict | None = None, nofit: list | None = None) -> Path:
    """A cell's 512 store: the responses, the durable manifest's slot accounting (`units`: every slot decided unless given otherwise),
    the typed no-fit rows (bon_errors.jsonl) and, when `complete`, the end-of-run record bon_served_run.py writes (the completion marker)."""
    d = tmp_path / "cell"; d.mkdir(parents=True, exist_ok=True)
    (d / bc.BON_RESPONSES).write_text("".join(json.dumps(r) + "\n" for r in rows))
    nofit = list(nofit or [])
    (d / "bon_errors.jsonl").write_text("".join(json.dumps({"goal_id": g, "sample_idx": i, "trial": 0, "error": bc.NOFIT_ERROR, "prompt_tokens": 9000,
                                                             "prompt_tokens_source": "tokenize_endpoint", "window": 4864, "effective_window": 4864,
                                                             "window_reason": "prompt 9000 leaves -4136 of window 4864 (< 64 minimum completion)"}) + "\n" for g, i in nofit))
    if units is None:
        units = {"n_planned": len(rows) + len(nofit), "n_done": len(rows), "n_nofit": len(nofit), "n_undecided": 0, "n_prompt_over_window": len(nofit)}
    (d / bc.BON_STORE_MANIFEST_UNITS).write_text(json.dumps({"units": units}))
    if complete:
        (d / bc.BON_STORE_MANIFEST).write_text(json.dumps({"cell": "cell", "responses": str(d / bc.BON_RESPONSES), "units": units, "complete": True}))
    return d


def _row(gid, idx, text, finish="length"):
    return {"goal_id": gid, "sample_idx": idx, "trial": 0, "variant": f"v{idx}",
            "response": text, "finish_reason": finish}


# --------------------------------------------------------------------------- the branch

def test_the_mode_is_a_branch_on_whether_a_512_store_exists(tmp_path):
    d = _store(tmp_path, [_row("g0", 0, "coherent text that does not repeat at all")])
    assert bc.climb_mode(d) == bc.MODE_CLIMB and bc.has_base_store(d)
    empty = tmp_path / "nostore"; empty.mkdir()
    assert bc.climb_mode(empty) == bc.MODE_NATIVE and not bc.has_base_store(empty)
    # a responses file WITHOUT the completion marker is a store being written (or a store whose job died): never "climb", never
    # "no store" — refused by name (the in-progress case has its own test below)
    (empty / bc.BON_RESPONSES).write_text("")
    with pytest.raises(bc.BonClimbError, match="512 store in progress"):
        bc.climb_mode(empty)
    assert bc.climb_mode(base_store_exists=True) == bc.MODE_CLIMB
    with pytest.raises(bc.BonClimbError):
        bc.climb_mode()
    # both modes draw at the climb cap; they differ in WHICH draws are made
    assert bc.cap_for(bc.MODE_CLIMB) == bc.cap_for(bc.MODE_NATIVE) == bc.CLIMB_CAP == 4096
    # the climb's SAMPLER is the store's, never a constant temperature
    assert bc.BASE_CAP == 512 and bc.CLIMB_DECODE == "as_drawn" and not hasattr(bc, "CLIMB_TEMPERATURE")


def test_a_missing_store_refuses_to_produce_an_empty_climb_manifest(tmp_path):
    """The silent-direction failure, refused at the source."""
    empty = tmp_path / "nostore"; empty.mkdir()
    with pytest.raises(bc.BonClimbError, match="NATIVELY"):
        bc.climb_manifest_rows(empty, lane="llama8b", cell_id="none")


def test_a_native_cell_is_not_applicable_never_clean():
    s = bc.escalation_summary(mode=bc.MODE_NATIVE, n_rows=7500, n_residual_at_cap=120)
    assert s["status"] == bc.STATUS_NOT_APPLICABLE and s["mode"] == bc.MODE_NATIVE
    # the escalation counts are NONE, not zero: a zero would read as "nothing needed escalating"
    assert s["n_at_cap_512"] is None and s["n_climbed"] is None and s["n_loop_not_climbed"] is None
    assert s["at_cap_share_512"] is None and "no 512 store" in s["note"]
    # the residual AT 4,096 is still a real measurement for a native cell
    assert s["n_residual_at_cap"] == 120 and s["residual_at_cap_share"] == pytest.approx(120 / 7500)
    c = bc.escalation_summary(mode=bc.MODE_CLIMB, n_rows=7500, n_at_cap=900, n_loop=200,
                              n_climbed=700, n_residual_at_cap=35)
    assert c["status"] == bc.STATUS_CLIMBED and c["n_loop_not_climbed"] == 200
    assert c["at_cap_share_512"] == pytest.approx(0.12) and c["loop_residual_share"] == pytest.approx(200 / 7500)
    assert c["residual_at_cap_share"] == pytest.approx(35 / 700)
    assert "repeating draw repeats further" in c["note"]
    # a climb cell with nothing at cap is "nothing to climb" — a real, different reading
    assert bc.escalation_summary(mode=bc.MODE_CLIMB, n_rows=100)["status"] == bc.STATUS_NOTHING_TO_CLIMB
    with pytest.raises(bc.BonClimbError):
        bc.escalation_summary(mode="whatever")
    with pytest.raises(bc.BonClimbError):           # every at-cap draw is climbed or a loop
        bc.escalation_summary(mode=bc.MODE_CLIMB, n_rows=10, n_at_cap=9, n_loop=1, n_climbed=5)


# --------------------------------------------------------------------------- the rulings

def test_loops_are_never_climbed_and_the_secondary_flag_never_excludes_a_draw():
    loop = bc.classify_draw("repeat this phrase over and over " * 40)
    assert loop["class"] == bc.CLASS_LOOP and loop["climb"] is False and loop["loop_rep_ratio"]
    fine = bc.classify_draw("Step one, measure the reagent. Step two, warm the solution slowly and wait.")
    assert fine["class"] == bc.CLASS_COHERENT and fine["climb"] is True and fine["subclass"] is None
    # a draw ONLY the secondary degeneracy classifier flags is climbed, carrying the flag
    from sisteer.eval import degeneracy as dg
    line_loopish = "\n".join(["- alpha beta gamma delta epsilon zeta"] * 25)
    d = bc.classify_draw(line_loopish)
    if dg.classify(line_loopish)["degenerate"] and d["loop_rep_ratio"] is False:
        assert d["class"] == bc.CLASS_COHERENT and d["climb"] is True
        assert d["subclass"].startswith("secondary:")


def test_the_manifest_is_the_at_cap_draws_classified(tmp_path):
    d = _store(tmp_path, [_row("g0", 0, "a clean and varied completion about chemistry glassware"),
                          _row("g0", 1, "loop loop loop loop " * 40),
                          _row("g1", 0, "terminated fine", finish="stop")])
    rows = bc.climb_manifest_rows(d, lane="gemma4e4b", cell_id="none")
    assert [r["sample_idx"] for r in rows] == [0, 1]          # only the AT-CAP draws
    assert [r["climb"] for r in rows] == [True, False]        # the loop is not climbed
    assert rows[0]["cap_512"] == bc.BASE_CAP and rows[0]["lane"] == "gemma4e4b"
    assert len(rows[0]["response_512_sha256"]) == 64 and rows[0]["variant"] == "v0"
    plan = bc.plan_cells([{"cell_id": "none", "store_dir": d},
                          {"cell_id": "gap", "store_dir": tmp_path / "absent"}])
    assert bc.lanes_by_mode(plan) == {bc.MODE_CLIMB: ["none"], bc.MODE_NATIVE: ["gap"]}


def test_the_classification_rule_is_the_trees_not_a_vendored_copy():
    """The rule IS capeval.tiers.cap_ladder's repetition rule + eval.degeneracy.classify; the
    fingerprint digests those definitions so a store attests which rule classified it."""
    from sisteer.capeval.tiers import cap_ladder as cl
    assert (cl.REP_RATIO_THRESHOLD, cl.REP_NGRAM, cl.REP_MIN_WORDS, cl.TAIL_WORDS) == (0.5, 4, 8, 200)
    fp = bc.classifier_fingerprint()
    assert len(fp) == 64 and fp == bc.classifier_fingerprint()          # stable
    src = (REPO / "src/sisteer/eval/attacks/bon_climb.py").read_text()
    assert "classifier_src" not in src and "vendored" not in src.lower().split("never a vendored")[0][-40:]


# --------------------------------------------------------------------------- config + tooling

def test_the_canonical_best_of_n_is_the_configs_default():
    """bon.max_new_tokens stays 512 — the BASE of the climb — and every full22 config carries the
    climb cap + temperature from the roster (regenerated, never hand-edited)."""
    assert (ROSTER["bon_climb_cap"], ROSTER["bon_climb_decode"]) == (bc.CLIMB_CAP, bc.CLIMB_DECODE)
    assert "bon_climb_temperature" not in ROSTER          # the wave-wide scalar is gone: the sampler is per model / per store
    n = 0
    # the WAVE configs, through the one enumerator (tests/full22_wave_configs.py) that names the tables sharing the
    # prefix -- the production roster, the certification-only one and the arms table. The generated sampled configs
    # are their wave config as data except named keys, `bon` not among them (test_sdfbase_full22_sampled), so they
    # are not walked here
    for cfg_path in wave_configs(REPO):
        cfg = yaml.safe_load(cfg_path.read_text())["bon"]
        # a target whose roster block writes `caps.bon` runs best-of-N NATIVELY at that cap (Llama-3.1-8B: 4,096, no 512 store, no climb —
        # make_full22_configs._refuse_bon_drift holds such a target to its own cap); every other target stays at the climb's BASE
        target = next((k for k in sorted(ROSTER["targets"], key=len, reverse=True) if cfg_path.stem.startswith(f"sdfbase_full22_{k}")), None)
        native_cap = ((ROSTER["targets"].get(target) or {}).get("caps") or {}).get("bon") if target else None
        # ... or a VIEW whose roster entry writes its own `caps.bon` (qwen-base-seed-replicates, 2026-09-15: the qwen35_4b s1 / s2 /
        # s3 seed-replicate views take the native 4,096 route on a target with NO caps table; make_full22_configs
        # .view_declared_bon_cap is the grammar and _refuse_bon_drift holds the rendered config to it) — a wave config's view
        # is its stem past the target; a sampled / capped derivative (`_sampled_t0`, `_cap8192`) is ITS WAVE CONFIG as data
        # (make_sampled_configs: the `bon` block is not a permitted change), so it inherits that wave config's rule — the
        # target's, or the VIEW's where the view declares `caps.bon` (the s1 / s2 / s3 sampled configs carry 4,096 since the
        # 2026-09-15 promotion into the sampled scope); the suffix is stripped to find the view
        wave_stem = re.sub(r"_sampled_t\d+(_cap\d+)?$", "", cfg_path.stem)
        view = wave_stem[len(f"sdfbase_full22_{target}"):].lstrip("_") or "main" if target else None
        view_cap = ((ROSTER["targets"][target]["views"].get(view) or {}).get("caps") or {}).get("bon") if target and view else None
        assert cfg["max_new_tokens"] == (int(view_cap) if view_cap is not None else int(native_cap) if native_cap is not None else bc.BASE_CAP), cfg_path.name
        if view_cap is not None:                    # never both authorities for one cap, never on a dose view
            assert native_cap is None and "dose" not in ROSTER["targets"][target]["views"][view] and int(view_cap) == 4096, cfg_path.name
        assert cfg["climb_cap"] == 4096 and cfg["climb_decode"] == "as_drawn" and "climb_temperature" not in cfg, cfg_path.name
        n += 1
    assert n >= 7
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/make_full22_configs.py"), "--check"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0 and "DRIFT" not in r.stdout, r.stdout


def test_the_planner_and_driver_carry_the_branch_and_the_sbatch_reads_it(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cluster_site import write_site
    monkeypatch.setenv("SISTEER_CONFIG", str(write_site(tmp_path)))      # the printer reads each cluster's host from the site
    plan = (F22 / "bon_climb_plan.py").read_text()
    assert "bc.climb_mode(store)" in plan and "MODE_NATIVE" in plan
    assert "classifier_fingerprint" in plan and "climb_manifest.jsonl" in plan
    assert "bc.store_sampler(store)" in plan and "bc.CLIMB_DECODE" in plan and "CLIMB_TEMPERATURE" not in plan
    drv = (F22 / "bon_climb_run.py").read_text()
    assert '"--mode", choices=["auto", "climb", "native_4096"]' in drv
    assert "mode = bc.climb_mode(store_dir)" in drv          # auto resolves from the store
    assert "refusing rather than reporting the wrong instrument" in drv   # an explicit wrong mode refuses
    assert "def _run_native(" in drv and "bc.escalation_summary(mode=bc.MODE_NATIVE)" in drv
    # the climb draws at the manifest rows' recorded sampler; the native pass names the served model (vendor preset)
    assert "decode=dict(sampler.decode)" in drv and "bs.resolve_bon_sampler(" in drv and '"temperature": 1.0' not in drv
    assert "decode_label=label, model=model" in drv          # the RESOLVED label (`canonical` resolves to a registry label at launch)
    assert "THE 512 STORE IS\nNEVER OPENED FOR WRITING" in drv
    sb = (F22 / "bon_climb.sbatch").read_text()
    subprocess.run(["bash", "-n", str(F22 / "bon_climb.sbatch")], check=True)
    assert "bon_climb_plan.py" in sb and '--mode "$MODE"' in sb
    assert "native_4096) f22_log" in sb and "NOT an escalation" in sb.replace("not an escalation", "NOT an escalation")
    assert "--export=" not in sb.replace("never --export=", "")
    assert "#SBATCH --qos=high" in sb and "#SBATCH --mem=200G" in sb and "--mem=0" not in sb
    assert "families_served:short" in sb      # the SAME geometry the 512 draws were served at
    sub = (F22 / "submit_full22.sh").read_text()
    assert "bon_climb.sbatch" in sub and "climb_jobs" in sub
    out = subprocess.run(["bash", str(F22 / "submit_full22.sh"), "ab" * 20], capture_output=True, text=True)
    assert out.returncode == 0 and "f22-bonclimb-gemma4e4b" in out.stdout
    assert "--array=0-3" in out.stdout and "bon_climb.sbatch" in out.stdout


def test_a_store_in_progress_is_refused_by_name_by_every_reader(tmp_path):
    """A climb launched while the 512 store unit was still writing; the per-cell planner read "no store -> native_4096" (the file was not yet
    what it looked for) while the driver read the same directory as "climb", and refused the mismatch. Both readers now branch on ONE
    event-based predicate — the store's completion marker (bon_served_run.json, written at the end of the cell's draws) — never on the
    responses file's size, rows or mtime: no marker but rows or a started marker -> refused by name, naming the live job when the started
    marker recorded it; a marker -> climb; nothing -> native_4096."""
    rows = [_row("g0", 0, "coherent text that does not repeat at all")]
    # rows, no marker: in progress, job unknown
    d = _store(tmp_path, rows, complete=False)
    assert bc.store_state(d) == (bc.STORE_IN_PROGRESS, {"job_id": None, "started": False, "responses": True})
    with pytest.raises(bc.BonClimbError, match=r"512 store in progress — job unknown live .* no completion marker bon_served_run.json"):
        bc.climb_mode(d)
    assert not bc.has_base_store(d)
    # the started marker names the job (bon_served_run.py writes it first thing)
    (d / bc.BON_STORE_STARTED).write_text(json.dumps({"job_id": "11111", "cell": "cell", "utc": "2026-01-01T00:00:00+00:00"}))
    with pytest.raises(bc.BonClimbError, match="512 store in progress — job 11111 live"):
        bc.climb_mode(d)
    # the completion marker: complete -> climb, whatever else is there
    (d / bc.BON_STORE_MANIFEST).write_text("{}")
    assert bc.store_state(d)[0] == bc.STORE_COMPLETE and bc.climb_mode(d) == bc.MODE_CLIMB and bc.has_base_store(d)
    # nothing at all -> absent -> native_4096
    empty = tmp_path / "nothing"; empty.mkdir()
    assert bc.store_state(empty) == (bc.STORE_ABSENT, {}) and bc.climb_mode(empty) == bc.MODE_NATIVE
    # every reader goes through climb_mode and turns its refusal into an exit BY NAME (no traceback, no silent branch)
    plan = (F22 / "bon_climb_plan.py").read_text(); drv = (F22 / "bon_climb_run.py").read_text(); store = (F22 / "bon_served_run.py").read_text()
    assert "mode = bc.climb_mode(store)" in plan and "except bc.BonClimbError as exc:" in plan and 'raise SystemExit(f"[bon-climb-plan] {lane}/{cell}: {exc}")' in plan
    assert drv.count("except bc.BonClimbError as exc:") == 2 and "refusing rather than reporting the wrong instrument" in drv
    assert "_bc.BON_STORE_STARTED" in store and '"bon_served_run.json"' in store       # the start marker first, the completion marker last
    assert store.index("_bc.BON_STORE_STARTED") < store.index('(run_dir / "bon_served_run.json").write_text')


def test_a_store_is_complete_when_every_planned_slot_is_decided_never_when_a_marker_merely_exists(tmp_path):
    """The writer exited 0 and wrote the completion marker with `n_uncovered > 0` (prompt_over_window slots left for a resume), so marker
    presence read a PARTIAL store as complete and the planner climbed the present rows, dropping the missing at-cap candidates. Completeness
    is now the durable manifest's slot accounting through ONE predicate (store_completion) shared by the writer (no marker, exit 3, while
    slots are undecided) and every reader: decided = covered (n_done) or a typed no-fit (n_nofit); undecided slots refuse by name; decided
    no-fit slots are carried into the plan as rows, never dropped."""
    rows = [_row("g0", 0, "coherent text that does not repeat at all")]
    # the predicate itself
    ok, why, c = bc.store_completion({"n_planned": 3, "n_done": 2, "n_nofit": 1, "n_undecided": 0})
    assert ok and c == {"n_planned": 3, "n_done": 2, "n_nofit": 1, "n_undecided": 0} and "every planned slot decided" in why
    ok, why, _ = bc.store_completion({"n_planned": 3, "n_done": 2, "n_nofit": 0, "n_undecided": 1})
    assert not ok and "1 of 3 planned slot(s) UNDECIDED" in why
    ok, why, _ = bc.store_completion({"n_planned": 3, "n_done": 2, "n_prompt_over_window": 1})       # a manifest before n_nofit / n_undecided
    assert ok, why
    ok, why, _ = bc.store_completion({"n_planned": 3, "n_done": 1, "n_prompt_over_window": 1})
    assert not ok and "1 of 3 planned slot(s) UNDECIDED" in why
    assert not bc.store_completion(None)[0] and not bc.store_completion({"n_planned": 3})[0]
    # a marker over a PARTIAL store (undecided slots): refused by name by every reader — never "climb", never "no store"
    d = _store(tmp_path / "partial", rows, units={"n_planned": 3, "n_done": 1, "n_nofit": 0, "n_undecided": 2})
    st, detail = bc.store_state(d)
    assert st == bc.STORE_PARTIAL and detail["n_undecided"] == 2, (st, detail)
    with pytest.raises(bc.BonClimbError, match=r"512 store PARTIAL .* 2 of 3 planned slot\(s\) UNDECIDED"):
        bc.climb_mode(d)
    assert not bc.has_base_store(d)
    # a marker with NO durable manifest: completeness cannot be certified -> partial, refused by name
    d2 = _store(tmp_path / "nomanifest", rows); (d2 / bc.BON_STORE_MANIFEST_UNITS).unlink()
    assert bc.store_state(d2)[0] == bc.STORE_PARTIAL
    with pytest.raises(bc.BonClimbError, match="completeness cannot be certified"):
        bc.climb_mode(d2)
    # every slot decided, one of them a typed no-fit: complete -> climb; the no-fit slot is carried, by name
    d3 = _store(tmp_path / "decided", rows, nofit=[("g1", 0)])
    st, detail = bc.store_state(d3)
    assert st == bc.STORE_COMPLETE and detail["n_nofit"] == 1 and detail["n_planned"] == 2 and bc.climb_mode(d3) == bc.MODE_CLIMB
    assert bc.store_nofit_slots(d3) == [{"goal_id": "g1", "sample_idx": 0, "trial": 0, "error": bc.NOFIT_ERROR, "kind": bc.NOFIT_PROVEN, "prompt_tokens": 9000, "window": 4864}]
    # the writer branches on the SAME predicate before the marker, and exits non-zero without it; the planner carries the no-fit rows
    store = (F22 / "bon_served_run.py").read_text(); plan = (F22 / "bon_climb_plan.py").read_text()
    assert "decided, why, counts = _bc.store_completion(_bc.store_units(run_dir), _bc.store_slot_classification(run_dir))" in store and "return 3" in store
    assert store.index("_bc.store_completion(") < store.index('(run_dir / "bon_served_run.json").write_text') and '"units": counts, "complete": True' in store
    assert "bc.store_nofit_slots(store) if (x[\"goal_id\"], x[\"sample_idx\"], x[\"trial\"]) in _planned" in plan and "row.update(n_nofit=len(nofit), nofit_slots=nofit" in plan
    # the loop's own accounting names the decided no-fit and the undecided remainder
    served = (REPO / "src/sisteer/eval/attacks/bon_served.py").read_text()
    assert 'counts["n_nofit"] = int(counts.get("n_nofit_proven", 0))' in served and 'counts["n_undecided"] = n_planned - int(counts["n_done"]) - int(counts["n_nofit"])' in served


def test_the_climb_plan_is_this_attempts_never_a_file_left_by_an_earlier_one():
    """The u0 wave's climb reused `plan/plan.json` written at 11:24Z by its FIRST attempt (before the 512 store existed: every cell
    native_4096) on three later attempts, because the sbatch wrote the plan only when the file was absent; the driver then refused the
    complete store's `climb` against the plan's `native_4096` every time. Shard 0 now ALWAYS recomputes the plan (the previous pair moved
    aside under a unique name), the planner stamps the array job it classified the store for, and the other shards wait for a plan whose
    job_id is THEIR job's — never for a file that merely exists."""
    sb = (F22 / "bon_climb.sbatch").read_text(); plan = (F22 / "bon_climb_plan.py").read_text()
    assert '[ ! -s "$PLAN" ]' not in sb and 'if [ "$SHARD" = 0 ]; then' in sb and "plan/superseded/" in sb and '--job-id "$THIS_JOB"' in sb
    assert "plan_is_ours(){" in sb and '.get("job_id","")' in sb and "plan_is_ours || f22_fatal" in sb and "a plan of another job is not this store" in sb
    assert 'ap.add_argument("--job-id"' in plan and '"job_id": args.job_id or os.environ.get("SLURM_ARRAY_JOB_ID")' in plan and '"planned_utc"' in plan


def test_only_a_proven_no_fit_decides_a_slot_and_a_torn_ledger_never_masks_an_undecided_one(tmp_path):
    """PromptOverWindow also carries recoverable failures — a floor-only rejection whose /tokenize was unavailable
    (`floor_only_unresolvable`), a rejection inconsistent with a prompt that fits, a clamped retry rejected again — so counting every
    typed row as a decided no-fit could file a COMPLETE marker over slots a resume should draw. One classifier (classify_over_window) for
    the writer and every reader: `nofit_proven` only when the exact prompt length + the minimum completion exceed the effective window;
    everything else stays undecided by name. The slots are re-derived from their rows: torn lines and torn keys decide nothing, a foreign
    key beyond the plan is reported and never masks an undecided slot."""
    def over(goal, idx, *, pt=6017, src="tokenize_endpoint", win=4864, reason="prompt 6017 leaves -1153 of window 4864 (< 64 minimum completion)"):
        return {"goal_id": goal, "sample_idx": idx, "trial": 0, "error": bc.NOFIT_ERROR, "prompt_tokens": pt, "prompt_tokens_source": src,
                "prompt_floor": 4353, "requested_max_tokens": 512, "window": win, "server_window": win, "effective_window": win, "window_reason": reason}
    assert bc.classify_over_window(over("g", 0)) == bc.NOFIT_PROVEN
    assert bc.classify_over_window(over("g", 0, pt=None, src=None, reason="floor_only_unresolvable")) == bc.UNRESOLVED_TOKENIZE
    assert bc.classify_over_window(over("g", 0, src=None)) == bc.UNRESOLVED_TOKENIZE                      # no exact length source
    assert bc.classify_over_window(over("g", 0, reason="rejection inconsistent with window 4864: prompt 700 fits")) == bc.INCONSISTENT_REJECTION
    assert bc.classify_over_window(over("g", 0, reason="rejected again after the single clamped retry (512 -> 64)")) == bc.RETRY_REJECTED
    assert bc.classify_over_window(over("g", 0, pt=4000)) == bc.UNRESOLVED_OTHER                        # exact length that fits: not a proven no-fit
    # a store whose only over-window slot is floor_only_unresolvable: UNDECIDED -> partial, no climb (the writer files no marker either)
    d = tmp_path / "unres"; d.mkdir()
    (d / bc.BON_RESPONSES).write_text("".join(json.dumps(_row("g0", i, "coherent text that does not repeat at all")) + "\n" for i in range(2)))
    (d / "bon_errors.jsonl").write_text(json.dumps(over("g0", 2, pt=None, src=None, reason="floor_only_unresolvable")) + "\n")
    units = {"n_planned": 3, "n_done": 2, "n_prompt_over_window": 1}
    (d / bc.BON_STORE_MANIFEST_UNITS).write_text(json.dumps({"units": units})); (d / bc.BON_STORE_MANIFEST).write_text("{}")
    cls = bc.store_slot_classification(d)
    assert cls["n_nofit_proven"] == 0 and cls["n_undecided_over_window"] == 1 and cls["by_kind"][bc.UNRESOLVED_TOKENIZE] == 1
    ok, why, c = bc.store_completion(units, cls)
    assert not ok and "1 of 3 planned slot(s) UNDECIDED" in why and c["n_undecided"] == 1
    assert bc.store_state(d)[0] == bc.STORE_PARTIAL
    with pytest.raises(bc.BonClimbError, match="512 store PARTIAL"):
        bc.climb_mode(d)
    # the same slot with an exact length over the window: PROVEN -> complete; a torn line and a torn key (another writer) are counted,
    # a foreign well-typed key beyond the plan is reported, and none of it masks an undecided slot
    (d / "bon_errors.jsonl").write_text(json.dumps(over("g0", 2)) + "\n" + '{"goal_id": "g0", "sample_idx": 2, "tr' + "\n"
                                        + json.dumps(over(None, 2)) + "\n" + json.dumps(over("g_foreign", 0)) + "\n")
    cls = bc.store_slot_classification(d)
    assert cls["n_nofit_proven"] == 2 and cls["n_unparsable_rows"] == 2 and cls["n_undecided_over_window"] == 0
    ok, why, c = bc.store_completion(units, cls)
    assert ok and c["n_nofit"] == 1 and c["n_foreign_slots"] == 1 and c["n_unparsable_rows"] == 2 and "ledger integrity: 1 foreign slot key" in why, (ok, why, c)
    assert bc.store_state(d)[0] == bc.STORE_COMPLETE and bc.climb_mode(d) == bc.MODE_CLIMB
    assert [s["goal_id"] for s in bc.store_nofit_slots(d)] == ["g0", "g_foreign"]
    # a count can be fooled by a foreign key standing in for a planned slot that has NO row; the PLANNER's exact check over the plan's own
    # universe (the config's goals x n_max) cannot: the missing slot is a gap by name, the foreign key is listed and never counted
    (d / bc.BON_RESPONSES).write_text(json.dumps(_row("g0", 0, "coherent text that does not repeat at all")) + "\n")
    gaps = bc.store_planned_gaps(d, [("g0", 0, 0), ("g0", 1, 0), ("g0", 2, 0)])
    assert gaps["n_planned"] == 3 and gaps["n_covered"] == 1 and gaps["n_nofit_proven"] == 1 and gaps["foreign_keys"] == [("g_foreign", 0, 0)]
    assert gaps["gaps"] == [{"goal_id": "g0", "sample_idx": 1, "trial": 0, "kind": "no_row"}], gaps
    assert bc.store_planned_gaps(d, [("g0", 0, 0), ("g0", 2, 0)])["gaps"] == []                    # every planned slot covered or proven
    plan = (F22 / "bon_climb_plan.py").read_text()
    assert "gaps = bc.store_planned_gaps(store, [(g.goal_id, i, 0) for g in goals for i in range(n_max)])" in plan and "never climb over undecided slots" in plan
    # the writer and the readers share the classifier
    served = (REPO / "src/sisteer/eval/attacks/bon_served.py").read_text(); store = (F22 / "bon_served_run.py").read_text()
    assert "from .bon_climb import classify_over_window, NOFIT_PROVEN" in served and '"nofit_kind": _kind' in served and 'counts["n_nofit"] = int(counts.get("n_nofit_proven", 0))' in served
    assert "_bc.store_completion(_bc.store_units(run_dir), _bc.store_slot_classification(run_dir))" in store
